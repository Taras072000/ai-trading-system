"""
Historical Data Manager для Peper Binance v4
Специализированный сбор исторических данных за год для всех AI моделей
Поддержка всех таймфреймов: 1m, 5m, 15m, 1h, 4h, 1d
"""

import asyncio
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import os
import json
from pathlib import Path
import aiohttp
import time
from dataclasses import dataclass
import sqlite3
import gzip
import pickle

from binance.client import Client
from binance.exceptions import BinanceAPIException

logger = logging.getLogger(__name__)

@dataclass
class DataRequest:
    """Запрос на загрузку данных"""
    symbol: str
    interval: str
    start_date: datetime
    end_date: datetime
    priority: int = 1

@dataclass
class DataStats:
    """Статистика загруженных данных"""
    symbol: str
    interval: str
    total_candles: int
    date_range: Tuple[datetime, datetime]
    file_size_mb: float
    last_updated: datetime

class HistoricalDataManager:
    """Менеджер исторических данных с оптимизацией и кэшированием"""
    
    def __init__(self, api_key: str = None, api_secret: str = None):
        self.client = Client(api_key, api_secret) if api_key and api_secret else None
        
        # Директории для данных
        self.data_dir = Path("historical_data")
        self.cache_dir = self.data_dir / "cache"
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        
        for dir_path in [self.data_dir, self.cache_dir, self.raw_dir, self.processed_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # База данных для метаданных
        self.db_path = self.data_dir / "metadata.db"
        self._init_database()
        
        # Настройки
        self.rate_limit_delay = 0.1  # Задержка между запросами
        self.max_retries = 3
        self.chunk_size = 1000  # Количество свечей за запрос
        
        # Поддерживаемые интервалы
        self.intervals = {
            '1m': {'minutes': 1, 'limit': 1000},
            '5m': {'minutes': 5, 'limit': 1000}, 
            '15m': {'minutes': 15, 'limit': 1000},
            '1h': {'minutes': 60, 'limit': 1000},
            '4h': {'minutes': 240, 'limit': 1000},
            '1d': {'minutes': 1440, 'limit': 1000}
        }
        
        # Статистика
        self.download_stats = {}
    
    def _init_database(self):
        """Инициализация базы данных для метаданных"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS data_files (
                    id INTEGER PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    interval TEXT NOT NULL,
                    start_date TEXT NOT NULL,
                    end_date TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    candles_count INTEGER,
                    file_size_bytes INTEGER,
                    created_at TEXT,
                    last_updated TEXT,
                    UNIQUE(symbol, interval, start_date, end_date)
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS download_log (
                    id INTEGER PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    interval TEXT NOT NULL,
                    start_date TEXT NOT NULL,
                    end_date TEXT NOT NULL,
                    status TEXT NOT NULL,
                    error_message TEXT,
                    download_time_seconds REAL,
                    created_at TEXT
                )
            ''')
    
    async def download_all_data(self, symbols: List[str], lookback_days: int = 365) -> Dict[str, Dict[str, DataStats]]:
        """Загрузка всех данных для всех символов и интервалов"""
        
        logger.info(f"🚀 Начинаем загрузку данных для {len(symbols)} символов за {lookback_days} дней")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)
        
        results = {}
        
        for symbol in symbols:
            logger.info(f"📊 Загрузка данных для {symbol}")
            symbol_results = {}
            
            for interval in self.intervals.keys():
                try:
                    logger.info(f"   ⏱️  Интервал {interval}")
                    
                    # Проверка существующих данных
                    if await self._data_exists(symbol, interval, start_date, end_date):
                        logger.info(f"   ✅ Данные уже существуют для {symbol} {interval}")
                        stats = await self._get_data_stats(symbol, interval)
                        symbol_results[interval] = stats
                        continue
                    
                    # Загрузка новых данных
                    data = await self._download_interval_data(symbol, interval, start_date, end_date)
                    
                    if data is not None and len(data) > 0:
                        # Сохранение данных
                        file_path = await self._save_data(symbol, interval, data, start_date, end_date)
                        
                        # Создание статистики
                        stats = DataStats(
                            symbol=symbol,
                            interval=interval,
                            total_candles=len(data),
                            date_range=(data.index[0], data.index[-1]),
                            file_size_mb=os.path.getsize(file_path) / 1024 / 1024,
                            last_updated=datetime.now()
                        )
                        
                        symbol_results[interval] = stats
                        
                        logger.info(f"   ✅ Загружено {len(data)} свечей для {symbol} {interval}")
                    
                    # Задержка для соблюдения лимитов API
                    await asyncio.sleep(self.rate_limit_delay)
                    
                except Exception as e:
                    logger.error(f"   ❌ Ошибка загрузки {symbol} {interval}: {e}")
                    await self._log_error(symbol, interval, start_date, end_date, str(e))
                    continue
            
            results[symbol] = symbol_results
        
        # Сохранение общей статистики
        await self._save_download_summary(results)
        
        return results
    
    async def _download_interval_data(self, symbol: str, interval: str, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
        """Загрузка данных для конкретного интервала"""
        
        if not self.client:
            logger.warning("Binance client не инициализирован, используем демо данные")
            return await self._generate_demo_data(symbol, interval, start_date, end_date)
        
        all_data = []
        current_start = start_date
        
        while current_start < end_date:
            try:
                # Расчет конечной даты для чанка
                interval_minutes = self.intervals[interval]['minutes']
                chunk_duration = timedelta(minutes=interval_minutes * self.chunk_size)
                current_end = min(current_start + chunk_duration, end_date)
                
                # Запрос к API
                klines = self.client.get_historical_klines(
                    symbol=symbol,
                    interval=interval,
                    start_str=int(current_start.timestamp() * 1000),
                    end_str=int(current_end.timestamp() * 1000),
                    limit=self.chunk_size
                )
                
                if not klines:
                    break
                
                # Преобразование в DataFrame
                df = pd.DataFrame(klines, columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_asset_volume', 'number_of_trades',
                    'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
                ])
                
                # Обработка данных
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                
                # Преобразование типов
                numeric_columns = ['open', 'high', 'low', 'close', 'volume']
                df[numeric_columns] = df[numeric_columns].astype(float)
                
                all_data.append(df[numeric_columns])
                
                # Обновление позиции
                current_start = current_end
                
                # Задержка
                await asyncio.sleep(self.rate_limit_delay)
                
            except BinanceAPIException as e:
                logger.error(f"Binance API ошибка для {symbol} {interval}: {e}")
                if e.code == -1121:  # Invalid symbol
                    break
                await asyncio.sleep(1)  # Увеличенная задержка при ошибке
                
            except Exception as e:
                logger.error(f"Общая ошибка загрузки {symbol} {interval}: {e}")
                break
        
        if all_data:
            combined_data = pd.concat(all_data)
            combined_data = combined_data[~combined_data.index.duplicated(keep='first')]
            combined_data.sort_index(inplace=True)
            return combined_data
        
        return None
    
    async def _generate_demo_data(self, symbol: str, interval: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Генерация демо данных для тестирования"""
        
        interval_minutes = self.intervals[interval]['minutes']
        periods = int((end_date - start_date).total_seconds() / (interval_minutes * 60))
        
        # Генерация случайных цен
        np.random.seed(hash(symbol) % 2**32)  # Детерминированная генерация
        
        base_price = 50000 if 'BTC' in symbol else 3000 if 'ETH' in symbol else 500
        
        # Случайное блуждание для цен
        returns = np.random.normal(0, 0.02, periods)  # 2% волатильность
        prices = [base_price]
        
        for ret in returns:
            prices.append(prices[-1] * (1 + ret))
        
        # Создание OHLCV данных
        data = []
        timestamps = pd.date_range(start=start_date, end=end_date, freq=f'{interval_minutes}T')[:periods]
        
        for i, ts in enumerate(timestamps):
            if i >= len(prices) - 1:
                break
                
            open_price = prices[i]
            close_price = prices[i + 1]
            
            # Генерация high/low
            high_price = max(open_price, close_price) * (1 + np.random.uniform(0, 0.01))
            low_price = min(open_price, close_price) * (1 - np.random.uniform(0, 0.01))
            
            # Генерация объема
            volume = np.random.uniform(100, 1000)
            
            data.append({
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': volume
            })
        
        df = pd.DataFrame(data, index=timestamps[:len(data)])
        return df
    
    async def _save_data(self, symbol: str, interval: str, data: pd.DataFrame, start_date: datetime, end_date: datetime) -> str:
        """Сохранение данных в файл"""
        
        # Создание имени файла
        start_str = start_date.strftime('%Y%m%d')
        end_str = end_date.strftime('%Y%m%d')
        filename = f"{symbol}_{interval}_{start_str}_{end_str}.pkl.gz"
        
        file_path = self.raw_dir / filename
        
        # Сжатое сохранение
        with gzip.open(file_path, 'wb') as f:
            pickle.dump(data, f)
        
        # Запись в базу данных
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT OR REPLACE INTO data_files 
                (symbol, interval, start_date, end_date, file_path, candles_count, file_size_bytes, created_at, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                symbol, interval, start_date.isoformat(), end_date.isoformat(),
                str(file_path), len(data), os.path.getsize(file_path),
                datetime.now().isoformat(), datetime.now().isoformat()
            ))
        
        return str(file_path)
    
    async def _data_exists(self, symbol: str, interval: str, start_date: datetime, end_date: datetime) -> bool:
        """Проверка существования данных"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                SELECT COUNT(*) FROM data_files 
                WHERE symbol = ? AND interval = ? 
                AND start_date <= ? AND end_date >= ?
            ''', (symbol, interval, start_date.isoformat(), end_date.isoformat()))
            
            count = cursor.fetchone()[0]
            return count > 0
    
    async def _get_data_stats(self, symbol: str, interval: str) -> DataStats:
        """Получение статистики данных"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                SELECT candles_count, start_date, end_date, file_size_bytes, last_updated
                FROM data_files 
                WHERE symbol = ? AND interval = ?
                ORDER BY last_updated DESC LIMIT 1
            ''', (symbol, interval))
            
            row = cursor.fetchone()
            
            if row:
                return DataStats(
                    symbol=symbol,
                    interval=interval,
                    total_candles=row[0],
                    date_range=(datetime.fromisoformat(row[1]), datetime.fromisoformat(row[2])),
                    file_size_mb=row[3] / 1024 / 1024,
                    last_updated=datetime.fromisoformat(row[4])
                )
        
        return DataStats(symbol, interval, 0, (datetime.now(), datetime.now()), 0.0, datetime.now())
    
    async def load_data(self, symbol: str, interval: str, start_date: datetime = None, end_date: datetime = None) -> Optional[pd.DataFrame]:
        """Загрузка данных из файла"""
        
        with sqlite3.connect(self.db_path) as conn:
            query = '''
                SELECT file_path FROM data_files 
                WHERE symbol = ? AND interval = ?
            '''
            params = [symbol, interval]
            
            if start_date:
                query += ' AND end_date >= ?'
                params.append(start_date.isoformat())
            
            if end_date:
                query += ' AND start_date <= ?'
                params.append(end_date.isoformat())
            
            query += ' ORDER BY start_date'
            
            cursor = conn.execute(query, params)
            files = cursor.fetchall()
        
        if not files:
            logger.warning(f"Данные не найдены для {symbol} {interval}")
            return None
        
        # Загрузка и объединение файлов
        all_data = []
        
        for file_path_tuple in files:
            file_path = Path(file_path_tuple[0])
            
            if file_path.exists():
                try:
                    with gzip.open(file_path, 'rb') as f:
                        data = pickle.load(f)
                        all_data.append(data)
                except Exception as e:
                    logger.error(f"Ошибка загрузки файла {file_path}: {e}")
        
        if all_data:
            combined_data = pd.concat(all_data)
            combined_data = combined_data[~combined_data.index.duplicated(keep='first')]
            combined_data.sort_index(inplace=True)
            
            # Фильтрация по датам если нужно
            if start_date:
                combined_data = combined_data[combined_data.index >= start_date]
            if end_date:
                combined_data = combined_data[combined_data.index <= end_date]
            
            return combined_data
        
        return None
    
    async def _log_error(self, symbol: str, interval: str, start_date: datetime, end_date: datetime, error: str):
        """Логирование ошибок"""
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO download_log 
                (symbol, interval, start_date, end_date, status, error_message, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                symbol, interval, start_date.isoformat(), end_date.isoformat(),
                'error', error, datetime.now().isoformat()
            ))
    
    async def _save_download_summary(self, results: Dict[str, Dict[str, DataStats]]):
        """Сохранение сводки загрузки"""
        
        summary = {
            'download_date': datetime.now().isoformat(),
            'total_symbols': len(results),
            'total_intervals': sum(len(intervals) for intervals in results.values()),
            'symbols': {}
        }
        
        for symbol, intervals in results.items():
            summary['symbols'][symbol] = {}
            
            for interval, stats in intervals.items():
                summary['symbols'][symbol][interval] = {
                    'candles': stats.total_candles,
                    'size_mb': round(stats.file_size_mb, 2),
                    'date_range': [stats.date_range[0].isoformat(), stats.date_range[1].isoformat()]
                }
        
        summary_file = self.data_dir / f"download_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"📊 Сводка сохранена в {summary_file}")
    
    def get_available_data(self) -> Dict[str, List[str]]:
        """Получение списка доступных данных"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                SELECT DISTINCT symbol, interval FROM data_files 
                ORDER BY symbol, interval
            ''')
            
            data = {}
            for symbol, interval in cursor.fetchall():
                if symbol not in data:
                    data[symbol] = []
                data[symbol].append(interval)
        
        return data
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Получение статистики хранилища"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                SELECT 
                    COUNT(*) as total_files,
                    SUM(candles_count) as total_candles,
                    SUM(file_size_bytes) as total_size_bytes,
                    COUNT(DISTINCT symbol) as unique_symbols,
                    COUNT(DISTINCT interval) as unique_intervals
                FROM data_files
            ''')
            
            row = cursor.fetchone()
            
            return {
                'total_files': row[0],
                'total_candles': row[1],
                'total_size_mb': round(row[2] / 1024 / 1024, 2) if row[2] else 0,
                'unique_symbols': row[3],
                'unique_intervals': row[4],
                'storage_path': str(self.data_dir)
            }
    
    async def cleanup_old_data(self, days_to_keep: int = 30):
        """Очистка старых данных"""
        
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        with sqlite3.connect(self.db_path) as conn:
            # Получение файлов для удаления
            cursor = conn.execute('''
                SELECT file_path FROM data_files 
                WHERE last_updated < ?
            ''', (cutoff_date.isoformat(),))
            
            files_to_delete = cursor.fetchall()
            
            # Удаление файлов
            deleted_count = 0
            for file_path_tuple in files_to_delete:
                file_path = Path(file_path_tuple[0])
                if file_path.exists():
                    try:
                        file_path.unlink()
                        deleted_count += 1
                    except Exception as e:
                        logger.error(f"Ошибка удаления файла {file_path}: {e}")
            
            # Удаление записей из БД
            conn.execute('''
                DELETE FROM data_files 
                WHERE last_updated < ?
            ''', (cutoff_date.isoformat(),))
            
            logger.info(f"🗑️ Удалено {deleted_count} старых файлов данных")

# Пример использования
async def main():
    """Основная функция для загрузки данных"""
    
    # Создание менеджера (без API ключей для демо)
    manager = HistoricalDataManager()
    
    # Символы для загрузки
    symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "SOLUSDT"]
    
    print("🚀 Запуск загрузки исторических данных...")
    print(f"📊 Символы: {', '.join(symbols)}")
    print(f"⏱️  Интервалы: {', '.join(manager.intervals.keys())}")
    print(f"📅 Период: 365 дней")
    
    # Загрузка данных
    results = await manager.download_all_data(symbols, lookback_days=365)
    
    # Вывод результатов
    print("\n" + "="*60)
    print("📈 РЕЗУЛЬТАТЫ ЗАГРУЗКИ")
    print("="*60)
    
    for symbol, intervals in results.items():
        print(f"\n💰 {symbol}")
        for interval, stats in intervals.items():
            print(f"   {interval}: {stats.total_candles:,} свечей ({stats.file_size_mb:.1f} MB)")
    
    # Общая статистика
    storage_stats = manager.get_storage_stats()
    print(f"\n📊 ОБЩАЯ СТАТИСТИКА:")
    print(f"   Всего файлов: {storage_stats['total_files']}")
    print(f"   Всего свечей: {storage_stats['total_candles']:,}")
    print(f"   Размер данных: {storage_stats['total_size_mb']} MB")
    print(f"   Символов: {storage_stats['unique_symbols']}")
    print(f"   Интервалов: {storage_stats['unique_intervals']}")

if __name__ == "__main__":
    asyncio.run(main())