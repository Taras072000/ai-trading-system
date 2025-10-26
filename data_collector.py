"""
Модуль для сбора данных с Binance API
Поддерживает загрузку исторических данных и real-time данных
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
import time
import json
from pathlib import Path

logger = logging.getLogger(__name__)

class BinanceDataCollector:
    """Класс для сбора данных с Binance API"""
    
    def __init__(self):
        self.base_url = "https://api.binance.com"
        self.session = None
        self.rate_limit_delay = 0.05  # Уменьшаем задержку для ускорения
        self.data_cache = {}
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def get_klines(self, symbol: str, interval: str = "1h", 
                        start_time: Optional[datetime] = None,
                        end_time: Optional[datetime] = None,
                        limit: int = 1000) -> pd.DataFrame:
        """
        Получение исторических данных свечей
        
        Args:
            symbol: Торговая пара (например, BTCUSDT)
            interval: Интервал (1m, 5m, 15m, 30m, 1h, 4h, 1d)
            start_time: Начальное время
            end_time: Конечное время
            limit: Максимальное количество свечей (до 1000)
        """
        if not self.session:
            raise RuntimeError("Сессия не инициализирована. Используйте async with.")
        
        url = f"{self.base_url}/api/v3/klines"
        
        params = {
            'symbol': symbol.upper(),
            'interval': interval,
            'limit': limit
        }
        
        if start_time:
            params['startTime'] = int(start_time.timestamp() * 1000)
        
        if end_time:
            params['endTime'] = int(end_time.timestamp() * 1000)
        
        try:
            await asyncio.sleep(self.rate_limit_delay)
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_klines(data)
                else:
                    error_text = await response.text()
                    logger.error(f"Ошибка API Binance: {response.status} - {error_text}")
                    raise Exception(f"API Error: {response.status}")
                    
        except Exception as e:
            logger.error(f"Ошибка при получении данных: {e}")
            raise
    
    def _parse_klines(self, klines_data: List) -> pd.DataFrame:
        """Парсинг данных свечей в DataFrame"""
        if not klines_data:
            return pd.DataFrame()
        
        columns = [
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ]
        
        df = pd.DataFrame(klines_data, columns=columns)
        
        # Конвертируем типы данных
        numeric_columns = ['open', 'high', 'low', 'close', 'volume', 
                          'quote_asset_volume', 'number_of_trades',
                          'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']
        
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Конвертируем timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
        
        # Оставляем только нужные колонки
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
        
        return df
    
    async def get_historical_data(self, symbol: str, interval: str = "1h", 
                                 days: int = 365) -> pd.DataFrame:
        """
        Получение исторических данных за указанный период
        
        Args:
            symbol: Торговая пара
            interval: Интервал свечей
            days: Количество дней назад
        """
        logger.info(f"Загрузка исторических данных {symbol} за {days} дней")
        
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        
        all_data = []
        current_start = start_time
        
        # Binance API ограничивает до 1000 свечей за запрос
        # Рассчитываем интервал для одного запроса
        interval_minutes = self._interval_to_minutes(interval)
        max_period = timedelta(minutes=interval_minutes * 1000)
        
        while current_start < end_time:
            current_end = min(current_start + max_period, end_time)
            
            try:
                batch_data = await self.get_klines(
                    symbol=symbol,
                    interval=interval,
                    start_time=current_start,
                    end_time=current_end
                )
                
                if not batch_data.empty:
                    all_data.append(batch_data)
                    logger.info(f"Загружено {len(batch_data)} свечей с {current_start} по {current_end}")
                
                current_start = current_end
                
            except Exception as e:
                logger.error(f"Ошибка при загрузке данных: {e}")
                break
        
        if all_data:
            result = pd.concat(all_data, ignore_index=True)
            result = result.drop_duplicates(subset=['timestamp']).sort_values('timestamp')
            result = result.reset_index(drop=True)
            
            logger.info(f"Всего загружено {len(result)} свечей")
            return result
        else:
            logger.warning("Не удалось загрузить данные")
            return pd.DataFrame()
    
    def _interval_to_minutes(self, interval: str) -> int:
        """Конвертация интервала в минуты"""
        interval_map = {
            '1m': 1, '3m': 3, '5m': 5, '15m': 15, '30m': 30,
            '1h': 60, '2h': 120, '4h': 240, '6h': 360, '8h': 480, '12h': 720,
            '1d': 1440, '3d': 4320, '1w': 10080, '1M': 43200
        }
        return interval_map.get(interval, 60)
    
    async def get_ticker_24hr(self, symbol: str) -> Dict:
        """Получение статистики за 24 часа"""
        if not self.session:
            raise RuntimeError("Сессия не инициализирована")
        
        url = f"{self.base_url}/api/v3/ticker/24hr"
        params = {'symbol': symbol.upper()}
        
        try:
            await asyncio.sleep(self.rate_limit_delay)
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    raise Exception(f"API Error: {response.status}")
                    
        except Exception as e:
            logger.error(f"Ошибка при получении ticker: {e}")
            raise
    
    async def get_current_price(self, symbol: str) -> float:
        """Получение текущей цены"""
        if not self.session:
            raise RuntimeError("Сессия не инициализирована")
        
        url = f"{self.base_url}/api/v3/ticker/price"
        params = {'symbol': symbol.upper()}
        
        try:
            await asyncio.sleep(self.rate_limit_delay)
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return float(data['price'])
                else:
                    raise Exception(f"API Error: {response.status}")
                    
        except Exception as e:
            logger.error(f"Ошибка при получении цены: {e}")
            raise
    
    def save_data(self, data: pd.DataFrame, symbol: str, interval: str, 
                  directory: str = "data") -> str:
        """Сохранение данных в файл"""
        Path(directory).mkdir(exist_ok=True)
        
        filename = f"{symbol}_{interval}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        filepath = Path(directory) / filename
        
        data.to_csv(filepath, index=False)
        logger.info(f"Данные сохранены в {filepath}")
        
        return str(filepath)
    
    def load_data(self, filepath: str) -> pd.DataFrame:
        """Загрузка данных из файла"""
        try:
            data = pd.read_csv(filepath)
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            logger.info(f"Данные загружены из {filepath}")
            return data
        except Exception as e:
            logger.error(f"Ошибка при загрузке данных: {e}")
            raise

class DataManager:
    """Менеджер для управления данными"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.collector = BinanceDataCollector()
    
    async def ensure_data_available(self, symbol: str, interval: str = "1h", 
                                   days: int = 365, force_update: bool = False) -> pd.DataFrame:
        """
        Обеспечивает наличие актуальных данных
        
        Args:
            symbol: Торговая пара
            interval: Интервал
            days: Количество дней
            force_update: Принудительное обновление
        """
        cache_file = self.data_dir / f"{symbol}_{interval}_cache.csv"
        
        # Проверяем кэш
        if cache_file.exists() and not force_update:
            try:
                cached_data = pd.read_csv(cache_file)
                cached_data['timestamp'] = pd.to_datetime(cached_data['timestamp'])
                
                # Проверяем актуальность данных
                last_timestamp = cached_data['timestamp'].max()
                hours_old = (datetime.now() - last_timestamp).total_seconds() / 3600
                
                if hours_old < 2:  # Данные свежие (менее 2 часов)
                    logger.info(f"Используем кэшированные данные для {symbol}")
                    return cached_data
                    
            except Exception as e:
                logger.warning(f"Ошибка при чтении кэша: {e}")
        
        # Загружаем новые данные
        logger.info(f"Загружаем новые данные для {symbol}")
        async with self.collector as collector:
            data = await collector.get_historical_data(symbol, interval, days)
        
        if not data.empty:
            # Сохраняем в кэш
            data.to_csv(cache_file, index=False)
            logger.info(f"Данные сохранены в кэш: {cache_file}")
        
        return data
    
    async def get_multiple_symbols_data(self, symbols: List[str], 
                                      interval: str = "1h", 
                                      days: int = 365) -> Dict[str, pd.DataFrame]:
        """Получение данных для нескольких символов"""
        results = {}
        
        for symbol in symbols:
            try:
                data = await self.ensure_data_available(symbol, interval, days)
                results[symbol] = data
                logger.info(f"Данные для {symbol}: {len(data)} записей")
            except Exception as e:
                logger.error(f"Ошибка при загрузке данных для {symbol}: {e}")
                results[symbol] = pd.DataFrame()
        
        return results

# Пример использования
async def main():
    """Пример использования модуля"""
    
    # Создаем менеджер данных
    data_manager = DataManager()
    
    # Загружаем данные для одного символа
    btc_data = await data_manager.ensure_data_available("BTCUSDT", "1h", 30)
    print(f"BTC данные: {len(btc_data)} записей")
    print(btc_data.head())
    
    # Загружаем данные для нескольких символов
    symbols = ["BTCUSDT", "ETHUSDT", "ADAUSDT"]
    all_data = await data_manager.get_multiple_symbols_data(symbols, "1h", 7)
    
    for symbol, data in all_data.items():
        print(f"{symbol}: {len(data)} записей")

if __name__ == "__main__":
    asyncio.run(main())