#!/usr/bin/env python3
"""
Скрипт для тестирования винрейта Lava AI на недельных исторических данных
Анализирует точность торговых сигналов за указанный период
"""

import asyncio
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import json
from pathlib import Path
import sys
import os

# Добавляем текущую директорию в путь для импорта модулей
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ai_modules.lava_ai import LavaAI
from historical_data_manager import HistoricalDataManager
from data_collector import BinanceDataCollector

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LavaWinrateTest:
    """Класс для тестирования винрейта Lava AI"""
    
    def __init__(self):
        self.lava_ai = LavaAI()
        self.data_manager = HistoricalDataManager()
        self.data_collector = BinanceDataCollector()
        
        # Настройки тестирования
        self.test_symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "SOLUSDT"]
        self.test_interval = "1h"  # Часовой интервал для тестирования
        self.test_days = 7  # Тестируем на недельных данных
        
        # Результаты тестирования
        self.test_results = {}
        
    async def initialize(self):
        """Инициализация компонентов"""
        logger.info("🚀 Инициализация Lava AI тестера...")
        await self.lava_ai.initialize()
        logger.info("✅ Lava AI инициализирован")
        
    async def load_test_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Загрузка тестовых данных для символа"""
        logger.info(f"📊 Загрузка данных для {symbol}...")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.test_days)
        
        # Сначала пробуем загрузить из HistoricalDataManager
        data = await self.data_manager.load_data(
            symbol=symbol,
            interval=self.test_interval,
            start_date=start_date,
            end_date=end_date
        )
        
        if data is not None and not data.empty:
            logger.info(f"✅ Загружено {len(data)} свечей из HistoricalDataManager")
            return data
        
        # Если нет данных, используем DataCollector
        logger.info(f"⚠️ Нет данных в HistoricalDataManager, используем DataCollector...")
        
        try:
            async with self.data_collector as collector:
                data = await collector.get_historical_data(
                    symbol=symbol,
                    interval=self.test_interval,
                    days=self.test_days
                )
                
                if data is not None and not data.empty:
                    logger.info(f"✅ Загружено {len(data)} свечей из DataCollector")
                    return data
                    
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки данных через DataCollector: {e}")
        
        # Если все не удалось, генерируем демо данные
        logger.warning(f"⚠️ Генерируем демо данные для {symbol}")
        return self._generate_demo_data(symbol, start_date, end_date)
    
    def _generate_demo_data(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Генерация демо данных для тестирования"""
        
        # Создаем временной ряд с часовым интервалом
        date_range = pd.date_range(start=start_date, end=end_date, freq='1H')
        
        # Базовая цена в зависимости от символа
        base_prices = {
            "BTCUSDT": 45000,
            "ETHUSDT": 3000,
            "BNBUSDT": 400,
            "ADAUSDT": 0.5,
            "SOLUSDT": 100
        }
        
        base_price = base_prices.get(symbol, 1000)
        
        # Генерируем случайные данные с трендом
        np.random.seed(42)  # Для воспроизводимости
        
        prices = []
        current_price = base_price
        
        for i in range(len(date_range)):
            # Добавляем случайные изменения
            change = np.random.normal(0, 0.02)  # 2% стандартное отклонение
            current_price *= (1 + change)
            prices.append(current_price)
        
        # Создаем OHLCV данные
        data = []
        for i, (timestamp, price) in enumerate(zip(date_range, prices)):
            # Генерируем OHLC на основе цены закрытия
            volatility = np.random.uniform(0.005, 0.02)  # 0.5-2% волатильность
            
            high = price * (1 + volatility)
            low = price * (1 - volatility)
            open_price = prices[i-1] if i > 0 else price
            close = price
            volume = np.random.uniform(1000, 10000)
            
            data.append({
                'timestamp': timestamp,
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume
            })
        
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        
        logger.info(f"✅ Сгенерировано {len(df)} демо свечей для {symbol}")
        return df
    
    async def test_symbol_winrate(self, symbol: str) -> Dict:
        """Тестирование винрейта для одного символа"""
        logger.info(f"🎯 Тестирование винрейта для {symbol}")
        
        # Загружаем данные
        data = await self.load_test_data(symbol)
        if data is None or data.empty:
            logger.error(f"❌ Не удалось загрузить данные для {symbol}")
            return {
                'symbol': symbol,
                'error': 'Нет данных',
                'total_signals': 0,
                'winning_signals': 0,
                'winrate': 0.0
            }
        
        # Анализируем каждую свечу и генерируем сигналы
        signals = []
        total_signals = 0
        winning_signals = 0
        
        # Используем скользящее окно для анализа
        window_size = 20  # Минимум данных для анализа
        
        for i in range(window_size, len(data) - 1):  # -1 чтобы проверить следующую свечу
            try:
                # Берем данные до текущего момента
                current_data = data.iloc[:i+1]
                
                # Генерируем сигнал
                signal_result = await self.lava_ai.generate_trading_signals(current_data)
                
                if signal_result and signal_result.get('signal') != 'HOLD':
                    signal = signal_result['signal']
                    confidence = signal_result.get('confidence', 0.5)
                    
                    # Проверяем результат на следующей свече
                    current_price = data.iloc[i]['close']
                    next_price = data.iloc[i+1]['close']
                    price_change = (next_price - current_price) / current_price
                    
                    # Определяем успешность сигнала
                    is_winning = False
                    if signal == 'BUY' and price_change > 0.001:  # Цена выросла больше чем на 0.1%
                        is_winning = True
                    elif signal == 'SELL' and price_change < -0.001:  # Цена упала больше чем на 0.1%
                        is_winning = True
                    
                    signals.append({
                        'timestamp': data.index[i],
                        'signal': signal,
                        'confidence': confidence,
                        'current_price': current_price,
                        'next_price': next_price,
                        'price_change': price_change,
                        'is_winning': is_winning
                    })
                    
                    total_signals += 1
                    if is_winning:
                        winning_signals += 1
                        
            except Exception as e:
                logger.error(f"Ошибка анализа свечи {i} для {symbol}: {e}")
                continue
        
        # Рассчитываем винрейт
        winrate = (winning_signals / total_signals * 100) if total_signals > 0 else 0
        
        result = {
            'symbol': symbol,
            'total_signals': total_signals,
            'winning_signals': winning_signals,
            'losing_signals': total_signals - winning_signals,
            'winrate': round(winrate, 2),
            'data_points': len(data),
            'test_period_days': self.test_days,
            'signals': signals[-10:] if signals else []  # Последние 10 сигналов для анализа
        }
        
        logger.info(f"📊 {symbol}: {total_signals} сигналов, винрейт {winrate:.2f}%")
        return result
    
    async def run_full_test(self) -> Dict:
        """Запуск полного тестирования для всех символов"""
        logger.info("🚀 Запуск полного тестирования винрейта Lava AI")
        logger.info(f"📊 Символы: {', '.join(self.test_symbols)}")
        logger.info(f"⏱️ Интервал: {self.test_interval}")
        logger.info(f"📅 Период: {self.test_days} дней")
        
        await self.initialize()
        
        results = {}
        total_signals = 0
        total_winning = 0
        
        for symbol in self.test_symbols:
            try:
                result = await self.test_symbol_winrate(symbol)
                results[symbol] = result
                
                total_signals += result['total_signals']
                total_winning += result['winning_signals']
                
            except Exception as e:
                logger.error(f"❌ Ошибка тестирования {symbol}: {e}")
                results[symbol] = {
                    'symbol': symbol,
                    'error': str(e),
                    'total_signals': 0,
                    'winning_signals': 0,
                    'winrate': 0.0
                }
        
        # Общий винрейт
        overall_winrate = (total_winning / total_signals * 100) if total_signals > 0 else 0
        
        summary = {
            'test_date': datetime.now().isoformat(),
            'test_period_days': self.test_days,
            'test_interval': self.test_interval,
            'symbols_tested': len(self.test_symbols),
            'total_signals': total_signals,
            'total_winning': total_winning,
            'overall_winrate': round(overall_winrate, 2),
            'symbol_results': results
        }
        
        # Сохраняем результаты
        await self.save_results(summary)
        
        return summary
    
    async def save_results(self, results: Dict):
        """Сохранение результатов тестирования"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"lava_winrate_test_{timestamp}.json"
        filepath = Path(filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 Результаты сохранены в {filepath}")
    
    def print_results(self, results: Dict):
        """Вывод результатов тестирования"""
        print("\n" + "="*80)
        print("🎯 РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ ВИНРЕЙТА LAVA AI")
        print("="*80)
        
        print(f"📅 Дата тестирования: {results['test_date']}")
        print(f"⏱️ Период: {results['test_period_days']} дней")
        print(f"📊 Интервал: {results['test_interval']}")
        print(f"💰 Символов протестировано: {results['symbols_tested']}")
        
        print(f"\n🎯 ОБЩИЕ РЕЗУЛЬТАТЫ:")
        print(f"   Всего сигналов: {results['total_signals']}")
        print(f"   Успешных сигналов: {results['total_winning']}")
        print(f"   ОБЩИЙ ВИНРЕЙТ: {results['overall_winrate']}%")
        
        print(f"\n📊 РЕЗУЛЬТАТЫ ПО СИМВОЛАМ:")
        for symbol, result in results['symbol_results'].items():
            if 'error' in result:
                print(f"   ❌ {symbol}: ОШИБКА - {result['error']}")
            else:
                print(f"   💰 {symbol}: {result['total_signals']} сигналов, "
                      f"винрейт {result['winrate']}% "
                      f"({result['winning_signals']}/{result['total_signals']})")
        
        print("\n" + "="*80)
    
    async def cleanup(self):
        """Очистка ресурсов"""
        await self.lava_ai.cleanup()

async def main():
    """Основная функция"""
    tester = LavaWinrateTest()
    
    try:
        # Запускаем тестирование
        results = await tester.run_full_test()
        
        # Выводим результаты
        tester.print_results(results)
        
    except KeyboardInterrupt:
        logger.info("❌ Тестирование прервано пользователем")
    except Exception as e:
        logger.error(f"❌ Критическая ошибка: {e}")
    finally:
        await tester.cleanup()

if __name__ == "__main__":
    asyncio.run(main())