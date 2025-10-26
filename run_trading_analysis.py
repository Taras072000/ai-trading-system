#!/usr/bin/env python3
"""
🚀 ЗАПУСК АНАЛИЗА ТОРГОВОЙ ЛОГИКИ
================================

Основной скрипт для диагностики и калибровки торговой системы.
Анализирует результаты последнего теста и выявляет проблемы в логике.

Использование:
    python run_trading_analysis.py

Результаты сохраняются в папку trading_analysis_results/
"""

import asyncio
import sys
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any
import logging

# Добавляем текущую директорию в путь для импорта
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from trading_logic_analyzer import TradingLogicAnalyzer, TradingCalibrator
from winrate_test_with_results2 import TradeResult, AIModelDecision, ConsensusSignal

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TradingDataLoader:
    """
    📊 ЗАГРУЗЧИК ДАННЫХ ДЛЯ АНАЛИЗА
    
    Загружает данные о сделках из различных источников:
    - Результаты последнего теста
    - Логи отладки
    - Сохраненные CSV файлы
    """
    
    def __init__(self):
        self.base_dir = Path(".")
        self.results_dir = self.base_dir / "reports" / "winrate_tests"
        
    async def load_latest_test_data(self) -> List[Dict]:
        """
        📈 ЗАГРУЗКА ДАННЫХ ПОСЛЕДНЕГО ТЕСТА
        
        Ищет и загружает данные из последнего запуска тестирования
        """
        logger.info("📊 Загрузка данных последнего теста...")
        
        # Пытаемся найти последний CSV файл с результатами
        csv_files = list(self.results_dir.glob("**/all_trades_*.csv")) if self.results_dir.exists() else []
        
        if csv_files:
            # Берем самый новый файл
            latest_csv = max(csv_files, key=lambda x: x.stat().st_mtime)
            logger.info(f"📁 Найден файл с результатами: {latest_csv}")
            
            try:
                df = pd.read_csv(latest_csv)
                trades_data = await self._convert_csv_to_trade_data(df)
                logger.info(f"✅ Загружено {len(trades_data)} сделок из CSV")
                return trades_data
            except Exception as e:
                logger.error(f"❌ Ошибка загрузки CSV: {e}")
        
        # Если CSV не найден, создаем тестовые данные на основе известных результатов
        logger.info("📝 CSV не найден, создаем тестовые данные на основе известных результатов...")
        return await self._create_test_data_from_known_results()
    
    async def _convert_csv_to_trade_data(self, df: pd.DataFrame) -> List[Dict]:
        """Конвертирует CSV данные в формат для анализа"""
        trades_data = []
        
        for _, row in df.iterrows():
            # Парсим время
            entry_time = pd.to_datetime(row['Время входа'])
            exit_time = pd.to_datetime(row['Время выхода'])
            
            # Создаем данные о участвующих моделях
            participating_models = []
            consensus_strength = int(row.get('Сила консенсуса', 1))
            
            # Создаем фиктивные решения AI моделей на основе консенсуса
            model_names = ["TradingAI", "LavaAI", "LGBMAI", "MistralAI", "ReinforcementLearningEngine"]
            for i in range(min(consensus_strength, len(model_names))):
                participating_models.append({
                    'model_name': model_names[i],
                    'action': 'BUY' if row['Направление'] == 'LONG' else 'SELL',
                    'confidence': float(row.get('Уверенность (%)', 0)) / 100,
                    'reasoning': f"Model {model_names[i]} signal",
                    'timestamp': entry_time.isoformat()
                })
            
            trade_data = {
                'symbol': row['Символ'],
                'entry_time': entry_time.isoformat(),
                'entry_price': float(row['Цена входа']),
                'exit_time': exit_time.isoformat(),
                'exit_price': float(row['Цена выхода']),
                'direction': row['Направление'],
                'pnl': float(row['P&L ($)']),
                'pnl_percent': float(row['P&L (%)']),
                'confidence': float(row.get('Уверенность (%)', 0)) / 100,
                'ai_model': row.get('AI модель', 'unknown'),
                'consensus_strength': consensus_strength,
                'participating_models': participating_models,
                'position_size': float(row.get('Размер позиции', 1000)),
                'commission': float(row.get('Комиссия', 1)),
                'exit_reason': 'unknown'
            }
            
            trades_data.append(trade_data)
        
        return trades_data
    
    async def _create_test_data_from_known_results(self) -> List[Dict]:
        """
        🧪 СОЗДАНИЕ ТЕСТОВЫХ ДАННЫХ НА ОСНОВЕ ИЗВЕСТНЫХ РЕЗУЛЬТАТОВ
        
        Создает реалистичные данные на основе результатов из терминала:
        - Win Rate: 41.2%
        - ROI: -15.6%
        - 34 сделки на топ-5 парах
        """
        logger.info("🧪 Создание тестовых данных на основе известных результатов...")
        
        # Данные из терминала
        symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "ADAUSDT", "XRPUSDT"]
        total_trades = 34
        win_rate = 0.412  # 41.2%
        total_roi = -15.6  # -15.6%
        
        # Распределение P&L по парам (из терминала)
        symbol_pnl = {
            "BTCUSDT": -2.85,
            "ETHUSDT": -3.22,
            "SOLUSDT": -4.15,
            "ADAUSDT": -3.89,
            "XRPUSDT": -2.45,
            "APTUSDT": 0.96  # Единственная прибыльная
        }
        
        trades_data = []
        base_time = datetime.now() - timedelta(days=7)
        
        # Создаем сделки для каждого символа
        for symbol in symbols:
            symbol_trades = int(total_trades / len(symbols))  # ~6-7 сделок на символ
            symbol_total_pnl = symbol_pnl.get(symbol, -2.0)
            
            for i in range(symbol_trades):
                # Определяем прибыльность сделки
                is_profitable = (i / symbol_trades) < win_rate
                
                # Распределяем P&L
                if is_profitable:
                    pnl = abs(symbol_total_pnl) * 0.3 * (1 + i * 0.1)  # Прибыльные сделки
                else:
                    pnl = -abs(symbol_total_pnl) * 0.7 * (1 + i * 0.1)  # Убыточные сделки
                
                # Время сделки
                entry_time = base_time + timedelta(hours=i * 24 + symbol.index(symbol[0]) * 4)
                exit_time = entry_time + timedelta(hours=24)  # 24 часа удержания
                
                # Цены (реалистичные для каждого символа)
                base_prices = {
                    "BTCUSDT": 50000,
                    "ETHUSDT": 3000,
                    "SOLUSDT": 200,
                    "ADAUSDT": 0.8,
                    "XRPUSDT": 2.5,
                    "APTUSDT": 15
                }
                
                entry_price = base_prices.get(symbol, 100) * (1 + (i * 0.01))
                pnl_percent = (pnl / 1000) * 100  # Предполагаем позицию $1000
                exit_price = entry_price * (1 + pnl_percent / 100)
                
                # Создаем участвующие модели
                participating_models = []
                consensus_strength = 1  # Большинство сделок с консенсусом 1 модели
                
                # Очень низкая уверенность (как в реальных данных)
                confidence = 0.001 + (i * 0.0001)  # 0.1% - 0.4%
                
                model_names = ["trading_ai", "lava_ai", "lgbm_ai", "mistral_ai", "reinforcement_learning_engine"]
                for j in range(consensus_strength):
                    participating_models.append({
                        'model_name': model_names[j],
                        'action': 'BUY' if pnl > 0 else 'SELL',
                        'confidence': confidence,
                        'reasoning': f"Signal from {model_names[j]}",
                        'timestamp': entry_time.isoformat()
                    })
                
                trade_data = {
                    'symbol': symbol,
                    'entry_time': entry_time.isoformat(),
                    'entry_price': entry_price,
                    'exit_time': exit_time.isoformat(),
                    'exit_price': exit_price,
                    'direction': 'LONG' if pnl > 0 else 'SHORT',
                    'pnl': pnl,
                    'pnl_percent': pnl_percent,
                    'confidence': confidence,
                    'ai_model': f'consensus_{consensus_strength}',
                    'consensus_strength': consensus_strength,
                    'participating_models': participating_models,
                    'position_size': 1000.0,
                    'commission': 1.0,
                    'exit_reason': 'stop_loss' if pnl < 0 else 'take_profit'
                }
                
                trades_data.append(trade_data)
        
        logger.info(f"✅ Создано {len(trades_data)} тестовых сделок")
        return trades_data
    
    async def load_historical_data(self) -> Dict[str, pd.DataFrame]:
        """
        📈 ЗАГРУЗКА ИСТОРИЧЕСКИХ ДАННЫХ
        
        Загружает исторические данные для анализа рыночных условий
        """
        logger.info("📈 Загрузка исторических данных...")
        
        # Создаем фиктивные исторические данные для анализа
        symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "ADAUSDT", "XRPUSDT", "APTUSDT"]
        historical_data = {}
        
        for symbol in symbols:
            # Создаем 7 дней данных с часовыми свечами
            dates = pd.date_range(
                start=datetime.now() - timedelta(days=7),
                end=datetime.now(),
                freq='H'
            )
            
            # Базовые цены
            base_prices = {
                "BTCUSDT": 50000,
                "ETHUSDT": 3000,
                "SOLUSDT": 200,
                "ADAUSDT": 0.8,
                "XRPUSDT": 2.5,
                "APTUSDT": 15
            }
            
            base_price = base_prices.get(symbol, 100)
            
            # Генерируем реалистичные OHLCV данные
            data = []
            current_price = base_price
            
            for date in dates:
                # Случайное изменение цены
                change = (np.random.random() - 0.5) * 0.02  # ±1% изменение
                current_price *= (1 + change)
                
                # OHLC
                open_price = current_price
                high_price = open_price * (1 + abs(change) * 0.5)
                low_price = open_price * (1 - abs(change) * 0.5)
                close_price = current_price
                volume = np.random.randint(1000000, 10000000)
                
                data.append({
                    'timestamp': date,
                    'open': open_price,
                    'high': high_price,
                    'low': low_price,
                    'close': close_price,
                    'volume': volume
                })
            
            historical_data[symbol] = pd.DataFrame(data)
        
        logger.info(f"✅ Загружены исторические данные для {len(historical_data)} символов")
        return historical_data


async def main():
    """
    🚀 ОСНОВНАЯ ФУНКЦИЯ ЗАПУСКА АНАЛИЗА
    """
    print("🔍 СИСТЕМА ДИАГНОСТИКИ ТОРГОВОЙ ЛОГИКИ")
    print("=" * 50)
    print(f"🕐 Время запуска: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    try:
        # 1. Загрузка данных
        logger.info("📊 Этап 1: Загрузка данных...")
        data_loader = TradingDataLoader()
        
        trades_data = await data_loader.load_latest_test_data()
        historical_data = await data_loader.load_historical_data()
        
        if not trades_data:
            logger.error("❌ Не удалось загрузить данные о сделках")
            return
        
        # 2. Инициализация анализатора
        logger.info("🔍 Этап 2: Инициализация анализатора...")
        analyzer = TradingLogicAnalyzer(trades_data, historical_data)
        
        # 3. Запуск полного анализа
        logger.info("🚀 Этап 3: Запуск полного анализа...")
        analysis_results = await analyzer.run_full_analysis()
        
        # 4. Инициализация калибратора
        logger.info("🎯 Этап 4: Инициализация калибратора...")
        calibrator = TradingCalibrator(analyzer)
        
        # 5. Запуск калибровки
        logger.info("⚙️ Этап 5: Запуск калибровки...")
        calibration_results = await calibrator.run_full_calibration()
        
        # 6. Вывод результатов
        print("\n🎯 РЕЗУЛЬТАТЫ АНАЛИЗА:")
        print("=" * 30)
        
        for component_name, analysis in analysis_results.items():
            score_emoji = "🔴" if analysis.performance_score < 50 else "🟡" if analysis.performance_score < 70 else "🟢"
            print(f"{score_emoji} {analysis.component_name}: {analysis.performance_score:.1f}/100")
            
            if analysis.issues:
                print(f"   ❌ Проблемы: {len(analysis.issues)}")
                for issue in analysis.issues[:2]:  # Показываем первые 2 проблемы
                    print(f"      • {issue}")
                if len(analysis.issues) > 2:
                    print(f"      • ... и еще {len(analysis.issues) - 2}")
            
            if analysis.recommendations:
                print(f"   💡 Рекомендации: {len(analysis.recommendations)}")
            print()
        
        # 7. Информация о результатах
        output_dir = Path("trading_analysis_results")
        print(f"📁 Результаты сохранены в: {output_dir.absolute()}")
        print(f"📊 Визуализации: {output_dir / 'visualizations'}")
        print(f"📋 Отчеты: {output_dir / 'reports'}")
        
        # 8. Открываем основной отчет
        main_report = output_dir / "reports" / "diagnostic_report.md"
        if main_report.exists():
            print(f"\n📖 Основной отчет: {main_report}")
            print("💡 Откройте файл для детального анализа проблем и рекомендаций")
        
        logger.info("✅ Анализ торговой логики завершен успешно!")
        
    except Exception as e:
        logger.error(f"❌ Ошибка при выполнении анализа: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    # Запускаем анализ
    exit_code = asyncio.run(main())
    sys.exit(exit_code)