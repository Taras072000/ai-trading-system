#!/usr/bin/env python3
"""
🤖 ИНДИВИДУАЛЬНЫЙ ТЕСТ AI МОДЕЛЕЙ
Тестирует каждую AI модель отдельно для калибровки параметров
"""

import os
import sys
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np

# Добавляем путь к модулям
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ai_modules.ai_manager import AIManager
from ai_modules.trading_ai import TradingAI
from ai_modules.lava_ai import LavaAI
from ai_modules.lgbm_ai import LGBMAI
from ai_modules.mistral_ai import MistralAI
from ai_modules.reinforcement_learning_engine import ReinforcementLearningEngine
from data_collector import BinanceDataCollector, DataManager

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ModelTestConfig:
    """Конфигурация для тестирования отдельной модели"""
    model_name: str
    min_confidence: float = 0.25
    test_symbols: List[str] = None
    test_days: int = 7
    
    def __post_init__(self):
        if self.test_symbols is None:
            self.test_symbols = ['TAOUSDT', 'CRVUSDT', 'ZRXUSDT', 'APTUSDT', 'SANDUSDT']

@dataclass
class ModelTestResult:
    """Результат тестирования модели"""
    model_name: str
    total_signals: int
    valid_signals: int
    avg_confidence: float
    confidence_distribution: Dict[str, int]
    signal_distribution: Dict[str, int]
    performance_score: float

class IndividualModelTester:
    """Тестер для индивидуальной калибровки AI моделей"""
    
    def __init__(self):
        self.data_manager = DataManager()
        self.models = {}
        self.results = {}
        
    def initialize_models(self):
        """Инициализация всех AI моделей"""
        logger.info("🚀 Инициализация AI моделей для индивидуального тестирования...")
        
        try:
            # Trading AI
            self.models['trading_ai'] = TradingAI()
            logger.info("✅ trading_ai инициализирована")
            
            # Lava AI
            self.models['lava_ai'] = LavaAI()
            logger.info("✅ lava_ai инициализирована")
            
            # LGBM AI
            self.models['lgbm_ai'] = LGBMAI()
            logger.info("✅ lgbm_ai инициализирована")
            
            # Mistral AI
            self.models['mistral_ai'] = MistralAI()
            logger.info("✅ mistral_ai инициализирована")
            
            # Reinforcement Learning Engine
            self.models['reinforcement_learning_engine'] = ReinforcementLearningEngine()
            logger.info("✅ reinforcement_learning_engine инициализирована")
            
        except Exception as e:
            logger.error(f"❌ Ошибка инициализации моделей: {e}")
            raise
    
    async def test_model(self, model_name: str, config: ModelTestConfig) -> ModelTestResult:
        """Тестирование отдельной модели"""
        logger.info(f"🔍 Тестирование модели: {model_name}")
        
        if model_name not in self.models:
            raise ValueError(f"Модель {model_name} не найдена")
        
        model = self.models[model_name]
        total_signals = 0
        valid_signals = 0
        confidences = []
        signal_types = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
        confidence_ranges = {'0-25%': 0, '25-50%': 0, '50-75%': 0, '75-100%': 0}
        
        # Тестируем на каждом символе
        for symbol in config.test_symbols:
            logger.info(f"📊 Тестирование {model_name} на {symbol}...")
            
            try:
                # Получаем данные
                end_time = datetime.now()
                start_time = end_time - timedelta(days=config.test_days)
                
                df = await self.data_manager.ensure_data_available(
                    symbol=symbol,
                    interval='1h',
                    days=config.test_days
                )
                
                if df is None or len(df) < 50:
                    logger.warning(f"⚠️ Недостаточно данных для {symbol}")
                    continue
                
                # Генерируем сигналы для каждого часа
                for i in range(50, len(df), 6):  # Каждые 6 часов
                    try:
                        current_data = df.iloc[:i+1].copy()
                        
                        # Получаем сигнал от модели
                        if model_name == 'trading_ai':
                            signal = model.get_signal(current_data, symbol)
                        elif model_name == 'lava_ai':
                            signal = model.get_signal(current_data, symbol)
                        elif model_name == 'lgbm_ai':
                            signal = model.get_signal(current_data, symbol)
                        elif model_name == 'mistral_ai':
                            signal = model.get_signal(current_data, symbol)
                        elif model_name == 'reinforcement_learning_engine':
                            # Для RL нужны сигналы других моделей
                            other_signals = {}
                            for other_model_name, other_model in self.models.items():
                                if other_model_name != 'reinforcement_learning_engine':
                                    try:
                                        other_signal = other_model.get_signal(current_data, symbol)
                                        if other_signal and 'confidence' in other_signal:
                                            other_signals[other_model_name] = other_signal
                                    except:
                                        pass
                            
                            if len(other_signals) >= 2:
                                signal = model.get_consensus_signal(other_signals, symbol)
                            else:
                                continue
                        else:
                            continue
                        
                        total_signals += 1
                        
                        if signal and 'confidence' in signal:
                            confidence = signal['confidence']
                            direction = signal.get('direction', 'HOLD')
                            
                            confidences.append(confidence)
                            
                            # Подсчет по типам сигналов
                            if direction == 1 or direction == 'BUY':
                                signal_types['BUY'] += 1
                            elif direction == -1 or direction == 'SELL':
                                signal_types['SELL'] += 1
                            else:
                                signal_types['HOLD'] += 1
                            
                            # Подсчет по диапазонам уверенности
                            if confidence < 0.25:
                                confidence_ranges['0-25%'] += 1
                            elif confidence < 0.50:
                                confidence_ranges['25-50%'] += 1
                            elif confidence < 0.75:
                                confidence_ranges['50-75%'] += 1
                            else:
                                confidence_ranges['75-100%'] += 1
                            
                            # Считаем валидными сигналы выше минимальной уверенности
                            if confidence >= config.min_confidence:
                                valid_signals += 1
                        
                    except Exception as e:
                        logger.debug(f"Ошибка получения сигнала: {e}")
                        continue
                        
            except Exception as e:
                logger.error(f"❌ Ошибка тестирования {symbol}: {e}")
                continue
        
        # Вычисляем метрики
        avg_confidence = np.mean(confidences) if confidences else 0.0
        performance_score = (valid_signals / total_signals * 100) if total_signals > 0 else 0.0
        
        result = ModelTestResult(
            model_name=model_name,
            total_signals=total_signals,
            valid_signals=valid_signals,
            avg_confidence=avg_confidence,
            confidence_distribution=confidence_ranges,
            signal_distribution=signal_types,
            performance_score=performance_score
        )
        
        logger.info(f"✅ {model_name}: {total_signals} сигналов, {valid_signals} валидных ({performance_score:.1f}%)")
        return result
    
    async def run_all_tests(self) -> Dict[str, ModelTestResult]:
        """Запуск тестов для всех моделей"""
        logger.info("🚀 Запуск индивидуального тестирования всех AI моделей...")
        
        self.initialize_models()
        
        config = ModelTestConfig(model_name="", min_confidence=0.25)
        results = {}
        
        for model_name in self.models.keys():
            try:
                config.model_name = model_name
                result = await self.test_model(model_name, config)
                results[model_name] = result
            except Exception as e:
                logger.error(f"❌ Ошибка тестирования {model_name}: {e}")
                continue
        
        return results
    
    def generate_report(self, results: Dict[str, ModelTestResult]):
        """Генерация отчета по результатам тестирования"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print("\n" + "="*80)
        print("🤖 ИНДИВИДУАЛЬНОЕ ТЕСТИРОВАНИЕ AI МОДЕЛЕЙ - ОТЧЕТ")
        print("="*80)
        
        # Сортируем модели по производительности
        sorted_models = sorted(results.items(), key=lambda x: x[1].performance_score, reverse=True)
        
        print(f"\n📊 РЕЙТИНГ МОДЕЛЕЙ ПО ПРОИЗВОДИТЕЛЬНОСТИ:")
        print("┌─────────────────────────────┬─────────┬─────────┬─────────┬─────────────┐")
        print("│           МОДЕЛЬ            │ СИГНАЛЫ │ ВАЛИДНЫЕ│ УСПЕХ % │ УВЕРЕННОСТЬ │")
        print("├─────────────────────────────┼─────────┼─────────┼─────────┼─────────────┤")
        
        for model_name, result in sorted_models:
            print(f"│ {model_name:27} │ {result.total_signals:7} │ {result.valid_signals:7} │ {result.performance_score:6.1f}% │ {result.avg_confidence:10.1%} │")
        
        print("└─────────────────────────────┴─────────┴─────────┴─────────┴─────────────┘")
        
        # Детальная статистика по каждой модели
        print(f"\n📈 ДЕТАЛЬНАЯ СТАТИСТИКА:")
        for model_name, result in results.items():
            print(f"\n🤖 {model_name.upper()}:")
            print(f"   📊 Всего сигналов: {result.total_signals}")
            print(f"   ✅ Валидных сигналов: {result.valid_signals}")
            print(f"   🎯 Средняя уверенность: {result.avg_confidence:.1%}")
            print(f"   📈 Оценка производительности: {result.performance_score:.1f}%")
            
            print(f"   📊 Распределение по уверенности:")
            for range_name, count in result.confidence_distribution.items():
                percentage = (count / result.total_signals * 100) if result.total_signals > 0 else 0
                print(f"      {range_name}: {count} ({percentage:.1f}%)")
            
            print(f"   🎯 Распределение сигналов:")
            for signal_type, count in result.signal_distribution.items():
                percentage = (count / result.total_signals * 100) if result.total_signals > 0 else 0
                print(f"      {signal_type}: {count} ({percentage:.1f}%)")
        
        # Рекомендации
        print(f"\n💡 РЕКОМЕНДАЦИИ ПО КАЛИБРОВКЕ:")
        
        best_model = sorted_models[0][1] if sorted_models else None
        worst_model = sorted_models[-1][1] if sorted_models else None
        
        if best_model:
            print(f"   🥇 Лучшая модель: {best_model.model_name} ({best_model.performance_score:.1f}% успеха)")
        
        if worst_model and worst_model.performance_score < 50:
            print(f"   ⚠️ Требует калибровки: {worst_model.model_name} ({worst_model.performance_score:.1f}% успеха)")
            
            if worst_model.avg_confidence < 0.3:
                print(f"      - Низкая уверенность ({worst_model.avg_confidence:.1%}) - нужна настройка параметров")
            
            low_confidence_signals = worst_model.confidence_distribution.get('0-25%', 0)
            if low_confidence_signals > worst_model.total_signals * 0.5:
                print(f"      - Слишком много сигналов низкой уверенности ({low_confidence_signals}) - пересмотреть алгоритм")
        
        # Сохранение результатов
        report_data = {
            'timestamp': timestamp,
            'results': {name: asdict(result) for name, result in results.items()},
            'summary': {
                'best_model': best_model.model_name if best_model else None,
                'worst_model': worst_model.model_name if worst_model else None,
                'total_models_tested': len(results)
            }
        }
        
        os.makedirs('reports/individual_tests', exist_ok=True)
        report_file = f'reports/individual_tests/individual_model_test_{timestamp}.json'
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Отчет сохранен: {report_file}")

async def main():
    """Основная функция"""
    try:
        tester = IndividualModelTester()
        results = await tester.run_all_tests()
        tester.generate_report(results)
        
    except Exception as e:
        logger.error(f"❌ Критическая ошибка: {e}")
        sys.exit(1)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())