#!/usr/bin/env python3
"""
Диагностический скрипт для тестирования всех AI моделей
Проверяет, что каждая модель генерирует BUY/SELL сигналы, а не только HOLD
"""

import asyncio
import sys
import os
import logging
from datetime import datetime, timedelta
import pandas as pd

# Добавляем путь к модулям
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ai_modules.trading_ai import TradingAI
from ai_modules.lava_ai import LavaAI
from ai_modules.lgbm_ai import LGBMAI
from ai_modules.mistral_ai import MistralAI
from ai_modules.multi_ai_orchestrator import MultiAIOrchestrator

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AIModelDiagnostic:
    def __init__(self):
        self.models = {}
        self.test_data = self._generate_test_data()
        
    def _generate_test_data(self):
        """Генерирует тестовые данные для разных рыночных условий"""
        base_price = 45000.0
        
        # Восходящий тренд
        uptrend_data = []
        for i in range(50):
            price = base_price + (i * 100) + (i * 10)  # Постепенный рост
            uptrend_data.append({
                'timestamp': datetime.now() - timedelta(minutes=50-i),
                'open': price - 50,
                'high': price + 100,
                'low': price - 100,
                'close': price,
                'volume': 1000 + (i * 10)
            })
            
        # Нисходящий тренд
        downtrend_data = []
        for i in range(50):
            price = base_price - (i * 100) - (i * 10)  # Постепенное падение
            downtrend_data.append({
                'timestamp': datetime.now() - timedelta(minutes=50-i),
                'open': price + 50,
                'high': price + 100,
                'low': price - 100,
                'close': price,
                'volume': 1000 + (i * 10)
            })
            
        # Боковой тренд
        sideways_data = []
        for i in range(50):
            price = base_price + ((-1) ** i * 50)  # Колебания вокруг базовой цены
            sideways_data.append({
                'timestamp': datetime.now() - timedelta(minutes=50-i),
                'open': price - 25,
                'high': price + 75,
                'low': price - 75,
                'close': price,
                'volume': 1000
            })
            
        return {
            'uptrend': uptrend_data,
            'downtrend': downtrend_data,
            'sideways': sideways_data
        }
    
    def _convert_to_dataframe(self, data_list):
        """Конвертирует список данных в DataFrame"""
        df = pd.DataFrame(data_list)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
        return df
    
    async def initialize_models(self):
        """Инициализация всех AI моделей"""
        logger.info("🔄 Инициализация AI моделей...")
        
        try:
            # TradingAI
            self.models['trading_ai'] = TradingAI()
            logger.info("✅ TradingAI инициализирован")
            
            # LavaAI
            self.models['lava_ai'] = LavaAI()
            logger.info("✅ LavaAI инициализирован")
            
            # LGBMAI
            self.models['lgbm_ai'] = LGBMAI()
            logger.info("✅ LGBMAI инициализирован")
            
            # MistralAI
            self.models['mistral_ai'] = MistralAI()
            await self.models['mistral_ai'].initialize()
            logger.info("✅ MistralAI инициализирован")
            
            # MultiAIOrchestrator
            self.models['orchestrator'] = MultiAIOrchestrator()
            logger.info("✅ MultiAIOrchestrator инициализирован")
            
        except Exception as e:
            logger.error(f"❌ Ошибка инициализации моделей: {e}")
            raise
    
    async def test_trading_ai(self, scenario_name: str, data: list):
        """Тестирование TradingAI"""
        try:
            symbol = "BTCUSDT"
            df = self._convert_to_dataframe(data)
            
            # Правильная сигнатура: analyze_market(symbol, data)
            signal = await self.models['trading_ai'].analyze_market(symbol, df)
            
            return {
                'model': 'TradingAI',
                'scenario': scenario_name,
                'signal': signal.action if hasattr(signal, 'action') else str(signal),
                'confidence': signal.confidence if hasattr(signal, 'confidence') else 0.5,
                'reason': signal.reason if hasattr(signal, 'reason') else 'N/A'
            }
            
        except Exception as e:
            logger.error(f"❌ Ошибка TradingAI для {scenario_name}: {e}")
            return {
                'model': 'TradingAI',
                'scenario': scenario_name,
                'signal': 'ERROR',
                'confidence': 0.0,
                'reason': str(e)
            }
    
    async def test_lava_ai(self, scenario_name: str, data: list):
        """Тестирование LavaAI"""
        try:
            df = self._convert_to_dataframe(data)
            
            # Правильная сигнатура: generate_trading_signals(data)
            signal = self.models['lava_ai'].generate_trading_signals(df)
            
            return {
                'model': 'LavaAI',
                'scenario': scenario_name,
                'signal': signal.get('signal', signal.get('action', 'HOLD')),
                'confidence': signal.get('confidence', 0.5),
                'reason': signal.get('reasoning', signal.get('reason', 'N/A'))
            }
            
        except Exception as e:
            logger.error(f"❌ Ошибка LavaAI для {scenario_name}: {e}")
            return {
                'model': 'LavaAI',
                'scenario': scenario_name,
                'signal': 'ERROR',
                'confidence': 0.0,
                'reason': str(e)
            }
    
    async def test_lgbm_ai(self, scenario_name: str, data: list):
        """Тестирование LGBMAI"""
        try:
            symbol = "BTCUSDT"
            df = self._convert_to_dataframe(data)
            
            # Правильная сигнатура: predict_market_direction(symbol, price_data)
            prediction = await self.models['lgbm_ai'].predict_market_direction(symbol, df)
            
            # Преобразуем direction в торговый сигнал
            direction = prediction.get('direction', 0.0)
            if direction > 0:
                signal = 'BUY'
            elif direction < 0:
                signal = 'SELL'
            else:
                signal = 'HOLD'
            
            return {
                'model': 'LGBMAI',
                'scenario': scenario_name,
                'signal': signal,
                'confidence': prediction.get('confidence', 0.5),
                'reason': prediction.get('reasoning', 'N/A')
            }
            
        except Exception as e:
            logger.error(f"❌ Ошибка LGBMAI для {scenario_name}: {e}")
            return {
                'model': 'LGBMAI',
                'scenario': scenario_name,
                'signal': 'ERROR',
                'confidence': 0.0,
                'reason': str(e)
            }
    
    async def test_mistral_ai(self, scenario_name: str, data: list):
        """Тестирование MistralAI"""
        try:
            symbol = "BTCUSDT"
            current_price = data[-1]['close']
            
            signal = await self.models['mistral_ai'].analyze_trading_opportunity(
                symbol=symbol,
                current_price=current_price,
                price_data=data
            )
            
            return {
                'model': 'MistralAI',
                'scenario': scenario_name,
                'signal': signal,
                'confidence': 0.7,  # MistralAI не возвращает confidence
                'reason': 'AI Analysis'
            }
            
        except Exception as e:
            logger.error(f"❌ Ошибка MistralAI для {scenario_name}: {e}")
            return {
                'model': 'MistralAI',
                'scenario': scenario_name,
                'signal': 'ERROR',
                'confidence': 0.0,
                'reason': str(e)
            }
    
    async def test_orchestrator(self, scenario_name: str, data: list):
        """Тестирование MultiAIOrchestrator"""
        try:
            # Конвертируем данные в DataFrame
            df = self._convert_to_dataframe(data)
            
            # Используем правильный метод analyze_and_decide
            decision = await self.models['orchestrator'].analyze_and_decide(
                symbol="SOLUSDT",
                data=df
            )
            
            # Извлекаем сигнал из решения
            signal = decision.action if hasattr(decision, 'action') else 'HOLD'
            confidence = decision.confidence if hasattr(decision, 'confidence') else 0.0
            reasoning = decision.reasoning if hasattr(decision, 'reasoning') else 'No reasoning'
            
            # Добавляем детальную отладочную информацию
            print(f"\n🔍 Детали Orchestrator для {scenario_name}:")
            print(f"  Финальный сигнал: {signal}")
            print(f"  Уверенность: {confidence:.3f}")
            
            # Попробуем получить детали из reasoning
            if "AI Сигналы:" in reasoning:
                signals_part = reasoning.split("AI Сигналы:")[1] if "AI Сигналы:" in reasoning else ""
                print(f"  Индивидуальные сигналы: {signals_part.strip()}")
            
            return {
                'model': 'Orchestrator',
                'scenario': scenario_name,
                'signal': signal,
                'confidence': confidence,
                'reason': reasoning
            }
            
        except Exception as e:
            logger.error(f"❌ Ошибка Orchestrator для {scenario_name}: {e}")
            import traceback
            print(f"  Полная ошибка: {traceback.format_exc()}")
            return {
                'model': 'Orchestrator',
                'scenario': scenario_name,
                'signal': 'ERROR',
                'confidence': 0.0,
                'reason': str(e)
            }
    
    async def run_comprehensive_test(self):
        """Запуск полного тестирования всех моделей"""
        logger.info("🚀 Начинаем комплексное тестирование AI моделей...")
        
        await self.initialize_models()
        
        results = []
        
        for scenario_name, data in self.test_data.items():
            logger.info(f"\n📊 Тестирование сценария: {scenario_name.upper()}")
            logger.info(f"Данные: {len(data)} свечей, цена от {data[0]['close']:.2f} до {data[-1]['close']:.2f}")
            
            # Тестируем каждую модель
            test_functions = [
                self.test_trading_ai,
                self.test_lava_ai,
                self.test_lgbm_ai,
                self.test_mistral_ai,
                self.test_orchestrator
            ]
            
            for test_func in test_functions:
                try:
                    result = await test_func(scenario_name, data)
                    results.append(result)
                    
                    signal_emoji = "🟢" if result['signal'] == 'BUY' else "🔴" if result['signal'] == 'SELL' else "🟡" if result['signal'] == 'HOLD' else "❌"
                    logger.info(f"{signal_emoji} {result['model']}: {result['signal']} (confidence: {result['confidence']:.2f})")
                    
                except Exception as e:
                    logger.error(f"❌ Ошибка тестирования {test_func.__name__} для {scenario_name}: {e}")
        
        return results
    
    def analyze_results(self, results):
        """Анализ результатов тестирования"""
        logger.info("\n" + "="*60)
        logger.info("📈 АНАЛИЗ РЕЗУЛЬТАТОВ ТЕСТИРОВАНИЯ")
        logger.info("="*60)
        
        # Группируем по моделям
        models_stats = {}
        for result in results:
            model = result['model']
            if model not in models_stats:
                models_stats[model] = {
                    'total': 0,
                    'buy': 0,
                    'sell': 0,
                    'hold': 0,
                    'error': 0,
                    'scenarios': {}
                }
            
            models_stats[model]['total'] += 1
            signal = result['signal']
            
            if signal in ['BUY', 'LONG']:
                models_stats[model]['buy'] += 1
            elif signal in ['SELL', 'SHORT']:
                models_stats[model]['sell'] += 1
            elif signal == 'HOLD':
                models_stats[model]['hold'] += 1
            else:
                models_stats[model]['error'] += 1
            
            # Статистика по сценариям
            scenario = result['scenario']
            if scenario not in models_stats[model]['scenarios']:
                models_stats[model]['scenarios'][scenario] = []
            models_stats[model]['scenarios'][scenario].append(signal)
        
        # Выводим статистику
        for model, stats in models_stats.items():
            logger.info(f"\n🤖 {model}:")
            logger.info(f"   Всего тестов: {stats['total']}")
            logger.info(f"   🟢 BUY:  {stats['buy']} ({stats['buy']/stats['total']*100:.1f}%)")
            logger.info(f"   🔴 SELL: {stats['sell']} ({stats['sell']/stats['total']*100:.1f}%)")
            logger.info(f"   🟡 HOLD: {stats['hold']} ({stats['hold']/stats['total']*100:.1f}%)")
            logger.info(f"   ❌ ERROR: {stats['error']} ({stats['error']/stats['total']*100:.1f}%)")
            
            # Активность модели (не только HOLD)
            active_signals = stats['buy'] + stats['sell']
            activity_rate = active_signals / stats['total'] * 100 if stats['total'] > 0 else 0
            
            if activity_rate < 30:
                logger.warning(f"   ⚠️  НИЗКАЯ АКТИВНОСТЬ: {activity_rate:.1f}% (модель слишком консервативна)")
            else:
                logger.info(f"   ✅ Активность: {activity_rate:.1f}%")
        
        # Проверяем проблемные модели
        logger.info(f"\n🔍 ДИАГНОСТИКА ПРОБЛЕМ:")
        
        problem_models = []
        for model, stats in models_stats.items():
            issues = []
            
            if stats['error'] > 0:
                issues.append(f"Ошибки: {stats['error']}")
            
            if stats['hold'] == stats['total']:
                issues.append("Только HOLD сигналы")
            
            active_rate = (stats['buy'] + stats['sell']) / stats['total'] * 100
            if active_rate < 20:
                issues.append(f"Низкая активность: {active_rate:.1f}%")
            
            if issues:
                problem_models.append(f"{model}: {', '.join(issues)}")
        
        if problem_models:
            logger.warning("❌ ПРОБЛЕМНЫЕ МОДЕЛИ:")
            for problem in problem_models:
                logger.warning(f"   - {problem}")
        else:
            logger.info("✅ Все модели работают корректно!")
        
        return models_stats

async def main():
    """Главная функция"""
    try:
        diagnostic = AIModelDiagnostic()
        results = await diagnostic.run_comprehensive_test()
        diagnostic.analyze_results(results)
        
        logger.info("\n🎯 Тестирование завершено!")
        
    except Exception as e:
        logger.error(f"❌ Критическая ошибка: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())