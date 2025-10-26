#!/usr/bin/env python3
"""
Простой калибратор для тестирования отдельных AI моделей
"""

import asyncio
import logging
import pandas as pd
from datetime import datetime, timedelta
from ai_modules.ai_manager import AIManager, AIModuleType
from ai_modules.trading_ai import TradingAI
from ai_modules.lava_ai import LavaAI
from ai_modules.lgbm_ai import LGBMAI
from ai_modules.mistral_ai import MistralAI
from ai_modules.reinforcement_learning_engine import ReinforcementLearningEngine
from data_collector import DataManager

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SimpleCalibrator:
    """Простой калибратор для тестирования AI моделей"""
    
    def __init__(self):
        self.data_manager = DataManager()
        self.ai_manager = None
        # Создаем прямые экземпляры моделей
        self.models = {
            'trading_ai': TradingAI(),
            'lava_ai': LavaAI(),
            'lgbm_ai': LGBMAI(),
            'mistral_ai': MistralAI(),
            'reinforcement_learning_engine': ReinforcementLearningEngine()
        }
        
    async def initialize(self):
        """Инициализация"""
        try:
            logger.info("🔄 Инициализация AI Manager...")
            self.ai_manager = AIManager()
            await self.ai_manager.initialize()
            
            logger.info("🔄 Инициализация моделей...")
            for name, model in self.models.items():
                if hasattr(model, 'initialize'):
                    await model.initialize()
                logger.info(f"✅ {name} инициализирована")
                
            logger.info("✅ Все модели инициализированы")
            return True
        except Exception as e:
            logger.error(f"❌ Ошибка инициализации: {e}")
            return False
    
    async def test_lava_ai(self):
        """Тест lava_ai модели"""
        try:
            logger.info("🧪 Тестирование lava_ai...")
            
            # Получаем данные
            symbol = "BTCUSDT"
            data = await self.data_manager.ensure_data_available(symbol, "1h", 168)
            
            if data is None or len(data) == 0:
                logger.error("❌ Нет данных для тестирования")
                return
                
            logger.info(f"📊 Загружено {len(data)} свечей для {symbol}")
            
            # Убеждаемся что индекс - datetime
            if not isinstance(data.index, pd.DatetimeIndex):
                if 'timestamp' in data.columns:
                    data['timestamp'] = pd.to_datetime(data['timestamp'])
                    data.set_index('timestamp', inplace=True)
                else:
                    logger.warning("⚠️ Неправильный формат индекса данных")
                    data.index = pd.to_datetime(data.index)
            
            # Тестируем lava_ai напрямую
            lava_ai = self.models['lava_ai']
            
            # Проверяем методы модели
            if hasattr(lava_ai, 'generate_trading_signals'):
                signal = await lava_ai.generate_trading_signals(data)
                logger.info(f"🎯 Сигнал от lava_ai: {signal}")
            elif hasattr(lava_ai, 'analyze_market_data'):
                analysis = await lava_ai.analyze_market_data(symbol, data)
                logger.info(f"📈 Анализ от lava_ai: {analysis}")
            elif hasattr(lava_ai, 'get_signal'):
                signal = await lava_ai.get_signal(data)
                logger.info(f"🎯 Сигнал от lava_ai: {signal}")
            elif hasattr(lava_ai, 'analyze'):
                analysis = await lava_ai.analyze(data)
                logger.info(f"📈 Анализ от lava_ai: {analysis}")
            else:
                logger.error("❌ lava_ai не имеет известных методов")
                
        except Exception as e:
            logger.error(f"❌ Ошибка тестирования lava_ai: {e}")
            import traceback
            traceback.print_exc()
    
    async def test_all_models(self):
        """Тест всех моделей"""
        try:
            logger.info("🧪 Тестирование всех моделей...")
            
            # Получаем данные
            symbol = "BTCUSDT"
            data = await self.data_manager.ensure_data_available(symbol, "1h", 168)
            
            if data is None or len(data) == 0:
                logger.error("❌ Нет данных для тестирования")
                return
                
            logger.info(f"📊 Загружено {len(data)} свечей для {symbol}")
            
            # Убеждаемся что индекс - datetime
            if not isinstance(data.index, pd.DatetimeIndex):
                if 'timestamp' in data.columns:
                    data['timestamp'] = pd.to_datetime(data['timestamp'])
                    data.set_index('timestamp', inplace=True)
                else:
                    logger.warning("⚠️ Неправильный формат индекса данных")
                    data.index = pd.to_datetime(data.index)
            
            # Тестируем каждую модель
            model_methods = {
                'trading_ai': 'generate_trading_signals',
                'lava_ai': 'generate_trading_signals', 
                'lgbm_ai': 'predict_market_direction',
                'mistral_ai': 'analyze_trading_opportunity',
                'reinforcement_learning_engine': 'get_action'
            }
            
            for name, model in self.models.items():
                try:
                    logger.info(f"🔍 Тестирование {name}...")
                    
                    method_name = model_methods.get(name)
                    if method_name and hasattr(model, method_name):
                        if name == 'lgbm_ai':
                            result = await getattr(model, method_name)(symbol, data)
                        elif name == 'mistral_ai':
                            result = await getattr(model, method_name)(symbol, data)
                        elif name == 'reinforcement_learning_engine':
                            result = await getattr(model, method_name)(symbol, data)
                        else:
                            result = await getattr(model, method_name)(data)
                        logger.info(f"🎯 Результат от {name}: {result}")
                    else:
                        logger.warning(f"⚠️ {name} не имеет метода {method_name}")
                        
                except Exception as e:
                    logger.error(f"❌ Ошибка тестирования {name}: {e}")
                    
        except Exception as e:
            logger.error(f"❌ Ошибка общего тестирования: {e}")
            import traceback
            traceback.print_exc()

async def main():
    """Главная функция"""
    calibrator = SimpleCalibrator()
    
    # Инициализация
    if not await calibrator.initialize():
        logger.error("❌ Не удалось инициализировать калибратор")
        return
    
    # Тестируем lava_ai
    await calibrator.test_lava_ai()
    
    # Тестируем все модели
    await calibrator.test_all_models()

if __name__ == "__main__":
    asyncio.run(main())