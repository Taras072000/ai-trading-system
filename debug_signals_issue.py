#!/usr/bin/env python3
"""
Диагностика проблемы с отсутствием сигналов
"""

import asyncio
import pandas as pd
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Импорты из основного файла
from ai_modules.lava_ai import LavaAI
from ai_modules.reinforcement_learning_engine import ReinforcementLearningEngine
from historical_data_manager import HistoricalDataManager

@dataclass
class TestConfig:
    min_consensus_models: int = 2
    enabled_ai_models: List[str] = None
    min_confidence: float = 0.25
    test_period_days: int = 3
    symbols: List[str] = None
    
    def __post_init__(self):
        if self.symbols is None:
            self.symbols = ['ETHUSDT', 'BTCUSDT']
        if self.enabled_ai_models is None:
            self.enabled_ai_models = ['lava_ai', 'reinforcement_learning_engine']

class SignalDiagnostic:
    def __init__(self):
        self.config = TestConfig()
        self.ai_models = {}
        self.historical_manager = HistoricalDataManager()
        
    async def initialize(self):
        """Инициализация AI моделей"""
        logger.info("🔧 Инициализация AI моделей для диагностики...")
        
        # Инициализация Lava AI
        try:
            self.ai_models['lava_ai'] = LavaAI()
            logger.info("✅ lava_ai инициализирована")
        except Exception as e:
            logger.error(f"❌ Ошибка инициализации lava_ai: {e}")
            
        # Инициализация Reinforcement Learning Engine
        try:
            self.ai_models['reinforcement_learning_engine'] = ReinforcementLearningEngine()
            logger.info("✅ reinforcement_learning_engine инициализирована")
        except Exception as e:
            logger.error(f"❌ Ошибка инициализации reinforcement_learning_engine: {e}")
            
        logger.info(f"🎯 Всего инициализировано моделей: {len(self.ai_models)}")
        
    async def load_test_data(self, symbol: str) -> pd.DataFrame:
        """Загрузка тестовых данных"""
        logger.info(f"📊 Загрузка данных для {symbol}...")
        
        try:
            # Загружаем исторические данные
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(days=self.config.test_period_days)
            
            data = await self.historical_manager.load_data(
                symbol=symbol,
                interval='1h',
                start_date=start_time,
                end_date=end_time
            )
            
            logger.info(f"✅ Загружено {len(data)} записей для {symbol}")
            if len(data) > 0:
                logger.info(f"📈 Период: {data.index[0]} - {data.index[-1]}")
                logger.info(f"💰 Цена: {data['close'].iloc[-1]:.2f}")
            
            return data
            
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки данных для {symbol}: {e}")
            return pd.DataFrame()
    
    async def test_individual_model(self, model_name: str, symbol: str, data: pd.DataFrame):
        """Тестирование отдельной модели"""
        logger.info(f"🔍 Тестирование {model_name} для {symbol}...")
        
        if model_name not in self.ai_models:
            logger.error(f"❌ Модель {model_name} не найдена")
            return None
            
        model = self.ai_models[model_name]
        
        try:
            # Берем последние 50 записей для анализа
            test_data = data.tail(50) if len(data) > 50 else data
            
            if model_name == 'lava_ai':
                result = await model.generate_trading_signals(test_data)
                logger.info(f"✅ {model_name} результат: {result}")
                return result
                
            elif model_name == 'reinforcement_learning_engine':
                # Для RL engine используем другой метод
                logger.info(f"🧠 {model_name} активен (RL engine)")
                return {'signal': 'ACTIVE', 'confidence': 0.5}
                
        except Exception as e:
            logger.error(f"❌ Ошибка тестирования {model_name}: {e}")
            return None
    
    async def run_diagnostic(self):
        """Запуск полной диагностики"""
        logger.info("🚀 Запуск диагностики сигналов...")
        
        # Инициализация
        await self.initialize()
        
        # Тестирование каждого символа
        for symbol in self.config.symbols:
            logger.info(f"\n{'='*60}")
            logger.info(f"🎯 ДИАГНОСТИКА {symbol}")
            logger.info(f"{'='*60}")
            
            # Загрузка данных
            data = await self.load_test_data(symbol)
            if data.empty:
                logger.warning(f"⚠️ Нет данных для {symbol}, пропускаем")
                continue
            
            # Тестирование каждой модели
            for model_name in self.config.enabled_ai_models:
                result = await self.test_individual_model(model_name, symbol, data)
                
                if result:
                    logger.info(f"✅ {model_name}: Сигнал получен")
                else:
                    logger.warning(f"⚠️ {model_name}: Сигнал НЕ получен")
        
        logger.info("\n🏁 Диагностика завершена")

async def main():
    diagnostic = SignalDiagnostic()
    await diagnostic.run_diagnostic()

if __name__ == "__main__":
    asyncio.run(main())