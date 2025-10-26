#!/usr/bin/env python3
"""
Отладочный скрипт для проверки генерации AI сигналов
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from pathlib import Path
import sys

# Добавляем путь к модулям
sys.path.append(str(Path(__file__).parent))

from winrate_test_with_results2 import RealWinrateTester, TestConfig

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def debug_signal_generation():
    """Отладка генерации сигналов"""
    logger.info("🔍 ОТЛАДКА ГЕНЕРАЦИИ СИГНАЛОВ")
    
    # Создаем конфигурацию с максимально мягкими фильтрами
    config = TestConfig(
        test_period_days=7,  # 7 дней для достаточного количества данных
        symbols=['BTCUSDT'],  # Только один символ
        min_confidence=0.001,  # Минимальная уверенность 0.1%
        min_volatility=0.001,  # Минимальная волатильность 0.1%
        min_volume_ratio=0.01,  # Минимальный объем 1%
        min_consensus_models=1,  # Только 1 модель нужна
        consensus_weight_threshold=0.001,  # Минимальный порог
        use_strict_filters=False,  # Отключаем строгие фильтры
        debug_mode=True  # Включаем отладку
    )
    
    # Создаем тестер
    tester = RealWinrateTester(config)
    
    try:
        # Инициализируем
        logger.info("🔧 Инициализация тестера...")
        await tester.initialize()
        
        # Загружаем данные для BTCUSDT
        symbol = 'BTCUSDT'
        logger.info(f"📊 Загрузка данных для {symbol}...")
        data = await tester.load_historical_data(symbol)
        
        if data is None or len(data) < 100:
            logger.error(f"❌ Недостаточно данных для {symbol}: {len(data) if data is not None else 0}")
            return
        
        logger.info(f"✅ Загружено {len(data)} записей для {symbol}")
        logger.info(f"📅 Период: {data.index[0]} - {data.index[-1]}")
        
        # Получаем AI сигналы
        logger.info(f"🤖 Получение AI сигналов для {symbol}...")
        signals = await tester.get_ai_signals(symbol, data)
        
        logger.info(f"📈 Получено {len(signals)} сигналов")
        
        if len(signals) == 0:
            logger.error("❌ НЕ ПОЛУЧЕНО НИ ОДНОГО СИГНАЛА!")
            logger.info("🔍 Проверяем отдельные AI модели...")
            
            # Проверяем каждую модель отдельно
            test_data = data.tail(50)  # Берем последние 50 записей
            
            for model_name in ['trading_ai', 'lava_ai', 'lgbm_ai', 'mistral_ai']:
                logger.info(f"🔍 Тестируем модель {model_name}...")
                try:
                    decision = await tester.get_individual_ai_signal(model_name, symbol, test_data)
                    if decision:
                        logger.info(f"✅ {model_name}: {decision.action} (confidence: {decision.confidence:.3f})")
                    else:
                        logger.warning(f"❌ {model_name}: Нет сигнала")
                except Exception as e:
                    logger.error(f"❌ {model_name}: Ошибка - {e}")
        else:
            logger.info("✅ СИГНАЛЫ ПОЛУЧЕНЫ!")
            for i, signal in enumerate(signals):
                logger.info(f"📈 Сигнал {i+1}: {signal.final_action} на {signal.price:.4f} (уверенность: {signal.confidence_avg:.3f})")
        
    except Exception as e:
        logger.error(f"❌ Ошибка отладки: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug_signal_generation())