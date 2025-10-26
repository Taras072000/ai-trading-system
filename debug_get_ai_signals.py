#!/usr/bin/env python3

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from winrate_test_with_results2 import RealWinrateTester, TestConfig
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def debug_get_ai_signals():
    """Отладка функции get_ai_signals"""
    
    # Создаем конфигурацию с максимально мягкими настройками
    config = TestConfig(
        test_period_days=1,  # Только 1 день для быстрого тестирования
        symbols=['BTCUSDT'],  # Только один символ
        min_confidence=0.001,  # Экстремально низкий порог
        min_consensus_models=1,  # Минимум 1 модель
        use_strict_filters=False,  # Отключить строгие фильтры
        debug_mode=True,
        enabled_ai_models=['trading_ai', 'lava_ai', 'lgbm_ai', 'mistral_ai']
    )
    
    # Создаем тестер
    tester = RealWinrateTester(config)
    
    try:
        # Инициализация
        logger.info("🚀 Инициализация тестера...")
        await tester.initialize()
        
        # Загружаем данные для BTCUSDT
        symbol = 'BTCUSDT'
        logger.info(f"📊 Загружаем данные для {symbol}...")
        data = await tester.load_historical_data(symbol)
        
        if data.empty:
            logger.error(f"❌ Нет данных для {symbol}")
            return
            
        logger.info(f"✅ Загружено {len(data)} строк данных для {symbol}")
        logger.info(f"📅 Период данных: {data.index[0]} - {data.index[-1]}")
        
        # Вызываем get_ai_signals с детальным логированием
        logger.info(f"🔍 Вызываем get_ai_signals для {symbol}...")
        signals = await tester.get_ai_signals(symbol, data)
        
        logger.info(f"🎯 РЕЗУЛЬТАТ: get_ai_signals вернул {len(signals) if signals else 0} сигналов")
        
        if signals:
            for i, signal in enumerate(signals):
                logger.info(f"📈 Сигнал {i+1}:")
                logger.info(f"   - Действие: {signal.final_action}")
                logger.info(f"   - Уверенность: {signal.confidence_avg:.2%}")
                logger.info(f"   - Время: {signal.timestamp}")
                logger.info(f"   - Цена: {signal.price}")
                logger.info(f"   - Сила консенсуса: {signal.consensus_strength}")
                logger.info(f"   - Участвующие модели: {[m.model_name for m in signal.participating_models]}")
        else:
            logger.warning("❌ Нет сигналов!")
            
    except Exception as e:
        logger.error(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug_get_ai_signals())