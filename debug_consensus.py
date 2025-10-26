#!/usr/bin/env python3
"""
Отладка консенсусных сигналов - проверяем, где именно сигналы отклоняются
"""

import asyncio
import pandas as pd
import logging
from datetime import datetime, timedelta
from winrate_test_with_results2 import RealWinrateTester, TestConfig

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def debug_consensus_signals():
    """Отладка консенсусных сигналов"""
    
    # Создаем максимально мягкую конфигурацию
    config = TestConfig(
        test_period_days=3,
        symbols=['BTCUSDT'],
        min_confidence=0.001,  # 0.1%
        min_consensus_models=1,
        use_strict_filters=False,
        debug_mode=True,
        min_volatility=0.001,  # 0.1%
        min_volume_ratio=0.01,  # 1%
        min_trend_strength=0.001,  # 0.1%
        min_volume_spike=0.01,  # 1%
        min_rsi_divergence=0.01,  # 1%
        consensus_weight_threshold=0.001  # 0.1%
    )
    
    logger.info("🚀 Запуск отладки консенсусных сигналов")
    logger.info(f"📊 Конфигурация: use_strict_filters={config.use_strict_filters}")
    logger.info(f"📊 min_confidence={config.min_confidence}")
    logger.info(f"📊 min_consensus_models={config.min_consensus_models}")
    
    # Создаем тестер
    tester = RealWinrateTester(config)
    await tester.initialize()
    
    symbol = 'BTCUSDT'
    logger.info(f"🔍 Загрузка данных для {symbol}")
    
    # Загружаем данные
    data = await tester.load_historical_data(symbol)
    
    if data is None or len(data) < 50:
        logger.error(f"❌ Недостаточно данных для {symbol}: {len(data) if data is not None else 0}")
        return
    
    logger.info(f"✅ Загружено {len(data)} записей для {symbol}")
    logger.info(f"📊 Период: {data.index[0]} - {data.index[-1]}")
    
    # Получаем сигналы AI
    logger.info(f"🤖 Получение AI сигналов для {symbol}")
    signals = await tester.get_ai_signals(symbol, data)
    
    logger.info(f"📈 Получено {len(signals)} консенсусных сигналов")
    
    if len(signals) == 0:
        logger.error("❌ НЕ ПОЛУЧЕНО НИ ОДНОГО КОНСЕНСУСНОГО СИГНАЛА!")
        logger.error("🔍 Проверим отдельные модели...")
        
        # Проверяем каждую модель отдельно
        test_data = data.tail(50)  # Берем последние 50 записей
        
        individual_signals = []
        for model_name in ['trading_ai', 'lava_ai', 'lgbm_ai', 'mistral_ai']:
            logger.info(f"🔍 Тестирование модели {model_name}")
            try:
                decision = await tester.get_individual_ai_signal(model_name, symbol, test_data)
                if decision:
                    logger.info(f"✅ {model_name}: {decision.action} (confidence: {decision.confidence:.3f})")
                    individual_signals.append((model_name, decision))
                else:
                    logger.warning(f"⚠️ {model_name}: Нет решения")
            except Exception as e:
                logger.error(f"❌ {model_name}: Ошибка - {e}")
        
        # Теперь попробуем создать консенсус вручную
        if len(individual_signals) > 0:
            logger.info("🔧 Попытка создания консенсуса вручную...")
            
            # Создаем список решений для консенсуса
            decisions = [signal[1] for signal in individual_signals]
            
            # Вызываем create_consensus_signal напрямую
            try:
                current_price = test_data['close'].iloc[-1]
                current_time = test_data.index[-1]
                
                logger.info(f"📊 Создание консенсуса: цена={current_price:.4f}, время={current_time}")
                logger.info(f"📊 Решения моделей: {[(d.action, d.confidence) for d in decisions]}")
                
                consensus = await tester.create_consensus_signal(
                    symbol, test_data, decisions
                )
                
                if consensus:
                    logger.info(f"✅ КОНСЕНСУС СОЗДАН: {consensus.final_action} (confidence: {consensus.confidence_avg:.3f})")
                else:
                    logger.error("❌ КОНСЕНСУС НЕ СОЗДАН - функция вернула None")
                    
            except Exception as e:
                logger.error(f"❌ Ошибка при создании консенсуса: {e}")
                import traceback
                logger.error(traceback.format_exc())
    else:
        logger.info("✅ КОНСЕНСУСНЫЕ СИГНАЛЫ ПОЛУЧЕНЫ!")
        for i, signal in enumerate(signals, 1):
            logger.info(f"📈 Сигнал {i}: {signal.final_action} на {signal.price:.4f} (уверенность: {signal.confidence_avg:.3f})")

if __name__ == "__main__":
    asyncio.run(debug_consensus_signals())