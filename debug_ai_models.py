#!/usr/bin/env python3
"""
Скрипт для диагностики и исправления AI моделей
Проверяет почему модели генерируют только HOLD сигналы
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import sys
import os

# Добавляем путь к проекту
sys.path.append('/Users/mac/Documents/Peper Binance v4')

from ai_modules.trading_ai import TradingAI
from ai_modules.lava_ai import LavaAI
from ai_modules.lgbm_ai import LGBMAI
from ai_modules.mistral_ai import MistralAI
from ai_modules.reinforcement_learning_engine import ReinforcementLearningEngine

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def create_test_data():
    """Создаем тестовые данные с трендом"""
    # Создаем данные с восходящим трендом
    dates = pd.date_range(start='2025-10-01', periods=100, freq='1H')
    
    # Базовая цена с трендом
    base_price = 50000
    trend = np.linspace(0, 2000, 100)  # Восходящий тренд +2000
    noise = np.random.normal(0, 100, 100)  # Шум
    
    prices = base_price + trend + noise
    
    # Создаем OHLCV данные
    data = pd.DataFrame({
        'open': prices * (1 + np.random.normal(0, 0.001, 100)),
        'high': prices * (1 + np.random.uniform(0, 0.01, 100)),
        'low': prices * (1 - np.random.uniform(0, 0.01, 100)),
        'close': prices,
        'volume': np.random.uniform(1000, 5000, 100)
    }, index=dates)
    
    # Убеждаемся что high >= max(open, close) и low <= min(open, close)
    data['high'] = np.maximum(data['high'], np.maximum(data['open'], data['close']))
    data['low'] = np.minimum(data['low'], np.minimum(data['open'], data['close']))
    
    return data

async def test_trading_ai(data):
    """Тестируем Trading AI"""
    logger.info("🤖 Тестирование Trading AI...")
    
    trading_ai = TradingAI()
    await trading_ai.initialize()
    
    # Тестируем с разными периодами данных
    for period in [20, 50, 100]:
        test_data = data.tail(period)
        signal = await trading_ai.analyze_market('BTCUSDT', test_data)
        
        logger.info(f"Trading AI ({period} свечей): {signal.action} (confidence: {signal.confidence:.3f}) - {signal.reason}")
        
        if signal.action != 'HOLD':
            return True
    
    return False

async def test_lava_ai(data):
    """Тестируем Lava AI"""
    logger.info("🌋 Тестирование Lava AI...")
    
    lava_ai = LavaAI()
    await lava_ai.initialize()
    
    # Тестируем с разными периодами данных
    for period in [30, 50, 100]:
        test_data = data.tail(period)
        signals = await lava_ai.generate_trading_signals(test_data)
        
        signal = signals.get('signal', 'UNKNOWN')
        confidence = signals.get('confidence', 0.0)
        
        logger.info(f"Lava AI ({period} свечей): {signal} (confidence: {confidence:.3f})")
        
        if signal != 'HOLD':
            return True
    
    return False

async def test_lgbm_ai(data):
    """Тестируем LGBM AI"""
    logger.info("🧠 Тестирование LGBM AI...")
    
    lgbm_ai = LGBMAI()
    await lgbm_ai.initialize()
    
    # Тестируем с разными периодами данных
    for period in [30, 50, 100]:
        test_data = data.tail(period)
        prediction = await lgbm_ai.predict_market_direction('BTCUSDT', test_data)
        
        if prediction:
            direction = prediction.get('direction', 0)
            confidence = prediction.get('confidence', 0.0)
            
            logger.info(f"LGBM AI ({period} свечей): direction={direction:.3f} (confidence: {confidence:.3f})")
            
            if abs(direction) > 0.1:  # Порог для генерации сигнала
                return True
        else:
            logger.info(f"LGBM AI ({period} свечей): None")
    
    return False

async def test_mistral_ai(data):
    """Тестируем Mistral AI"""
    logger.info("🔮 Тестирование Mistral AI...")
    
    mistral_ai = MistralAI()
    await mistral_ai.initialize()
    
    # Тестируем с разными периодами данных
    for period in [20, 30, 50]:
        test_data = data.tail(period)
        current_price = float(test_data['close'].iloc[-1])
        
        # Конвертируем в формат для mistral_ai
        price_data = [
            {
                'timestamp': str(row.name),
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': float(row['volume'])
            }
            for _, row in test_data.iterrows()
        ]
        
        try:
            analysis = await mistral_ai.analyze_trading_opportunity('BTCUSDT', current_price, price_data)
            
            logger.info(f"Mistral AI ({period} свечей): {analysis}")
            
            if analysis and isinstance(analysis, str) and analysis.upper() in ['BUY', 'SELL']:
                return True
        except Exception as e:
            logger.warning(f"Mistral AI ошибка: {e}")
    
    return False

async def test_reinforcement_learning(data):
    """Тестируем Reinforcement Learning Engine"""
    logger.info("🧠 Тестирование Reinforcement Learning Engine...")
    
    rl_engine = ReinforcementLearningEngine()
    await rl_engine.initialize()
    
    # Тестируем логику из winrate_test_with_results2.py
    for period in [20, 30, 50]:
        test_data = data.tail(period)
        
        if len(test_data) >= 10:
            price_change = (test_data['close'].iloc[-1] - test_data['close'].iloc[-5]) / test_data['close'].iloc[-5]
            volume_ratio = test_data['volume'].iloc[-1] / test_data['volume'].iloc[-10:].mean()
            
            logger.info(f"RL Engine ({period} свечей): price_change={price_change:.3f}, volume_ratio={volume_ratio:.2f}")
            
            # Логика из winrate_test_with_results2.py
            if price_change > 0.01 and volume_ratio > 1.2:
                logger.info(f"RL Engine: BUY сигнал")
                return True
            elif price_change < -0.01 and volume_ratio > 1.2:
                logger.info(f"RL Engine: SELL сигнал")
                return True
    
    return False

async def fix_ai_models():
    """Исправляем AI модели для генерации более агрессивных сигналов"""
    logger.info("🔧 Исправление AI моделей...")
    
    # Исправляем Trading AI - делаем пороги более агрессивными
    trading_ai_fix = '''
    # В методе _calculate_trading_signal изменяем логику:
    
    # Старая логика (слишком консервативная):
    if sma_5 > sma_20 and rsi < 70:
        action = 'BUY'
        confidence = min(0.8, (sma_5 - sma_20) / sma_20 * 10)
    
    # Новая логика (более агрессивная):
    if sma_5 > sma_20 and rsi < 80:  # Увеличили порог RSI
        action = 'BUY'
        confidence = min(0.8, max(0.3, (sma_5 - sma_20) / sma_20 * 20))  # Минимум 0.3
    elif sma_5 < sma_20 and rsi > 20:  # Уменьшили порог RSI
        action = 'SELL'
        confidence = min(0.8, max(0.3, (sma_20 - sma_5) / sma_20 * 20))  # Минимум 0.3
    '''
    
    # Исправляем Lava AI - уменьшаем пороги
    lava_ai_fix = '''
    # В методе generate_trading_signals изменяем пороги:
    
    # Старые пороги (слишком высокие):
    if signal_score > 0.4:
        signal = 'BUY'
    elif signal_score < -0.4:
        signal = 'SELL'
    
    # Новые пороги (более низкие):
    if signal_score > 0.2:  # Уменьшили с 0.4 до 0.2
        signal = 'BUY'
    elif signal_score < -0.2:  # Уменьшили с -0.4 до -0.2
        signal = 'SELL'
    '''
    
    # Исправляем LGBM AI - уменьшаем порог direction
    lgbm_ai_fix = '''
    # В методе predict_market_direction изменяем порог:
    
    # Старый порог (слишком высокий):
    if abs(direction) > 0.1:
        action = 'BUY' if direction > 0 else 'SELL'
    
    # Новый порог (более низкий):
    if abs(direction) > 0.05:  # Уменьшили с 0.1 до 0.05
        action = 'BUY' if direction > 0 else 'SELL'
    '''
    
    logger.info("📝 Рекомендации по исправлению:")
    logger.info("1. Trading AI: Увеличить пороги RSI (70→80, 30→20), добавить минимальную confidence")
    logger.info("2. Lava AI: Уменьшить пороги signal_score (0.4→0.2)")
    logger.info("3. LGBM AI: Уменьшить порог direction (0.1→0.05)")
    logger.info("4. Mistral AI: Проверить работу Ollama сервера")
    logger.info("5. RL Engine: Уменьшить пороги price_change и volume_ratio")

async def main():
    """Главная функция"""
    logger.info("🚀 Запуск диагностики AI моделей...")
    
    # Создаем тестовые данные с трендом
    data = await create_test_data()
    logger.info(f"📊 Создали тестовые данные: {len(data)} свечей")
    logger.info(f"📈 Тренд: {data['close'].iloc[0]:.2f} → {data['close'].iloc[-1]:.2f} (+{((data['close'].iloc[-1]/data['close'].iloc[0])-1)*100:.1f}%)")
    
    # Тестируем каждую модель
    results = {}
    
    try:
        results['trading_ai'] = await test_trading_ai(data)
    except Exception as e:
        logger.error(f"Ошибка Trading AI: {e}")
        results['trading_ai'] = False
    
    try:
        results['lava_ai'] = await test_lava_ai(data)
    except Exception as e:
        logger.error(f"Ошибка Lava AI: {e}")
        results['lava_ai'] = False
    
    try:
        results['lgbm_ai'] = await test_lgbm_ai(data)
    except Exception as e:
        logger.error(f"Ошибка LGBM AI: {e}")
        results['lgbm_ai'] = False
    
    try:
        results['mistral_ai'] = await test_mistral_ai(data)
    except Exception as e:
        logger.error(f"Ошибка Mistral AI: {e}")
        results['mistral_ai'] = False
    
    try:
        results['reinforcement_learning'] = await test_reinforcement_learning(data)
    except Exception as e:
        logger.error(f"Ошибка RL Engine: {e}")
        results['reinforcement_learning'] = False
    
    # Выводим результаты
    logger.info("\n" + "="*50)
    logger.info("📊 РЕЗУЛЬТАТЫ ДИАГНОСТИКИ:")
    logger.info("="*50)
    
    working_models = 0
    for model, is_working in results.items():
        status = "✅ РАБОТАЕТ" if is_working else "❌ НЕ РАБОТАЕТ"
        logger.info(f"{model}: {status}")
        if is_working:
            working_models += 1
    
    logger.info(f"\n📈 Работающих моделей: {working_models}/5")
    
    if working_models < 5:
        logger.info("\n🔧 Требуется исправление моделей!")
        await fix_ai_models()
    else:
        logger.info("\n🎉 Все модели работают корректно!")

if __name__ == "__main__":
    asyncio.run(main())