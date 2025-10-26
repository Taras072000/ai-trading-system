#!/usr/bin/env python3
"""
Тест исправлений AI модулей
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Добавляем путь к модулям
sys.path.append('/Users/mac/Documents/Peper Binance v4')

from ai_modules.lava_ai import LavaAI
from ai_modules.reinforcement_learning_engine import ReinforcementLearningEngine

async def test_ai_modules():
    """Тестирование AI модулей"""
    print("🧪 Тестирование исправлений AI модулей...")
    
    # Создаем тестовые данные
    dates = pd.date_range(start='2024-01-01', periods=100, freq='1H')
    np.random.seed(42)
    
    # Генерируем реалистичные данные
    base_price = 100.0
    prices = []
    for i in range(100):
        if i == 0:
            prices.append(base_price)
        else:
            change = np.random.normal(0, 0.02)  # 2% волатильность
            prices.append(prices[-1] * (1 + change))
    
    data = pd.DataFrame({
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'close': prices,
        'volume': [np.random.randint(1000, 10000) for _ in range(100)]
    }, index=dates)
    
    print(f"📊 Создали тестовые данные: {len(data)} свечей")
    print(f"💰 Цена: {data['close'].iloc[0]:.2f} → {data['close'].iloc[-1]:.2f}")
    
    # Тест LavaAI
    print("\n🌋 Тестирование LavaAI...")
    try:
        lava_ai = LavaAI()
        await lava_ai.initialize()
        
        # Тест generate_trading_signals (асинхронный)
        signals = await lava_ai.generate_trading_signals(data)
        print(f"✅ LavaAI.generate_trading_signals: {signals}")
        
        # Тест analyze_patterns (асинхронный)
        patterns = await lava_ai.analyze_patterns(data)
        print(f"✅ LavaAI.analyze_patterns: {patterns}")
        
    except Exception as e:
        print(f"❌ Ошибка LavaAI: {e}")
        import traceback
        traceback.print_exc()
    
    # Тест ReinforcementLearningEngine
    print("\n🧠 Тестирование ReinforcementLearningEngine...")
    try:
        rl_engine = ReinforcementLearningEngine()
        
        # Тест get_model_weights
        weights = rl_engine.get_model_weights()
        print(f"✅ ReinforcementLearningEngine.get_model_weights: {weights}")
        
        # Тест get_current_weights
        current_weights = rl_engine.get_current_weights()
        print(f"✅ ReinforcementLearningEngine.get_current_weights: {current_weights}")
        
    except Exception as e:
        print(f"❌ Ошибка ReinforcementLearningEngine: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n✅ Тестирование завершено!")

if __name__ == "__main__":
    asyncio.run(test_ai_modules())