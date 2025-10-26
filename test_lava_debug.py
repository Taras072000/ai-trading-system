#!/usr/bin/env python3
"""
Тест для отладки lava_ai - почему генерирует 0 сигналов
"""

import asyncio
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Добавляем путь к модулям
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ai_modules.lava_ai import LavaAI

def create_test_data():
    """Создаем тестовые данные с разными сценариями"""
    
    # Базовые данные
    dates = pd.date_range(start='2024-01-01', periods=100, freq='1H')
    base_price = 100.0
    
    # Сценарий 1: Восходящий тренд
    uptrend_prices = [base_price + i * 0.5 + np.random.normal(0, 0.1) for i in range(100)]
    uptrend_volumes = [1000 + np.random.normal(0, 100) for _ in range(100)]
    
    uptrend_data = pd.DataFrame({
        'timestamp': dates,
        'open': uptrend_prices,
        'high': [p + np.random.uniform(0, 1) for p in uptrend_prices],
        'low': [p - np.random.uniform(0, 1) for p in uptrend_prices],
        'close': uptrend_prices,
        'volume': uptrend_volumes
    })
    
    # Сценарий 2: Нисходящий тренд
    downtrend_prices = [base_price - i * 0.3 + np.random.normal(0, 0.1) for i in range(100)]
    downtrend_volumes = [1000 + np.random.normal(0, 100) for _ in range(100)]
    
    downtrend_data = pd.DataFrame({
        'timestamp': dates,
        'open': downtrend_prices,
        'high': [p + np.random.uniform(0, 0.5) for p in downtrend_prices],
        'low': [p - np.random.uniform(0, 0.5) for p in downtrend_prices],
        'close': downtrend_prices,
        'volume': downtrend_volumes
    })
    
    # Сценарий 3: Сильная волатильность
    volatile_prices = [base_price + np.sin(i/10) * 5 + np.random.normal(0, 2) for i in range(100)]
    volatile_volumes = [1000 + np.random.normal(0, 500) for _ in range(100)]
    
    volatile_data = pd.DataFrame({
        'timestamp': dates,
        'open': volatile_prices,
        'high': [p + np.random.uniform(1, 3) for p in volatile_prices],
        'low': [p - np.random.uniform(1, 3) for p in volatile_prices],
        'close': volatile_prices,
        'volume': volatile_volumes
    })
    
    return {
        'uptrend': uptrend_data,
        'downtrend': downtrend_data,
        'volatile': volatile_data
    }

async def test_lava_ai_signals():
    """Тестируем генерацию сигналов lava_ai"""
    
    print("🔍 Тестирование lava_ai генерации сигналов...")
    
    # Инициализируем lava_ai
    lava_ai = LavaAI()
    await lava_ai.initialize()
    
    # Создаем тестовые данные
    test_scenarios = create_test_data()
    
    for scenario_name, data in test_scenarios.items():
        print(f"\n📊 Тестирование сценария: {scenario_name}")
        print(f"   Данные: {len(data)} записей")
        print(f"   Цена: {data['close'].iloc[0]:.2f} → {data['close'].iloc[-1]:.2f}")
        print(f"   Изменение: {((data['close'].iloc[-1] / data['close'].iloc[0]) - 1) * 100:.2f}%")
        
        try:
            # Генерируем сигнал
            signal_result = await lava_ai.generate_trading_signals(data)
            
            print(f"   ✅ Результат: {signal_result}")
            
            # Анализируем паттерны отдельно
            pattern_result = await lava_ai.analyze_patterns(data)
            print(f"   📈 Паттерны: {pattern_result}")
            
        except Exception as e:
            print(f"   ❌ Ошибка: {e}")
    
    # Тестируем с реальными рыночными условиями
    print(f"\n🔬 Тестирование с экстремальными условиями...")
    
    # Экстремальный рост
    extreme_up_data = test_scenarios['uptrend'].copy()
    extreme_up_data['close'] = extreme_up_data['close'] * 1.1  # +10% рост
    extreme_up_data['volume'] = extreme_up_data['volume'] * 2  # Удвоенный объем
    
    print(f"📈 Экстремальный рост:")
    signal = await lava_ai.generate_trading_signals(extreme_up_data)
    print(f"   Сигнал: {signal}")
    
    # Экстремальное падение
    extreme_down_data = test_scenarios['downtrend'].copy()
    extreme_down_data['close'] = extreme_down_data['close'] * 0.9  # -10% падение
    extreme_down_data['volume'] = extreme_down_data['volume'] * 3  # Утроенный объем
    
    print(f"📉 Экстремальное падение:")
    signal = await lava_ai.generate_trading_signals(extreme_down_data)
    print(f"   Сигнал: {signal}")

if __name__ == "__main__":
    asyncio.run(test_lava_ai_signals())