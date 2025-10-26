#!/usr/bin/env python3
"""
Тест исправленного LavaAI для проверки генерации сигналов
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Добавляем путь к модулям
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ai_modules.lava_ai import LavaAI

async def test_lava_ai_fixed():
    """Тест исправленного LavaAI"""
    print("🧪 Тестирование исправленного LavaAI...")
    
    # Инициализация LavaAI
    lava_ai = LavaAI()
    await lava_ai.initialize()
    
    # Создание тестовых данных с сильным трендом
    dates = pd.date_range(start='2024-01-01', periods=100, freq='1H')
    
    # Сильный восходящий тренд
    base_price = 50000
    prices = []
    volumes = []
    
    for i in range(100):
        # Восходящий тренд с небольшими колебаниями
        trend_component = i * 50  # Сильный тренд
        noise = np.random.normal(0, 100)  # Небольшой шум
        price = base_price + trend_component + noise
        prices.append(price)
        
        # Высокий объем для подтверждения тренда
        volume = np.random.uniform(1000000, 2000000)
        volumes.append(volume)
    
    # Создание DataFrame
    test_data = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': [p * 1.01 for p in prices],
        'low': [p * 0.99 for p in prices],
        'close': prices,
        'volume': volumes
    })
    
    print(f"📊 Тестовые данные: {len(test_data)} свечей")
    print(f"💰 Цена: {test_data['close'].iloc[0]:.2f} → {test_data['close'].iloc[-1]:.2f}")
    print(f"📈 Изменение: {((test_data['close'].iloc[-1] / test_data['close'].iloc[0]) - 1) * 100:.2f}%")
    
    # Генерация сигналов
    try:
        signals = await lava_ai.generate_trading_signals(test_data)
        
        print("\n🎯 Результат генерации сигналов:")
        print(f"   Сигнал: {signals.get('signal', 'UNKNOWN')}")
        print(f"   Уверенность: {signals.get('confidence', 0)*100:.1f}%")
        print(f"   Signal Score: {signals.get('signal_score', 0):.3f}")
        print(f"   Качество: {signals.get('signal_quality', 0)*100:.1f}%")
        print(f"   Обоснование: {signals.get('reasoning', 'Нет')}")
        
        # Проверяем рыночные условия
        market_conditions = signals.get('market_conditions', {})
        print(f"\n📊 Рыночные условия:")
        for key, value in market_conditions.items():
            print(f"   {key}: {value}")
            
        # Проверяем технические индикаторы
        tech_indicators = signals.get('technical_indicators', {})
        print(f"\n📈 Технические индикаторы:")
        for key, value in tech_indicators.items():
            if isinstance(value, (int, float)):
                print(f"   {key}: {value:.3f}")
            else:
                print(f"   {key}: {value}")
        
        return signals
        
    except Exception as e:
        print(f"❌ Ошибка при генерации сигналов: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    asyncio.run(test_lava_ai_fixed())