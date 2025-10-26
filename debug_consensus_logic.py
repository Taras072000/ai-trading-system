#!/usr/bin/env python3
"""
Диагностический скрипт для проверки логики консенсуса AI моделей
"""

import asyncio
import pandas as pd
from datetime import datetime
from historical_data_manager import HistoricalDataManager
from ai_modules.lava_ai import LavaAI
from ai_modules.reinforcement_learning_engine import ReinforcementLearningEngine

async def test_consensus_logic():
    """Тестирует логику консенсуса с реальными данными"""
    
    print("🔍 ДИАГНОСТИКА КОНСЕНСУСА AI МОДЕЛЕЙ")
    print("=" * 50)
    
    # Загружаем данные
    data_manager = HistoricalDataManager()
    symbol = "BTCUSDT"
    
    print(f"📊 Загружаем данные для {symbol}...")
    data = await data_manager.load_data(symbol, "1h")
    
    if data is None or len(data) < 50:
        print("❌ Недостаточно данных для анализа")
        return
    
    print(f"✅ Загружено {len(data)} записей")
    
    # Инициализируем AI модели
    print("\n🤖 Инициализируем AI модели...")
    lava_ai = LavaAI()
    rl_engine = ReinforcementLearningEngine()
    
    # Получаем сигналы от каждой модели
    print("\n📡 Получаем сигналы от AI моделей...")
    
    try:
        lava_result = await lava_ai.get_signal(symbol, data)
        print(f"🔥 LavaAI: {lava_result}")
    except Exception as e:
        print(f"❌ Ошибка LavaAI: {e}")
        lava_result = None
    
    try:
        rl_result = await rl_engine.get_signal(symbol, data)
        print(f"🧠 RL Engine: {rl_result}")
    except Exception as e:
        print(f"❌ Ошибка RL Engine: {e}")
        rl_result = None
    
    # Анализируем результаты
    print("\n📊 АНАЛИЗ КОНСЕНСУСА:")
    print("-" * 30)
    
    if lava_result and rl_result:
        # Извлекаем действия
        lava_action = "HOLD"
        rl_action = "HOLD"
        
        if hasattr(lava_result, 'action'):
            lava_action = lava_result.action
        elif hasattr(lava_result, 'signal'):
            if lava_result.signal > 0.5:
                lava_action = "BUY"
            elif lava_result.signal < -0.5:
                lava_action = "SELL"
        
        if hasattr(rl_result, 'action'):
            rl_action = rl_result.action
        elif hasattr(rl_result, 'signal'):
            if rl_result.signal > 0.5:
                rl_action = "BUY"
            elif rl_result.signal < -0.5:
                rl_action = "SELL"
        
        print(f"🔥 LavaAI действие: {lava_action}")
        print(f"🧠 RL Engine действие: {rl_action}")
        
        # Проверяем консенсус
        if lava_action == rl_action and lava_action != "HOLD":
            print(f"✅ КОНСЕНСУС ДОСТИГНУТ: {lava_action}")
            print("   Обе модели согласны - сигнал должен быть создан")
        elif lava_action != rl_action:
            print(f"❌ КОНСЕНСУС НЕ ДОСТИГНУТ: {lava_action} vs {rl_action}")
            print("   Модели не согласны - сигнал НЕ будет создан")
            print("   ⚠️  ЭТО ОСНОВНАЯ ПРИЧИНА ОТСУТСТВИЯ СДЕЛОК!")
        else:
            print(f"⚪ Обе модели рекомендуют HOLD")
    
    # Симулируем логику консенсуса из основного файла
    print("\n🔧 СИМУЛЯЦИЯ ЛОГИКИ КОНСЕНСУСА:")
    print("-" * 40)
    
    min_consensus_models = 3
    consensus_weight_threshold = 0.3
    
    print(f"📋 Настройки:")
    print(f"   min_consensus_models = {min_consensus_models}")
    print(f"   consensus_weight_threshold = {consensus_weight_threshold}")
    print(f"   enabled_models = ['trading_ai', 'lava_ai', 'lgbm_ai', 'mistral_ai', 'reinforcement_learning_engine']")
    
    print(f"\n💡 ПРОБЛЕМА:")
    print(f"   Если модели дают разные сигналы (BUY vs SELL),")
    print(f"   то каждое действие получает только 1 голос,")
    print(f"   что меньше требуемых {min_consensus_models} голосов.")
    print(f"   Результат: НИ ОДИН сигнал не проходит консенсус!")
    
    print(f"\n🔧 РЕШЕНИЕ:")
    print(f"   1. Изменить min_consensus_models с 2 на 1")
    print(f"   2. Или добавить больше AI моделей")
    print(f"   3. Или изменить логику консенсуса")

if __name__ == "__main__":
    asyncio.run(test_consensus_logic())