#!/usr/bin/env python3
"""
Простая диагностика сигналов
"""

import asyncio
import sys
import os
from datetime import datetime, timedelta
import pandas as pd

# Добавляем путь к проекту
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from historical_data_manager import HistoricalDataManager
from lava_ai import LavaAI
from reinforcement_learning_engine import ReinforcementLearningEngine

async def main():
    print("🔍 Диагностика генерации сигналов...")
    
    # Инициализация
    data_manager = HistoricalDataManager()
    lava_ai = LavaAI()
    rl_engine = ReinforcementLearningEngine()
    
    symbols = ['BTCUSDT', 'ETHUSDT']
    timeframe = '1h'
    
    # Период тестирования
    end_date = datetime.now()
    start_date = end_date - timedelta(days=3)
    
    for symbol in symbols:
        print(f"\n📊 Анализ {symbol}:")
        
        # Загружаем данные
        data = data_manager.load_data(symbol, timeframe, start_date, end_date)
        if data is None or len(data) == 0:
            print(f"❌ Нет данных для {symbol}")
            continue
            
        print(f"✅ Загружено {len(data)} свечей")
        
        # Генерируем сигналы
        try:
            lava_signals = await lava_ai.generate_signal(symbol, timeframe, data)
            print(f"🤖 LavaAI сигнал: {lava_signals}")
            
            rl_signals = await rl_engine.generate_signal(symbol, timeframe, data)
            print(f"🧠 RL сигнал: {rl_signals}")
            
            # Проверяем консенсус
            if lava_signals and rl_signals:
                if (lava_signals.get('action') == rl_signals.get('action') and 
                    lava_signals.get('action') in ['BUY', 'SELL']):
                    print(f"✅ КОНСЕНСУС: {lava_signals.get('action')}")
                    print(f"   LavaAI confidence: {lava_signals.get('confidence', 0):.3f}")
                    print(f"   RL confidence: {rl_signals.get('confidence', 0):.3f}")
                else:
                    print(f"❌ НЕТ КОНСЕНСУСА: LavaAI={lava_signals.get('action')}, RL={rl_signals.get('action')}")
            else:
                print("❌ Один из сигналов пустой")
                
        except Exception as e:
            print(f"❌ Ошибка генерации сигналов: {e}")

if __name__ == "__main__":
    asyncio.run(main())