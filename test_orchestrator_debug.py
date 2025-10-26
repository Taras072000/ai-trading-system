#!/usr/bin/env python3
"""
Отладочный скрипт для тестирования MultiAIOrchestrator
"""

import asyncio
import sys
import os
import pandas as pd
from datetime import datetime, timedelta

# Добавляем путь к модулям
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ai_modules.multi_ai_orchestrator import MultiAIOrchestrator

async def test_orchestrator_debug():
    """Детальное тестирование Orchestrator"""
    print("🔍 Отладочное тестирование MultiAIOrchestrator")
    
    # Создаем тестовые данные
    dates = [datetime.now() - timedelta(hours=i) for i in range(100, 0, -1)]
    test_data = pd.DataFrame({
        'timestamp': dates,
        'open': [100 + i * 0.1 for i in range(100)],
        'high': [101 + i * 0.1 for i in range(100)],
        'low': [99 + i * 0.1 for i in range(100)],
        'close': [100.5 + i * 0.1 for i in range(100)],
        'volume': [1000 + i * 10 for i in range(100)]
    })
    
    # Инициализируем Orchestrator
    orchestrator = MultiAIOrchestrator()
    
    try:
        print("\n📊 Анализируем данные...")
        decision = await orchestrator.analyze_and_decide(
            symbol="SOLUSDT",
            data=test_data
        )
        
        print(f"\n✅ Результат анализа:")
        print(f"  Действие: {decision.action}")
        print(f"  Уверенность: {decision.confidence:.3f}")
        print(f"  Размер позиции: {decision.position_size:.3f}")
        print(f"  Стоп-лосс: {decision.stop_loss:.2f}")
        print(f"  Тейк-профиты: {len(decision.take_profits)}")
        
        print(f"\n📝 Обоснование:")
        print(decision.reasoning)
        
        # Получаем историю сигналов
        if orchestrator.signal_history:
            print(f"\n🤖 Индивидуальные сигналы AI модулей:")
            latest_signals = orchestrator.signal_history[-1]
            for signal in latest_signals:
                print(f"  {signal.module_name}: {signal.signal_type} (уверенность: {signal.confidence:.3f})")
                print(f"    Обоснование: {signal.reasoning[:100]}...")
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        await orchestrator.cleanup()

if __name__ == "__main__":
    asyncio.run(test_orchestrator_debug())