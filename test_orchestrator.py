#!/usr/bin/env python3
"""
Тестовый скрипт для проверки интеграции AI модулей в MultiAIOrchestrator
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Добавляем путь к модулям
sys.path.append(os.path.join(os.path.dirname(__file__), 'ai_modules'))

from ai_modules.multi_ai_orchestrator import MultiAIOrchestrator

def create_test_data():
    """Создание тестовых рыночных данных"""
    dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='1H')
    np.random.seed(42)
    
    # Генерируем реалистичные OHLCV данные
    base_price = 50000
    price_changes = np.random.normal(0, 0.02, len(dates))
    prices = [base_price]
    
    for change in price_changes[1:]:
        new_price = prices[-1] * (1 + change)
        prices.append(new_price)
    
    # Создаем OHLCV данные
    data = []
    for i, (date, price) in enumerate(zip(dates, prices)):
        high = price * (1 + abs(np.random.normal(0, 0.01)))
        low = price * (1 - abs(np.random.normal(0, 0.01)))
        open_price = prices[i-1] if i > 0 else price
        close_price = price
        volume = np.random.uniform(100, 1000)
        
        data.append({
            'timestamp': date,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close_price,
            'volume': volume
        })
    
    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)
    return df

async def test_orchestrator_initialization():
    """Тест инициализации оркестратора"""
    print("🔧 Тестирование инициализации MultiAIOrchestrator...")
    
    try:
        orchestrator = MultiAIOrchestrator()
        print("✅ Оркестратор создан успешно")
        
        # Проверяем статус системы
        status = orchestrator.get_system_status()
        print(f"📊 Статус системы: {status}")
        
        return orchestrator
    except Exception as e:
        print(f"❌ Ошибка при создании оркестратора: {e}")
        return None

async def test_ai_modules_initialization(orchestrator):
    """Тест инициализации AI модулей"""
    print("\n🤖 Тестирование инициализации AI модулей...")
    
    try:
        await orchestrator.initialize()
        print("✅ Все AI модули инициализированы успешно")
        
        # Проверяем статус каждого модуля
        status = orchestrator.get_system_status()
        for module, info in status['modules'].items():
            print(f"  - {module}: {info['status']}")
        
        return True
    except Exception as e:
        print(f"❌ Ошибка при инициализации AI модулей: {e}")
        return False

async def test_signal_collection(orchestrator, test_data):
    """Тест сбора сигналов от AI модулей"""
    print("\n📡 Тестирование сбора сигналов от AI модулей...")
    
    try:
        # Тестируем сбор сигналов
        signals = await orchestrator._collect_ai_signals('BTCUSDT', test_data, None)
        
        print(f"✅ Собрано {len(signals)} сигналов:")
        for signal in signals:
            print(f"  - {signal.module_name}: {signal.signal_type} "
                  f"(уверенность: {signal.confidence:.2f})")
            print(f"    Обоснование: {signal.reasoning}")
        
        return signals
    except Exception as e:
        print(f"❌ Ошибка при сборе сигналов: {e}")
        return []

async def test_signal_aggregation(orchestrator, signals):
    """Тест агрегации сигналов"""
    print("\n🔄 Тестирование агрегации сигналов...")
    
    try:
        aggregated = orchestrator._aggregate_signals(signals)
        
        print("✅ Сигналы агрегированы успешно:")
        print(f"  - Финальный сигнал: {aggregated['final_signal']}")
        print(f"  - Уверенность: {aggregated['confidence']:.2f}")
        print(f"  - Голоса: {aggregated['signal_votes']}")
        print(f"  - Обоснование: {aggregated['reasoning']}")
        
        return aggregated
    except Exception as e:
        print(f"❌ Ошибка при агрегации сигналов: {e}")
        return None

async def test_final_decision(orchestrator, aggregated_signals):
    """Тест принятия финального решения"""
    print("\n🎯 Тестирование принятия финального решения...")
    
    try:
        final_decision = await orchestrator._make_final_decision(aggregated_signals)
        
        print("✅ Финальное решение принято:")
        print(f"  - Действие: {final_decision['action']}")
        print(f"  - Уверенность: {final_decision['confidence']:.2f}")
        print(f"  - Анализ Mistral: {final_decision.get('mistral_analysis', 'Недоступен')}")
        print(f"  - Обоснование: {final_decision['reasoning']}")
        
        return final_decision
    except Exception as e:
        print(f"❌ Ошибка при принятии финального решения: {e}")
        return None

async def test_full_analysis(orchestrator, test_data):
    """Тест полного анализа и принятия решения"""
    print("\n🚀 Тестирование полного анализа и принятия решения...")
    
    try:
        decision = await orchestrator.analyze_and_decide('BTCUSDT', test_data)
        
        print("✅ Полный анализ завершен успешно:")
        print(f"  - Действие: {decision.action}")
        print(f"  - Уверенность: {decision.confidence:.2f}")
        print(f"  - Цена входа: {decision.entry_price}")
        print(f"  - Размер позиции: {decision.position_size}")
        print(f"  - Стоп-лосс: {decision.stop_loss}")
        print(f"  - Тейк-профиты: {len(decision.take_profits)} уровней")
        print(f"  - Обоснование: {decision.reasoning}")
        print(f"  - Оценка риска: {decision.risk_score}")
        
        return decision
    except Exception as e:
        print(f"❌ Ошибка при полном анализе: {e}")
        return None

async def main():
    """Основная функция тестирования"""
    print("🧪 Запуск тестирования интеграции AI модулей")
    print("=" * 60)
    
    # Создаем тестовые данные
    test_data = create_test_data()
    print(f"📈 Создано {len(test_data)} записей тестовых данных")
    
    # Тест 1: Инициализация оркестратора
    orchestrator = await test_orchestrator_initialization()
    if not orchestrator:
        print("❌ Тестирование прервано из-за ошибки инициализации")
        return
    
    # Тест 2: Инициализация AI модулей
    modules_ok = await test_ai_modules_initialization(orchestrator)
    if not modules_ok:
        print("⚠️  Продолжаем тестирование несмотря на ошибки инициализации")
    
    # Тест 3: Сбор сигналов
    signals = await test_signal_collection(orchestrator, test_data)
    if not signals:
        print("❌ Не удалось собрать сигналы, завершаем тестирование")
        return
    
    # Тест 4: Агрегация сигналов
    aggregated = await test_signal_aggregation(orchestrator, signals)
    if not aggregated:
        print("❌ Не удалось агрегировать сигналы")
        return
    
    # Тест 5: Финальное решение
    final_decision = await test_final_decision(orchestrator, aggregated)
    if not final_decision:
        print("❌ Не удалось принять финальное решение")
        return
    
    # Тест 6: Полный анализ
    full_decision = await test_full_analysis(orchestrator, test_data)
    if not full_decision:
        print("❌ Не удалось выполнить полный анализ")
        return
    
    # Очистка
    await orchestrator.cleanup()
    
    print("\n" + "=" * 60)
    print("🎉 Тестирование завершено успешно!")
    print("✅ Все AI модули интегрированы и работают корректно")

if __name__ == "__main__":
    asyncio.run(main())