#!/usr/bin/env python3
"""
Тестирование улучшенного риск-менеджмента
"""

from ai_modules.multi_ai_orchestrator import MultiAIOrchestrator
import pandas as pd
import numpy as np

def test_risk_management():
    """Тестирование улучшенной системы риск-менеджмента"""
    
    # Создаем тестовые данные
    test_data = {
        'close': 50000,
        'atr_percent': 2.5,
        'volume': 1000000
    }

    # Инициализируем оркестратор
    orchestrator = MultiAIOrchestrator()

    # Тестируем для каждого актива
    symbols = ['BTCUSDT']

    print('🧪 Тестирование улучшенного риск-менеджмента')
    print('=' * 60)

    results = {}
    
    for symbol in symbols:
        print(f'\n📊 Тестирование {symbol}:')
        
        try:
            # Тест динамических параметров риска
            dynamic_risk = orchestrator._calculate_dynamic_risk_parameters(symbol, test_data)
            print(f'  🎯 Динамический стоп-лосс: {dynamic_risk["stop_loss_percent"]:.2f}%')
            print(f'  📈 Количество тейк-профитов: {len(dynamic_risk["take_profit_levels"])}')
            print(f'  ⚖️ Соотношение риск/прибыль: 1:{dynamic_risk["risk_reward_ratio"]:.1f}')
            print(f'  📊 Волатильность: {dynamic_risk["volatility_ratio"]:.2f}x')
            
            # Тест влияния комиссий
            commission_impact = orchestrator._calculate_commission_impact(
                symbol, 0.1, test_data['close'], test_data['close'] * 1.02, 'market'
            )
            print(f'  💸 Общая комиссия: ${commission_impact["total_commission"]:.2f}')
            print(f'  💰 Чистая прибыль (2% рост): ${commission_impact["net_pnl"]:.2f}')
            
            # Тест корректировки уровней
            adjusted = orchestrator._adjust_levels_for_commission(
                symbol, test_data['close'], 
                dynamic_risk['stop_loss_percent'], 
                dynamic_risk['take_profit_levels'], 
                0.1
            )
            print(f'  🛡️ Скорректированный SL: {adjusted["adjusted_stop_loss_percent"]:.2f}%')
            print(f'  📈 Первый скорректированный TP: {adjusted["adjusted_take_profit_levels"][0]:.2f}%')
            
            # Проверка соотношения риск/прибыль
            risk_reward = adjusted['adjusted_take_profit_levels'][0] / adjusted['adjusted_stop_loss_percent']
            print(f'  ✅ Фактическое R/R: 1:{risk_reward:.2f}')
            
            if risk_reward >= 2.0:
                print(f'  ✅ Соотношение 1:2 соблюдено!')
                status = "PASSED"
            else:
                print(f'  ❌ Соотношение 1:2 НЕ соблюдено!')
                status = "FAILED"
            
            results[symbol] = {
                'status': status,
                'risk_reward': risk_reward,
                'stop_loss': adjusted["adjusted_stop_loss_percent"],
                'first_tp': adjusted["adjusted_take_profit_levels"][0],
                'commission_impact': commission_impact["total_commission"]
            }
            
        except Exception as e:
            print(f'  ❌ Ошибка тестирования {symbol}: {e}')
            results[symbol] = {'status': 'ERROR', 'error': str(e)}

    print('\n' + '=' * 60)
    print('📋 СВОДКА РЕЗУЛЬТАТОВ:')
    print('=' * 60)
    
    passed = 0
    failed = 0
    errors = 0
    
    for symbol, result in results.items():
        if result['status'] == 'PASSED':
            print(f'✅ {symbol}: R/R = 1:{result["risk_reward"]:.2f}, SL = {result["stop_loss"]:.2f}%, TP1 = {result["first_tp"]:.2f}%')
            passed += 1
        elif result['status'] == 'FAILED':
            print(f'❌ {symbol}: R/R = 1:{result["risk_reward"]:.2f} (< 2.0)')
            failed += 1
        else:
            print(f'🔥 {symbol}: ОШИБКА - {result.get("error", "Неизвестная ошибка")}')
            errors += 1
    
    print(f'\n📊 Итого: ✅ {passed} прошли | ❌ {failed} не прошли | 🔥 {errors} ошибок')
    
    if failed == 0 and errors == 0:
        print('🎉 ВСЕ ТЕСТЫ ПРОШЛИ УСПЕШНО!')
        print('✅ Риск-менеджмент с соотношением 1:2 реализован корректно')
        print('✅ Динамическое управление рисками работает')
        print('✅ Учет комиссий интегрирован')
    else:
        print('⚠️ Требуется доработка системы риск-менеджмента')
    
    return results

if __name__ == "__main__":
    test_risk_management()