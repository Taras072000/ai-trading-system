#!/usr/bin/env python3
"""
Анализ результатов индивидуального тестирования моделей
"""

import json
import os
from datetime import datetime

def analyze_test_results(filename):
    """Анализ результатов тестирования"""
    
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print('🎯 РЕЗУЛЬТАТЫ ИНДИВИДУАЛЬНОГО ТЕСТИРОВАНИЯ МОДЕЛЕЙ')
    print('='*60)

    summary = {}
    
    for model_name, model_data in data.items():
        print(f'\n📊 МОДЕЛЬ: {model_name.upper()}')
        print('-' * 40)
        
        total_signals = 0
        working_pairs = 0
        total_methods = 0
        successful_methods = 0
        
        for symbol, result in model_data.items():
            if 'performance_analysis' in result:
                analysis = result['performance_analysis']
                status = analysis['status']
                signals = analysis['signals_generated']
                success_rate = analysis['success_rate']
                
                print(f'  {symbol}: {status} | Сигналы: {signals} | Успех: {success_rate:.1f}%')
                
                if status == 'working':
                    working_pairs += 1
                total_signals += signals
                total_methods += analysis['total_methods_tested']
                successful_methods += analysis['successful_methods']
                
                # Показываем рекомендации
                if 'recommendations' in result:
                    for rec in result['recommendations']:
                        print(f'    {rec}')
        
        total_pairs = len(model_data)
        overall_success = (working_pairs / total_pairs * 100) if total_pairs > 0 else 0
        method_success = (successful_methods / total_methods * 100) if total_methods > 0 else 0
        
        print(f'  📈 ИТОГО: {working_pairs}/{total_pairs} пар работают ({overall_success:.1f}%)')
        print(f'  🎯 Всего сигналов: {total_signals}')
        print(f'  🔧 Методы: {successful_methods}/{total_methods} работают ({method_success:.1f}%)')
        
        if overall_success == 0:
            status_text = '❌ НЕ РАБОТАЕТ'
        elif overall_success < 50:
            status_text = '⚠️ РАБОТАЕТ ПЛОХО'
        else:
            status_text = '✅ РАБОТАЕТ ХОРОШО'
            
        print(f'  📊 СТАТУС: {status_text}')
        
        summary[model_name] = {
            'working_pairs': working_pairs,
            'total_pairs': total_pairs,
            'success_rate': overall_success,
            'total_signals': total_signals,
            'method_success': method_success,
            'status': status_text
        }
    
    # Общий анализ
    print(f'\n🎯 ОБЩИЙ АНАЛИЗ')
    print('='*60)
    
    working_models = 0
    total_models = len(summary)
    
    for model_name, stats in summary.items():
        if stats['success_rate'] > 0:
            working_models += 1
    
    print(f'📊 Работающие модели: {working_models}/{total_models}')
    
    # Рекомендации по исправлению
    print(f'\n🔧 РЕКОМЕНДАЦИИ ПО ИСПРАВЛЕНИЮ')
    print('='*60)
    
    for model_name, stats in summary.items():
        if stats['success_rate'] == 0:
            print(f'❌ {model_name.upper()}: Модель не работает - нужно проверить методы и API')
        elif stats['success_rate'] < 100:
            print(f'⚠️ {model_name.upper()}: Работает частично - нужна оптимизация')
        else:
            print(f'✅ {model_name.upper()}: Работает хорошо')

if __name__ == "__main__":
    # Найти последний файл результатов
    results_dir = "individual_test_results"
    if os.path.exists(results_dir):
        files = [f for f in os.listdir(results_dir) if f.endswith('.json')]
        if files:
            latest_file = max(files, key=lambda x: os.path.getctime(os.path.join(results_dir, x)))
            analyze_test_results(os.path.join(results_dir, latest_file))
        else:
            print("Файлы результатов не найдены")
    else:
        print("Папка с результатами не найдена")
"""
Анализ результатов индивидуального тестирования моделей
"""

import json
import os
from datetime import datetime

def analyze_test_results(filename):
    """Анализ результатов тестирования"""
    
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print('🎯 РЕЗУЛЬТАТЫ ИНДИВИДУАЛЬНОГО ТЕСТИРОВАНИЯ МОДЕЛЕЙ')
    print('='*60)

    summary = {}
    
    for model_name, model_data in data.items():
        print(f'\n📊 МОДЕЛЬ: {model_name.upper()}')
        print('-' * 40)
        
        total_signals = 0
        working_pairs = 0
        total_methods = 0
        successful_methods = 0
        
        for symbol, result in model_data.items():
            if 'performance_analysis' in result:
                analysis = result['performance_analysis']
                status = analysis['status']
                signals = analysis['signals_generated']
                success_rate = analysis['success_rate']
                
                print(f'  {symbol}: {status} | Сигналы: {signals} | Успех: {success_rate:.1f}%')
                
                if status == 'working':
                    working_pairs += 1
                total_signals += signals
                total_methods += analysis['total_methods_tested']
                successful_methods += analysis['successful_methods']
                
                # Показываем рекомендации
                if 'recommendations' in result:
                    for rec in result['recommendations']:
                        print(f'    {rec}')
        
        total_pairs = len(model_data)
        overall_success = (working_pairs / total_pairs * 100) if total_pairs > 0 else 0
        method_success = (successful_methods / total_methods * 100) if total_methods > 0 else 0
        
        print(f'  📈 ИТОГО: {working_pairs}/{total_pairs} пар работают ({overall_success:.1f}%)')
        print(f'  🎯 Всего сигналов: {total_signals}')
        print(f'  🔧 Методы: {successful_methods}/{total_methods} работают ({method_success:.1f}%)')
        
        if overall_success == 0:
            status_text = '❌ НЕ РАБОТАЕТ'
        elif overall_success < 50:
            status_text = '⚠️ РАБОТАЕТ ПЛОХО'
        else:
            status_text = '✅ РАБОТАЕТ ХОРОШО'
            
        print(f'  📊 СТАТУС: {status_text}')
        
        summary[model_name] = {
            'working_pairs': working_pairs,
            'total_pairs': total_pairs,
            'success_rate': overall_success,
            'total_signals': total_signals,
            'method_success': method_success,
            'status': status_text
        }
    
    # Общий анализ
    print(f'\n🎯 ОБЩИЙ АНАЛИЗ')
    print('='*60)
    
    working_models = 0
    total_models = len(summary)
    
    for model_name, stats in summary.items():
        if stats['success_rate'] > 0:
            working_models += 1
    
    print(f'📊 Работающие модели: {working_models}/{total_models}')
    
    # Рекомендации по исправлению
    print(f'\n🔧 РЕКОМЕНДАЦИИ ПО ИСПРАВЛЕНИЮ')
    print('='*60)
    
    for model_name, stats in summary.items():
        if stats['success_rate'] == 0:
            print(f'❌ {model_name.upper()}: Модель не работает - нужно проверить методы и API')
        elif stats['success_rate'] < 100:
            print(f'⚠️ {model_name.upper()}: Работает частично - нужна оптимизация')
        else:
            print(f'✅ {model_name.upper()}: Работает хорошо')

if __name__ == "__main__":
    # Найти последний файл результатов
    results_dir = "individual_test_results"
    if os.path.exists(results_dir):
        files = [f for f in os.listdir(results_dir) if f.endswith('.json')]
        if files:
            latest_file = max(files, key=lambda x: os.path.getctime(os.path.join(results_dir, x)))
            analyze_test_results(os.path.join(results_dir, latest_file))
        else:
            print("Файлы результатов не найдены")
    else:
        print("Папка с результатами не найдена")