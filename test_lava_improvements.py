#!/usr/bin/env python3
"""
Тестирование улучшений модуля Lava AI
"""

import sys
import os
import pandas as pd
import numpy as np
import asyncio
from datetime import datetime, timedelta
import logging

# Добавляем путь к модулям
sys.path.append(os.path.join(os.path.dirname(__file__), 'ai_modules'))

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_test_data(days=30, symbol='BTCUSDT'):
    """Генерация тестовых данных для проверки"""
    np.random.seed(42)
    
    # Базовые параметры
    start_price = 50000
    periods = days * 24 * 4  # 15-минутные свечи
    
    # Генерация цен с трендом и волатильностью
    price_changes = np.random.normal(0, 0.002, periods)
    trend = np.linspace(0, 0.1, periods)  # Восходящий тренд
    
    prices = [start_price]
    for i in range(1, periods):
        new_price = prices[-1] * (1 + price_changes[i] + trend[i]/periods)
        prices.append(max(new_price, 1000))  # Минимальная цена
    
    # Генерация OHLC данных
    data = []
    for i in range(periods):
        open_price = prices[i]
        close_price = prices[i] * (1 + np.random.normal(0, 0.001))
        high_price = max(open_price, close_price) * (1 + abs(np.random.normal(0, 0.002)))
        low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, 0.002)))
        volume = np.random.lognormal(15, 1)  # Логнормальное распределение объемов
        
        timestamp = datetime.now() - timedelta(minutes=15*(periods-i))
        
        data.append({
            'timestamp': timestamp,
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': volume,
            'symbol': symbol
        })
    
    return pd.DataFrame(data)

async def test_lava_ai_improvements():
    """Основная функция тестирования"""
    try:
        logger.info("Начинаем тестирование улучшений Lava AI...")
        
        # Импорт модуля
        from lava_ai import LavaAI
        
        # Создание экземпляра
        lava_ai = LavaAI()
        
        # Генерация тестовых данных
        logger.info("Генерация тестовых данных...")
        test_data = generate_test_data(days=30)
        
        # Тестирование различных рыночных условий
        test_scenarios = [
            {
                'name': 'Нормальные условия',
                'data': test_data,
                'description': 'Стандартные рыночные данные с умеренной волатильностью'
            },
            {
                'name': 'Высокая волатильность',
                'data': add_volatility(test_data.copy(), multiplier=3),
                'description': 'Данные с повышенной волатильностью'
            },
            {
                'name': 'Боковой тренд',
                'data': create_sideways_trend(test_data.copy()),
                'description': 'Данные с боковым движением цены'
            },
            {
                'name': 'Низкий объем',
                'data': reduce_volume(test_data.copy(), factor=0.3),
                'description': 'Данные с пониженными объемами'
            }
        ]
        
        results = []
        
        for scenario in test_scenarios:
            logger.info(f"\nТестирование сценария: {scenario['name']}")
            logger.info(f"Описание: {scenario['description']}")
            
            # Генерация сигналов
            signals_result = await lava_ai.generate_trading_signals(scenario['data'])
            
            # Проверяем, что результат является словарем
            if not isinstance(signals_result, dict):
                logger.error(f"Ошибка: generate_trading_signals вернул {type(signals_result)}, ожидался dict")
                signals_result = {}
            
            # Анализ результатов
            analysis = analyze_signals_quality(signals_result, scenario['data'])
            
            result = {
                'scenario': scenario['name'],
                'description': scenario['description'],
                'signal': signals_result.get('signal', 'UNKNOWN'),
                'confidence': signals_result.get('confidence', 0),
                'signal_quality': signals_result.get('signal_quality', 0),
                'market_conditions': signals_result.get('market_conditions', {}),
                'adapted_thresholds': signals_result.get('adapted_thresholds', {}),
                'technical_indicators': signals_result.get('technical_indicators', {}),
                'analysis': analysis
            }
            
            results.append(result)
            
            # Вывод результатов
            print_scenario_results(result)
        
        # Общий анализ
        logger.info("\n" + "="*60)
        logger.info("ОБЩИЙ АНАЛИЗ РЕЗУЛЬТАТОВ")
        logger.info("="*60)
        
        print_summary_analysis(results)
        
        return results
        
    except Exception as e:
        logger.error(f"Ошибка при тестировании: {e}")
        return None

def add_volatility(data, multiplier=2):
    """Добавление волатильности к данным"""
    for i in range(1, len(data)):
        change = (data.loc[i, 'close'] - data.loc[i-1, 'close']) / data.loc[i-1, 'close']
        enhanced_change = change * multiplier
        data.loc[i, 'close'] = data.loc[i-1, 'close'] * (1 + enhanced_change)
        data.loc[i, 'high'] = max(data.loc[i, 'close'], data.loc[i, 'high']) * 1.02
        data.loc[i, 'low'] = min(data.loc[i, 'close'], data.loc[i, 'low']) * 0.98
    return data

def create_sideways_trend(data):
    """Создание бокового тренда"""
    base_price = data['close'].iloc[0]
    for i in range(len(data)):
        noise = np.random.normal(0, 0.005)
        data.loc[i, 'close'] = base_price * (1 + noise)
        data.loc[i, 'high'] = data.loc[i, 'close'] * 1.01
        data.loc[i, 'low'] = data.loc[i, 'close'] * 0.99
    return data

def reduce_volume(data, factor=0.5):
    """Снижение объемов торгов"""
    data['volume'] = data['volume'] * factor
    return data

def analyze_signals_quality(signals_result, data):
    """Анализ качества сгенерированных сигналов"""
    analysis = {}
    
    # Проверка наличия ключевых компонентов
    analysis['has_technical_indicators'] = 'technical_indicators' in signals_result
    analysis['has_market_conditions'] = 'market_conditions' in signals_result
    analysis['has_adapted_thresholds'] = 'adapted_thresholds' in signals_result
    analysis['has_signal_quality'] = 'signal_quality' in signals_result
    
    # Анализ адаптации
    if 'adapted_thresholds' in signals_result:
        adapted = signals_result['adapted_thresholds']
        analysis['threshold_adaptation'] = {
            'rsi_oversold': adapted.get('rsi_oversold', 30),
            'rsi_overbought': adapted.get('rsi_overbought', 70),
            'volume_multiplier': adapted.get('volume_multiplier', 1.5)
        }
    
    # Оценка согласованности
    if 'technical_indicators' in signals_result:
        indicators = signals_result['technical_indicators']
        analysis['indicator_consistency'] = evaluate_indicator_consistency(indicators)
    
    return analysis

def evaluate_indicator_consistency(indicators):
    """Оценка согласованности индикаторов"""
    bullish_count = 0
    bearish_count = 0
    
    # RSI
    rsi = indicators.get('rsi', 50)
    if rsi < 30:
        bullish_count += 1
    elif rsi > 70:
        bearish_count += 1
    
    # MACD
    macd_hist = indicators.get('macd_histogram', 0)
    if macd_hist > 0:
        bullish_count += 1
    elif macd_hist < 0:
        bearish_count += 1
    
    # Bollinger Bands
    bb_position = indicators.get('bb_position', 0.5)
    if bb_position < 0.2:
        bullish_count += 1
    elif bb_position > 0.8:
        bearish_count += 1
    
    total_signals = bullish_count + bearish_count
    if total_signals == 0:
        return 'neutral'
    elif bullish_count > bearish_count:
        return f'bullish_consensus_{bullish_count}/{total_signals}'
    elif bearish_count > bullish_count:
        return f'bearish_consensus_{bearish_count}/{total_signals}'
    else:
        return 'mixed_signals'

def print_scenario_results(result):
    """Вывод результатов сценария"""
    print(f"\n📊 Сценарий: {result['scenario']}")
    print(f"📝 Описание: {result['description']}")
    print(f"🎯 Сигнал: {result['signal']} (уверенность: {result['confidence']:.2f})")
    print(f"⭐ Качество сигнала: {result['signal_quality']:.2f}")
    
    # Рыночные условия
    conditions = result['market_conditions']
    print(f"🌊 Рыночные условия:")
    print(f"   - Волатильность: {conditions.get('volatility', 'unknown')}")
    print(f"   - Тренд: {conditions.get('trend', 'unknown')}")
    print(f"   - Объем: {conditions.get('volume', 'unknown')}")
    print(f"   - Моментум: {conditions.get('momentum', 'unknown')}")
    
    # Технические индикаторы
    indicators = result['technical_indicators']
    print(f"📈 Технические индикаторы:")
    print(f"   - RSI: {indicators.get('rsi', 0):.1f}")
    print(f"   - MACD гистограмма: {indicators.get('macd_histogram', 0):.6f}")
    print(f"   - BB позиция: {indicators.get('bb_position', 0):.2f}")
    print(f"   - Объем (отношение): {indicators.get('volume_ratio', 0):.2f}")
    
    # Анализ качества
    analysis = result['analysis']
    print(f"🔍 Анализ качества:")
    print(f"   - Согласованность индикаторов: {analysis.get('indicator_consistency', 'unknown')}")
    print(f"   - Адаптация порогов: {'✅' if analysis.get('has_adapted_thresholds') else '❌'}")
    print(f"   - Фильтрация сигналов: {'✅' if analysis.get('has_signal_quality') else '❌'}")

def print_summary_analysis(results):
    """Вывод общего анализа"""
    total_scenarios = len(results)
    
    # Статистика по сигналам
    signal_stats = {}
    quality_scores = []
    confidence_scores = []
    
    for result in results:
        signal = result['signal']
        signal_stats[signal] = signal_stats.get(signal, 0) + 1
        quality_scores.append(result['signal_quality'])
        confidence_scores.append(result['confidence'])
    
    print(f"📊 Статистика по {total_scenarios} сценариям:")
    print(f"   - Распределение сигналов: {signal_stats}")
    print(f"   - Среднее качество сигналов: {np.mean(quality_scores):.2f}")
    print(f"   - Средняя уверенность: {np.mean(confidence_scores):.2f}")
    print(f"   - Диапазон качества: {min(quality_scores):.2f} - {max(quality_scores):.2f}")
    print(f"   - Диапазон уверенности: {min(confidence_scores):.2f} - {max(confidence_scores):.2f}")
    
    # Проверка функциональности
    features_check = {
        'Адаптивные пороги': all(r['analysis'].get('has_adapted_thresholds', False) for r in results),
        'Рыночные условия': all(r['analysis'].get('has_market_conditions', False) for r in results),
        'Технические индикаторы': all(r['analysis'].get('has_technical_indicators', False) for r in results),
        'Качество сигналов': all(r['analysis'].get('has_signal_quality', False) for r in results)
    }
    
    print(f"\n✅ Проверка функциональности:")
    for feature, status in features_check.items():
        print(f"   - {feature}: {'✅ Работает' if status else '❌ Не работает'}")
    
    # Рекомендации
    print(f"\n💡 Рекомендации:")
    if np.mean(quality_scores) > 0.7:
        print("   - Качество сигналов высокое, система работает хорошо")
    elif np.mean(quality_scores) > 0.5:
        print("   - Качество сигналов среднее, возможны улучшения")
    else:
        print("   - Качество сигналов низкое, требуется доработка")
    
    if max(confidence_scores) - min(confidence_scores) > 0.4:
        print("   - Хорошая адаптация уверенности к различным условиям")
    else:
        print("   - Слабая адаптация уверенности, возможно улучшение")

if __name__ == "__main__":
    # Запуск тестирования
    asyncio.run(test_lava_ai_improvements())