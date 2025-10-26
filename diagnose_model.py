#!/usr/bin/env python3
"""
Скрипт для диагностики модели и анализа предсказаний
"""

import asyncio
import joblib
import numpy as np
import pandas as pd
import json
from data_collector import BinanceDataCollector
from enhanced_indicators import EnhancedTechnicalIndicators

async def diagnose_model():
    """Диагностика модели BTCUSDT"""
    print('🔍 ДИАГНОСТИКА МОДЕЛИ BTCUSDT')
    print('=' * 50)
    
    try:
        # Загружаем модель и связанные файлы
        model_path = 'models/trading_ai/BTCUSDT_trading_model.joblib'
        scaler_path = 'models/trading_ai/BTCUSDT_scaler.joblib'
        features_path = 'models/trading_ai/BTCUSDT_features.json'
        
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        
        with open(features_path, 'r') as f:
            feature_names = json.load(f)
        
        print(f'✅ Модель загружена: {type(model).__name__}')
        print(f'📊 Количество признаков: {len(feature_names)}')
        
        # Загружаем тестовые данные
        collector = BinanceDataCollector()
        async with collector:
            data = await collector.get_historical_data('BTCUSDT', '1h', 100)
        indicators = EnhancedTechnicalIndicators()
        
        print(f'📈 Загружено {len(data)} записей данных')
        
        # Подготавливаем признаки для последних 10 записей
        features_list = []
        for i in range(len(data)-10, len(data)):
            window_data = data.iloc[max(0, i-50):i+1]
            if len(window_data) >= 1:
                try:
                    features = indicators.calculate_all_indicators(window_data)
                    features_list.append(features)
                except Exception as e:
                    print(f"⚠️ Ошибка расчета индикаторов для записи {i}: {e}")
                    continue
        
        if not features_list:
            print("❌ Не удалось подготовить признаки")
            return
            
        features_df = pd.DataFrame(features_list)
        features_df = features_df.fillna(method='ffill').fillna(0)
        
        # Выбираем только нужные признаки
        available_features = [f for f in feature_names if f in features_df.columns]
        missing_features = [f for f in feature_names if f not in features_df.columns]
        
        print(f'✅ Доступно признаков: {len(available_features)}/{len(feature_names)}')
        if missing_features:
            print(f'⚠️ Отсутствуют признаки: {missing_features[:5]}...')
        
        X = features_df[available_features]
        
        # Добавляем недостающие признаки как нули
        for feature in missing_features:
            X[feature] = 0
            
        # Переупорядочиваем колонки согласно обученной модели
        X = X[feature_names]
        
        # Масштабируем
        X_scaled = scaler.transform(X)
        
        # Получаем предсказания и вероятности
        predictions = model.predict(X_scaled)
        probabilities = model.predict_proba(X_scaled)
        
        print(f'\n📊 Анализ последних {len(X)} предсказаний:')
        print(f'Предсказания: {predictions}')
        print(f'Уникальные предсказания: {np.unique(predictions)}')
        
        print('\n📈 Вероятности по классам (первые 5 записей):')
        for i, probs in enumerate(probabilities[:5]):
            print(f'  Запись {i+1}: HOLD={probs[0]:.4f}, BUY={probs[1]:.4f}, SELL={probs[2]:.4f}')
        
        print('\n📊 Средние вероятности:')
        mean_probs = np.mean(probabilities, axis=0)
        print(f'  HOLD: {mean_probs[0]:.4f}')
        print(f'  BUY: {mean_probs[1]:.4f}')
        print(f'  SELL: {mean_probs[2]:.4f}')
        
        # Анализ распределения вероятностей
        print('\n📈 Распределение максимальных вероятностей:')
        max_probs = np.max(probabilities, axis=1)
        print(f'  Минимум: {np.min(max_probs):.4f}')
        print(f'  Максимум: {np.max(max_probs):.4f}')
        print(f'  Среднее: {np.mean(max_probs):.4f}')
        print(f'  Медиана: {np.median(max_probs):.4f}')
        
        # Проверяем, есть ли вообще предсказания BUY/SELL с высокой вероятностью
        buy_probs = probabilities[:, 1]
        sell_probs = probabilities[:, 2]
        
        print(f'\n🔍 Анализ вероятностей BUY:')
        print(f'  Максимальная: {np.max(buy_probs):.4f}')
        print(f'  Средняя: {np.mean(buy_probs):.4f}')
        print(f'  Записей с BUY > 0.3: {np.sum(buy_probs > 0.3)}')
        print(f'  Записей с BUY > 0.4: {np.sum(buy_probs > 0.4)}')
        
        print(f'\n🔍 Анализ вероятностей SELL:')
        print(f'  Максимальная: {np.max(sell_probs):.4f}')
        print(f'  Средняя: {np.mean(sell_probs):.4f}')
        print(f'  Записей с SELL > 0.3: {np.sum(sell_probs > 0.3)}')
        print(f'  Записей с SELL > 0.4: {np.sum(sell_probs > 0.4)}')
        
    except Exception as e:
        print(f"❌ Ошибка диагностики: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(diagnose_model())