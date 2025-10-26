#!/usr/bin/env python3
"""
Скрипт для тестирования и диагностики обученных AI моделей
"""

import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Добавляем текущую директорию в путь
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_collector import BinanceDataCollector, DataManager
from enhanced_indicators import EnhancedTechnicalIndicators

class ModelTester:
    def __init__(self):
        self.data_manager = DataManager()
        self.indicators = EnhancedTechnicalIndicators()
        self.models_dir = "models/trading_ai"
        
    def load_model_files(self, symbol):
        """Загружает все файлы модели для символа"""
        try:
            model_path = f"{self.models_dir}/{symbol}_trading_model.joblib"
            scaler_path = f"{self.models_dir}/{symbol}_scaler.joblib"
            config_path = f"{self.models_dir}/{symbol}_config.json"
            features_path = f"{self.models_dir}/{symbol}_features.json"
            
            if not all(os.path.exists(path) for path in [model_path, scaler_path, config_path, features_path]):
                return None
                
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            
            with open(config_path, 'r') as f:
                config = json.load(f)
                
            with open(features_path, 'r') as f:
                features = json.load(f)
                
            # Проверяем, является ли features списком или словарем
            if isinstance(features, list):
                feature_names = features
            elif isinstance(features, dict) and 'feature_names' in features:
                feature_names = features['feature_names']
            else:
                print(f"⚠️ Неожиданная структура файла признаков для {symbol}")
                return None
                
            return {
                'model': model,
                'scaler': scaler,
                'config': config,
                'features': feature_names
            }
        except Exception as e:
            print(f"❌ Ошибка загрузки модели {symbol}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def prepare_features(self, df, feature_names):
        """Подготавливает признаки для предсказания"""
        try:
            # Создаем DataFrame для признаков
            features_list = []
            
            # Вычисляем индикаторы для каждой строки (скользящее окно)
            for i in range(len(df)):
                # Берем данные до текущего момента (минимум 20 записей для индикаторов)
                start_idx = max(0, i - 100)  # Уменьшаем окно для более быстрой обработки
                window_data = df.iloc[start_idx:i+1].copy()
                
                if len(window_data) >= 1:  # Минимум 1 запись для базовых признаков
                    indicators = self.indicators.calculate_all_indicators(window_data)
                    
                    # Добавляем базовые ценовые данные если их нет в индикаторах
                    current_row = df.iloc[i]
                    if 'open' not in indicators:
                        indicators['open'] = current_row['open']
                    if 'high' not in indicators:
                        indicators['high'] = current_row['high']
                    if 'low' not in indicators:
                        indicators['low'] = current_row['low']
                    if 'close' not in indicators:
                        indicators['close'] = current_row['close']
                    if 'volume' not in indicators:
                        indicators['volume'] = current_row.get('volume', 0)
                    
                    features_list.append(indicators)
                else:
                    # Для первых записей используем базовые значения
                    current_row = df.iloc[i]
                    basic_features = {
                        'open': current_row['open'],
                        'high': current_row['high'], 
                        'low': current_row['low'],
                        'close': current_row['close'],
                        'volume': current_row.get('volume', 0),
                        'hour': 12.0,
                        'day_of_week': 2.0,
                        'day_of_month': 15.0,
                        'price_change': 0.0
                    }
                    features_list.append(basic_features)
            
            # Создаем DataFrame из списка индикаторов
            features_df = pd.DataFrame(features_list, index=df.index)
            
            # Заполняем пропущенные значения
            features_df = features_df.fillna(method='ffill').fillna(0)
            
            # Проверяем наличие всех необходимых признаков
            missing_features = [f for f in feature_names if f not in features_df.columns]
            if missing_features:
                print(f"⚠️ Отсутствующие признаки: {missing_features[:5]}...")
                # Создаем недостающие признаки с нулевыми значениями
                for feature in missing_features:
                    features_df[feature] = 0.0
            
            # Выбираем только нужные признаки
            features_df = features_df[feature_names].copy()
            
            # Заменяем inf и -inf на NaN, затем заполняем
            features_df = features_df.replace([np.inf, -np.inf], np.nan)
            features_df = features_df.fillna(method='ffill').fillna(0)
            
            return features_df
            
        except Exception as e:
            print(f"❌ Ошибка подготовки признаков: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def apply_confidence_thresholds(self, raw_predictions, probabilities, max_probs, config):
        """Применяет адаптивные пороги уверенности для генерации торговых сигналов"""
        try:
            # Получаем пороги из конфигурации или используем адаптивные значения
            min_confidence = config.get('min_confidence', 0.6)
            
            # Анализируем распределение предсказаний и уверенности
            unique_preds, pred_counts = np.unique(raw_predictions, return_counts=True)
            print(f"📊 Исходное распределение предсказаний:")
            for pred, count in zip(unique_preds, pred_counts):
                pred_name = {0: 'HOLD', 1: 'BUY', 2: 'SELL'}.get(pred, f'Class_{pred}')
                print(f"   - {pred_name}: {count} ({count/len(raw_predictions)*100:.1f}%)")
            
            # Адаптивные пороги на основе распределения уверенности
            confidence_percentile_50 = np.percentile(max_probs, 50)
            confidence_percentile_75 = np.percentile(max_probs, 75)
            confidence_percentile_85 = np.percentile(max_probs, 85)
            
            # Используем экстремально низкие пороги для генерации сигналов
            buy_threshold = min(0.25, confidence_percentile_50 * 0.7)  # Экстремально низкий порог
            sell_threshold = min(0.25, confidence_percentile_50 * 0.7)  # Экстремально низкий порог
            
            print(f"📊 Адаптивные пороги уверенности:")
            print(f"   - BUY порог: {buy_threshold:.3f}")
            print(f"   - SELL порог: {sell_threshold:.3f}")
            print(f"   - 50-й процентиль уверенности: {confidence_percentile_50:.3f}")
            print(f"   - 75-й процентиль уверенности: {confidence_percentile_75:.3f}")
            print(f"   - 85-й процентиль уверенности: {confidence_percentile_85:.3f}")
            
            # Создаем копию предсказаний
            adjusted_predictions = raw_predictions.copy()
            
            # Применяем пороги уверенности
            for i in range(len(raw_predictions)):
                prediction = raw_predictions[i]
                confidence = max_probs[i]
                
                # Если это BUY сигнал (класс 1)
                if prediction == 1:
                    if confidence < buy_threshold:
                        adjusted_predictions[i] = 0  # Переводим в HOLD
                        
                # Если это SELL сигнал (класс 2)
                elif prediction == 2:
                    if confidence < sell_threshold:
                        adjusted_predictions[i] = 0  # Переводим в HOLD
                        
                # HOLD сигналы (класс 0) оставляем как есть
            
            # Показываем результат применения порогов
            unique_adj, adj_counts = np.unique(adjusted_predictions, return_counts=True)
            print(f"📊 Распределение после применения порогов:")
            for pred, count in zip(unique_adj, adj_counts):
                pred_name = {0: 'HOLD', 1: 'BUY', 2: 'SELL'}.get(pred, f'Class_{pred}')
                print(f"   - {pred_name}: {count} ({count/len(adjusted_predictions)*100:.1f}%)")
                
            return adjusted_predictions
            
        except Exception as e:
            print(f"⚠️ Ошибка применения порогов уверенности: {e}")
            return raw_predictions  # Возвращаем исходные предсказания при ошибке
    
    def test_model_predictions(self, symbol, days=30):
        """Тестирует модель на последних данных"""
        print(f"\n🔍 Тестирование модели {symbol}...")
        
        # Загружаем модель
        model_data = self.load_model_files(symbol)
        if not model_data:
            print(f"❌ Не удалось загрузить модель для {symbol}")
            return None
            
        try:
            # Получаем свежие данные
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days + 100)  # +100 для индикаторов
            
            # Используем асинхронный метод для получения данных
            import asyncio
            df = asyncio.run(self.data_manager.ensure_data_available(
                symbol=symbol,
                interval='1h',
                days=days + 100,
                force_update=False
            ))
            
            if df is None or len(df) < 100:
                print(f"❌ Недостаточно данных для {symbol}")
                return None
                
            print(f"📊 Загружено {len(df)} записей")
            
            # Подготавливаем признаки
            features_df = self.prepare_features(df, model_data['features'])
            if features_df is None:
                return None
                
            # Берем последние записи для тестирования
            test_data = features_df.tail(days * 24).copy()  # 24 часа в сутках
            
            if len(test_data) == 0:
                print(f"❌ Нет данных для тестирования {symbol}")
                return None
                
            print(f"🧪 Тестируем на {len(test_data)} записях")
            
            # Нормализуем данные
            X_test = model_data['scaler'].transform(test_data)
            
            # Получаем предсказания и вероятности
            raw_predictions = model_data['model'].predict(X_test)
            probabilities = model_data['model'].predict_proba(X_test)
            
            # Вычисляем максимальные вероятности (уверенность)
            max_probs = np.max(probabilities, axis=1)
            
            # Применяем адаптивные пороги уверенности
            predictions = self.apply_confidence_thresholds(raw_predictions, probabilities, max_probs, model_data['config'])
            
            # Анализируем распределение предсказаний
            unique, counts = np.unique(predictions, return_counts=True)
            pred_distribution = dict(zip(unique, counts))
            
            avg_confidence = np.mean(max_probs)
            
            # Подсчитываем сигналы
            hold_signals = np.sum(predictions == 0)
            buy_signals = np.sum(predictions == 1) 
            sell_signals = np.sum(predictions == 2)
            
            results = {
                'symbol': symbol,
                'total_predictions': len(predictions),
                'hold_signals': hold_signals,
                'buy_signals': buy_signals,
                'sell_signals': sell_signals,
                'hold_pct': (hold_signals / len(predictions)) * 100,
                'buy_pct': (buy_signals / len(predictions)) * 100,
                'sell_pct': (sell_signals / len(predictions)) * 100,
                'avg_confidence': avg_confidence,
                'prediction_distribution': pred_distribution,
                'recent_predictions': predictions[-10].tolist(),
                'recent_probabilities': probabilities[-10].tolist(),
                'feature_count': len(model_data['features']),
                'model_type': str(type(model_data['model']).__name__)
            }
            
            return results
            
        except Exception as e:
            print(f"❌ Ошибка тестирования {symbol}: {e}")
            return None
    
    def diagnose_model_issues(self, symbol):
        """Диагностирует проблемы с моделью"""
        print(f"\n🔧 Диагностика модели {symbol}...")
        
        model_data = self.load_model_files(symbol)
        if not model_data:
            return None
            
        try:
            # Проверяем конфигурацию
            config = model_data['config']
            print(f"📋 Конфигурация модели:")
            print(f"   - Целевой winrate: {config.get('target_winrate', 'N/A')}")
            print(f"   - Минимальная уверенность: {config.get('min_confidence', 'N/A')}")
            print(f"   - Risk/Reward: {config.get('risk_reward_ratio', 'N/A')}")
            print(f"   - Stop Loss: {config.get('stop_loss_pct', 'N/A')}%")
            print(f"   - Take Profit: {config.get('take_profit_pct', 'N/A')}%")
            
            # Проверяем количество признаков
            print(f"🔢 Количество признаков: {len(model_data['features'])}")
            
            # Проверяем тип модели
            model = model_data['model']
            print(f"🤖 Тип модели: {type(model).__name__}")
            
            # Если это ансамбль, проверяем его состав
            if hasattr(model, 'estimators_'):
                print(f"🌳 Количество деревьев/оценщиков: {len(model.estimators_)}")
                
            # Проверяем важность признаков (если доступна)
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                top_features_idx = np.argsort(importances)[-10:][::-1]
                print(f"🎯 Топ-10 важных признаков:")
                for i, idx in enumerate(top_features_idx):
                    feature_name = model_data['features'][idx]
                    importance = importances[idx]
                    print(f"   {i+1}. {feature_name}: {importance:.4f}")
                    
            return True
            
        except Exception as e:
            print(f"❌ Ошибка диагностики {symbol}: {e}")
            return None
    
    def run_comprehensive_test(self):
        """Запускает комплексное тестирование всех моделей"""
        print("🚀 КОМПЛЕКСНОЕ ТЕСТИРОВАНИЕ ОБУЧЕННЫХ МОДЕЛЕЙ")
        print("=" * 60)
        
        # Находим все обученные модели
        if not os.path.exists(self.models_dir):
            print(f"❌ Папка с моделями не найдена: {self.models_dir}")
            return
            
        model_files = [f for f in os.listdir(self.models_dir) if f.endswith('_trading_model.joblib')]
        symbols = [f.replace('_trading_model.joblib', '') for f in model_files]
        
        if not symbols:
            print("❌ Обученные модели не найдены")
            return
            
        print(f"📋 Найдено моделей: {len(symbols)}")
        print(f"💰 Символы: {', '.join(symbols)}")
        
        all_results = {}
        
        for symbol in symbols:
            print(f"\n{'='*50}")
            print(f"🔍 ТЕСТИРОВАНИЕ {symbol}")
            print(f"{'='*50}")
            
            # Диагностика модели
            self.diagnose_model_issues(symbol)
            
            # Тестирование предсказаний
            test_results = self.test_model_predictions(symbol, days=7)
            if test_results:
                all_results[symbol] = test_results
                
                print(f"\n📊 РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ {symbol}:")
                print(f"   🔢 Всего предсказаний: {test_results['total_predictions']}")
                print(f"   ⏸️  HOLD сигналов: {test_results['hold_signals']} ({test_results['hold_pct']:.1f}%)")
                print(f"   📈 BUY сигналов: {test_results['buy_signals']} ({test_results['buy_pct']:.1f}%)")
                print(f"   📉 SELL сигналов: {test_results['sell_signals']} ({test_results['sell_pct']:.1f}%)")
                print(f"   🎯 Средняя уверенность: {test_results['avg_confidence']:.3f}")
                print(f"   🤖 Тип модели: {test_results['model_type']}")
                
                # Показываем последние предсказания
                print(f"   📋 Последние 10 предсказаний: {test_results['recent_predictions']}")
        
        # Общая сводка
        print(f"\n{'='*60}")
        print("📈 ОБЩАЯ СВОДКА ТЕСТИРОВАНИЯ")
        print(f"{'='*60}")
        
        if all_results:
            total_predictions = sum(r['total_predictions'] for r in all_results.values())
            total_buy = sum(r['buy_signals'] for r in all_results.values())
            total_sell = sum(r['sell_signals'] for r in all_results.values())
            total_hold = sum(r['hold_signals'] for r in all_results.values())
            avg_confidence = np.mean([r['avg_confidence'] for r in all_results.values()])
            
            print(f"🔢 Всего предсказаний: {total_predictions}")
            print(f"📈 Всего BUY сигналов: {total_buy} ({(total_buy/total_predictions)*100:.1f}%)")
            print(f"📉 Всего SELL сигналов: {total_sell} ({(total_sell/total_predictions)*100:.1f}%)")
            print(f"⏸️  Всего HOLD сигналов: {total_hold} ({(total_hold/total_predictions)*100:.1f}%)")
            print(f"🎯 Средняя уверенность: {avg_confidence:.3f}")
            
            # Сохраняем результаты
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = f"model_test_results_{timestamp}.json"
            
            # Конвертируем numpy типы в стандартные Python типы для JSON
            def convert_numpy_types(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {str(k): convert_numpy_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(item) for item in obj]
                return obj
            
            # Конвертируем результаты
            converted_results = convert_numpy_types(all_results)
            
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(converted_results, f, indent=2, ensure_ascii=False)
                
            print(f"💾 Результаты сохранены в: {results_file}")
            
            # Диагностика проблем
            print(f"\n🔧 ДИАГНОСТИКА ПРОБЛЕМ:")
            
            if total_buy == 0 and total_sell == 0:
                print("❌ КРИТИЧЕСКАЯ ПРОБЛЕМА: Модели не генерируют торговые сигналы!")
                print("   Возможные причины:")
                print("   1. Слишком строгие пороги уверенности")
                print("   2. Несбалансированные данные обучения")
                print("   3. Проблемы с разметкой целевых меток")
                print("   4. Переобучение на класс HOLD")
                
            elif total_buy < total_predictions * 0.01:  # Менее 1% BUY сигналов
                print("⚠️ ПРОБЛЕМА: Очень мало BUY сигналов")
                print("   Рекомендации:")
                print("   1. Снизить порог уверенности для BUY")
                print("   2. Пересмотреть критерии разметки BUY сигналов")
                print("   3. Увеличить вес класса BUY при обучении")
                
            elif total_sell < total_predictions * 0.01:  # Менее 1% SELL сигналов
                print("⚠️ ПРОБЛЕМА: Очень мало SELL сигналов")
                print("   Рекомендации:")
                print("   1. Снизить порог уверенности для SELL")
                print("   2. Пересмотреть критерии разметки SELL сигналов")
                print("   3. Увеличить вес класса SELL при обучении")
                
            if avg_confidence < 0.6:
                print("⚠️ ПРОБЛЕМА: Низкая уверенность моделей")
                print("   Рекомендации:")
                print("   1. Добавить больше данных для обучения")
                print("   2. Улучшить качество признаков")
                print("   3. Настроить гиперпараметры моделей")
        
        else:
            print("❌ Не удалось протестировать ни одну модель")

def main():
    """Главная функция"""
    tester = ModelTester()
    tester.run_comprehensive_test()

if __name__ == "__main__":
    main()