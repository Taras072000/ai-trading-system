#!/usr/bin/env python3
"""
Trading AI Trainer - Скрипт обучения торговой модели
Цель: достижение винрейта 75%+

Использование:
    python trading_ai_trainer.py --symbol BTCUSDT --days 365 --train
    python trading_ai_trainer.py --symbol BTCUSDT --backtest
"""

import asyncio
import argparse
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import joblib
import json
import os
from pathlib import Path

# ML библиотеки
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.inspection import permutation_importance
import xgboost as xgb
import lightgbm as lgb

# Локальные импорты
from enhanced_indicators import EnhancedTechnicalIndicators
from ai_modules.trading_ai import TradingSignal
import config
from data_collector import BinanceDataCollector, DataManager
from improved_labeling_strategy import ImprovedLabelingStrategy

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_ai_trainer.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TradingAITrainer:
    """Класс для обучения торговой модели"""
    
    def __init__(self, symbol: str = "BTCUSDT"):
        self.symbol = symbol
        self.indicators = EnhancedTechnicalIndicators()
        self.models = {}
        self.scalers = {}
        self.feature_names = []
        self.model_dir = Path("models/trading_ai")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Инициализируем менеджер данных для реальной загрузки
        self.data_manager = DataManager()
        
        # Инициализируем улучшенную стратегию разметки
        self.labeling_strategy = ImprovedLabelingStrategy({
            'prediction_horizon': 1,
            'target_class_balance': 0.3,
            'min_return_threshold': 0.002,
            'volatility_multiplier': 1.5,
            'use_percentile_thresholds': True,
            'buy_percentile': 70,  # Более консервативные пороги
            'sell_percentile': 30,
            'momentum_weight': 0.3,
            'volume_weight': 0.2,
        })
        
        # Параметры обучения
        self.config = {
            'lookback_period': 50,  # Период для анализа
            'prediction_horizon': 1,  # Горизонт прогноза (1 = следующая свеча)
            'min_confidence': 0.6,  # Минимальная уверенность для сигнала
            'target_winrate': 0.75,  # Целевой винрейт
            'risk_reward_ratio': 2.0,  # Соотношение риск/прибыль
            'stop_loss_pct': 0.005,  # Стоп-лосс в процентах (снижено с 0.02 до 0.005)
            'take_profit_pct': 0.01,  # Тейк-профит в процентах (снижено с 0.04 до 0.01)
            'use_improved_labeling': True,  # Использовать улучшенную разметку
            'use_class_weights': True,  # Использовать веса классов для балансировки
        }
    
    async def load_market_data(self, days: int = 365) -> pd.DataFrame:
        """Загрузка исторических данных с Binance API"""
        logger.info(f"Загрузка реальных данных для {self.symbol} за {days} дней")
        
        try:
            # Используем DataManager для загрузки реальных данных
            data = await self.data_manager.ensure_data_available(
                symbol=self.symbol,
                interval="1h",  # Часовые свечи для обучения
                days=days,
                force_update=False  # Используем кэш если данные свежие
            )
            
            if data is None or len(data) == 0:
                logger.error(f"Не удалось загрузить данные для {self.symbol}")
                return None
            
            # Убеждаемся что данные отсортированы по времени
            data = data.sort_values('timestamp').reset_index(drop=True)
            
            logger.info(f"Загружено {len(data)} записей для {self.symbol}")
            logger.info(f"Период данных: {data['timestamp'].min()} - {data['timestamp'].max()}")
            
            return data
            
        except Exception as e:
            logger.error(f"Ошибка загрузки данных для {self.symbol}: {str(e)}")
            # В случае ошибки возвращаем None, чтобы обработать в вызывающем коде
            return None
    
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Подготовка признаков для обучения"""
        logger.info("Подготовка признаков...")
        
        features_list = []
        
        for i in range(self.config['lookback_period'], len(data)):
            # Получаем данные для текущего окна
            window_data = data.iloc[i-self.config['lookback_period']:i].copy()
            
            # Вычисляем технические индикаторы
            indicators = self.indicators.calculate_all_indicators(window_data)
            
            # Добавляем временные признаки
            current_time = data.iloc[i]['timestamp']
            indicators['hour'] = current_time.hour
            indicators['day_of_week'] = current_time.weekday()
            indicators['day_of_month'] = current_time.day
            
            # Добавляем ценовые признаки
            current_price = data.iloc[i]['close']
            prev_price = data.iloc[i-1]['close']
            indicators['price_change'] = (current_price - prev_price) / prev_price
            indicators['current_price'] = current_price
            
            # Добавляем индекс для связи с целевой переменной
            indicators['index'] = i
            
            features_list.append(indicators)
        
        features_df = pd.DataFrame(features_list)
        
        # Удаляем строки с NaN
        features_df = features_df.dropna()
        
        logger.info(f"Подготовлено {len(features_df)} образцов с {len(features_df.columns)} признаками")
        return features_df
    
    def balance_indicators(self, X: pd.DataFrame, y: np.ndarray, top_k_features: int = 50) -> Tuple[pd.DataFrame, List[str]]:
        """
        🎯 БАЛАНСИРОВКА ИНДИКАТОРОВ - отбор наиболее значимых признаков
        Цель: Повысить качество данных для обучения, отсеивая слабые индикаторы
        """
        logger.info(f"🔍 Начинаю балансировку индикаторов (отбор {top_k_features} лучших из {len(X.columns)})")
        
        # Удаляем служебные колонки для анализа
        feature_cols = [col for col in X.columns if col not in ['index', 'current_price']]
        X_features = X[feature_cols].copy()
        
        # Заполняем NaN значения медианой
        X_features = X_features.fillna(X_features.median())
        
        # 1. СТАТИСТИЧЕСКИЙ АНАЛИЗ - F-статистика
        logger.info("📊 Анализ статистической значимости (F-статистика)...")
        f_selector = SelectKBest(score_func=f_classif, k=min(top_k_features * 2, len(feature_cols)))
        X_f_selected = f_selector.fit_transform(X_features, y)
        f_scores = f_selector.scores_
        f_selected_features = [feature_cols[i] for i in f_selector.get_support(indices=True)]
        
        # 2. ИНФОРМАЦИОННЫЙ АНАЛИЗ - Mutual Information
        logger.info("🧠 Анализ взаимной информации (Mutual Information)...")
        mi_selector = SelectKBest(score_func=mutual_info_classif, k=min(top_k_features * 2, len(feature_cols)))
        X_mi_selected = mi_selector.fit_transform(X_features, y)
        mi_scores = mi_selector.scores_
        mi_selected_features = [feature_cols[i] for i in mi_selector.get_support(indices=True)]
        
        # 3. МОДЕЛЬНЫЙ АНАЛИЗ - Feature Importance с Random Forest
        logger.info("🌲 Анализ важности признаков (Random Forest)...")
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf_model.fit(X_features, y)
        rf_importance = rf_model.feature_importances_
        
        # 4. КОМБИНИРОВАННЫЙ СКОР
        logger.info("⚖️ Вычисление комбинированного скора важности...")
        feature_scores = {}
        
        for i, feature in enumerate(feature_cols):
            # Нормализуем скоры к диапазону [0, 1]
            f_score_norm = f_scores[i] / (np.max(f_scores) + 1e-8) if i < len(f_scores) else 0
            mi_score_norm = mi_scores[i] / (np.max(mi_scores) + 1e-8) if i < len(mi_scores) else 0
            rf_score_norm = rf_importance[i] / (np.max(rf_importance) + 1e-8)
            
            # Комбинированный скор: 30% F-статистика + 30% MI + 40% RF важность
            combined_score = (f_score_norm * 0.3) + (mi_score_norm * 0.3) + (rf_score_norm * 0.4)
            feature_scores[feature] = {
                'combined_score': combined_score,
                'f_score': f_score_norm,
                'mi_score': mi_score_norm,
                'rf_importance': rf_score_norm
            }
        
        # 5. ОТБОР ЛУЧШИХ ПРИЗНАКОВ
        sorted_features = sorted(feature_scores.items(), key=lambda x: x[1]['combined_score'], reverse=True)
        selected_features = [feature for feature, _ in sorted_features[:top_k_features]]
        
        # Добавляем обратно служебные колонки
        final_features = selected_features + ['index', 'current_price']
        X_balanced = X[final_features].copy()
        
        # 6. ЛОГИРОВАНИЕ РЕЗУЛЬТАТОВ
        logger.info(f"✅ Балансировка завершена! Отобрано {len(selected_features)} признаков из {len(feature_cols)}")
        logger.info(f"🏆 ТОП-10 лучших индикаторов:")
        for i, (feature, scores) in enumerate(sorted_features[:10]):
            logger.info(f"   {i+1:2d}. {feature:25s} - скор: {scores['combined_score']:.4f} "
                       f"(F:{scores['f_score']:.3f}, MI:{scores['mi_score']:.3f}, RF:{scores['rf_importance']:.3f})")
        
        logger.info(f"❌ Отфильтрованные слабые индикаторы ({len(feature_cols) - len(selected_features)}):")
        weak_features = [feature for feature, _ in sorted_features[top_k_features:]]
        for i, (feature, scores) in enumerate(sorted_features[top_k_features:top_k_features+5]):
            logger.info(f"   {feature:25s} - скор: {scores['combined_score']:.4f}")
        if len(weak_features) > 5:
            logger.info(f"   ... и ещё {len(weak_features) - 5} слабых признаков")
        
        # 7. СОХРАНЕНИЕ ИНФОРМАЦИИ О ВАЖНОСТИ ПРИЗНАКОВ
        self.feature_importance_analysis = {
            'selected_features': selected_features,
            'feature_scores': feature_scores,
            'selection_method': 'combined_f_mi_rf',
            'top_k': top_k_features,
            'total_features_before': len(feature_cols),
            'total_features_after': len(selected_features)
        }
        
        return X_balanced, selected_features
    
    def analyze_feature_quality(self, X: pd.DataFrame, y: np.ndarray, selected_features: List[str]) -> Dict:
        """
        📈 АНАЛИЗ КАЧЕСТВА ОТОБРАННЫХ ПРИЗНАКОВ
        Проверяет, насколько хорошо отобранные признаки разделяют классы
        """
        logger.info("🔬 Анализ качества отобранных признаков...")
        
        feature_cols = [col for col in selected_features if col not in ['index', 'current_price']]
        X_features = X[feature_cols].fillna(X[feature_cols].median())
        
        # Анализ разделимости классов
        class_separation = {}
        for feature in feature_cols:
            feature_values = X_features[feature].values
            class_means = {}
            class_stds = {}
            
            for class_label in np.unique(y):
                class_mask = (y == class_label)
                class_values = feature_values[class_mask]
                class_means[class_label] = np.mean(class_values)
                class_stds[class_label] = np.std(class_values)
            
            # Вычисляем коэффициент разделимости (отношение межклассового к внутриклассовому разбросу)
            between_class_var = np.var(list(class_means.values()))
            within_class_var = np.mean(list(class_stds.values())) ** 2
            separation_ratio = between_class_var / (within_class_var + 1e-8)
            
            class_separation[feature] = {
                'separation_ratio': separation_ratio,
                'class_means': class_means,
                'class_stds': class_stds
            }
        
        # Топ признаки по разделимости
        top_separating_features = sorted(class_separation.items(), 
                                       key=lambda x: x[1]['separation_ratio'], 
                                       reverse=True)[:10]
        
        logger.info("🎯 ТОП-10 признаков по разделимости классов:")
        for i, (feature, data) in enumerate(top_separating_features):
            logger.info(f"   {i+1:2d}. {feature:25s} - разделимость: {data['separation_ratio']:.4f}")
        
        quality_analysis = {
            'class_separation': class_separation,
            'top_separating_features': [f[0] for f in top_separating_features],
            'average_separation_ratio': np.mean([data['separation_ratio'] for data in class_separation.values()]),
            'feature_count': len(feature_cols)
        }
        
        return quality_analysis
    
    def create_labels(self, data: pd.DataFrame, features_df: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame]:
        """Создание меток для обучения с использованием улучшенной стратегии"""
        logger.info("Создание меток...")
        
        if self.config.get('use_improved_labeling', True):
            logger.info("Использование улучшенной стратегии разметки")
            labels, filtered_features_df = self.labeling_strategy.create_enhanced_labels(features_df)
            return labels, filtered_features_df
        else:
            # Оригинальная стратегия разметки (для сравнения)
            logger.info("Использование оригинальной стратегии разметки")
            labels = []
            returns = []
            
            for idx in features_df['index']:
                if idx + self.config['prediction_horizon'] >= len(data):
                    continue
                    
                current_price = data.iloc[idx]['close']
                future_price = data.iloc[idx + self.config['prediction_horizon']]['close']
                
                # Вычисляем будущий возврат
                future_return = (future_price - current_price) / current_price
                returns.append(future_return)
                
                # Создаем метки на основе стратегии
                if future_return > self.config['take_profit_pct']:
                    label = 1  # BUY
                elif future_return < -self.config['stop_loss_pct']:
                    label = 2  # SELL
                else:
                    label = 0  # HOLD
                
                labels.append(label)
            
            # Обрезаем features_df до размера labels
            filtered_features_df = features_df.iloc[:len(labels)].copy()
            
            # Детальная диагностика
            returns_array = np.array(returns)
            logger.info(f"Создано {len(labels)} меток")
            logger.info(f"Распределение: HOLD={labels.count(0)}, BUY={labels.count(1)}, SELL={labels.count(2)}")
            logger.info(f"Статистика доходности:")
            logger.info(f"  Средняя доходность: {returns_array.mean():.6f}")
            logger.info(f"  Стандартное отклонение: {returns_array.std():.6f}")
            logger.info(f"  Минимум: {returns_array.min():.6f}")
            logger.info(f"  Максимум: {returns_array.max():.6f}")
            logger.info(f"  Процентили: 5%={np.percentile(returns_array, 5):.6f}, 95%={np.percentile(returns_array, 95):.6f}")
            logger.info(f"Пороги: take_profit={self.config['take_profit_pct']:.6f}, stop_loss={self.config['stop_loss_pct']:.6f}")
            
            # Проверяем, сколько точек превышают пороги
            buy_candidates = np.sum(returns_array > self.config['take_profit_pct'])
            sell_candidates = np.sum(returns_array < -self.config['stop_loss_pct'])
            logger.info(f"Кандидаты: BUY={buy_candidates}, SELL={sell_candidates}")
            
            return np.array(labels), filtered_features_df
    
    def train_models(self, X: pd.DataFrame, y: np.ndarray) -> Dict:
        """Обучение различных моделей машинного обучения с балансировкой классов и индикаторов"""
        logger.info("Начало обучения моделей с балансировкой индикаторов...")
        
        # 🎯 НОВАЯ ФУНКЦИЯ: БАЛАНСИРОВКА ИНДИКАТОРОВ
        logger.info("=" * 60)
        logger.info("🔍 ЭТАП 1: БАЛАНСИРОВКА ИНДИКАТОРОВ")
        logger.info("=" * 60)
        
        # Применяем балансировку индикаторов для отбора лучших признаков
        X_balanced, selected_features = self.balance_indicators(X, y, top_k_features=50)
        
        # Анализируем качество отобранных признаков
        quality_analysis = self.analyze_feature_quality(X_balanced, y, selected_features)
        
        logger.info("=" * 60)
        logger.info("🤖 ЭТАП 2: ОБУЧЕНИЕ МОДЕЛЕЙ НА ОТОБРАННЫХ ПРИЗНАКАХ")
        logger.info("=" * 60)
        
        # Используем отобранные признаки для обучения
        feature_cols = [col for col in selected_features if col not in ['index', 'current_price']]
        X_features = X_balanced[feature_cols]
        self.feature_names = feature_cols
        
        logger.info(f"📊 Обучение на {len(feature_cols)} отобранных признаках (было {len(X.columns)-2})")
        logger.info(f"🎯 Средняя разделимость классов: {quality_analysis['average_separation_ratio']:.4f}")
        
        # Вычисляем веса классов для балансировки
        class_weights = None
        if self.config.get('use_class_weights', True):
            unique_classes = np.unique(y)
            class_weights_array = compute_class_weight('balanced', classes=unique_classes, y=y)
            class_weights = dict(zip(unique_classes, class_weights_array))
            logger.info(f"Веса классов для балансировки: {class_weights}")
        
        # Разделение на обучающую и тестовую выборки
        X_train, X_test, y_train, y_test = train_test_split(
            X_features, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Масштабирование признаков
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scalers['main'] = scaler
        
        # Модели для обучения с улучшенной балансировкой
        models_config = {
            'random_forest': {
                'model': RandomForestClassifier(
                    random_state=42, 
                    class_weight=class_weights if class_weights else 'balanced',
                    n_jobs=-1
                ),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            },
            'xgboost': {
                'model': xgb.XGBClassifier(
                    random_state=42, 
                    n_jobs=-1,
                    eval_metric='mlogloss'
                ),
                'params': {
                    'n_estimators': [100, 200],
                    'max_depth': [6, 10],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.8, 1.0],
                    'scale_pos_weight': [1, 2, 3] if len(np.unique(y)) == 2 else [1]
                }
            },
            'lightgbm': {
                'model': lgb.LGBMClassifier(
                    random_state=42, 
                    verbose=-1, 
                    class_weight=class_weights if class_weights else 'balanced',
                    n_jobs=-1
                ),
                'params': {
                    'n_estimators': [100, 200],
                    'max_depth': [6, 10],
                    'learning_rate': [0.01, 0.1],
                    'num_leaves': [31, 50]
                }
            },
            'gradient_boosting': {
                'model': GradientBoostingClassifier(random_state=42),
                'params': {
                    'n_estimators': [100, 200],
                    'max_depth': [6, 10],
                    'learning_rate': [0.01, 0.1]
                }
            }
        }
        
        results = {}
        
        for name, config in models_config.items():
            logger.info(f"Обучение модели: {name}")
            
            try:
                # Настройка весов для XGBoost (специальная обработка)
                if name == 'xgboost' and class_weights and len(np.unique(y)) > 2:
                    # Для многоклассовой классификации в XGBoost используем sample_weight
                    sample_weights = np.array([class_weights[label] for label in y_train])
                    
                    # Обучение без GridSearch для XGBoost с весами
                    model = config['model']
                    model.fit(X_train_scaled, y_train, sample_weight=sample_weights)
                    best_model = model
                    best_params = model.get_params()
                else:
                    # Обычное обучение с GridSearch
                    grid_search = GridSearchCV(
                        config['model'], 
                        config['params'], 
                        cv=3, 
                        scoring='f1_macro',
                        n_jobs=-1,
                        verbose=1
                    )
                    grid_search.fit(X_train_scaled, y_train)
                    best_model = grid_search.best_estimator_
                    best_params = grid_search.best_params_
                
                # Предсказания
                y_pred = best_model.predict(X_test_scaled)
                y_pred_proba = best_model.predict_proba(X_test_scaled)
                
                # Сохранение результатов
                results[name] = {
                    'model': best_model,
                    'params': best_params,
                    'predictions': y_pred,
                    'probabilities': y_pred_proba,
                    'y_test': y_test,
                    'accuracy': accuracy_score(y_test, y_pred),
                    'classification_report': classification_report(y_test, y_pred, output_dict=True),
                    'confusion_matrix': confusion_matrix(y_test, y_pred)
                }
                
                logger.info(f"Модель {name} обучена. Точность: {results[name]['accuracy']:.4f}")
                
            except Exception as e:
                logger.error(f"Ошибка при обучении модели {name}: {str(e)}")
                continue
        
        return results
    
    def evaluate_model(self, results: Dict) -> Dict:
        """Оценка качества модели"""
        logger.info("Оценка качества модели...")
        
        evaluation = {}
        
        for name, result in results.items():
            y_test = result['y_test']
            y_pred = result['predictions']
            
            # Основные метрики
            accuracy = accuracy_score(y_test, y_pred)
            
            # Детальный отчет
            report = classification_report(y_test, y_pred, output_dict=True)
            
            # Матрица ошибок
            cm = confusion_matrix(y_test, y_pred)
            
            evaluation[name] = {
                'accuracy': accuracy,
                'classification_report': report,
                'confusion_matrix': cm.tolist(),
                'meets_target': accuracy >= self.config['target_winrate']
            }
            
            logger.info(f"\n{name} - Детальная оценка:")
            logger.info(f"Accuracy: {accuracy:.4f}")
            logger.info(f"Достигает целевой точности ({self.config['target_winrate']}): {accuracy >= self.config['target_winrate']}")
        
        return evaluation
    
    def save_model(self):
        """Сохранение обученной модели"""
        logger.info("Сохранение модели...")
        
        model_path = self.model_dir / f"{self.symbol}_trading_model.joblib"
        scaler_path = self.model_dir / f"{self.symbol}_scaler.joblib"
        config_path = self.model_dir / f"{self.symbol}_config.json"
        features_path = self.model_dir / f"{self.symbol}_features.json"
        
        # Сохраняем модель
        joblib.dump(self.models['best'], model_path)
        
        # Сохраняем скейлер
        joblib.dump(self.scalers['main'], scaler_path)
        
        # Сохраняем конфигурацию
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        # Сохраняем список признаков
        with open(features_path, 'w') as f:
            json.dump(self.feature_names, f, indent=2)
        
        logger.info(f"Модель сохранена в {model_path}")
    
    def load_model(self):
        """Загрузка сохраненной модели"""
        logger.info("Загрузка модели...")
        
        model_path = self.model_dir / f"{self.symbol}_trading_model.joblib"
        scaler_path = self.model_dir / f"{self.symbol}_scaler.joblib"
        config_path = self.model_dir / f"{self.symbol}_config.json"
        features_path = self.model_dir / f"{self.symbol}_features.json"
        
        if not all(p.exists() for p in [model_path, scaler_path, config_path, features_path]):
            raise FileNotFoundError("Не найдены файлы модели")
        
        # Загружаем модель
        self.models['best'] = joblib.load(model_path)
        
        # Загружаем скейлер
        self.scalers['main'] = joblib.load(scaler_path)
        
        # Загружаем конфигурацию
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Загружаем список признаков
        with open(features_path, 'r') as f:
            self.feature_names = json.load(f)
        
        logger.info("Модель загружена успешно")
    
    async def backtest(self, data: pd.DataFrame) -> Dict:
        """Бэктестинг стратегии"""
        logger.info("Запуск бэктестинга...")
        
        if 'best' not in self.models:
            raise ValueError("Модель не обучена. Сначала запустите обучение.")
        
        # Подготавливаем данные для бэктестинга
        features_df = self.prepare_features(data)
        
        # Удаляем служебные колонки
        X = features_df[self.feature_names]
        
        # Масштабируем признаки
        X_scaled = self.scalers['main'].transform(X)
        
        # Получаем предсказания
        predictions = self.models['best'].predict(X_scaled)
        probabilities = self.models['best'].predict_proba(X_scaled)
        
        # Симуляция торговли
        initial_balance = 10000
        balance = initial_balance
        position = 0
        trades = []
        
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            if i + self.config['lookback_period'] >= len(data):
                break
                
            current_price = data.iloc[i + self.config['lookback_period']]['close']
            confidence = np.max(prob)
            
            # Торговая логика
            if pred == 1 and confidence > self.config['min_confidence'] and position <= 0:  # BUY
                if position < 0:  # Закрываем короткую позицию
                    balance += position * current_price
                    trades.append({
                        'type': 'close_short',
                        'price': current_price,
                        'balance': balance,
                        'confidence': confidence
                    })
                
                # Открываем длинную позицию
                position = balance / current_price
                balance = 0
                trades.append({
                    'type': 'buy',
                    'price': current_price,
                    'position': position,
                    'confidence': confidence
                })
                
            elif pred == 2 and confidence > self.config['min_confidence'] and position >= 0:  # SELL
                if position > 0:  # Закрываем длинную позицию
                    balance += position * current_price
                    trades.append({
                        'type': 'close_long',
                        'price': current_price,
                        'balance': balance,
                        'confidence': confidence
                    })
                
                # Открываем короткую позицию
                position = -balance / current_price
                balance = 0
                trades.append({
                    'type': 'sell',
                    'price': current_price,
                    'position': position,
                    'confidence': confidence
                })
        
        # Закрываем оставшуюся позицию
        final_price = data.iloc[-1]['close']
        if position != 0:
            balance += position * final_price
        
        # Расчет результатов
        total_return = (balance - initial_balance) / initial_balance * 100
        num_trades = len(trades)
        
        # Подсчет выигрышных сделок
        winning_trades = 0
        for i in range(1, len(trades)):
            if trades[i]['type'] in ['close_long', 'close_short']:
                if trades[i]['balance'] > trades[i-1].get('balance', initial_balance):
                    winning_trades += 1
        
        win_rate = winning_trades / (num_trades // 2) if num_trades > 0 else 0
        
        results = {
            'initial_balance': initial_balance,
            'final_balance': balance,
            'total_return_pct': total_return,
            'num_trades': num_trades,
            'win_rate': win_rate,
            'trades': trades,
            'meets_target_winrate': win_rate >= self.config['target_winrate']
        }
        
        logger.info(f"Результаты бэктестинга:")
        logger.info(f"Общая доходность: {total_return:.2f}%")
        logger.info(f"Количество сделок: {num_trades}")
        logger.info(f"Винрейт: {win_rate:.2f}")
        logger.info(f"Достигает целевой винрейт ({self.config['target_winrate']}): {win_rate >= self.config['target_winrate']}")
        
        return results

async def main():
    parser = argparse.ArgumentParser(description='Trading AI Trainer')
    parser.add_argument('--symbol', default='BTCUSDT', help='Торговая пара')
    parser.add_argument('--days', type=int, default=365, help='Количество дней для загрузки данных')
    parser.add_argument('--train', action='store_true', help='Запустить обучение')
    parser.add_argument('--backtest', action='store_true', help='Запустить бэктестинг')
    parser.add_argument('--evaluate', action='store_true', help='Оценить модель')
    
    args = parser.parse_args()
    
    trainer = TradingAITrainer(args.symbol)
    
    if args.train:
        logger.info("=== РЕЖИМ ОБУЧЕНИЯ ===")
        
        # Загружаем данные
        data = await trainer.load_market_data(args.days)
        
        # Подготавливаем признаки
        features_df = trainer.prepare_features(data)
        
        # Создаем метки
        labels, features_df = trainer.create_labels(data, features_df)
        
        # Обучаем модели
        results = trainer.train_models(features_df, labels)
        
        # Оцениваем модели
        evaluation = trainer.evaluate_model(results)
        
        # Сохраняем лучшую модель
        trainer.save_model()
        
        logger.info("Обучение завершено!")
    
    if args.backtest:
        logger.info("=== РЕЖИМ БЭКТЕСТИНГА ===")
        
        # Загружаем модель
        trainer.load_model()
        
        # Загружаем данные
        data = await trainer.load_market_data(args.days)
        
        # Запускаем бэктестинг
        backtest_results = await trainer.backtest(data)
        
        logger.info("Бэктестинг завершен!")
    
    if args.evaluate:
        logger.info("=== РЕЖИМ ОЦЕНКИ ===")
        
        # Загружаем модель
        trainer.load_model()
        
        # Загружаем данные
        data = await trainer.load_market_data(args.days)
        
        # Подготавливаем данные
        features_df = trainer.prepare_features(data)
        labels, features_df = trainer.create_labels(data, features_df)
        
        # Оцениваем на новых данных
        X = features_df[trainer.feature_names]
        X_scaled = trainer.scalers['main'].transform(X)
        predictions = trainer.models['best'].predict(X_scaled)
        
        from sklearn.metrics import accuracy_score
        accuracy = accuracy_score(labels, predictions)
        
        logger.info(f"Точность на новых данных: {accuracy:.4f}")

if __name__ == "__main__":
    asyncio.run(main())