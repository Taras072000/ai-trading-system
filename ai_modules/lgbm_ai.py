"""
LGBM AI модуль для Peper Binance v4
Легковесная реализация машинного обучения с использованием LightGBM
"""

import asyncio
import logging
import numpy as np
import pandas as pd
import lightgbm as lgb
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
import gc
import pickle
import os
from dataclasses import dataclass
import config
from utils.timezone_utils import get_utc_now
from config_params import CONFIG_PARAMS
from utils.indicators_cache import indicators_cache
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error

logger = logging.getLogger(__name__)

@dataclass
class LGBMPrediction:
    """Результат предсказания LGBM модели"""
    prediction: Union[float, List[float]]
    confidence: float
    feature_importance: Dict[str, float]
    model_type: str
    timestamp: datetime
    metadata: Dict[str, Any]

class LGBMModelManager:
    """Менеджер моделей LGBM с оптимизацией памяти"""
    
    def __init__(self, max_models: int = 3):
        self.max_models = max_models
        self.models = {}
        self.scalers = {}
        self.model_metadata = {}
        self.last_used = {}
    
    def add_model(self, name: str, model: lgb.LGBMModel, scaler: StandardScaler = None):
        """Добавление модели с контролем памяти"""
        if len(self.models) >= self.max_models:
            self._remove_oldest_model()
        
        self.models[name] = model
        self.scalers[name] = scaler
        self.last_used[name] = get_utc_now()
        self.model_metadata[name] = {
            'created': get_utc_now(),
            'size_mb': self._estimate_model_size(model)
        }
    
    def get_model(self, name: str) -> Tuple[Optional[lgb.LGBMModel], Optional[StandardScaler]]:
        """Получение модели"""
        if name in self.models:
            self.last_used[name] = get_utc_now()
            return self.models[name], self.scalers.get(name)
        return None, None
    
    def _remove_oldest_model(self):
        """Удаление самой старой модели"""
        if not self.last_used:
            return
        
        oldest_model = min(self.last_used.items(), key=lambda x: x[1])[0]
        self.remove_model(oldest_model)
    
    def remove_model(self, name: str):
        """Удаление модели"""
        self.models.pop(name, None)
        self.scalers.pop(name, None)
        self.last_used.pop(name, None)
        self.model_metadata.pop(name, None)
        gc.collect()
    
    def _estimate_model_size(self, model: lgb.LGBMModel) -> float:
        """Приблизительная оценка размера модели в МБ"""
        try:
            # Простая оценка на основе количества деревьев и листьев
            if hasattr(model, 'booster_'):
                return model.booster_.num_trees() * 0.1  # Примерно 0.1 МБ на дерево
            return 1.0  # Базовая оценка
        except:
            return 1.0

class LGBMAI:
    """
    LGBM AI модуль с оптимизацией ресурсов
    Поддерживает различные типы задач машинного обучения
    """
    
    def __init__(self):
        # Получаем конфигурацию LGBM AI из CONFIG_PARAMS
        ai_config = CONFIG_PARAMS.get('ai_modules', {})
        lgbm_config = ai_config.get('lgbm', {})
        
        self.config = lgbm_config
        self.is_initialized = False
        self.model_manager = LGBMModelManager()
        self.feature_cache = {}
        self.last_cleanup = get_utc_now()
        
        # Оптимизированные параметры LGBM для лучшего качества
        self.lgbm_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': lgbm_config.get('num_leaves', 15),  # Оптимизировано с 31 до 15
            'max_depth': lgbm_config.get('max_depth', 4),     # Оптимизировано с 6 до 4
            'learning_rate': lgbm_config.get('learning_rate', 0.05),  # Оптимизировано с 0.1 до 0.05
            'n_estimators': lgbm_config.get('n_estimators', 100),
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': 1,  # Ограничиваем количество потоков
            'verbose': -1,
            'force_row_wise': True  # Экономия памяти
        }
        
        logger.info("LGBM AI инициализирован с оптимизацией ресурсов")
    
    async def initialize(self):
        """Ленивая инициализация модуля"""
        if self.is_initialized:
            return True
        
        try:
            logger.info("Инициализация LGBM AI модуля...")
            
            # Создаем базовые модели
            await self._create_base_models()
            
            self.is_initialized = True
            logger.info("LGBM AI модуль успешно инициализирован")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка инициализации LGBM AI: {e}")
            return False
    
    async def _create_base_models(self):
        """Создание базовых моделей"""
        # Создаем простые синтетические данные для демонстрации
        np.random.seed(42)
        X_demo = np.random.randn(100, 5)
        y_demo = np.random.randn(100)
        
        # Модель для предсказания цены
        price_model = lgb.LGBMRegressor(**self.lgbm_params)
        price_model.fit(X_demo, y_demo)
        
        # Модель для классификации тренда
        trend_params = self.lgbm_params.copy()
        trend_params['objective'] = 'binary'
        trend_params['metric'] = 'binary_logloss'
        
        y_trend = (y_demo > 0).astype(int)
        trend_model = lgb.LGBMClassifier(**trend_params)
        trend_model.fit(X_demo, y_trend)
        
        # Добавляем модели в менеджер
        self.model_manager.add_model('price_prediction', price_model)
        self.model_manager.add_model('trend_classification', trend_model)
        
        logger.info("Базовые LGBM модели созданы")
    
    async def train_model(self, model_name: str, X: pd.DataFrame, y: pd.Series, 
                         model_type: str = 'regression') -> Dict[str, Any]:
        """
        Обучение модели с кросс-валидацией TimeSeriesSplit
        """
        if not self.is_initialized:
            await self.initialize()
        
        try:
            logger.info(f"Обучение модели {model_name} с кросс-валидацией...")
            
            # Подготовка данных
            X_processed, scaler = await self._prepare_features(X)
            
            # Настройка параметров в зависимости от типа задачи
            params = self.lgbm_params.copy()
            if model_type == 'classification':
                params['objective'] = 'binary'
                params['metric'] = 'binary_logloss'
                model = lgb.LGBMClassifier(**params)
                scoring = 'accuracy'
            else:
                model = lgb.LGBMRegressor(**params)
                scoring = 'neg_mean_squared_error'
            
            # Кросс-валидация с TimeSeriesSplit (5 фолдов)
            tscv = TimeSeriesSplit(n_splits=5)
            cv_scores = cross_val_score(model, X_processed, y, cv=tscv, scoring=scoring, n_jobs=1)
            
            # Финальное обучение на всех данных
            # Разделение данных для финального обучения
            X_train, X_test, y_train, y_test = train_test_split(
                X_processed, y, test_size=0.2, random_state=42, shuffle=False  # Без перемешивания для временных рядов
            )
            
            # Обучение с ранней остановкой
            model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)]
            )
            
            # Оценка качества на тестовых данных
            if model_type == 'classification':
                y_pred = model.predict(X_test)
                test_score = accuracy_score(y_test, y_pred)
                metric_name = 'accuracy'
                cv_mean = cv_scores.mean()
                cv_std = cv_scores.std()
            else:
                y_pred = model.predict(X_test)
                test_score = mean_squared_error(y_test, y_pred, squared=False)
                metric_name = 'rmse'
                # Для MSE нужно инвертировать отрицательные значения
                cv_mean = np.sqrt(-cv_scores.mean())
                cv_std = np.sqrt(cv_scores.std())
            
            # Сохраняем модель
            self.model_manager.add_model(model_name, model, scaler)
            
            # Периодическая очистка памяти
            await self._periodic_cleanup()
            
            logger.info(f"Модель {model_name} обучена. CV Score: {cv_mean:.4f} ± {cv_std:.4f}, Test Score: {test_score:.4f}")
            
            return {
                'model_name': model_name,
                'model_type': model_type,
                'test_score': test_score,
                'cv_score_mean': cv_mean,
                'cv_score_std': cv_std,
                'cv_scores': cv_scores.tolist(),
                'metric': metric_name,
                'feature_importance': dict(zip(X.columns, model.feature_importances_)),
                'training_samples': len(X_train),
                'test_samples': len(X_test),
                'cross_validation': {
                    'method': 'TimeSeriesSplit',
                    'n_splits': 5,
                    'scores': cv_scores.tolist()
                }
            }
            
        except Exception as e:
            logger.error(f"Ошибка обучения модели {model_name}: {e}")
            return {'error': str(e)}
    
    async def predict(self, model_name: str, X: pd.DataFrame) -> LGBMPrediction:
        """
        Предсказание с использованием обученной модели
        """
        if not self.is_initialized:
            await self.initialize()
        
        try:
            # Получаем модель
            model, scaler = self.model_manager.get_model(model_name)
            if model is None:
                raise ValueError(f"Модель {model_name} не найдена")
            
            # Подготавливаем данные
            X_processed = X.copy()
            
            # Проверяем количество признаков
            if hasattr(model, 'n_features_in_'):
                expected_features = model.n_features_in_
                if X_processed.shape[1] != expected_features:
                    logger.warning(f"Несоответствие признаков: ожидается {expected_features}, получено {X_processed.shape[1]}")
                    # Используем только первые n признаков или дополняем нулями
                    if X_processed.shape[1] > expected_features:
                        X_processed = X_processed.iloc[:, :expected_features]
                    else:
                        # Дополняем нулями до нужного количества
                        missing_cols = expected_features - X_processed.shape[1]
                        for i in range(missing_cols):
                            X_processed[f'feature_{X_processed.shape[1] + i}'] = 0.0
            
            if scaler is not None:
                X_processed = pd.DataFrame(
                    scaler.transform(X_processed),
                    columns=X_processed.columns,
                    index=X_processed.index
                )
            
            # Делаем предсказание
            prediction = model.predict(X_processed)
            
            # Вычисляем уверенность (для классификации)
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X_processed)
                confidence = np.max(proba, axis=1).mean()
            else:
                # Для регрессии используем обратную величину стандартного отклонения
                confidence = 1.0 / (1.0 + np.std(prediction))
            
            # Важность признаков
            feature_importance = {}
            if hasattr(model, 'feature_importances_'):
                feature_importance = dict(zip(X.columns, model.feature_importances_))
            
            return LGBMPrediction(
                prediction=prediction.tolist() if len(prediction) > 1 else float(prediction[0]),
                confidence=float(confidence),
                feature_importance=feature_importance,
                model_type=type(model).__name__,
                timestamp=get_utc_now(),
                metadata={
                    'model_name': model_name,
                    'samples_predicted': len(X),
                    'features_used': list(X.columns)
                }
            )
            
        except Exception as e:
            logger.error(f"Ошибка предсказания модели {model_name}: {e}")
            return LGBMPrediction(
                prediction=0.0,
                confidence=0.0,
                feature_importance={},
                model_type='error',
                timestamp=get_utc_now(),
                metadata={'error': str(e)}
            )
    
    async def _prepare_features(self, X: pd.DataFrame) -> Tuple[pd.DataFrame, StandardScaler]:
        """Подготовка признаков с кэшированием"""
        # Простая подготовка данных
        X_processed = X.copy()
        
        # Обработка пропущенных значений
        X_processed = X_processed.fillna(X_processed.mean())
        
        # Нормализация (опционально)
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(
            scaler.fit_transform(X_processed),
            columns=X_processed.columns,
            index=X_processed.index
        )
        
        return X_scaled, scaler
    
    async def create_trading_features(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """Создание торговых признаков из ценовых данных (ровно 5 признаков для совместимости)"""
        try:
            features = pd.DataFrame(index=price_data.index)
            
            # Создаем ровно 5 признаков для совместимости с обученной моделью
            # 1. Нормализованная цена (используем кэш для SMA)
            sma_20 = indicators_cache.get_sma(price_data['close'], 20)
            if sma_20 is not None:
                sma_20_std = price_data['close'].rolling(20).std()
                features['price_norm'] = (price_data['close'] - sma_20) / sma_20_std
            else:
                features['price_norm'] = 0.0
            
            # 2. RSI (используем кэш)
            rsi = indicators_cache.get_rsi(price_data['close'], 14)
            if rsi is not None:
                features['rsi'] = rsi / 100.0  # Нормализуем к [0,1]
            else:
                features['rsi'] = 0.5  # Нейтральное значение
            
            # 3. Отношение SMA (используем кэш)
            sma_5 = indicators_cache.get_sma(price_data['close'], 5)
            if sma_5 is not None and sma_20 is not None:
                features['sma_ratio'] = sma_5 / sma_20 - 1.0  # Центрируем вокруг 0
            else:
                features['sma_ratio'] = 0.0
            
            # 4. Волатильность (используем кэш)
            volatility = indicators_cache.get_volatility(price_data['close'], 10)
            if volatility is not None:
                features['volatility'] = volatility
            else:
                features['volatility'] = 0.0
            
            # 5. Ценовое изменение
            features['price_change'] = price_data['close'].pct_change()
            
            # Удаляем NaN и заполняем нулями
            features = features.fillna(0.0)
            
            # Убеждаемся, что у нас ровно 5 колонок
            if features.shape[1] != 5:
                logger.warning(f"Неожиданное количество признаков: {features.shape[1]}, ожидается 5")
                # Обрезаем до 5 или дополняем
                if features.shape[1] > 5:
                    features = features.iloc[:, :5]
                else:
                    for i in range(5 - features.shape[1]):
                        features[f'dummy_{i}'] = 0.0
            
            return features
            
        except Exception as e:
            logger.error(f"Ошибка создания торговых признаков: {e}")
            return pd.DataFrame()
    
    async def predict_price_movement(self, price_data: pd.DataFrame) -> LGBMPrediction:
        """Предсказание движения цены"""
        try:
            # Создаем признаки
            features = await self.create_trading_features(price_data)
            if features.empty:
                raise ValueError("Не удалось создать признаки")
            
            # Используем последние данные для предсказания
            latest_features = features.tail(1)
            
            # Предсказываем с помощью модели тренда
            prediction = await self.predict('trend_classification', latest_features)
            
            return prediction
            
        except Exception as e:
            logger.error(f"Ошибка предсказания движения цены: {e}")
            return LGBMPrediction(
                prediction=0.5,  # Нейтральное предсказание
                confidence=0.0,
                feature_importance={},
                model_type='error',
                timestamp=get_utc_now(),
                metadata={'error': str(e)}
            )
    
    async def predict_market_direction(self, symbol: str, price_data: pd.DataFrame) -> Dict[str, Any]:
        """Предсказание направления рынка для совместимости с тестами"""
        try:
            if not self.is_initialized:
                await self.initialize()
            
            # Создаем торговые признаки
            features = await self.create_trading_features(price_data)
            if features.empty:
                logger.warning(f"Не удалось создать признаки для {symbol}")
                return {
                    'direction': 0.0,
                    'confidence': 0.1,
                    'reasoning': 'Не удалось создать признаки',
                    'model_type': 'lgbm_ai'
                }
            
            # Используем последние данные для предсказания
            latest_features = features.tail(1)
            
            # Получаем предсказание от модели классификации тренда
            prediction = await self.predict('trend_classification', latest_features)
            
            # Преобразуем результат в формат, ожидаемый тестами
            if prediction.model_type == 'error':
                direction = 0.0
                confidence = 0.1
                reasoning = f"Ошибка модели: {prediction.metadata.get('error', 'Неизвестная ошибка')}"
            else:
                # Для классификации prediction.prediction - это вероятность класса 1 (рост)
                raw_prediction = prediction.prediction
                if isinstance(raw_prediction, list):
                    raw_prediction = raw_prediction[0] if raw_prediction else 0.5
                
                # Преобразуем в направление: >0.5 = рост, <0.5 = падение
                direction = 1.0 if raw_prediction > 0.5 else -1.0
                confidence = prediction.confidence
                
                # Создаем объяснение
                trend_word = "рост" if direction > 0 else "падение"
                reasoning = f"LGBM AI: предсказание {trend_word} с вероятностью {raw_prediction:.3f}"
                
                # Добавляем информацию о важности признаков
                if prediction.feature_importance:
                    top_features = sorted(prediction.feature_importance.items(), 
                                        key=lambda x: x[1], reverse=True)[:2]
                    feature_info = ", ".join([f"{feat}: {imp:.3f}" for feat, imp in top_features])
                    reasoning += f" (ключевые признаки: {feature_info})"
            
            logger.info(f"🤖 LGBM AI сигнал для {symbol}: направление={direction}, уверенность={confidence*100:.1f}%")
            
            return {
                'direction': direction,
                'confidence': confidence,
                'reasoning': reasoning,
                'model_type': 'lgbm_ai',
                'symbol': symbol,
                'prediction_raw': prediction.prediction,
                'feature_importance': prediction.feature_importance,
                'timestamp': get_utc_now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Ошибка предсказания направления рынка для {symbol}: {e}")
            return {
                'direction': 0.0,
                'confidence': 0.1,
                'reasoning': f'Ошибка LGBM AI: {str(e)}',
                'model_type': 'lgbm_ai',
                'symbol': symbol,
                'error': str(e)
            }
    
    async def _periodic_cleanup(self):
        """Периодическая очистка памяти"""
        now = get_utc_now()
        if (now - self.last_cleanup).seconds > 300:  # Каждые 5 минут
            # Очищаем кэш признаков
            if len(self.feature_cache) > 10:
                # Удаляем половину самых старых записей
                sorted_items = sorted(self.feature_cache.items())
                for key in sorted_items[:len(sorted_items)//2]:
                    del self.feature_cache[key[0]]
            
            gc.collect()
            self.last_cleanup = now
            logger.debug("Выполнена очистка памяти LGBM AI")
    
    async def save_model(self, model_name: str, filepath: str) -> bool:
        """Сохранение модели на диск"""
        try:
            model, scaler = self.model_manager.get_model(model_name)
            if model is None:
                return False
            
            # Сохраняем модель и скейлер
            model_data = {
                'model': model,
                'scaler': scaler,
                'metadata': self.model_manager.model_metadata.get(model_name, {}),
                'timestamp': get_utc_now()
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info(f"Модель {model_name} сохранена в {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка сохранения модели {model_name}: {e}")
            return False
    
    async def load_model(self, model_name: str, filepath: str) -> bool:
        """Загрузка модели с диска"""
        try:
            if not os.path.exists(filepath):
                return False
            
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model_manager.add_model(
                model_name,
                model_data['model'],
                model_data.get('scaler')
            )
            
            logger.info(f"Модель {model_name} загружена из {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка загрузки модели {model_name}: {e}")
            return False
    
    async def cleanup(self):
        """Очистка ресурсов модуля"""
        logger.info("Очистка ресурсов LGBM AI...")
        
        # Очищаем все модели
        for model_name in list(self.model_manager.models.keys()):
            self.model_manager.remove_model(model_name)
        
        self.feature_cache.clear()
        
        # Принудительная сборка мусора
        gc.collect()
        
        self.is_initialized = False
        logger.info("LGBM AI ресурсы очищены")
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Получение информации об использовании памяти"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        total_model_size = sum(
            metadata.get('size_mb', 0) 
            for metadata in self.model_manager.model_metadata.values()
        )
        
        return {
            'rss_mb': memory_info.rss / 1024 / 1024,
            'models_loaded': len(self.model_manager.models),
            'total_model_size_mb': total_model_size,
            'feature_cache_size': len(self.feature_cache),
            'model_names': list(self.model_manager.models.keys())
        }
    
    async def generate_trading_signals(self, symbol: str, data: pd.DataFrame) -> Dict[str, Any]:
        """Генерация торговых сигналов для совместимости с системой тестирования"""
        try:
            if not self.is_initialized:
                await self.initialize()
            
            # Используем существующий метод predict_market_direction
            market_prediction = await self.predict_market_direction(symbol, data)
            
            # Преобразуем в формат торгового сигнала
            direction = market_prediction.get('direction', 0.0)
            confidence = market_prediction.get('confidence', 0.1)
            
            # Определяем тип сигнала на основе направления и уверенности
            if abs(direction) < 0.1 or confidence < 0.3:
                signal_type = 'no_signal'
                action = 'HOLD'
            elif direction > 0:
                signal_type = 'buy_signal'
                action = 'BUY'
            else:
                signal_type = 'sell_signal'
                action = 'SELL'
            
            # Рассчитываем take profit и stop loss на основе волатильности
            if len(data) >= 20:
                volatility = data['close'].pct_change().rolling(20).std().iloc[-1]
                if pd.isna(volatility):
                    volatility = 0.02  # Дефолтная волатильность 2%
            else:
                volatility = 0.02
            
            take_profit = 2.0 * volatility * 100  # 2x волатильность в процентах
            stop_loss = 1.5 * volatility * 100    # 1.5x волатильность в процентах
            
            current_price = data['close'].iloc[-1] if len(data) > 0 else 0
            
            signal = {
                'signal_type': signal_type,
                'action': action,
                'confidence': confidence,
                'symbol': symbol,
                'price': current_price,
                'take_profit_pct': take_profit,
                'stop_loss_pct': stop_loss,
                'reasoning': market_prediction.get('reasoning', 'LGBM AI анализ'),
                'model_name': 'lgbm_ai',
                'timestamp': get_utc_now().isoformat(),
                'direction': direction,
                'feature_importance': market_prediction.get('feature_importance', {}),
                'metadata': {
                    'volatility': volatility,
                    'data_points': len(data),
                    'prediction_raw': market_prediction.get('prediction_raw', 0.5)
                }
            }
            
            logger.info(f"🤖 LGBM AI сигнал для {symbol}: {action} (уверенность: {confidence*100:.1f}%)")
            
            return signal
            
        except Exception as e:
            logger.error(f"Ошибка генерации торгового сигнала LGBM AI для {symbol}: {e}")
            return {
                'signal_type': 'no_signal',
                'action': 'HOLD',
                'confidence': 0.0,
                'symbol': symbol,
                'price': 0,
                'reasoning': f'Ошибка LGBM AI: {str(e)}',
                'model_name': 'lgbm_ai',
                'error': str(e),
                'timestamp': get_utc_now().isoformat()
            }
    
    async def analyze_market_data(self, symbol: str, data: pd.DataFrame) -> Dict[str, Any]:
        """Анализ рыночных данных для совместимости с системой тестирования"""
        try:
            if not self.is_initialized:
                await self.initialize()
            
            if len(data) < 20:
                return {
                    'analysis_type': 'insufficient_data',
                    'symbol': symbol,
                    'confidence': 0.0,
                    'reasoning': 'Недостаточно данных для анализа (минимум 20 свечей)',
                    'model_name': 'lgbm_ai',
                    'timestamp': get_utc_now().isoformat()
                }
            
            # Создаем торговые признаки для анализа
            features = await self.create_trading_features(data)
            if features.empty:
                return {
                    'analysis_type': 'feature_error',
                    'symbol': symbol,
                    'confidence': 0.0,
                    'reasoning': 'Не удалось создать торговые признаки',
                    'model_name': 'lgbm_ai',
                    'timestamp': get_utc_now().isoformat()
                }
            
            # Получаем предсказание движения цены
            prediction = await self.predict_market_direction(symbol, data)
            
            # Анализируем рыночные условия
            current_price = data['close'].iloc[-1]
            price_change_24h = ((current_price - data['close'].iloc[-24]) / data['close'].iloc[-24] * 100) if len(data) >= 24 else 0
            
            # Рассчитываем волатильность
            volatility = data['close'].pct_change().rolling(20).std().iloc[-1] * 100
            if pd.isna(volatility):
                volatility = 2.0
            
            # Определяем рыночные условия
            if volatility > 5.0:
                market_condition = 'high_volatility'
            elif volatility < 1.0:
                market_condition = 'low_volatility'
            else:
                market_condition = 'normal_volatility'
            
            # Анализ тренда
            sma_20 = data['close'].rolling(20).mean().iloc[-1]
            if current_price > sma_20 * 1.02:
                trend = 'uptrend'
            elif current_price < sma_20 * 0.98:
                trend = 'downtrend'
            else:
                trend = 'sideways'
            
            analysis = {
                'analysis_type': 'market_analysis',
                'symbol': symbol,
                'confidence': prediction.get('confidence', 0.5),
                'reasoning': f"LGBM AI анализ: {trend}, {market_condition}",
                'model_name': 'lgbm_ai',
                'timestamp': get_utc_now().isoformat(),
                'market_data': {
                    'current_price': current_price,
                    'price_change_24h': price_change_24h,
                    'volatility': volatility,
                    'trend': trend,
                    'market_condition': market_condition,
                    'sma_20': sma_20
                },
                'prediction': {
                    'direction': prediction.get('direction', 0.0),
                    'confidence': prediction.get('confidence', 0.5),
                    'feature_importance': prediction.get('feature_importance', {})
                },
                'technical_indicators': {
                    'rsi': features['rsi'].iloc[-1] * 100 if not features.empty else 50,
                    'sma_ratio': features['sma_ratio'].iloc[-1] if not features.empty else 0,
                    'volatility_norm': features['volatility'].iloc[-1] if not features.empty else 0
                }
            }
            
            logger.info(f"📊 LGBM AI анализ для {symbol}: {trend}, волатильность {volatility:.2f}%")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Ошибка анализа рыночных данных LGBM AI для {symbol}: {e}")
            return {
                'analysis_type': 'error',
                'symbol': symbol,
                'confidence': 0.0,
                'reasoning': f'Ошибка LGBM AI: {str(e)}',
                'model_name': 'lgbm_ai',
                'error': str(e),
                'timestamp': get_utc_now().isoformat()
            }