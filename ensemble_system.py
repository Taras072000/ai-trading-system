"""
Ensemble System для Peper Binance v4
Система ансамбля для объединения всех 4 AI моделей
Цель: достижение общего винрейта 75%+ через умное комбинирование предсказаний
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import asyncio
import json
from pathlib import Path
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
import joblib
import warnings
warnings.filterwarnings('ignore')

# Импорты AI моделей
from lgbm_ai import LGBMAI
from lava_ai import LavaAI
from mistral_ai import MistralAI
from trading_ai import TradingAI
from multi_ai_trainer import MultiAITrainer
from historical_data_manager import HistoricalDataManager
from indicators_optimizer import IndicatorsOptimizer

logger = logging.getLogger(__name__)

@dataclass
class ModelPrediction:
    """Предсказание от одной модели"""
    model_name: str
    signal: int  # 0 = продажа, 1 = покупка
    confidence: float  # 0.0 - 1.0
    probability: float  # Вероятность класса
    features_used: List[str]
    timestamp: datetime
    timeframe: str

@dataclass
class EnsemblePrediction:
    """Финальное предсказание ансамбля"""
    final_signal: int
    confidence: float
    individual_predictions: List[ModelPrediction]
    voting_method: str
    weights: Dict[str, float]
    timestamp: datetime
    
@dataclass
class EnsembleMetrics:
    """Метрики ансамбля"""
    winrate: float
    precision: float
    recall: float
    f1_score: float
    sharpe_ratio: float
    max_drawdown: float
    total_trades: int
    profitable_trades: int
    avg_return_per_trade: float
    model_contributions: Dict[str, float]

class EnsembleSystem:
    """Система ансамбля AI моделей"""
    
    def __init__(self, config_path: str = None):
        # Инициализация AI моделей
        self.models = {
            'lgbm': LGBMAI(),
            'lava': LavaAI(), 
            'mistral': MistralAI(),
            'trading_ai': TradingAI()
        }
        
        # Менеджеры
        self.data_manager = HistoricalDataManager()
        self.trainer = MultiAITrainer()
        self.optimizer = IndicatorsOptimizer()
        
        # Конфигурация ансамбля
        self.config = self._load_config(config_path) if config_path else self._default_config()
        
        # Веса моделей (обучаются автоматически)
        self.model_weights = {
            'lgbm': 0.25,
            'lava': 0.25,
            'mistral': 0.25,
            'trading_ai': 0.25
        }
        
        # Мета-модель для стекинга
        self.meta_model = LogisticRegression(random_state=42)
        self.is_meta_trained = False
        
        # Статистика
        self.prediction_history = []
        self.performance_metrics = {}
        
        # Настройки
        self.min_confidence_threshold = 0.6
        self.consensus_threshold = 0.7  # Минимальный консенсус для сигнала
        self.max_models_per_prediction = 4
        
        # Таймфреймы для каждой модели
        self.model_timeframes = {
            'lgbm': ['1m', '5m'],           # Скальпинг
            'lava': ['15m', '1h'],          # Свинг
            'mistral': ['4h', '1d'],        # Позиционная
            'trading_ai': ['5m', '15m', '1h'] # Универсальная
        }
    
    def _default_config(self) -> Dict[str, Any]:
        """Конфигурация по умолчанию"""
        return {
            'ensemble_methods': ['voting', 'stacking', 'weighted_average'],
            'voting_strategy': 'soft',  # 'hard' или 'soft'
            'confidence_weighting': True,
            'dynamic_weights': True,
            'performance_window': 100,  # Окно для расчета производительности
            'rebalance_frequency': 50,  # Частота пересчета весов
            'min_models_agreement': 2,  # Минимум моделей для согласия
            'target_winrate': 0.75,
            'risk_management': {
                'max_position_size': 0.1,
                'stop_loss': 0.02,
                'take_profit': 0.04,
                'max_daily_trades': 20
            }
        }
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Загрузка конфигурации из файла"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Ошибка загрузки конфигурации: {e}, используем настройки по умолчанию")
            return self._default_config()
    
    async def train_ensemble(self, symbols: List[str], lookback_days: int = 365) -> Dict[str, Any]:
        """Обучение всего ансамбля"""
        
        logger.info(f"🚀 Начинаем обучение ансамбля для {len(symbols)} символов")
        
        training_results = {}
        
        # 1. Загрузка исторических данных
        logger.info("📊 Загрузка исторических данных...")
        all_data = await self.data_manager.download_all_data(symbols, lookback_days)
        
        # 2. Обучение индивидуальных моделей
        logger.info("🤖 Обучение индивидуальных AI моделей...")
        
        for model_name, model in self.models.items():
            try:
                logger.info(f"   Обучение {model_name}...")
                
                # Получение данных для модели
                model_data = self._prepare_model_data(all_data, model_name)
                
                if model_data is None or len(model_data) < 1000:
                    logger.warning(f"   Недостаточно данных для {model_name}")
                    continue
                
                # Оптимизация индикаторов
                optimization_result = self.optimizer.optimize_indicators(model_name, model_data)
                
                # Обучение модели
                training_result = await self.trainer.train_individual_model(
                    model_name, model_data, optimization_result.best_indicators
                )
                
                training_results[model_name] = {
                    'optimization': optimization_result,
                    'training': training_result
                }
                
                logger.info(f"   ✅ {model_name} обучена, винрейт: {optimization_result.winrate:.1%}")
                
            except Exception as e:
                logger.error(f"   ❌ Ошибка обучения {model_name}: {e}")
                continue
        
        # 3. Обучение мета-модели для стекинга
        logger.info("🧠 Обучение мета-модели...")
        await self._train_meta_model(all_data, training_results)
        
        # 4. Оптимизация весов ансамбля
        logger.info("⚖️ Оптимизация весов ансамбля...")
        await self._optimize_ensemble_weights(all_data, training_results)
        
        # 5. Валидация ансамбля
        logger.info("✅ Валидация ансамбля...")
        validation_metrics = await self._validate_ensemble(all_data)
        
        # Сохранение результатов
        ensemble_results = {
            'training_date': datetime.now().isoformat(),
            'symbols': symbols,
            'lookback_days': lookback_days,
            'individual_models': training_results,
            'ensemble_weights': self.model_weights,
            'validation_metrics': validation_metrics,
            'config': self.config
        }
        
        await self._save_ensemble_model(ensemble_results)
        
        logger.info("🎉 Обучение ансамбля завершено!")
        logger.info(f"   Общий винрейт: {validation_metrics.winrate:.1%}")
        logger.info(f"   Sharpe ratio: {validation_metrics.sharpe_ratio:.3f}")
        
        return ensemble_results
    
    def _prepare_model_data(self, all_data: Dict[str, Dict[str, Any]], model_name: str) -> Optional[pd.DataFrame]:
        """Подготовка данных для конкретной модели"""
        
        # Выбор подходящих таймфреймов для модели
        suitable_timeframes = self.model_timeframes.get(model_name, ['1h'])
        
        # Поиск доступных данных
        for symbol_data in all_data.values():
            for timeframe in suitable_timeframes:
                if timeframe in symbol_data:
                    stats = symbol_data[timeframe]
                    if stats.total_candles > 1000:
                        # Загрузка данных
                        return asyncio.run(self.data_manager.load_data(
                            stats.symbol, timeframe
                        ))
        
        return None
    
    async def _train_meta_model(self, all_data: Dict[str, Dict[str, Any]], training_results: Dict[str, Any]):
        """Обучение мета-модели для стекинга"""
        
        # Сбор предсказаний от всех моделей
        meta_features = []
        meta_targets = []
        
        for symbol_data in all_data.values():
            for timeframe_data in symbol_data.values():
                try:
                    # Загрузка данных
                    data = await self.data_manager.load_data(
                        timeframe_data.symbol, timeframe_data.interval
                    )
                    
                    if data is None or len(data) < 100:
                        continue
                    
                    # Создание целевой переменной
                    target = self._create_target_variable(data)
                    
                    # Получение предсказаний от каждой модели
                    model_predictions = []
                    
                    for model_name, model in self.models.items():
                        if model_name in training_results:
                            try:
                                # Простое предсказание (заглушка)
                                pred = np.random.random(len(data))  # TODO: реальные предсказания
                                model_predictions.append(pred)
                            except:
                                model_predictions.append(np.zeros(len(data)))
                    
                    if len(model_predictions) >= 2:
                        # Объединение предсказаний
                        combined_predictions = np.column_stack(model_predictions)
                        
                        # Удаление NaN
                        valid_mask = ~(np.isnan(combined_predictions).any(axis=1) | np.isnan(target))
                        
                        if valid_mask.sum() > 50:
                            meta_features.append(combined_predictions[valid_mask])
                            meta_targets.append(target[valid_mask])
                
                except Exception as e:
                    logger.error(f"Ошибка подготовки мета-данных: {e}")
                    continue
        
        if meta_features:
            # Объединение всех данных
            X_meta = np.vstack(meta_features)
            y_meta = np.hstack(meta_targets)
            
            # Обучение мета-модели
            self.meta_model.fit(X_meta, y_meta)
            self.is_meta_trained = True
            
            # Оценка качества
            cv_scores = cross_val_score(self.meta_model, X_meta, y_meta, cv=5)
            logger.info(f"   Мета-модель CV score: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    
    async def _optimize_ensemble_weights(self, all_data: Dict[str, Dict[str, Any]], training_results: Dict[str, Any]):
        """Оптимизация весов ансамбля"""
        
        import optuna
        
        def objective(trial):
            """Целевая функция для оптимизации весов"""
            
            # Предлагаемые веса
            weights = {}
            total_weight = 0
            
            for model_name in self.models.keys():
                if model_name in training_results:
                    weight = trial.suggest_float(f"weight_{model_name}", 0.1, 1.0)
                    weights[model_name] = weight
                    total_weight += weight
            
            # Нормализация весов
            for model_name in weights:
                weights[model_name] /= total_weight
            
            # Симуляция торговли с этими весами
            try:
                total_accuracy = 0
                total_samples = 0
                
                for symbol_data in list(all_data.values())[:3]:  # Ограничиваем для скорости
                    for timeframe_data in list(symbol_data.values())[:2]:
                        
                        # Простая симуляция
                        accuracy = sum(weights.values()) * 0.7  # Базовая точность
                        
                        # Бонус за сбалансированность весов
                        weight_variance = np.var(list(weights.values()))
                        balance_bonus = max(0, 0.1 - weight_variance)
                        
                        total_accuracy += accuracy + balance_bonus
                        total_samples += 1
                
                return total_accuracy / total_samples if total_samples > 0 else 0.0
                
            except Exception as e:
                return 0.0
        
        # Оптимизация
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=50, show_progress_bar=False)
        
        # Применение лучших весов
        best_params = study.best_params
        total_weight = sum(best_params.values())
        
        for model_name in self.models.keys():
            if f"weight_{model_name}" in best_params:
                self.model_weights[model_name] = best_params[f"weight_{model_name}"] / total_weight
            else:
                self.model_weights[model_name] = 0.0
        
        logger.info(f"   Оптимизированные веса: {self.model_weights}")
    
    async def predict(self, symbol: str, timeframe: str = '1h', method: str = 'weighted_voting') -> EnsemblePrediction:
        """Получение предсказания от ансамбля"""
        
        # Загрузка последних данных
        data = await self.data_manager.load_data(symbol, timeframe)
        
        if data is None or len(data) < 100:
            raise ValueError(f"Недостаточно данных для {symbol} {timeframe}")
        
        # Получение предсказаний от каждой модели
        individual_predictions = []
        
        for model_name, model in self.models.items():
            try:
                # Проверка совместимости таймфрейма с моделью
                if timeframe not in self.model_timeframes.get(model_name, [timeframe]):
                    continue
                
                # Получение предсказания (заглушка)
                signal = np.random.choice([0, 1])  # TODO: реальное предсказание
                confidence = np.random.uniform(0.5, 1.0)
                probability = np.random.uniform(0.4, 0.9)
                
                prediction = ModelPrediction(
                    model_name=model_name,
                    signal=signal,
                    confidence=confidence,
                    probability=probability,
                    features_used=[],  # TODO: реальные фичи
                    timestamp=datetime.now(),
                    timeframe=timeframe
                )
                
                individual_predictions.append(prediction)
                
            except Exception as e:
                logger.error(f"Ошибка предсказания {model_name}: {e}")
                continue
        
        if not individual_predictions:
            raise ValueError("Не удалось получить предсказания ни от одной модели")
        
        # Комбинирование предсказаний
        if method == 'weighted_voting':
            final_prediction = self._weighted_voting(individual_predictions)
        elif method == 'stacking':
            final_prediction = self._stacking_prediction(individual_predictions)
        elif method == 'consensus':
            final_prediction = self._consensus_prediction(individual_predictions)
        else:
            final_prediction = self._simple_voting(individual_predictions)
        
        # Сохранение в историю
        self.prediction_history.append(final_prediction)
        
        return final_prediction
    
    def _weighted_voting(self, predictions: List[ModelPrediction]) -> EnsemblePrediction:
        """Взвешенное голосование"""
        
        total_weight = 0
        weighted_signal = 0
        weighted_confidence = 0
        
        active_weights = {}
        
        for pred in predictions:
            weight = self.model_weights.get(pred.model_name, 0.25)
            confidence_weight = weight * pred.confidence
            
            weighted_signal += pred.signal * confidence_weight
            weighted_confidence += pred.confidence * weight
            total_weight += weight  # Исправлено: используем базовый вес, а не confidence_weight
            
            active_weights[pred.model_name] = weight
        
        if total_weight == 0:
            final_signal = 0
            final_confidence = 0.5
        else:
            # Для сигнала используем confidence-взвешенное голосование
            final_signal = 1 if weighted_signal / sum(pred.confidence * self.model_weights.get(pred.model_name, 0.25) for pred in predictions) > 0.5 else 0
            # Для уверенности используем простое взвешенное среднее
            final_confidence = weighted_confidence / total_weight
        
        return EnsemblePrediction(
            final_signal=final_signal,
            confidence=final_confidence,
            individual_predictions=predictions,
            voting_method='weighted_voting',
            weights=active_weights,
            timestamp=datetime.now()
        )
    
    def _stacking_prediction(self, predictions: List[ModelPrediction]) -> EnsemblePrediction:
        """Предсказание через стекинг"""
        
        if not self.is_meta_trained:
            # Fallback к взвешенному голосованию
            return self._weighted_voting(predictions)
        
        # Подготовка входных данных для мета-модели
        meta_input = np.array([[pred.probability for pred in predictions]])
        
        try:
            # Предсказание мета-модели
            final_probability = self.meta_model.predict_proba(meta_input)[0][1]
            final_signal = 1 if final_probability > 0.5 else 0
            final_confidence = abs(final_probability - 0.5) * 2  # Нормализация к [0, 1]
            
        except Exception as e:
            logger.error(f"Ошибка стекинга: {e}")
            return self._weighted_voting(predictions)
        
        return EnsemblePrediction(
            final_signal=final_signal,
            confidence=final_confidence,
            individual_predictions=predictions,
            voting_method='stacking',
            weights={pred.model_name: 1/len(predictions) for pred in predictions},
            timestamp=datetime.now()
        )
    
    def _consensus_prediction(self, predictions: List[ModelPrediction]) -> EnsemblePrediction:
        """Консенсусное предсказание"""
        
        # Подсчет голосов
        buy_votes = sum(1 for pred in predictions if pred.signal == 1)
        total_votes = len(predictions)
        
        # Средняя уверенность
        avg_confidence = np.mean([pred.confidence for pred in predictions])
        
        # Консенсус
        consensus_ratio = buy_votes / total_votes
        
        if consensus_ratio >= self.consensus_threshold:
            final_signal = 1
            final_confidence = avg_confidence * consensus_ratio
        elif consensus_ratio <= (1 - self.consensus_threshold):
            final_signal = 0
            final_confidence = avg_confidence * (1 - consensus_ratio)
        else:
            # Нет консенсуса - нейтральная позиция
            final_signal = 0
            final_confidence = 0.5
        
        return EnsemblePrediction(
            final_signal=final_signal,
            confidence=final_confidence,
            individual_predictions=predictions,
            voting_method='consensus',
            weights={pred.model_name: 1/len(predictions) for pred in predictions},
            timestamp=datetime.now()
        )
    
    def _simple_voting(self, predictions: List[ModelPrediction]) -> EnsemblePrediction:
        """Простое голосование"""
        
        # Подсчет голосов
        votes = [pred.signal for pred in predictions]
        final_signal = 1 if sum(votes) > len(votes) / 2 else 0
        
        # Средняя уверенность
        final_confidence = np.mean([pred.confidence for pred in predictions])
        
        return EnsemblePrediction(
            final_signal=final_signal,
            confidence=final_confidence,
            individual_predictions=predictions,
            voting_method='simple_voting',
            weights={pred.model_name: 1/len(predictions) for pred in predictions},
            timestamp=datetime.now()
        )
    
    async def _validate_ensemble(self, all_data: Dict[str, Dict[str, Any]]) -> EnsembleMetrics:
        """Валидация ансамбля"""
        
        all_predictions = []
        all_targets = []
        all_returns = []
        
        # Симуляция торговли на исторических данных
        for symbol_data in list(all_data.values())[:3]:  # Ограничиваем для скорости
            for timeframe_data in list(symbol_data.values())[:2]:
                
                try:
                    # Загрузка данных
                    data = await self.data_manager.load_data(
                        timeframe_data.symbol, timeframe_data.interval
                    )
                    
                    if data is None or len(data) < 200:
                        continue
                    
                    # Создание целевой переменной
                    target = self._create_target_variable(data)
                    
                    # Симуляция предсказаний
                    for i in range(100, len(data) - 10):  # Скользящее окно
                        try:
                            # Получение предсказания (заглушка)
                            prediction = np.random.choice([0, 1])
                            
                            all_predictions.append(prediction)
                            all_targets.append(target[i])
                            
                            # Расчет доходности
                            if prediction == 1:  # Покупка
                                future_return = (data.iloc[i+5]['close'] - data.iloc[i]['close']) / data.iloc[i]['close']
                                all_returns.append(future_return)
                            else:
                                all_returns.append(0)
                        
                        except Exception as e:
                            continue
                
                except Exception as e:
                    logger.error(f"Ошибка валидации: {e}")
                    continue
        
        if not all_predictions:
            # Возвращаем базовые метрики
            return EnsembleMetrics(
                winrate=0.5,
                precision=0.5,
                recall=0.5,
                f1_score=0.5,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                total_trades=0,
                profitable_trades=0,
                avg_return_per_trade=0.0,
                model_contributions=self.model_weights
            )
        
        # Расчет метрик
        winrate = accuracy_score(all_targets, all_predictions)
        precision = precision_score(all_targets, all_predictions, zero_division=0)
        recall = recall_score(all_targets, all_predictions, zero_division=0)
        f1 = f1_score(all_targets, all_predictions, zero_division=0)
        
        # Финансовые метрики
        returns_series = pd.Series(all_returns)
        sharpe_ratio = self._calculate_sharpe_ratio(returns_series)
        max_drawdown = self._calculate_max_drawdown(returns_series)
        
        profitable_trades = sum(1 for r in all_returns if r > 0)
        avg_return = np.mean(all_returns) if all_returns else 0
        
        return EnsembleMetrics(
            winrate=winrate,
            precision=precision,
            recall=recall,
            f1_score=f1,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            total_trades=len(all_predictions),
            profitable_trades=profitable_trades,
            avg_return_per_trade=avg_return,
            model_contributions=self.model_weights
        )
    
    def _create_target_variable(self, data: pd.DataFrame, lookforward: int = 5) -> np.ndarray:
        """Создание целевой переменной"""
        
        future_returns = data['close'].pct_change(lookforward).shift(-lookforward)
        target = (future_returns > 0.005).astype(int)  # 0.5% прибыль
        
        return target.fillna(0).values
    
    def _calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Расчет коэффициента Шарпа"""
        
        if len(returns) == 0 or returns.std() == 0:
            return 0.0
        
        excess_returns = returns.mean() * 252 - risk_free_rate
        volatility = returns.std() * np.sqrt(252)
        
        return excess_returns / volatility if volatility > 0 else 0.0
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Расчет максимальной просадки"""
        
        if len(returns) == 0:
            return 0.0
        
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        return abs(drawdown.min()) if len(drawdown) > 0 else 0.0
    
    async def _save_ensemble_model(self, results: Dict[str, Any]):
        """Сохранение обученного ансамбля"""
        
        model_dir = Path("ensemble_models")
        model_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Сохранение конфигурации и результатов
        config_file = model_dir / f"ensemble_config_{timestamp}.json"
        with open(config_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Сохранение мета-модели
        if self.is_meta_trained:
            meta_model_file = model_dir / f"meta_model_{timestamp}.joblib"
            joblib.dump(self.meta_model, meta_model_file)
        
        # Сохранение весов
        weights_file = model_dir / f"ensemble_weights_{timestamp}.json"
        with open(weights_file, 'w') as f:
            json.dump(self.model_weights, f, indent=2)
        
        logger.info(f"💾 Ансамбль сохранен в {model_dir}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Получение сводки производительности"""
        
        if not self.prediction_history:
            return {"message": "Нет истории предсказаний"}
        
        recent_predictions = self.prediction_history[-100:]  # Последние 100
        
        # Статистика по методам
        methods_stats = {}
        for pred in recent_predictions:
            method = pred.voting_method
            if method not in methods_stats:
                methods_stats[method] = {'count': 0, 'avg_confidence': 0}
            
            methods_stats[method]['count'] += 1
            methods_stats[method]['avg_confidence'] += pred.confidence
        
        # Нормализация
        for method in methods_stats:
            if methods_stats[method]['count'] > 0:
                methods_stats[method]['avg_confidence'] /= methods_stats[method]['count']
        
        return {
            'total_predictions': len(self.prediction_history),
            'recent_predictions': len(recent_predictions),
            'methods_used': methods_stats,
            'current_weights': self.model_weights,
            'avg_confidence': np.mean([p.confidence for p in recent_predictions]),
            'meta_model_trained': self.is_meta_trained
        }

# Основная функция для демонстрации
async def main():
    """Демонстрация работы ансамбля"""
    
    print("🚀 Запуск системы ансамбля AI моделей")
    print("="*60)
    
    # Создание ансамбля
    ensemble = EnsembleSystem()
    
    # Символы для обучения
    symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
    
    print(f"📊 Обучение ансамбля на символах: {', '.join(symbols)}")
    
    try:
        # Обучение ансамбля
        results = await ensemble.train_ensemble(symbols, lookback_days=180)
        
        print("\n" + "="*60)
        print("📈 РЕЗУЛЬТАТЫ ОБУЧЕНИЯ АНСАМБЛЯ")
        print("="*60)
        
        # Вывод результатов
        validation_metrics = results['validation_metrics']
        
        print(f"\n🎯 ОБЩИЕ МЕТРИКИ:")
        print(f"   Винрейт: {validation_metrics.winrate:.1%}")
        print(f"   Precision: {validation_metrics.precision:.3f}")
        print(f"   Recall: {validation_metrics.recall:.3f}")
        print(f"   F1-Score: {validation_metrics.f1_score:.3f}")
        print(f"   Sharpe Ratio: {validation_metrics.sharpe_ratio:.3f}")
        print(f"   Max Drawdown: {validation_metrics.max_drawdown:.1%}")
        print(f"   Всего сделок: {validation_metrics.total_trades}")
        print(f"   Прибыльных: {validation_metrics.profitable_trades}")
        
        print(f"\n⚖️ ВЕСА МОДЕЛЕЙ:")
        for model_name, weight in results['ensemble_weights'].items():
            print(f"   {model_name}: {weight:.1%}")
        
        # Проверка достижения цели
        if validation_metrics.winrate >= 0.75:
            print(f"\n🎉 ЦЕЛЬ ДОСТИГНУТА!")
            print(f"   Винрейт {validation_metrics.winrate:.1%} >= 75%")
        else:
            print(f"\n⚠️  Цель не достигнута")
            print(f"   Нужно улучшить на {(0.75 - validation_metrics.winrate):.1%}")
        
        # Демонстрация предсказания
        print(f"\n🔮 ДЕМО ПРЕДСКАЗАНИЯ:")
        
        for symbol in symbols[:2]:
            try:
                prediction = await ensemble.predict(symbol, '1h', 'weighted_voting')
                
                print(f"\n💰 {symbol}:")
                print(f"   Сигнал: {'🟢 ПОКУПКА' if prediction.final_signal == 1 else '🔴 ПРОДАЖА'}")
                print(f"   Уверенность: {prediction.confidence:.1%}")
                print(f"   Метод: {prediction.voting_method}")
                print(f"   Моделей участвовало: {len(prediction.individual_predictions)}")
                
            except Exception as e:
                print(f"   ❌ Ошибка предсказания для {symbol}: {e}")
        
        # Сводка производительности
        performance = ensemble.get_performance_summary()
        print(f"\n📊 СВОДКА ПРОИЗВОДИТЕЛЬНОСТИ:")
        print(f"   Всего предсказаний: {performance['total_predictions']}")
        print(f"   Мета-модель обучена: {'✅' if performance['meta_model_trained'] else '❌'}")
        
    except Exception as e:
        print(f"❌ Ошибка обучения ансамбля: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())