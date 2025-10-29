"""
Система автоматической оптимизации параметров для торговой системы
Включает ML адаптацию, генетические алгоритмы, байесовскую оптимизацию
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Callable
import logging
from datetime import datetime, timedelta
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

# Оптимизационные библиотеки
from scipy.optimize import differential_evolution, minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
import optuna

# Генетические алгоритмы
from deap import base, creator, tools, algorithms
import random

class ParameterOptimizer:
    """Главный класс для автоматической оптимизации параметров"""
    
    def __init__(self, config_path: str = None):
        self.logger = logging.getLogger(__name__)
        self.config_path = config_path
        self.optimization_history = []
        self.best_parameters = {}
        self.performance_metrics = {}
        
        # Настройки оптимизации
        self.optimization_methods = {
            'bayesian': self._bayesian_optimization,
            'genetic': self._genetic_optimization,
            'differential_evolution': self._differential_evolution,
            'optuna': self._optuna_optimization,
            'ml_adaptive': self._ml_adaptive_optimization
        }
        
        # Параметры для оптимизации
        self.parameter_bounds = {
            'rsi_buy_threshold': (50, 80),
            'rsi_sell_threshold': (20, 50),
            'confidence_multiplier': (5, 25),
            'min_confidence': (0.1, 0.5),
            'atr_multiplier': (1.5, 4.0),
            'max_position_size': (0.01, 0.1),
            'stop_loss_pct': (0.005, 0.03),
            'take_profit_ratio': (1.0, 3.0),
            'lgbm_num_leaves': (10, 50),
            'lgbm_max_depth': (3, 8),
            'lgbm_learning_rate': (0.01, 0.2),
            'volatility_threshold': (0.01, 0.05)
        }
        
        # Веса для мульти-объективной оптимизации
        self.objective_weights = {
            'win_rate': 0.3,
            'roi': 0.25,
            'sharpe_ratio': 0.2,
            'max_drawdown': -0.15,  # Отрицательный вес (минимизируем)
            'profit_factor': 0.1
        }
        
        self.executor = ThreadPoolExecutor(max_workers=4)
        
    def optimize_parameters(self, 
                          method: str = 'bayesian',
                          n_trials: int = 100,
                          target_metrics: Dict[str, float] = None) -> Dict[str, Any]:
        """
        Основной метод оптимизации параметров
        
        Args:
            method: Метод оптимизации ('bayesian', 'genetic', 'optuna', etc.)
            n_trials: Количество итераций оптимизации
            target_metrics: Целевые метрики для достижения
            
        Returns:
            Словарь с оптимальными параметрами и метриками
        """
        self.logger.info(f"Начинаем оптимизацию параметров методом: {method}")
        
        if target_metrics is None:
            target_metrics = {
                'win_rate': 0.75,
                'roi': 0.08,
                'sharpe_ratio': 1.5,
                'max_drawdown': 0.05,
                'profit_factor': 2.0
            }
        
        start_time = datetime.now()
        
        try:
            if method in self.optimization_methods:
                result = self.optimization_methods[method](n_trials, target_metrics)
            else:
                raise ValueError(f"Неизвестный метод оптимизации: {method}")
                
            optimization_time = (datetime.now() - start_time).total_seconds()
            
            result.update({
                'optimization_method': method,
                'optimization_time': optimization_time,
                'timestamp': datetime.now().isoformat(),
                'target_metrics': target_metrics
            })
            
            self.optimization_history.append(result)
            self._save_optimization_results(result)
            
            self.logger.info(f"Оптимизация завершена за {optimization_time:.2f} секунд")
            return result
            
        except Exception as e:
            self.logger.error(f"Ошибка при оптимизации: {e}")
            raise
    
    def _bayesian_optimization(self, n_trials: int, target_metrics: Dict) -> Dict[str, Any]:
        """Байесовская оптимизация с использованием Gaussian Process"""
        
        # Создаем Gaussian Process модель
        kernel = Matern(length_scale=1.0, nu=2.5)
        gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True)
        
        # Инициализация случайными точками
        n_initial = min(20, n_trials // 5)
        X_init = []
        y_init = []
        
        for _ in range(n_initial):
            params = self._sample_random_parameters()
            score = self._evaluate_parameters(params)
            X_init.append(list(params.values()))
            y_init.append(score)
        
        X_init = np.array(X_init)
        y_init = np.array(y_init)
        
        best_score = max(y_init)
        best_params = dict(zip(self.parameter_bounds.keys(), X_init[np.argmax(y_init)]))
        
        # Основной цикл байесовской оптимизации
        for i in range(n_initial, n_trials):
            # Обучаем GP на текущих данных
            gp.fit(X_init, y_init)
            
            # Находим следующую точку для оценки (acquisition function)
            next_params = self._acquisition_function(gp, X_init, y_init)
            next_score = self._evaluate_parameters(next_params)
            
            # Обновляем данные
            X_init = np.vstack([X_init, list(next_params.values())])
            y_init = np.append(y_init, next_score)
            
            if next_score > best_score:
                best_score = next_score
                best_params = next_params.copy()
                
            if i % 10 == 0:
                self.logger.info(f"Байесовская оптимизация: итерация {i}, лучший score: {best_score:.4f}")
        
        return {
            'best_parameters': best_params,
            'best_score': best_score,
            'all_scores': y_init.tolist(),
            'convergence_iteration': np.argmax(y_init)
        }
    
    def _genetic_optimization(self, n_trials: int, target_metrics: Dict) -> Dict[str, Any]:
        """Генетический алгоритм для оптимизации параметров"""
        
        # Настройка DEAP
        if not hasattr(creator, "FitnessMax"):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMax)
        
        toolbox = base.Toolbox()
        
        # Генерация индивидуумов
        def create_individual():
            params = self._sample_random_parameters()
            return creator.Individual(list(params.values()))
        
        toolbox.register("individual", create_individual)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        
        # Функция оценки
        def evaluate_individual(individual):
            params = dict(zip(self.parameter_bounds.keys(), individual))
            score = self._evaluate_parameters(params)
            return (score,)
        
        toolbox.register("evaluate", evaluate_individual)
        toolbox.register("mate", tools.cxBlend, alpha=0.5)
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
        toolbox.register("select", tools.selTournament, tournsize=3)
        
        # Параметры ГА
        population_size = min(50, n_trials // 10)
        n_generations = n_trials // population_size
        
        # Создание начальной популяции
        population = toolbox.population(n=population_size)
        
        # Статистика
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("max", np.max)
        stats.register("min", np.min)
        
        # Запуск эволюции
        population, logbook = algorithms.eaSimple(
            population, toolbox, cxpb=0.7, mutpb=0.3, 
            ngen=n_generations, stats=stats, verbose=False
        )
        
        # Лучший индивидуум
        best_individual = tools.selBest(population, 1)[0]
        best_params = dict(zip(self.parameter_bounds.keys(), best_individual))
        best_score = best_individual.fitness.values[0]
        
        return {
            'best_parameters': best_params,
            'best_score': best_score,
            'evolution_stats': logbook,
            'final_population_size': len(population)
        }
    
    def _optuna_optimization(self, n_trials: int, target_metrics: Dict) -> Dict[str, Any]:
        """Оптимизация с использованием Optuna"""
        
        def objective(trial):
            params = {}
            for param_name, (low, high) in self.parameter_bounds.items():
                if isinstance(low, int) and isinstance(high, int):
                    params[param_name] = trial.suggest_int(param_name, low, high)
                else:
                    params[param_name] = trial.suggest_float(param_name, low, high)
            
            return self._evaluate_parameters(params)
        
        # Создание исследования
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        return {
            'best_parameters': study.best_params,
            'best_score': study.best_value,
            'n_trials': len(study.trials),
            'optimization_history': [trial.value for trial in study.trials if trial.value is not None]
        }
    
    def _differential_evolution(self, n_trials: int, target_metrics: Dict) -> Dict[str, Any]:
        """Дифференциальная эволюция"""
        
        bounds = list(self.parameter_bounds.values())
        
        def objective(x):
            params = dict(zip(self.parameter_bounds.keys(), x))
            return -self._evaluate_parameters(params)  # Минимизируем отрицательный score
        
        result = differential_evolution(
            objective, bounds, maxiter=n_trials//10, 
            popsize=10, seed=42, atol=1e-6
        )
        
        best_params = dict(zip(self.parameter_bounds.keys(), result.x))
        best_score = -result.fun
        
        return {
            'best_parameters': best_params,
            'best_score': best_score,
            'n_iterations': result.nit,
            'success': result.success
        }
    
    def _ml_adaptive_optimization(self, n_trials: int, target_metrics: Dict) -> Dict[str, Any]:
        """ML-адаптивная оптимизация с обучением на исторических данных"""
        
        # Собираем данные для обучения
        X_train = []
        y_train = []
        
        # Генерируем начальные данные
        for _ in range(min(50, n_trials // 2)):
            params = self._sample_random_parameters()
            score = self._evaluate_parameters(params)
            X_train.append(list(params.values()))
            y_train.append(score)
        
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        # Обучаем модель предсказания производительности
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        
        best_score = max(y_train)
        best_params = dict(zip(self.parameter_bounds.keys(), X_train[np.argmax(y_train)]))
        
        # Адаптивная оптимизация
        for i in range(len(X_train), n_trials):
            # Генерируем кандидатов
            candidates = []
            for _ in range(20):
                candidate = self._sample_random_parameters()
                candidates.append(list(candidate.values()))
            
            candidates = np.array(candidates)
            
            # Предсказываем производительность
            predicted_scores = rf_model.predict(candidates)
            
            # Выбираем лучшего кандидата
            best_candidate_idx = np.argmax(predicted_scores)
            next_params = dict(zip(self.parameter_bounds.keys(), candidates[best_candidate_idx]))
            
            # Оцениваем реальную производительность
            actual_score = self._evaluate_parameters(next_params)
            
            # Обновляем данные и переобучаем модель
            X_train = np.vstack([X_train, candidates[best_candidate_idx]])
            y_train = np.append(y_train, actual_score)
            rf_model.fit(X_train, y_train)
            
            if actual_score > best_score:
                best_score = actual_score
                best_params = next_params.copy()
            
            if i % 10 == 0:
                self.logger.info(f"ML-адаптивная оптимизация: итерация {i}, лучший score: {best_score:.4f}")
        
        # Важность признаков
        feature_importance = dict(zip(self.parameter_bounds.keys(), rf_model.feature_importances_))
        
        return {
            'best_parameters': best_params,
            'best_score': best_score,
            'feature_importance': feature_importance,
            'model_score': rf_model.score(X_train, y_train)
        }
    
    def _sample_random_parameters(self) -> Dict[str, float]:
        """Генерация случайных параметров в заданных границах"""
        params = {}
        for param_name, (low, high) in self.parameter_bounds.items():
            if isinstance(low, int) and isinstance(high, int):
                params[param_name] = random.randint(low, high)
            else:
                params[param_name] = random.uniform(low, high)
        return params
    
    def _acquisition_function(self, gp, X, y, n_candidates=1000):
        """Acquisition function для байесовской оптимизации (Upper Confidence Bound)"""
        
        # Генерируем кандидатов
        candidates = []
        for _ in range(n_candidates):
            candidate = self._sample_random_parameters()
            candidates.append(list(candidate.values()))
        
        candidates = np.array(candidates)
        
        # Предсказания GP
        mu, sigma = gp.predict(candidates, return_std=True)
        
        # UCB acquisition function
        kappa = 2.0  # Параметр исследования
        acquisition_values = mu + kappa * sigma
        
        # Выбираем лучшего кандидата
        best_idx = np.argmax(acquisition_values)
        best_candidate = dict(zip(self.parameter_bounds.keys(), candidates[best_idx]))
        
        return best_candidate
    
    def _evaluate_parameters(self, params: Dict[str, float]) -> float:
        """
        Оценка качества параметров
        Здесь должна быть интеграция с системой бэктестинга
        """
        # Симуляция оценки (в реальной системе здесь будет бэктестинг)
        # Возвращаем комбинированный score на основе целевых метрик
        
        # Простая эвристическая оценка для демонстрации
        score = 0.0
        
        # RSI параметры
        rsi_buy = params.get('rsi_buy_threshold', 65)
        rsi_sell = params.get('rsi_sell_threshold', 35)
        if 55 <= rsi_buy <= 75 and 25 <= rsi_sell <= 45:
            score += 0.2
        
        # Confidence параметры
        conf_mult = params.get('confidence_multiplier', 10)
        min_conf = params.get('min_confidence', 0.25)
        if 8 <= conf_mult <= 15 and 0.15 <= min_conf <= 0.35:
            score += 0.2
        
        # Risk management
        atr_mult = params.get('atr_multiplier', 2.5)
        max_pos = params.get('max_position_size', 0.05)
        if 2.0 <= atr_mult <= 3.5 and 0.02 <= max_pos <= 0.08:
            score += 0.2
        
        # LGBM параметры
        num_leaves = params.get('lgbm_num_leaves', 20)
        max_depth = params.get('lgbm_max_depth', 5)
        if 15 <= num_leaves <= 30 and 4 <= max_depth <= 6:
            score += 0.2
        
        # Добавляем случайность для симуляции реальной оценки
        score += random.uniform(-0.1, 0.3)
        
        return max(0.0, min(1.0, score))
    
    def _save_optimization_results(self, result: Dict[str, Any]):
        """Сохранение результатов оптимизации"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"optimization_results_{timestamp}.json"
            filepath = f"/Users/mac/Documents/Peper Binance v4 Clean/optimization/{filename}"
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False, default=str)
                
            self.logger.info(f"Результаты оптимизации сохранены: {filepath}")
            
        except Exception as e:
            self.logger.error(f"Ошибка при сохранении результатов: {e}")
    
    async def continuous_optimization(self, 
                                    interval_hours: int = 24,
                                    method: str = 'bayesian',
                                    n_trials: int = 50):
        """Непрерывная оптимизация параметров"""
        
        self.logger.info(f"Запуск непрерывной оптимизации каждые {interval_hours} часов")
        
        while True:
            try:
                self.logger.info("Начинаем цикл непрерывной оптимизации")
                
                # Запускаем оптимизацию
                result = await asyncio.get_event_loop().run_in_executor(
                    self.executor, 
                    self.optimize_parameters, 
                    method, 
                    n_trials
                )
                
                # Применяем новые параметры если они лучше
                if self._should_apply_parameters(result):
                    await self._apply_optimized_parameters(result['best_parameters'])
                
                # Ждем до следующего цикла
                await asyncio.sleep(interval_hours * 3600)
                
            except Exception as e:
                self.logger.error(f"Ошибка в непрерывной оптимизации: {e}")
                await asyncio.sleep(3600)  # Ждем час перед повтором
    
    def _should_apply_parameters(self, result: Dict[str, Any]) -> bool:
        """Определяет, стоит ли применять новые параметры"""
        current_best = self.best_parameters.get('score', 0)
        new_score = result.get('best_score', 0)
        
        # Применяем если улучшение больше 5%
        return new_score > current_best * 1.05
    
    async def _apply_optimized_parameters(self, parameters: Dict[str, float]):
        """Применение оптимизированных параметров к системе"""
        try:
            # Здесь должна быть интеграция с системой конфигурации
            self.logger.info("Применяем оптимизированные параметры:")
            for param, value in parameters.items():
                self.logger.info(f"  {param}: {value}")
            
            self.best_parameters = {
                'parameters': parameters,
                'score': self._evaluate_parameters(parameters),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Ошибка при применении параметров: {e}")

class AutoRetrainingManager:
    """Менеджер автоматического переобучения моделей"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.retrain_schedule = {}
        self.performance_threshold = 0.05  # 5% снижение производительности
        
    async def schedule_retraining(self, 
                                model_name: str, 
                                interval_days: int = 7,
                                performance_check_hours: int = 6):
        """Планирование автоматического переобучения"""
        
        self.logger.info(f"Планируем переобучение модели {model_name} каждые {interval_days} дней")
        
        while True:
            try:
                # Проверяем производительность
                current_performance = await self._check_model_performance(model_name)
                baseline_performance = self._get_baseline_performance(model_name)
                
                # Если производительность упала значительно
                if (baseline_performance - current_performance) > self.performance_threshold:
                    self.logger.warning(f"Производительность модели {model_name} упала, запускаем переобучение")
                    await self._retrain_model(model_name)
                
                # Плановое переобучение
                elif self._should_retrain_scheduled(model_name, interval_days):
                    self.logger.info(f"Плановое переобучение модели {model_name}")
                    await self._retrain_model(model_name)
                
                await asyncio.sleep(performance_check_hours * 3600)
                
            except Exception as e:
                self.logger.error(f"Ошибка в планировщике переобучения: {e}")
                await asyncio.sleep(3600)
    
    async def _check_model_performance(self, model_name: str) -> float:
        """Проверка текущей производительности модели"""
        # Здесь должна быть интеграция с системой мониторинга
        return 0.75  # Заглушка
    
    def _get_baseline_performance(self, model_name: str) -> float:
        """Получение базовой производительности модели"""
        # Здесь должна быть интеграция с базой данных метрик
        return 0.80  # Заглушка
    
    def _should_retrain_scheduled(self, model_name: str, interval_days: int) -> bool:
        """Проверка необходимости планового переобучения"""
        last_retrain = self.retrain_schedule.get(model_name)
        if last_retrain is None:
            return True
        
        return (datetime.now() - last_retrain).days >= interval_days
    
    async def _retrain_model(self, model_name: str):
        """Переобучение модели"""
        try:
            self.logger.info(f"Начинаем переобучение модели {model_name}")
            
            # Здесь должна быть интеграция с системой обучения моделей
            # Симуляция переобучения
            await asyncio.sleep(5)
            
            self.retrain_schedule[model_name] = datetime.now()
            self.logger.info(f"Переобучение модели {model_name} завершено")
            
        except Exception as e:
            self.logger.error(f"Ошибка при переобучении модели {model_name}: {e}")

# Пример использования
if __name__ == "__main__":
    # Настройка логирования
    logging.basicConfig(level=logging.INFO)
    
    # Создание оптимизатора
    optimizer = ParameterOptimizer()
    
    # Запуск оптимизации
    result = optimizer.optimize_parameters(method='bayesian', n_trials=50)
    print("Результаты оптимизации:")
    print(f"Лучшие параметры: {result['best_parameters']}")
    print(f"Лучший score: {result['best_score']:.4f}")