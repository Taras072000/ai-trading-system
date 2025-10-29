"""
Продвинутые AI возможности для торговой системы
Deep Learning прогнозирование, Reinforcement Learning, Ensemble методы, NLP анализ новостей
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import tensorflow as tf
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import gym
from gym import spaces
import stable_baselines3 as sb3
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
import transformers
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import requests
import feedparser
import re
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import asyncio
import aiohttp
import json
import pickle
import warnings
warnings.filterwarnings('ignore')

@dataclass
class PredictionResult:
    """Результат прогнозирования"""
    timestamp: datetime
    symbol: str
    prediction_type: str  # price, direction, volatility
    predicted_value: float
    confidence: float
    model_name: str
    features_used: List[str]
    metadata: Dict[str, Any] = None

@dataclass
class NewsAnalysisResult:
    """Результат анализа новостей"""
    timestamp: datetime
    headline: str
    content: str
    sentiment_score: float  # -1 to 1
    relevance_score: float  # 0 to 1
    impact_prediction: str  # bullish, bearish, neutral
    confidence: float
    keywords: List[str]
    source: str

@dataclass
class EnsemblePrediction:
    """Результат ансамблевого прогнозирования"""
    timestamp: datetime
    symbol: str
    final_prediction: float
    confidence: float
    model_predictions: Dict[str, float]
    model_weights: Dict[str, float]
    consensus_strength: float

class DeepLearningPredictor:
    """Deep Learning модель для прогнозирования цен"""
    
    def __init__(self, sequence_length: int = 60, features_count: int = 10):
        self.sequence_length = sequence_length
        self.features_count = features_count
        self.model = None
        self.scaler = MinMaxScaler()
        self.is_trained = False
        self.logger = logging.getLogger(__name__)
        
        # Создание модели
        self._build_model()
    
    def _build_model(self):
        """Построение LSTM модели"""
        
        class LSTMPredictor(nn.Module):
            def __init__(self, input_size, hidden_size=128, num_layers=3, dropout=0.2):
                super(LSTMPredictor, self).__init__()
                
                self.hidden_size = hidden_size
                self.num_layers = num_layers
                
                # LSTM слои
                self.lstm = nn.LSTM(
                    input_size, 
                    hidden_size, 
                    num_layers, 
                    batch_first=True,
                    dropout=dropout
                )
                
                # Attention механизм
                self.attention = nn.MultiheadAttention(hidden_size, num_heads=8)
                
                # Полносвязные слои
                self.fc_layers = nn.Sequential(
                    nn.Linear(hidden_size, hidden_size // 2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_size // 2, hidden_size // 4),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_size // 4, 1)
                )
                
                # Batch normalization
                self.batch_norm = nn.BatchNorm1d(hidden_size)
            
            def forward(self, x):
                batch_size = x.size(0)
                
                # LSTM forward pass
                lstm_out, (hidden, cell) = self.lstm(x)
                
                # Attention
                attn_out, _ = self.attention(
                    lstm_out.transpose(0, 1),
                    lstm_out.transpose(0, 1),
                    lstm_out.transpose(0, 1)
                )
                attn_out = attn_out.transpose(0, 1)
                
                # Используем последний выход
                last_output = attn_out[:, -1, :]
                
                # Batch normalization
                last_output = self.batch_norm(last_output)
                
                # Полносвязные слои
                output = self.fc_layers(last_output)
                
                return output
        
        self.model = LSTMPredictor(self.features_count)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-5)
        self.criterion = nn.MSELoss()
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=10, factor=0.5
        )
    
    def prepare_data(self, data: pd.DataFrame, target_column: str = 'close') -> Tuple[np.ndarray, np.ndarray]:
        """Подготовка данных для обучения"""
        
        # Создание технических индикаторов
        data = self._add_technical_indicators(data)
        
        # Выбор признаков
        feature_columns = [
            'open', 'high', 'low', 'close', 'volume',
            'sma_20', 'ema_12', 'rsi', 'macd', 'bb_upper', 'bb_lower',
            'atr', 'adx', 'stoch_k', 'stoch_d'
        ]
        
        # Фильтрация доступных колонок
        available_features = [col for col in feature_columns if col in data.columns]
        
        if len(available_features) < 5:
            # Создаем базовые признаки если их недостаточно
            data['returns'] = data[target_column].pct_change()
            data['volatility'] = data['returns'].rolling(20).std()
            data['price_ma_ratio'] = data[target_column] / data[target_column].rolling(20).mean()
            available_features.extend(['returns', 'volatility', 'price_ma_ratio'])
        
        # Нормализация данных
        features_data = data[available_features].fillna(method='ffill').fillna(0)
        target_data = data[target_column].values
        
        # Масштабирование
        features_scaled = self.scaler.fit_transform(features_data)
        
        # Создание последовательностей
        X, y = [], []
        for i in range(self.sequence_length, len(features_scaled)):
            X.append(features_scaled[i-self.sequence_length:i])
            y.append(target_data[i])
        
        return np.array(X), np.array(y)
    
    def _add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Добавление технических индикаторов"""
        
        df = data.copy()
        
        # Простые скользящие средние
        df['sma_20'] = df['close'].rolling(20).mean()
        df['ema_12'] = df['close'].ewm(span=12).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema_12 = df['close'].ewm(span=12).mean()
        ema_26 = df['close'].ewm(span=26).mean()
        df['macd'] = ema_12 - ema_26
        
        # Bollinger Bands
        sma_20 = df['close'].rolling(20).mean()
        std_20 = df['close'].rolling(20).std()
        df['bb_upper'] = sma_20 + (std_20 * 2)
        df['bb_lower'] = sma_20 - (std_20 * 2)
        
        # ATR
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['atr'] = true_range.rolling(14).mean()
        
        # ADX (упрощенная версия)
        plus_dm = df['high'].diff()
        minus_dm = df['low'].diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0
        df['adx'] = (plus_dm.rolling(14).mean() + abs(minus_dm.rolling(14).mean())) / df['atr']
        
        # Stochastic
        low_14 = df['low'].rolling(14).min()
        high_14 = df['high'].rolling(14).max()
        df['stoch_k'] = 100 * ((df['close'] - low_14) / (high_14 - low_14))
        df['stoch_d'] = df['stoch_k'].rolling(3).mean()
        
        return df
    
    def train(self, data: pd.DataFrame, epochs: int = 100, batch_size: int = 32) -> Dict[str, Any]:
        """Обучение модели"""
        
        self.logger.info("Начало обучения Deep Learning модели")
        
        # Подготовка данных
        X, y = self.prepare_data(data)
        
        if len(X) < 100:
            raise ValueError("Недостаточно данных для обучения (минимум 100 образцов)")
        
        # Разделение на обучающую и валидационную выборки
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Преобразование в тензоры
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
        X_val_tensor = torch.FloatTensor(X_val)
        y_val_tensor = torch.FloatTensor(y_val).unsqueeze(1)
        
        # Создание DataLoader
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Обучение
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Обучение
            self.model.train()
            epoch_train_loss = 0
            
            for batch_X, batch_y in train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                epoch_train_loss += loss.item()
            
            # Валидация
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val_tensor)
                val_loss = self.criterion(val_outputs, y_val_tensor)
            
            train_losses.append(epoch_train_loss / len(train_loader))
            val_losses.append(val_loss.item())
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Сохранение лучшей модели
                torch.save(self.model.state_dict(), 'best_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= 20:
                    self.logger.info(f"Early stopping на эпохе {epoch}")
                    break
            
            # Обновление learning rate
            self.scheduler.step(val_loss)
            
            if epoch % 10 == 0:
                self.logger.info(f"Эпоха {epoch}: Train Loss = {train_losses[-1]:.6f}, Val Loss = {val_losses[-1]:.6f}")
        
        # Загрузка лучшей модели
        self.model.load_state_dict(torch.load('best_model.pth'))
        self.is_trained = True
        
        # Расчет метрик
        self.model.eval()
        with torch.no_grad():
            train_pred = self.model(X_train_tensor).numpy()
            val_pred = self.model(X_val_tensor).numpy()
        
        train_mse = mean_squared_error(y_train, train_pred)
        val_mse = mean_squared_error(y_val, val_pred)
        train_mae = mean_absolute_error(y_train, train_pred)
        val_mae = mean_absolute_error(y_val, val_pred)
        
        training_results = {
            'train_mse': train_mse,
            'val_mse': val_mse,
            'train_mae': train_mae,
            'val_mae': val_mae,
            'epochs_trained': epoch + 1,
            'best_val_loss': best_val_loss,
            'train_losses': train_losses,
            'val_losses': val_losses
        }
        
        self.logger.info(f"Обучение завершено. Val MSE: {val_mse:.6f}, Val MAE: {val_mae:.6f}")
        
        return training_results
    
    def predict(self, data: pd.DataFrame, symbol: str = "UNKNOWN") -> PredictionResult:
        """Прогнозирование"""
        
        if not self.is_trained:
            raise ValueError("Модель не обучена")
        
        # Подготовка данных
        data_with_indicators = self._add_technical_indicators(data)
        
        # Получение последней последовательности
        feature_columns = [
            'open', 'high', 'low', 'close', 'volume',
            'sma_20', 'ema_12', 'rsi', 'macd', 'bb_upper', 'bb_lower',
            'atr', 'adx', 'stoch_k', 'stoch_d'
        ]
        
        available_features = [col for col in feature_columns if col in data_with_indicators.columns]
        
        if len(available_features) < 5:
            # Создаем базовые признаки
            data_with_indicators['returns'] = data_with_indicators['close'].pct_change()
            data_with_indicators['volatility'] = data_with_indicators['returns'].rolling(20).std()
            data_with_indicators['price_ma_ratio'] = data_with_indicators['close'] / data_with_indicators['close'].rolling(20).mean()
            available_features.extend(['returns', 'volatility', 'price_ma_ratio'])
        
        features_data = data_with_indicators[available_features].fillna(method='ffill').fillna(0)
        
        if len(features_data) < self.sequence_length:
            raise ValueError(f"Недостаточно данных для прогнозирования (нужно минимум {self.sequence_length} точек)")
        
        # Масштабирование
        features_scaled = self.scaler.transform(features_data)
        
        # Получение последней последовательности
        last_sequence = features_scaled[-self.sequence_length:].reshape(1, self.sequence_length, -1)
        
        # Прогнозирование
        self.model.eval()
        with torch.no_grad():
            sequence_tensor = torch.FloatTensor(last_sequence)
            prediction = self.model(sequence_tensor).item()
        
        # Расчет уверенности (упрощенный)
        recent_volatility = data['close'].pct_change().tail(20).std()
        confidence = max(0.1, min(0.95, 1 - recent_volatility * 10))
        
        return PredictionResult(
            timestamp=datetime.now(),
            symbol=symbol,
            prediction_type="price",
            predicted_value=prediction,
            confidence=confidence,
            model_name="DeepLearning_LSTM",
            features_used=available_features,
            metadata={
                'sequence_length': self.sequence_length,
                'recent_volatility': recent_volatility
            }
        )

class ReinforcementLearningTrader:
    """Reinforcement Learning агент для торговли"""
    
    def __init__(self, observation_space_size: int = 20, action_space_size: int = 3):
        self.observation_space_size = observation_space_size
        self.action_space_size = action_space_size  # 0: Hold, 1: Buy, 2: Sell
        self.model = None
        self.env = None
        self.is_trained = False
        self.logger = logging.getLogger(__name__)
        
        # Создание среды
        self._create_environment()
    
    def _create_environment(self):
        """Создание торговой среды"""
        
        class TradingEnvironment(gym.Env):
            def __init__(self, data: pd.DataFrame = None):
                super(TradingEnvironment, self).__init__()
                
                # Пространства действий и наблюдений
                self.action_space = spaces.Discrete(3)  # Hold, Buy, Sell
                self.observation_space = spaces.Box(
                    low=-np.inf, high=np.inf, 
                    shape=(20,), dtype=np.float32
                )
                
                # Данные
                self.data = data
                self.current_step = 0
                self.max_steps = len(data) - 1 if data is not None else 1000
                
                # Состояние портфеля
                self.balance = 10000.0
                self.shares = 0
                self.net_worth = self.balance
                self.max_net_worth = self.balance
                
                # История для расчета наград
                self.price_history = []
                self.action_history = []
                self.reward_history = []
            
            def reset(self):
                self.current_step = 20  # Начинаем с 20-го шага для истории
                self.balance = 10000.0
                self.shares = 0
                self.net_worth = self.balance
                self.max_net_worth = self.balance
                self.price_history = []
                self.action_history = []
                self.reward_history = []
                
                return self._get_observation()
            
            def step(self, action):
                if self.data is None:
                    # Симуляция для тестирования
                    return self._simulate_step(action)
                
                current_price = self.data.iloc[self.current_step]['close']
                
                # Выполнение действия
                reward = self._execute_action(action, current_price)
                
                # Обновление состояния
                self.current_step += 1
                self.net_worth = self.balance + self.shares * current_price
                self.max_net_worth = max(self.max_net_worth, self.net_worth)
                
                # Проверка завершения эпизода
                done = self.current_step >= self.max_steps
                
                # Штраф за большие просадки
                drawdown = (self.max_net_worth - self.net_worth) / self.max_net_worth
                if drawdown > 0.2:
                    reward -= 100
                    done = True
                
                obs = self._get_observation()
                info = {
                    'balance': self.balance,
                    'shares': self.shares,
                    'net_worth': self.net_worth,
                    'current_price': current_price
                }
                
                return obs, reward, done, info
            
            def _execute_action(self, action, current_price):
                reward = 0
                
                if action == 1:  # Buy
                    if self.balance > current_price:
                        shares_to_buy = self.balance // current_price
                        self.shares += shares_to_buy
                        self.balance -= shares_to_buy * current_price
                        reward = 1  # Небольшая награда за покупку
                
                elif action == 2:  # Sell
                    if self.shares > 0:
                        self.balance += self.shares * current_price
                        # Награда за прибыльную продажу
                        if len(self.price_history) > 0:
                            last_buy_price = self.price_history[-1]
                            if current_price > last_buy_price:
                                reward = (current_price - last_buy_price) / last_buy_price * 100
                            else:
                                reward = -10  # Штраф за убыточную продажу
                        self.shares = 0
                
                # Награда за удержание позиции в правильном направлении
                if len(self.price_history) > 0:
                    price_change = (current_price - self.price_history[-1]) / self.price_history[-1]
                    if action == 0:  # Hold
                        if self.shares > 0 and price_change > 0:
                            reward += price_change * 50
                        elif self.shares == 0 and price_change < 0:
                            reward += abs(price_change) * 25
                
                self.price_history.append(current_price)
                self.action_history.append(action)
                self.reward_history.append(reward)
                
                return reward
            
            def _get_observation(self):
                if self.data is None:
                    # Симуляция наблюдений
                    return np.random.randn(20).astype(np.float32)
                
                if self.current_step < 20:
                    # Заполняем нулями если недостаточно истории
                    obs = np.zeros(20, dtype=np.float32)
                    available_data = self.data.iloc[:self.current_step+1]
                    if len(available_data) > 0:
                        obs[:len(available_data)] = available_data['close'].values
                    return obs
                
                # Получаем последние 20 цен
                recent_prices = self.data.iloc[self.current_step-19:self.current_step+1]['close'].values
                
                # Нормализация
                if len(recent_prices) == 20:
                    normalized_prices = (recent_prices - recent_prices.mean()) / (recent_prices.std() + 1e-8)
                    return normalized_prices.astype(np.float32)
                else:
                    return np.zeros(20, dtype=np.float32)
            
            def _simulate_step(self, action):
                # Симуляция для тестирования без реальных данных
                obs = np.random.randn(20).astype(np.float32)
                reward = np.random.randn() * 10
                done = self.current_step >= 100
                self.current_step += 1
                info = {}
                return obs, reward, done, info
        
        self.env_class = TradingEnvironment
    
    def train(self, data: pd.DataFrame, total_timesteps: int = 100000) -> Dict[str, Any]:
        """Обучение RL агента"""
        
        self.logger.info("Начало обучения Reinforcement Learning агента")
        
        # Создание среды с данными
        self.env = self.env_class(data)
        
        # Создание модели PPO
        self.model = PPO(
            "MlpPolicy",
            self.env,
            verbose=1,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            tensorboard_log="./ppo_trading_tensorboard/"
        )
        
        # Callback для мониторинга
        class TrainingCallback(BaseCallback):
            def __init__(self, verbose=0):
                super(TrainingCallback, self).__init__(verbose)
                self.episode_rewards = []
                self.episode_lengths = []
            
            def _on_step(self) -> bool:
                if self.locals.get('dones')[0]:
                    episode_reward = self.locals.get('infos')[0].get('episode', {}).get('r', 0)
                    episode_length = self.locals.get('infos')[0].get('episode', {}).get('l', 0)
                    
                    if episode_reward != 0:
                        self.episode_rewards.append(episode_reward)
                        self.episode_lengths.append(episode_length)
                
                return True
        
        callback = TrainingCallback()
        
        # Обучение
        self.model.learn(total_timesteps=total_timesteps, callback=callback)
        self.is_trained = True
        
        # Сохранение модели
        self.model.save("rl_trading_model")
        
        training_results = {
            'total_timesteps': total_timesteps,
            'episode_rewards': callback.episode_rewards[-100:],  # Последние 100 эпизодов
            'average_reward': np.mean(callback.episode_rewards[-100:]) if callback.episode_rewards else 0,
            'episode_lengths': callback.episode_lengths[-100:],
            'average_length': np.mean(callback.episode_lengths[-100:]) if callback.episode_lengths else 0
        }
        
        self.logger.info(f"Обучение RL завершено. Средняя награда: {training_results['average_reward']:.2f}")
        
        return training_results
    
    def predict_action(self, data: pd.DataFrame, symbol: str = "UNKNOWN") -> PredictionResult:
        """Прогнозирование действия"""
        
        if not self.is_trained:
            raise ValueError("Модель не обучена")
        
        # Создание временной среды для получения наблюдения
        temp_env = self.env_class(data)
        temp_env.current_step = len(data) - 1
        obs = temp_env._get_observation()
        
        # Получение действия от модели
        action, _states = self.model.predict(obs, deterministic=True)
        
        # Преобразование действия в торговое решение
        action_map = {0: "HOLD", 1: "BUY", 2: "SELL"}
        predicted_action = action_map[action]
        
        # Расчет уверенности (упрощенный)
        confidence = 0.7  # Базовая уверенность для RL
        
        return PredictionResult(
            timestamp=datetime.now(),
            symbol=symbol,
            prediction_type="action",
            predicted_value=float(action),
            confidence=confidence,
            model_name="ReinforcementLearning_PPO",
            features_used=["price_history"],
            metadata={
                'action_name': predicted_action,
                'observation_shape': obs.shape
            }
        )

class EnsemblePredictor:
    """Ансамблевый предиктор, объединяющий несколько моделей"""
    
    def __init__(self):
        self.models = {}
        self.model_weights = {}
        self.performance_history = {}
        self.logger = logging.getLogger(__name__)
        
        # Инициализация базовых моделей
        self._initialize_base_models()
    
    def _initialize_base_models(self):
        """Инициализация базовых моделей"""
        
        # Random Forest
        self.models['random_forest'] = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
        # Gradient Boosting
        self.models['gradient_boosting'] = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
        
        # Начальные веса (равномерное распределение)
        num_models = len(self.models)
        for model_name in self.models.keys():
            self.model_weights[model_name] = 1.0 / num_models
            self.performance_history[model_name] = []
    
    def add_model(self, name: str, model: Any, weight: float = None):
        """Добавление модели в ансамбль"""
        
        self.models[name] = model
        
        if weight is None:
            # Пересчитываем веса равномерно
            num_models = len(self.models)
            for model_name in self.models.keys():
                self.model_weights[model_name] = 1.0 / num_models
        else:
            self.model_weights[name] = weight
            # Нормализуем веса
            total_weight = sum(self.model_weights.values())
            for model_name in self.model_weights.keys():
                self.model_weights[model_name] /= total_weight
        
        self.performance_history[name] = []
        self.logger.info(f"Модель {name} добавлена в ансамбль с весом {self.model_weights[name]:.3f}")
    
    def train(self, data: pd.DataFrame, target_column: str = 'close') -> Dict[str, Any]:
        """Обучение всех моделей в ансамбле"""
        
        self.logger.info("Начало обучения ансамбля моделей")
        
        # Подготовка признаков
        features = self._prepare_features(data)
        target = data[target_column].values
        
        # Разделение данных
        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=0.2, random_state=42
        )
        
        training_results = {}
        
        # Обучение каждой модели
        for name, model in self.models.items():
            try:
                if hasattr(model, 'fit'):
                    # Sklearn модели
                    model.fit(X_train, y_train)
                    
                    # Оценка производительности
                    train_pred = model.predict(X_train)
                    test_pred = model.predict(X_test)
                    
                    train_mse = mean_squared_error(y_train, train_pred)
                    test_mse = mean_squared_error(y_test, test_pred)
                    
                    training_results[name] = {
                        'train_mse': train_mse,
                        'test_mse': test_mse,
                        'train_mae': mean_absolute_error(y_train, train_pred),
                        'test_mae': mean_absolute_error(y_test, test_pred)
                    }
                    
                    # Обновление истории производительности
                    self.performance_history[name].append(test_mse)
                    
                elif hasattr(model, 'predict'):
                    # Уже обученные модели (например, Deep Learning или RL)
                    try:
                        test_pred = []
                        for i in range(len(X_test)):
                            # Создаем DataFrame для совместимости
                            sample_df = pd.DataFrame([X_test[i]], columns=features.columns)
                            pred_result = model.predict(sample_df)
                            if hasattr(pred_result, 'predicted_value'):
                                test_pred.append(pred_result.predicted_value)
                            else:
                                test_pred.append(pred_result)
                        
                        test_pred = np.array(test_pred)
                        test_mse = mean_squared_error(y_test, test_pred)
                        
                        training_results[name] = {
                            'test_mse': test_mse,
                            'test_mae': mean_absolute_error(y_test, test_pred)
                        }
                        
                        self.performance_history[name].append(test_mse)
                        
                    except Exception as e:
                        self.logger.warning(f"Не удалось оценить модель {name}: {e}")
                        training_results[name] = {'error': str(e)}
                
            except Exception as e:
                self.logger.error(f"Ошибка обучения модели {name}: {e}")
                training_results[name] = {'error': str(e)}
        
        # Обновление весов на основе производительности
        self._update_weights()
        
        self.logger.info("Обучение ансамбля завершено")
        
        return {
            'model_results': training_results,
            'updated_weights': self.model_weights.copy(),
            'ensemble_performance': self._calculate_ensemble_performance(X_test, y_test)
        }
    
    def _prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Подготовка признаков для обучения"""
        
        df = data.copy()
        
        # Базовые признаки
        features = ['open', 'high', 'low', 'close', 'volume']
        
        # Технические индикаторы
        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['returns'].rolling(20).std()
        df['sma_5'] = df['close'].rolling(5).mean()
        df['sma_20'] = df['close'].rolling(20).mean()
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['rsi'] = self._calculate_rsi(df['close'])
        df['macd'] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
        
        # Лаговые признаки
        for lag in [1, 2, 3, 5]:
            df[f'close_lag_{lag}'] = df['close'].shift(lag)
            df[f'volume_lag_{lag}'] = df['volume'].shift(lag)
        
        # Статистические признаки
        df['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        df['price_change'] = df['close'].pct_change()
        df['volume_change'] = df['volume'].pct_change()
        
        # Выбор финальных признаков
        feature_columns = [
            'open', 'high', 'low', 'volume', 'returns', 'volatility',
            'sma_5', 'sma_20', 'ema_12', 'rsi', 'macd',
            'close_lag_1', 'close_lag_2', 'close_lag_3', 'close_lag_5',
            'volume_lag_1', 'volume_lag_2', 'volume_lag_3', 'volume_lag_5',
            'price_position', 'price_change', 'volume_change'
        ]
        
        # Фильтрация существующих колонок
        available_features = [col for col in feature_columns if col in df.columns]
        
        return df[available_features].fillna(method='ffill').fillna(0)
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Расчет RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _update_weights(self):
        """Обновление весов моделей на основе производительности"""
        
        # Расчет весов на основе обратной производительности (меньше MSE = больше вес)
        model_scores = {}
        
        for name, history in self.performance_history.items():
            if history:
                # Используем последние результаты
                recent_performance = np.mean(history[-5:])  # Последние 5 результатов
                model_scores[name] = 1.0 / (recent_performance + 1e-8)  # Обратная величина
        
        if model_scores:
            # Нормализация весов
            total_score = sum(model_scores.values())
            for name in model_scores:
                self.model_weights[name] = model_scores[name] / total_score
            
            self.logger.info("Веса моделей обновлены на основе производительности")
    
    def _calculate_ensemble_performance(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Расчет производительности ансамбля"""
        
        try:
            ensemble_predictions = []
            
            for i in range(len(X_test)):
                sample_df = pd.DataFrame([X_test[i]], columns=self._prepare_features(pd.DataFrame()).columns)
                ensemble_pred = self.predict(sample_df)
                ensemble_predictions.append(ensemble_pred.final_prediction)
            
            ensemble_predictions = np.array(ensemble_predictions)
            
            return {
                'ensemble_mse': mean_squared_error(y_test, ensemble_predictions),
                'ensemble_mae': mean_absolute_error(y_test, ensemble_predictions)
            }
        except Exception as e:
            self.logger.error(f"Ошибка расчета производительности ансамбля: {e}")
            return {'error': str(e)}
    
    def predict(self, data: pd.DataFrame, symbol: str = "UNKNOWN") -> EnsemblePrediction:
        """Ансамблевое прогнозирование"""
        
        # Подготовка признаков
        features = self._prepare_features(data)
        
        if len(features) == 0:
            raise ValueError("Нет данных для прогнозирования")
        
        # Получение последней строки признаков
        last_features = features.iloc[-1:].values
        
        model_predictions = {}
        valid_predictions = {}
        
        # Получение прогнозов от каждой модели
        for name, model in self.models.items():
            try:
                if hasattr(model, 'predict'):
                    if hasattr(model, 'predict_action'):
                        # RL модель
                        pred_result = model.predict_action(data, symbol)
                        prediction = pred_result.predicted_value
                    elif hasattr(model, 'predict') and not hasattr(model, 'fit'):
                        # Deep Learning модель
                        pred_result = model.predict(data, symbol)
                        prediction = pred_result.predicted_value
                    else:
                        # Sklearn модель
                        prediction = model.predict(last_features)[0]
                    
                    model_predictions[name] = prediction
                    valid_predictions[name] = prediction
                    
            except Exception as e:
                self.logger.warning(f"Ошибка прогнозирования модели {name}: {e}")
                model_predictions[name] = None
        
        if not valid_predictions:
            raise ValueError("Ни одна модель не смогла сделать прогноз")
        
        # Взвешенное усреднение прогнозов
        weighted_sum = 0
        total_weight = 0
        
        for name, prediction in valid_predictions.items():
            weight = self.model_weights.get(name, 0)
            weighted_sum += prediction * weight
            total_weight += weight
        
        if total_weight == 0:
            final_prediction = np.mean(list(valid_predictions.values()))
            consensus_strength = 0.5
        else:
            final_prediction = weighted_sum / total_weight
            
            # Расчет силы консенсуса
            predictions_array = np.array(list(valid_predictions.values()))
            consensus_strength = 1.0 - (np.std(predictions_array) / (np.mean(predictions_array) + 1e-8))
            consensus_strength = max(0, min(1, consensus_strength))
        
        # Расчет общей уверенности
        confidence = consensus_strength * 0.8 + 0.2  # Базовая уверенность 20%
        
        return EnsemblePrediction(
            timestamp=datetime.now(),
            symbol=symbol,
            final_prediction=final_prediction,
            confidence=confidence,
            model_predictions=model_predictions,
            model_weights=self.model_weights.copy(),
            consensus_strength=consensus_strength
        )

class NewsAnalyzer:
    """Анализатор новостей с использованием NLP"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.sentiment_analyzer = None
        self.news_sources = [
            'https://feeds.finance.yahoo.com/rss/2.0/headline',
            'https://feeds.bloomberg.com/markets/news.rss',
            'https://www.coindesk.com/arc/outboundfeeds/rss/'
        ]
        
        # Инициализация NLP моделей
        self._initialize_nlp_models()
    
    def _initialize_nlp_models(self):
        """Инициализация NLP моделей"""
        
        try:
            # Sentiment analysis
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="nlptown/bert-base-multilingual-uncased-sentiment",
                return_all_scores=True
            )
            
            self.logger.info("NLP модели инициализированы успешно")
            
        except Exception as e:
            self.logger.warning(f"Ошибка инициализации NLP моделей: {e}")
            # Fallback к простому анализу
            self.sentiment_analyzer = None
    
    async def fetch_news(self, symbols: List[str] = None) -> List[Dict[str, Any]]:
        """Получение новостей"""
        
        all_news = []
        
        for source_url in self.news_sources:
            try:
                # Получение RSS ленты
                feed = feedparser.parse(source_url)
                
                for entry in feed.entries[:10]:  # Последние 10 новостей
                    news_item = {
                        'title': entry.get('title', ''),
                        'description': entry.get('description', ''),
                        'link': entry.get('link', ''),
                        'published': entry.get('published', ''),
                        'source': source_url
                    }
                    
                    # Фильтрация по символам если указаны
                    if symbols:
                        content = f"{news_item['title']} {news_item['description']}".lower()
                        if any(symbol.lower() in content for symbol in symbols):
                            all_news.append(news_item)
                    else:
                        all_news.append(news_item)
                        
            except Exception as e:
                self.logger.error(f"Ошибка получения новостей из {source_url}: {e}")
        
        return all_news
    
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Анализ тональности текста"""
        
        if not text:
            return {'sentiment_score': 0.0, 'confidence': 0.0}
        
        try:
            if self.sentiment_analyzer:
                # Используем BERT модель
                results = self.sentiment_analyzer(text[:512])  # Ограничиваем длину
                
                # Преобразуем результаты в единую оценку
                sentiment_score = 0.0
                confidence = 0.0
                
                for result in results[0]:
                    label = result['label']
                    score = result['score']
                    
                    if 'positive' in label.lower() or '4' in label or '5' in label:
                        sentiment_score += score
                    elif 'negative' in label.lower() or '1' in label or '2' in label:
                        sentiment_score -= score
                    
                    confidence = max(confidence, score)
                
                # Нормализация
                sentiment_score = max(-1, min(1, sentiment_score))
                
                return {
                    'sentiment_score': sentiment_score,
                    'confidence': confidence
                }
            else:
                # Простой анализ по ключевым словам
                return self._simple_sentiment_analysis(text)
                
        except Exception as e:
            self.logger.error(f"Ошибка анализа тональности: {e}")
            return self._simple_sentiment_analysis(text)
    
    def _simple_sentiment_analysis(self, text: str) -> Dict[str, float]:
        """Простой анализ тональности по ключевым словам"""
        
        positive_words = [
            'bull', 'bullish', 'rise', 'rising', 'up', 'gain', 'gains', 'profit',
            'growth', 'increase', 'positive', 'strong', 'surge', 'rally', 'boom'
        ]
        
        negative_words = [
            'bear', 'bearish', 'fall', 'falling', 'down', 'loss', 'losses', 'decline',
            'decrease', 'negative', 'weak', 'crash', 'dump', 'correction', 'drop'
        ]
        
        text_lower = text.lower()
        
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        total_words = len(text.split())
        
        if total_words == 0:
            return {'sentiment_score': 0.0, 'confidence': 0.0}
        
        sentiment_score = (positive_count - negative_count) / max(1, total_words) * 10
        sentiment_score = max(-1, min(1, sentiment_score))
        
        confidence = min(1.0, (positive_count + negative_count) / max(1, total_words) * 5)
        
        return {
            'sentiment_score': sentiment_score,
            'confidence': confidence
        }
    
    def extract_keywords(self, text: str) -> List[str]:
        """Извлечение ключевых слов"""
        
        # Простое извлечение ключевых слов
        crypto_keywords = [
            'bitcoin', 'btc', 'ethereum', 'eth', 'crypto', 'cryptocurrency',
            'blockchain', 'defi', 'nft', 'altcoin', 'trading', 'market'
        ]
        
        financial_keywords = [
            'fed', 'interest', 'rate', 'inflation', 'gdp', 'unemployment',
            'stock', 'market', 'economy', 'recession', 'bull', 'bear'
        ]
        
        all_keywords = crypto_keywords + financial_keywords
        text_lower = text.lower()
        
        found_keywords = [keyword for keyword in all_keywords if keyword in text_lower]
        
        return found_keywords
    
    def calculate_relevance(self, text: str, symbols: List[str]) -> float:
        """Расчет релевантности новости для символов"""
        
        if not symbols:
            return 0.5  # Базовая релевантность
        
        text_lower = text.lower()
        relevance_score = 0.0
        
        for symbol in symbols:
            symbol_lower = symbol.lower()
            
            # Прямое упоминание символа
            if symbol_lower in text_lower:
                relevance_score += 0.5
            
            # Упоминание базовой валюты (например, BTC для BTCUSDT)
            base_currency = symbol_lower.replace('usdt', '').replace('usd', '')
            if base_currency in text_lower:
                relevance_score += 0.3
        
        return min(1.0, relevance_score)
    
    async def analyze_news(self, symbols: List[str] = None) -> List[NewsAnalysisResult]:
        """Полный анализ новостей"""
        
        self.logger.info("Начало анализа новостей")
        
        # Получение новостей
        news_items = await self.fetch_news(symbols)
        
        analysis_results = []
        
        for news_item in news_items:
            try:
                # Объединение заголовка и описания
                full_text = f"{news_item['title']} {news_item['description']}"
                
                # Анализ тональности
                sentiment_result = self.analyze_sentiment(full_text)
                
                # Извлечение ключевых слов
                keywords = self.extract_keywords(full_text)
                
                # Расчет релевантности
                relevance_score = self.calculate_relevance(full_text, symbols or [])
                
                # Определение влияния на рынок
                impact_prediction = "neutral"
                if sentiment_result['sentiment_score'] > 0.3:
                    impact_prediction = "bullish"
                elif sentiment_result['sentiment_score'] < -0.3:
                    impact_prediction = "bearish"
                
                # Создание результата анализа
                analysis_result = NewsAnalysisResult(
                    timestamp=datetime.now(),
                    headline=news_item['title'],
                    content=news_item['description'],
                    sentiment_score=sentiment_result['sentiment_score'],
                    relevance_score=relevance_score,
                    impact_prediction=impact_prediction,
                    confidence=sentiment_result['confidence'],
                    keywords=keywords,
                    source=news_item['source']
                )
                
                analysis_results.append(analysis_result)
                
            except Exception as e:
                self.logger.error(f"Ошибка анализа новости: {e}")
        
        # Сортировка по релевантности и времени
        analysis_results.sort(key=lambda x: (x.relevance_score, x.confidence), reverse=True)
        
        self.logger.info(f"Анализ завершен. Обработано {len(analysis_results)} новостей")
        
        return analysis_results

class AdvancedAISystem:
    """Главный класс продвинутой AI системы"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Инициализация компонентов
        self.deep_learning_predictor = DeepLearningPredictor()
        self.rl_trader = ReinforcementLearningTrader()
        self.ensemble_predictor = EnsemblePredictor()
        self.news_analyzer = NewsAnalyzer()
        
        # Добавление моделей в ансамбль
        self.ensemble_predictor.add_model("deep_learning", self.deep_learning_predictor, 0.3)
        self.ensemble_predictor.add_model("reinforcement_learning", self.rl_trader, 0.2)
        
        self.is_trained = False
    
    async def train_system(self, data: pd.DataFrame, epochs: int = 50, rl_timesteps: int = 50000) -> Dict[str, Any]:
        """Обучение всей AI системы"""
        
        self.logger.info("Начало обучения продвинутой AI системы")
        
        training_results = {}
        
        try:
            # Обучение Deep Learning модели
            self.logger.info("Обучение Deep Learning модели...")
            dl_results = self.deep_learning_predictor.train(data, epochs=epochs)
            training_results['deep_learning'] = dl_results
            
            # Обучение Reinforcement Learning агента
            self.logger.info("Обучение Reinforcement Learning агента...")
            rl_results = self.rl_trader.train(data, total_timesteps=rl_timesteps)
            training_results['reinforcement_learning'] = rl_results
            
            # Обучение ансамбля
            self.logger.info("Обучение ансамбля моделей...")
            ensemble_results = self.ensemble_predictor.train(data)
            training_results['ensemble'] = ensemble_results
            
            self.is_trained = True
            
            self.logger.info("Обучение AI системы завершено успешно")
            
        except Exception as e:
            self.logger.error(f"Ошибка обучения AI системы: {e}")
            training_results['error'] = str(e)
        
        return training_results
    
    async def get_comprehensive_prediction(
        self, 
        data: pd.DataFrame, 
        symbol: str = "UNKNOWN",
        include_news: bool = True
    ) -> Dict[str, Any]:
        """Комплексное прогнозирование с использованием всех AI компонентов"""
        
        if not self.is_trained:
            self.logger.warning("Система не обучена, используются базовые прогнозы")
        
        comprehensive_result = {
            'timestamp': datetime.now(),
            'symbol': symbol,
            'predictions': {},
            'news_analysis': None,
            'final_recommendation': None,
            'confidence': 0.0
        }
        
        try:
            # Получение прогнозов от всех моделей
            if self.is_trained:
                # Deep Learning прогноз
                try:
                    dl_prediction = self.deep_learning_predictor.predict(data, symbol)
                    comprehensive_result['predictions']['deep_learning'] = {
                        'predicted_value': dl_prediction.predicted_value,
                        'confidence': dl_prediction.confidence,
                        'type': dl_prediction.prediction_type
                    }
                except Exception as e:
                    self.logger.error(f"Ошибка DL прогноза: {e}")
                
                # Reinforcement Learning прогноз
                try:
                    rl_prediction = self.rl_trader.predict_action(data, symbol)
                    comprehensive_result['predictions']['reinforcement_learning'] = {
                        'predicted_action': rl_prediction.metadata.get('action_name', 'UNKNOWN'),
                        'confidence': rl_prediction.confidence,
                        'action_value': rl_prediction.predicted_value
                    }
                except Exception as e:
                    self.logger.error(f"Ошибка RL прогноза: {e}")
                
                # Ансамблевый прогноз
                try:
                    ensemble_prediction = self.ensemble_predictor.predict(data, symbol)
                    comprehensive_result['predictions']['ensemble'] = {
                        'final_prediction': ensemble_prediction.final_prediction,
                        'confidence': ensemble_prediction.confidence,
                        'consensus_strength': ensemble_prediction.consensus_strength,
                        'model_predictions': ensemble_prediction.model_predictions,
                        'model_weights': ensemble_prediction.model_weights
                    }
                except Exception as e:
                    self.logger.error(f"Ошибка ансамблевого прогноза: {e}")
            
            # Анализ новостей
            if include_news:
                try:
                    news_results = await self.news_analyzer.analyze_news([symbol])
                    if news_results:
                        # Берем топ-5 наиболее релевантных новостей
                        top_news = news_results[:5]
                        
                        comprehensive_result['news_analysis'] = {
                            'total_news': len(news_results),
                            'average_sentiment': np.mean([news.sentiment_score for news in top_news]),
                            'average_relevance': np.mean([news.relevance_score for news in top_news]),
                            'bullish_news': len([news for news in top_news if news.impact_prediction == 'bullish']),
                            'bearish_news': len([news for news in top_news if news.impact_prediction == 'bearish']),
                            'top_news': [
                                {
                                    'headline': news.headline,
                                    'sentiment_score': news.sentiment_score,
                                    'impact_prediction': news.impact_prediction,
                                    'confidence': news.confidence
                                }
                                for news in top_news
                            ]
                        }
                except Exception as e:
                    self.logger.error(f"Ошибка анализа новостей: {e}")
            
            # Формирование финальной рекомендации
            comprehensive_result['final_recommendation'] = self._generate_final_recommendation(comprehensive_result)
            
        except Exception as e:
            self.logger.error(f"Ошибка комплексного прогнозирования: {e}")
            comprehensive_result['error'] = str(e)
        
        return comprehensive_result
    
    def _generate_final_recommendation(self, comprehensive_result: Dict[str, Any]) -> Dict[str, Any]:
        """Генерация финальной рекомендации на основе всех прогнозов"""
        
        recommendations = []
        total_confidence = 0.0
        confidence_count = 0
        
        # Анализ прогнозов моделей
        predictions = comprehensive_result.get('predictions', {})
        
        # Deep Learning рекомендация
        if 'deep_learning' in predictions:
            dl_pred = predictions['deep_learning']
            current_price = comprehensive_result.get('current_price', 100)  # Заглушка
            
            if dl_pred['predicted_value'] > current_price * 1.02:
                recommendations.append(('BUY', dl_pred['confidence'], 'Deep Learning прогнозирует рост цены'))
            elif dl_pred['predicted_value'] < current_price * 0.98:
                recommendations.append(('SELL', dl_pred['confidence'], 'Deep Learning прогнозирует падение цены'))
            else:
                recommendations.append(('HOLD', dl_pred['confidence'], 'Deep Learning прогнозирует стабильность'))
            
            total_confidence += dl_pred['confidence']
            confidence_count += 1
        
        # Reinforcement Learning рекомендация
        if 'reinforcement_learning' in predictions:
            rl_pred = predictions['reinforcement_learning']
            action = rl_pred['predicted_action']
            recommendations.append((action, rl_pred['confidence'], 'Reinforcement Learning рекомендация'))
            
            total_confidence += rl_pred['confidence']
            confidence_count += 1
        
        # Ансамблевая рекомендация
        if 'ensemble' in predictions:
            ensemble_pred = predictions['ensemble']
            current_price = comprehensive_result.get('current_price', 100)  # Заглушка
            
            if ensemble_pred['final_prediction'] > current_price * 1.015:
                recommendations.append(('BUY', ensemble_pred['confidence'], 'Ансамбль прогнозирует рост'))
            elif ensemble_pred['final_prediction'] < current_price * 0.985:
                recommendations.append(('SELL', ensemble_pred['confidence'], 'Ансамбль прогнозирует падение'))
            else:
                recommendations.append(('HOLD', ensemble_pred['confidence'], 'Ансамбль рекомендует удержание'))
            
            total_confidence += ensemble_pred['confidence']
            confidence_count += 1
        
        # Анализ новостей
        news_sentiment = 0.0
        if comprehensive_result.get('news_analysis'):
            news_analysis = comprehensive_result['news_analysis']
            news_sentiment = news_analysis.get('average_sentiment', 0.0)
            
            if news_sentiment > 0.3:
                recommendations.append(('BUY', 0.6, 'Позитивные новости'))
            elif news_sentiment < -0.3:
                recommendations.append(('SELL', 0.6, 'Негативные новости'))
        
        # Подсчет голосов
        buy_votes = len([r for r in recommendations if r[0] == 'BUY'])
        sell_votes = len([r for r in recommendations if r[0] == 'SELL'])
        hold_votes = len([r for r in recommendations if r[0] == 'HOLD'])
        
        # Взвешенное голосование
        buy_weight = sum([r[1] for r in recommendations if r[0] == 'BUY'])
        sell_weight = sum([r[1] for r in recommendations if r[0] == 'SELL'])
        hold_weight = sum([r[1] for r in recommendations if r[0] == 'HOLD'])
        
        # Определение финальной рекомендации
        if buy_weight > sell_weight and buy_weight > hold_weight:
            final_action = 'BUY'
        elif sell_weight > buy_weight and sell_weight > hold_weight:
            final_action = 'SELL'
        else:
            final_action = 'HOLD'
        
        # Расчет общей уверенности
        if confidence_count > 0:
            average_confidence = total_confidence / confidence_count
        else:
            average_confidence = 0.5
        
        # Корректировка уверенности на основе консенсуса
        total_votes = buy_votes + sell_votes + hold_votes
        if total_votes > 0:
            max_votes = max(buy_votes, sell_votes, hold_votes)
            consensus_strength = max_votes / total_votes
            final_confidence = average_confidence * consensus_strength
        else:
            final_confidence = 0.5
        
        comprehensive_result['confidence'] = final_confidence
        
        return {
            'action': final_action,
            'confidence': final_confidence,
            'reasoning': [r[2] for r in recommendations],
            'vote_breakdown': {
                'buy_votes': buy_votes,
                'sell_votes': sell_votes,
                'hold_votes': hold_votes
            },
            'weight_breakdown': {
                'buy_weight': buy_weight,
                'sell_weight': sell_weight,
                'hold_weight': hold_weight
            },
            'news_sentiment': news_sentiment
        }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Получение статуса AI системы"""
        
        return {
            'timestamp': datetime.now(),
            'is_trained': self.is_trained,
            'components': {
                'deep_learning': {
                    'initialized': self.deep_learning_predictor is not None,
                    'trained': self.deep_learning_predictor.is_trained if self.deep_learning_predictor else False
                },
                'reinforcement_learning': {
                    'initialized': self.rl_trader is not None,
                    'trained': self.rl_trader.is_trained if self.rl_trader else False
                },
                'ensemble': {
                    'initialized': self.ensemble_predictor is not None,
                    'models_count': len(self.ensemble_predictor.models) if self.ensemble_predictor else 0
                },
                'news_analyzer': {
                    'initialized': self.news_analyzer is not None,
                    'nlp_models_loaded': self.news_analyzer.sentiment_analyzer is not None if self.news_analyzer else False
                }
            }
        }

# Функция для тестирования системы
async def test_advanced_ai_system():
    """Тестирование продвинутой AI системы"""
    
    print("=== Тестирование продвинутой AI системы ===")
    
    # Создание тестовых данных
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='1H')
    np.random.seed(42)
    
    # Генерация реалистичных ценовых данных
    price = 100
    prices = [price]
    
    for i in range(1, len(dates)):
        # Случайное изменение цены с трендом
        change = np.random.normal(0, 0.02) + 0.0001  # Небольшой восходящий тренд
        price = price * (1 + change)
        prices.append(price)
    
    test_data = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'close': prices,
        'volume': [np.random.randint(1000, 10000) for _ in prices]
    })
    
    # Инициализация системы
    ai_system = AdvancedAISystem()
    
    # Проверка статуса
    status = ai_system.get_system_status()
    print(f"Статус системы: {status}")
    
    try:
        # Обучение системы (с уменьшенными параметрами для тестирования)
        print("\nНачало обучения AI системы...")
        training_results = await ai_system.train_system(
            test_data, 
            epochs=5,  # Уменьшено для тестирования
            rl_timesteps=1000  # Уменьшено для тестирования
        )
        
        print("Результаты обучения:")
        for component, results in training_results.items():
            if isinstance(results, dict) and 'error' not in results:
                print(f"  {component}: Успешно")
            else:
                print(f"  {component}: {results}")
        
        # Тестирование прогнозирования
        print("\nТестирование прогнозирования...")
        prediction_result = await ai_system.get_comprehensive_prediction(
            test_data.tail(100),  # Последние 100 точек
            symbol="BTCUSDT",
            include_news=False  # Отключаем новости для тестирования
        )
        
        print("Результат прогнозирования:")
        print(f"  Символ: {prediction_result['symbol']}")
        print(f"  Время: {prediction_result['timestamp']}")
        print(f"  Финальная рекомендация: {prediction_result.get('final_recommendation', {})}")
        print(f"  Общая уверенность: {prediction_result['confidence']:.3f}")
        
        # Проверка компонентов
        predictions = prediction_result.get('predictions', {})
        for component, pred in predictions.items():
            print(f"  {component}: {pred}")
        
        print("\n=== Тестирование завершено успешно ===")
        
        return True
        
    except Exception as e:
        print(f"Ошибка тестирования: {e}")
        return False

if __name__ == "__main__":
    # Настройка логирования
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Запуск тестирования
    import asyncio
    asyncio.run(test_advanced_ai_system())