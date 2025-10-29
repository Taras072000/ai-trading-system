"""
Enterprise Predictive Market Analytics - Предиктивная аналитика рынка
Обеспечивает прогнозирование рыночных движений с использованием ML/AI
"""

import asyncio
import json
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import logging
import redis.asyncio as redis
from collections import deque, defaultdict
import warnings
warnings.filterwarnings('ignore')

# ML/AI библиотеки
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

# Временные ряды
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import pmdarima as pm

# Технический анализ
import talib
import yfinance as yf

# Обработка данных
from scipy import stats
from scipy.signal import savgol_filter
import networkx as nx

# Метрики
from prometheus_client import Counter, Histogram, Gauge

class PredictionType(Enum):
    """Типы прогнозов"""
    PRICE = "price"
    DIRECTION = "direction"
    VOLATILITY = "volatility"
    VOLUME = "volume"
    TREND = "trend"
    SUPPORT_RESISTANCE = "support_resistance"
    BREAKOUT = "breakout"
    REVERSAL = "reversal"

class TimeHorizon(Enum):
    """Временные горизонты прогнозирования"""
    ULTRA_SHORT = "1m"  # 1 минута
    SHORT = "5m"        # 5 минут
    MEDIUM = "1h"       # 1 час
    LONG = "1d"         # 1 день
    ULTRA_LONG = "1w"   # 1 неделя

class ModelType(Enum):
    """Типы моделей"""
    LSTM = "lstm"
    GRU = "gru"
    TRANSFORMER = "transformer"
    CNN = "cnn"
    ENSEMBLE = "ensemble"
    ARIMA = "arima"
    PROPHET = "prophet"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"

@dataclass
class MarketPrediction:
    """Рыночный прогноз"""
    symbol: str
    prediction_type: PredictionType
    time_horizon: TimeHorizon
    predicted_value: float
    confidence: float
    probability: Optional[float] = None
    upper_bound: Optional[float] = None
    lower_bound: Optional[float] = None
    features_importance: Optional[Dict[str, float]] = None
    model_used: str = ""
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

@dataclass
class MarketSignal:
    """Рыночный сигнал"""
    symbol: str
    signal_type: str
    strength: float  # 0-1
    direction: str   # "bullish", "bearish", "neutral"
    confidence: float
    reasoning: str
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

@dataclass
class ModelPerformance:
    """Производительность модели"""
    model_name: str
    prediction_type: PredictionType
    time_horizon: TimeHorizon
    accuracy: float = 0.0
    mse: float = 0.0
    mae: float = 0.0
    r2_score: float = 0.0
    directional_accuracy: float = 0.0
    sharpe_ratio: float = 0.0
    total_predictions: int = 0
    last_updated: datetime = None
    
    def __post_init__(self):
        if self.last_updated is None:
            self.last_updated = datetime.now()

# Метрики
PREDICTIONS_TOTAL = Counter('predictions_total', 'Total predictions made', ['model', 'symbol', 'type'])
PREDICTION_ACCURACY = Gauge('prediction_accuracy', 'Prediction accuracy', ['model', 'symbol', 'type'])
PREDICTION_CONFIDENCE = Histogram('prediction_confidence', 'Prediction confidence', ['model', 'symbol'])
MODEL_TRAINING_TIME = Histogram('model_training_time_seconds', 'Model training time', ['model'])

class LSTMPredictor(nn.Module):
    """LSTM модель для прогнозирования"""
    
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2, 
                 output_size: int = 1, dropout: float = 0.2):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Используем последний выход
        last_output = lstm_out[:, -1, :]
        
        # Dropout и полносвязный слой
        output = self.dropout(last_output)
        output = self.fc(output)
        
        return output

class TransformerPredictor(nn.Module):
    """Transformer модель для прогнозирования"""
    
    def __init__(self, input_size: int, d_model: int = 128, nhead: int = 8, 
                 num_layers: int = 6, output_size: int = 1, dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.input_projection = nn.Linear(input_size, d_model)
        
        # Позиционное кодирование
        self.pos_encoding = PositionalEncoding(d_model, dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Выходной слой
        self.fc = nn.Linear(d_model, output_size)
        
    def forward(self, x):
        # Проекция входных данных
        x = self.input_projection(x)
        
        # Позиционное кодирование
        x = self.pos_encoding(x)
        
        # Transformer
        transformer_out = self.transformer(x)
        
        # Используем последний выход
        last_output = transformer_out[:, -1, :]
        
        # Финальная проекция
        output = self.fc(last_output)
        
        return output

class PositionalEncoding(nn.Module):
    """Позиционное кодирование для Transformer"""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:x.size(0), :].transpose(0, 1)
        return self.dropout(x)

class CNNPredictor(nn.Module):
    """CNN модель для прогнозирования временных рядов"""
    
    def __init__(self, input_channels: int, sequence_length: int, output_size: int = 1):
        super().__init__()
        
        # Сверточные слои
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        
        # Пулинг
        self.pool = nn.MaxPool1d(2)
        
        # Dropout
        self.dropout = nn.Dropout(0.2)
        
        # Полносвязные слои
        conv_output_size = 256 * (sequence_length // 8)  # После 3 пулингов
        self.fc1 = nn.Linear(conv_output_size, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, output_size)
        
    def forward(self, x):
        # Сверточные слои
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        
        # Выравнивание
        x = x.view(x.size(0), -1)
        
        # Полносвязные слои
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        
        x = self.fc3(x)
        
        return x

class FeatureEngineer:
    """Инженер признаков для рыночных данных"""
    
    def __init__(self):
        self.scalers = {}
        
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Создание признаков из рыночных данных"""
        df = df.copy()
        
        # Базовые признаки
        df = self._add_price_features(df)
        df = self._add_volume_features(df)
        df = self._add_technical_indicators(df)
        df = self._add_statistical_features(df)
        df = self._add_time_features(df)
        df = self._add_market_microstructure_features(df)
        
        return df
        
    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Добавление ценовых признаков"""
        # Доходности
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Ценовые изменения
        df['price_change'] = df['close'] - df['open']
        df['price_change_pct'] = df['price_change'] / df['open']
        
        # Диапазоны
        df['high_low_ratio'] = df['high'] / df['low']
        df['close_open_ratio'] = df['close'] / df['open']
        
        # Тени свечей
        df['upper_shadow'] = df['high'] - np.maximum(df['open'], df['close'])
        df['lower_shadow'] = np.minimum(df['open'], df['close']) - df['low']
        df['body_size'] = np.abs(df['close'] - df['open'])
        
        # Нормализация теней
        df['upper_shadow_ratio'] = df['upper_shadow'] / (df['high'] - df['low'])
        df['lower_shadow_ratio'] = df['lower_shadow'] / (df['high'] - df['low'])
        df['body_ratio'] = df['body_size'] / (df['high'] - df['low'])
        
        return df
        
    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Добавление объемных признаков"""
        # Объемные индикаторы
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # VWAP
        df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
        df['price_vwap_ratio'] = df['close'] / df['vwap']
        
        # Volume Price Trend
        df['vpt'] = (df['volume'] * df['returns']).cumsum()
        
        # On Balance Volume
        df['obv'] = (df['volume'] * np.sign(df['returns'])).cumsum()
        
        return df
        
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Добавление технических индикаторов"""
        # Скользящие средние
        for period in [5, 10, 20, 50, 100, 200]:
            df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
            df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
            
        # RSI
        df['rsi'] = self._calculate_rsi(df['close'])
        
        # MACD
        ema_12 = df['close'].ewm(span=12).mean()
        ema_26 = df['close'].ewm(span=26).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_width'] = df['bb_upper'] - df['bb_lower']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Stochastic
        df['stoch_k'] = self._calculate_stochastic(df)
        df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
        
        # Williams %R
        df['williams_r'] = self._calculate_williams_r(df)
        
        # ATR
        df['atr'] = self._calculate_atr(df)
        
        # CCI
        df['cci'] = self._calculate_cci(df)
        
        return df
        
    def _add_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Добавление статистических признаков"""
        # Волатильность
        for period in [5, 10, 20, 50]:
            df[f'volatility_{period}'] = df['returns'].rolling(window=period).std()
            df[f'volatility_ratio_{period}'] = df[f'volatility_{period}'] / df[f'volatility_{period}'].rolling(window=50).mean()
            
        # Скользящие статистики
        for period in [10, 20, 50]:
            df[f'skewness_{period}'] = df['returns'].rolling(window=period).skew()
            df[f'kurtosis_{period}'] = df['returns'].rolling(window=period).kurt()
            
        # Z-score
        df['z_score'] = (df['close'] - df['close'].rolling(window=20).mean()) / df['close'].rolling(window=20).std()
        
        # Momentum
        for period in [5, 10, 20]:
            df[f'momentum_{period}'] = df['close'] / df['close'].shift(period) - 1
            
        # Rate of Change
        for period in [5, 10, 20]:
            df[f'roc_{period}'] = df['close'].pct_change(periods=period)
            
        return df
        
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Добавление временных признаков"""
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Циклические признаки
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['day_of_month'] = df['timestamp'].dt.day
            df['month'] = df['timestamp'].dt.month
            
            # Синусоидальные кодировки для цикличности
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
            
        return df
        
    def _add_market_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Добавление признаков микроструктуры рынка"""
        # Bid-Ask Spread (если доступен)
        if 'bid' in df.columns and 'ask' in df.columns:
            df['spread'] = df['ask'] - df['bid']
            df['spread_ratio'] = df['spread'] / df['close']
            df['mid_price'] = (df['bid'] + df['ask']) / 2
            df['price_mid_ratio'] = df['close'] / df['mid_price']
            
        # Ликвидность (приблизительно через объем)
        df['liquidity_proxy'] = df['volume'] / df['volatility_20']
        
        # Давление покупки/продажи
        df['buy_pressure'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        df['sell_pressure'] = (df['high'] - df['close']) / (df['high'] - df['low'])
        
        return df
        
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Расчет RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
        
    def _calculate_stochastic(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Расчет Stochastic %K"""
        lowest_low = df['low'].rolling(window=period).min()
        highest_high = df['high'].rolling(window=period).max()
        stoch_k = 100 * (df['close'] - lowest_low) / (highest_high - lowest_low)
        return stoch_k
        
    def _calculate_williams_r(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Расчет Williams %R"""
        highest_high = df['high'].rolling(window=period).max()
        lowest_low = df['low'].rolling(window=period).min()
        williams_r = -100 * (highest_high - df['close']) / (highest_high - lowest_low)
        return williams_r
        
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Расчет Average True Range"""
        high_low = df['high'] - df['low']
        high_close_prev = np.abs(df['high'] - df['close'].shift())
        low_close_prev = np.abs(df['low'] - df['close'].shift())
        
        true_range = np.maximum(high_low, np.maximum(high_close_prev, low_close_prev))
        atr = true_range.rolling(window=period).mean()
        return atr
        
    def _calculate_cci(self, df: pd.DataFrame, period: int = 20) -> pd.Series:
        """Расчет Commodity Channel Index"""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        sma_tp = typical_price.rolling(window=period).mean()
        mad = typical_price.rolling(window=period).apply(lambda x: np.mean(np.abs(x - x.mean())))
        cci = (typical_price - sma_tp) / (0.015 * mad)
        return cci

class MarketRegimeDetector:
    """Детектор рыночных режимов"""
    
    def __init__(self):
        self.regimes = ['trending_up', 'trending_down', 'sideways', 'volatile', 'low_volatility']
        
    def detect_regime(self, df: pd.DataFrame, window: int = 50) -> str:
        """Определение текущего рыночного режима"""
        recent_data = df.tail(window)
        
        # Расчет метрик
        returns = recent_data['close'].pct_change().dropna()
        volatility = returns.std()
        trend = (recent_data['close'].iloc[-1] - recent_data['close'].iloc[0]) / recent_data['close'].iloc[0]
        
        # Определение режима
        if abs(trend) > 0.1:  # Сильный тренд
            return 'trending_up' if trend > 0 else 'trending_down'
        elif volatility > 0.03:  # Высокая волатильность
            return 'volatile'
        elif volatility < 0.01:  # Низкая волатильность
            return 'low_volatility'
        else:
            return 'sideways'

class EnsemblePredictor:
    """Ансамблевый предиктор"""
    
    def __init__(self):
        self.models = {}
        self.weights = {}
        self.feature_engineer = FeatureEngineer()
        
    def add_model(self, name: str, model: Any, weight: float = 1.0):
        """Добавление модели в ансамбль"""
        self.models[name] = model
        self.weights[name] = weight
        
    def predict(self, X: np.ndarray) -> Tuple[float, float]:
        """Ансамблевое предсказание"""
        predictions = []
        confidences = []
        
        for name, model in self.models.items():
            try:
                if hasattr(model, 'predict_proba'):
                    pred = model.predict(X.reshape(1, -1))[0]
                    conf = np.max(model.predict_proba(X.reshape(1, -1)))
                else:
                    pred = model.predict(X.reshape(1, -1))[0]
                    conf = 0.8  # Базовая уверенность
                    
                predictions.append(pred * self.weights[name])
                confidences.append(conf * self.weights[name])
                
            except Exception as e:
                continue
                
        if not predictions:
            return 0.0, 0.0
            
        # Взвешенное среднее
        total_weight = sum(self.weights.values())
        final_prediction = sum(predictions) / total_weight
        final_confidence = sum(confidences) / total_weight
        
        return final_prediction, final_confidence

class EnterprisePredictiveAnalytics:
    """Enterprise система предиктивной аналитики рынка"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.redis_client = None
        self.logger = self._setup_logging()
        
        # Модели
        self.models: Dict[str, Any] = {}
        self.model_performances: Dict[str, ModelPerformance] = {}
        
        # Инженер признаков
        self.feature_engineer = FeatureEngineer()
        
        # Детектор режимов
        self.regime_detector = MarketRegimeDetector()
        
        # Кеш данных
        self.data_cache: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        
        # Предсказания
        self.predictions_cache: Dict[str, List[MarketPrediction]] = defaultdict(list)
        
        # Сигналы
        self.signals_cache: Dict[str, List[MarketSignal]] = defaultdict(list)
        
    def _setup_logging(self) -> logging.Logger:
        """Настройка логирования"""
        logger = logging.getLogger('enterprise_predictive_analytics')
        logger.setLevel(logging.INFO)
        
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
        
    async def start(self):
        """Запуск системы предиктивной аналитики"""
        # Подключение к Redis
        self.redis_client = redis.Redis(
            host=self.config.get('redis_host', 'localhost'),
            port=self.config.get('redis_port', 6379),
            decode_responses=True
        )
        
        # Инициализация моделей
        await self._initialize_models()
        
        # Запуск фоновых задач
        asyncio.create_task(self._data_processor())
        asyncio.create_task(self._prediction_engine())
        asyncio.create_task(self._signal_generator())
        asyncio.create_task(self._model_trainer())
        asyncio.create_task(self._performance_evaluator())
        
        self.logger.info("Enterprise Predictive Analytics started")
        
    async def stop(self):
        """Остановка системы"""
        # Сохранение моделей
        await self._save_models()
        
        if self.redis_client:
            await self.redis_client.close()
            
    async def _initialize_models(self):
        """Инициализация моделей"""
        symbols = self.config.get('symbols', ['BTCUSDT', 'ETHUSDT'])
        
        for symbol in symbols:
            # Получение исторических данных
            historical_data = await self._get_historical_data(symbol)
            
            if len(historical_data) < 100:
                continue
                
            # Создание признаков
            features_df = self.feature_engineer.create_features(historical_data)
            features_df = features_df.dropna()
            
            if len(features_df) < 50:
                continue
                
            # Инициализация моделей для каждого типа предсказания
            for pred_type in [PredictionType.PRICE, PredictionType.DIRECTION, PredictionType.VOLATILITY]:
                for horizon in [TimeHorizon.SHORT, TimeHorizon.MEDIUM, TimeHorizon.LONG]:
                    model_key = f"{symbol}_{pred_type.value}_{horizon.value}"
                    
                    # Создание и обучение моделей
                    await self._create_and_train_models(model_key, features_df, pred_type, horizon)
                    
        self.logger.info(f"Initialized {len(self.models)} models")
        
    async def _create_and_train_models(self, model_key: str, data: pd.DataFrame, 
                                     pred_type: PredictionType, horizon: TimeHorizon):
        """Создание и обучение моделей"""
        try:
            # Подготовка данных
            X, y = self._prepare_training_data(data, pred_type, horizon)
            
            if len(X) < 50:
                return
                
            # Разделение данных
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, shuffle=False
            )
            
            # Создание ансамбля моделей
            ensemble = EnsemblePredictor()
            
            # XGBoost
            xgb_model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
            xgb_model.fit(X_train, y_train)
            ensemble.add_model('xgboost', xgb_model, weight=0.3)
            
            # LightGBM
            lgb_model = lgb.LGBMRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
            lgb_model.fit(X_train, y_train)
            ensemble.add_model('lightgbm', lgb_model, weight=0.3)
            
            # Random Forest
            rf_model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            rf_model.fit(X_train, y_train)
            ensemble.add_model('random_forest', rf_model, weight=0.2)
            
            # Neural Network (если данных достаточно)
            if len(X_train) > 200:
                nn_model = self._create_neural_network(X_train.shape[1])
                nn_model = self._train_neural_network(nn_model, X_train, y_train)
                ensemble.add_model('neural_network', nn_model, weight=0.2)
                
            self.models[model_key] = ensemble
            
            # Оценка производительности
            y_pred = []
            for i in range(len(X_test)):
                pred, _ = ensemble.predict(X_test[i])
                y_pred.append(pred)
                
            y_pred = np.array(y_pred)
            
            # Расчет метрик
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Направленная точность (для ценовых предсказаний)
            if pred_type == PredictionType.DIRECTION:
                directional_accuracy = np.mean(np.sign(y_test) == np.sign(y_pred))
            else:
                directional_accuracy = 0.0
                
            # Сохранение производительности
            self.model_performances[model_key] = ModelPerformance(
                model_name=model_key,
                prediction_type=pred_type,
                time_horizon=horizon,
                mse=mse,
                mae=mae,
                r2_score=r2,
                directional_accuracy=directional_accuracy
            )
            
            self.logger.info(f"Trained model {model_key}: MSE={mse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")
            
        except Exception as e:
            self.logger.error(f"Error training model {model_key}: {e}")
            
    def _prepare_training_data(self, data: pd.DataFrame, pred_type: PredictionType, 
                             horizon: TimeHorizon) -> Tuple[np.ndarray, np.ndarray]:
        """Подготовка данных для обучения"""
        # Определение горизонта в периодах
        horizon_periods = {
            TimeHorizon.ULTRA_SHORT: 1,
            TimeHorizon.SHORT: 5,
            TimeHorizon.MEDIUM: 60,
            TimeHorizon.LONG: 1440,
            TimeHorizon.ULTRA_LONG: 10080
        }
        
        periods = horizon_periods[horizon]
        
        # Выбор целевой переменной
        if pred_type == PredictionType.PRICE:
            target = data['close'].shift(-periods)
        elif pred_type == PredictionType.DIRECTION:
            target = np.sign(data['close'].shift(-periods) - data['close'])
        elif pred_type == PredictionType.VOLATILITY:
            target = data['returns'].rolling(window=periods).std().shift(-periods)
        else:
            target = data['close'].pct_change(periods).shift(-periods)
            
        # Выбор признаков
        feature_columns = [col for col in data.columns if col not in ['timestamp', 'close']]
        features = data[feature_columns]
        
        # Удаление NaN
        valid_indices = ~(target.isna() | features.isna().any(axis=1))
        X = features[valid_indices].values
        y = target[valid_indices].values
        
        # Нормализация признаков
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        return X, y
        
    def _create_neural_network(self, input_size: int) -> keras.Model:
        """Создание нейронной сети"""
        model = keras.Sequential([
            layers.Dense(256, activation='relu', input_shape=(input_size,)),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.1),
            layers.Dense(1)
        ])
        
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        return model
        
    def _train_neural_network(self, model: keras.Model, X_train: np.ndarray, 
                            y_train: np.ndarray) -> keras.Model:
        """Обучение нейронной сети"""
        early_stopping = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=0
        )
        
        return model
        
    async def _get_historical_data(self, symbol: str, days: int = 365) -> pd.DataFrame:
        """Получение исторических данных"""
        # Заглушка для исторических данных
        # В реальной реализации здесь должно быть подключение к API биржи
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        dates = pd.date_range(start=start_date, end=end_date, freq='1H')
        
        # Генерация синтетических данных
        np.random.seed(hash(symbol) % 2**32)
        base_price = 50000 if symbol == 'BTCUSDT' else 3000 if symbol == 'ETHUSDT' else 1.0
        
        # Случайное блуждание с трендом
        returns = np.random.normal(0.0001, 0.02, len(dates))
        prices = base_price * np.exp(np.cumsum(returns))
        
        # Объемы
        volumes = np.random.lognormal(mean=5, sigma=1, size=len(dates))
        
        df = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': prices * (1 + np.random.uniform(0, 0.02, len(dates))),
            'low': prices * (1 - np.random.uniform(0, 0.02, len(dates))),
            'close': prices,
            'volume': volumes
        })
        
        return df
        
    async def _data_processor(self):
        """Обработчик данных"""
        while True:
            try:
                symbols = self.config.get('symbols', ['BTCUSDT', 'ETHUSDT'])
                
                for symbol in symbols:
                    # Получение новых данных
                    new_data = await self._get_latest_market_data(symbol)
                    
                    if new_data is not None:
                        # Добавление в кеш
                        self.data_cache[symbol].append(new_data)
                        
                        # Обновление признаков
                        if len(self.data_cache[symbol]) > 100:
                            df = pd.DataFrame(list(self.data_cache[symbol]))
                            features_df = self.feature_engineer.create_features(df)
                            
                            # Сохранение в Redis
                            await self.redis_client.set(
                                f"features:{symbol}",
                                features_df.tail(1).to_json(),
                                ex=3600
                            )
                            
                await asyncio.sleep(60)  # Обновление каждую минуту
                
            except Exception as e:
                self.logger.error(f"Data processor error: {e}")
                await asyncio.sleep(300)
                
    async def _get_latest_market_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Получение последних рыночных данных"""
        # Симуляция получения данных
        # В реальной реализации здесь должен быть API вызов
        
        if symbol in self.data_cache and len(self.data_cache[symbol]) > 0:
            last_data = list(self.data_cache[symbol])[-1]
            base_price = last_data['close']
        else:
            base_price = 50000 if symbol == 'BTCUSDT' else 3000
            
        price_change = np.random.normal(0, base_price * 0.001)
        new_price = base_price + price_change
        
        return {
            'timestamp': datetime.now(),
            'open': base_price,
            'high': max(base_price, new_price) * (1 + np.random.uniform(0, 0.001)),
            'low': min(base_price, new_price) * (1 - np.random.uniform(0, 0.001)),
            'close': new_price,
            'volume': np.random.lognormal(mean=5, sigma=0.5)
        }
        
    async def _prediction_engine(self):
        """Движок предсказаний"""
        while True:
            try:
                for model_key, model in self.models.items():
                    symbol = model_key.split('_')[0]
                    pred_type = PredictionType(model_key.split('_')[1])
                    horizon = TimeHorizon(model_key.split('_')[2])
                    
                    # Получение последних признаков
                    features_json = await self.redis_client.get(f"features:{symbol}")
                    
                    if features_json:
                        features_df = pd.read_json(features_json)
                        
                        if not features_df.empty:
                            # Подготовка данных для предсказания
                            feature_columns = [col for col in features_df.columns 
                                             if col not in ['timestamp', 'close']]
                            X = features_df[feature_columns].values[-1]
                            
                            # Предсказание
                            prediction, confidence = model.predict(X)
                            
                            # Создание объекта предсказания
                            market_prediction = MarketPrediction(
                                symbol=symbol,
                                prediction_type=pred_type,
                                time_horizon=horizon,
                                predicted_value=prediction,
                                confidence=confidence,
                                model_used=model_key
                            )
                            
                            # Сохранение предсказания
                            self.predictions_cache[symbol].append(market_prediction)
                            
                            # Ограничение размера кеша
                            if len(self.predictions_cache[symbol]) > 1000:
                                self.predictions_cache[symbol] = self.predictions_cache[symbol][-1000:]
                                
                            # Сохранение в Redis
                            await self.redis_client.lpush(
                                f"predictions:{symbol}",
                                json.dumps(asdict(market_prediction), default=str)
                            )
                            await self.redis_client.ltrim(f"predictions:{symbol}", 0, 999)
                            
                            # Метрики
                            PREDICTIONS_TOTAL.labels(
                                model=model_key,
                                symbol=symbol,
                                type=pred_type.value
                            ).inc()
                            
                            PREDICTION_CONFIDENCE.labels(
                                model=model_key,
                                symbol=symbol
                            ).observe(confidence)
                            
                await asyncio.sleep(300)  # Предсказания каждые 5 минут
                
            except Exception as e:
                self.logger.error(f"Prediction engine error: {e}")
                await asyncio.sleep(600)
                
    async def _signal_generator(self):
        """Генератор торговых сигналов"""
        while True:
            try:
                symbols = self.config.get('symbols', ['BTCUSDT', 'ETHUSDT'])
                
                for symbol in symbols:
                    # Получение последних предсказаний
                    predictions = self.predictions_cache.get(symbol, [])
                    
                    if len(predictions) < 3:
                        continue
                        
                    # Анализ предсказаний для генерации сигналов
                    signals = await self._analyze_predictions_for_signals(symbol, predictions)
                    
                    for signal in signals:
                        self.signals_cache[symbol].append(signal)
                        
                        # Сохранение в Redis
                        await self.redis_client.lpush(
                            f"signals:{symbol}",
                            json.dumps(asdict(signal), default=str)
                        )
                        await self.redis_client.ltrim(f"signals:{symbol}", 0, 499)
                        
                    # Ограничение размера кеша
                    if len(self.signals_cache[symbol]) > 500:
                        self.signals_cache[symbol] = self.signals_cache[symbol][-500:]
                        
                await asyncio.sleep(180)  # Сигналы каждые 3 минуты
                
            except Exception as e:
                self.logger.error(f"Signal generator error: {e}")
                await asyncio.sleep(300)
                
    async def _analyze_predictions_for_signals(self, symbol: str, 
                                             predictions: List[MarketPrediction]) -> List[MarketSignal]:
        """Анализ предсказаний для генерации сигналов"""
        signals = []
        
        # Получение последних предсказаний по типам
        price_predictions = [p for p in predictions[-10:] if p.prediction_type == PredictionType.PRICE]
        direction_predictions = [p for p in predictions[-10:] if p.prediction_type == PredictionType.DIRECTION]
        volatility_predictions = [p for p in predictions[-10:] if p.prediction_type == PredictionType.VOLATILITY]
        
        # Сигнал направления цены
        if direction_predictions:
            latest_direction = direction_predictions[-1]
            
            if latest_direction.confidence > 0.7:
                direction = "bullish" if latest_direction.predicted_value > 0 else "bearish"
                
                signal = MarketSignal(
                    symbol=symbol,
                    signal_type="direction",
                    strength=latest_direction.confidence,
                    direction=direction,
                    confidence=latest_direction.confidence,
                    reasoning=f"Direction prediction: {latest_direction.predicted_value:.4f}"
                )
                signals.append(signal)
                
        # Сигнал волатильности
        if volatility_predictions:
            latest_volatility = volatility_predictions[-1]
            
            if latest_volatility.predicted_value > 0.03:  # Высокая волатильность
                signal = MarketSignal(
                    symbol=symbol,
                    signal_type="high_volatility",
                    strength=min(latest_volatility.predicted_value * 20, 1.0),
                    direction="neutral",
                    confidence=latest_volatility.confidence,
                    reasoning=f"High volatility expected: {latest_volatility.predicted_value:.4f}"
                )
                signals.append(signal)
                
        # Консенсус сигнал
        if len(price_predictions) >= 3:
            recent_prices = [p.predicted_value for p in price_predictions[-3:]]
            price_trend = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
            
            if abs(price_trend) > 0.02:  # Значительное изменение цены
                avg_confidence = np.mean([p.confidence for p in price_predictions[-3:]])
                
                signal = MarketSignal(
                    symbol=symbol,
                    signal_type="price_trend",
                    strength=min(abs(price_trend) * 50, 1.0),
                    direction="bullish" if price_trend > 0 else "bearish",
                    confidence=avg_confidence,
                    reasoning=f"Price trend: {price_trend:.4f}"
                )
                signals.append(signal)
                
        return signals
        
    async def _model_trainer(self):
        """Тренер моделей"""
        while True:
            try:
                # Периодическое переобучение моделей
                for model_key in list(self.models.keys()):
                    symbol = model_key.split('_')[0]
                    pred_type = PredictionType(model_key.split('_')[1])
                    horizon = TimeHorizon(model_key.split('_')[2])
                    
                    # Получение новых данных
                    if len(self.data_cache[symbol]) > 500:
                        df = pd.DataFrame(list(self.data_cache[symbol]))
                        features_df = self.feature_engineer.create_features(df)
                        features_df = features_df.dropna()
                        
                        if len(features_df) > 100:
                            self.logger.info(f"Retraining model {model_key}")
                            await self._create_and_train_models(model_key, features_df, pred_type, horizon)
                            
                await asyncio.sleep(3600 * 6)  # Переобучение каждые 6 часов
                
            except Exception as e:
                self.logger.error(f"Model trainer error: {e}")
                await asyncio.sleep(3600)
                
    async def _performance_evaluator(self):
        """Оценщик производительности"""
        while True:
            try:
                for model_key, performance in self.model_performances.items():
                    # Обновление метрик
                    PREDICTION_ACCURACY.labels(
                        model=model_key,
                        symbol=model_key.split('_')[0],
                        type=performance.prediction_type.value
                    ).set(performance.directional_accuracy)
                    
                await asyncio.sleep(1800)  # Обновление каждые 30 минут
                
            except Exception as e:
                self.logger.error(f"Performance evaluator error: {e}")
                await asyncio.sleep(3600)
                
    async def _save_models(self):
        """Сохранение моделей"""
        for model_key, model in self.models.items():
            try:
                # Сохранение ансамблевых моделей
                model_path = f"models/predictive_{model_key}_{int(time.time())}"
                
                # Здесь должна быть логика сохранения моделей
                # В зависимости от типа модели
                
                self.logger.info(f"Saved model {model_key}")
                
            except Exception as e:
                self.logger.error(f"Error saving model {model_key}: {e}")
                
    async def get_predictions(self, symbol: str, prediction_type: Optional[PredictionType] = None,
                            time_horizon: Optional[TimeHorizon] = None) -> List[MarketPrediction]:
        """Получение предсказаний"""
        predictions = self.predictions_cache.get(symbol, [])
        
        if prediction_type:
            predictions = [p for p in predictions if p.prediction_type == prediction_type]
            
        if time_horizon:
            predictions = [p for p in predictions if p.time_horizon == time_horizon]
            
        return predictions[-10:]  # Последние 10 предсказаний
        
    async def get_signals(self, symbol: str, signal_type: Optional[str] = None) -> List[MarketSignal]:
        """Получение сигналов"""
        signals = self.signals_cache.get(symbol, [])
        
        if signal_type:
            signals = [s for s in signals if s.signal_type == signal_type]
            
        return signals[-20:]  # Последние 20 сигналов
        
    async def get_model_performance(self, model_key: str) -> Optional[ModelPerformance]:
        """Получение производительности модели"""
        return self.model_performances.get(model_key)

async def main():
    """Основная функция запуска"""
    config = {
        'redis_host': 'localhost',
        'redis_port': 6379,
        'symbols': ['BTCUSDT', 'ETHUSDT', 'ADAUSDT'],
        'prediction_interval': 300,  # 5 минут
        'signal_interval': 180,     # 3 минуты
        'retrain_interval': 21600   # 6 часов
    }
    
    analytics_system = EnterprisePredictiveAnalytics(config)
    await analytics_system.start()
    
    print("Enterprise Predictive Market Analytics started")
    print(f"Monitoring symbols: {config['symbols']}")
    print(f"Models initialized: {len(analytics_system.models)}")
    
    try:
        # Демонстрация работы
        await asyncio.sleep(10)
        
        for symbol in config['symbols']:
            predictions = await analytics_system.get_predictions(symbol)
            signals = await analytics_system.get_signals(symbol)
            
            print(f"\n{symbol}:")
            print(f"  Predictions: {len(predictions)}")
            print(f"  Signals: {len(signals)}")
            
            if predictions:
                latest = predictions[-1]
                print(f"  Latest prediction: {latest.prediction_type.value} = {latest.predicted_value:.4f} (confidence: {latest.confidence:.2f})")
                
            if signals:
                latest_signal = signals[-1]
                print(f"  Latest signal: {latest_signal.signal_type} - {latest_signal.direction} (strength: {latest_signal.strength:.2f})")
                
        await asyncio.Future()  # Бесконечное ожидание
        
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        await analytics_system.stop()

if __name__ == '__main__':
    asyncio.run(main())