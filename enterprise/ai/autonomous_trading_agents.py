"""
Enterprise Autonomous Trading Agents - Автономные торговые агенты
Обеспечивает интеллектуальную автономную торговлю с использованием RL и AI
"""

import asyncio
import json
import time
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import logging
import redis.asyncio as redis
import pandas as pd
from collections import deque, defaultdict
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import gym
from gym import spaces
import stable_baselines3 as sb3
from stable_baselines3 import PPO, A2C, DQN, SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from prometheus_client import Counter, Histogram, Gauge
import warnings
warnings.filterwarnings('ignore')

class AgentType(Enum):
    """Типы торговых агентов"""
    SCALPER = "scalper"  # Скальпер
    DAY_TRADER = "day_trader"  # Дневной трейдер
    SWING_TRADER = "swing_trader"  # Свинг трейдер
    ARBITRAGE = "arbitrage"  # Арбитражный
    MARKET_MAKER = "market_maker"  # Маркет-мейкер
    TREND_FOLLOWER = "trend_follower"  # Следующий за трендом
    MEAN_REVERSION = "mean_reversion"  # Возврат к среднему

class ActionType(Enum):
    """Типы действий"""
    HOLD = 0
    BUY = 1
    SELL = 2
    BUY_STRONG = 3
    SELL_STRONG = 4

class RiskLevel(Enum):
    """Уровни риска"""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"

@dataclass
class TradingAction:
    """Торговое действие"""
    agent_id: str
    symbol: str
    action_type: ActionType
    quantity: float
    price: Optional[float] = None
    confidence: float = 0.0
    reasoning: str = ""
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

@dataclass
class MarketState:
    """Состояние рынка"""
    symbol: str
    price: float
    volume: float
    bid: float
    ask: float
    spread: float
    volatility: float
    trend: float
    rsi: float
    macd: float
    bollinger_upper: float
    bollinger_lower: float
    support: float
    resistance: float
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

@dataclass
class AgentPerformance:
    """Производительность агента"""
    agent_id: str
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    win_rate: float = 0.0
    avg_trade_duration: float = 0.0
    risk_adjusted_return: float = 0.0
    last_updated: datetime = None
    
    def __post_init__(self):
        if self.last_updated is None:
            self.last_updated = datetime.now()

# Метрики
AGENT_ACTIONS_TOTAL = Counter('agent_actions_total', 'Total agent actions', ['agent_id', 'action_type'])
AGENT_PNL = Gauge('agent_pnl', 'Agent PnL', ['agent_id'])
AGENT_WIN_RATE = Gauge('agent_win_rate', 'Agent win rate', ['agent_id'])
AGENT_SHARPE_RATIO = Gauge('agent_sharpe_ratio', 'Agent Sharpe ratio', ['agent_id'])
AGENT_DECISION_TIME = Histogram('agent_decision_time_seconds', 'Agent decision time', ['agent_id'])

class TradingEnvironment(gym.Env):
    """Торговая среда для обучения с подкреплением"""
    
    def __init__(self, market_data: pd.DataFrame, initial_balance: float = 10000.0,
                 transaction_cost: float = 0.001, max_position: float = 1.0):
        super().__init__()
        
        self.market_data = market_data
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.max_position = max_position
        
        # Пространство действий: [HOLD, BUY, SELL, BUY_STRONG, SELL_STRONG]
        self.action_space = spaces.Discrete(5)
        
        # Пространство состояний: цена, объем, технические индикаторы
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(20,), dtype=np.float32
        )
        
        self.reset()
        
    def reset(self):
        """Сброс среды"""
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0.0
        self.total_pnl = 0.0
        self.trades = []
        self.max_drawdown = 0.0
        self.peak_balance = self.initial_balance
        
        return self._get_observation()
        
    def step(self, action):
        """Выполнение действия"""
        if self.current_step >= len(self.market_data) - 1:
            return self._get_observation(), 0, True, {}
            
        current_price = self.market_data.iloc[self.current_step]['close']
        
        # Выполнение действия
        reward = self._execute_action(action, current_price)
        
        # Переход к следующему шагу
        self.current_step += 1
        
        # Проверка завершения эпизода
        done = self.current_step >= len(self.market_data) - 1
        
        # Расчет дополнительных метрик
        info = self._get_info()
        
        return self._get_observation(), reward, done, info
        
    def _execute_action(self, action, price):
        """Выполнение торгового действия"""
        action_type = ActionType(action)
        
        old_position = self.position
        old_balance = self.balance
        
        if action_type == ActionType.BUY:
            # Покупка 25% от максимальной позиции
            target_position = min(self.position + 0.25, self.max_position)
        elif action_type == ActionType.BUY_STRONG:
            # Сильная покупка 50% от максимальной позиции
            target_position = min(self.position + 0.5, self.max_position)
        elif action_type == ActionType.SELL:
            # Продажа 25% от максимальной позиции
            target_position = max(self.position - 0.25, -self.max_position)
        elif action_type == ActionType.SELL_STRONG:
            # Сильная продажа 50% от максимальной позиции
            target_position = max(self.position - 0.5, -self.max_position)
        else:  # HOLD
            target_position = self.position
            
        # Расчет изменения позиции
        position_change = target_position - self.position
        
        if abs(position_change) > 0.001:  # Минимальное изменение
            # Расчет стоимости транзакции
            transaction_value = abs(position_change) * price * self.initial_balance
            transaction_fee = transaction_value * self.transaction_cost
            
            # Обновление баланса и позиции
            self.balance -= transaction_fee
            self.position = target_position
            
            # Запись сделки
            self.trades.append({
                'step': self.current_step,
                'action': action_type,
                'price': price,
                'position_change': position_change,
                'fee': transaction_fee
            })
            
        # Расчет текущей стоимости портфеля
        portfolio_value = self.balance + self.position * price * self.initial_balance
        
        # Обновление максимума и просадки
        if portfolio_value > self.peak_balance:
            self.peak_balance = portfolio_value
        else:
            drawdown = (self.peak_balance - portfolio_value) / self.peak_balance
            self.max_drawdown = max(self.max_drawdown, drawdown)
            
        # Расчет награды
        pnl_change = portfolio_value - (old_balance + old_position * price * self.initial_balance)
        
        # Нормализация награды
        reward = pnl_change / self.initial_balance
        
        # Штраф за большую просадку
        if self.max_drawdown > 0.1:  # 10% просадка
            reward -= self.max_drawdown * 10
            
        self.total_pnl = portfolio_value - self.initial_balance
        
        return reward
        
    def _get_observation(self):
        """Получение наблюдения состояния"""
        if self.current_step >= len(self.market_data):
            return np.zeros(20, dtype=np.float32)
            
        row = self.market_data.iloc[self.current_step]
        
        # Технические индикаторы и состояние рынка
        obs = np.array([
            row.get('close', 0),
            row.get('volume', 0),
            row.get('high', 0),
            row.get('low', 0),
            row.get('rsi', 50),
            row.get('macd', 0),
            row.get('macd_signal', 0),
            row.get('bb_upper', 0),
            row.get('bb_lower', 0),
            row.get('sma_20', 0),
            row.get('sma_50', 0),
            row.get('ema_12', 0),
            row.get('ema_26', 0),
            row.get('volatility', 0),
            row.get('atr', 0),
            self.position,
            self.balance / self.initial_balance,
            self.total_pnl / self.initial_balance,
            self.max_drawdown,
            len(self.trades)
        ], dtype=np.float32)
        
        return obs
        
    def _get_info(self):
        """Получение дополнительной информации"""
        portfolio_value = self.balance + self.position * self.market_data.iloc[self.current_step]['close'] * self.initial_balance
        
        return {
            'portfolio_value': portfolio_value,
            'position': self.position,
            'balance': self.balance,
            'total_pnl': self.total_pnl,
            'max_drawdown': self.max_drawdown,
            'num_trades': len(self.trades)
        }

class DeepQNetwork(nn.Module):
    """Глубокая Q-сеть для DQN агента"""
    
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 256):
        super().__init__()
        
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc4 = nn.Linear(hidden_size // 2, action_size)
        
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class ActorCriticNetwork(nn.Module):
    """Сеть актор-критик для PPO агента"""
    
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 256):
        super().__init__()
        
        # Общие слои
        self.shared_fc1 = nn.Linear(state_size, hidden_size)
        self.shared_fc2 = nn.Linear(hidden_size, hidden_size)
        
        # Актор (политика)
        self.actor_fc = nn.Linear(hidden_size, hidden_size // 2)
        self.actor_out = nn.Linear(hidden_size // 2, action_size)
        
        # Критик (функция ценности)
        self.critic_fc = nn.Linear(hidden_size, hidden_size // 2)
        self.critic_out = nn.Linear(hidden_size // 2, 1)
        
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        # Общие признаки
        shared = F.relu(self.shared_fc1(x))
        shared = F.relu(self.shared_fc2(shared))
        
        # Актор
        actor = F.relu(self.actor_fc(shared))
        actor = self.dropout(actor)
        action_probs = F.softmax(self.actor_out(actor), dim=-1)
        
        # Критик
        critic = F.relu(self.critic_fc(shared))
        critic = self.dropout(critic)
        state_value = self.critic_out(critic)
        
        return action_probs, state_value

class AutonomousTradingAgent:
    """Автономный торговый агент"""
    
    def __init__(self, agent_id: str, agent_type: AgentType, config: Dict[str, Any]):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.config = config
        
        # Модели RL
        self.rl_model = None
        self.model_type = config.get('model_type', 'PPO')
        
        # Состояние агента
        self.is_active = True
        self.current_position = 0.0
        self.balance = config.get('initial_balance', 10000.0)
        self.risk_level = RiskLevel(config.get('risk_level', 'moderate'))
        
        # История торговли
        self.trade_history: List[TradingAction] = []
        self.performance = AgentPerformance(agent_id=agent_id)
        
        # Технические индикаторы
        self.technical_analyzer = TechnicalAnalyzer()
        
        # Обучение
        self.training_data = deque(maxlen=10000)
        self.is_training = False
        
        # Логирование
        self.logger = logging.getLogger(f'agent_{agent_id}')
        
    async def initialize(self, market_data: pd.DataFrame):
        """Инициализация агента"""
        # Создание торговой среды
        self.env = TradingEnvironment(
            market_data=market_data,
            initial_balance=self.balance,
            transaction_cost=self.config.get('transaction_cost', 0.001),
            max_position=self.config.get('max_position', 1.0)
        )
        
        # Создание модели RL
        await self._create_rl_model()
        
        self.logger.info(f"Agent {self.agent_id} initialized with {self.model_type}")
        
    async def _create_rl_model(self):
        """Создание модели обучения с подкреплением"""
        if self.model_type == 'PPO':
            self.rl_model = PPO(
                'MlpPolicy',
                self.env,
                learning_rate=self.config.get('learning_rate', 3e-4),
                n_steps=self.config.get('n_steps', 2048),
                batch_size=self.config.get('batch_size', 64),
                n_epochs=self.config.get('n_epochs', 10),
                gamma=self.config.get('gamma', 0.99),
                gae_lambda=self.config.get('gae_lambda', 0.95),
                clip_range=self.config.get('clip_range', 0.2),
                verbose=0
            )
        elif self.model_type == 'DQN':
            self.rl_model = DQN(
                'MlpPolicy',
                self.env,
                learning_rate=self.config.get('learning_rate', 1e-4),
                buffer_size=self.config.get('buffer_size', 100000),
                learning_starts=self.config.get('learning_starts', 1000),
                batch_size=self.config.get('batch_size', 32),
                tau=self.config.get('tau', 1.0),
                gamma=self.config.get('gamma', 0.99),
                train_freq=self.config.get('train_freq', 4),
                gradient_steps=self.config.get('gradient_steps', 1),
                target_update_interval=self.config.get('target_update_interval', 1000),
                verbose=0
            )
        elif self.model_type == 'SAC':
            self.rl_model = SAC(
                'MlpPolicy',
                self.env,
                learning_rate=self.config.get('learning_rate', 3e-4),
                buffer_size=self.config.get('buffer_size', 100000),
                learning_starts=self.config.get('learning_starts', 1000),
                batch_size=self.config.get('batch_size', 256),
                tau=self.config.get('tau', 0.005),
                gamma=self.config.get('gamma', 0.99),
                train_freq=self.config.get('train_freq', 1),
                gradient_steps=self.config.get('gradient_steps', 1),
                verbose=0
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
            
    async def make_decision(self, market_state: MarketState) -> TradingAction:
        """Принятие торгового решения"""
        start_time = time.time()
        
        try:
            # Получение наблюдения
            observation = self._prepare_observation(market_state)
            
            # Предсказание действия с помощью RL модели
            action, confidence = await self._predict_action(observation)
            
            # Применение фильтров риска
            action = self._apply_risk_filters(action, market_state)
            
            # Создание торгового действия
            trading_action = TradingAction(
                agent_id=self.agent_id,
                symbol=market_state.symbol,
                action_type=action,
                quantity=self._calculate_quantity(action, market_state),
                price=market_state.price,
                confidence=confidence,
                reasoning=self._generate_reasoning(action, market_state)
            )
            
            # Обновление истории
            self.trade_history.append(trading_action)
            
            # Метрики
            AGENT_ACTIONS_TOTAL.labels(
                agent_id=self.agent_id,
                action_type=action.value
            ).inc()
            
            AGENT_DECISION_TIME.labels(agent_id=self.agent_id).observe(
                time.time() - start_time
            )
            
            return trading_action
            
        except Exception as e:
            self.logger.error(f"Decision making error: {e}")
            # Возврат безопасного действия
            return TradingAction(
                agent_id=self.agent_id,
                symbol=market_state.symbol,
                action_type=ActionType.HOLD,
                quantity=0.0,
                confidence=0.0,
                reasoning="Error in decision making"
            )
            
    def _prepare_observation(self, market_state: MarketState) -> np.ndarray:
        """Подготовка наблюдения для модели"""
        # Нормализация данных рынка
        obs = np.array([
            market_state.price / 100000,  # Нормализация цены
            market_state.volume / 1000000,  # Нормализация объема
            market_state.spread / market_state.price,  # Относительный спред
            market_state.volatility,
            market_state.trend,
            (market_state.rsi - 50) / 50,  # Нормализация RSI
            market_state.macd / market_state.price,
            (market_state.bollinger_upper - market_state.price) / market_state.price,
            (market_state.price - market_state.bollinger_lower) / market_state.price,
            (market_state.support - market_state.price) / market_state.price,
            (market_state.resistance - market_state.price) / market_state.price,
            self.current_position,
            self.balance / self.config.get('initial_balance', 10000),
            self.performance.total_pnl / self.config.get('initial_balance', 10000),
            self.performance.max_drawdown,
            self.performance.win_rate,
            self.performance.sharpe_ratio,
            len(self.trade_history) / 1000,  # Нормализация количества сделок
            self._get_time_features(),
            self._get_agent_type_encoding()
        ], dtype=np.float32)
        
        return obs
        
    async def _predict_action(self, observation: np.ndarray) -> Tuple[ActionType, float]:
        """Предсказание действия с помощью RL модели"""
        if self.rl_model is None:
            return ActionType.HOLD, 0.0
            
        try:
            # Предсказание действия
            action, _states = self.rl_model.predict(observation, deterministic=False)
            
            # Расчет уверенности (упрощенно)
            confidence = 0.8  # Базовая уверенность
            
            # Корректировка уверенности на основе производительности
            if self.performance.win_rate > 0.6:
                confidence += 0.1
            elif self.performance.win_rate < 0.4:
                confidence -= 0.1
                
            confidence = max(0.1, min(0.95, confidence))
            
            return ActionType(action), confidence
            
        except Exception as e:
            self.logger.error(f"Action prediction error: {e}")
            return ActionType.HOLD, 0.0
            
    def _apply_risk_filters(self, action: ActionType, market_state: MarketState) -> ActionType:
        """Применение фильтров риска"""
        # Фильтр максимальной позиции
        max_position = self.config.get('max_position', 1.0)
        
        if action in [ActionType.BUY, ActionType.BUY_STRONG]:
            if self.current_position >= max_position:
                return ActionType.HOLD
        elif action in [ActionType.SELL, ActionType.SELL_STRONG]:
            if self.current_position <= -max_position:
                return ActionType.HOLD
                
        # Фильтр максимальной просадки
        if self.performance.max_drawdown > 0.15:  # 15% просадка
            if action in [ActionType.BUY_STRONG, ActionType.SELL_STRONG]:
                # Снижение агрессивности при большой просадке
                return ActionType.BUY if action == ActionType.BUY_STRONG else ActionType.SELL
                
        # Фильтр волатильности
        if market_state.volatility > 0.05:  # Высокая волатильность
            if self.risk_level == RiskLevel.CONSERVATIVE:
                if action in [ActionType.BUY_STRONG, ActionType.SELL_STRONG]:
                    return ActionType.HOLD
                    
        # Фильтр времени (избегание торговли в нерабочие часы)
        current_hour = datetime.now().hour
        if current_hour < 6 or current_hour > 22:  # Ночное время
            if action != ActionType.HOLD:
                return ActionType.HOLD
                
        return action
        
    def _calculate_quantity(self, action: ActionType, market_state: MarketState) -> float:
        """Расчет количества для торговли"""
        base_quantity = self.balance * 0.1  # 10% от баланса
        
        # Корректировка на основе типа действия
        if action == ActionType.BUY:
            quantity = base_quantity * 0.5
        elif action == ActionType.BUY_STRONG:
            quantity = base_quantity * 1.0
        elif action == ActionType.SELL:
            quantity = base_quantity * 0.5
        elif action == ActionType.SELL_STRONG:
            quantity = base_quantity * 1.0
        else:
            quantity = 0.0
            
        # Корректировка на основе уровня риска
        if self.risk_level == RiskLevel.CONSERVATIVE:
            quantity *= 0.5
        elif self.risk_level == RiskLevel.AGGRESSIVE:
            quantity *= 1.5
            
        # Корректировка на основе волатильности
        volatility_factor = 1.0 - min(market_state.volatility * 10, 0.5)
        quantity *= volatility_factor
        
        # Корректировка на основе уверенности
        # confidence_factor = self.performance.win_rate if self.performance.win_rate > 0 else 0.5
        # quantity *= confidence_factor
        
        return max(0.0, quantity)
        
    def _generate_reasoning(self, action: ActionType, market_state: MarketState) -> str:
        """Генерация обоснования решения"""
        reasons = []
        
        if action == ActionType.BUY or action == ActionType.BUY_STRONG:
            if market_state.rsi < 30:
                reasons.append("RSI oversold")
            if market_state.price < market_state.bollinger_lower:
                reasons.append("Price below Bollinger lower band")
            if market_state.trend > 0.02:
                reasons.append("Strong upward trend")
            if market_state.macd > 0:
                reasons.append("MACD bullish signal")
                
        elif action == ActionType.SELL or action == ActionType.SELL_STRONG:
            if market_state.rsi > 70:
                reasons.append("RSI overbought")
            if market_state.price > market_state.bollinger_upper:
                reasons.append("Price above Bollinger upper band")
            if market_state.trend < -0.02:
                reasons.append("Strong downward trend")
            if market_state.macd < 0:
                reasons.append("MACD bearish signal")
                
        else:  # HOLD
            reasons.append("Market conditions unclear")
            
        return "; ".join(reasons) if reasons else "No clear signal"
        
    def _get_time_features(self) -> float:
        """Получение временных признаков"""
        now = datetime.now()
        # Нормализация часа дня
        return now.hour / 24.0
        
    def _get_agent_type_encoding(self) -> float:
        """Кодирование типа агента"""
        type_mapping = {
            AgentType.SCALPER: 0.1,
            AgentType.DAY_TRADER: 0.2,
            AgentType.SWING_TRADER: 0.3,
            AgentType.ARBITRAGE: 0.4,
            AgentType.MARKET_MAKER: 0.5,
            AgentType.TREND_FOLLOWER: 0.6,
            AgentType.MEAN_REVERSION: 0.7
        }
        return type_mapping.get(self.agent_type, 0.0)
        
    async def train(self, training_steps: int = 10000):
        """Обучение агента"""
        if self.rl_model is None:
            return
            
        self.is_training = True
        
        try:
            self.logger.info(f"Starting training for {training_steps} steps")
            
            # Обучение модели
            self.rl_model.learn(total_timesteps=training_steps)
            
            self.logger.info("Training completed")
            
        except Exception as e:
            self.logger.error(f"Training error: {e}")
        finally:
            self.is_training = False
            
    async def update_performance(self, executed_action: TradingAction, 
                               result_pnl: float, trade_duration: float):
        """Обновление производительности агента"""
        self.performance.total_trades += 1
        
        if result_pnl > 0:
            self.performance.winning_trades += 1
        else:
            self.performance.losing_trades += 1
            
        self.performance.total_pnl += result_pnl
        
        # Обновление win rate
        self.performance.win_rate = (
            self.performance.winning_trades / self.performance.total_trades
            if self.performance.total_trades > 0 else 0.0
        )
        
        # Обновление средней продолжительности сделки
        if self.performance.total_trades == 1:
            self.performance.avg_trade_duration = trade_duration
        else:
            self.performance.avg_trade_duration = (
                (self.performance.avg_trade_duration * (self.performance.total_trades - 1) + trade_duration) /
                self.performance.total_trades
            )
            
        # Расчет Sharpe ratio (упрощенно)
        if len(self.trade_history) > 10:
            returns = [trade.confidence for trade in self.trade_history[-10:]]
            if np.std(returns) > 0:
                self.performance.sharpe_ratio = np.mean(returns) / np.std(returns)
                
        # Обновление максимальной просадки
        if result_pnl < 0:
            current_drawdown = abs(result_pnl) / self.balance
            self.performance.max_drawdown = max(self.performance.max_drawdown, current_drawdown)
            
        self.performance.last_updated = datetime.now()
        
        # Обновление метрик
        AGENT_PNL.labels(agent_id=self.agent_id).set(self.performance.total_pnl)
        AGENT_WIN_RATE.labels(agent_id=self.agent_id).set(self.performance.win_rate)
        AGENT_SHARPE_RATIO.labels(agent_id=self.agent_id).set(self.performance.sharpe_ratio)
        
    def save_model(self, filepath: str):
        """Сохранение модели агента"""
        if self.rl_model:
            self.rl_model.save(filepath)
            
    def load_model(self, filepath: str):
        """Загрузка модели агента"""
        if self.model_type == 'PPO':
            self.rl_model = PPO.load(filepath, env=self.env)
        elif self.model_type == 'DQN':
            self.rl_model = DQN.load(filepath, env=self.env)
        elif self.model_type == 'SAC':
            self.rl_model = SAC.load(filepath, env=self.env)

class TechnicalAnalyzer:
    """Анализатор технических индикаторов"""
    
    def __init__(self):
        self.indicators = {}
        
    def calculate_rsi(self, prices: List[float], period: int = 14) -> float:
        """Расчет RSI"""
        if len(prices) < period + 1:
            return 50.0
            
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
            
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
        
    def calculate_macd(self, prices: List[float], fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[float, float]:
        """Расчет MACD"""
        if len(prices) < slow:
            return 0.0, 0.0
            
        prices_array = np.array(prices)
        
        # Экспоненциальные скользящие средние
        ema_fast = self._ema(prices_array, fast)
        ema_slow = self._ema(prices_array, slow)
        
        # MACD линия
        macd_line = ema_fast[-1] - ema_slow[-1]
        
        # Сигнальная линия (упрощенно)
        signal_line = macd_line * 0.9  # Упрощение
        
        return macd_line, signal_line
        
    def calculate_bollinger_bands(self, prices: List[float], period: int = 20, std_dev: float = 2.0) -> Tuple[float, float, float]:
        """Расчет полос Боллинджера"""
        if len(prices) < period:
            price = prices[-1] if prices else 0.0
            return price, price, price
            
        prices_array = np.array(prices[-period:])
        sma = np.mean(prices_array)
        std = np.std(prices_array)
        
        upper_band = sma + (std_dev * std)
        lower_band = sma - (std_dev * std)
        
        return upper_band, sma, lower_band
        
    def _ema(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Экспоненциальная скользящая средняя"""
        alpha = 2.0 / (period + 1.0)
        ema = np.zeros_like(prices)
        ema[0] = prices[0]
        
        for i in range(1, len(prices)):
            ema[i] = alpha * prices[i] + (1 - alpha) * ema[i-1]
            
        return ema

class EnterpriseAutonomousTradingSystem:
    """Enterprise система автономных торговых агентов"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.redis_client = None
        self.logger = self._setup_logging()
        
        # Агенты
        self.agents: Dict[str, AutonomousTradingAgent] = {}
        self.agent_configs = config.get('agent_configs', {})
        
        # Состояние рынка
        self.market_states: Dict[str, MarketState] = {}
        self.market_data_cache: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # Координация агентов
        self.agent_coordinator = AgentCoordinator()
        
        # Мониторинг производительности
        self.performance_monitor = PerformanceMonitor()
        
    def _setup_logging(self) -> logging.Logger:
        """Настройка логирования"""
        logger = logging.getLogger('enterprise_autonomous_trading')
        logger.setLevel(logging.INFO)
        
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
        
    async def start(self):
        """Запуск системы автономной торговли"""
        # Подключение к Redis
        self.redis_client = redis.Redis(
            host=self.config.get('redis_host', 'localhost'),
            port=self.config.get('redis_port', 6379),
            decode_responses=True
        )
        
        # Создание агентов
        await self._create_agents()
        
        # Запуск фоновых задач
        asyncio.create_task(self._market_data_processor())
        asyncio.create_task(self._agent_decision_loop())
        asyncio.create_task(self._performance_monitor_loop())
        asyncio.create_task(self._agent_trainer())
        
        self.logger.info("Enterprise Autonomous Trading System started")
        
    async def stop(self):
        """Остановка системы"""
        # Сохранение моделей агентов
        for agent in self.agents.values():
            model_path = f"models/agent_{agent.agent_id}_{int(time.time())}"
            agent.save_model(model_path)
            
        if self.redis_client:
            await self.redis_client.close()
            
    async def _create_agents(self):
        """Создание торговых агентов"""
        for agent_config in self.agent_configs:
            agent_id = agent_config['id']
            agent_type = AgentType(agent_config['type'])
            
            agent = AutonomousTradingAgent(agent_id, agent_type, agent_config)
            
            # Инициализация с историческими данными
            market_data = await self._get_historical_data(agent_config.get('symbol', 'BTCUSDT'))
            await agent.initialize(market_data)
            
            self.agents[agent_id] = agent
            
        self.logger.info(f"Created {len(self.agents)} trading agents")
        
    async def _get_historical_data(self, symbol: str) -> pd.DataFrame:
        """Получение исторических данных"""
        # Заглушка для исторических данных
        # В реальной реализации здесь должно быть подключение к API биржи
        dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='1H')
        
        # Генерация синтетических данных
        np.random.seed(42)
        prices = 50000 + np.cumsum(np.random.randn(len(dates)) * 100)
        volumes = np.random.uniform(100, 1000, len(dates))
        
        df = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': prices * 1.01,
            'low': prices * 0.99,
            'close': prices,
            'volume': volumes
        })
        
        # Добавление технических индикаторов
        df = self._add_technical_indicators(df)
        
        return df
        
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Добавление технических индикаторов"""
        # RSI
        df['rsi'] = df['close'].rolling(window=14).apply(
            lambda x: self._calculate_rsi(x.values), raw=False
        )
        
        # MACD
        ema_12 = df['close'].ewm(span=12).mean()
        ema_26 = df['close'].ewm(span=26).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        
        # Bollinger Bands
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['bb_std'] = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['sma_20'] + (df['bb_std'] * 2)
        df['bb_lower'] = df['sma_20'] - (df['bb_std'] * 2)
        
        # Дополнительные индикаторы
        df['sma_50'] = df['close'].rolling(window=50).mean()
        df['ema_12'] = ema_12
        df['ema_26'] = ema_26
        df['volatility'] = df['close'].rolling(window=20).std() / df['close'].rolling(window=20).mean()
        df['atr'] = (df['high'] - df['low']).rolling(window=14).mean()
        
        return df.fillna(0)
        
    def _calculate_rsi(self, prices: np.ndarray) -> float:
        """Расчет RSI"""
        if len(prices) < 2:
            return 50.0
            
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        
        if avg_loss == 0:
            return 100.0
            
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
        
    async def _market_data_processor(self):
        """Обработчик рыночных данных"""
        while True:
            try:
                # Получение актуальных рыночных данных
                # В реальной реализации здесь должно быть подключение к WebSocket биржи
                
                # Симуляция обновления данных
                for symbol in ['BTCUSDT', 'ETHUSDT', 'ADAUSDT']:
                    market_state = await self._get_current_market_state(symbol)
                    self.market_states[symbol] = market_state
                    
                    # Добавление в кеш
                    self.market_data_cache[symbol].append(market_state)
                    
                await asyncio.sleep(1)  # Обновление каждую секунду
                
            except Exception as e:
                self.logger.error(f"Market data processor error: {e}")
                await asyncio.sleep(5)
                
    async def _get_current_market_state(self, symbol: str) -> MarketState:
        """Получение текущего состояния рынка"""
        # Симуляция рыночных данных
        # В реальной реализации здесь должен быть API вызов
        
        base_price = 50000 if symbol == 'BTCUSDT' else 3000 if symbol == 'ETHUSDT' else 1.0
        price = base_price + np.random.randn() * base_price * 0.001
        
        return MarketState(
            symbol=symbol,
            price=price,
            volume=np.random.uniform(100, 1000),
            bid=price * 0.999,
            ask=price * 1.001,
            spread=price * 0.002,
            volatility=np.random.uniform(0.01, 0.05),
            trend=np.random.uniform(-0.05, 0.05),
            rsi=np.random.uniform(20, 80),
            macd=np.random.uniform(-100, 100),
            bollinger_upper=price * 1.02,
            bollinger_lower=price * 0.98,
            support=price * 0.95,
            resistance=price * 1.05
        )
        
    async def _agent_decision_loop(self):
        """Цикл принятия решений агентами"""
        while True:
            try:
                for agent in self.agents.values():
                    if not agent.is_active or agent.is_training:
                        continue
                        
                    # Получение состояния рынка для символа агента
                    symbol = agent.config.get('symbol', 'BTCUSDT')
                    market_state = self.market_states.get(symbol)
                    
                    if market_state:
                        # Принятие решения
                        trading_action = await agent.make_decision(market_state)
                        
                        # Выполнение действия (если не HOLD)
                        if trading_action.action_type != ActionType.HOLD:
                            await self._execute_trading_action(agent, trading_action)
                            
                await asyncio.sleep(self.config.get('decision_interval', 5))  # 5 секунд
                
            except Exception as e:
                self.logger.error(f"Agent decision loop error: {e}")
                await asyncio.sleep(10)
                
    async def _execute_trading_action(self, agent: AutonomousTradingAgent, 
                                    action: TradingAction):
        """Выполнение торгового действия"""
        try:
            # Симуляция выполнения сделки
            # В реальной реализации здесь должен быть API вызов к бирже
            
            execution_price = action.price * (1 + np.random.uniform(-0.001, 0.001))
            execution_time = datetime.now()
            
            # Расчет PnL (упрощенно)
            if action.action_type in [ActionType.BUY, ActionType.BUY_STRONG]:
                pnl = (execution_price - action.price) * action.quantity
                agent.current_position += action.quantity
            else:  # SELL
                pnl = (action.price - execution_price) * action.quantity
                agent.current_position -= action.quantity
                
            # Обновление производительности агента
            trade_duration = 300.0  # 5 минут (упрощенно)
            await agent.update_performance(action, pnl, trade_duration)
            
            # Сохранение результата
            await self._save_trade_result(agent, action, pnl, execution_time)
            
            self.logger.info(f"Executed trade: {agent.agent_id} {action.action_type.value} {action.quantity} {action.symbol}")
            
        except Exception as e:
            self.logger.error(f"Trade execution error: {e}")
            
    async def _save_trade_result(self, agent: AutonomousTradingAgent, 
                               action: TradingAction, pnl: float, execution_time: datetime):
        """Сохранение результата сделки"""
        trade_result = {
            'agent_id': agent.agent_id,
            'action': asdict(action),
            'pnl': pnl,
            'execution_time': execution_time.isoformat(),
            'agent_performance': asdict(agent.performance)
        }
        
        await self.redis_client.lpush(
            f"trade_results:{agent.agent_id}",
            json.dumps(trade_result, default=str)
        )
        
        # Ограничение размера списка
        await self.redis_client.ltrim(f"trade_results:{agent.agent_id}", 0, 999)
        
    async def _performance_monitor_loop(self):
        """Цикл мониторинга производительности"""
        while True:
            try:
                for agent in self.agents.values():
                    await self._monitor_agent_performance(agent)
                    
                await asyncio.sleep(300)  # 5 минут
                
            except Exception as e:
                self.logger.error(f"Performance monitor error: {e}")
                await asyncio.sleep(60)
                
    async def _monitor_agent_performance(self, agent: AutonomousTradingAgent):
        """Мониторинг производительности агента"""
        # Проверка критических метрик
        if agent.performance.max_drawdown > 0.2:  # 20% просадка
            self.logger.warning(f"Agent {agent.agent_id} has high drawdown: {agent.performance.max_drawdown:.2%}")
            
        if agent.performance.win_rate < 0.3 and agent.performance.total_trades > 10:
            self.logger.warning(f"Agent {agent.agent_id} has low win rate: {agent.performance.win_rate:.2%}")
            
        # Автоматическое отключение неэффективных агентов
        if (agent.performance.total_trades > 50 and 
            agent.performance.win_rate < 0.25 and 
            agent.performance.total_pnl < -agent.balance * 0.1):
            
            agent.is_active = False
            self.logger.warning(f"Agent {agent.agent_id} deactivated due to poor performance")
            
    async def _agent_trainer(self):
        """Тренер агентов"""
        while True:
            try:
                for agent in self.agents.values():
                    # Периодическое переобучение агентов
                    if (agent.performance.total_trades > 0 and 
                        agent.performance.total_trades % 100 == 0):  # Каждые 100 сделок
                        
                        self.logger.info(f"Starting retraining for agent {agent.agent_id}")
                        await agent.train(training_steps=5000)
                        
                await asyncio.sleep(3600)  # 1 час
                
            except Exception as e:
                self.logger.error(f"Agent trainer error: {e}")
                await asyncio.sleep(1800)

class AgentCoordinator:
    """Координатор агентов"""
    
    def __init__(self):
        pass
        
    async def coordinate_agents(self, agents: List[AutonomousTradingAgent], 
                              market_state: MarketState) -> List[TradingAction]:
        """Координация действий агентов"""
        # Здесь может быть логика координации между агентами
        # Например, предотвращение конфликтующих действий
        pass

class PerformanceMonitor:
    """Монитор производительности"""
    
    def __init__(self):
        pass
        
    async def generate_performance_report(self, agents: List[AutonomousTradingAgent]) -> Dict[str, Any]:
        """Генерация отчета о производительности"""
        total_pnl = sum(agent.performance.total_pnl for agent in agents)
        avg_win_rate = np.mean([agent.performance.win_rate for agent in agents])
        
        return {
            'total_agents': len(agents),
            'active_agents': sum(1 for agent in agents if agent.is_active),
            'total_pnl': total_pnl,
            'average_win_rate': avg_win_rate,
            'timestamp': datetime.now().isoformat()
        }

async def main():
    """Основная функция запуска"""
    config = {
        'redis_host': 'localhost',
        'redis_port': 6379,
        'decision_interval': 5,
        'agent_configs': [
            {
                'id': 'scalper_001',
                'type': 'scalper',
                'symbol': 'BTCUSDT',
                'model_type': 'DQN',
                'initial_balance': 10000,
                'risk_level': 'aggressive',
                'max_position': 0.5,
                'transaction_cost': 0.001
            },
            {
                'id': 'swing_trader_001',
                'type': 'swing_trader',
                'symbol': 'ETHUSDT',
                'model_type': 'PPO',
                'initial_balance': 15000,
                'risk_level': 'moderate',
                'max_position': 1.0,
                'transaction_cost': 0.001
            },
            {
                'id': 'arbitrage_001',
                'type': 'arbitrage',
                'symbol': 'ADAUSDT',
                'model_type': 'SAC',
                'initial_balance': 20000,
                'risk_level': 'conservative',
                'max_position': 0.3,
                'transaction_cost': 0.0005
            }
        ]
    }
    
    trading_system = EnterpriseAutonomousTradingSystem(config)
    await trading_system.start()
    
    print("Enterprise Autonomous Trading System started")
    print(f"Active agents: {len(trading_system.agents)}")
    
    try:
        await asyncio.Future()  # Бесконечное ожидание
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        await trading_system.stop()

if __name__ == '__main__':
    asyncio.run(main())