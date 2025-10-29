"""
Enterprise Social Trading Platform - Платформа социального трейдинга
Обеспечивает копи-трейдинг, социальные функции и управление сообществом трейдеров
"""

import asyncio
import json
import time
import hashlib
from typing import Dict, List, Optional, Any, Tuple, Union, Set
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import redis.asyncio as redis
from collections import defaultdict, deque
import uuid
import numpy as np
from scipy import stats
import websockets
import aiohttp

# Метрики и мониторинг
from prometheus_client import Counter, Histogram, Gauge

# Машинное обучение для рекомендаций
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

class TraderTier(Enum):
    """Уровни трейдеров"""
    NOVICE = "novice"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"
    MASTER = "master"

class CopyMode(Enum):
    """Режимы копирования"""
    PROPORTIONAL = "proportional"  # Пропорциональное копирование
    FIXED_AMOUNT = "fixed_amount"  # Фиксированная сумма
    PERCENTAGE = "percentage"      # Процент от депозита
    MIRROR = "mirror"             # Зеркальное копирование

class SignalType(Enum):
    """Типы сигналов"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    CLOSE = "close"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"

class PostType(Enum):
    """Типы постов"""
    ANALYSIS = "analysis"
    SIGNAL = "signal"
    EDUCATION = "education"
    NEWS = "news"
    DISCUSSION = "discussion"

class FollowType(Enum):
    """Типы подписок"""
    FOLLOW = "follow"           # Просто подписка
    COPY_TRADE = "copy_trade"   # Копирование сделок
    SIGNAL_ONLY = "signal_only" # Только сигналы

@dataclass
class TraderProfile:
    """Профиль трейдера"""
    user_id: str
    username: str
    display_name: str
    tier: TraderTier
    bio: Optional[str] = None
    avatar_url: Optional[str] = None
    verified: bool = False
    created_at: datetime = field(default_factory=datetime.now)
    
    # Статистика
    total_followers: int = 0
    total_copiers: int = 0
    total_posts: int = 0
    reputation_score: float = 0.0
    
    # Торговая статистика
    win_rate: float = 0.0
    total_trades: int = 0
    avg_profit: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    total_pnl: float = 0.0
    
    # Настройки
    is_public: bool = True
    allow_copying: bool = True
    copy_fee_percentage: float = 0.0
    min_copy_amount: float = 100.0
    max_copiers: int = 1000

@dataclass
class TradingSignal:
    """Торговый сигнал"""
    id: str
    trader_id: str
    signal_type: SignalType
    symbol: str
    price: float
    quantity: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    confidence: float = 0.0  # 0-100
    reasoning: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    
    # Результаты
    executed_price: Optional[float] = None
    executed_at: Optional[datetime] = None
    closed_price: Optional[float] = None
    closed_at: Optional[datetime] = None
    pnl: Optional[float] = None
    is_active: bool = True

@dataclass
class CopyTradingSettings:
    """Настройки копи-трейдинга"""
    id: str
    follower_id: str
    trader_id: str
    copy_mode: CopyMode
    is_active: bool = True
    
    # Параметры копирования
    copy_amount: Optional[float] = None  # Для FIXED_AMOUNT
    copy_percentage: Optional[float] = None  # Для PERCENTAGE
    proportion_factor: float = 1.0  # Для PROPORTIONAL
    
    # Ограничения
    max_daily_loss: Optional[float] = None
    max_position_size: Optional[float] = None
    allowed_symbols: Optional[List[str]] = None
    excluded_symbols: Optional[List[str]] = None
    
    # Управление рисками
    stop_loss_percentage: Optional[float] = None
    take_profit_percentage: Optional[float] = None
    max_open_positions: int = 10
    
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class SocialPost:
    """Социальный пост"""
    id: str
    author_id: str
    post_type: PostType
    title: str
    content: str
    tags: List[str] = field(default_factory=list)
    attachments: List[str] = field(default_factory=list)  # URLs к изображениям/файлам
    created_at: datetime = field(default_factory=datetime.now)
    
    # Взаимодействие
    likes: int = 0
    comments: int = 0
    shares: int = 0
    views: int = 0
    
    # Связанные данные
    related_signal_id: Optional[str] = None
    related_symbols: List[str] = field(default_factory=list)

@dataclass
class Comment:
    """Комментарий"""
    id: str
    post_id: str
    author_id: str
    content: str
    parent_comment_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    likes: int = 0

@dataclass
class Follow:
    """Подписка"""
    id: str
    follower_id: str
    following_id: str
    follow_type: FollowType
    created_at: datetime = field(default_factory=datetime.now)
    is_active: bool = True

@dataclass
class CopyTradeExecution:
    """Выполнение копи-сделки"""
    id: str
    original_signal_id: str
    original_trader_id: str
    copier_id: str
    copy_settings_id: str
    
    # Детали сделки
    symbol: str
    side: str
    original_quantity: float
    copied_quantity: float
    original_price: float
    executed_price: float
    
    # Результаты
    pnl: Optional[float] = None
    fee_paid: float = 0.0
    
    executed_at: datetime = field(default_factory=datetime.now)
    closed_at: Optional[datetime] = None

@dataclass
class LeaderboardEntry:
    """Запись в рейтинге"""
    trader_id: str
    rank: int
    score: float
    period: str  # daily, weekly, monthly, all_time
    metric_type: str  # pnl, win_rate, sharpe_ratio, etc.

# Метрики Prometheus
SOCIAL_POSTS_TOTAL = Counter('social_posts_total', 'Total social posts created', ['post_type', 'author_tier'])
SIGNALS_GENERATED = Counter('trading_signals_total', 'Total trading signals generated', ['signal_type', 'trader_tier'])
COPY_TRADES_EXECUTED = Counter('copy_trades_total', 'Total copy trades executed', ['copy_mode'])
FOLLOWERS_TOTAL = Gauge('followers_total', 'Total followers', ['trader_id'])
SOCIAL_ENGAGEMENT = Counter('social_engagement_total', 'Total social engagement', ['action_type'])

class RecommendationEngine:
    """Движок рекомендаций"""
    
    def __init__(self):
        self.user_features: Dict[str, np.ndarray] = {}
        self.trader_features: Dict[str, np.ndarray] = {}
        self.interaction_matrix: Dict[Tuple[str, str], float] = {}
        
    def update_user_features(self, user_id: str, features: Dict[str, float]):
        """Обновление характеристик пользователя"""
        feature_vector = np.array([
            features.get('risk_tolerance', 0.5),
            features.get('trading_experience', 0.0),
            features.get('preferred_timeframe', 0.5),
            features.get('capital_size', 0.5),
            features.get('win_rate_preference', 0.5),
            features.get('drawdown_tolerance', 0.5)
        ])
        self.user_features[user_id] = feature_vector
        
    def update_trader_features(self, trader_id: str, profile: TraderProfile):
        """Обновление характеристик трейдера"""
        feature_vector = np.array([
            profile.win_rate / 100.0,
            min(profile.sharpe_ratio / 3.0, 1.0),  # Нормализация
            1.0 - min(abs(profile.max_drawdown) / 20.0, 1.0),  # Обратная просадка
            min(profile.reputation_score / 100.0, 1.0),
            min(profile.total_followers / 1000.0, 1.0),
            profile.tier.value == TraderTier.EXPERT.value or profile.tier.value == TraderTier.MASTER.value
        ])
        self.trader_features[trader_id] = feature_vector
        
    def record_interaction(self, user_id: str, trader_id: str, interaction_type: str, weight: float = 1.0):
        """Запись взаимодействия"""
        interaction_weights = {
            'view': 0.1,
            'like': 0.3,
            'comment': 0.5,
            'follow': 0.7,
            'copy': 1.0
        }
        
        weight = interaction_weights.get(interaction_type, weight)
        key = (user_id, trader_id)
        self.interaction_matrix[key] = self.interaction_matrix.get(key, 0.0) + weight
        
    def get_trader_recommendations(self, user_id: str, limit: int = 10) -> List[Tuple[str, float]]:
        """Получение рекомендаций трейдеров"""
        if user_id not in self.user_features:
            return []
            
        user_vector = self.user_features[user_id]
        recommendations = []
        
        for trader_id, trader_vector in self.trader_features.items():
            # Косинусное сходство
            similarity = cosine_similarity([user_vector], [trader_vector])[0][0]
            
            # Бонус за взаимодействия
            interaction_bonus = self.interaction_matrix.get((user_id, trader_id), 0.0) * 0.1
            
            # Итоговый скор
            score = similarity + interaction_bonus
            recommendations.append((trader_id, score))
            
        # Сортировка по скору
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:limit]

class SocialFeedManager:
    """Менеджер социальной ленты"""
    
    def __init__(self):
        self.posts: Dict[str, SocialPost] = {}
        self.comments: Dict[str, List[Comment]] = defaultdict(list)
        self.user_feeds: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
    def create_post(self, post: SocialPost) -> str:
        """Создание поста"""
        self.posts[post.id] = post
        
        # Добавление в ленты подписчиков
        # В реальной системе это должно быть асинхронно
        
        SOCIAL_POSTS_TOTAL.labels(
            post_type=post.post_type.value,
            author_tier="unknown"  # Нужно получить из профиля
        ).inc()
        
        return post.id
        
    def add_comment(self, comment: Comment):
        """Добавление комментария"""
        self.comments[comment.post_id].append(comment)
        
        if comment.post_id in self.posts:
            self.posts[comment.post_id].comments += 1
            
        SOCIAL_ENGAGEMENT.labels(action_type='comment').inc()
        
    def like_post(self, post_id: str, user_id: str):
        """Лайк поста"""
        if post_id in self.posts:
            self.posts[post_id].likes += 1
            
        SOCIAL_ENGAGEMENT.labels(action_type='like').inc()
        
    def get_user_feed(self, user_id: str, limit: int = 50) -> List[SocialPost]:
        """Получение ленты пользователя"""
        feed_post_ids = list(self.user_feeds[user_id])[-limit:]
        return [self.posts[post_id] for post_id in feed_post_ids if post_id in self.posts]
        
    def get_trending_posts(self, timeframe_hours: int = 24, limit: int = 20) -> List[SocialPost]:
        """Получение трендовых постов"""
        cutoff_time = datetime.now() - timedelta(hours=timeframe_hours)
        
        recent_posts = [
            post for post in self.posts.values()
            if post.created_at >= cutoff_time
        ]
        
        # Сортировка по engagement score
        def engagement_score(post: SocialPost) -> float:
            age_hours = (datetime.now() - post.created_at).total_seconds() / 3600
            decay_factor = max(0.1, 1.0 - (age_hours / 24))  # Затухание за 24 часа
            
            return (post.likes * 1.0 + post.comments * 2.0 + post.shares * 3.0) * decay_factor
            
        recent_posts.sort(key=engagement_score, reverse=True)
        return recent_posts[:limit]

class CopyTradingEngine:
    """Движок копи-трейдинга"""
    
    def __init__(self):
        self.copy_settings: Dict[str, CopyTradingSettings] = {}
        self.active_copies: Dict[str, List[str]] = defaultdict(list)  # trader_id -> [copier_ids]
        self.copy_executions: List[CopyTradeExecution] = []
        
    def add_copy_settings(self, settings: CopyTradingSettings):
        """Добавление настроек копирования"""
        self.copy_settings[settings.id] = settings
        self.active_copies[settings.trader_id].append(settings.follower_id)
        
    def remove_copy_settings(self, settings_id: str):
        """Удаление настроек копирования"""
        if settings_id in self.copy_settings:
            settings = self.copy_settings[settings_id]
            self.active_copies[settings.trader_id].remove(settings.follower_id)
            del self.copy_settings[settings_id]
            
    async def execute_copy_trades(self, signal: TradingSignal) -> List[CopyTradeExecution]:
        """Выполнение копи-сделок по сигналу"""
        executions = []
        
        # Получение всех копировщиков этого трейдера
        copier_ids = self.active_copies.get(signal.trader_id, [])
        
        for copier_id in copier_ids:
            # Поиск настроек копирования
            copy_settings = None
            for settings in self.copy_settings.values():
                if (settings.follower_id == copier_id and 
                    settings.trader_id == signal.trader_id and 
                    settings.is_active):
                    copy_settings = settings
                    break
                    
            if not copy_settings:
                continue
                
            # Проверка ограничений
            if not self._check_copy_constraints(copy_settings, signal):
                continue
                
            # Расчет количества для копирования
            copy_quantity = self._calculate_copy_quantity(copy_settings, signal)
            if copy_quantity <= 0:
                continue
                
            # Создание копи-сделки
            execution = CopyTradeExecution(
                id=str(uuid.uuid4()),
                original_signal_id=signal.id,
                original_trader_id=signal.trader_id,
                copier_id=copier_id,
                copy_settings_id=copy_settings.id,
                symbol=signal.symbol,
                side=signal.signal_type.value,
                original_quantity=signal.quantity or 0,
                copied_quantity=copy_quantity,
                original_price=signal.price,
                executed_price=signal.price  # В реальности может отличаться
            )
            
            executions.append(execution)
            self.copy_executions.append(execution)
            
            COPY_TRADES_EXECUTED.labels(copy_mode=copy_settings.copy_mode.value).inc()
            
        return executions
        
    def _check_copy_constraints(self, settings: CopyTradingSettings, signal: TradingSignal) -> bool:
        """Проверка ограничений копирования"""
        # Проверка разрешенных символов
        if settings.allowed_symbols and signal.symbol not in settings.allowed_symbols:
            return False
            
        # Проверка исключенных символов
        if settings.excluded_symbols and signal.symbol in settings.excluded_symbols:
            return False
            
        # Дополнительные проверки (лимиты, риски и т.д.)
        return True
        
    def _calculate_copy_quantity(self, settings: CopyTradingSettings, signal: TradingSignal) -> float:
        """Расчет количества для копирования"""
        if not signal.quantity:
            return 0.0
            
        if settings.copy_mode == CopyMode.PROPORTIONAL:
            return signal.quantity * settings.proportion_factor
            
        elif settings.copy_mode == CopyMode.FIXED_AMOUNT:
            if settings.copy_amount:
                return settings.copy_amount / signal.price
            return 0.0
            
        elif settings.copy_mode == CopyMode.PERCENTAGE:
            # Нужен баланс пользователя для расчета
            # В реальной системе получаем из базы данных
            user_balance = 10000.0  # Заглушка
            if settings.copy_percentage:
                amount = user_balance * (settings.copy_percentage / 100.0)
                return amount / signal.price
            return 0.0
            
        elif settings.copy_mode == CopyMode.MIRROR:
            return signal.quantity
            
        return 0.0

class LeaderboardManager:
    """Менеджер рейтингов"""
    
    def __init__(self):
        self.leaderboards: Dict[str, List[LeaderboardEntry]] = defaultdict(list)
        
    def update_leaderboard(self, period: str, metric_type: str, trader_profiles: Dict[str, TraderProfile]):
        """Обновление рейтинга"""
        entries = []
        
        for trader_id, profile in trader_profiles.items():
            score = self._calculate_score(profile, metric_type)
            entry = LeaderboardEntry(
                trader_id=trader_id,
                rank=0,  # Будет установлен после сортировки
                score=score,
                period=period,
                metric_type=metric_type
            )
            entries.append(entry)
            
        # Сортировка и установка рангов
        entries.sort(key=lambda x: x.score, reverse=True)
        for i, entry in enumerate(entries):
            entry.rank = i + 1
            
        self.leaderboards[f"{period}_{metric_type}"] = entries
        
    def _calculate_score(self, profile: TraderProfile, metric_type: str) -> float:
        """Расчет скора для рейтинга"""
        if metric_type == "pnl":
            return profile.total_pnl
        elif metric_type == "win_rate":
            return profile.win_rate
        elif metric_type == "sharpe_ratio":
            return profile.sharpe_ratio
        elif metric_type == "followers":
            return float(profile.total_followers)
        elif metric_type == "reputation":
            return profile.reputation_score
        elif metric_type == "composite":
            # Композитный скор
            return (
                profile.win_rate * 0.3 +
                profile.sharpe_ratio * 20 * 0.3 +
                min(profile.reputation_score, 100) * 0.2 +
                min(profile.total_followers / 100, 100) * 0.2
            )
        return 0.0
        
    def get_leaderboard(self, period: str, metric_type: str, limit: int = 100) -> List[LeaderboardEntry]:
        """Получение рейтинга"""
        key = f"{period}_{metric_type}"
        return self.leaderboards.get(key, [])[:limit]

class EnterpriseSocialTradingPlatform:
    """Enterprise платформа социального трейдинга"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.redis_client = None
        self.logger = self._setup_logging()
        
        # Компоненты системы
        self.recommendation_engine = RecommendationEngine()
        self.social_feed_manager = SocialFeedManager()
        self.copy_trading_engine = CopyTradingEngine()
        self.leaderboard_manager = LeaderboardManager()
        
        # Данные
        self.trader_profiles: Dict[str, TraderProfile] = {}
        self.trading_signals: Dict[str, TradingSignal] = {}
        self.follows: Dict[str, Follow] = {}
        
        # WebSocket соединения
        self.websocket_connections: Dict[str, websockets.WebSocketServerProtocol] = {}
        
    def _setup_logging(self) -> logging.Logger:
        """Настройка логирования"""
        logger = logging.getLogger('social_trading')
        logger.setLevel(logging.INFO)
        
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
        
    async def start(self):
        """Запуск платформы"""
        # Подключение к Redis
        self.redis_client = redis.Redis(
            host=self.config.get('redis_host', 'localhost'),
            port=self.config.get('redis_port', 6379),
            decode_responses=True
        )
        
        # Запуск фоновых задач
        asyncio.create_task(self._signal_processing_loop())
        asyncio.create_task(self._leaderboard_update_loop())
        asyncio.create_task(self._recommendation_update_loop())
        asyncio.create_task(self._websocket_server())
        
        self.logger.info("Enterprise Social Trading Platform started")
        
    async def stop(self):
        """Остановка платформы"""
        if self.redis_client:
            await self.redis_client.close()
            
    async def create_trader_profile(self, profile: TraderProfile) -> str:
        """Создание профиля трейдера"""
        self.trader_profiles[profile.user_id] = profile
        
        # Обновление характеристик для рекомендаций
        self.recommendation_engine.update_trader_features(profile.user_id, profile)
        
        # Сохранение в Redis
        await self.redis_client.set(
            f"trader_profile_{profile.user_id}",
            json.dumps(asdict(profile), default=str),
            ex=86400
        )
        
        self.logger.info(f"Trader profile created: {profile.username}")
        return profile.user_id
        
    async def publish_signal(self, signal: TradingSignal) -> str:
        """Публикация торгового сигнала"""
        self.trading_signals[signal.id] = signal
        
        # Выполнение копи-сделок
        copy_executions = await self.copy_trading_engine.execute_copy_trades(signal)
        
        # Создание поста в социальной ленте
        post = SocialPost(
            id=str(uuid.uuid4()),
            author_id=signal.trader_id,
            post_type=PostType.SIGNAL,
            title=f"Trading Signal: {signal.signal_type.value.upper()} {signal.symbol}",
            content=f"Signal: {signal.signal_type.value} {signal.symbol} at {signal.price}\n"
                   f"Confidence: {signal.confidence}%\n"
                   f"Reasoning: {signal.reasoning or 'No reasoning provided'}",
            related_signal_id=signal.id,
            related_symbols=[signal.symbol]
        )
        
        self.social_feed_manager.create_post(post)
        
        # Уведомление подписчиков через WebSocket
        await self._notify_followers(signal.trader_id, {
            'type': 'new_signal',
            'signal': asdict(signal),
            'copy_executions': len(copy_executions)
        })
        
        # Сохранение в Redis
        await self.redis_client.set(
            f"signal_{signal.id}",
            json.dumps(asdict(signal), default=str),
            ex=86400
        )
        
        SIGNALS_GENERATED.labels(
            signal_type=signal.signal_type.value,
            trader_tier=self.trader_profiles.get(signal.trader_id, TraderProfile("", "", "", TraderTier.NOVICE)).tier.value
        ).inc()
        
        self.logger.info(f"Signal published: {signal.id} by {signal.trader_id}")
        return signal.id
        
    async def follow_trader(self, follower_id: str, trader_id: str, follow_type: FollowType) -> str:
        """Подписка на трейдера"""
        follow_id = str(uuid.uuid4())
        
        follow = Follow(
            id=follow_id,
            follower_id=follower_id,
            following_id=trader_id,
            follow_type=follow_type
        )
        
        self.follows[follow_id] = follow
        
        # Обновление счетчика подписчиков
        if trader_id in self.trader_profiles:
            self.trader_profiles[trader_id].total_followers += 1
            FOLLOWERS_TOTAL.labels(trader_id=trader_id).set(
                self.trader_profiles[trader_id].total_followers
            )
            
        # Запись взаимодействия для рекомендаций
        self.recommendation_engine.record_interaction(follower_id, trader_id, 'follow')
        
        # Сохранение в Redis
        await self.redis_client.set(
            f"follow_{follow_id}",
            json.dumps(asdict(follow), default=str),
            ex=86400 * 30
        )
        
        self.logger.info(f"User {follower_id} followed {trader_id} with type {follow_type.value}")
        return follow_id
        
    async def setup_copy_trading(self, settings: CopyTradingSettings) -> str:
        """Настройка копи-трейдинга"""
        self.copy_trading_engine.add_copy_settings(settings)
        
        # Сохранение в Redis
        await self.redis_client.set(
            f"copy_settings_{settings.id}",
            json.dumps(asdict(settings), default=str),
            ex=86400 * 30
        )
        
        self.logger.info(f"Copy trading setup: {settings.follower_id} -> {settings.trader_id}")
        return settings.id
        
    async def create_social_post(self, post: SocialPost) -> str:
        """Создание социального поста"""
        post_id = self.social_feed_manager.create_post(post)
        
        # Уведомление подписчиков
        await self._notify_followers(post.author_id, {
            'type': 'new_post',
            'post': asdict(post)
        })
        
        # Сохранение в Redis
        await self.redis_client.set(
            f"post_{post.id}",
            json.dumps(asdict(post), default=str),
            ex=86400 * 7
        )
        
        self.logger.info(f"Social post created: {post.id} by {post.author_id}")
        return post_id
        
    async def get_recommendations(self, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Получение рекомендаций трейдеров"""
        recommendations = self.recommendation_engine.get_trader_recommendations(user_id, limit)
        
        result = []
        for trader_id, score in recommendations:
            if trader_id in self.trader_profiles:
                profile = self.trader_profiles[trader_id]
                result.append({
                    'trader_id': trader_id,
                    'profile': asdict(profile),
                    'recommendation_score': score
                })
                
        return result
        
    async def get_leaderboard(self, period: str = "monthly", metric: str = "composite", limit: int = 50) -> List[Dict[str, Any]]:
        """Получение рейтинга"""
        entries = self.leaderboard_manager.get_leaderboard(period, metric, limit)
        
        result = []
        for entry in entries:
            if entry.trader_id in self.trader_profiles:
                profile = self.trader_profiles[entry.trader_id]
                result.append({
                    'rank': entry.rank,
                    'trader_id': entry.trader_id,
                    'profile': asdict(profile),
                    'score': entry.score
                })
                
        return result
        
    async def get_social_feed(self, user_id: str, feed_type: str = "following", limit: int = 50) -> List[Dict[str, Any]]:
        """Получение социальной ленты"""
        if feed_type == "trending":
            posts = self.social_feed_manager.get_trending_posts(limit=limit)
        else:
            posts = self.social_feed_manager.get_user_feed(user_id, limit)
            
        result = []
        for post in posts:
            author_profile = self.trader_profiles.get(post.author_id)
            result.append({
                'post': asdict(post),
                'author': asdict(author_profile) if author_profile else None
            })
            
        return result
        
    async def _signal_processing_loop(self):
        """Цикл обработки сигналов"""
        while True:
            try:
                # Проверка истекших сигналов
                current_time = datetime.now()
                expired_signals = []
                
                for signal_id, signal in self.trading_signals.items():
                    if (signal.expires_at and 
                        signal.expires_at <= current_time and 
                        signal.is_active):
                        expired_signals.append(signal_id)
                        
                # Деактивация истекших сигналов
                for signal_id in expired_signals:
                    self.trading_signals[signal_id].is_active = False
                    
                await asyncio.sleep(60)  # Проверка каждую минуту
                
            except Exception as e:
                self.logger.error(f"Signal processing loop error: {e}")
                await asyncio.sleep(300)
                
    async def _leaderboard_update_loop(self):
        """Цикл обновления рейтингов"""
        while True:
            try:
                # Обновление рейтингов
                periods = ["daily", "weekly", "monthly", "all_time"]
                metrics = ["pnl", "win_rate", "sharpe_ratio", "followers", "composite"]
                
                for period in periods:
                    for metric in metrics:
                        self.leaderboard_manager.update_leaderboard(
                            period, metric, self.trader_profiles
                        )
                        
                await asyncio.sleep(3600)  # Обновление каждый час
                
            except Exception as e:
                self.logger.error(f"Leaderboard update loop error: {e}")
                await asyncio.sleep(7200)
                
    async def _recommendation_update_loop(self):
        """Цикл обновления рекомендаций"""
        while True:
            try:
                # Обновление характеристик трейдеров
                for trader_id, profile in self.trader_profiles.items():
                    self.recommendation_engine.update_trader_features(trader_id, profile)
                    
                await asyncio.sleep(1800)  # Обновление каждые 30 минут
                
            except Exception as e:
                self.logger.error(f"Recommendation update loop error: {e}")
                await asyncio.sleep(3600)
                
    async def _websocket_server(self):
        """WebSocket сервер для реального времени"""
        async def handle_client(websocket, path):
            try:
                # Аутентификация клиента
                auth_message = await websocket.recv()
                auth_data = json.loads(auth_message)
                user_id = auth_data.get('user_id')
                
                if user_id:
                    self.websocket_connections[user_id] = websocket
                    self.logger.info(f"WebSocket client connected: {user_id}")
                    
                    # Отправка приветственного сообщения
                    await websocket.send(json.dumps({
                        'type': 'connected',
                        'message': 'Connected to social trading platform'
                    }))
                    
                    # Ожидание сообщений
                    async for message in websocket:
                        await self._handle_websocket_message(user_id, json.loads(message))
                        
            except websockets.exceptions.ConnectionClosed:
                if user_id in self.websocket_connections:
                    del self.websocket_connections[user_id]
                    self.logger.info(f"WebSocket client disconnected: {user_id}")
            except Exception as e:
                self.logger.error(f"WebSocket error: {e}")
                
        # Запуск WebSocket сервера
        try:
            start_server = websockets.serve(
                handle_client, 
                self.config.get('websocket_host', 'localhost'),
                self.config.get('websocket_port', 8765)
            )
            await start_server
        except Exception as e:
            self.logger.error(f"WebSocket server error: {e}")
            
    async def _handle_websocket_message(self, user_id: str, message: Dict[str, Any]):
        """Обработка WebSocket сообщения"""
        message_type = message.get('type')
        
        if message_type == 'subscribe_signals':
            trader_id = message.get('trader_id')
            # Подписка на сигналы трейдера
            
        elif message_type == 'like_post':
            post_id = message.get('post_id')
            self.social_feed_manager.like_post(post_id, user_id)
            
        elif message_type == 'get_feed':
            feed = await self.get_social_feed(user_id)
            await self._send_to_user(user_id, {
                'type': 'feed_update',
                'feed': feed
            })
            
    async def _notify_followers(self, trader_id: str, notification: Dict[str, Any]):
        """Уведомление подписчиков"""
        # Поиск подписчиков
        followers = [
            follow.follower_id for follow in self.follows.values()
            if follow.following_id == trader_id and follow.is_active
        ]
        
        # Отправка уведомлений
        for follower_id in followers:
            await self._send_to_user(follower_id, notification)
            
    async def _send_to_user(self, user_id: str, message: Dict[str, Any]):
        """Отправка сообщения пользователю через WebSocket"""
        if user_id in self.websocket_connections:
            try:
                await self.websocket_connections[user_id].send(json.dumps(message))
            except Exception as e:
                self.logger.error(f"Failed to send message to {user_id}: {e}")
                # Удаление неактивного соединения
                if user_id in self.websocket_connections:
                    del self.websocket_connections[user_id]

async def main():
    """Основная функция запуска"""
    config = {
        'redis_host': 'localhost',
        'redis_port': 6379,
        'websocket_host': 'localhost',
        'websocket_port': 8765
    }
    
    platform = EnterpriseSocialTradingPlatform(config)
    await platform.start()
    
    print("Enterprise Social Trading Platform started")
    
    try:
        # Создание тестовых данных
        
        # Создание профиля трейдера
        trader_profile = TraderProfile(
            user_id="trader_001",
            username="crypto_master",
            display_name="Crypto Master",
            tier=TraderTier.EXPERT,
            bio="Professional crypto trader with 5+ years experience",
            verified=True,
            win_rate=78.5,
            total_trades=1250,
            avg_profit=2.3,
            max_drawdown=-4.2,
            sharpe_ratio=2.8,
            total_pnl=125000.0,
            allow_copying=True,
            copy_fee_percentage=10.0
        )
        
        await platform.create_trader_profile(trader_profile)
        
        # Создание торгового сигнала
        signal = TradingSignal(
            id=str(uuid.uuid4()),
            trader_id="trader_001",
            signal_type=SignalType.BUY,
            symbol="BTCUSDT",
            price=45000.0,
            quantity=0.1,
            stop_loss=43000.0,
            take_profit=48000.0,
            confidence=85.0,
            reasoning="Strong bullish momentum with volume confirmation"
        )
        
        await platform.publish_signal(signal)
        
        # Создание социального поста
        post = SocialPost(
            id=str(uuid.uuid4()),
            author_id="trader_001",
            post_type=PostType.ANALYSIS,
            title="Bitcoin Market Analysis",
            content="Bitcoin is showing strong bullish signals with increasing volume...",
            tags=["bitcoin", "analysis", "bullish"],
            related_symbols=["BTCUSDT"]
        )
        
        await platform.create_social_post(post)
        
        # Получение рейтинга
        leaderboard = await platform.get_leaderboard()
        print(f"Leaderboard entries: {len(leaderboard)}")
        
        # Непрерывная работа
        print("Platform running... Press Ctrl+C to stop")
        await asyncio.Future()  # Бесконечное ожидание
        
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        await platform.stop()

if __name__ == '__main__':
    asyncio.run(main())