"""
Educational Platform for Phase 5
VR simulations, strategy marketplace, and certification system
"""

import asyncio
import logging
import json
import time
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum
from datetime import datetime, timezone, timedelta
import uuid
import numpy as np
from decimal import Decimal
import warnings
warnings.filterwarnings('ignore')

class CourseLevel(Enum):
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"

class CourseType(Enum):
    THEORY = "theory"
    PRACTICAL = "practical"
    VR_SIMULATION = "vr_simulation"
    LIVE_TRADING = "live_trading"
    STRATEGY_ANALYSIS = "strategy_analysis"

class CertificationLevel(Enum):
    BRONZE = "bronze"
    SILVER = "silver"
    GOLD = "gold"
    PLATINUM = "platinum"
    DIAMOND = "diamond"

class StrategyCategory(Enum):
    SCALPING = "scalping"
    DAY_TRADING = "day_trading"
    SWING_TRADING = "swing_trading"
    POSITION_TRADING = "position_trading"
    ARBITRAGE = "arbitrage"
    ALGORITHMIC = "algorithmic"
    AI_ML = "ai_ml"
    DeFi = "defi"

class StrategyStatus(Enum):
    DRAFT = "draft"
    PENDING_REVIEW = "pending_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    FEATURED = "featured"
    DEPRECATED = "deprecated"

@dataclass
class User:
    """Platform user"""
    user_id: str
    username: str
    email: str
    
    # Profile
    first_name: str = ""
    last_name: str = ""
    avatar_url: str = ""
    bio: str = ""
    
    # Experience
    trading_experience: str = "beginner"  # beginner, intermediate, advanced, expert
    preferred_markets: List[str] = field(default_factory=list)
    
    # Progress
    completed_courses: List[str] = field(default_factory=list)
    certifications: List[str] = field(default_factory=list)
    total_study_hours: float = 0.0
    
    # VR/AR preferences
    vr_enabled: bool = False
    ar_enabled: bool = False
    preferred_vr_platform: str = ""
    
    # Marketplace
    published_strategies: List[str] = field(default_factory=list)
    purchased_strategies: List[str] = field(default_factory=list)
    total_earnings: float = 0.0
    total_spent: float = 0.0
    
    # Timestamps
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_active: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def __post_init__(self):
        if not self.user_id:
            self.user_id = str(uuid.uuid4())

@dataclass
class Course:
    """Educational course"""
    course_id: str
    title: str
    description: str
    
    # Course metadata
    level: CourseLevel
    course_type: CourseType
    category: str
    
    # Content
    modules: List[Dict[str, Any]] = field(default_factory=list)
    duration_hours: float = 0.0
    
    # VR/AR content
    vr_scenes: List[Dict[str, Any]] = field(default_factory=list)
    ar_overlays: List[Dict[str, Any]] = field(default_factory=list)
    
    # Requirements
    prerequisites: List[str] = field(default_factory=list)
    required_certifications: List[str] = field(default_factory=list)
    
    # Pricing
    price: float = 0.0
    currency: str = "USD"
    is_free: bool = True
    
    # Instructor
    instructor_id: str = ""
    instructor_name: str = ""
    
    # Statistics
    enrolled_count: int = 0
    completion_rate: float = 0.0
    average_rating: float = 0.0
    total_ratings: int = 0
    
    # Status
    is_published: bool = False
    is_featured: bool = False
    
    # Timestamps
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def __post_init__(self):
        if not self.course_id:
            self.course_id = str(uuid.uuid4())

@dataclass
class VRSimulation:
    """VR trading simulation"""
    simulation_id: str
    title: str
    description: str
    
    # Simulation parameters
    market_scenario: str
    initial_balance: float = 10000.0
    duration_minutes: int = 60
    
    # Market conditions
    volatility_level: str = "medium"  # low, medium, high, extreme
    trend_direction: str = "sideways"  # bullish, bearish, sideways
    news_events: List[Dict[str, Any]] = field(default_factory=list)
    
    # VR environment
    environment_type: str = "trading_floor"  # trading_floor, home_office, exchange
    visual_complexity: str = "standard"  # minimal, standard, advanced
    
    # Objectives
    learning_objectives: List[str] = field(default_factory=list)
    success_criteria: Dict[str, float] = field(default_factory=dict)
    
    # Difficulty
    difficulty_level: CourseLevel = CourseLevel.BEGINNER
    
    # Statistics
    completion_count: int = 0
    average_score: float = 0.0
    average_duration: float = 0.0
    
    # Timestamps
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def __post_init__(self):
        if not self.simulation_id:
            self.simulation_id = str(uuid.uuid4())

@dataclass
class TradingStrategy:
    """Trading strategy for marketplace"""
    strategy_id: str
    name: str
    description: str
    
    # Strategy details
    category: StrategyCategory
    timeframe: str
    markets: List[str] = field(default_factory=list)
    
    # Performance metrics
    win_rate: float = 0.0
    roi: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    
    # Backtesting results
    backtest_period: str = ""
    total_trades: int = 0
    profitable_trades: int = 0
    
    # Code and configuration
    strategy_code: str = ""
    configuration: Dict[str, Any] = field(default_factory=dict)
    
    # Pricing
    price: float = 0.0
    currency: str = "USD"
    license_type: str = "single_use"  # single_use, unlimited, subscription
    
    # Author
    author_id: str = ""
    author_name: str = ""
    
    # Status
    status: StrategyStatus = StrategyStatus.DRAFT
    
    # Marketplace statistics
    views: int = 0
    downloads: int = 0
    ratings: List[Dict[str, Any]] = field(default_factory=list)
    average_rating: float = 0.0
    
    # Timestamps
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def __post_init__(self):
        if not self.strategy_id:
            self.strategy_id = str(uuid.uuid4())

@dataclass
class Certification:
    """Trading certification"""
    certification_id: str
    name: str
    description: str
    
    # Certification details
    level: CertificationLevel
    category: str
    
    # Requirements
    required_courses: List[str] = field(default_factory=list)
    required_simulations: List[str] = field(default_factory=list)
    minimum_score: float = 80.0
    
    # Exam
    exam_questions: int = 100
    exam_duration_minutes: int = 120
    passing_score: float = 80.0
    
    # Validity
    validity_months: int = 24
    renewal_required: bool = True
    
    # Statistics
    issued_count: int = 0
    pass_rate: float = 0.0
    
    # Timestamps
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def __post_init__(self):
        if not self.certification_id:
            self.certification_id = str(uuid.uuid4())

@dataclass
class UserProgress:
    """User learning progress"""
    user_id: str
    course_id: str
    
    # Progress tracking
    current_module: int = 0
    completion_percentage: float = 0.0
    
    # Time tracking
    time_spent_hours: float = 0.0
    last_accessed: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Scores and assessments
    quiz_scores: List[float] = field(default_factory=list)
    assignment_scores: List[float] = field(default_factory=list)
    overall_score: float = 0.0
    
    # VR simulation results
    simulation_results: List[Dict[str, Any]] = field(default_factory=list)
    
    # Status
    is_completed: bool = False
    completion_date: Optional[datetime] = None
    
    # Timestamps
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

class EducationPlatform:
    """Educational platform with VR simulations and strategy marketplace"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Users and authentication
        self.users: Dict[str, User] = {}
        self.user_sessions: Dict[str, Dict[str, Any]] = {}
        
        # Educational content
        self.courses: Dict[str, Course] = {}
        self.vr_simulations: Dict[str, VRSimulation] = {}
        self.certifications: Dict[str, Certification] = {}
        
        # Progress tracking
        self.user_progress: Dict[str, List[UserProgress]] = {}  # user_id -> [progress]
        
        # Strategy marketplace
        self.strategies: Dict[str, TradingStrategy] = {}
        self.strategy_purchases: Dict[str, List[Dict[str, Any]]] = {}  # user_id -> [purchases]
        
        # Analytics
        self.platform_analytics: Dict[str, Any] = {}
        
        # Initialize platform content
        asyncio.create_task(self._initialize_platform_content())
    
    async def _initialize_platform_content(self):
        """Initialize platform with default content"""
        try:
            # Create default courses
            await self._create_default_courses()
            
            # Create VR simulations
            await self._create_vr_simulations()
            
            # Create certifications
            await self._create_certifications()
            
            # Create sample strategies
            await self._create_sample_strategies()
            
            self.logger.info("Platform content initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Platform initialization failed: {e}")
    
    async def _create_default_courses(self):
        """Create default educational courses"""
        try:
            courses_data = [
                {
                    'title': 'Trading Fundamentals',
                    'description': 'Learn the basics of financial markets and trading',
                    'level': CourseLevel.BEGINNER,
                    'course_type': CourseType.THEORY,
                    'category': 'Fundamentals',
                    'duration_hours': 10.0,
                    'modules': [
                        {'title': 'Introduction to Markets', 'duration': 2.0},
                        {'title': 'Order Types and Execution', 'duration': 2.0},
                        {'title': 'Risk Management Basics', 'duration': 3.0},
                        {'title': 'Technical Analysis Introduction', 'duration': 3.0}
                    ],
                    'is_free': True
                },
                {
                    'title': 'Advanced Technical Analysis',
                    'description': 'Master advanced charting techniques and indicators',
                    'level': CourseLevel.ADVANCED,
                    'course_type': CourseType.PRACTICAL,
                    'category': 'Technical Analysis',
                    'duration_hours': 25.0,
                    'price': 299.0,
                    'is_free': False,
                    'modules': [
                        {'title': 'Advanced Chart Patterns', 'duration': 5.0},
                        {'title': 'Custom Indicators', 'duration': 6.0},
                        {'title': 'Multi-Timeframe Analysis', 'duration': 7.0},
                        {'title': 'Market Structure Analysis', 'duration': 7.0}
                    ]
                },
                {
                    'title': 'Cryptocurrency Trading Mastery',
                    'description': 'Complete guide to crypto trading and DeFi',
                    'level': CourseLevel.INTERMEDIATE,
                    'course_type': CourseType.PRACTICAL,
                    'category': 'Cryptocurrency',
                    'duration_hours': 20.0,
                    'price': 199.0,
                    'is_free': False,
                    'modules': [
                        {'title': 'Crypto Market Fundamentals', 'duration': 4.0},
                        {'title': 'DeFi Protocols and Yield Farming', 'duration': 6.0},
                        {'title': 'NFT Trading Strategies', 'duration': 5.0},
                        {'title': 'Cross-Chain Trading', 'duration': 5.0}
                    ]
                },
                {
                    'title': 'VR Trading Simulation Mastery',
                    'description': 'Master trading in virtual reality environments',
                    'level': CourseLevel.ADVANCED,
                    'course_type': CourseType.VR_SIMULATION,
                    'category': 'VR Trading',
                    'duration_hours': 15.0,
                    'price': 399.0,
                    'is_free': False,
                    'vr_scenes': [
                        {'name': 'NYSE Trading Floor', 'complexity': 'high'},
                        {'name': 'Crypto Exchange Hub', 'complexity': 'medium'},
                        {'name': 'Forex Trading Room', 'complexity': 'high'}
                    ],
                    'modules': [
                        {'title': 'VR Interface Mastery', 'duration': 3.0},
                        {'title': 'Immersive Market Analysis', 'duration': 4.0},
                        {'title': 'Virtual Risk Management', 'duration': 4.0},
                        {'title': 'Multi-Asset VR Trading', 'duration': 4.0}
                    ]
                },
                {
                    'title': 'AI and Machine Learning for Trading',
                    'description': 'Build intelligent trading systems with AI/ML',
                    'level': CourseLevel.EXPERT,
                    'course_type': CourseType.PRACTICAL,
                    'category': 'AI/ML',
                    'duration_hours': 40.0,
                    'price': 599.0,
                    'is_free': False,
                    'modules': [
                        {'title': 'Machine Learning Fundamentals', 'duration': 8.0},
                        {'title': 'Deep Learning for Trading', 'duration': 10.0},
                        {'title': 'Reinforcement Learning Strategies', 'duration': 10.0},
                        {'title': 'Quantum ML Applications', 'duration': 12.0}
                    ]
                }
            ]
            
            for course_data in courses_data:
                course = Course(**course_data)
                course.instructor_name = "PEPER AI Academy"
                self.courses[course.course_id] = course
            
        except Exception as e:
            self.logger.error(f"Default courses creation failed: {e}")
    
    async def _create_vr_simulations(self):
        """Create VR trading simulations"""
        try:
            simulations_data = [
                {
                    'title': 'Market Crash Simulation',
                    'description': 'Navigate through a major market crash scenario',
                    'market_scenario': 'bear_market_crash',
                    'initial_balance': 100000.0,
                    'duration_minutes': 45,
                    'volatility_level': 'extreme',
                    'trend_direction': 'bearish',
                    'difficulty_level': CourseLevel.ADVANCED,
                    'environment_type': 'trading_floor',
                    'learning_objectives': [
                        'Risk management during extreme volatility',
                        'Position sizing in bear markets',
                        'Hedging strategies implementation'
                    ],
                    'success_criteria': {
                        'max_drawdown': 15.0,
                        'minimum_return': -10.0,
                        'risk_score': 8.0
                    }
                },
                {
                    'title': 'Crypto Bull Run Trading',
                    'description': 'Trade during a cryptocurrency bull market',
                    'market_scenario': 'crypto_bull_run',
                    'initial_balance': 50000.0,
                    'duration_minutes': 30,
                    'volatility_level': 'high',
                    'trend_direction': 'bullish',
                    'difficulty_level': CourseLevel.INTERMEDIATE,
                    'environment_type': 'crypto_exchange',
                    'learning_objectives': [
                        'Momentum trading strategies',
                        'Altcoin selection and timing',
                        'Profit taking strategies'
                    ],
                    'success_criteria': {
                        'minimum_return': 25.0,
                        'max_drawdown': 20.0,
                        'trades_count': 10
                    }
                },
                {
                    'title': 'Forex Scalping Challenge',
                    'description': 'Master high-frequency forex scalping',
                    'market_scenario': 'forex_high_frequency',
                    'initial_balance': 25000.0,
                    'duration_minutes': 20,
                    'volatility_level': 'medium',
                    'trend_direction': 'sideways',
                    'difficulty_level': CourseLevel.EXPERT,
                    'environment_type': 'forex_desk',
                    'learning_objectives': [
                        'Scalping technique mastery',
                        'News trading execution',
                        'Spread and commission optimization'
                    ],
                    'success_criteria': {
                        'minimum_return': 5.0,
                        'max_drawdown': 3.0,
                        'win_rate': 70.0
                    }
                },
                {
                    'title': 'DeFi Yield Farming Simulation',
                    'description': 'Optimize yield farming strategies in DeFi',
                    'market_scenario': 'defi_yield_farming',
                    'initial_balance': 75000.0,
                    'duration_minutes': 60,
                    'volatility_level': 'high',
                    'trend_direction': 'sideways',
                    'difficulty_level': CourseLevel.ADVANCED,
                    'environment_type': 'defi_dashboard',
                    'learning_objectives': [
                        'Liquidity pool optimization',
                        'Impermanent loss management',
                        'Cross-protocol strategies'
                    ],
                    'success_criteria': {
                        'minimum_apy': 15.0,
                        'max_impermanent_loss': 5.0,
                        'gas_efficiency': 90.0
                    }
                }
            ]
            
            for sim_data in simulations_data:
                simulation = VRSimulation(**sim_data)
                self.vr_simulations[simulation.simulation_id] = simulation
            
        except Exception as e:
            self.logger.error(f"VR simulations creation failed: {e}")
    
    async def _create_certifications(self):
        """Create trading certifications"""
        try:
            certifications_data = [
                {
                    'name': 'Certified Trading Fundamentals',
                    'description': 'Basic trading knowledge certification',
                    'level': CertificationLevel.BRONZE,
                    'category': 'Fundamentals',
                    'exam_questions': 50,
                    'exam_duration_minutes': 60,
                    'passing_score': 75.0,
                    'validity_months': 12
                },
                {
                    'name': 'Advanced Technical Analyst',
                    'description': 'Advanced technical analysis certification',
                    'level': CertificationLevel.SILVER,
                    'category': 'Technical Analysis',
                    'exam_questions': 75,
                    'exam_duration_minutes': 90,
                    'passing_score': 80.0,
                    'validity_months': 18
                },
                {
                    'name': 'Cryptocurrency Trading Expert',
                    'description': 'Expert-level crypto trading certification',
                    'level': CertificationLevel.GOLD,
                    'category': 'Cryptocurrency',
                    'exam_questions': 100,
                    'exam_duration_minutes': 120,
                    'passing_score': 85.0,
                    'validity_months': 24
                },
                {
                    'name': 'VR Trading Specialist',
                    'description': 'Virtual reality trading specialist certification',
                    'level': CertificationLevel.PLATINUM,
                    'category': 'VR Trading',
                    'exam_questions': 80,
                    'exam_duration_minutes': 100,
                    'passing_score': 88.0,
                    'validity_months': 24
                },
                {
                    'name': 'AI Trading Systems Master',
                    'description': 'Master-level AI trading systems certification',
                    'level': CertificationLevel.DIAMOND,
                    'category': 'AI/ML',
                    'exam_questions': 150,
                    'exam_duration_minutes': 180,
                    'passing_score': 90.0,
                    'validity_months': 36
                }
            ]
            
            for cert_data in certifications_data:
                certification = Certification(**cert_data)
                self.certifications[certification.certification_id] = certification
            
        except Exception as e:
            self.logger.error(f"Certifications creation failed: {e}")
    
    async def _create_sample_strategies(self):
        """Create sample trading strategies for marketplace"""
        try:
            strategies_data = [
                {
                    'name': 'AI-Powered Momentum Strategy',
                    'description': 'Machine learning-based momentum trading strategy',
                    'category': StrategyCategory.AI_ML,
                    'timeframe': '1h',
                    'markets': ['BTCUSDT', 'ETHUSDT', 'ADAUSDT'],
                    'win_rate': 78.5,
                    'roi': 45.2,
                    'max_drawdown': 8.3,
                    'sharpe_ratio': 2.85,
                    'price': 299.0,
                    'status': StrategyStatus.FEATURED,
                    'author_name': 'PEPER AI Lab'
                },
                {
                    'name': 'DeFi Arbitrage Bot',
                    'description': 'Cross-DEX arbitrage opportunities scanner',
                    'category': StrategyCategory.ARBITRAGE,
                    'timeframe': '1m',
                    'markets': ['Uniswap', 'SushiSwap', 'PancakeSwap'],
                    'win_rate': 92.1,
                    'roi': 28.7,
                    'max_drawdown': 2.1,
                    'sharpe_ratio': 4.12,
                    'price': 499.0,
                    'status': StrategyStatus.APPROVED,
                    'author_name': 'DeFi Strategies Inc'
                },
                {
                    'name': 'Scalping Master Pro',
                    'description': 'High-frequency scalping strategy for forex',
                    'category': StrategyCategory.SCALPING,
                    'timeframe': '1m',
                    'markets': ['EURUSD', 'GBPUSD', 'USDJPY'],
                    'win_rate': 68.9,
                    'roi': 32.4,
                    'max_drawdown': 5.7,
                    'sharpe_ratio': 2.34,
                    'price': 199.0,
                    'status': StrategyStatus.APPROVED,
                    'author_name': 'Forex Masters'
                },
                {
                    'name': 'Swing Trading Algorithm',
                    'description': 'Multi-timeframe swing trading system',
                    'category': StrategyCategory.SWING_TRADING,
                    'timeframe': '4h',
                    'markets': ['SPY', 'QQQ', 'IWM'],
                    'win_rate': 71.2,
                    'roi': 38.9,
                    'max_drawdown': 12.4,
                    'sharpe_ratio': 1.98,
                    'price': 149.0,
                    'status': StrategyStatus.APPROVED,
                    'author_name': 'Stock Swing Pro'
                },
                {
                    'name': 'Quantum Trading Engine',
                    'description': 'Quantum computing-enhanced trading algorithm',
                    'category': StrategyCategory.AI_ML,
                    'timeframe': '15m',
                    'markets': ['BTCUSDT', 'ETHUSDT', 'SOLUSDT'],
                    'win_rate': 84.3,
                    'roi': 67.8,
                    'max_drawdown': 6.2,
                    'sharpe_ratio': 3.45,
                    'price': 999.0,
                    'status': StrategyStatus.FEATURED,
                    'author_name': 'Quantum Labs'
                }
            ]
            
            for strategy_data in strategies_data:
                strategy = TradingStrategy(**strategy_data)
                strategy.strategy_code = f"# {strategy.name} Implementation\n# Advanced trading algorithm\npass"
                self.strategies[strategy.strategy_id] = strategy
            
        except Exception as e:
            self.logger.error(f"Sample strategies creation failed: {e}")
    
    async def register_user(self, username: str, email: str, password: str,
                          trading_experience: str = "beginner") -> Optional[str]:
        """Register new user"""
        try:
            # Check if user already exists
            for user in self.users.values():
                if user.username == username or user.email == email:
                    return None
            
            # Create new user
            user = User(
                user_id="",
                username=username,
                email=email,
                trading_experience=trading_experience
            )
            
            self.users[user.user_id] = user
            
            # Initialize user progress tracking
            self.user_progress[user.user_id] = []
            self.strategy_purchases[user.user_id] = []
            
            self.logger.info(f"User registered: {username}")
            return user.user_id
            
        except Exception as e:
            self.logger.error(f"User registration failed: {e}")
            return None
    
    async def enroll_in_course(self, user_id: str, course_id: str) -> bool:
        """Enroll user in course"""
        try:
            if user_id not in self.users or course_id not in self.courses:
                return False
            
            user = self.users[user_id]
            course = self.courses[course_id]
            
            # Check if already enrolled
            for progress in self.user_progress[user_id]:
                if progress.course_id == course_id:
                    return False
            
            # Create progress tracking
            progress = UserProgress(
                user_id=user_id,
                course_id=course_id
            )
            
            self.user_progress[user_id].append(progress)
            
            # Update course statistics
            course.enrolled_count += 1
            
            # Add to user's completed courses if not already there
            if course_id not in user.completed_courses:
                user.completed_courses.append(course_id)
            
            self.logger.info(f"User {user.username} enrolled in {course.title}")
            return True
            
        except Exception as e:
            self.logger.error(f"Course enrollment failed: {e}")
            return False
    
    async def start_vr_simulation(self, user_id: str, simulation_id: str) -> Optional[Dict[str, Any]]:
        """Start VR simulation session"""
        try:
            if user_id not in self.users or simulation_id not in self.vr_simulations:
                return None
            
            user = self.users[user_id]
            simulation = self.vr_simulations[simulation_id]
            
            # Check VR capability
            if not user.vr_enabled:
                return None
            
            # Create simulation session
            session_id = str(uuid.uuid4())
            session_data = {
                'session_id': session_id,
                'user_id': user_id,
                'simulation_id': simulation_id,
                'start_time': datetime.now(timezone.utc),
                'initial_balance': simulation.initial_balance,
                'current_balance': simulation.initial_balance,
                'trades': [],
                'status': 'active',
                'environment': {
                    'type': simulation.environment_type,
                    'volatility': simulation.volatility_level,
                    'trend': simulation.trend_direction,
                    'duration_minutes': simulation.duration_minutes
                }
            }
            
            # Store session
            if 'vr_sessions' not in self.user_sessions:
                self.user_sessions['vr_sessions'] = {}
            self.user_sessions['vr_sessions'][session_id] = session_data
            
            self.logger.info(f"VR simulation started: {simulation.title} for {user.username}")
            return session_data
            
        except Exception as e:
            self.logger.error(f"VR simulation start failed: {e}")
            return None
    
    async def purchase_strategy(self, user_id: str, strategy_id: str) -> Optional[str]:
        """Purchase trading strategy"""
        try:
            if user_id not in self.users or strategy_id not in self.strategies:
                return None
            
            user = self.users[user_id]
            strategy = self.strategies[strategy_id]
            
            # Check if strategy is approved
            if strategy.status not in [StrategyStatus.APPROVED, StrategyStatus.FEATURED]:
                return None
            
            # Check if already purchased
            for purchase in self.strategy_purchases[user_id]:
                if purchase['strategy_id'] == strategy_id:
                    return None
            
            # Create purchase record
            purchase_id = str(uuid.uuid4())
            purchase_data = {
                'purchase_id': purchase_id,
                'strategy_id': strategy_id,
                'strategy_name': strategy.name,
                'price': strategy.price,
                'currency': strategy.currency,
                'license_type': strategy.license_type,
                'purchased_at': datetime.now(timezone.utc),
                'author_id': strategy.author_id
            }
            
            # Add to user purchases
            self.strategy_purchases[user_id].append(purchase_data)
            user.purchased_strategies.append(strategy_id)
            user.total_spent += strategy.price
            
            # Update strategy statistics
            strategy.downloads += 1
            
            # Update author earnings (if author exists in users)
            if strategy.author_id in self.users:
                author = self.users[strategy.author_id]
                author.total_earnings += strategy.price * 0.7  # 70% to author, 30% platform fee
            
            self.logger.info(f"Strategy purchased: {strategy.name} by {user.username}")
            return purchase_id
            
        except Exception as e:
            self.logger.error(f"Strategy purchase failed: {e}")
            return None
    
    async def submit_strategy(self, user_id: str, strategy_data: Dict[str, Any]) -> Optional[str]:
        """Submit strategy to marketplace"""
        try:
            if user_id not in self.users:
                return None
            
            user = self.users[user_id]
            
            # Create strategy
            strategy = TradingStrategy(
                strategy_id="",
                author_id=user_id,
                author_name=user.username,
                status=StrategyStatus.PENDING_REVIEW,
                **strategy_data
            )
            
            self.strategies[strategy.strategy_id] = strategy
            user.published_strategies.append(strategy.strategy_id)
            
            self.logger.info(f"Strategy submitted: {strategy.name} by {user.username}")
            return strategy.strategy_id
            
        except Exception as e:
            self.logger.error(f"Strategy submission failed: {e}")
            return None
    
    async def take_certification_exam(self, user_id: str, certification_id: str) -> Optional[Dict[str, Any]]:
        """Take certification exam"""
        try:
            if user_id not in self.users or certification_id not in self.certifications:
                return None
            
            user = self.users[user_id]
            certification = self.certifications[certification_id]
            
            # Check if already certified
            if certification_id in user.certifications:
                return None
            
            # Simulate exam (in real implementation, this would be an actual exam)
            exam_score = np.random.uniform(70, 95)  # Random score for simulation
            
            exam_result = {
                'exam_id': str(uuid.uuid4()),
                'user_id': user_id,
                'certification_id': certification_id,
                'score': exam_score,
                'passing_score': certification.passing_score,
                'passed': exam_score >= certification.passing_score,
                'taken_at': datetime.now(timezone.utc),
                'valid_until': datetime.now(timezone.utc) + timedelta(days=certification.validity_months * 30)
            }
            
            if exam_result['passed']:
                user.certifications.append(certification_id)
                certification.issued_count += 1
                
                self.logger.info(f"Certification earned: {certification.name} by {user.username}")
            
            return exam_result
            
        except Exception as e:
            self.logger.error(f"Certification exam failed: {e}")
            return None
    
    async def get_user_dashboard(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user dashboard data"""
        try:
            if user_id not in self.users:
                return None
            
            user = self.users[user_id]
            user_progress_list = self.user_progress.get(user_id, [])
            
            # Calculate progress statistics
            total_courses = len(user_progress_list)
            completed_courses = sum(1 for p in user_progress_list if p.is_completed)
            total_study_hours = sum(p.time_spent_hours for p in user_progress_list)
            
            # Get recent activity
            recent_courses = sorted(user_progress_list, key=lambda p: p.last_accessed, reverse=True)[:5]
            
            # Get purchased strategies
            purchased_strategies = self.strategy_purchases.get(user_id, [])
            
            dashboard_data = {
                'user_info': {
                    'username': user.username,
                    'trading_experience': user.trading_experience,
                    'total_study_hours': total_study_hours,
                    'certifications_count': len(user.certifications),
                    'strategies_published': len(user.published_strategies),
                    'strategies_purchased': len(purchased_strategies),
                    'total_earnings': user.total_earnings,
                    'total_spent': user.total_spent
                },
                'learning_progress': {
                    'total_courses': total_courses,
                    'completed_courses': completed_courses,
                    'completion_rate': (completed_courses / total_courses * 100) if total_courses > 0 else 0,
                    'recent_courses': [
                        {
                            'course_id': p.course_id,
                            'course_title': self.courses[p.course_id].title if p.course_id in self.courses else 'Unknown',
                            'completion_percentage': p.completion_percentage,
                            'last_accessed': p.last_accessed.isoformat()
                        }
                        for p in recent_courses
                    ]
                },
                'certifications': [
                    {
                        'certification_id': cert_id,
                        'name': self.certifications[cert_id].name if cert_id in self.certifications else 'Unknown',
                        'level': self.certifications[cert_id].level.value if cert_id in self.certifications else 'unknown'
                    }
                    for cert_id in user.certifications
                ],
                'marketplace_activity': {
                    'published_strategies': len(user.published_strategies),
                    'purchased_strategies': len(purchased_strategies),
                    'total_earnings': user.total_earnings,
                    'total_spent': user.total_spent
                }
            }
            
            return dashboard_data
            
        except Exception as e:
            self.logger.error(f"Get user dashboard failed: {e}")
            return None
    
    async def get_course_catalog(self, level: Optional[CourseLevel] = None,
                               course_type: Optional[CourseType] = None,
                               category: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get course catalog"""
        try:
            catalog = []
            
            for course in self.courses.values():
                # Apply filters
                if level and course.level != level:
                    continue
                if course_type and course.course_type != course_type:
                    continue
                if category and course.category.lower() != category.lower():
                    continue
                
                catalog.append({
                    'course_id': course.course_id,
                    'title': course.title,
                    'description': course.description,
                    'level': course.level.value,
                    'course_type': course.course_type.value,
                    'category': course.category,
                    'duration_hours': course.duration_hours,
                    'price': course.price,
                    'currency': course.currency,
                    'is_free': course.is_free,
                    'instructor_name': course.instructor_name,
                    'enrolled_count': course.enrolled_count,
                    'average_rating': course.average_rating,
                    'is_featured': course.is_featured,
                    'has_vr_content': len(course.vr_scenes) > 0,
                    'has_ar_content': len(course.ar_overlays) > 0
                })
            
            # Sort by featured first, then by rating
            catalog.sort(key=lambda c: (not c['is_featured'], -c['average_rating']))
            
            return catalog
            
        except Exception as e:
            self.logger.error(f"Get course catalog failed: {e}")
            return []
    
    async def get_strategy_marketplace(self, category: Optional[StrategyCategory] = None,
                                     min_win_rate: Optional[float] = None,
                                     max_price: Optional[float] = None) -> List[Dict[str, Any]]:
        """Get strategy marketplace listings"""
        try:
            marketplace = []
            
            for strategy in self.strategies.values():
                # Only show approved and featured strategies
                if strategy.status not in [StrategyStatus.APPROVED, StrategyStatus.FEATURED]:
                    continue
                
                # Apply filters
                if category and strategy.category != category:
                    continue
                if min_win_rate and strategy.win_rate < min_win_rate:
                    continue
                if max_price and strategy.price > max_price:
                    continue
                
                marketplace.append({
                    'strategy_id': strategy.strategy_id,
                    'name': strategy.name,
                    'description': strategy.description,
                    'category': strategy.category.value,
                    'timeframe': strategy.timeframe,
                    'markets': strategy.markets,
                    'win_rate': strategy.win_rate,
                    'roi': strategy.roi,
                    'max_drawdown': strategy.max_drawdown,
                    'sharpe_ratio': strategy.sharpe_ratio,
                    'price': strategy.price,
                    'currency': strategy.currency,
                    'license_type': strategy.license_type,
                    'author_name': strategy.author_name,
                    'views': strategy.views,
                    'downloads': strategy.downloads,
                    'average_rating': strategy.average_rating,
                    'is_featured': strategy.status == StrategyStatus.FEATURED
                })
            
            # Sort by featured first, then by performance
            marketplace.sort(key=lambda s: (not s['is_featured'], -s['roi']))
            
            return marketplace
            
        except Exception as e:
            self.logger.error(f"Get strategy marketplace failed: {e}")
            return []
    
    async def get_platform_analytics(self) -> Dict[str, Any]:
        """Get platform analytics"""
        try:
            # User analytics
            user_stats = {
                'total_users': len(self.users),
                'by_experience': {},
                'active_users_24h': 0,
                'vr_enabled_users': sum(1 for user in self.users.values() if user.vr_enabled),
                'ar_enabled_users': sum(1 for user in self.users.values() if user.ar_enabled)
            }
            
            for user in self.users.values():
                exp = user.trading_experience
                if exp not in user_stats['by_experience']:
                    user_stats['by_experience'][exp] = 0
                user_stats['by_experience'][exp] += 1
                
                # Check if active in last 24 hours
                if (datetime.now(timezone.utc) - user.last_active).total_seconds() < 86400:
                    user_stats['active_users_24h'] += 1
            
            # Course analytics
            course_stats = {
                'total_courses': len(self.courses),
                'by_level': {},
                'by_type': {},
                'total_enrollments': sum(course.enrolled_count for course in self.courses.values()),
                'vr_courses': sum(1 for course in self.courses.values() if len(course.vr_scenes) > 0),
                'ar_courses': sum(1 for course in self.courses.values() if len(course.ar_overlays) > 0)
            }
            
            for course in self.courses.values():
                # By level
                level = course.level.value
                if level not in course_stats['by_level']:
                    course_stats['by_level'][level] = 0
                course_stats['by_level'][level] += 1
                
                # By type
                course_type = course.course_type.value
                if course_type not in course_stats['by_type']:
                    course_stats['by_type'][course_type] = 0
                course_stats['by_type'][course_type] += 1
            
            # VR simulation analytics
            vr_stats = {
                'total_simulations': len(self.vr_simulations),
                'total_completions': sum(sim.completion_count for sim in self.vr_simulations.values()),
                'average_score': np.mean([sim.average_score for sim in self.vr_simulations.values() if sim.average_score > 0]) if self.vr_simulations else 0
            }
            
            # Strategy marketplace analytics
            strategy_stats = {
                'total_strategies': len(self.strategies),
                'approved_strategies': sum(1 for s in self.strategies.values() if s.status == StrategyStatus.APPROVED),
                'featured_strategies': sum(1 for s in self.strategies.values() if s.status == StrategyStatus.FEATURED),
                'total_downloads': sum(s.downloads for s in self.strategies.values()),
                'total_revenue': sum(len(purchases) * 0.3 for purchases in self.strategy_purchases.values()),  # 30% platform fee
                'by_category': {}
            }
            
            for strategy in self.strategies.values():
                category = strategy.category.value
                if category not in strategy_stats['by_category']:
                    strategy_stats['by_category'][category] = 0
                strategy_stats['by_category'][category] += 1
            
            # Certification analytics
            cert_stats = {
                'total_certifications': len(self.certifications),
                'total_issued': sum(cert.issued_count for cert in self.certifications.values()),
                'by_level': {}
            }
            
            for cert in self.certifications.values():
                level = cert.level.value
                if level not in cert_stats['by_level']:
                    cert_stats['by_level'][level] = 0
                cert_stats['by_level'][level] += cert.issued_count
            
            return {
                'users': user_stats,
                'courses': course_stats,
                'vr_simulations': vr_stats,
                'strategy_marketplace': strategy_stats,
                'certifications': cert_stats,
                'generated_at': datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Get platform analytics failed: {e}")
            return {}

# Example usage
async def main():
    """
    Example usage of Education Platform
    """
    print("🎓 Education Platform - Phase 5 Testing")
    print("=" * 50)
    
    # Initialize platform
    platform = EducationPlatform()
    
    # Wait for initialization
    await asyncio.sleep(3)
    
    # Register test users
    print("\n👥 Registering Test Users:")
    
    test_users = [
        ('alice_trader', 'alice@example.com', 'beginner'),
        ('bob_expert', 'bob@example.com', 'expert'),
        ('charlie_vr', 'charlie@example.com', 'intermediate')
    ]
    
    user_ids = []
    for username, email, experience in test_users:
        user_id = await platform.register_user(username, email, 'password123', experience)
        if user_id:
            user_ids.append(user_id)
            print(f"  ✅ {username} registered")
            
            # Enable VR for charlie
            if username == 'charlie_vr':
                platform.users[user_id].vr_enabled = True
    
    # Test course enrollment
    print(f"\n📚 Testing Course Enrollment:")
    
    course_catalog = await platform.get_course_catalog()
    print(f"  Available courses: {len(course_catalog)}")
    
    if user_ids and course_catalog:
        # Enroll first user in first course
        success = await platform.enroll_in_course(user_ids[0], course_catalog[0]['course_id'])
        print(f"  Enrollment: {'✅' if success else '❌'}")
    
    # Test VR simulation
    print(f"\n🥽 Testing VR Simulation:")
    
    vr_simulations = list(platform.vr_simulations.values())
    if user_ids and vr_simulations:
        # Find VR-enabled user
        vr_user_id = None
        for user_id in user_ids:
            if platform.users[user_id].vr_enabled:
                vr_user_id = user_id
                break
        
        if vr_user_id:
            session = await platform.start_vr_simulation(vr_user_id, vr_simulations[0].simulation_id)
            if session:
                print(f"  VR Session started: {session['session_id']}")
                print(f"  Simulation: {vr_simulations[0].title}")
            else:
                print("  ❌ VR session failed to start")
        else:
            print("  ❌ No VR-enabled users found")
    
    # Test strategy marketplace
    print(f"\n🛒 Testing Strategy Marketplace:")
    
    marketplace = await platform.get_strategy_marketplace()
    print(f"  Available strategies: {len(marketplace)}")
    
    if user_ids and marketplace:
        # Purchase first strategy
        purchase_id = await platform.purchase_strategy(user_ids[0], marketplace[0]['strategy_id'])
        if purchase_id:
            print(f"  Strategy purchased: {purchase_id}")
            print(f"  Strategy: {marketplace[0]['name']}")
        else:
            print("  ❌ Strategy purchase failed")
    
    # Test strategy submission
    print(f"\n📝 Testing Strategy Submission:")
    
    if user_ids:
        strategy_data = {
            'name': 'Test Moving Average Strategy',
            'description': 'Simple moving average crossover strategy',
            'category': StrategyCategory.ALGORITHMIC,
            'timeframe': '1h',
            'markets': ['BTCUSDT'],
            'win_rate': 65.0,
            'roi': 25.0,
            'max_drawdown': 10.0,
            'sharpe_ratio': 1.8,
            'price': 99.0,
            'strategy_code': 'def strategy(): pass'
        }
        
        strategy_id = await platform.submit_strategy(user_ids[1], strategy_data)
        if strategy_id:
            print(f"  Strategy submitted: {strategy_id}")
        else:
            print("  ❌ Strategy submission failed")
    
    # Test certification exam
    print(f"\n🏆 Testing Certification Exam:")
    
    certifications = list(platform.certifications.values())
    if user_ids and certifications:
        exam_result = await platform.take_certification_exam(user_ids[0], certifications[0].certification_id)
        if exam_result:
            print(f"  Exam taken: {exam_result['passed']}")
            print(f"  Score: {exam_result['score']:.1f}%")
            print(f"  Certification: {certifications[0].name}")
        else:
            print("  ❌ Certification exam failed")
    
    # Test user dashboard
    print(f"\n📊 Testing User Dashboard:")
    
    if user_ids:
        dashboard = await platform.get_user_dashboard(user_ids[0])
        if dashboard:
            print(f"  User: {dashboard['user_info']['username']}")
            print(f"  Study hours: {dashboard['user_info']['total_study_hours']}")
            print(f"  Certifications: {dashboard['user_info']['certifications_count']}")
            print(f"  Completion rate: {dashboard['learning_progress']['completion_rate']:.1f}%")
        else:
            print("  ❌ Dashboard retrieval failed")
    
    # Get platform analytics
    print(f"\n📈 Platform Analytics:")
    
    analytics = await platform.get_platform_analytics()
    if analytics:
        print(f"  Total users: {analytics['users']['total_users']}")
        print(f"  Total courses: {analytics['courses']['total_courses']}")
        print(f"  VR simulations: {analytics['vr_simulations']['total_simulations']}")
        print(f"  Marketplace strategies: {analytics['strategy_marketplace']['total_strategies']}")
        print(f"  Certifications issued: {analytics['certifications']['total_issued']}")
        print(f"  VR-enabled users: {analytics['users']['vr_enabled_users']}")
    
    print(f"\n✅ Education Platform testing completed!")

if __name__ == "__main__":
    asyncio.run(main())