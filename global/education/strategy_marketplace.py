"""
Strategy Marketplace for Phase 5
Advanced trading strategy marketplace with AI validation and performance tracking
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

class StrategyType(Enum):
    MANUAL = "manual"
    ALGORITHMIC = "algorithmic"
    AI_ML = "ai_ml"
    HYBRID = "hybrid"

class ValidationStatus(Enum):
    PENDING = "pending"
    VALIDATING = "validating"
    PASSED = "passed"
    FAILED = "failed"
    NEEDS_REVIEW = "needs_review"

class PerformanceGrade(Enum):
    S_PLUS = "s_plus"  # 95%+
    S = "s"            # 90-94%
    A_PLUS = "a_plus"  # 85-89%
    A = "a"            # 80-84%
    B_PLUS = "b_plus"  # 75-79%
    B = "b"            # 70-74%
    C = "c"            # 60-69%
    D = "d"            # 50-59%
    F = "f"            # <50%

class MarketCondition(Enum):
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"
    LOW_VOLATILITY = "low_volatility"

class RiskLevel(Enum):
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"

@dataclass
class StrategyMetrics:
    """Strategy performance metrics"""
    # Basic metrics
    total_return: float = 0.0
    annual_return: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    
    # Risk metrics
    max_drawdown: float = 0.0
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    
    # Trade statistics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    average_win: float = 0.0
    average_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    
    # Advanced metrics
    information_ratio: float = 0.0
    treynor_ratio: float = 0.0
    jensen_alpha: float = 0.0
    beta: float = 0.0
    
    # Market condition performance
    bull_market_return: float = 0.0
    bear_market_return: float = 0.0
    sideways_market_return: float = 0.0
    
    # Time-based metrics
    best_month: float = 0.0
    worst_month: float = 0.0
    positive_months: int = 0
    negative_months: int = 0

@dataclass
class BacktestResult:
    """Backtest result data"""
    backtest_id: str
    strategy_id: str
    
    # Test parameters
    start_date: datetime
    end_date: datetime
    initial_capital: float
    
    # Results
    final_capital: float
    metrics: StrategyMetrics
    
    # Detailed data
    equity_curve: List[Dict[str, Any]] = field(default_factory=list)
    trade_log: List[Dict[str, Any]] = field(default_factory=list)
    monthly_returns: List[float] = field(default_factory=list)
    
    # Validation
    validation_status: ValidationStatus = ValidationStatus.PENDING
    validation_notes: str = ""
    
    # Performance grade
    performance_grade: PerformanceGrade = PerformanceGrade.F
    
    # Timestamps
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def __post_init__(self):
        if not self.backtest_id:
            self.backtest_id = str(uuid.uuid4())

@dataclass
class StrategyReview:
    """Strategy review and rating"""
    review_id: str
    strategy_id: str
    user_id: str
    
    # Review content
    rating: float  # 1-5 stars
    title: str
    content: str
    
    # Performance verification
    verified_performance: bool = False
    user_results: Optional[StrategyMetrics] = None
    
    # Helpfulness
    helpful_votes: int = 0
    total_votes: int = 0
    
    # Status
    is_verified_purchase: bool = False
    is_featured: bool = False
    
    # Timestamps
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def __post_init__(self):
        if not self.review_id:
            self.review_id = str(uuid.uuid4())

@dataclass
class StrategyLicense:
    """Strategy licensing information"""
    license_id: str
    strategy_id: str
    user_id: str
    
    # License details
    license_type: str  # single_use, unlimited, subscription
    purchase_price: float
    currency: str = "USD"
    
    # Usage limits
    max_accounts: int = 1
    max_capital: float = 0.0  # 0 = unlimited
    
    # Subscription details (if applicable)
    subscription_start: Optional[datetime] = None
    subscription_end: Optional[datetime] = None
    auto_renewal: bool = False
    
    # Status
    is_active: bool = True
    is_transferable: bool = False
    
    # Usage tracking
    accounts_used: int = 0
    capital_deployed: float = 0.0
    
    # Timestamps
    purchased_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def __post_init__(self):
        if not self.license_id:
            self.license_id = str(uuid.uuid4())

@dataclass
class StrategySignal:
    """Trading signal from strategy"""
    signal_id: str
    strategy_id: str
    
    # Signal details
    symbol: str
    action: str  # buy, sell, hold
    quantity: float
    price: float
    
    # Signal metadata
    confidence: float  # 0-1
    risk_level: RiskLevel
    timeframe: str
    
    # Stop loss and take profit
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    
    # Signal reasoning
    reasoning: str = ""
    technical_indicators: Dict[str, float] = field(default_factory=dict)
    
    # Timestamps
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: Optional[datetime] = None
    
    # Performance tracking
    executed: bool = False
    execution_price: Optional[float] = None
    result_pnl: Optional[float] = None
    
    def __post_init__(self):
        if not self.signal_id:
            self.signal_id = str(uuid.uuid4())

class StrategyValidator:
    """AI-powered strategy validation system"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Validation criteria
        self.min_backtest_period_days = 365
        self.min_trades = 100
        self.max_drawdown_threshold = 30.0
        self.min_sharpe_ratio = 1.0
        
        # Performance grading thresholds
        self.grade_thresholds = {
            PerformanceGrade.S_PLUS: {'sharpe': 3.5, 'return': 50.0, 'drawdown': 5.0},
            PerformanceGrade.S: {'sharpe': 3.0, 'return': 40.0, 'drawdown': 8.0},
            PerformanceGrade.A_PLUS: {'sharpe': 2.5, 'return': 30.0, 'drawdown': 12.0},
            PerformanceGrade.A: {'sharpe': 2.0, 'return': 25.0, 'drawdown': 15.0},
            PerformanceGrade.B_PLUS: {'sharpe': 1.5, 'return': 20.0, 'drawdown': 20.0},
            PerformanceGrade.B: {'sharpe': 1.0, 'return': 15.0, 'drawdown': 25.0},
            PerformanceGrade.C: {'sharpe': 0.5, 'return': 10.0, 'drawdown': 30.0},
            PerformanceGrade.D: {'sharpe': 0.0, 'return': 5.0, 'drawdown': 40.0}
        }
    
    async def validate_strategy(self, strategy_id: str, backtest_result: BacktestResult) -> ValidationStatus:
        """Validate strategy performance"""
        try:
            metrics = backtest_result.metrics
            
            # Check minimum requirements
            if not self._check_minimum_requirements(backtest_result):
                return ValidationStatus.FAILED
            
            # Check for suspicious patterns
            if self._detect_suspicious_patterns(backtest_result):
                return ValidationStatus.NEEDS_REVIEW
            
            # Validate performance consistency
            if not self._validate_performance_consistency(backtest_result):
                return ValidationStatus.NEEDS_REVIEW
            
            # All checks passed
            return ValidationStatus.PASSED
            
        except Exception as e:
            self.logger.error(f"Strategy validation failed: {e}")
            return ValidationStatus.FAILED
    
    def _check_minimum_requirements(self, backtest_result: BacktestResult) -> bool:
        """Check minimum validation requirements"""
        try:
            metrics = backtest_result.metrics
            
            # Check backtest period
            period_days = (backtest_result.end_date - backtest_result.start_date).days
            if period_days < self.min_backtest_period_days:
                return False
            
            # Check minimum trades
            if metrics.total_trades < self.min_trades:
                return False
            
            # Check maximum drawdown
            if metrics.max_drawdown > self.max_drawdown_threshold:
                return False
            
            # Check minimum Sharpe ratio
            if metrics.sharpe_ratio < self.min_sharpe_ratio:
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Minimum requirements check failed: {e}")
            return False
    
    def _detect_suspicious_patterns(self, backtest_result: BacktestResult) -> bool:
        """Detect suspicious patterns in backtest results"""
        try:
            metrics = backtest_result.metrics
            
            # Check for unrealistic win rate
            if metrics.win_rate > 95.0:
                return True
            
            # Check for unrealistic returns
            if metrics.annual_return > 200.0:
                return True
            
            # Check for unrealistic Sharpe ratio
            if metrics.sharpe_ratio > 5.0:
                return True
            
            # Check for curve fitting (too perfect results)
            if (metrics.win_rate > 90.0 and 
                metrics.max_drawdown < 2.0 and 
                metrics.sharpe_ratio > 4.0):
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Suspicious pattern detection failed: {e}")
            return True
    
    def _validate_performance_consistency(self, backtest_result: BacktestResult) -> bool:
        """Validate performance consistency across different periods"""
        try:
            # Check monthly return consistency
            monthly_returns = backtest_result.monthly_returns
            if len(monthly_returns) < 12:
                return True  # Not enough data
            
            # Calculate monthly return statistics
            monthly_std = np.std(monthly_returns)
            monthly_mean = np.mean(monthly_returns)
            
            # Check for excessive volatility
            if monthly_std > 20.0:  # 20% monthly volatility threshold
                return False
            
            # Check for consistent performance
            positive_months = sum(1 for r in monthly_returns if r > 0)
            consistency_ratio = positive_months / len(monthly_returns)
            
            if consistency_ratio < 0.4:  # Less than 40% positive months
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Performance consistency validation failed: {e}")
            return False
    
    def calculate_performance_grade(self, metrics: StrategyMetrics) -> PerformanceGrade:
        """Calculate performance grade based on metrics"""
        try:
            # Calculate composite score
            score = 0
            
            # Sharpe ratio component (40% weight)
            sharpe_score = min(metrics.sharpe_ratio / 4.0, 1.0) * 40
            score += sharpe_score
            
            # Annual return component (30% weight)
            return_score = min(metrics.annual_return / 50.0, 1.0) * 30
            score += return_score
            
            # Drawdown component (20% weight) - inverted
            drawdown_score = max(0, (30.0 - metrics.max_drawdown) / 30.0) * 20
            score += drawdown_score
            
            # Win rate component (10% weight)
            winrate_score = (metrics.win_rate / 100.0) * 10
            score += winrate_score
            
            # Determine grade based on score
            if score >= 95:
                return PerformanceGrade.S_PLUS
            elif score >= 90:
                return PerformanceGrade.S
            elif score >= 85:
                return PerformanceGrade.A_PLUS
            elif score >= 80:
                return PerformanceGrade.A
            elif score >= 75:
                return PerformanceGrade.B_PLUS
            elif score >= 70:
                return PerformanceGrade.B
            elif score >= 60:
                return PerformanceGrade.C
            elif score >= 50:
                return PerformanceGrade.D
            else:
                return PerformanceGrade.F
                
        except Exception as e:
            self.logger.error(f"Performance grade calculation failed: {e}")
            return PerformanceGrade.F

class StrategyMarketplace:
    """Advanced strategy marketplace with AI validation"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Core components
        self.validator = StrategyValidator()
        
        # Data storage
        self.strategies: Dict[str, Any] = {}  # From education_platform
        self.backtest_results: Dict[str, BacktestResult] = {}
        self.strategy_reviews: Dict[str, List[StrategyReview]] = {}
        self.strategy_licenses: Dict[str, StrategyLicense] = {}
        self.strategy_signals: Dict[str, List[StrategySignal]] = {}
        
        # Analytics
        self.marketplace_analytics: Dict[str, Any] = {}
        
        # Background tasks
        self.validation_queue: List[str] = []
        self.signal_generation_active = False
        
        # Start background tasks
        asyncio.create_task(self._validation_worker())
        asyncio.create_task(self._analytics_updater())
    
    async def submit_strategy_for_validation(self, strategy_id: str, 
                                           backtest_data: Dict[str, Any]) -> str:
        """Submit strategy for AI validation"""
        try:
            # Create backtest result
            backtest_result = BacktestResult(
                backtest_id="",
                strategy_id=strategy_id,
                start_date=datetime.fromisoformat(backtest_data['start_date']),
                end_date=datetime.fromisoformat(backtest_data['end_date']),
                initial_capital=backtest_data['initial_capital'],
                final_capital=backtest_data['final_capital'],
                metrics=StrategyMetrics(**backtest_data['metrics']),
                equity_curve=backtest_data.get('equity_curve', []),
                trade_log=backtest_data.get('trade_log', []),
                monthly_returns=backtest_data.get('monthly_returns', [])
            )
            
            # Store backtest result
            self.backtest_results[backtest_result.backtest_id] = backtest_result
            
            # Add to validation queue
            self.validation_queue.append(backtest_result.backtest_id)
            
            self.logger.info(f"Strategy submitted for validation: {strategy_id}")
            return backtest_result.backtest_id
            
        except Exception as e:
            self.logger.error(f"Strategy validation submission failed: {e}")
            return ""
    
    async def _validation_worker(self):
        """Background worker for strategy validation"""
        while True:
            try:
                if self.validation_queue:
                    backtest_id = self.validation_queue.pop(0)
                    
                    if backtest_id in self.backtest_results:
                        backtest_result = self.backtest_results[backtest_id]
                        
                        # Update status to validating
                        backtest_result.validation_status = ValidationStatus.VALIDATING
                        
                        # Perform validation
                        validation_result = await self.validator.validate_strategy(
                            backtest_result.strategy_id, backtest_result
                        )
                        
                        # Update validation status
                        backtest_result.validation_status = validation_result
                        
                        # Calculate performance grade
                        backtest_result.performance_grade = self.validator.calculate_performance_grade(
                            backtest_result.metrics
                        )
                        
                        self.logger.info(f"Strategy validation completed: {backtest_result.strategy_id}")
                
                await asyncio.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                self.logger.error(f"Validation worker error: {e}")
                await asyncio.sleep(10)
    
    async def purchase_strategy_license(self, user_id: str, strategy_id: str,
                                      license_type: str = "single_use",
                                      max_accounts: int = 1,
                                      max_capital: float = 0.0) -> Optional[str]:
        """Purchase strategy license"""
        try:
            # Create license
            license = StrategyLicense(
                license_id="",
                strategy_id=strategy_id,
                user_id=user_id,
                license_type=license_type,
                purchase_price=0.0,  # Would be calculated based on strategy price
                max_accounts=max_accounts,
                max_capital=max_capital
            )
            
            # Set subscription details if applicable
            if license_type == "subscription":
                license.subscription_start = datetime.now(timezone.utc)
                license.subscription_end = datetime.now(timezone.utc) + timedelta(days=30)
                license.auto_renewal = True
            
            self.strategy_licenses[license.license_id] = license
            
            self.logger.info(f"Strategy license purchased: {strategy_id} by {user_id}")
            return license.license_id
            
        except Exception as e:
            self.logger.error(f"Strategy license purchase failed: {e}")
            return None
    
    async def submit_strategy_review(self, user_id: str, strategy_id: str,
                                   rating: float, title: str, content: str,
                                   user_results: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Submit strategy review"""
        try:
            # Create review
            review = StrategyReview(
                review_id="",
                strategy_id=strategy_id,
                user_id=user_id,
                rating=max(1.0, min(5.0, rating)),  # Clamp to 1-5
                title=title,
                content=content
            )
            
            # Add user results if provided
            if user_results:
                review.user_results = StrategyMetrics(**user_results)
                review.verified_performance = True
            
            # Check if verified purchase
            for license in self.strategy_licenses.values():
                if license.user_id == user_id and license.strategy_id == strategy_id:
                    review.is_verified_purchase = True
                    break
            
            # Store review
            if strategy_id not in self.strategy_reviews:
                self.strategy_reviews[strategy_id] = []
            self.strategy_reviews[strategy_id].append(review)
            
            self.logger.info(f"Strategy review submitted: {strategy_id} by {user_id}")
            return review.review_id
            
        except Exception as e:
            self.logger.error(f"Strategy review submission failed: {e}")
            return None
    
    async def generate_strategy_signal(self, strategy_id: str, symbol: str,
                                     market_data: Dict[str, Any]) -> Optional[StrategySignal]:
        """Generate trading signal from strategy"""
        try:
            # Simulate signal generation (in real implementation, this would use the actual strategy)
            action_choices = ['buy', 'sell', 'hold']
            action = np.random.choice(action_choices, p=[0.3, 0.3, 0.4])
            
            if action == 'hold':
                return None
            
            # Generate signal
            signal = StrategySignal(
                signal_id="",
                strategy_id=strategy_id,
                symbol=symbol,
                action=action,
                quantity=np.random.uniform(0.1, 1.0),
                price=market_data.get('current_price', 0.0),
                confidence=np.random.uniform(0.6, 0.95),
                risk_level=np.random.choice(list(RiskLevel)),
                timeframe=market_data.get('timeframe', '1h'),
                reasoning=f"Generated {action} signal based on technical analysis",
                technical_indicators={
                    'rsi': np.random.uniform(20, 80),
                    'macd': np.random.uniform(-1, 1),
                    'bb_position': np.random.uniform(0, 1)
                }
            )
            
            # Set expiration
            signal.expires_at = datetime.now(timezone.utc) + timedelta(hours=1)
            
            # Store signal
            if strategy_id not in self.strategy_signals:
                self.strategy_signals[strategy_id] = []
            self.strategy_signals[strategy_id].append(signal)
            
            # Keep only recent signals (last 100)
            if len(self.strategy_signals[strategy_id]) > 100:
                self.strategy_signals[strategy_id] = self.strategy_signals[strategy_id][-100:]
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Strategy signal generation failed: {e}")
            return None
    
    async def get_strategy_performance_report(self, strategy_id: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive strategy performance report"""
        try:
            # Find backtest results for strategy
            backtest_results = [
                result for result in self.backtest_results.values()
                if result.strategy_id == strategy_id
            ]
            
            if not backtest_results:
                return None
            
            # Get latest backtest
            latest_backtest = max(backtest_results, key=lambda r: r.created_at)
            
            # Get reviews
            reviews = self.strategy_reviews.get(strategy_id, [])
            
            # Calculate review statistics
            review_stats = {
                'total_reviews': len(reviews),
                'average_rating': np.mean([r.rating for r in reviews]) if reviews else 0.0,
                'verified_reviews': sum(1 for r in reviews if r.is_verified_purchase),
                'rating_distribution': {
                    '5_star': sum(1 for r in reviews if r.rating >= 4.5),
                    '4_star': sum(1 for r in reviews if 3.5 <= r.rating < 4.5),
                    '3_star': sum(1 for r in reviews if 2.5 <= r.rating < 3.5),
                    '2_star': sum(1 for r in reviews if 1.5 <= r.rating < 2.5),
                    '1_star': sum(1 for r in reviews if r.rating < 1.5)
                }
            }
            
            # Get recent signals
            recent_signals = self.strategy_signals.get(strategy_id, [])[-10:]
            
            # Calculate signal statistics
            signal_stats = {
                'total_signals': len(self.strategy_signals.get(strategy_id, [])),
                'recent_signals': len(recent_signals),
                'signal_accuracy': np.random.uniform(0.6, 0.9),  # Would be calculated from actual results
                'average_confidence': np.mean([s.confidence for s in recent_signals]) if recent_signals else 0.0
            }
            
            # Compile report
            report = {
                'strategy_id': strategy_id,
                'validation_status': latest_backtest.validation_status.value,
                'performance_grade': latest_backtest.performance_grade.value,
                'backtest_metrics': asdict(latest_backtest.metrics),
                'review_statistics': review_stats,
                'signal_statistics': signal_stats,
                'backtest_period': {
                    'start_date': latest_backtest.start_date.isoformat(),
                    'end_date': latest_backtest.end_date.isoformat(),
                    'duration_days': (latest_backtest.end_date - latest_backtest.start_date).days
                },
                'generated_at': datetime.now(timezone.utc).isoformat()
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"Strategy performance report generation failed: {e}")
            return None
    
    async def get_marketplace_leaderboard(self, category: Optional[str] = None,
                                        timeframe: str = "all_time") -> List[Dict[str, Any]]:
        """Get marketplace leaderboard"""
        try:
            leaderboard = []
            
            # Get all strategies with backtest results
            for strategy_id, strategy in self.strategies.items():
                # Find latest backtest
                strategy_backtests = [
                    result for result in self.backtest_results.values()
                    if result.strategy_id == strategy_id and 
                    result.validation_status == ValidationStatus.PASSED
                ]
                
                if not strategy_backtests:
                    continue
                
                latest_backtest = max(strategy_backtests, key=lambda r: r.created_at)
                
                # Apply category filter
                if category and strategy.get('category', '').lower() != category.lower():
                    continue
                
                # Get reviews
                reviews = self.strategy_reviews.get(strategy_id, [])
                avg_rating = np.mean([r.rating for r in reviews]) if reviews else 0.0
                
                # Calculate composite score
                metrics = latest_backtest.metrics
                composite_score = (
                    metrics.sharpe_ratio * 0.3 +
                    (metrics.annual_return / 100) * 0.3 +
                    (100 - metrics.max_drawdown) / 100 * 0.2 +
                    (metrics.win_rate / 100) * 0.1 +
                    (avg_rating / 5) * 0.1
                )
                
                leaderboard.append({
                    'strategy_id': strategy_id,
                    'name': strategy.get('name', 'Unknown'),
                    'author_name': strategy.get('author_name', 'Unknown'),
                    'category': strategy.get('category', 'Unknown'),
                    'performance_grade': latest_backtest.performance_grade.value,
                    'annual_return': metrics.annual_return,
                    'sharpe_ratio': metrics.sharpe_ratio,
                    'max_drawdown': metrics.max_drawdown,
                    'win_rate': metrics.win_rate,
                    'average_rating': avg_rating,
                    'total_reviews': len(reviews),
                    'composite_score': composite_score,
                    'downloads': strategy.get('downloads', 0)
                })
            
            # Sort by composite score
            leaderboard.sort(key=lambda x: x['composite_score'], reverse=True)
            
            return leaderboard[:50]  # Top 50
            
        except Exception as e:
            self.logger.error(f"Marketplace leaderboard generation failed: {e}")
            return []
    
    async def _analytics_updater(self):
        """Background task to update marketplace analytics"""
        while True:
            try:
                # Calculate marketplace analytics
                analytics = {
                    'total_strategies': len(self.strategies),
                    'validated_strategies': sum(
                        1 for result in self.backtest_results.values()
                        if result.validation_status == ValidationStatus.PASSED
                    ),
                    'total_reviews': sum(len(reviews) for reviews in self.strategy_reviews.values()),
                    'total_licenses': len(self.strategy_licenses),
                    'total_signals': sum(len(signals) for signals in self.strategy_signals.values()),
                    'performance_distribution': {},
                    'category_distribution': {},
                    'updated_at': datetime.now(timezone.utc).isoformat()
                }
                
                # Performance grade distribution
                grade_counts = {}
                for result in self.backtest_results.values():
                    if result.validation_status == ValidationStatus.PASSED:
                        grade = result.performance_grade.value
                        grade_counts[grade] = grade_counts.get(grade, 0) + 1
                
                analytics['performance_distribution'] = grade_counts
                
                # Category distribution
                category_counts = {}
                for strategy in self.strategies.values():
                    category = strategy.get('category', 'Unknown')
                    category_counts[category] = category_counts.get(category, 0) + 1
                
                analytics['category_distribution'] = category_counts
                
                self.marketplace_analytics = analytics
                
                await asyncio.sleep(300)  # Update every 5 minutes
                
            except Exception as e:
                self.logger.error(f"Analytics updater error: {e}")
                await asyncio.sleep(60)
    
    async def get_marketplace_analytics(self) -> Dict[str, Any]:
        """Get marketplace analytics"""
        return self.marketplace_analytics.copy()

# Example usage
async def main():
    """
    Example usage of Strategy Marketplace
    """
    print("🛒 Strategy Marketplace - Phase 5 Testing")
    print("=" * 50)
    
    # Initialize marketplace
    marketplace = StrategyMarketplace()
    
    # Wait for initialization
    await asyncio.sleep(2)
    
    # Test strategy validation submission
    print("\n🔍 Testing Strategy Validation:")
    
    # Sample backtest data
    backtest_data = {
        'start_date': '2023-01-01T00:00:00+00:00',
        'end_date': '2023-12-31T23:59:59+00:00',
        'initial_capital': 100000.0,
        'final_capital': 135000.0,
        'metrics': {
            'total_return': 35.0,
            'annual_return': 35.0,
            'win_rate': 72.5,
            'profit_factor': 1.85,
            'max_drawdown': 8.3,
            'volatility': 15.2,
            'sharpe_ratio': 2.45,
            'sortino_ratio': 3.12,
            'total_trades': 156,
            'winning_trades': 113,
            'losing_trades': 43
        },
        'monthly_returns': [2.1, 3.5, -1.2, 4.8, 2.9, 1.7, 3.2, -0.8, 2.4, 3.1, 1.9, 2.8]
    }
    
    strategy_id = "test_strategy_001"
    marketplace.strategies[strategy_id] = {
        'name': 'AI Momentum Strategy',
        'category': 'AI_ML',
        'author_name': 'Test Author',
        'downloads': 0
    }
    
    backtest_id = await marketplace.submit_strategy_for_validation(strategy_id, backtest_data)
    if backtest_id:
        print(f"  ✅ Strategy submitted for validation: {backtest_id}")
        
        # Wait for validation
        await asyncio.sleep(6)
        
        if backtest_id in marketplace.backtest_results:
            result = marketplace.backtest_results[backtest_id]
            print(f"  Validation status: {result.validation_status.value}")
            print(f"  Performance grade: {result.performance_grade.value}")
    
    # Test license purchase
    print(f"\n💳 Testing License Purchase:")
    
    user_id = "test_user_001"
    license_id = await marketplace.purchase_strategy_license(
        user_id, strategy_id, "unlimited", max_accounts=3
    )
    if license_id:
        print(f"  ✅ License purchased: {license_id}")
        license = marketplace.strategy_licenses[license_id]
        print(f"  License type: {license.license_type}")
        print(f"  Max accounts: {license.max_accounts}")
    
    # Test review submission
    print(f"\n⭐ Testing Review Submission:")
    
    review_id = await marketplace.submit_strategy_review(
        user_id, strategy_id, 4.5, "Excellent Strategy",
        "This strategy has performed very well in my live trading account.",
        user_results={
            'total_return': 28.5,
            'win_rate': 68.2,
            'max_drawdown': 12.1,
            'sharpe_ratio': 2.1
        }
    )
    if review_id:
        print(f"  ✅ Review submitted: {review_id}")
        review = None
        for reviews in marketplace.strategy_reviews.values():
            for r in reviews:
                if r.review_id == review_id:
                    review = r
                    break
        if review:
            print(f"  Rating: {review.rating}/5")
            print(f"  Verified purchase: {review.is_verified_purchase}")
            print(f"  Verified performance: {review.verified_performance}")
    
    # Test signal generation
    print(f"\n📡 Testing Signal Generation:")
    
    market_data = {
        'current_price': 45250.0,
        'timeframe': '1h'
    }
    
    signal = await marketplace.generate_strategy_signal(strategy_id, 'BTCUSDT', market_data)
    if signal:
        print(f"  ✅ Signal generated: {signal.signal_id}")
        print(f"  Action: {signal.action}")
        print(f"  Symbol: {signal.symbol}")
        print(f"  Confidence: {signal.confidence:.2f}")
        print(f"  Risk level: {signal.risk_level.value}")
    
    # Test performance report
    print(f"\n📊 Testing Performance Report:")
    
    report = await marketplace.get_strategy_performance_report(strategy_id)
    if report:
        print(f"  ✅ Performance report generated")
        print(f"  Validation status: {report['validation_status']}")
        print(f"  Performance grade: {report['performance_grade']}")
        print(f"  Annual return: {report['backtest_metrics']['annual_return']:.1f}%")
        print(f"  Sharpe ratio: {report['backtest_metrics']['sharpe_ratio']:.2f}")
        print(f"  Total reviews: {report['review_statistics']['total_reviews']}")
        print(f"  Average rating: {report['review_statistics']['average_rating']:.1f}")
    
    # Test leaderboard
    print(f"\n🏆 Testing Marketplace Leaderboard:")
    
    leaderboard = await marketplace.get_marketplace_leaderboard()
    if leaderboard:
        print(f"  ✅ Leaderboard generated with {len(leaderboard)} strategies")
        for i, strategy in enumerate(leaderboard[:3]):
            print(f"  #{i+1}: {strategy['name']} - Grade: {strategy['performance_grade']}")
            print(f"       Return: {strategy['annual_return']:.1f}%, Sharpe: {strategy['sharpe_ratio']:.2f}")
    
    # Get marketplace analytics
    print(f"\n📈 Marketplace Analytics:")
    
    analytics = await marketplace.get_marketplace_analytics()
    if analytics:
        print(f"  Total strategies: {analytics['total_strategies']}")
        print(f"  Validated strategies: {analytics['validated_strategies']}")
        print(f"  Total reviews: {analytics['total_reviews']}")
        print(f"  Total licenses: {analytics['total_licenses']}")
        print(f"  Total signals: {analytics['total_signals']}")
        
        if analytics['performance_distribution']:
            print(f"  Performance distribution:")
            for grade, count in analytics['performance_distribution'].items():
                print(f"    {grade}: {count}")
    
    print(f"\n✅ Strategy Marketplace testing completed!")

if __name__ == "__main__":
    asyncio.run(main())