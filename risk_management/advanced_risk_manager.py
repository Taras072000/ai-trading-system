"""
Продвинутый риск-менеджер для Peper Binance v4
Архитектурные улучшения - Фаза 2
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import threading
from collections import defaultdict, deque

from config.unified_config_manager import get_config_manager
from market_analysis.market_phase_detector import MarketPhase, MarketCondition

logger = logging.getLogger(__name__)

class RiskLevel(Enum):
    """Уровни риска"""
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"

class PositionType(Enum):
    """Типы позиций"""
    LONG = "long"
    SHORT = "short"

@dataclass
class Position:
    """Информация о позиции"""
    symbol: str
    position_type: PositionType
    entry_price: float
    quantity: float
    timestamp: datetime
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    trailing_stop: Optional[float] = None
    unrealized_pnl: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RiskMetrics:
    """Метрики риска"""
    portfolio_risk: float
    position_risk: float
    correlation_risk: float
    volatility_risk: float
    drawdown_risk: float
    var_1d: float  # Value at Risk 1 день
    var_7d: float  # Value at Risk 7 дней
    max_drawdown: float
    sharpe_ratio: float
    risk_level: RiskLevel
    timestamp: datetime

@dataclass
class RiskLimits:
    """Лимиты риска"""
    max_position_size: float
    max_portfolio_risk: float
    max_correlation: float
    max_daily_loss: float
    max_weekly_loss: float
    max_drawdown: float
    max_open_positions: int

class CorrelationAnalyzer:
    """Анализатор корреляций между активами"""
    
    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self.price_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        self.correlation_matrix: Dict[Tuple[str, str], float] = {}
        self.last_update = datetime.now()
        
    def update_prices(self, symbol: str, price: float) -> None:
        """Обновление цен для расчета корреляции"""
        self.price_history[symbol].append({
            'price': price,
            'timestamp': datetime.now()
        })
        
        # Обновление корреляций каждые 5 минут
        if datetime.now() - self.last_update > timedelta(minutes=5):
            self._update_correlations()
            self.last_update = datetime.now()
    
    def _update_correlations(self) -> None:
        """Обновление матрицы корреляций"""
        symbols = list(self.price_history.keys())
        
        for i, symbol1 in enumerate(symbols):
            for j, symbol2 in enumerate(symbols[i+1:], i+1):
                correlation = self._calculate_correlation(symbol1, symbol2)
                self.correlation_matrix[(symbol1, symbol2)] = correlation
                self.correlation_matrix[(symbol2, symbol1)] = correlation
    
    def _calculate_correlation(self, symbol1: str, symbol2: str) -> float:
        """Расчет корреляции между двумя активами"""
        try:
            history1 = list(self.price_history[symbol1])
            history2 = list(self.price_history[symbol2])
            
            if len(history1) < 10 or len(history2) < 10:
                return 0.0
            
            # Синхронизация по времени
            prices1 = [h['price'] for h in history1[-min(len(history1), len(history2)):]]
            prices2 = [h['price'] for h in history2[-min(len(history1), len(history2)):]]
            
            if len(prices1) < 2 or len(prices2) < 2:
                return 0.0
            
            # Расчет доходностей
            returns1 = np.diff(np.log(prices1))
            returns2 = np.diff(np.log(prices2))
            
            # Корреляция
            correlation = np.corrcoef(returns1, returns2)[0, 1]
            return correlation if not np.isnan(correlation) else 0.0
            
        except Exception as e:
            logger.error(f"Ошибка расчета корреляции {symbol1}-{symbol2}: {e}")
            return 0.0
    
    def get_correlation(self, symbol1: str, symbol2: str) -> float:
        """Получение корреляции между активами"""
        return self.correlation_matrix.get((symbol1, symbol2), 0.0)
    
    def get_portfolio_correlations(self, symbols: List[str]) -> Dict[Tuple[str, str], float]:
        """Получение корреляций для портфеля"""
        correlations = {}
        for i, symbol1 in enumerate(symbols):
            for j, symbol2 in enumerate(symbols[i+1:], i+1):
                correlation = self.get_correlation(symbol1, symbol2)
                correlations[(symbol1, symbol2)] = correlation
        return correlations

class VolatilityCalculator:
    """Калькулятор волатильности"""
    
    @staticmethod
    def calculate_historical_volatility(prices: List[float], window: int = 20) -> float:
        """Расчет исторической волатильности"""
        if len(prices) < window:
            return 0.02  # Значение по умолчанию
        
        returns = np.diff(np.log(prices[-window:]))
        volatility = np.std(returns) * np.sqrt(252)  # Годовая волатильность
        return volatility
    
    @staticmethod
    def calculate_garch_volatility(prices: List[float]) -> float:
        """Расчет GARCH волатильности (упрощенная версия)"""
        if len(prices) < 30:
            return VolatilityCalculator.calculate_historical_volatility(prices)
        
        returns = np.diff(np.log(prices))
        
        # Упрощенная GARCH(1,1)
        alpha = 0.1
        beta = 0.85
        omega = 0.000001
        
        variance = np.var(returns)
        garch_variance = omega
        
        for ret in returns[-20:]:  # Последние 20 наблюдений
            garch_variance = omega + alpha * (ret ** 2) + beta * garch_variance
        
        return np.sqrt(garch_variance * 252)  # Годовая волатильность

class AdvancedRiskManager:
    """Продвинутый риск-менеджер"""
    
    def __init__(self):
        self.config_manager = get_config_manager()
        self.risk_config = self.config_manager.get_risk_config()
        
        # Компоненты
        self.correlation_analyzer = CorrelationAnalyzer()
        self.volatility_calculator = VolatilityCalculator()
        
        # Текущие позиции
        self.positions: Dict[str, Position] = {}
        
        # История P&L
        self.pnl_history: deque = deque(maxlen=1000)
        self.daily_pnl: Dict[str, float] = defaultdict(float)  # По дням
        self.weekly_pnl: Dict[str, float] = defaultdict(float)  # По неделям
        
        # Лимиты риска
        self.risk_limits = RiskLimits(
            max_position_size=self.risk_config.max_position_size,
            max_portfolio_risk=0.15,  # 15% портфеля
            max_correlation=self.risk_config.max_correlation,
            max_daily_loss=self.risk_config.daily_loss_limit,
            max_weekly_loss=0.10,  # 10% в неделю
            max_drawdown=0.20,  # 20% максимальная просадка
            max_open_positions=self.risk_config.max_open_positions
        )
        
        # Метрики
        self.current_metrics: Optional[RiskMetrics] = None
        
        # Блокировка
        self.lock = threading.Lock()
        
        logger.info("Продвинутый риск-менеджер инициализирован")
    
    async def calculate_position_size(self, symbol: str, entry_price: float, 
                                    stop_loss: float, market_condition: Optional[MarketCondition] = None,
                                    account_balance: float = 10000.0) -> Tuple[float, Dict[str, Any]]:
        """Расчет размера позиции с учетом риска"""
        try:
            # Базовый размер позиции
            base_size = self.risk_config.base_position_size
            
            # Корректировка на основе фазы рынка
            if market_condition:
                phase_config = self.config_manager.get_market_phase_config(market_condition.primary_phase.value)
                if phase_config:
                    risk_multiplier = phase_config.risk_multiplier
                    base_size *= risk_multiplier
            
            # Корректировка на основе волатильности
            volatility_adjustment = await self._calculate_volatility_adjustment(symbol, entry_price)
            base_size *= volatility_adjustment
            
            # Корректировка на основе корреляции
            correlation_adjustment = await self._calculate_correlation_adjustment(symbol)
            base_size *= correlation_adjustment
            
            # Расчет риска на сделку
            risk_per_trade = abs(entry_price - stop_loss) / entry_price
            if risk_per_trade > 0:
                # Ограничение риска на сделку до 1.5%
                max_risk_per_trade = 0.015
                if risk_per_trade > max_risk_per_trade:
                    base_size *= max_risk_per_trade / risk_per_trade
            
            # Ограничения
            max_size = min(
                self.risk_limits.max_position_size,
                account_balance * base_size
            )
            
            # Проверка лимитов портфеля
            portfolio_risk = await self._calculate_portfolio_risk()
            if portfolio_risk > self.risk_limits.max_portfolio_risk * 0.8:  # 80% от лимита
                max_size *= 0.5  # Уменьшаем размер позиции
            
            position_size = min(max_size, account_balance * base_size)
            
            # Метаданные расчета
            calculation_metadata = {
                'base_size_percent': base_size,
                'volatility_adjustment': volatility_adjustment,
                'correlation_adjustment': correlation_adjustment,
                'risk_per_trade': risk_per_trade,
                'portfolio_risk': portfolio_risk,
                'final_size_usd': position_size,
                'market_phase': market_condition.primary_phase.value if market_condition else None
            }
            
            logger.info(f"Размер позиции для {symbol}: ${position_size:.2f} "
                       f"({position_size/account_balance*100:.2f}% от баланса)")
            
            return position_size, calculation_metadata
            
        except Exception as e:
            logger.error(f"Ошибка расчета размера позиции: {e}")
            # Возвращаем минимальный безопасный размер
            safe_size = account_balance * 0.01  # 1% от баланса
            return safe_size, {'error': str(e), 'safe_mode': True}
    
    async def calculate_stop_loss(self, symbol: str, entry_price: float, 
                                position_type: PositionType, atr: float,
                                market_condition: Optional[MarketCondition] = None) -> Tuple[float, bool]:
        """Расчет стоп-лосса на основе ATR"""
        try:
            # Базовый множитель ATR
            atr_multiplier = self.risk_config.stop_loss_atr_multiplier
            
            # Корректировка на основе фазы рынка
            if market_condition:
                if market_condition.primary_phase == MarketPhase.HIGH_VOLATILITY:
                    atr_multiplier *= 1.5  # Увеличиваем стоп в волатильном рынке
                elif market_condition.primary_phase == MarketPhase.LOW_VOLATILITY:
                    atr_multiplier *= 0.8  # Уменьшаем стоп в спокойном рынке
            
            # Расчет стоп-лосса
            if position_type == PositionType.LONG:
                stop_loss = entry_price - (atr * atr_multiplier)
            else:
                stop_loss = entry_price + (atr * atr_multiplier)
            
            # Проверка максимального риска
            risk_percent = abs(entry_price - stop_loss) / entry_price
            max_risk = 0.02  # Максимум 2% риска на сделку
            
            trailing_enabled = True
            if risk_percent > max_risk:
                # Корректируем стоп-лосс
                if position_type == PositionType.LONG:
                    stop_loss = entry_price * (1 - max_risk)
                else:
                    stop_loss = entry_price * (1 + max_risk)
                trailing_enabled = False  # Отключаем трейлинг для жестких стопов
            
            logger.info(f"Стоп-лосс для {symbol}: {stop_loss:.6f} "
                       f"(риск: {risk_percent*100:.2f}%)")
            
            return stop_loss, trailing_enabled
            
        except Exception as e:
            logger.error(f"Ошибка расчета стоп-лосса: {e}")
            # Безопасный стоп-лосс 1.5%
            if position_type == PositionType.LONG:
                return entry_price * 0.985, False
            else:
                return entry_price * 1.015, False
    
    async def calculate_take_profit(self, symbol: str, entry_price: float, 
                                  stop_loss: float, position_type: PositionType,
                                  market_condition: Optional[MarketCondition] = None) -> List[float]:
        """Расчет уровней тейк-профита"""
        try:
            # Базовое соотношение риск/прибыль
            risk_reward_ratio = self.risk_config.take_profit_ratio
            
            # Корректировка на основе фазы рынка
            if market_condition:
                if market_condition.primary_phase in [MarketPhase.UPTREND, MarketPhase.DOWNTREND]:
                    risk_reward_ratio *= 1.2  # Увеличиваем цели в трендовом рынке
                elif market_condition.primary_phase == MarketPhase.SIDEWAYS:
                    risk_reward_ratio *= 0.8  # Уменьшаем цели в боковом рынке
            
            # Расчет риска на сделку
            risk_amount = abs(entry_price - stop_loss)
            
            # Уровни тейк-профита
            take_profits = []
            
            if position_type == PositionType.LONG:
                # Первый уровень (50% позиции)
                tp1 = entry_price + (risk_amount * risk_reward_ratio * 0.5)
                # Второй уровень (30% позиции)
                tp2 = entry_price + (risk_amount * risk_reward_ratio)
                # Третий уровень (20% позиции)
                tp3 = entry_price + (risk_amount * risk_reward_ratio * 1.5)
            else:
                tp1 = entry_price - (risk_amount * risk_reward_ratio * 0.5)
                tp2 = entry_price - (risk_amount * risk_reward_ratio)
                tp3 = entry_price - (risk_amount * risk_reward_ratio * 1.5)
            
            take_profits = [tp1, tp2, tp3]
            
            logger.info(f"Тейк-профиты для {symbol}: {[f'{tp:.6f}' for tp in take_profits]}")
            
            return take_profits
            
        except Exception as e:
            logger.error(f"Ошибка расчета тейк-профита: {e}")
            return []
    
    async def validate_trade(self, symbol: str, position_type: PositionType, 
                           position_size: float, entry_price: float) -> Tuple[bool, str]:
        """Валидация сделки перед открытием"""
        try:
            # Проверка лимитов позиций
            if len(self.positions) >= self.risk_limits.max_open_positions:
                return False, f"Превышен лимит открытых позиций ({self.risk_limits.max_open_positions})"
            
            # Проверка размера позиции
            if position_size > self.risk_limits.max_position_size * 10000:  # Предполагаем баланс 10k
                return False, f"Превышен максимальный размер позиции"
            
            # Проверка корреляции
            correlation_risk = await self._check_correlation_risk(symbol)
            if correlation_risk > self.risk_limits.max_correlation:
                return False, f"Высокий корреляционный риск: {correlation_risk:.3f}"
            
            # Проверка дневных потерь
            today = datetime.now().strftime('%Y-%m-%d')
            daily_loss = self.daily_pnl.get(today, 0.0)
            if daily_loss < -self.risk_limits.max_daily_loss * 10000:  # Предполагаем баланс 10k
                return False, f"Превышен дневной лимит потерь"
            
            # Проверка портфельного риска
            portfolio_risk = await self._calculate_portfolio_risk()
            if portfolio_risk > self.risk_limits.max_portfolio_risk:
                return False, f"Превышен портфельный риск: {portfolio_risk:.3f}"
            
            return True, "Сделка прошла валидацию"
            
        except Exception as e:
            logger.error(f"Ошибка валидации сделки: {e}")
            return False, f"Ошибка валидации: {str(e)}"
    
    async def add_position(self, position: Position) -> None:
        """Добавление позиции в портфель"""
        with self.lock:
            self.positions[position.symbol] = position
            
        # Обновление корреляций
        self.correlation_analyzer.update_prices(position.symbol, position.entry_price)
        
        logger.info(f"Добавлена позиция: {position.symbol} {position.position_type.value} "
                   f"размер: {position.quantity}")
    
    async def remove_position(self, symbol: str, exit_price: float, 
                            realized_pnl: float) -> None:
        """Удаление позиции из портфеля"""
        with self.lock:
            if symbol in self.positions:
                position = self.positions.pop(symbol)
                
                # Обновление P&L
                self._update_pnl(realized_pnl)
                
                logger.info(f"Закрыта позиция: {symbol} P&L: {realized_pnl:.2f}")
    
    async def update_position_prices(self, symbol: str, current_price: float) -> None:
        """Обновление цен позиций"""
        with self.lock:
            if symbol in self.positions:
                position = self.positions[symbol]
                
                # Расчет нереализованной прибыли
                if position.position_type == PositionType.LONG:
                    position.unrealized_pnl = (current_price - position.entry_price) * position.quantity
                else:
                    position.unrealized_pnl = (position.entry_price - current_price) * position.quantity
        
        # Обновление корреляций
        self.correlation_analyzer.update_prices(symbol, current_price)
    
    async def calculate_risk_metrics(self, account_balance: float = 10000.0) -> RiskMetrics:
        """Расчет метрик риска"""
        try:
            # Портфельный риск
            portfolio_risk = await self._calculate_portfolio_risk()
            
            # Позиционный риск
            position_risk = await self._calculate_position_risk()
            
            # Корреляционный риск
            correlation_risk = await self._calculate_correlation_risk()
            
            # Волатильность портфеля
            volatility_risk = await self._calculate_volatility_risk()
            
            # Риск просадки
            drawdown_risk = await self._calculate_drawdown_risk()
            
            # Value at Risk
            var_1d, var_7d = await self._calculate_var(account_balance)
            
            # Максимальная просадка
            max_drawdown = await self._calculate_max_drawdown()
            
            # Коэффициент Шарпа
            sharpe_ratio = await self._calculate_sharpe_ratio()
            
            # Общий уровень риска
            risk_level = await self._determine_risk_level(portfolio_risk, volatility_risk, drawdown_risk)
            
            metrics = RiskMetrics(
                portfolio_risk=portfolio_risk,
                position_risk=position_risk,
                correlation_risk=correlation_risk,
                volatility_risk=volatility_risk,
                drawdown_risk=drawdown_risk,
                var_1d=var_1d,
                var_7d=var_7d,
                max_drawdown=max_drawdown,
                sharpe_ratio=sharpe_ratio,
                risk_level=risk_level,
                timestamp=datetime.now()
            )
            
            self.current_metrics = metrics
            return metrics
            
        except Exception as e:
            logger.error(f"Ошибка расчета метрик риска: {e}")
            return self._get_default_metrics()
    
    async def _calculate_volatility_adjustment(self, symbol: str, price: float) -> float:
        """Корректировка размера позиции на основе волатильности"""
        try:
            # Получаем историю цен (заглушка)
            # В реальной реализации здесь должен быть запрос к базе данных
            prices = [price * (1 + np.random.normal(0, 0.02)) for _ in range(30)]
            
            volatility = self.volatility_calculator.calculate_historical_volatility(prices)
            
            # Корректировка: чем выше волатильность, тем меньше позиция
            if volatility > 0.5:  # Высокая волатильность
                return 0.5
            elif volatility > 0.3:  # Средняя волатильность
                return 0.7
            elif volatility < 0.1:  # Низкая волатильность
                return 1.2
            else:
                return 1.0
                
        except Exception as e:
            logger.error(f"Ошибка расчета волатильности: {e}")
            return 1.0
    
    async def _calculate_correlation_adjustment(self, symbol: str) -> float:
        """Корректировка размера позиции на основе корреляции"""
        try:
            if not self.positions:
                return 1.0
            
            max_correlation = 0.0
            for existing_symbol in self.positions.keys():
                correlation = abs(self.correlation_analyzer.get_correlation(symbol, existing_symbol))
                max_correlation = max(max_correlation, correlation)
            
            # Корректировка: чем выше корреляция, тем меньше позиция
            if max_correlation > 0.8:
                return 0.3
            elif max_correlation > 0.6:
                return 0.5
            elif max_correlation > 0.4:
                return 0.7
            else:
                return 1.0
                
        except Exception as e:
            logger.error(f"Ошибка расчета корреляции: {e}")
            return 1.0
    
    async def _calculate_portfolio_risk(self) -> float:
        """Расчет портфельного риска"""
        try:
            if not self.positions:
                return 0.0
            
            total_risk = 0.0
            for position in self.positions.values():
                # Риск позиции как процент от стоп-лосса
                if position.stop_loss:
                    position_risk = abs(position.entry_price - position.stop_loss) / position.entry_price
                    position_weight = position.quantity * position.entry_price / 10000  # Предполагаем баланс 10k
                    total_risk += position_risk * position_weight
            
            return total_risk
            
        except Exception as e:
            logger.error(f"Ошибка расчета портфельного риска: {e}")
            return 0.0
    
    async def _check_correlation_risk(self, new_symbol: str) -> float:
        """Проверка корреляционного риска для нового символа"""
        if not self.positions:
            return 0.0
        
        max_correlation = 0.0
        for existing_symbol in self.positions.keys():
            correlation = abs(self.correlation_analyzer.get_correlation(new_symbol, existing_symbol))
            max_correlation = max(max_correlation, correlation)
        
        return max_correlation
    
    def _update_pnl(self, pnl: float) -> None:
        """Обновление истории P&L"""
        today = datetime.now().strftime('%Y-%m-%d')
        week = datetime.now().strftime('%Y-W%U')
        
        self.daily_pnl[today] += pnl
        self.weekly_pnl[week] += pnl
        
        self.pnl_history.append({
            'pnl': pnl,
            'timestamp': datetime.now(),
            'cumulative': sum(self.daily_pnl.values())
        })
    
    async def _calculate_position_risk(self) -> float:
        """Расчет позиционного риска"""
        # Упрощенная реализация
        return len(self.positions) / self.risk_limits.max_open_positions
    
    async def _calculate_correlation_risk(self) -> float:
        """Расчет корреляционного риска"""
        if len(self.positions) < 2:
            return 0.0
        
        symbols = list(self.positions.keys())
        correlations = []
        
        for i, symbol1 in enumerate(symbols):
            for symbol2 in symbols[i+1:]:
                correlation = abs(self.correlation_analyzer.get_correlation(symbol1, symbol2))
                correlations.append(correlation)
        
        return max(correlations) if correlations else 0.0
    
    async def _calculate_volatility_risk(self) -> float:
        """Расчет риска волатильности"""
        # Упрощенная реализация
        return 0.5  # Средний уровень
    
    async def _calculate_drawdown_risk(self) -> float:
        """Расчет риска просадки"""
        if not self.pnl_history:
            return 0.0
        
        cumulative_pnl = [entry['cumulative'] for entry in self.pnl_history]
        peak = max(cumulative_pnl)
        current = cumulative_pnl[-1]
        
        if peak > 0:
            drawdown = (peak - current) / peak
            return min(1.0, drawdown / 0.2)  # Нормализация к 20% максимальной просадки
        
        return 0.0
    
    async def _calculate_var(self, account_balance: float) -> Tuple[float, float]:
        """Расчет Value at Risk"""
        if len(self.pnl_history) < 30:
            return 0.0, 0.0
        
        daily_returns = [entry['pnl'] / account_balance for entry in self.pnl_history[-30:]]
        
        # VaR на уровне 95%
        var_1d = abs(np.percentile(daily_returns, 5)) * account_balance
        var_7d = var_1d * np.sqrt(7)  # Масштабирование на неделю
        
        return var_1d, var_7d
    
    async def _calculate_max_drawdown(self) -> float:
        """Расчет максимальной просадки"""
        if not self.pnl_history:
            return 0.0
        
        cumulative_pnl = [entry['cumulative'] for entry in self.pnl_history]
        peak = cumulative_pnl[0]
        max_drawdown = 0.0
        
        for pnl in cumulative_pnl:
            if pnl > peak:
                peak = pnl
            drawdown = (peak - pnl) / max(peak, 1)
            max_drawdown = max(max_drawdown, drawdown)
        
        return max_drawdown
    
    async def _calculate_sharpe_ratio(self) -> float:
        """Расчет коэффициента Шарпа"""
        if len(self.pnl_history) < 30:
            return 0.0
        
        returns = [entry['pnl'] for entry in self.pnl_history[-30:]]
        
        if not returns:
            return 0.0
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0.0
        
        # Предполагаем безрисковую ставку 2% годовых
        risk_free_rate = 0.02 / 252  # Дневная ставка
        sharpe = (mean_return - risk_free_rate) / std_return
        
        return sharpe * np.sqrt(252)  # Годовой коэффициент Шарпа
    
    async def _determine_risk_level(self, portfolio_risk: float, 
                                  volatility_risk: float, drawdown_risk: float) -> RiskLevel:
        """Определение общего уровня риска"""
        avg_risk = (portfolio_risk + volatility_risk + drawdown_risk) / 3
        
        if avg_risk < 0.2:
            return RiskLevel.VERY_LOW
        elif avg_risk < 0.4:
            return RiskLevel.LOW
        elif avg_risk < 0.6:
            return RiskLevel.MEDIUM
        elif avg_risk < 0.8:
            return RiskLevel.HIGH
        else:
            return RiskLevel.VERY_HIGH
    
    def _get_default_metrics(self) -> RiskMetrics:
        """Метрики по умолчанию"""
        return RiskMetrics(
            portfolio_risk=0.0,
            position_risk=0.0,
            correlation_risk=0.0,
            volatility_risk=0.5,
            drawdown_risk=0.0,
            var_1d=0.0,
            var_7d=0.0,
            max_drawdown=0.0,
            sharpe_ratio=0.0,
            risk_level=RiskLevel.LOW,
            timestamp=datetime.now()
        )
    
    def get_current_positions(self) -> Dict[str, Position]:
        """Получение текущих позиций"""
        with self.lock:
            return self.positions.copy()
    
    def get_risk_status(self) -> Dict[str, Any]:
        """Получение статуса риска"""
        return {
            'positions_count': len(self.positions),
            'max_positions': self.risk_limits.max_open_positions,
            'daily_pnl': self.daily_pnl.get(datetime.now().strftime('%Y-%m-%d'), 0.0),
            'weekly_pnl': self.weekly_pnl.get(datetime.now().strftime('%Y-W%U'), 0.0),
            'current_metrics': self.current_metrics,
            'risk_limits': self.risk_limits
        }