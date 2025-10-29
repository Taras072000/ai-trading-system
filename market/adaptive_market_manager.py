"""
Адаптивная система управления под разные типы рынков
Автоматическое определение рыночных условий и динамическая адаптация стратегий
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import logging
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json

# Технические индикаторы
import talib

class MarketType(Enum):
    """Типы рыночных условий"""
    TRENDING_BULL = "trending_bull"      # Восходящий тренд
    TRENDING_BEAR = "trending_bear"      # Нисходящий тренд
    RANGING = "ranging"                  # Боковое движение
    VOLATILE = "volatile"                # Высокая волатильность
    BREAKOUT = "breakout"                # Пробой уровней
    CONSOLIDATION = "consolidation"      # Консолидация
    REVERSAL = "reversal"                # Разворот тренда

class MarketRegime(Enum):
    """Рыночные режимы"""
    BULL_MARKET = "bull_market"
    BEAR_MARKET = "bear_market"
    SIDEWAYS_MARKET = "sideways_market"
    CRISIS_MARKET = "crisis_market"

@dataclass
class MarketCondition:
    """Состояние рынка"""
    market_type: MarketType
    regime: MarketRegime
    volatility_level: float
    trend_strength: float
    momentum: float
    volume_profile: str
    support_resistance: Dict[str, float]
    confidence: float
    timestamp: datetime

@dataclass
class StrategyAdaptation:
    """Адаптация стратегии под рыночные условия"""
    strategy_name: str
    parameters: Dict[str, Any]
    risk_multiplier: float
    position_sizing: float
    stop_loss_adjustment: float
    take_profit_adjustment: float
    entry_conditions: List[str]
    exit_conditions: List[str]
    active: bool

@dataclass
class PortfolioAllocation:
    """Распределение портфеля"""
    symbol: str
    allocation_percent: float
    risk_weight: float
    correlation_factor: float
    market_beta: float
    expected_return: float
    max_position_size: float

class AdaptiveMarketManager:
    """Главный класс адаптивного управления рынком"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Параметры для определения типа рынка
        self.market_detection_params = {
            'trend_threshold': 0.02,      # 2% для определения тренда
            'volatility_threshold': 0.03,  # 3% для высокой волатильности
            'ranging_threshold': 0.015,    # 1.5% для бокового движения
            'breakout_threshold': 0.025,   # 2.5% для пробоя
            'volume_threshold': 1.5,       # 150% от среднего объема
            'trend_periods': [20, 50, 200] # Периоды для анализа тренда
        }
        
        # Стратегии для разных типов рынка
        self.market_strategies = {
            MarketType.TRENDING_BULL: {
                'strategy': 'trend_following',
                'risk_multiplier': 1.2,
                'position_sizing': 0.8,
                'stop_loss': 0.02,
                'take_profit': 0.06,
                'indicators': ['EMA', 'MACD', 'RSI'],
                'entry_conditions': ['price_above_ema', 'macd_bullish', 'rsi_oversold_recovery']
            },
            MarketType.TRENDING_BEAR: {
                'strategy': 'short_selling',
                'risk_multiplier': 1.1,
                'position_sizing': 0.6,
                'stop_loss': 0.025,
                'take_profit': 0.05,
                'indicators': ['EMA', 'MACD', 'RSI'],
                'entry_conditions': ['price_below_ema', 'macd_bearish', 'rsi_overbought_decline']
            },
            MarketType.RANGING: {
                'strategy': 'mean_reversion',
                'risk_multiplier': 0.8,
                'position_sizing': 0.5,
                'stop_loss': 0.015,
                'take_profit': 0.03,
                'indicators': ['BB', 'RSI', 'Stochastic'],
                'entry_conditions': ['price_at_bb_bands', 'rsi_extreme', 'stoch_oversold_overbought']
            },
            MarketType.VOLATILE: {
                'strategy': 'volatility_breakout',
                'risk_multiplier': 0.6,
                'position_sizing': 0.3,
                'stop_loss': 0.03,
                'take_profit': 0.08,
                'indicators': ['ATR', 'BB', 'VIX'],
                'entry_conditions': ['high_volatility', 'bb_squeeze_release', 'volume_spike']
            },
            MarketType.BREAKOUT: {
                'strategy': 'momentum_breakout',
                'risk_multiplier': 1.5,
                'position_sizing': 1.0,
                'stop_loss': 0.02,
                'take_profit': 0.1,
                'indicators': ['Volume', 'ATR', 'Support_Resistance'],
                'entry_conditions': ['level_breakout', 'volume_confirmation', 'momentum_acceleration']
            },
            MarketType.CONSOLIDATION: {
                'strategy': 'range_trading',
                'risk_multiplier': 0.7,
                'position_sizing': 0.4,
                'stop_loss': 0.01,
                'take_profit': 0.025,
                'indicators': ['Support_Resistance', 'RSI', 'MACD'],
                'entry_conditions': ['price_at_support_resistance', 'rsi_neutral', 'low_volatility']
            },
            MarketType.REVERSAL: {
                'strategy': 'reversal_trading',
                'risk_multiplier': 1.0,
                'position_sizing': 0.6,
                'stop_loss': 0.025,
                'take_profit': 0.07,
                'indicators': ['Divergence', 'RSI', 'MACD', 'Volume'],
                'entry_conditions': ['divergence_signal', 'rsi_reversal', 'volume_confirmation']
            }
        }
        
        # Корреляционная матрица активов (пример)
        self.correlation_matrix = {}
        
        # Портфельные веса
        self.portfolio_weights = {}
        
        # История рыночных условий
        self.market_history = []
        
        self.executor = ThreadPoolExecutor(max_workers=4)
        
    async def analyze_market_conditions(self, 
                                      market_data: Dict[str, pd.DataFrame],
                                      symbols: List[str]) -> Dict[str, MarketCondition]:
        """
        Анализ рыночных условий для множественных активов
        
        Args:
            market_data: Данные по активам
            symbols: Список символов для анализа
            
        Returns:
            Словарь с условиями рынка для каждого символа
        """
        self.logger.info(f"Анализ рыночных условий для {len(symbols)} активов")
        
        market_conditions = {}
        
        # Параллельный анализ всех активов
        tasks = []
        for symbol in symbols:
            if symbol in market_data:
                task = asyncio.get_event_loop().run_in_executor(
                    self.executor, self._analyze_single_market, symbol, market_data[symbol]
                )
                tasks.append((symbol, task))
        
        # Получение результатов
        for symbol, task in tasks:
            try:
                condition = await task
                market_conditions[symbol] = condition
            except Exception as e:
                self.logger.error(f"Ошибка анализа рынка для {symbol}: {e}")
                market_conditions[symbol] = self._create_default_condition()
        
        # Анализ общих рыночных условий
        overall_condition = self._analyze_overall_market(market_conditions)
        
        # Обновление истории
        self.market_history.append({
            'timestamp': datetime.now(),
            'conditions': market_conditions,
            'overall': overall_condition
        })
        
        return market_conditions
    
    def _analyze_single_market(self, symbol: str, data: pd.DataFrame) -> MarketCondition:
        """Анализ рыночных условий для одного актива"""
        
        if len(data) < 200:
            return self._create_default_condition()
        
        try:
            # Расчет технических индикаторов
            indicators = self._calculate_market_indicators(data)
            
            # Определение типа рынка
            market_type = self._determine_market_type(indicators, data)
            
            # Определение режима рынка
            regime = self._determine_market_regime(indicators, data)
            
            # Анализ волатильности
            volatility_level = self._analyze_volatility(data)
            
            # Сила тренда
            trend_strength = self._calculate_trend_strength(indicators)
            
            # Моментум
            momentum = self._calculate_momentum(indicators)
            
            # Объемный профиль
            volume_profile = self._analyze_volume_profile(data)
            
            # Уровни поддержки и сопротивления
            support_resistance = self._find_support_resistance_levels(data)
            
            # Уверенность в определении
            confidence = self._calculate_market_confidence(indicators, market_type)
            
            return MarketCondition(
                market_type=market_type,
                regime=regime,
                volatility_level=volatility_level,
                trend_strength=trend_strength,
                momentum=momentum,
                volume_profile=volume_profile,
                support_resistance=support_resistance,
                confidence=confidence,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Ошибка в анализе рынка для {symbol}: {e}")
            return self._create_default_condition()
    
    def _calculate_market_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Расчет индикаторов для анализа рынка"""
        
        close = data['close'].values
        high = data['high'].values
        low = data['low'].values
        volume = data['volume'].values if 'volume' in data.columns else np.ones(len(close))
        
        indicators = {}
        
        try:
            # Трендовые индикаторы
            indicators['ema_20'] = talib.EMA(close, timeperiod=20)
            indicators['ema_50'] = talib.EMA(close, timeperiod=50)
            indicators['ema_200'] = talib.EMA(close, timeperiod=200)
            indicators['sma_20'] = talib.SMA(close, timeperiod=20)
            indicators['sma_50'] = talib.SMA(close, timeperiod=50)
            
            # MACD
            macd, macd_signal, macd_hist = talib.MACD(close)
            indicators['macd'] = macd
            indicators['macd_signal'] = macd_signal
            indicators['macd_histogram'] = macd_hist
            
            # Осцилляторы
            indicators['rsi'] = talib.RSI(close, timeperiod=14)
            indicators['stoch_k'], indicators['stoch_d'] = talib.STOCH(high, low, close)
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = talib.BBANDS(close)
            indicators['bb_upper'] = bb_upper
            indicators['bb_middle'] = bb_middle
            indicators['bb_lower'] = bb_lower
            indicators['bb_width'] = (bb_upper - bb_lower) / bb_middle
            
            # Волатильность
            indicators['atr'] = talib.ATR(high, low, close, timeperiod=14)
            indicators['volatility'] = np.std(np.diff(np.log(close[-20:]))) * np.sqrt(252)
            
            # ADX для силы тренда
            indicators['adx'] = talib.ADX(high, low, close, timeperiod=14)
            indicators['plus_di'] = talib.PLUS_DI(high, low, close, timeperiod=14)
            indicators['minus_di'] = talib.MINUS_DI(high, low, close, timeperiod=14)
            
            # Объемные индикаторы
            indicators['volume_sma'] = talib.SMA(volume, timeperiod=20)
            indicators['volume_ratio'] = volume / indicators['volume_sma']
            indicators['obv'] = talib.OBV(close, volume)
            
            # Дополнительные индикаторы
            indicators['cci'] = talib.CCI(high, low, close, timeperiod=14)
            indicators['williams_r'] = talib.WILLR(high, low, close, timeperiod=14)
            
            # Ценовые изменения
            indicators['price_change_1d'] = (close[-1] - close[-2]) / close[-2] if len(close) > 1 else 0
            indicators['price_change_5d'] = (close[-1] - close[-6]) / close[-6] if len(close) > 5 else 0
            indicators['price_change_20d'] = (close[-1] - close[-21]) / close[-21] if len(close) > 20 else 0
            
        except Exception as e:
            self.logger.error(f"Ошибка расчета индикаторов: {e}")
            # Возвращаем базовые значения
            indicators = {
                'ema_20': close, 'rsi': np.full(len(close), 50),
                'atr': np.full(len(close), close[-1] * 0.02),
                'adx': np.full(len(close), 25)
            }
        
        return indicators
    
    def _determine_market_type(self, indicators: Dict[str, Any], data: pd.DataFrame) -> MarketType:
        """Определение типа рынка"""
        
        close = data['close'].values
        
        # Анализ тренда
        trend_score = 0
        if len(indicators['ema_20']) > 0 and len(indicators['ema_50']) > 0:
            if indicators['ema_20'][-1] > indicators['ema_50'][-1]:
                trend_score += 1
            elif indicators['ema_20'][-1] < indicators['ema_50'][-1]:
                trend_score -= 1
        
        if len(indicators['ema_50']) > 0 and len(indicators['ema_200']) > 0:
            if indicators['ema_50'][-1] > indicators['ema_200'][-1]:
                trend_score += 1
            elif indicators['ema_50'][-1] < indicators['ema_200'][-1]:
                trend_score -= 1
        
        # Сила тренда
        adx = indicators['adx'][-1] if len(indicators['adx']) > 0 else 25
        
        # Волатильность
        volatility = indicators['volatility'] if 'volatility' in indicators else 0.02
        
        # Ценовые изменения
        price_change_20d = abs(indicators.get('price_change_20d', 0))
        
        # Bollinger Bands ширина
        bb_width = indicators['bb_width'][-1] if len(indicators['bb_width']) > 0 else 0.04
        
        # Определение типа рынка
        if adx > 30 and trend_score >= 1:
            if indicators.get('price_change_20d', 0) > 0:
                return MarketType.TRENDING_BULL
            else:
                return MarketType.TRENDING_BEAR
        
        elif volatility > self.market_detection_params['volatility_threshold']:
            return MarketType.VOLATILE
        
        elif price_change_20d > self.market_detection_params['breakout_threshold']:
            return MarketType.BREAKOUT
        
        elif bb_width < 0.02:  # Узкие Bollinger Bands
            return MarketType.CONSOLIDATION
        
        elif adx < 20 and price_change_20d < self.market_detection_params['ranging_threshold']:
            return MarketType.RANGING
        
        else:
            # Проверка на разворот
            rsi = indicators['rsi'][-1] if len(indicators['rsi']) > 0 else 50
            macd_hist = indicators['macd_histogram'][-1] if len(indicators['macd_histogram']) > 0 else 0
            
            if (rsi > 70 and macd_hist < 0) or (rsi < 30 and macd_hist > 0):
                return MarketType.REVERSAL
            
            return MarketType.RANGING
    
    def _determine_market_regime(self, indicators: Dict[str, Any], data: pd.DataFrame) -> MarketRegime:
        """Определение режима рынка"""
        
        # Долгосрочный тренд
        long_term_change = indicators.get('price_change_20d', 0)
        
        # Волатильность
        volatility = indicators.get('volatility', 0.02)
        
        # ADX
        adx = indicators['adx'][-1] if len(indicators['adx']) > 0 else 25
        
        # Определение режима
        if volatility > 0.05:  # Очень высокая волатильность
            return MarketRegime.CRISIS_MARKET
        elif long_term_change > 0.1 and adx > 25:  # Сильный восходящий тренд
            return MarketRegime.BULL_MARKET
        elif long_term_change < -0.1 and adx > 25:  # Сильный нисходящий тренд
            return MarketRegime.BEAR_MARKET
        else:
            return MarketRegime.SIDEWAYS_MARKET
    
    def _analyze_volatility(self, data: pd.DataFrame) -> float:
        """Анализ уровня волатильности"""
        
        close = data['close'].values
        
        if len(close) < 20:
            return 0.02
        
        # Историческая волатильность
        returns = np.diff(np.log(close))
        volatility = np.std(returns) * np.sqrt(252)  # Годовая волатильность
        
        return volatility
    
    def _calculate_trend_strength(self, indicators: Dict[str, Any]) -> float:
        """Расчет силы тренда"""
        
        adx = indicators['adx'][-1] if len(indicators['adx']) > 0 else 25
        
        # Нормализация ADX (0-100) к (0-1)
        trend_strength = min(1.0, adx / 100.0)
        
        return trend_strength
    
    def _calculate_momentum(self, indicators: Dict[str, Any]) -> float:
        """Расчет моментума"""
        
        rsi = indicators['rsi'][-1] if len(indicators['rsi']) > 0 else 50
        macd_hist = indicators['macd_histogram'][-1] if len(indicators['macd_histogram']) > 0 else 0
        
        # Нормализация RSI к (-1, 1)
        rsi_momentum = (rsi - 50) / 50
        
        # Нормализация MACD гистограммы
        macd_momentum = np.tanh(macd_hist * 100)  # Используем tanh для ограничения
        
        # Комбинированный моментум
        momentum = (rsi_momentum + macd_momentum) / 2
        
        return momentum
    
    def _analyze_volume_profile(self, data: pd.DataFrame) -> str:
        """Анализ объемного профиля"""
        
        if 'volume' not in data.columns:
            return 'normal'
        
        volume = data['volume'].values
        
        if len(volume) < 20:
            return 'normal'
        
        # Средний объем за последние 20 периодов
        avg_volume = np.mean(volume[-20:])
        current_volume = volume[-1]
        
        volume_ratio = current_volume / avg_volume
        
        if volume_ratio > 2.0:
            return 'high'
        elif volume_ratio > 1.5:
            return 'above_average'
        elif volume_ratio < 0.5:
            return 'low'
        else:
            return 'normal'
    
    def _find_support_resistance_levels(self, data: pd.DataFrame) -> Dict[str, float]:
        """Поиск уровней поддержки и сопротивления"""
        
        high = data['high'].values
        low = data['low'].values
        close = data['close'].values
        
        if len(high) < 50:
            return {'support': close[-1] * 0.98, 'resistance': close[-1] * 1.02}
        
        # Поиск локальных максимумов и минимумов
        from scipy.signal import argrelextrema
        
        # Локальные максимумы (сопротивления)
        max_indices = argrelextrema(high, np.greater, order=5)[0]
        resistance_levels = high[max_indices] if len(max_indices) > 0 else [close[-1] * 1.02]
        
        # Локальные минимумы (поддержки)
        min_indices = argrelextrema(low, np.less, order=5)[0]
        support_levels = low[min_indices] if len(min_indices) > 0 else [close[-1] * 0.98]
        
        # Ближайшие уровни
        current_price = close[-1]
        
        # Ближайшее сопротивление выше текущей цены
        resistance_above = [r for r in resistance_levels if r > current_price]
        nearest_resistance = min(resistance_above) if resistance_above else current_price * 1.02
        
        # Ближайшая поддержка ниже текущей цены
        support_below = [s for s in support_levels if s < current_price]
        nearest_support = max(support_below) if support_below else current_price * 0.98
        
        return {
            'support': nearest_support,
            'resistance': nearest_resistance,
            'support_strength': len([s for s in support_levels if abs(s - nearest_support) < current_price * 0.005]),
            'resistance_strength': len([r for r in resistance_levels if abs(r - nearest_resistance) < current_price * 0.005])
        }
    
    def _calculate_market_confidence(self, indicators: Dict[str, Any], market_type: MarketType) -> float:
        """Расчет уверенности в определении типа рынка"""
        
        confidence = 0.5  # Базовая уверенность
        
        # ADX для силы тренда
        adx = indicators['adx'][-1] if len(indicators['adx']) > 0 else 25
        confidence += min(0.3, adx / 100)
        
        # Согласованность индикаторов
        rsi = indicators['rsi'][-1] if len(indicators['rsi']) > 0 else 50
        macd_hist = indicators['macd_histogram'][-1] if len(indicators['macd_histogram']) > 0 else 0
        
        # Проверка согласованности для трендовых рынков
        if market_type in [MarketType.TRENDING_BULL, MarketType.TRENDING_BEAR]:
            if market_type == MarketType.TRENDING_BULL and rsi > 50 and macd_hist > 0:
                confidence += 0.2
            elif market_type == MarketType.TRENDING_BEAR and rsi < 50 and macd_hist < 0:
                confidence += 0.2
        
        # Объемное подтверждение
        volume_ratio = indicators.get('volume_ratio', np.array([1.0]))[-1]
        if volume_ratio > 1.2:
            confidence += 0.1
        
        return max(0.0, min(1.0, confidence))
    
    def _analyze_overall_market(self, market_conditions: Dict[str, MarketCondition]) -> Dict[str, Any]:
        """Анализ общих рыночных условий"""
        
        if not market_conditions:
            return {'regime': MarketRegime.SIDEWAYS_MARKET, 'confidence': 0.5}
        
        # Подсчет типов рынков
        market_types = [condition.market_type for condition in market_conditions.values()]
        regime_counts = {}
        
        for condition in market_conditions.values():
            regime = condition.regime
            regime_counts[regime] = regime_counts.get(regime, 0) + 1
        
        # Доминирующий режим
        dominant_regime = max(regime_counts, key=regime_counts.get)
        
        # Средняя волатильность
        avg_volatility = np.mean([condition.volatility_level for condition in market_conditions.values()])
        
        # Средняя уверенность
        avg_confidence = np.mean([condition.confidence for condition in market_conditions.values()])
        
        return {
            'regime': dominant_regime,
            'volatility': avg_volatility,
            'confidence': avg_confidence,
            'market_types': market_types,
            'regime_distribution': regime_counts
        }
    
    async def adapt_strategies(self, 
                             market_conditions: Dict[str, MarketCondition],
                             current_strategies: Dict[str, Any]) -> Dict[str, StrategyAdaptation]:
        """
        Адаптация стратегий под рыночные условия
        
        Args:
            market_conditions: Условия рынка для каждого актива
            current_strategies: Текущие стратегии
            
        Returns:
            Адаптированные стратегии
        """
        self.logger.info("Адаптация стратегий под рыночные условия")
        
        adapted_strategies = {}
        
        for symbol, condition in market_conditions.items():
            try:
                # Получение базовой стратегии для типа рынка
                base_strategy = self.market_strategies.get(condition.market_type, 
                                                         self.market_strategies[MarketType.RANGING])
                
                # Адаптация параметров под конкретные условия
                adapted_params = self._adapt_strategy_parameters(base_strategy, condition)
                
                # Создание адаптированной стратегии
                strategy_adaptation = StrategyAdaptation(
                    strategy_name=f"{base_strategy['strategy']}_{symbol}",
                    parameters=adapted_params,
                    risk_multiplier=self._calculate_risk_multiplier(condition),
                    position_sizing=self._calculate_position_sizing(condition),
                    stop_loss_adjustment=self._calculate_stop_loss_adjustment(condition),
                    take_profit_adjustment=self._calculate_take_profit_adjustment(condition),
                    entry_conditions=base_strategy['entry_conditions'],
                    exit_conditions=self._generate_exit_conditions(condition),
                    active=condition.confidence > 0.6  # Активируем только при высокой уверенности
                )
                
                adapted_strategies[symbol] = strategy_adaptation
                
            except Exception as e:
                self.logger.error(f"Ошибка адаптации стратегии для {symbol}: {e}")
        
        return adapted_strategies
    
    def _adapt_strategy_parameters(self, base_strategy: Dict[str, Any], condition: MarketCondition) -> Dict[str, Any]:
        """Адаптация параметров стратегии"""
        
        params = base_strategy.copy()
        
        # Корректировка на основе волатильности
        volatility_multiplier = 1 + (condition.volatility_level - 0.02) / 0.02
        params['stop_loss'] *= volatility_multiplier
        params['take_profit'] *= volatility_multiplier
        
        # Корректировка на основе силы тренда
        if condition.trend_strength > 0.7:
            params['take_profit'] *= 1.2  # Увеличиваем цели при сильном тренде
        elif condition.trend_strength < 0.3:
            params['stop_loss'] *= 0.8   # Уменьшаем стопы при слабом тренде
        
        # Корректировка на основе моментума
        if abs(condition.momentum) > 0.5:
            params['position_sizing'] *= 1.1  # Увеличиваем позицию при сильном моментуме
        
        return params
    
    def _calculate_risk_multiplier(self, condition: MarketCondition) -> float:
        """Расчет множителя риска"""
        
        base_multiplier = 1.0
        
        # Корректировка на волатильность
        if condition.volatility_level > 0.05:
            base_multiplier *= 0.7  # Снижаем риск при высокой волатильности
        elif condition.volatility_level < 0.02:
            base_multiplier *= 1.2  # Увеличиваем риск при низкой волатильности
        
        # Корректировка на уверенность
        base_multiplier *= condition.confidence
        
        # Корректировка на режим рынка
        if condition.regime == MarketRegime.CRISIS_MARKET:
            base_multiplier *= 0.5
        elif condition.regime == MarketRegime.BULL_MARKET:
            base_multiplier *= 1.1
        
        return max(0.1, min(2.0, base_multiplier))
    
    def _calculate_position_sizing(self, condition: MarketCondition) -> float:
        """Расчет размера позиции"""
        
        base_size = 0.5  # 50% базовый размер
        
        # Корректировка на основе уверенности
        size_multiplier = 0.5 + condition.confidence
        
        # Корректировка на волатильность
        volatility_adjustment = 1 - (condition.volatility_level - 0.02) / 0.1
        volatility_adjustment = max(0.3, min(1.5, volatility_adjustment))
        
        final_size = base_size * size_multiplier * volatility_adjustment
        
        return max(0.1, min(1.0, final_size))
    
    def _calculate_stop_loss_adjustment(self, condition: MarketCondition) -> float:
        """Расчет корректировки стоп-лосса"""
        
        base_stop = 0.02  # 2% базовый стоп
        
        # Корректировка на волатильность
        volatility_multiplier = 1 + condition.volatility_level / 0.02
        
        # Корректировка на силу тренда
        trend_multiplier = 1 + condition.trend_strength * 0.5
        
        adjusted_stop = base_stop * volatility_multiplier * trend_multiplier
        
        return max(0.005, min(0.1, adjusted_stop))
    
    def _calculate_take_profit_adjustment(self, condition: MarketCondition) -> float:
        """Расчет корректировки тейк-профита"""
        
        base_tp = 0.04  # 4% базовый тейк-профит
        
        # Корректировка на силу тренда
        if condition.trend_strength > 0.7:
            trend_multiplier = 1.5
        elif condition.trend_strength < 0.3:
            trend_multiplier = 0.8
        else:
            trend_multiplier = 1.0
        
        # Корректировка на моментум
        momentum_multiplier = 1 + abs(condition.momentum) * 0.5
        
        adjusted_tp = base_tp * trend_multiplier * momentum_multiplier
        
        return max(0.01, min(0.2, adjusted_tp))
    
    def _generate_exit_conditions(self, condition: MarketCondition) -> List[str]:
        """Генерация условий выхода"""
        
        exit_conditions = ['stop_loss_hit', 'take_profit_hit']
        
        # Дополнительные условия в зависимости от типа рынка
        if condition.market_type == MarketType.VOLATILE:
            exit_conditions.extend(['volatility_spike', 'volume_dry_up'])
        
        elif condition.market_type in [MarketType.TRENDING_BULL, MarketType.TRENDING_BEAR]:
            exit_conditions.extend(['trend_reversal', 'momentum_divergence'])
        
        elif condition.market_type == MarketType.RANGING:
            exit_conditions.extend(['range_breakout', 'time_based_exit'])
        
        return exit_conditions
    
    async def optimize_portfolio_allocation(self, 
                                          market_conditions: Dict[str, MarketCondition],
                                          symbols: List[str],
                                          current_allocations: Dict[str, float]) -> Dict[str, PortfolioAllocation]:
        """
        Оптимизация распределения портфеля
        
        Args:
            market_conditions: Условия рынка
            symbols: Список активов
            current_allocations: Текущие распределения
            
        Returns:
            Оптимизированные распределения
        """
        self.logger.info("Оптимизация портфельного распределения")
        
        # Расчет корреляций между активами
        correlations = await self._calculate_asset_correlations(symbols)
        
        # Расчет ожидаемых доходностей
        expected_returns = self._calculate_expected_returns(market_conditions)
        
        # Расчет рисков
        risk_metrics = self._calculate_risk_metrics(market_conditions)
        
        # Оптимизация по Марковицу (упрощенная версия)
        optimal_weights = self._optimize_portfolio_weights(
            expected_returns, risk_metrics, correlations
        )
        
        # Создание распределений
        allocations = {}
        for symbol in symbols:
            if symbol in market_conditions:
                condition = market_conditions[symbol]
                
                allocation = PortfolioAllocation(
                    symbol=symbol,
                    allocation_percent=optimal_weights.get(symbol, 0.0),
                    risk_weight=risk_metrics.get(symbol, 0.5),
                    correlation_factor=self._calculate_avg_correlation(symbol, correlations),
                    market_beta=self._calculate_market_beta(condition),
                    expected_return=expected_returns.get(symbol, 0.0),
                    max_position_size=self._calculate_max_position_size(condition)
                )
                
                allocations[symbol] = allocation
        
        return allocations
    
    async def _calculate_asset_correlations(self, symbols: List[str]) -> Dict[Tuple[str, str], float]:
        """Расчет корреляций между активами"""
        
        # В реальной системе здесь был бы расчет корреляций на основе исторических данных
        # Для примера используем случайные корреляции
        correlations = {}
        
        for i, symbol1 in enumerate(symbols):
            for j, symbol2 in enumerate(symbols):
                if i < j:
                    # Генерируем реалистичные корреляции
                    correlation = np.random.uniform(-0.3, 0.7)
                    correlations[(symbol1, symbol2)] = correlation
                    correlations[(symbol2, symbol1)] = correlation
                elif i == j:
                    correlations[(symbol1, symbol2)] = 1.0
        
        return correlations
    
    def _calculate_expected_returns(self, market_conditions: Dict[str, MarketCondition]) -> Dict[str, float]:
        """Расчет ожидаемых доходностей"""
        
        expected_returns = {}
        
        for symbol, condition in market_conditions.items():
            # Базовая доходность на основе типа рынка
            base_return = 0.0
            
            if condition.market_type == MarketType.TRENDING_BULL:
                base_return = 0.08
            elif condition.market_type == MarketType.TRENDING_BEAR:
                base_return = -0.05
            elif condition.market_type == MarketType.BREAKOUT:
                base_return = 0.12
            elif condition.market_type == MarketType.VOLATILE:
                base_return = 0.03
            else:
                base_return = 0.02
            
            # Корректировка на моментум и силу тренда
            momentum_adjustment = condition.momentum * 0.05
            trend_adjustment = condition.trend_strength * 0.03
            
            # Корректировка на уверенность
            confidence_adjustment = (condition.confidence - 0.5) * 0.02
            
            expected_return = base_return + momentum_adjustment + trend_adjustment + confidence_adjustment
            expected_returns[symbol] = expected_return
        
        return expected_returns
    
    def _calculate_risk_metrics(self, market_conditions: Dict[str, MarketCondition]) -> Dict[str, float]:
        """Расчет метрик риска"""
        
        risk_metrics = {}
        
        for symbol, condition in market_conditions.items():
            # Базовый риск на основе волатильности
            base_risk = condition.volatility_level
            
            # Корректировка на режим рынка
            if condition.regime == MarketRegime.CRISIS_MARKET:
                regime_multiplier = 2.0
            elif condition.regime == MarketRegime.BULL_MARKET:
                regime_multiplier = 0.8
            elif condition.regime == MarketRegime.BEAR_MARKET:
                regime_multiplier = 1.2
            else:
                regime_multiplier = 1.0
            
            # Корректировка на уверенность (низкая уверенность = высокий риск)
            confidence_multiplier = 2.0 - condition.confidence
            
            total_risk = base_risk * regime_multiplier * confidence_multiplier
            risk_metrics[symbol] = min(1.0, total_risk)
        
        return risk_metrics
    
    def _optimize_portfolio_weights(self, 
                                  expected_returns: Dict[str, float],
                                  risk_metrics: Dict[str, float],
                                  correlations: Dict[Tuple[str, str], float]) -> Dict[str, float]:
        """Оптимизация весов портфеля (упрощенная версия)"""
        
        symbols = list(expected_returns.keys())
        n_assets = len(symbols)
        
        if n_assets == 0:
            return {}
        
        # Равномерное распределение как базовая линия
        equal_weight = 1.0 / n_assets
        
        # Корректировка весов на основе соотношения доходность/риск
        adjusted_weights = {}
        total_score = 0
        
        for symbol in symbols:
            expected_return = expected_returns[symbol]
            risk = risk_metrics[symbol]
            
            # Sharpe-подобный скор
            score = expected_return / max(risk, 0.01)  # Избегаем деления на ноль
            adjusted_weights[symbol] = max(0, score)  # Только положительные веса
            total_score += adjusted_weights[symbol]
        
        # Нормализация весов
        if total_score > 0:
            for symbol in symbols:
                adjusted_weights[symbol] /= total_score
        else:
            # Если все скоры отрицательные, используем равномерное распределение
            for symbol in symbols:
                adjusted_weights[symbol] = equal_weight
        
        # Ограничение максимального веса одного актива
        max_weight = 0.4
        for symbol in symbols:
            adjusted_weights[symbol] = min(adjusted_weights[symbol], max_weight)
        
        # Перенормализация после ограничения
        total_weight = sum(adjusted_weights.values())
        if total_weight > 0:
            for symbol in symbols:
                adjusted_weights[symbol] /= total_weight
        
        return adjusted_weights
    
    def _calculate_avg_correlation(self, symbol: str, correlations: Dict[Tuple[str, str], float]) -> float:
        """Расчет средней корреляции актива с остальными"""
        
        correlations_list = []
        for (s1, s2), corr in correlations.items():
            if s1 == symbol and s1 != s2:
                correlations_list.append(corr)
            elif s2 == symbol and s1 != s2:
                correlations_list.append(corr)
        
        return np.mean(correlations_list) if correlations_list else 0.0
    
    def _calculate_market_beta(self, condition: MarketCondition) -> float:
        """Расчет бета-коэффициента относительно рынка"""
        
        # Упрощенный расчет беты на основе характеристик рынка
        base_beta = 1.0
        
        # Корректировка на волатильность
        volatility_adjustment = condition.volatility_level / 0.02  # Нормализация к 2%
        
        # Корректировка на силу тренда
        trend_adjustment = 1 + (condition.trend_strength - 0.5) * 0.5
        
        beta = base_beta * volatility_adjustment * trend_adjustment
        
        return max(0.1, min(3.0, beta))
    
    def _calculate_max_position_size(self, condition: MarketCondition) -> float:
        """Расчет максимального размера позиции"""
        
        base_max_size = 0.2  # 20% базовый максимум
        
        # Корректировка на уверенность
        confidence_multiplier = 0.5 + condition.confidence
        
        # Корректировка на волатильность (обратная зависимость)
        volatility_multiplier = max(0.5, 1 - condition.volatility_level / 0.1)
        
        max_size = base_max_size * confidence_multiplier * volatility_multiplier
        
        return max(0.05, min(0.5, max_size))
    
    def _create_default_condition(self) -> MarketCondition:
        """Создание условий рынка по умолчанию"""
        return MarketCondition(
            market_type=MarketType.RANGING,
            regime=MarketRegime.SIDEWAYS_MARKET,
            volatility_level=0.02,
            trend_strength=0.3,
            momentum=0.0,
            volume_profile='normal',
            support_resistance={'support': 0.0, 'resistance': 0.0},
            confidence=0.5,
            timestamp=datetime.now()
        )

# Пример использования
if __name__ == "__main__":
    # Настройка логирования
    logging.basicConfig(level=logging.INFO)
    
    # Создание менеджера
    manager = AdaptiveMarketManager()
    
    # Пример данных
    sample_data = {
        'BTCUSDT': pd.DataFrame({
            'open': np.random.randn(200) + 50000,
            'high': np.random.randn(200) + 51000,
            'low': np.random.randn(200) + 49000,
            'close': np.random.randn(200) + 50000,
            'volume': np.random.randint(1000, 10000, 200)
        }),
        'ETHUSDT': pd.DataFrame({
            'open': np.random.randn(200) + 3000,
            'high': np.random.randn(200) + 3100,
            'low': np.random.randn(200) + 2900,
            'close': np.random.randn(200) + 3000,
            'volume': np.random.randint(1000, 10000, 200)
        })
    }
    
    symbols = ['BTCUSDT', 'ETHUSDT']
    
    # Запуск анализа
    async def test_analysis():
        # Анализ рыночных условий
        conditions = await manager.analyze_market_conditions(sample_data, symbols)
        print("Рыночные условия:")
        for symbol, condition in conditions.items():
            print(f"{symbol}: {condition.market_type.value}, confidence: {condition.confidence:.3f}")
        
        # Адаптация стратегий
        strategies = await manager.adapt_strategies(conditions, {})
        print("\nАдаптированные стратегии:")
        for symbol, strategy in strategies.items():
            print(f"{symbol}: {strategy.strategy_name}, active: {strategy.active}")
        
        # Оптимизация портфеля
        allocations = await manager.optimize_portfolio_allocation(conditions, symbols, {})
        print("\nРаспределение портфеля:")
        for symbol, allocation in allocations.items():
            print(f"{symbol}: {allocation.allocation_percent:.1%}, risk: {allocation.risk_weight:.3f}")
    
    # asyncio.run(test_analysis())