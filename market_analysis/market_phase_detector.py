"""
Детектор фаз рынка для Peper Binance v4
Архитектурные улучшения - Фаза 2
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import asyncio
from collections import deque
import threading

from config.unified_config_manager import get_config_manager

logger = logging.getLogger(__name__)

class MarketPhase(Enum):
    """Фазы рынка"""
    UPTREND = "uptrend"
    DOWNTREND = "downtrend"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    BREAKOUT = "breakout"
    CONSOLIDATION = "consolidation"

@dataclass
class MarketCondition:
    """Состояние рынка"""
    primary_phase: MarketPhase
    secondary_phases: List[MarketPhase]
    confidence: float
    volatility: float
    trend_strength: float
    volume_profile: str
    timestamp: datetime
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class TechnicalIndicators:
    """Технические индикаторы для анализа фаз рынка"""
    
    @staticmethod
    def calculate_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
        """Расчет Average True Range"""
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        atr = pd.Series(tr).rolling(window=period).mean().values
        return atr
    
    @staticmethod
    def calculate_adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Расчет ADX (Average Directional Index)"""
        # True Range
        tr = TechnicalIndicators.calculate_atr(high, low, close, 1)
        
        # Directional Movement
        dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                          np.maximum(high - np.roll(high, 1), 0), 0)
        dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                           np.maximum(np.roll(low, 1) - low, 0), 0)
        
        # Smoothed values
        tr_smooth = pd.Series(tr).rolling(window=period).mean().values
        dm_plus_smooth = pd.Series(dm_plus).rolling(window=period).mean().values
        dm_minus_smooth = pd.Series(dm_minus).rolling(window=period).mean().values
        
        # Directional Indicators
        di_plus = 100 * dm_plus_smooth / tr_smooth
        di_minus = 100 * dm_minus_smooth / tr_smooth
        
        # ADX
        dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
        adx = pd.Series(dx).rolling(window=period).mean().values
        
        return adx, di_plus, di_minus
    
    @staticmethod
    def calculate_bollinger_bands(close: np.ndarray, period: int = 20, std_dev: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Расчет полос Боллинджера"""
        sma = pd.Series(close).rolling(window=period).mean().values
        std = pd.Series(close).rolling(window=period).std().values
        
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        
        return upper_band, sma, lower_band
    
    @staticmethod
    def calculate_rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
        """Расчет RSI"""
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = pd.Series(gain).rolling(window=period).mean().values
        avg_loss = pd.Series(loss).rolling(window=period).mean().values
        
        rs = avg_gain / np.where(avg_loss != 0, avg_loss, 1e-10)
        rsi = 100 - (100 / (1 + rs))
        
        return np.concatenate([[50], rsi])  # Добавляем первое значение
    
    @staticmethod
    def calculate_macd(close: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Расчет MACD"""
        ema_fast = pd.Series(close).ewm(span=fast).mean().values
        ema_slow = pd.Series(close).ewm(span=slow).mean().values
        
        macd_line = ema_fast - ema_slow
        signal_line = pd.Series(macd_line).ewm(span=signal).mean().values
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram

class MarketPhaseDetector:
    """Детектор фаз рынка"""
    
    def __init__(self):
        self.config_manager = get_config_manager()
        self.phase_config = self.config_manager.get_system_config('market_phases')
        
        # Параметры детекции
        self.volatility_window = self.phase_config.get('detection', {}).get('volatility_window', 20)
        self.trend_window = self.phase_config.get('detection', {}).get('trend_window', 50)
        self.volume_window = self.phase_config.get('detection', {}).get('volume_window', 20)
        
        # Пороги
        thresholds = self.phase_config.get('thresholds', {})
        self.high_volatility_threshold = thresholds.get('high_volatility', 0.03)
        self.low_volatility_threshold = thresholds.get('low_volatility', 0.01)
        self.strong_trend_threshold = thresholds.get('strong_trend', 0.02)
        self.weak_trend_threshold = thresholds.get('weak_trend', 0.005)
        
        # История состояний рынка
        self.market_history: deque = deque(maxlen=1000)
        self.current_condition: Optional[MarketCondition] = None
        
        # Кэш индикаторов
        self.indicator_cache: Dict[str, Any] = {}
        self.cache_timestamp: Optional[datetime] = None
        self.cache_ttl = timedelta(minutes=5)
        
        # Блокировка для потокобезопасности
        self.lock = threading.Lock()
        
        logger.info("Детектор фаз рынка инициализирован")
    
    async def analyze_market_phase(self, market_data: pd.DataFrame) -> MarketCondition:
        """Анализ текущей фазы рынка"""
        try:
            if market_data.empty or len(market_data) < max(self.trend_window, self.volatility_window):
                logger.warning("Недостаточно данных для анализа фазы рынка")
                return self._get_default_condition()
            
            # Подготовка данных
            close = market_data['close'].values
            high = market_data['high'].values
            low = market_data['low'].values
            volume = market_data['volume'].values if 'volume' in market_data.columns else None
            
            # Расчет индикаторов
            indicators = await self._calculate_indicators(high, low, close, volume)
            
            # Определение фаз
            primary_phase = await self._determine_primary_phase(indicators)
            secondary_phases = await self._determine_secondary_phases(indicators)
            
            # Расчет уверенности
            confidence = await self._calculate_confidence(indicators, primary_phase)
            
            # Анализ волатильности
            volatility = indicators['atr'][-1] / close[-1] if len(indicators['atr']) > 0 else 0.02
            
            # Сила тренда
            trend_strength = await self._calculate_trend_strength(indicators)
            
            # Профиль объема
            volume_profile = await self._analyze_volume_profile(volume) if volume is not None else "normal"
            
            # Создание условия рынка
            condition = MarketCondition(
                primary_phase=primary_phase,
                secondary_phases=secondary_phases,
                confidence=confidence,
                volatility=volatility,
                trend_strength=trend_strength,
                volume_profile=volume_profile,
                timestamp=datetime.now(),
                metadata={
                    'indicators': {k: v[-1] if isinstance(v, np.ndarray) and len(v) > 0 else v 
                                 for k, v in indicators.items()},
                    'price': close[-1],
                    'analysis_window': len(market_data)
                }
            )
            
            # Сохранение в историю
            with self.lock:
                self.market_history.append(condition)
                self.current_condition = condition
            
            logger.info(f"Фаза рынка: {primary_phase.value}, уверенность: {confidence:.3f}, "
                       f"волатильность: {volatility:.4f}")
            
            return condition
            
        except Exception as e:
            logger.error(f"Ошибка анализа фазы рынка: {e}")
            return self._get_default_condition()
    
    async def _calculate_indicators(self, high: np.ndarray, low: np.ndarray, 
                                  close: np.ndarray, volume: Optional[np.ndarray]) -> Dict[str, Any]:
        """Расчет технических индикаторов"""
        # Проверка кэша
        current_time = datetime.now()
        if (self.cache_timestamp and 
            current_time - self.cache_timestamp < self.cache_ttl and
            len(self.indicator_cache) > 0):
            return self.indicator_cache
        
        indicators = {}
        
        try:
            # ATR для волатильности
            indicators['atr'] = TechnicalIndicators.calculate_atr(high, low, close, self.volatility_window)
            
            # ADX для силы тренда
            adx, di_plus, di_minus = TechnicalIndicators.calculate_adx(high, low, close, 14)
            indicators['adx'] = adx
            indicators['di_plus'] = di_plus
            indicators['di_minus'] = di_minus
            
            # Bollinger Bands для волатильности и тренда
            bb_upper, bb_middle, bb_lower = TechnicalIndicators.calculate_bollinger_bands(close, 20, 2.0)
            indicators['bb_upper'] = bb_upper
            indicators['bb_middle'] = bb_middle
            indicators['bb_lower'] = bb_lower
            indicators['bb_width'] = (bb_upper - bb_lower) / bb_middle
            
            # RSI для перекупленности/перепроданности
            indicators['rsi'] = TechnicalIndicators.calculate_rsi(close, 14)
            
            # MACD для тренда
            macd, macd_signal, macd_hist = TechnicalIndicators.calculate_macd(close)
            indicators['macd'] = macd
            indicators['macd_signal'] = macd_signal
            indicators['macd_histogram'] = macd_hist
            
            # Скользящие средние
            indicators['sma_20'] = pd.Series(close).rolling(window=20).mean().values
            indicators['sma_50'] = pd.Series(close).rolling(window=50).mean().values
            indicators['ema_12'] = pd.Series(close).ewm(span=12).mean().values
            indicators['ema_26'] = pd.Series(close).ewm(span=26).mean().values
            
            # Анализ объема
            if volume is not None:
                indicators['volume_sma'] = pd.Series(volume).rolling(window=self.volume_window).mean().values
                indicators['volume_ratio'] = volume / indicators['volume_sma']
            
            # Кэширование
            self.indicator_cache = indicators
            self.cache_timestamp = current_time
            
        except Exception as e:
            logger.error(f"Ошибка расчета индикаторов: {e}")
            indicators = self._get_default_indicators()
        
        return indicators
    
    async def _determine_primary_phase(self, indicators: Dict[str, Any]) -> MarketPhase:
        """Определение основной фазы рынка"""
        try:
            # Анализ тренда
            trend_score = 0
            
            # ADX анализ
            if 'adx' in indicators and len(indicators['adx']) > 0:
                adx_value = indicators['adx'][-1]
                di_plus = indicators['di_plus'][-1] if 'di_plus' in indicators else 0
                di_minus = indicators['di_minus'][-1] if 'di_minus' in indicators else 0
                
                if adx_value > 25:  # Сильный тренд
                    if di_plus > di_minus:
                        trend_score += 2
                    else:
                        trend_score -= 2
                elif adx_value < 20:  # Слабый тренд
                    trend_score = 0
            
            # MACD анализ
            if 'macd' in indicators and 'macd_signal' in indicators:
                macd = indicators['macd'][-1] if len(indicators['macd']) > 0 else 0
                macd_signal = indicators['macd_signal'][-1] if len(indicators['macd_signal']) > 0 else 0
                
                if macd > macd_signal and macd > 0:
                    trend_score += 1
                elif macd < macd_signal and macd < 0:
                    trend_score -= 1
            
            # Анализ скользящих средних
            if 'sma_20' in indicators and 'sma_50' in indicators:
                sma_20 = indicators['sma_20'][-1] if len(indicators['sma_20']) > 0 else 0
                sma_50 = indicators['sma_50'][-1] if len(indicators['sma_50']) > 0 else 0
                
                if sma_20 > sma_50:
                    trend_score += 1
                elif sma_20 < sma_50:
                    trend_score -= 1
            
            # Анализ волатильности
            volatility_high = False
            if 'bb_width' in indicators and len(indicators['bb_width']) > 0:
                bb_width = indicators['bb_width'][-1]
                if bb_width > self.high_volatility_threshold:
                    volatility_high = True
            
            # Определение фазы
            if volatility_high:
                return MarketPhase.HIGH_VOLATILITY
            elif abs(trend_score) < 2:
                return MarketPhase.SIDEWAYS
            elif trend_score >= 2:
                return MarketPhase.UPTREND
            elif trend_score <= -2:
                return MarketPhase.DOWNTREND
            else:
                return MarketPhase.CONSOLIDATION
                
        except Exception as e:
            logger.error(f"Ошибка определения основной фазы: {e}")
            return MarketPhase.SIDEWAYS
    
    async def _determine_secondary_phases(self, indicators: Dict[str, Any]) -> List[MarketPhase]:
        """Определение дополнительных фаз рынка"""
        secondary_phases = []
        
        try:
            # Анализ волатильности
            if 'bb_width' in indicators and len(indicators['bb_width']) > 0:
                bb_width = indicators['bb_width'][-1]
                
                if bb_width > self.high_volatility_threshold:
                    secondary_phases.append(MarketPhase.HIGH_VOLATILITY)
                elif bb_width < self.low_volatility_threshold:
                    secondary_phases.append(MarketPhase.LOW_VOLATILITY)
            
            # Анализ прорывов
            if 'bb_upper' in indicators and 'bb_lower' in indicators:
                # Здесь нужна цена закрытия для сравнения
                # Пока добавим логику для определения прорыва
                pass
            
            # Анализ консолидации
            if 'atr' in indicators and len(indicators['atr']) > 1:
                current_atr = indicators['atr'][-1]
                prev_atr = indicators['atr'][-2]
                
                if current_atr < prev_atr * 0.8:  # Снижение волатильности
                    secondary_phases.append(MarketPhase.CONSOLIDATION)
        
        except Exception as e:
            logger.error(f"Ошибка определения дополнительных фаз: {e}")
        
        return secondary_phases
    
    async def _calculate_confidence(self, indicators: Dict[str, Any], primary_phase: MarketPhase) -> float:
        """Расчет уверенности в определении фазы"""
        confidence_factors = []
        
        try:
            # Уверенность на основе ADX
            if 'adx' in indicators and len(indicators['adx']) > 0:
                adx_value = indicators['adx'][-1]
                adx_confidence = min(1.0, adx_value / 50.0)  # Нормализация к 1.0
                confidence_factors.append(adx_confidence)
            
            # Уверенность на основе MACD
            if 'macd' in indicators and 'macd_signal' in indicators:
                macd = indicators['macd'][-1] if len(indicators['macd']) > 0 else 0
                macd_signal = indicators['macd_signal'][-1] if len(indicators['macd_signal']) > 0 else 0
                
                macd_diff = abs(macd - macd_signal)
                macd_confidence = min(1.0, macd_diff * 10)  # Масштабирование
                confidence_factors.append(macd_confidence)
            
            # Уверенность на основе Bollinger Bands
            if 'bb_width' in indicators and len(indicators['bb_width']) > 0:
                bb_width = indicators['bb_width'][-1]
                bb_confidence = min(1.0, bb_width / 0.05)  # Нормализация
                confidence_factors.append(bb_confidence)
            
            # Итоговая уверенность
            if confidence_factors:
                return sum(confidence_factors) / len(confidence_factors)
            else:
                return 0.5  # Нейтральная уверенность
                
        except Exception as e:
            logger.error(f"Ошибка расчета уверенности: {e}")
            return 0.5
    
    async def _calculate_trend_strength(self, indicators: Dict[str, Any]) -> float:
        """Расчет силы тренда"""
        try:
            if 'adx' in indicators and len(indicators['adx']) > 0:
                adx_value = indicators['adx'][-1]
                return min(1.0, adx_value / 50.0)  # Нормализация к 1.0
            return 0.5
        except Exception as e:
            logger.error(f"Ошибка расчета силы тренда: {e}")
            return 0.5
    
    async def _analyze_volume_profile(self, volume: np.ndarray) -> str:
        """Анализ профиля объема"""
        try:
            if len(volume) < self.volume_window:
                return "normal"
            
            recent_volume = volume[-self.volume_window:]
            avg_volume = np.mean(recent_volume)
            current_volume = volume[-1]
            
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
            
            if volume_ratio > 1.5:
                return "high"
            elif volume_ratio < 0.5:
                return "low"
            else:
                return "normal"
                
        except Exception as e:
            logger.error(f"Ошибка анализа объема: {e}")
            return "normal"
    
    def get_current_condition(self) -> Optional[MarketCondition]:
        """Получение текущего состояния рынка"""
        with self.lock:
            return self.current_condition
    
    def get_phase_history(self, limit: int = 100) -> List[MarketCondition]:
        """Получение истории фаз рынка"""
        with self.lock:
            return list(self.market_history)[-limit:]
    
    def get_adaptive_parameters(self, phase: MarketPhase) -> Dict[str, Any]:
        """Получение адаптивных параметров для фазы рынка"""
        phase_config = self.config_manager.get_market_phase_config(phase.value)
        if phase_config:
            return {
                'ai_weights': phase_config.ai_weights,
                'risk_multiplier': phase_config.risk_multiplier,
                'parameters': phase_config.parameters
            }
        return {}
    
    def _get_default_condition(self) -> MarketCondition:
        """Получение состояния рынка по умолчанию"""
        return MarketCondition(
            primary_phase=MarketPhase.SIDEWAYS,
            secondary_phases=[],
            confidence=0.5,
            volatility=0.02,
            trend_strength=0.5,
            volume_profile="normal",
            timestamp=datetime.now(),
            metadata={'default': True}
        )
    
    def _get_default_indicators(self) -> Dict[str, Any]:
        """Получение индикаторов по умолчанию"""
        return {
            'atr': np.array([0.02]),
            'adx': np.array([20]),
            'di_plus': np.array([20]),
            'di_minus': np.array([20]),
            'bb_width': np.array([0.02]),
            'rsi': np.array([50]),
            'macd': np.array([0]),
            'macd_signal': np.array([0])
        }