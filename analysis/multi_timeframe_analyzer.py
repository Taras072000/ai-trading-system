"""
Система мульти-таймфреймного анализа для торговой системы
Интеграция сигналов с разных временных интервалов с иерархической системой решений
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import logging
from datetime import datetime, timedelta
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
from dataclasses import dataclass
from enum import Enum

# Технические индикаторы
import talib

class TimeFrame(Enum):
    """Временные интервалы для анализа"""
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"

class SignalStrength(Enum):
    """Сила сигнала"""
    VERY_WEAK = 1
    WEAK = 2
    MODERATE = 3
    STRONG = 4
    VERY_STRONG = 5

@dataclass
class TimeFrameSignal:
    """Сигнал с определенного таймфрейма"""
    timeframe: TimeFrame
    signal_type: str  # 'BUY', 'SELL', 'HOLD'
    strength: SignalStrength
    confidence: float
    indicators: Dict[str, float]
    timestamp: datetime
    price: float
    volume: float

@dataclass
class MultiTimeFrameDecision:
    """Итоговое решение на основе мульти-таймфреймного анализа"""
    final_signal: str
    confidence: float
    timeframe_votes: Dict[TimeFrame, TimeFrameSignal]
    hierarchical_weight: float
    risk_assessment: Dict[str, float]
    execution_priority: int
    reasoning: List[str]

class MultiTimeFrameAnalyzer:
    """Главный класс мульти-таймфреймного анализа"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Веса для разных таймфреймов (иерархическая важность)
        self.timeframe_weights = {
            TimeFrame.D1: 0.30,   # Долгосрочный тренд - наивысший приоритет
            TimeFrame.H4: 0.25,   # Среднесрочный тренд
            TimeFrame.H1: 0.20,   # Краткосрочный тренд
            TimeFrame.M15: 0.15,  # Тактические сигналы
            TimeFrame.M5: 0.07,   # Точки входа
            TimeFrame.M1: 0.03    # Микро-тайминг
        }
        
        # Минимальные требования по таймфреймам для разных типов сигналов
        self.signal_requirements = {
            'STRONG_BUY': {
                'min_timeframes': 4,
                'required_major': [TimeFrame.D1, TimeFrame.H4],
                'min_confidence': 0.7
            },
            'BUY': {
                'min_timeframes': 3,
                'required_major': [TimeFrame.H4],
                'min_confidence': 0.6
            },
            'WEAK_BUY': {
                'min_timeframes': 2,
                'required_major': [],
                'min_confidence': 0.5
            },
            'HOLD': {
                'min_timeframes': 1,
                'required_major': [],
                'min_confidence': 0.3
            }
        }
        
        # Конфликт-резолюция между таймфреймами
        self.conflict_resolution = {
            'trend_alignment': 0.4,    # Совпадение трендов
            'momentum_sync': 0.3,      # Синхронизация моментума
            'volume_confirmation': 0.2, # Подтверждение объемом
            'volatility_factor': 0.1   # Фактор волатильности
        }
        
        self.executor = ThreadPoolExecutor(max_workers=6)
        self.signal_cache = {}
        
    async def analyze_multi_timeframe(self, 
                                    symbol: str, 
                                    current_price: float,
                                    market_data: Dict[TimeFrame, pd.DataFrame]) -> MultiTimeFrameDecision:
        """
        Основной метод мульти-таймфреймного анализа
        
        Args:
            symbol: Торговый символ
            current_price: Текущая цена
            market_data: Данные по всем таймфреймам
            
        Returns:
            Итоговое решение с учетом всех таймфреймов
        """
        self.logger.info(f"Начинаем мульти-таймфреймный анализ для {symbol}")
        
        try:
            # Параллельный анализ всех таймфреймов
            timeframe_signals = await self._analyze_all_timeframes(symbol, market_data)
            
            # Иерархическое взвешивание сигналов
            weighted_decision = self._calculate_hierarchical_decision(timeframe_signals)
            
            # Разрешение конфликтов между таймфреймами
            resolved_decision = self._resolve_timeframe_conflicts(weighted_decision, timeframe_signals)
            
            # Оценка рисков
            risk_assessment = self._assess_multi_timeframe_risk(timeframe_signals, current_price)
            
            # Формирование финального решения
            final_decision = self._create_final_decision(
                resolved_decision, timeframe_signals, risk_assessment, current_price
            )
            
            self.logger.info(f"Мульти-таймфреймный анализ завершен: {final_decision.final_signal} "
                           f"(confidence: {final_decision.confidence:.3f})")
            
            return final_decision
            
        except Exception as e:
            self.logger.error(f"Ошибка в мульти-таймфреймном анализе: {e}")
            raise
    
    async def _analyze_all_timeframes(self, 
                                    symbol: str, 
                                    market_data: Dict[TimeFrame, pd.DataFrame]) -> Dict[TimeFrame, TimeFrameSignal]:
        """Параллельный анализ всех таймфреймов"""
        
        tasks = []
        for timeframe, data in market_data.items():
            task = asyncio.get_event_loop().run_in_executor(
                self.executor, self._analyze_single_timeframe, timeframe, data, symbol
            )
            tasks.append((timeframe, task))
        
        timeframe_signals = {}
        for timeframe, task in tasks:
            try:
                signal = await task
                timeframe_signals[timeframe] = signal
            except Exception as e:
                self.logger.error(f"Ошибка анализа таймфрейма {timeframe}: {e}")
                # Создаем нейтральный сигнал при ошибке
                timeframe_signals[timeframe] = self._create_neutral_signal(timeframe)
        
        return timeframe_signals
    
    def _analyze_single_timeframe(self, 
                                timeframe: TimeFrame, 
                                data: pd.DataFrame, 
                                symbol: str) -> TimeFrameSignal:
        """Анализ одного таймфрейма"""
        
        if len(data) < 50:
            return self._create_neutral_signal(timeframe)
        
        try:
            # Расчет технических индикаторов
            indicators = self._calculate_timeframe_indicators(data, timeframe)
            
            # Определение тренда
            trend_signal = self._determine_trend_signal(indicators, timeframe)
            
            # Анализ моментума
            momentum_signal = self._analyze_momentum(indicators, timeframe)
            
            # Анализ объемов
            volume_signal = self._analyze_volume_pattern(data, timeframe)
            
            # Комбинирование сигналов
            combined_signal = self._combine_timeframe_signals(
                trend_signal, momentum_signal, volume_signal, timeframe
            )
            
            # Оценка силы сигнала
            signal_strength = self._evaluate_signal_strength(indicators, timeframe)
            
            # Расчет уверенности
            confidence = self._calculate_timeframe_confidence(indicators, combined_signal, timeframe)
            
            return TimeFrameSignal(
                timeframe=timeframe,
                signal_type=combined_signal,
                strength=signal_strength,
                confidence=confidence,
                indicators=indicators,
                timestamp=datetime.now(),
                price=data['close'].iloc[-1],
                volume=data['volume'].iloc[-1] if 'volume' in data.columns else 0
            )
            
        except Exception as e:
            self.logger.error(f"Ошибка в анализе таймфрейма {timeframe}: {e}")
            return self._create_neutral_signal(timeframe)
    
    def _calculate_timeframe_indicators(self, data: pd.DataFrame, timeframe: TimeFrame) -> Dict[str, float]:
        """Расчет технических индикаторов для таймфрейма"""
        
        close = data['close'].values
        high = data['high'].values
        low = data['low'].values
        volume = data['volume'].values if 'volume' in data.columns else np.ones(len(close))
        
        indicators = {}
        
        try:
            # Трендовые индикаторы
            indicators['sma_20'] = talib.SMA(close, timeperiod=20)[-1]
            indicators['sma_50'] = talib.SMA(close, timeperiod=50)[-1]
            indicators['ema_12'] = talib.EMA(close, timeperiod=12)[-1]
            indicators['ema_26'] = talib.EMA(close, timeperiod=26)[-1]
            
            # MACD
            macd, macd_signal, macd_hist = talib.MACD(close)
            indicators['macd'] = macd[-1]
            indicators['macd_signal'] = macd_signal[-1]
            indicators['macd_histogram'] = macd_hist[-1]
            
            # Осцилляторы
            indicators['rsi'] = talib.RSI(close, timeperiod=14)[-1]
            indicators['stoch_k'], indicators['stoch_d'] = talib.STOCH(high, low, close)
            indicators['stoch_k'] = indicators['stoch_k'][-1]
            indicators['stoch_d'] = indicators['stoch_d'][-1]
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = talib.BBANDS(close)
            indicators['bb_upper'] = bb_upper[-1]
            indicators['bb_middle'] = bb_middle[-1]
            indicators['bb_lower'] = bb_lower[-1]
            indicators['bb_position'] = (close[-1] - bb_lower[-1]) / (bb_upper[-1] - bb_lower[-1])
            
            # ATR для волатильности
            indicators['atr'] = talib.ATR(high, low, close, timeperiod=14)[-1]
            
            # Объемные индикаторы
            indicators['volume_sma'] = talib.SMA(volume, timeperiod=20)[-1]
            indicators['volume_ratio'] = volume[-1] / indicators['volume_sma']
            
            # ADX для силы тренда
            indicators['adx'] = talib.ADX(high, low, close, timeperiod=14)[-1]
            
            # Дополнительные индикаторы в зависимости от таймфрейма
            if timeframe in [TimeFrame.D1, TimeFrame.H4]:
                # Долгосрочные индикаторы
                indicators['sma_200'] = talib.SMA(close, timeperiod=200)[-1] if len(close) >= 200 else indicators['sma_50']
                indicators['rsi_weekly'] = talib.RSI(close, timeperiod=21)[-1]
            
            elif timeframe in [TimeFrame.M1, TimeFrame.M5]:
                # Краткосрочные индикаторы
                indicators['cci'] = talib.CCI(high, low, close, timeperiod=14)[-1]
                indicators['williams_r'] = talib.WILLR(high, low, close, timeperiod=14)[-1]
            
        except Exception as e:
            self.logger.error(f"Ошибка расчета индикаторов для {timeframe}: {e}")
            # Возвращаем базовые значения при ошибке
            indicators = {
                'sma_20': close[-1], 'sma_50': close[-1], 'rsi': 50.0,
                'macd': 0.0, 'atr': close[-1] * 0.02, 'adx': 25.0
            }
        
        return indicators
    
    def _determine_trend_signal(self, indicators: Dict[str, float], timeframe: TimeFrame) -> str:
        """Определение трендового сигнала"""
        
        trend_score = 0
        
        # SMA тренд
        if 'sma_20' in indicators and 'sma_50' in indicators:
            if indicators['sma_20'] > indicators['sma_50']:
                trend_score += 1
            elif indicators['sma_20'] < indicators['sma_50']:
                trend_score -= 1
        
        # EMA тренд
        if 'ema_12' in indicators and 'ema_26' in indicators:
            if indicators['ema_12'] > indicators['ema_26']:
                trend_score += 1
            elif indicators['ema_12'] < indicators['ema_26']:
                trend_score -= 1
        
        # MACD тренд
        if 'macd' in indicators and 'macd_signal' in indicators:
            if indicators['macd'] > indicators['macd_signal']:
                trend_score += 1
            elif indicators['macd'] < indicators['macd_signal']:
                trend_score -= 1
        
        # ADX сила тренда
        if 'adx' in indicators:
            if indicators['adx'] > 25:
                trend_score *= 1.5  # Усиливаем сигнал при сильном тренде
        
        # Определение сигнала
        if trend_score >= 2:
            return 'BUY'
        elif trend_score <= -2:
            return 'SELL'
        else:
            return 'HOLD'
    
    def _analyze_momentum(self, indicators: Dict[str, float], timeframe: TimeFrame) -> str:
        """Анализ моментума"""
        
        momentum_score = 0
        
        # RSI моментум
        if 'rsi' in indicators:
            rsi = indicators['rsi']
            if rsi > 70:
                momentum_score -= 1  # Перекупленность
            elif rsi > 60:
                momentum_score += 1  # Сильный моментум
            elif rsi < 30:
                momentum_score += 1  # Перепроданность
            elif rsi < 40:
                momentum_score -= 1  # Слабый моментум
        
        # Stochastic моментум
        if 'stoch_k' in indicators and 'stoch_d' in indicators:
            if indicators['stoch_k'] > indicators['stoch_d'] and indicators['stoch_k'] < 80:
                momentum_score += 1
            elif indicators['stoch_k'] < indicators['stoch_d'] and indicators['stoch_k'] > 20:
                momentum_score -= 1
        
        # MACD гистограмма
        if 'macd_histogram' in indicators:
            if indicators['macd_histogram'] > 0:
                momentum_score += 1
            else:
                momentum_score -= 1
        
        if momentum_score >= 2:
            return 'BUY'
        elif momentum_score <= -2:
            return 'SELL'
        else:
            return 'HOLD'
    
    def _analyze_volume_pattern(self, data: pd.DataFrame, timeframe: TimeFrame) -> str:
        """Анализ паттернов объема"""
        
        if 'volume' not in data.columns:
            return 'HOLD'
        
        volume = data['volume'].values
        close = data['close'].values
        
        if len(volume) < 20:
            return 'HOLD'
        
        # Средний объем
        avg_volume = np.mean(volume[-20:])
        current_volume = volume[-1]
        
        # Изменение цены
        price_change = (close[-1] - close[-2]) / close[-2] if len(close) > 1 else 0
        
        # Анализ объема
        volume_ratio = current_volume / avg_volume
        
        if volume_ratio > 1.5 and price_change > 0:
            return 'BUY'  # Высокий объем + рост цены
        elif volume_ratio > 1.5 and price_change < 0:
            return 'SELL'  # Высокий объем + падение цены
        else:
            return 'HOLD'
    
    def _combine_timeframe_signals(self, 
                                 trend_signal: str, 
                                 momentum_signal: str, 
                                 volume_signal: str, 
                                 timeframe: TimeFrame) -> str:
        """Комбинирование сигналов внутри таймфрейма"""
        
        signals = [trend_signal, momentum_signal, volume_signal]
        
        # Подсчет голосов
        buy_votes = signals.count('BUY')
        sell_votes = signals.count('SELL')
        hold_votes = signals.count('HOLD')
        
        # Веса для разных типов сигналов
        trend_weight = 0.5
        momentum_weight = 0.3
        volume_weight = 0.2
        
        # Взвешенный счет
        weighted_score = 0
        if trend_signal == 'BUY':
            weighted_score += trend_weight
        elif trend_signal == 'SELL':
            weighted_score -= trend_weight
            
        if momentum_signal == 'BUY':
            weighted_score += momentum_weight
        elif momentum_signal == 'SELL':
            weighted_score -= momentum_weight
            
        if volume_signal == 'BUY':
            weighted_score += volume_weight
        elif volume_signal == 'SELL':
            weighted_score -= volume_weight
        
        # Определение итогового сигнала
        if weighted_score > 0.4:
            return 'BUY'
        elif weighted_score < -0.4:
            return 'SELL'
        else:
            return 'HOLD'
    
    def _evaluate_signal_strength(self, indicators: Dict[str, float], timeframe: TimeFrame) -> SignalStrength:
        """Оценка силы сигнала"""
        
        strength_score = 0
        
        # ADX - сила тренда
        if 'adx' in indicators:
            adx = indicators['adx']
            if adx > 40:
                strength_score += 2
            elif adx > 25:
                strength_score += 1
        
        # RSI - экстремальные значения
        if 'rsi' in indicators:
            rsi = indicators['rsi']
            if rsi > 80 or rsi < 20:
                strength_score += 2
            elif rsi > 70 or rsi < 30:
                strength_score += 1
        
        # Bollinger Bands - позиция
        if 'bb_position' in indicators:
            bb_pos = indicators['bb_position']
            if bb_pos > 0.9 or bb_pos < 0.1:
                strength_score += 1
        
        # Объем
        if 'volume_ratio' in indicators:
            vol_ratio = indicators['volume_ratio']
            if vol_ratio > 2.0:
                strength_score += 2
            elif vol_ratio > 1.5:
                strength_score += 1
        
        # Преобразование в enum
        if strength_score >= 5:
            return SignalStrength.VERY_STRONG
        elif strength_score >= 4:
            return SignalStrength.STRONG
        elif strength_score >= 2:
            return SignalStrength.MODERATE
        elif strength_score >= 1:
            return SignalStrength.WEAK
        else:
            return SignalStrength.VERY_WEAK
    
    def _calculate_timeframe_confidence(self, 
                                      indicators: Dict[str, float], 
                                      signal: str, 
                                      timeframe: TimeFrame) -> float:
        """Расчет уверенности в сигнале таймфрейма"""
        
        confidence = 0.5  # Базовая уверенность
        
        # Согласованность индикаторов
        trend_indicators = ['sma_20', 'sma_50', 'ema_12', 'ema_26']
        trend_agreement = 0
        
        for i in range(len(trend_indicators) - 1):
            if trend_indicators[i] in indicators and trend_indicators[i+1] in indicators:
                if signal == 'BUY' and indicators[trend_indicators[i]] > indicators[trend_indicators[i+1]]:
                    trend_agreement += 1
                elif signal == 'SELL' and indicators[trend_indicators[i]] < indicators[trend_indicators[i+1]]:
                    trend_agreement += 1
        
        confidence += trend_agreement * 0.1
        
        # Сила тренда (ADX)
        if 'adx' in indicators:
            adx = indicators['adx']
            confidence += min(0.2, adx / 100)
        
        # Объемное подтверждение
        if 'volume_ratio' in indicators:
            vol_ratio = indicators['volume_ratio']
            if vol_ratio > 1.2:
                confidence += 0.1
        
        # Экстремальные значения RSI
        if 'rsi' in indicators:
            rsi = indicators['rsi']
            if signal == 'BUY' and rsi < 40:
                confidence += 0.15
            elif signal == 'SELL' and rsi > 60:
                confidence += 0.15
        
        # Ограничиваем диапазон
        return max(0.0, min(1.0, confidence))
    
    def _calculate_hierarchical_decision(self, 
                                       timeframe_signals: Dict[TimeFrame, TimeFrameSignal]) -> Dict[str, float]:
        """Иерархическое взвешивание решений по таймфреймам"""
        
        weighted_votes = {'BUY': 0.0, 'SELL': 0.0, 'HOLD': 0.0}
        total_weight = 0.0
        
        for timeframe, signal in timeframe_signals.items():
            weight = self.timeframe_weights.get(timeframe, 0.1)
            confidence_weight = weight * signal.confidence
            
            weighted_votes[signal.signal_type] += confidence_weight
            total_weight += confidence_weight
        
        # Нормализация
        if total_weight > 0:
            for signal_type in weighted_votes:
                weighted_votes[signal_type] /= total_weight
        
        return weighted_votes
    
    def _resolve_timeframe_conflicts(self, 
                                   weighted_decision: Dict[str, float],
                                   timeframe_signals: Dict[TimeFrame, TimeFrameSignal]) -> Dict[str, Any]:
        """Разрешение конфликтов между таймфреймами"""
        
        # Анализ согласованности трендов
        trend_alignment = self._analyze_trend_alignment(timeframe_signals)
        
        # Синхронизация моментума
        momentum_sync = self._analyze_momentum_synchronization(timeframe_signals)
        
        # Подтверждение объемом
        volume_confirmation = self._analyze_volume_confirmation(timeframe_signals)
        
        # Фактор волатильности
        volatility_factor = self._analyze_volatility_factor(timeframe_signals)
        
        # Корректировка решения на основе конфликт-резолюции
        conflict_adjustment = (
            trend_alignment * self.conflict_resolution['trend_alignment'] +
            momentum_sync * self.conflict_resolution['momentum_sync'] +
            volume_confirmation * self.conflict_resolution['volume_confirmation'] +
            volatility_factor * self.conflict_resolution['volatility_factor']
        )
        
        # Применяем корректировку
        adjusted_decision = weighted_decision.copy()
        
        if conflict_adjustment > 0.1:
            # Усиливаем сигнал покупки
            adjusted_decision['BUY'] *= (1 + conflict_adjustment)
        elif conflict_adjustment < -0.1:
            # Усиливаем сигнал продажи
            adjusted_decision['SELL'] *= (1 + abs(conflict_adjustment))
        
        return {
            'decision': adjusted_decision,
            'conflict_metrics': {
                'trend_alignment': trend_alignment,
                'momentum_sync': momentum_sync,
                'volume_confirmation': volume_confirmation,
                'volatility_factor': volatility_factor,
                'total_adjustment': conflict_adjustment
            }
        }
    
    def _analyze_trend_alignment(self, timeframe_signals: Dict[TimeFrame, TimeFrameSignal]) -> float:
        """Анализ согласованности трендов между таймфреймами"""
        
        major_timeframes = [TimeFrame.D1, TimeFrame.H4, TimeFrame.H1]
        signals = []
        
        for tf in major_timeframes:
            if tf in timeframe_signals:
                signal = timeframe_signals[tf].signal_type
                if signal == 'BUY':
                    signals.append(1)
                elif signal == 'SELL':
                    signals.append(-1)
                else:
                    signals.append(0)
        
        if not signals:
            return 0.0
        
        # Согласованность как стандартное отклонение
        alignment = 1.0 - (np.std(signals) / 1.0) if len(signals) > 1 else 1.0
        return alignment * np.mean(signals) if signals else 0.0
    
    def _analyze_momentum_synchronization(self, timeframe_signals: Dict[TimeFrame, TimeFrameSignal]) -> float:
        """Анализ синхронизации моментума"""
        
        rsi_values = []
        for signal in timeframe_signals.values():
            if 'rsi' in signal.indicators:
                rsi_values.append(signal.indicators['rsi'])
        
        if len(rsi_values) < 2:
            return 0.0
        
        # Синхронизация как обратная величина разброса RSI
        rsi_std = np.std(rsi_values)
        sync_score = max(0.0, 1.0 - rsi_std / 50.0)  # Нормализация
        
        # Направление моментума
        avg_rsi = np.mean(rsi_values)
        if avg_rsi > 60:
            return sync_score
        elif avg_rsi < 40:
            return -sync_score
        else:
            return 0.0
    
    def _analyze_volume_confirmation(self, timeframe_signals: Dict[TimeFrame, TimeFrameSignal]) -> float:
        """Анализ подтверждения объемом"""
        
        volume_ratios = []
        signal_directions = []
        
        for signal in timeframe_signals.values():
            if 'volume_ratio' in signal.indicators:
                volume_ratios.append(signal.indicators['volume_ratio'])
                if signal.signal_type == 'BUY':
                    signal_directions.append(1)
                elif signal.signal_type == 'SELL':
                    signal_directions.append(-1)
                else:
                    signal_directions.append(0)
        
        if not volume_ratios:
            return 0.0
        
        # Средний объем и направление
        avg_volume_ratio = np.mean(volume_ratios)
        avg_direction = np.mean(signal_directions)
        
        # Подтверждение = высокий объем * согласованное направление
        confirmation = min(1.0, avg_volume_ratio / 2.0) * abs(avg_direction)
        
        return confirmation * avg_direction
    
    def _analyze_volatility_factor(self, timeframe_signals: Dict[TimeFrame, TimeFrameSignal]) -> float:
        """Анализ фактора волатильности"""
        
        atr_values = []
        for signal in timeframe_signals.values():
            if 'atr' in signal.indicators:
                atr_values.append(signal.indicators['atr'])
        
        if not atr_values:
            return 0.0
        
        # Нормализованная волатильность
        avg_atr = np.mean(atr_values)
        
        # Высокая волатильность снижает уверенность
        volatility_penalty = min(0.3, avg_atr / 100.0)
        
        return -volatility_penalty
    
    def _assess_multi_timeframe_risk(self, 
                                   timeframe_signals: Dict[TimeFrame, TimeFrameSignal],
                                   current_price: float) -> Dict[str, float]:
        """Оценка рисков на основе мульти-таймфреймного анализа"""
        
        risk_metrics = {}
        
        # Риск расхождения таймфреймов
        signal_types = [signal.signal_type for signal in timeframe_signals.values()]
        unique_signals = len(set(signal_types))
        risk_metrics['divergence_risk'] = min(1.0, unique_signals / len(signal_types))
        
        # Волатильность риск
        atr_values = [signal.indicators.get('atr', 0) for signal in timeframe_signals.values()]
        avg_atr = np.mean([atr for atr in atr_values if atr > 0])
        risk_metrics['volatility_risk'] = min(1.0, avg_atr / current_price) if avg_atr > 0 else 0.5
        
        # Риск ложного сигнала
        confidences = [signal.confidence for signal in timeframe_signals.values()]
        avg_confidence = np.mean(confidences)
        risk_metrics['false_signal_risk'] = 1.0 - avg_confidence
        
        # Временной риск (конфликт между краткосрочными и долгосрочными сигналами)
        short_term_signals = [timeframe_signals[tf].signal_type 
                            for tf in [TimeFrame.M1, TimeFrame.M5, TimeFrame.M15] 
                            if tf in timeframe_signals]
        long_term_signals = [timeframe_signals[tf].signal_type 
                           for tf in [TimeFrame.H4, TimeFrame.D1] 
                           if tf in timeframe_signals]
        
        if short_term_signals and long_term_signals:
            short_consensus = max(set(short_term_signals), key=short_term_signals.count)
            long_consensus = max(set(long_term_signals), key=long_term_signals.count)
            risk_metrics['temporal_risk'] = 1.0 if short_consensus != long_consensus else 0.0
        else:
            risk_metrics['temporal_risk'] = 0.5
        
        return risk_metrics
    
    def _create_final_decision(self, 
                             resolved_decision: Dict[str, Any],
                             timeframe_signals: Dict[TimeFrame, TimeFrameSignal],
                             risk_assessment: Dict[str, float],
                             current_price: float) -> MultiTimeFrameDecision:
        """Создание финального решения"""
        
        decision_scores = resolved_decision['decision']
        
        # Определение финального сигнала
        final_signal = max(decision_scores, key=decision_scores.get)
        
        # Расчет итоговой уверенности с учетом рисков
        base_confidence = decision_scores[final_signal]
        risk_penalty = np.mean(list(risk_assessment.values()))
        final_confidence = base_confidence * (1 - risk_penalty * 0.5)
        
        # Иерархический вес (важность долгосрочных таймфреймов)
        major_timeframes_weight = sum(
            self.timeframe_weights[tf] * timeframe_signals[tf].confidence
            for tf in [TimeFrame.D1, TimeFrame.H4] 
            if tf in timeframe_signals
        )
        
        # Приоритет исполнения
        execution_priority = self._calculate_execution_priority(
            final_signal, final_confidence, risk_assessment
        )
        
        # Обоснование решения
        reasoning = self._generate_decision_reasoning(
            timeframe_signals, resolved_decision, risk_assessment
        )
        
        return MultiTimeFrameDecision(
            final_signal=final_signal,
            confidence=final_confidence,
            timeframe_votes=timeframe_signals,
            hierarchical_weight=major_timeframes_weight,
            risk_assessment=risk_assessment,
            execution_priority=execution_priority,
            reasoning=reasoning
        )
    
    def _calculate_execution_priority(self, 
                                    signal: str, 
                                    confidence: float, 
                                    risk_assessment: Dict[str, float]) -> int:
        """Расчет приоритета исполнения (1-10, где 10 - наивысший)"""
        
        if signal == 'HOLD':
            return 1
        
        # Базовый приоритет на основе уверенности
        base_priority = int(confidence * 10)
        
        # Корректировка на основе рисков
        avg_risk = np.mean(list(risk_assessment.values()))
        risk_adjustment = int((1 - avg_risk) * 3)
        
        priority = base_priority + risk_adjustment
        return max(1, min(10, priority))
    
    def _generate_decision_reasoning(self, 
                                   timeframe_signals: Dict[TimeFrame, TimeFrameSignal],
                                   resolved_decision: Dict[str, Any],
                                   risk_assessment: Dict[str, float]) -> List[str]:
        """Генерация обоснования решения"""
        
        reasoning = []
        
        # Анализ согласованности таймфреймов
        signal_counts = {}
        for signal in timeframe_signals.values():
            signal_counts[signal.signal_type] = signal_counts.get(signal.signal_type, 0) + 1
        
        dominant_signal = max(signal_counts, key=signal_counts.get)
        reasoning.append(f"Доминирующий сигнал: {dominant_signal} ({signal_counts[dominant_signal]}/{len(timeframe_signals)} таймфреймов)")
        
        # Анализ долгосрочного тренда
        major_signals = [timeframe_signals[tf].signal_type 
                        for tf in [TimeFrame.D1, TimeFrame.H4] 
                        if tf in timeframe_signals]
        if major_signals:
            reasoning.append(f"Долгосрочный тренд: {', '.join(major_signals)}")
        
        # Анализ рисков
        high_risks = [risk_type for risk_type, value in risk_assessment.items() if value > 0.6]
        if high_risks:
            reasoning.append(f"Высокие риски: {', '.join(high_risks)}")
        
        # Конфликт-резолюция
        conflict_metrics = resolved_decision.get('conflict_metrics', {})
        if conflict_metrics.get('total_adjustment', 0) > 0.1:
            reasoning.append("Положительная корректировка на основе согласованности индикаторов")
        elif conflict_metrics.get('total_adjustment', 0) < -0.1:
            reasoning.append("Отрицательная корректировка из-за конфликтов между таймфреймами")
        
        return reasoning
    
    def _create_neutral_signal(self, timeframe: TimeFrame) -> TimeFrameSignal:
        """Создание нейтрального сигнала при ошибках"""
        return TimeFrameSignal(
            timeframe=timeframe,
            signal_type='HOLD',
            strength=SignalStrength.VERY_WEAK,
            confidence=0.3,
            indicators={'rsi': 50.0, 'atr': 0.02},
            timestamp=datetime.now(),
            price=0.0,
            volume=0.0
        )

# Пример использования
if __name__ == "__main__":
    # Настройка логирования
    logging.basicConfig(level=logging.INFO)
    
    # Создание анализатора
    analyzer = MultiTimeFrameAnalyzer()
    
    # Пример данных (в реальной системе данные приходят из API)
    sample_data = {
        TimeFrame.D1: pd.DataFrame({
            'open': np.random.randn(100) + 100,
            'high': np.random.randn(100) + 102,
            'low': np.random.randn(100) + 98,
            'close': np.random.randn(100) + 100,
            'volume': np.random.randint(1000, 10000, 100)
        }),
        TimeFrame.H4: pd.DataFrame({
            'open': np.random.randn(200) + 100,
            'high': np.random.randn(200) + 102,
            'low': np.random.randn(200) + 98,
            'close': np.random.randn(200) + 100,
            'volume': np.random.randint(1000, 10000, 200)
        })
    }
    
    # Запуск анализа
    async def test_analysis():
        decision = await analyzer.analyze_multi_timeframe("BTCUSDT", 50000.0, sample_data)
        print(f"Финальное решение: {decision.final_signal}")
        print(f"Уверенность: {decision.confidence:.3f}")
        print(f"Обоснование: {decision.reasoning}")
    
    # asyncio.run(test_analysis())