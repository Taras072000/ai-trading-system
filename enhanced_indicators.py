"""
Расширенные технические индикаторы для Trading AI
Цель: достижение винрейта 75%+
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
import talib
from scipy import stats
import logging

logger = logging.getLogger(__name__)

class EnhancedTechnicalIndicators:
    """Класс для вычисления расширенного набора технических индикаторов"""
    
    def __init__(self):
        self.indicators_cache = {}
    
    def calculate_all_indicators(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Вычисляет полный набор технических индикаторов
        """
        if len(data) < 20:  # Уменьшаем минимальное требование с 50 до 20
            logger.warning("Недостаточно данных для расчета всех индикаторов")
            # Возвращаем базовые признаки даже при недостатке данных
            indicators = {}
            if len(data) > 0:
                # Добавляем базовые ценовые данные
                current_row = data.iloc[-1]
                indicators.update({
                    'open': current_row['open'],
                    'high': current_row['high'], 
                    'low': current_row['low'],
                    'close': current_row['close'],
                    'volume': current_row.get('volume', 0)
                })
                
                # Добавляем временные признаки и price_change
                if hasattr(data.index, 'to_pydatetime'):
                    last_datetime = data.index[-1].to_pydatetime()
                    indicators['hour'] = float(last_datetime.hour)
                    indicators['day_of_week'] = float(last_datetime.weekday())
                    indicators['day_of_month'] = float(last_datetime.day)
                else:
                    indicators['hour'] = 12.0
                    indicators['day_of_week'] = 2.0
                    indicators['day_of_month'] = 15.0
                
                # Изменение цены
                if len(data) > 1:
                    indicators['price_change'] = (data['close'].iloc[-1] / data['close'].iloc[-2] - 1) * 100
                else:
                    indicators['price_change'] = 0.0
            
            return indicators
        
        indicators = {}
        
        # Основные цены
        high = data['high'].values
        low = data['low'].values
        close = data['close'].values
        volume = data['volume'].values if 'volume' in data.columns else None
        
        try:
            # 1. Трендовые индикаторы
            indicators.update(self._trend_indicators(high, low, close))
            
            # 2. Осцилляторы
            indicators.update(self._oscillators(high, low, close))
            
            # 3. Волатильность
            indicators.update(self._volatility_indicators(high, low, close))
            
            # 4. Объемные индикаторы (если есть данные по объему)
            if volume is not None:
                indicators.update(self._volume_indicators(high, low, close, volume))
            
            # 5. Паттерны свечей
            indicators.update(self._candlestick_patterns(data))
            
            # 6. Статистические индикаторы
            indicators.update(self._statistical_indicators(close))
            
            # 7. Кастомные индикаторы
            indicators.update(self._custom_indicators(data))
            
        except Exception as e:
            logger.error(f"Ошибка при расчете индикаторов: {e}")
            
        return indicators
    
    def _trend_indicators(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> Dict[str, float]:
        """Трендовые индикаторы"""
        indicators = {}
        
        try:
            # Moving Averages
            indicators['sma_5'] = talib.SMA(close, timeperiod=5)[-1]
            indicators['sma_10'] = talib.SMA(close, timeperiod=10)[-1]
            indicators['sma_20'] = talib.SMA(close, timeperiod=20)[-1]
            indicators['sma_50'] = talib.SMA(close, timeperiod=50)[-1] if len(close) >= 50 else np.nan
            
            indicators['ema_5'] = talib.EMA(close, timeperiod=5)[-1]
            indicators['ema_10'] = talib.EMA(close, timeperiod=10)[-1]
            indicators['ema_20'] = talib.EMA(close, timeperiod=20)[-1]
            indicators['ema_50'] = talib.EMA(close, timeperiod=50)[-1] if len(close) >= 50 else np.nan
            
            # MACD
            macd, macd_signal, macd_hist = talib.MACD(close)
            indicators['macd'] = macd[-1]
            indicators['macd_signal'] = macd_signal[-1]
            indicators['macd_histogram'] = macd_hist[-1]
            
            # Parabolic SAR
            indicators['sar'] = talib.SAR(high, low)[-1]
            
            # ADX (Average Directional Index)
            indicators['adx'] = talib.ADX(high, low, close)[-1]
            indicators['plus_di'] = talib.PLUS_DI(high, low, close)[-1]
            indicators['minus_di'] = talib.MINUS_DI(high, low, close)[-1]
            
            # Aroon
            aroon_down, aroon_up = talib.AROON(high, low)
            indicators['aroon_up'] = aroon_up[-1]
            indicators['aroon_down'] = aroon_down[-1]
            indicators['aroon_osc'] = talib.AROONOSC(high, low)[-1]
            
        except Exception as e:
            logger.error(f"Ошибка в трендовых индикаторах: {e}")
            
        return indicators
    
    def _oscillators(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> Dict[str, float]:
        """Осцилляторы"""
        indicators = {}
        
        try:
            # RSI
            indicators['rsi_14'] = talib.RSI(close, timeperiod=14)[-1]
            indicators['rsi_21'] = talib.RSI(close, timeperiod=21)[-1]
            
            # Stochastic
            slowk, slowd = talib.STOCH(high, low, close)
            indicators['stoch_k'] = slowk[-1]
            indicators['stoch_d'] = slowd[-1]
            
            # Stochastic RSI
            fastk, fastd = talib.STOCHRSI(close)
            indicators['stochrsi_k'] = fastk[-1]
            indicators['stochrsi_d'] = fastd[-1]
            
            # Williams %R
            indicators['williams_r'] = talib.WILLR(high, low, close)[-1]
            
            # CCI (Commodity Channel Index)
            indicators['cci'] = talib.CCI(high, low, close)[-1]
            
            # ROC (Rate of Change)
            indicators['roc'] = talib.ROC(close)[-1]
            
            # MFI (Money Flow Index) - требует volume
            # indicators['mfi'] = talib.MFI(high, low, close, volume)[-1]
            
        except Exception as e:
            logger.error(f"Ошибка в осцилляторах: {e}")
            
        return indicators
    
    def _volatility_indicators(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> Dict[str, float]:
        """Индикаторы волатильности"""
        indicators = {}
        
        try:
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = talib.BBANDS(close)
            indicators['bb_upper'] = bb_upper[-1]
            indicators['bb_middle'] = bb_middle[-1]
            indicators['bb_lower'] = bb_lower[-1]
            indicators['bb_width'] = (bb_upper[-1] - bb_lower[-1]) / bb_middle[-1] * 100
            indicators['bb_position'] = (close[-1] - bb_lower[-1]) / (bb_upper[-1] - bb_lower[-1])
            
            # Average True Range
            indicators['atr'] = talib.ATR(high, low, close)[-1]
            
            # True Range
            indicators['trange'] = talib.TRANGE(high, low, close)[-1]
            
            # Normalized ATR
            indicators['natr'] = talib.NATR(high, low, close)[-1]
            
        except Exception as e:
            logger.error(f"Ошибка в индикаторах волатильности: {e}")
            
        return indicators
    
    def _volume_indicators(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray) -> Dict[str, float]:
        """Объемные индикаторы"""
        indicators = {}
        
        try:
            # On Balance Volume
            indicators['obv'] = talib.OBV(close, volume)[-1]
            
            # Accumulation/Distribution Line
            indicators['ad'] = talib.AD(high, low, close, volume)[-1]
            
            # Chaikin A/D Oscillator
            indicators['adosc'] = talib.ADOSC(high, low, close, volume)[-1]
            
            # Money Flow Index
            indicators['mfi'] = talib.MFI(high, low, close, volume)[-1]
            
            # Volume SMA
            indicators['volume_sma'] = talib.SMA(volume, timeperiod=20)[-1]
            indicators['volume_ratio'] = volume[-1] / indicators['volume_sma']
            
        except Exception as e:
            logger.error(f"Ошибка в объемных индикаторах: {e}")
            
        return indicators
    
    def _candlestick_patterns(self, data: pd.DataFrame) -> Dict[str, float]:
        """Паттерны японских свечей"""
        indicators = {}
        
        try:
            high = data['high'].values
            low = data['low'].values
            open_price = data['open'].values if 'open' in data.columns else data['close'].values
            close = data['close'].values
            
            # Основные паттерны
            indicators['doji'] = talib.CDLDOJI(open_price, high, low, close)[-1]
            indicators['hammer'] = talib.CDLHAMMER(open_price, high, low, close)[-1]
            indicators['hanging_man'] = talib.CDLHANGINGMAN(open_price, high, low, close)[-1]
            indicators['shooting_star'] = talib.CDLSHOOTINGSTAR(open_price, high, low, close)[-1]
            indicators['engulfing'] = talib.CDLENGULFING(open_price, high, low, close)[-1]
            indicators['harami'] = talib.CDLHARAMI(open_price, high, low, close)[-1]
            
        except Exception as e:
            logger.error(f"Ошибка в паттернах свечей: {e}")
            
        return indicators
    
    def _statistical_indicators(self, close: np.ndarray) -> Dict[str, float]:
        """Статистические индикаторы"""
        indicators = {}
        
        try:
            # Линейная регрессия
            x = np.arange(len(close))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x[-20:], close[-20:])
            indicators['linear_reg_slope'] = slope
            indicators['linear_reg_r2'] = r_value ** 2
            
            # Z-Score
            mean_20 = np.mean(close[-20:])
            std_20 = np.std(close[-20:])
            indicators['zscore'] = (close[-1] - mean_20) / std_20 if std_20 > 0 else 0
            
            # Коэффициент вариации
            indicators['cv'] = std_20 / mean_20 if mean_20 > 0 else 0
            
        except Exception as e:
            logger.error(f"Ошибка в статистических индикаторах: {e}")
            
        return indicators
    
    def _custom_indicators(self, data: pd.DataFrame) -> Dict[str, float]:
        """Кастомные индикаторы"""
        indicators = {}
        
        try:
            close = data['close'].values
            
            # Временные признаки (если есть индекс с датой)
            if hasattr(data.index, 'to_pydatetime'):
                last_datetime = data.index[-1].to_pydatetime()
                indicators['hour'] = float(last_datetime.hour)
                indicators['day_of_week'] = float(last_datetime.weekday())
                indicators['day_of_month'] = float(last_datetime.day)
            else:
                # Если нет временного индекса, используем значения по умолчанию
                indicators['hour'] = 12.0  # Полдень как среднее значение
                indicators['day_of_week'] = 2.0  # Среда как среднее значение недели
                indicators['day_of_month'] = 15.0  # Середина месяца
            
            # Изменение цены
            if len(close) > 1:
                indicators['price_change'] = (close[-1] / close[-2] - 1) * 100
            else:
                indicators['price_change'] = 0.0
            
            # Momentum индикаторы
            indicators['momentum_5'] = (close[-1] / close[-6] - 1) * 100 if len(close) > 5 else 0
            indicators['momentum_10'] = (close[-1] / close[-11] - 1) * 100 if len(close) > 10 else 0
            
            # Процент от максимума/минимума
            high_20 = np.max(data['high'].tail(20))
            low_20 = np.min(data['low'].tail(20))
            indicators['pct_from_high'] = (close[-1] / high_20 - 1) * 100
            indicators['pct_from_low'] = (close[-1] / low_20 - 1) * 100
            
            # Индекс силы тренда
            sma_5 = np.mean(close[-5:])
            sma_20 = np.mean(close[-20:])
            indicators['trend_strength'] = (sma_5 / sma_20 - 1) * 100
            
            # Волатильность (стандартное отклонение)
            indicators['volatility_10'] = np.std(close[-10:]) / np.mean(close[-10:]) * 100
            indicators['volatility_20'] = np.std(close[-20:]) / np.mean(close[-20:]) * 100
            
        except Exception as e:
            logger.error(f"Ошибка в кастомных индикаторах: {e}")
            
        return indicators
    
    def get_signal_strength(self, indicators: Dict[str, float]) -> Tuple[str, float, str]:
        """
        Анализирует все индикаторы и возвращает сигнал с уверенностью
        """
        if not indicators:
            return 'HOLD', 0.0, 'Нет данных для анализа'
        
        buy_signals = 0
        sell_signals = 0
        total_signals = 0
        reasons = []
        
        try:
            # Трендовые сигналы
            if 'sma_5' in indicators and 'sma_20' in indicators:
                if indicators['sma_5'] > indicators['sma_20']:
                    buy_signals += 2
                    reasons.append("SMA5>SMA20")
                else:
                    sell_signals += 2
                    reasons.append("SMA5<SMA20")
                total_signals += 2
            
            # RSI сигналы
            if 'rsi_14' in indicators:
                rsi = indicators['rsi_14']
                if rsi < 30:
                    buy_signals += 3
                    reasons.append(f"RSI oversold ({rsi:.1f})")
                elif rsi > 70:
                    sell_signals += 3
                    reasons.append(f"RSI overbought ({rsi:.1f})")
                total_signals += 3
            
            # MACD сигналы
            if all(k in indicators for k in ['macd', 'macd_signal']):
                if indicators['macd'] > indicators['macd_signal']:
                    buy_signals += 2
                    reasons.append("MACD bullish")
                else:
                    sell_signals += 2
                    reasons.append("MACD bearish")
                total_signals += 2
            
            # Bollinger Bands
            if 'bb_position' in indicators:
                bb_pos = indicators['bb_position']
                if bb_pos < 0.2:
                    buy_signals += 2
                    reasons.append("BB oversold")
                elif bb_pos > 0.8:
                    sell_signals += 2
                    reasons.append("BB overbought")
                total_signals += 2
            
            # Stochastic
            if 'stoch_k' in indicators and 'stoch_d' in indicators:
                if indicators['stoch_k'] < 20 and indicators['stoch_k'] > indicators['stoch_d']:
                    buy_signals += 1
                    reasons.append("Stoch bullish")
                elif indicators['stoch_k'] > 80 and indicators['stoch_k'] < indicators['stoch_d']:
                    sell_signals += 1
                    reasons.append("Stoch bearish")
                total_signals += 1
            
            # Определение итогового сигнала
            if total_signals == 0:
                return 'HOLD', 0.0, 'Недостаточно сигналов'
            
            buy_ratio = buy_signals / total_signals
            sell_ratio = sell_signals / total_signals
            
            if buy_ratio > 0.6:
                action = 'BUY'
                confidence = min(0.95, buy_ratio)
            elif sell_ratio > 0.6:
                action = 'SELL'
                confidence = min(0.95, sell_ratio)
            else:
                action = 'HOLD'
                confidence = 0.5
            
            reason = f"Signals: {buy_signals}B/{sell_signals}S. " + ", ".join(reasons[:3])
            
            return action, confidence, reason
            
        except Exception as e:
            logger.error(f"Ошибка в анализе сигналов: {e}")
            return 'HOLD', 0.0, f'Ошибка анализа: {str(e)}'