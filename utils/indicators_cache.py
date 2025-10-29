"""
Централизованный кэш технических индикаторов
Устраняет дублирование расчетов SMA, волатильности и других индикаторов
"""

import pandas as pd
import numpy as np
import hashlib
import time
from typing import Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class IndicatorsCache:
    """Централизованный кэш для технических индикаторов"""
    
    def __init__(self, max_cache_size: int = 1000, ttl_seconds: int = 300):
        """
        Инициализация кэша индикаторов
        
        Args:
            max_cache_size: Максимальный размер кэша
            ttl_seconds: Время жизни кэша в секундах (по умолчанию 5 минут)
        """
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.access_times: Dict[str, float] = {}
        self.max_cache_size = max_cache_size
        self.ttl_seconds = ttl_seconds
        self.last_cleanup = time.time()
        
        logger.info(f"Инициализирован кэш индикаторов: размер={max_cache_size}, TTL={ttl_seconds}с")
    
    def _generate_cache_key(self, symbol: str, data_hash: str, indicator_type: str, **params) -> str:
        """Генерация ключа кэша"""
        params_str = "_".join([f"{k}={v}" for k, v in sorted(params.items())])
        return f"{symbol}_{indicator_type}_{data_hash}_{params_str}"
    
    def _get_data_hash(self, data) -> str:
        """Получение хэша данных для кэширования"""
        if hasattr(data, 'empty') and data.empty:
            return "empty"
        
        # Работаем как с DataFrame, так и с Series
        if isinstance(data, pd.Series):
            if len(data) == 0:
                return "empty"
            last_values = data.tail(5)
            data_str = f"{len(data)}_{last_values.iloc[-1]}_{last_values.iloc[0]}"
        else:
            # DataFrame
            last_rows = data.tail(5)
            data_str = f"{len(data)}_{last_rows['close'].iloc[-1]}_{last_rows['close'].iloc[0]}"
        
        return hashlib.md5(data_str.encode()).hexdigest()[:8]
    
    def _cleanup_expired(self):
        """Очистка устаревших записей"""
        current_time = time.time()
        
        # Очистка каждые 60 секунд
        if current_time - self.last_cleanup < 60:
            return
        
        expired_keys = []
        for key, access_time in self.access_times.items():
            if current_time - access_time > self.ttl_seconds:
                expired_keys.append(key)
        
        for key in expired_keys:
            self.cache.pop(key, None)
            self.access_times.pop(key, None)
        
        # Если кэш все еще переполнен, удаляем самые старые записи
        if len(self.cache) > self.max_cache_size:
            sorted_keys = sorted(self.access_times.items(), key=lambda x: x[1])
            keys_to_remove = sorted_keys[:len(self.cache) - self.max_cache_size + 100]
            
            for key, _ in keys_to_remove:
                self.cache.pop(key, None)
                self.access_times.pop(key, None)
        
        self.last_cleanup = current_time
        
        if expired_keys:
            logger.debug(f"Очищено {len(expired_keys)} устаревших записей из кэша индикаторов")
    
    def get_sma(self, symbol: str, data, period: int) -> Optional[pd.Series]:
        """Получение SMA с кэшированием"""
        try:
            self._cleanup_expired()
            
            data_hash = self._get_data_hash(data)
            cache_key = self._generate_cache_key(symbol, data_hash, "sma", period=period)
            
            # Проверяем кэш
            if cache_key in self.cache:
                self.access_times[cache_key] = time.time()
                return self.cache[cache_key]['value']
            
            # Вычисляем SMA
            if len(data) < period:
                return None
            
            # Работаем как с DataFrame, так и с Series
            if isinstance(data, pd.Series):
                sma_series = data.rolling(period).mean()
            else:
                sma_series = data['close'].rolling(period).mean()
            
            # Сохраняем в кэш
            self.cache[cache_key] = {
                'value': sma_series,
                'timestamp': time.time()
            }
            self.access_times[cache_key] = time.time()
            
            return sma_series
            
        except Exception as e:
            logger.error(f"Ошибка расчета SMA для {symbol}: {e}")
            return None
    
    def get_volatility(self, symbol: str, data, period: int = 20, 
                      method: str = 'std') -> Optional[float]:
        """Получение волатильности с кэшированием"""
        try:
            self._cleanup_expired()
            
            data_hash = self._get_data_hash(data)
            cache_key = self._generate_cache_key(symbol, data_hash, "volatility", 
                                               period=period, method=method)
            
            # Проверяем кэш
            if cache_key in self.cache:
                self.access_times[cache_key] = time.time()
                return self.cache[cache_key]['value']
            
            # Вычисляем волатильность
            if len(data) < period:
                return None
            
            # Работаем как с DataFrame, так и с Series
            if isinstance(data, pd.Series):
                returns = data.pct_change().dropna()
            else:
                returns = data['close'].pct_change().dropna()
            
            if method == 'std':
                volatility = returns.rolling(period).std().iloc[-1]
            elif method == 'daily_std':
                volatility = returns.std() * np.sqrt(24 * 60)  # Дневная волатильность для 1m данных
            else:
                volatility = returns.rolling(period).std().iloc[-1]
            
            # Сохраняем в кэш
            self.cache[cache_key] = {
                'value': float(volatility),
                'timestamp': time.time()
            }
            self.access_times[cache_key] = time.time()
            
            return float(volatility)
            
        except Exception as e:
            logger.error(f"Ошибка расчета волатильности для {symbol}: {e}")
            return None
    
    def get_rsi(self, symbol: str, data, period: int = 14) -> Optional[pd.Series]:
        """Получение RSI с кэшированием"""
        try:
            self._cleanup_expired()
            
            data_hash = self._get_data_hash(data)
            cache_key = self._generate_cache_key(symbol, data_hash, "rsi", period=period)
            
            # Проверяем кэш
            if cache_key in self.cache:
                self.access_times[cache_key] = time.time()
                return self.cache[cache_key]['value']
            
            # Вычисляем RSI
            if len(data) < period + 1:
                return None
            
            # Работаем как с DataFrame, так и с Series
            if isinstance(data, pd.Series):
                delta = data.diff()
            else:
                delta = data['close'].diff()
                
            gain = (delta.where(delta > 0, 0)).rolling(period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            # Сохраняем в кэш
            self.cache[cache_key] = {
                'value': rsi,
                'timestamp': time.time()
            }
            self.access_times[cache_key] = time.time()
            
            return rsi
            
        except Exception as e:
            logger.error(f"Ошибка расчета RSI для {symbol}: {e}")
            return None
    
    def get_atr(self, symbol: str, data: pd.DataFrame, period: int = 14) -> Optional[float]:
        """Получение ATR с кэшированием"""
        try:
            self._cleanup_expired()
            
            data_hash = self._get_data_hash(data)
            cache_key = self._generate_cache_key(symbol, data_hash, "atr", period=period)
            
            # Проверяем кэш
            if cache_key in self.cache:
                self.access_times[cache_key] = time.time()
                return self.cache[cache_key]['value']
            
            # Вычисляем ATR
            if len(data) < period:
                return None
            
            high_low = data['high'] - data['low']
            high_close = np.abs(data['high'] - data['close'].shift())
            low_close = np.abs(data['low'] - data['close'].shift())
            true_range = np.maximum(high_low, np.maximum(high_close, low_close))
            atr = true_range.rolling(period).mean().iloc[-1]
            
            # Сохраняем в кэш
            self.cache[cache_key] = {
                'value': float(atr),
                'timestamp': time.time()
            }
            self.access_times[cache_key] = time.time()
            
            return float(atr)
            
        except Exception as e:
            logger.error(f"Ошибка расчета ATR для {symbol}: {e}")
            return None
    
    def get_multiple_sma(self, symbol: str, data: pd.DataFrame, 
                        periods: list) -> Dict[int, Optional[float]]:
        """Получение нескольких SMA одновременно"""
        result = {}
        for period in periods:
            result[period] = self.get_sma(symbol, data, period)
        return result
    
    def get_bollinger_bands(self, symbol: str, data: pd.DataFrame, 
                           period: int = 20, std_dev: float = 2.0) -> Optional[Dict[str, float]]:
        """Получение полос Боллинджера с кэшированием"""
        try:
            self._cleanup_expired()
            
            data_hash = self._get_data_hash(data)
            cache_key = self._generate_cache_key(symbol, data_hash, "bollinger", 
                                               period=period, std_dev=std_dev)
            
            # Проверяем кэш
            if cache_key in self.cache:
                self.access_times[cache_key] = time.time()
                return self.cache[cache_key]['value']
            
            # Вычисляем полосы Боллинджера
            if len(data) < period:
                return None
            
            sma = data['close'].rolling(period).mean()
            std = data['close'].rolling(period).std()
            
            upper_band = sma + (std * std_dev)
            lower_band = sma - (std * std_dev)
            
            result = {
                'upper': float(upper_band.iloc[-1]),
                'middle': float(sma.iloc[-1]),
                'lower': float(lower_band.iloc[-1])
            }
            
            # Сохраняем в кэш
            self.cache[cache_key] = {
                'value': result,
                'timestamp': time.time()
            }
            self.access_times[cache_key] = time.time()
            
            return result
            
        except Exception as e:
            logger.error(f"Ошибка расчета полос Боллинджера для {symbol}: {e}")
            return None
    
    def clear_cache(self):
        """Очистка всего кэша"""
        self.cache.clear()
        self.access_times.clear()
        logger.info("Кэш индикаторов полностью очищен")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Получение статистики кэша"""
        return {
            'cache_size': len(self.cache),
            'max_size': self.max_cache_size,
            'ttl_seconds': self.ttl_seconds,
            'last_cleanup': datetime.fromtimestamp(self.last_cleanup).isoformat()
        }

# Глобальный экземпляр кэша
indicators_cache = IndicatorsCache()