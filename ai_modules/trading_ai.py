"""
Trading AI модуль для Peper Binance v4
Легковесная реализация для трейдинга с минимальным потреблением ресурсов
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import gc
from dataclasses import dataclass
import config
from config_params import CONFIG_PARAMS
from utils.indicators_cache import indicators_cache

logger = logging.getLogger(__name__)

@dataclass
class TradingSignal:
    """Структура торгового сигнала"""
    symbol: str
    action: str  # 'BUY', 'SELL', 'HOLD'
    confidence: float
    price: float
    timestamp: datetime
    reason: str

class MemoryOptimizedCache:
    """Оптимизированный кэш для экономии памяти"""
    
    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self.cache = {}
        self.access_times = {}
    
    def get(self, key: str) -> Optional[Any]:
        if key in self.cache:
            self.access_times[key] = datetime.now()
            return self.cache[key]
        return None
    
    def set(self, key: str, value: Any):
        if len(self.cache) >= self.max_size:
            self._cleanup_old_entries()
        
        self.cache[key] = value
        self.access_times[key] = datetime.now()
    
    def _cleanup_old_entries(self):
        """Удаляем старые записи для освобождения памяти"""
        if not self.access_times:
            return
            
        # Удаляем 20% самых старых записей
        sorted_items = sorted(self.access_times.items(), key=lambda x: x[1])
        to_remove = int(len(sorted_items) * 0.2)
        
        for key, _ in sorted_items[:to_remove]:
            self.cache.pop(key, None)
            self.access_times.pop(key, None)
        
        # Принудительная сборка мусора
        gc.collect()

class TradingAI:
    """
    Легковесный Trading AI модуль
    Оптимизирован для минимального потребления ресурсов
    """
    
    def __init__(self):
        # Получаем конфигурацию Trading AI из CONFIG_PARAMS
        ai_config = CONFIG_PARAMS.get('ai_modules', {})
        trading_config = ai_config.get('trading', {})
        
        self.config = trading_config
        self.is_initialized = False
        self.cache = MemoryOptimizedCache(trading_config.get('model_cache_size', 10))
        self.models = {}
        self.last_cleanup = datetime.now()
        
        logger.info("Trading AI инициализирован с оптимизацией ресурсов")
    
    async def initialize(self):
        """Ленивая инициализация модуля"""
        if self.is_initialized:
            return True
        
        try:
            logger.info("Инициализация Trading AI модуля...")
            
            # Инициализируем только базовые компоненты
            await self._load_lightweight_models()
            
            self.is_initialized = True
            logger.info("Trading AI модуль успешно инициализирован")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка инициализации Trading AI: {e}")
            return False
    
    async def _load_lightweight_models(self):
        """Загрузка легковесных моделей"""
        # Простая модель на основе технических индикаторов
        self.models['technical'] = {
            'sma_periods': [5, 10, 20],
            'rsi_period': 14,
            'macd_params': (12, 26, 9)
        }
        
        # Модель волатильности
        self.models['volatility'] = {
            'window': 20,
            'threshold': 0.02
        }
        
        logger.info("Легковесные модели загружены")
    
    async def analyze_market(self, symbol: str, data: pd.DataFrame) -> TradingSignal:
        """
        Анализ рынка с минимальным потреблением ресурсов
        """
        if not self.is_initialized:
            await self.initialize()
        
        try:
            # Проверяем кэш
            cache_key = f"{symbol}_{len(data)}"
            cached_result = self.cache.get(cache_key)
            if cached_result:
                return cached_result
            
            # Вычисляем технические индикаторы эффективно
            signal = await self._calculate_trading_signal(symbol, data)
            
            # Кэшируем результат
            self.cache.set(cache_key, signal)
            
            # Периодическая очистка памяти
            await self._periodic_cleanup()
            
            return signal
            
        except Exception as e:
            logger.error(f"Ошибка анализа рынка для {symbol}: {e}")
            return TradingSignal(
                symbol=symbol,
                action='HOLD',
                confidence=0.0,
                price=data['close'].iloc[-1] if not data.empty else 0.0,
                timestamp=datetime.now(),
                reason=f"Ошибка анализа: {str(e)}"
            )
    
    async def _calculate_trading_signal(self, symbol: str, data: pd.DataFrame) -> TradingSignal:
        """Вычисление торгового сигнала с оптимизацией"""
        if data.empty or len(data) < 20:
            return TradingSignal(
                symbol=symbol,
                action='HOLD',
                confidence=0.0,
                price=0.0,
                timestamp=datetime.now(),
                reason="Недостаточно данных"
            )
        
        # Используем только последние данные для экономии памяти
        recent_data = data.tail(50).copy()
        
        # Быстрые технические индикаторы с использованием кэша
        sma_5_series = indicators_cache.get_sma(symbol, recent_data, 5)
        sma_20_series = indicators_cache.get_sma(symbol, recent_data, 20)
        current_price = recent_data['close'].iloc[-1]
        
        # RSI с использованием кэша
        rsi_series = indicators_cache.get_rsi(symbol, recent_data, 14)
        
        # Проверяем валидность индикаторов и извлекаем последние значения
        if sma_5_series is None or sma_20_series is None or rsi_series is None:
            return TradingSignal(
                symbol=symbol,
                action='HOLD',
                confidence=0.0,
                price=current_price,
                timestamp=datetime.now(),
                reason="Недостаточно данных для расчета индикаторов"
            )
        
        # Извлекаем последние значения
        sma_5 = sma_5_series.iloc[-1]
        sma_20 = sma_20_series.iloc[-1]
        rsi = rsi_series.iloc[-1]
        
        # Логика принятия решений - СНИЖАЕМ ПОРОГИ для более активной торговли
        action = 'HOLD'
        confidence = 0.5
        reason = "Нейтральный сигнал"
        
        # Вычисляем дополнительные индикаторы для более точных сигналов
        price_change = (current_price - recent_data['close'].iloc[-5]) / recent_data['close'].iloc[-5] * 100
        volume_avg = recent_data.get('volume', pd.Series([1])).rolling(10).mean().iloc[-1] if 'volume' in recent_data.columns else 1
        current_volume = recent_data.get('volume', pd.Series([1])).iloc[-1] if 'volume' in recent_data.columns else 1
        volume_ratio = current_volume / volume_avg if volume_avg > 0 else 1
        
        # Оптимизированные пороги для лучшего качества сигналов
        if sma_5 > sma_20 and rsi < 65:  # Оптимизировано с 75 до 65
            # Дополнительные условия для BUY
            if price_change > -2 or volume_ratio > 1.1:  # Добавлены альтернативные условия
                action = 'BUY'
                # Увеличиваем confidence на основе силы сигнала
                base_confidence = (sma_5 - sma_20) / sma_20 * 10  # Оптимизирован множитель с 20 до 10
                rsi_bonus = (65 - rsi) / 65 * 0.2 if rsi < 65 else 0
                volume_bonus = min(0.1, (volume_ratio - 1) * 0.1) if volume_ratio > 1 else 0
                confidence = min(0.85, 0.5 + base_confidence + rsi_bonus + volume_bonus)
                reason = f"BUY: SMA5 > SMA20 ({(sma_5/sma_20-1)*100:.1f}%), RSI: {rsi:.1f}, Vol: {volume_ratio:.1f}x"
        
        elif sma_5 < sma_20 and rsi > 35:  # Оптимизировано с 25 до 35
            # Дополнительные условия для SELL
            if price_change < 2 or volume_ratio > 1.1:  # Добавлены альтернативные условия
                action = 'SELL'
                # Увеличиваем confidence на основе силы сигнала
                base_confidence = (sma_20 - sma_5) / sma_20 * 10  # Оптимизирован множитель с 20 до 10
                rsi_bonus = (rsi - 35) / 65 * 0.2 if rsi > 35 else 0
                volume_bonus = min(0.1, (volume_ratio - 1) * 0.1) if volume_ratio > 1 else 0
                confidence = min(0.85, 0.5 + base_confidence + rsi_bonus + volume_bonus)
                reason = f"SELL: SMA5 < SMA20 ({(1-sma_5/sma_20)*100:.1f}%), RSI: {rsi:.1f}, Vol: {volume_ratio:.1f}x"
        
        # Дополнительные сигналы на основе RSI экстремумов
        elif rsi <= 25:  # Сильно перепродано
            action = 'BUY'
            confidence = min(0.8, 0.6 + (30 - rsi) / 30 * 0.3)
            reason = f"BUY: RSI перепродан ({rsi:.1f})"
        elif rsi >= 75:  # Сильно перекуплено
            action = 'SELL'
            confidence = min(0.8, 0.6 + (rsi - 70) / 30 * 0.3)
            reason = f"SELL: RSI перекуплен ({rsi:.1f})"
        
        return TradingSignal(
            symbol=symbol,
            action=action,
            confidence=confidence,
            price=current_price,
            timestamp=datetime.now(),
            reason=reason
        )
    
    async def _periodic_cleanup(self):
        """Периодическая очистка памяти"""
        now = datetime.now()
        if (now - self.last_cleanup).seconds > 300:  # Каждые 5 минут
            gc.collect()
            self.last_cleanup = now
            logger.debug("Выполнена очистка памяти Trading AI")
    
    async def get_portfolio_recommendation(self, portfolio: Dict[str, float], 
                                         market_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Рекомендации по портфелю с оптимизацией"""
        recommendations = {}
        
        for symbol, data in market_data.items():
            if symbol in portfolio:
                signal = await self.analyze_market(symbol, data)
                recommendations[symbol] = {
                    'signal': signal,
                    'current_position': portfolio[symbol],
                    'recommended_action': signal.action
                }
        
        return recommendations
    
    async def analyze_risk_management(self, symbol: str, data: pd.DataFrame, 
                                     current_price: float = None, **kwargs) -> Dict[str, Any]:
        """Анализ управления рисками"""
        try:
            if current_price is None:
                current_price = data['close'].iloc[-1]
            
            # Расчет волатильности с использованием кэша
            volatility = indicators_cache.get_volatility(symbol, data, period=20, method='daily_std')
            if volatility is None:
                volatility = 0.02  # Значение по умолчанию
            
            # Расчет ATR для стоп-лосса с использованием кэша (оптимизировано до 2.5 ATR)
            atr = indicators_cache.get_atr(symbol, data, period=14)
            if atr is None:
                atr = current_price * 0.02  # Значение по умолчанию 2% от цены
            
            # Рекомендации по управлению рисками с оптимизированными параметрами
            risk_level = "LOW" if volatility < 0.02 else "MEDIUM" if volatility < 0.05 else "HIGH"
            
            # Максимальная позиция: 5% от капитала
            max_position_size = 0.05
            recommended_position_size = min(max_position_size, 0.03 if risk_level == "LOW" else 0.02 if risk_level == "MEDIUM" else 0.01)
            
            return {
                'risk_level': risk_level,
                'volatility': float(volatility),
                'atr': float(atr),
                'recommended_stop_loss': float(current_price - atr * 2.5),  # Оптимизировано до 2.5 ATR
                'recommended_position_size': recommended_position_size,
                'max_position_size': max_position_size,  # 5% от капитала
                'daily_loss_limit': 0.03,  # Дневной лимит убытков: 3%
                'confidence': 0.8,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Ошибка анализа рисков: {e}")
            return {
                'risk_level': 'UNKNOWN',
                'volatility': 0.0,
                'atr': 0.0,
                'recommended_stop_loss': current_price * 0.99 if current_price else 0.0,
                'recommended_position_size': 0.01,
                'confidence': 0.0,
                'error': str(e)
            }

    async def optimize_position_sizing(self, price_data: pd.DataFrame = None, 
                                      volatility: float = None, **kwargs) -> Dict[str, Any]:
        """Оптимизация размера позиции"""
        try:
            if volatility is None and price_data is not None:
                volatility = price_data['close'].pct_change().std()
            elif volatility is None:
                volatility = 0.02  # Значение по умолчанию
            
            # Расчет оптимального размера позиции на основе волатильности
            if volatility < 0.01:
                position_size = 0.03  # Низкая волатильность - больший размер
            elif volatility < 0.03:
                position_size = 0.02  # Средняя волатильность
            else:
                position_size = 0.01  # Высокая волатильность - меньший размер
            
            return {
                'recommended_position_size': position_size,
                'max_position_size': position_size * 1.5,
                'min_position_size': position_size * 0.5,
                'volatility_adjusted': True,
                'confidence': 0.8
            }
            
        except Exception as e:
            logger.error(f"Ошибка оптимизации позиции: {e}")
            return {
                'recommended_position_size': 0.01,
                'max_position_size': 0.015,
                'min_position_size': 0.005,
                'volatility_adjusted': False,
                'confidence': 0.5,
                'error': str(e)
            }

    async def cleanup(self):
        """Очистка ресурсов модуля"""
        logger.info("Очистка ресурсов Trading AI...")
        
        self.cache.cache.clear()
        self.cache.access_times.clear()
        self.models.clear()
        
        # Принудительная сборка мусора
        gc.collect()
        
        self.is_initialized = False
        logger.info("Trading AI ресурсы очищены")
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Получение информации об использовании памяти"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        return {
            'rss_mb': memory_info.rss / 1024 / 1024,
            'vms_mb': memory_info.vms / 1024 / 1024,
            'cache_size': len(self.cache.cache),
            'models_loaded': len(self.models)
        }