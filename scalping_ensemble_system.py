#!/usr/bin/env python3
"""
Гибридная скальпинг система ансамбля AI моделей
Peper Binance v4 - Командная работа всех AI для скальпинга
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Импорт AI моделей из существующих модулей
from ai_modules.lgbm_ai import LGBMAI
from ai_modules.lava_ai import LavaAI
from ai_modules.mistral_ai import MistralAI
from ai_modules.trading_ai import TradingAI
from ai_modules.ai_manager import AIManager
from historical_data_manager import HistoricalDataManager
from data_collector import BinanceDataCollector

logger = logging.getLogger(__name__)

class MarketPhase(Enum):
    """Фазы рынка"""
    ACCUMULATION = "accumulation"      # Накопление
    MARKUP = "markup"                  # Разметка (рост)
    DISTRIBUTION = "distribution"      # Распределение
    MARKDOWN = "markdown"              # Снижение
    CONSOLIDATION = "consolidation"    # Консолидация
    BREAKOUT = "breakout"             # Пробой
    REVERSAL = "reversal"             # Разворот

@dataclass
class ScalpingSignal:
    """Скальпинг сигнал"""
    action: str  # 'BUY', 'SELL', 'HOLD'
    confidence: float  # 0.0 - 1.0
    entry_price: float
    stop_loss: float
    take_profit: float
    expected_profit_pips: float
    expected_profit_after_fees: float
    market_phase: MarketPhase
    ai_consensus: Dict[str, float]  # Консенсус каждой AI
    reasoning: str
    timeframe: str
    timestamp: datetime

@dataclass
class AIExpertise:
    """Экспертиза каждой AI"""
    model_name: str
    specialty: str
    confidence: float
    analysis: Dict[str, Any]
    market_phase_vote: MarketPhase
    signal_strength: float  # -1.0 до 1.0

class ScalpingEnsembleSystem:
    """Гибридная скальпинг система ансамбля"""
    
    def __init__(self):
        # AI модели с их специализациями
        self.ai_models = {
            'lgbm': {
                'model': LGBMAI(),
                'specialty': 'price_action',  # Ценовое действие
                'expertise': ['support_resistance', 'breakouts', 'momentum'],
                'weight': 0.25
            },
            'lava': {
                'model': LavaAI(),
                'specialty': 'volume_analysis',  # Объемный анализ
                'expertise': ['volume_patterns', 'liquidity', 'order_flow'],
                'weight': 0.25
            },
            'mistral': {
                'model': MistralAI(),
                'specialty': 'pattern_recognition',  # Распознавание паттернов
                'expertise': ['chart_patterns', 'market_structure', 'sentiment'],
                'weight': 0.25
            },
            'trading_ai': {
                'model': TradingAI(),
                'specialty': 'technical_indicators',  # Технические индикаторы
                'expertise': ['oscillators', 'trend_indicators', 'volatility'],
                'weight': 0.25
            }
        }
        
        # AI Manager для управления моделями
        self.ai_manager = AIManager()
        
        # Добавляем модели в ai_manager для совместимости
        self.ai_manager.models = {
            'lgbm_ai': self.ai_models['lgbm']['model'],
            'lava_ai': self.ai_models['lava']['model'],
            'mistral_ai': self.ai_models['mistral']['model'],
            'trading_ai': self.ai_models['trading_ai']['model']
        }
        
        # Менеджер данных
        self.data_manager = HistoricalDataManager()
        
        # Коллектор данных для загрузки новых данных
        self.data_collector = BinanceDataCollector()
        
        # Конфигурация скальпинга
        self.scalping_config = {
            'timeframes': ['1m', '3m', '5m'],  # Только быстрые таймфреймы
            'min_profit_pips': 5,              # Увеличено с 3 до 5 пипсов прибыли
            'max_risk_pips': 3,                # Уменьшено с 5 до 3 пипсов риска
            'binance_fee': 0.001,              # 0.1% комиссия Binance
            'min_confidence': 0.85,            # Увеличено с 0.75 до 0.85 (85%)
            'max_trades_per_hour': 5,          # Уменьшено с 10 до 5 сделок в час
            'risk_reward_ratio': 2.0,          # Увеличено с 1.5 до 2.0
            'market_phases_for_trading': [     # Торгуем только в самых сильных фазах
                MarketPhase.BREAKOUT,          # Только пробои - самые надежные
                MarketPhase.MARKUP             # И сильный рост
            ]
        }
        
        # Статистика
        self.trade_history = []
        self.performance_metrics = {}
        
    async def analyze_market_phase(self, symbol: str, timeframe: str = '1m') -> Tuple[MarketPhase, Dict[str, AIExpertise]]:
        """Совместное определение фазы рынка всеми AI"""
        
        # Получаем данные
        data = await self.data_manager.load_data(symbol, timeframe)
        
        if data is None or len(data) < 50:
            # Если нет данных в HistoricalDataManager, используем data_collector
            async with self.data_collector as collector:
                data = await collector.get_historical_data(symbol, timeframe, days=7)
        
        if data is None or len(data) < 50:
            # Генерируем демо-данные для тестирования
            logger.info(f"Генерируем демо-данные для {symbol}")
            end_date = datetime.now()
            start_date = end_date - timedelta(days=7)
            data = await self.data_manager._generate_demo_data(symbol, timeframe, start_date, end_date)
        
        if data is None or len(data) < 50:
            raise ValueError(f"Недостаточно данных для анализа {symbol}")
        
        # Каждая AI анализирует свою область экспертизы
        ai_analyses = {}
        
        # LGBM AI - Ценовое действие
        lgbm_analysis = await self._lgbm_price_action_analysis(data)
        ai_analyses['lgbm'] = AIExpertise(
            model_name='lgbm',
            specialty='price_action',
            confidence=lgbm_analysis['confidence'],
            analysis=lgbm_analysis,
            market_phase_vote=lgbm_analysis['phase_vote'],
            signal_strength=lgbm_analysis['signal_strength']
        )
        
        # Lava AI - Объемный анализ
        lava_analysis = await self._lava_volume_analysis(data)
        ai_analyses['lava'] = AIExpertise(
            model_name='lava',
            specialty='volume_analysis',
            confidence=lava_analysis['confidence'],
            analysis=lava_analysis,
            market_phase_vote=lava_analysis['phase_vote'],
            signal_strength=lava_analysis['signal_strength']
        )
        
        # Mistral AI - Паттерны
        mistral_analysis = await self._mistral_pattern_analysis(data)
        ai_analyses['mistral'] = AIExpertise(
            model_name='mistral',
            specialty='pattern_recognition',
            confidence=mistral_analysis['confidence'],
            analysis=mistral_analysis,
            market_phase_vote=mistral_analysis['phase_vote'],
            signal_strength=mistral_analysis['signal_strength']
        )
        
        # Trading AI - Технические индикаторы
        trading_analysis = await self._trading_ai_technical_analysis(data)
        ai_analyses['trading_ai'] = AIExpertise(
            model_name='trading_ai',
            specialty='technical_indicators',
            confidence=trading_analysis['confidence'],
            analysis=trading_analysis,
            market_phase_vote=trading_analysis['phase_vote'],
            signal_strength=trading_analysis['signal_strength']
        )
        
        # Определяем итоговую фазу рынка через голосование
        final_phase = self._determine_market_phase_consensus(ai_analyses)
        
        return final_phase, ai_analyses
    
    async def generate_scalping_signal(self, symbol: str, timeframe: str = '1m') -> Optional[ScalpingSignal]:
        """Генерация скальпинг сигнала командой AI"""
        
        try:
            # Анализируем фазу рынка
            market_phase, ai_analyses = await self.analyze_market_phase(symbol, timeframe)
            
            # Проверяем, подходит ли фаза для торговли
            if market_phase not in self.scalping_config['market_phases_for_trading']:
                logger.info(f"Фаза рынка {market_phase.value} не подходит для скальпинга")
                return None
            
            # Получаем текущие данные
            current_data = await self.data_manager.load_data(symbol, timeframe)
            
            if current_data is None or len(current_data) < 50:
                # Если нет данных в HistoricalDataManager, используем data_collector
                async with self.data_collector as collector:
                    current_data = await collector.get_historical_data(symbol, timeframe, days=7)
            
            if current_data is None or len(current_data) < 50:
                logger.warning(f"Недостаточно данных для генерации сигнала {symbol}")
                return None
                
            current_price = float(current_data['close'].iloc[-1])
            
            # Собираем консенсус AI
            ai_consensus = {}
            total_signal_strength = 0
            total_confidence = 0
            
            for ai_name, analysis in ai_analyses.items():
                ai_consensus[ai_name] = analysis.confidence * analysis.signal_strength
                total_signal_strength += analysis.signal_strength * self.ai_models[ai_name]['weight']
                total_confidence += analysis.confidence * self.ai_models[ai_name]['weight']
            
            # Проверяем минимальную уверенность
            if total_confidence < self.scalping_config['min_confidence']:
                logger.info(f"Недостаточная уверенность: {total_confidence:.3f}")
                return None
            
            # Определяем направление сделки с фильтрами
            
            # Фильтр тренда - анализируем EMA на более высоком таймфрейме
            trend_filter_passed = await self._check_trend_filter(symbol, current_price)
            if not trend_filter_passed:
                logger.info(f"Сигнал отклонен фильтром тренда для {symbol}")
                return None
            
            # Фильтр объемов - проверяем достаточный объем
            volume_filter_passed = await self._check_volume_filter(symbol)
            if not volume_filter_passed:
                logger.info(f"Сигнал отклонен фильтром объемов для {symbol}")
                return None
            
            # Фильтр волатильности - избегаем экстремальных условий
            volatility_filter_passed = await self._check_volatility_filter(symbol, current_price)
            if not volatility_filter_passed:
                logger.info(f"Сигнал отклонен фильтром волатильности для {symbol}")
                return None
            
            # Дополнительный фильтр качества сигнала
            signal_quality_score = await self._calculate_signal_quality(symbol, total_signal_strength, total_confidence)
            if signal_quality_score < 0.8:  # Повышен порог качества с 0.6 до 0.8
                logger.info(f"Сигнал отклонен по качеству: {signal_quality_score:.3f} для {symbol}")
                return None
            
            # Определяем направление сделки только после всех фильтров - ужесточенные условия
            if total_signal_strength >= 0.7:  # Повышен порог для BUY с 0.4 до 0.7
                action = 'BUY'
            elif total_signal_strength <= -0.7:  # Повышен порог для SELL с -0.4 до -0.7
                action = 'SELL'
            else:
                logger.info(f"Недостаточная сила сигнала: {total_signal_strength:.3f}")
                return None
            
            # Рассчитываем уровни входа, стопа и тейка
            entry_price = current_price
            
            # Динамический стоп-лосс на основе волатильности
            dynamic_stop_loss_pct = await self._calculate_dynamic_stop_loss(symbol, current_price)
            
            if action == 'BUY':
                stop_loss = entry_price * (1 - dynamic_stop_loss_pct)
                # Улучшенный тейк-профит с учетом соотношения риск/прибыль
                profit_target_pct = dynamic_stop_loss_pct * self.scalping_config['risk_reward_ratio']
                take_profit = entry_price * (1 + profit_target_pct)
            else:  # SELL
                stop_loss = entry_price * (1 + dynamic_stop_loss_pct)
                # Улучшенный тейк-профит с учетом соотношения риск/прибыль
                profit_target_pct = dynamic_stop_loss_pct * self.scalping_config['risk_reward_ratio']
                take_profit = entry_price * (1 - profit_target_pct)
            
            # Рассчитываем ожидаемую прибыль с учетом комиссий
            expected_profit_pips = abs(take_profit - entry_price) * 10000
            expected_profit_after_fees = expected_profit_pips - (2 * self.scalping_config['binance_fee'] * 10000)
            
            # Проверяем прибыльность после комиссий
            if expected_profit_after_fees < self.scalping_config['min_profit_pips']:
                logger.info(f"Недостаточная прибыль после комиссий: {expected_profit_after_fees:.2f} пипсов")
                return None
            
            # Создаем обоснование
            reasoning = self._create_signal_reasoning(ai_analyses, market_phase, total_signal_strength)
            
            # Создаем сигнал
            signal = ScalpingSignal(
                action=action,
                confidence=total_confidence,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                expected_profit_pips=expected_profit_pips,
                expected_profit_after_fees=expected_profit_after_fees,
                market_phase=market_phase,
                ai_consensus=ai_consensus,
                reasoning=reasoning,
                timeframe=timeframe,
                timestamp=datetime.now()
            )
            
            return signal
            
        except Exception as e:
            logger.error(f"Ошибка генерации сигнала: {e}")
            return None
    
    async def _lgbm_price_action_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """LGBM AI анализирует технические индикаторы (RSI, MACD, Bollinger Bands)"""
        
        try:
            # Используем реальный LGBM AI модуль
            lgbm_model = self.ai_models['lgbm']['model']
            
            # Инициализируем модель если нужно
            if not hasattr(lgbm_model, 'initialized'):
                await lgbm_model.initialize()
                lgbm_model.initialized = True
            
            # Создаем торговые фичи для LGBM
            features_df = await lgbm_model.create_trading_features(data)
            
            # Получаем предсказание движения цены
            prediction = await lgbm_model.predict_price_movement(data)
            
            # Определяем фазу рынка на основе предсказания
            pred_value = prediction.prediction if hasattr(prediction, 'prediction') else 0
            pred_confidence = prediction.confidence if hasattr(prediction, 'confidence') else 0
            
            if pred_confidence > 0.7:
                if pred_value > 0.005:  # Сильный рост
                    phase_vote = MarketPhase.BREAKOUT
                    signal_strength = 0.8
                elif pred_value > 0.002:  # Умеренный рост
                    phase_vote = MarketPhase.MARKUP
                    signal_strength = 0.5
                elif pred_value < -0.005:  # Сильное падение
                    phase_vote = MarketPhase.MARKDOWN
                    signal_strength = -0.8
                elif pred_value < -0.002:  # Умеренное падение
                    phase_vote = MarketPhase.MARKDOWN
                    signal_strength = -0.5
                else:
                    phase_vote = MarketPhase.CONSOLIDATION
                    signal_strength = 0.0
            else:
                phase_vote = MarketPhase.CONSOLIDATION
                signal_strength = 0.0
            
            return {
                'confidence': pred_confidence,
                'phase_vote': phase_vote,
                'signal_strength': signal_strength,
                'prediction': pred_value,
                'feature_importance': prediction.feature_importance if hasattr(prediction, 'feature_importance') else {},
                'model_type': prediction.model_type
            }
            
        except Exception as e:
            logger.warning(f"LGBM анализ не удался, используем fallback: {e}")
            # Fallback к простому анализу
            current_price = data['close'].iloc[-1]
            price_change = (current_price - data['close'].iloc[-20]) / data['close'].iloc[-20]
            
            if abs(price_change) > 0.005:
                phase_vote = MarketPhase.BREAKOUT if price_change > 0 else MarketPhase.MARKDOWN
                signal_strength = 0.6 if price_change > 0 else -0.6
            else:
                phase_vote = MarketPhase.CONSOLIDATION
                signal_strength = 0.0
            
            return {
                'confidence': 0.5,
                'phase_vote': phase_vote,
                'signal_strength': signal_strength,
                'prediction': price_change,
                'fallback': True
            }
    
    async def _lava_volume_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Lava AI анализирует паттерны и уровни поддержки/сопротивления"""
        
        try:
            # Используем реальный Lava AI модуль
            lava_model = self.ai_models['lava']['model']
            
            # Инициализируем модель если нужно
            if not hasattr(lava_model, 'initialized'):
                await lava_model.initialize()
                lava_model.initialized = True
            
            # Анализируем паттерны
            patterns = await lava_model.analyze_patterns(data)
            
            # Определяем уровни поддержки/сопротивления
            levels = await lava_model.identify_support_resistance(data)
            
            # Получаем торговые сигналы
            signals = await lava_model.generate_trading_signals(data)
            
            # Определяем фазу рынка на основе паттернов
            pattern_strength = patterns.pattern_strength if hasattr(patterns, 'pattern_strength') else 0
            pattern_type = patterns.pattern_type if hasattr(patterns, 'pattern_type') else ''
            
            if pattern_strength > 0.7:
                if pattern_type in ['bullish_breakout', 'ascending_triangle']:
                    phase_vote = MarketPhase.BREAKOUT
                    signal_strength = 0.8
                elif pattern_type in ['bullish_flag', 'cup_and_handle']:
                    phase_vote = MarketPhase.MARKUP
                    signal_strength = 0.6
                elif pattern_type in ['bearish_breakout', 'descending_triangle']:
                    phase_vote = MarketPhase.MARKDOWN
                    signal_strength = -0.8
                elif pattern_type in ['rectangle', 'sideways']:
                    phase_vote = MarketPhase.CONSOLIDATION
                    signal_strength = 0.0
                else:
                    phase_vote = MarketPhase.CONSOLIDATION
                    signal_strength = 0.0
            else:
                phase_vote = MarketPhase.CONSOLIDATION
                signal_strength = 0.0
            
            return {
                'confidence': pattern_strength,
                'phase_vote': phase_vote,
                'signal_strength': signal_strength,
                'pattern_type': pattern_type,
                'support_levels': getattr(levels, 'support_levels', []),
                'resistance_levels': getattr(levels, 'resistance_levels', []),
                'trading_signals': getattr(signals, 'signals', [])
            }
            
        except Exception as e:
            logger.warning(f"Lava анализ не удался, используем fallback: {e}")
            # Fallback к простому анализу объемов
            volume_ma = data['volume'].rolling(window=20).mean()
            current_volume = data['volume'].iloc[-5:].mean()
            volume_ratio = current_volume / volume_ma.iloc[-1]
            
            # Анализ ликвидности
            price_volume_trend = np.corrcoef(data['close'].iloc[-20:], data['volume'].iloc[-20:])[0, 1]
            
            # Опреденение фазы по объемам
            if volume_ratio > 1.5 and price_volume_trend > 0.3:
                phase_vote = MarketPhase.BREAKOUT
                signal_strength = 0.6
            elif volume_ratio > 1.2:
                phase_vote = MarketPhase.MARKUP if data['close'].iloc[-1] > data['close'].iloc[-10] else MarketPhase.MARKDOWN
                signal_strength = 0.3 if data['close'].iloc[-1] > data['close'].iloc[-10] else -0.3
            else:
                phase_vote = MarketPhase.CONSOLIDATION
                signal_strength = 0.0
            
            return {
                'confidence': min(0.7, volume_ratio * 0.3 + abs(price_volume_trend) * 0.5),
                'phase_vote': phase_vote,
                'signal_strength': signal_strength,
                'volume_ratio': volume_ratio,
                'price_volume_trend': price_volume_trend,
                'liquidity_score': volume_ratio * abs(price_volume_trend),
                'fallback': True
            }
    
    async def _mistral_pattern_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Mistral AI анализирует новости и настроения рынка"""
        
        try:
            # Используем реальный Mistral AI модуль
            mistral_model = self.ai_models['mistral']['model']
            
            # Инициализируем модель если нужно
            if not hasattr(mistral_model, 'initialized'):
                await mistral_model.initialize()
                mistral_model.initialized = True
            
            # Анализируем рыночные новости и настроения
            current_price = data['close'].iloc[-1]
            symbol = "BTCUSDT"  # Можно передавать как параметр
            
            # Получаем анализ новостей
            news_analysis = await mistral_model.analyze_trading_data(
                symbol=symbol,
                current_price=current_price,
                price_data=data.tail(100).to_dict('records')
            )
            
            # Определяем фазу рынка на основе анализа новостей
            news_confidence = news_analysis.confidence if hasattr(news_analysis, 'confidence') else 0
            sentiment_score = news_analysis.sentiment_score if hasattr(news_analysis, 'sentiment_score') else 0.5
            
            if news_confidence > 0.7:
                if sentiment_score > 0.6:
                    phase_vote = MarketPhase.MARKUP
                    signal_strength = 0.7
                elif sentiment_score > 0.3:
                    phase_vote = MarketPhase.ACCUMULATION
                    signal_strength = 0.4
                elif sentiment_score < -0.6:
                    phase_vote = MarketPhase.MARKDOWN
                    signal_strength = -0.7
                elif sentiment_score < -0.3:
                    phase_vote = MarketPhase.DISTRIBUTION
                    signal_strength = -0.4
                else:
                    phase_vote = MarketPhase.CONSOLIDATION
                    signal_strength = 0.0
            else:
                phase_vote = MarketPhase.CONSOLIDATION
                signal_strength = 0.0
            
            return {
                'confidence': news_confidence,
                'phase_vote': phase_vote,
                'signal_strength': signal_strength,
                'sentiment_score': sentiment_score,
                'key_factors': getattr(news_analysis, 'key_factors', ['Анализ новостей']),
                'market_outlook': getattr(news_analysis, 'market_outlook', 'Нейтральный')
            }
            
        except Exception as e:
            logger.warning(f"Mistral анализ не удался, используем fallback: {e}")
            # Fallback к простому анализу паттернов
            closes = data['close'].values
            
            # Паттерн "флаг"
            recent_trend = (closes[-1] - closes[-10]) / closes[-10]
            consolidation = np.std(closes[-5:]) / np.mean(closes[-5:])
            
            # Паттерн "треугольник"
            highs_trend = np.polyfit(range(10), data['high'].iloc[-10:], 1)[0]
            lows_trend = np.polyfit(range(10), data['low'].iloc[-10:], 1)[0]
            
            # Определение фазы
            if abs(recent_trend) > 0.01 and consolidation < 0.005:
                phase_vote = MarketPhase.BREAKOUT
                signal_strength = 0.5 if recent_trend > 0 else -0.5
            elif consolidation < 0.003:
                phase_vote = MarketPhase.CONSOLIDATION
                signal_strength = 0.0
            else:
                phase_vote = MarketPhase.DISTRIBUTION if recent_trend > 0 else MarketPhase.ACCUMULATION
                signal_strength = 0.2 if recent_trend > 0 else -0.2
            
            return {
                'confidence': min(0.6, abs(recent_trend) * 50 + (1 - consolidation) * 0.5),
                'phase_vote': phase_vote,
                'signal_strength': signal_strength,
                'recent_trend': recent_trend,
                'consolidation': consolidation,
                'pattern_strength': abs(highs_trend - lows_trend),
                'fallback': True
            }
    
    async def _trading_ai_technical_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Trading AI анализирует риск-менеджмент и управление позициями"""
        
        try:
            # Используем реальный Trading AI модуль
            trading_model = self.ai_models['trading_ai']['model']
            
            # Инициализируем модель если нужно
            if not hasattr(trading_model, 'initialized'):
                await trading_model.initialize()
                trading_model.initialized = True
            
            # Анализируем риски и управление позициями
            current_price = data['close'].iloc[-1]
            
            # Получаем анализ рисков
            risk_analysis = await trading_model.analyze_risk_management(
                symbol="BTCUSDT",
                data=data.tail(100),
                current_price=current_price
            )
            
            # Получаем рекомендации по позициям
            position_analysis = await trading_model.optimize_position_sizing(
                price_data=data.tail(100),
                volatility=data['close'].pct_change().rolling(20).std().iloc[-1]
            )
            
            # Определяем фазу рынка на основе риск-анализа
            risk_confidence = risk_analysis.get('confidence', 0)
            risk_level = risk_analysis.get('risk_level', 'MEDIUM')
            position_size = position_analysis.get('recommended_position_size', 0.01)
            
            # Преобразуем risk_level в числовое значение
            risk_numeric = 0.5
            if risk_level == 'LOW':
                risk_numeric = 0.2
            elif risk_level == 'HIGH':
                risk_numeric = 0.8
            
            if risk_confidence > 0.7:
                if risk_numeric < 0.3 and position_size > 0.02:
                    phase_vote = MarketPhase.MARKUP
                    signal_strength = 0.6
                elif risk_numeric < 0.5 and position_size > 0.015:
                    phase_vote = MarketPhase.ACCUMULATION
                    signal_strength = 0.4
                elif risk_numeric > 0.7:
                    phase_vote = MarketPhase.DISTRIBUTION
                    signal_strength = -0.6
                elif risk_numeric > 0.5:
                    phase_vote = MarketPhase.MARKDOWN
                    signal_strength = -0.4
                else:
                    phase_vote = MarketPhase.CONSOLIDATION
                    signal_strength = 0.0
            else:
                phase_vote = MarketPhase.CONSOLIDATION
                signal_strength = 0.0
            
            return {
                'confidence': risk_confidence,
                'phase_vote': phase_vote,
                'signal_strength': signal_strength,
                'risk_level': risk_numeric,
                'position_size': position_size,
                'stop_loss_level': risk_analysis.get('recommended_stop_loss', current_price * 0.99),
                'take_profit_level': current_price * 1.01  # Простой расчет
            }
            
        except Exception as e:
            logger.warning(f"Trading AI анализ не удался, используем fallback: {e}")
            # Fallback к простому техническому анализу
            # RSI
            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            current_rsi = rsi.iloc[-1]
            
            # MACD
            ema12 = data['close'].ewm(span=12).mean()
            ema26 = data['close'].ewm(span=26).mean()
            macd = ema12 - ema26
            signal_line = macd.ewm(span=9).mean()
            macd_histogram = macd - signal_line
            
            # Bollinger Bands
            bb_middle = data['close'].rolling(window=20).mean()
            bb_std = data['close'].rolling(window=20).std()
            bb_upper = bb_middle + (bb_std * 2)
            bb_lower = bb_middle - (bb_std * 2)
            
            current_price = data['close'].iloc[-1]
            bb_position = (current_price - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1])
            
            # Определение фазы
            if current_rsi > 70 and macd_histogram.iloc[-1] > 0:
                phase_vote = MarketPhase.DISTRIBUTION
                signal_strength = -0.4
            elif current_rsi < 30 and macd_histogram.iloc[-1] < 0:
                phase_vote = MarketPhase.ACCUMULATION
                signal_strength = 0.4
            elif macd_histogram.iloc[-1] > macd_histogram.iloc[-2] and current_rsi > 50:
                phase_vote = MarketPhase.MARKUP
                signal_strength = 0.3
            elif macd_histogram.iloc[-1] < macd_histogram.iloc[-2] and current_rsi < 50:
                phase_vote = MarketPhase.MARKDOWN
                signal_strength = -0.3
            else:
                phase_vote = MarketPhase.CONSOLIDATION
                signal_strength = 0.0
            
            return {
                'confidence': min(0.7, abs(current_rsi - 50) / 50 + abs(macd_histogram.iloc[-1]) * 10),
                'phase_vote': phase_vote,
                'signal_strength': signal_strength,
                'rsi': current_rsi,
                'macd_histogram': macd_histogram.iloc[-1],
                'bb_position': bb_position,
                'fallback': True
            }
    
    def _determine_market_phase_consensus(self, ai_analyses: Dict[str, AIExpertise]) -> MarketPhase:
        """Определение консенсуса по фазе рынка"""
        
        # Взвешенное голосование
        phase_votes = {}
        
        for ai_name, analysis in ai_analyses.items():
            phase = analysis.market_phase_vote
            weight = self.ai_models[ai_name]['weight'] * analysis.confidence
            
            if phase not in phase_votes:
                phase_votes[phase] = 0
            phase_votes[phase] += weight
        
        # Возвращаем фазу с максимальным весом
        return max(phase_votes, key=phase_votes.get)
    
    def _create_signal_reasoning(self, ai_analyses: Dict[str, AIExpertise], market_phase: MarketPhase, signal_strength: float) -> str:
        """Создание обоснования сигнала"""
        
        reasoning_parts = [
            f"Фаза рынка: {market_phase.value}",
            f"Общая сила сигнала: {signal_strength:.3f}"
        ]
        
        for ai_name, analysis in ai_analyses.items():
            specialty = self.ai_models[ai_name]['specialty']
            reasoning_parts.append(
                f"{ai_name.upper()} ({specialty}): уверенность {analysis.confidence:.2f}, "
                f"сила {analysis.signal_strength:.2f}"
            )
        
        return " | ".join(reasoning_parts)
    
    async def backtest_scalping_system(self, symbol: str, days: int = 7) -> Dict[str, Any]:
        """Бэктест скальпинг системы"""
        
        logger.info(f"Запуск бэктеста для {symbol} за {days} дней")
        
        # Используем BinanceDataCollector вместо HistoricalDataManager
        async with BinanceDataCollector() as collector:
            data = await collector.get_historical_data(symbol, '1m', days)
        
        if data is None or len(data) < 1000:
            raise ValueError("Недостаточно данных для бэктеста")
        
        trades = []
        balance = 10000  # Начальный баланс $10,000
        
        # Симуляция торговли
        for i in range(100, len(data) - 50, 10):  # Каждые 10 минут
            try:
                # Получаем данные для анализа
                analysis_data = data.iloc[i-100:i]
                
                # Генерируем сигнал (упрощенная версия)
                market_phase, ai_analyses = await self.analyze_market_phase_from_data(analysis_data)
                
                if market_phase in self.scalping_config['market_phases_for_trading']:
                    # Симулируем сделку
                    entry_price = data['close'].iloc[i]
                    
                    # Простая логика для бэктеста
                    total_signal = sum(a.signal_strength for a in ai_analyses.values()) / len(ai_analyses)
                    
                    if abs(total_signal) > 0.3:
                        action = 'BUY' if total_signal > 0 else 'SELL'
                        
                        # Рассчитываем результат через 5 минут
                        if i + 5 < len(data):
                            exit_price = data['close'].iloc[i + 5]
                            
                            if action == 'BUY':
                                profit_pct = (exit_price - entry_price) / entry_price
                            else:
                                profit_pct = (entry_price - exit_price) / entry_price
                            
                            # Учитываем комиссии
                            profit_pct -= 2 * self.scalping_config['binance_fee']
                            
                            trades.append({
                                'timestamp': data.index[i],
                                'action': action,
                                'entry_price': entry_price,
                                'exit_price': exit_price,
                                'profit_pct': profit_pct,
                                'market_phase': market_phase.value
                            })
                            
                            balance *= (1 + profit_pct)
            
            except Exception as e:
                continue
        
        # Анализ результатов
        if not trades:
            return {'error': 'Нет сделок для анализа'}
        
        trades_df = pd.DataFrame(trades)
        
        total_trades = len(trades)
        winning_trades = len(trades_df[trades_df['profit_pct'] > 0])
        winrate = winning_trades / total_trades if total_trades > 0 else 0
        
        avg_profit = trades_df['profit_pct'].mean()
        total_return = (balance - 10000) / 10000
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'winrate': winrate,
            'avg_profit_per_trade': avg_profit,
            'total_return': total_return,
            'final_balance': balance,
            'trades': trades[:10]  # Первые 10 сделок для примера
        }
    
    async def analyze_market_phase_from_data(self, data: pd.DataFrame) -> Tuple[MarketPhase, Dict[str, AIExpertise]]:
        """Анализ фазы рынка из готовых данных (для бэктеста)"""
        
        # Упрощенная версия анализа для бэктеста
        ai_analyses = {}
        
        # Простой анализ тренда
        price_change = (data['close'].iloc[-1] - data['close'].iloc[-20]) / data['close'].iloc[-20]
        volume_change = (data['volume'].iloc[-5:].mean() - data['volume'].iloc[-15:-5].mean()) / data['volume'].iloc[-15:-5].mean()
        
        # Создаем упрощенные анализы для каждой AI
        for ai_name in ['lgbm', 'lava', 'mistral', 'trading_ai']:
            if price_change > 0.01:
                phase_vote = MarketPhase.BREAKOUT
                signal_strength = 0.5
            elif price_change > 0.003:
                phase_vote = MarketPhase.MARKUP
                signal_strength = 0.3
            elif price_change < -0.003:
                phase_vote = MarketPhase.MARKDOWN
                signal_strength = -0.3
            else:
                phase_vote = MarketPhase.CONSOLIDATION
                signal_strength = 0.0
            
            ai_analyses[ai_name] = AIExpertise(
                model_name=ai_name,
                specialty=self.ai_models[ai_name]['specialty'],
                confidence=min(0.9, abs(price_change) * 50 + 0.5),
                analysis={'price_change': price_change, 'volume_change': volume_change},
                market_phase_vote=phase_vote,
                signal_strength=signal_strength
            )
        
        final_phase = self._determine_market_phase_consensus(ai_analyses)
        return final_phase, ai_analyses
    
    async def _check_trend_filter(self, symbol: str, current_price: float) -> bool:
        """Улучшенный фильтр тренда - строгая проверка направления тренда"""
        try:
            # Получаем данные 5m для анализа тренда
            data_collector = BinanceDataCollector()
            trend_data = await data_collector.get_historical_data(symbol, '5m', days=1)
            
            if trend_data is None or len(trend_data) < 50:
                return False  # Без достаточных данных не торгуем
            
            # Рассчитываем множественные EMA для более точного анализа тренда
            trend_data['ema_9'] = trend_data['close'].ewm(span=9).mean()
            trend_data['ema_21'] = trend_data['close'].ewm(span=21).mean()
            trend_data['ema_50'] = trend_data['close'].ewm(span=50).mean()
            
            latest_ema_9 = trend_data['ema_9'].iloc[-1]
            latest_ema_21 = trend_data['ema_21'].iloc[-1]
            latest_ema_50 = trend_data['ema_50'].iloc[-1]
            
            # Проверяем силу тренда - все EMA должны быть выстроены правильно
            strong_uptrend = (latest_ema_9 > latest_ema_21 > latest_ema_50 and 
                             current_price > latest_ema_9)
            strong_downtrend = (latest_ema_9 < latest_ema_21 < latest_ema_50 and 
                               current_price < latest_ema_9)
            
            # Дополнительная проверка - тренд должен быть устойчивым
            # Проверяем последние 3 свечи
            recent_closes = trend_data['close'].tail(3).values
            trend_consistency = True
            
            if strong_uptrend:
                # Для восходящего тренда - последние закрытия должны расти
                trend_consistency = all(recent_closes[i] >= recent_closes[i-1] * 0.999 
                                      for i in range(1, len(recent_closes)))
            elif strong_downtrend:
                # Для нисходящего тренда - последние закрытия должны падать
                trend_consistency = all(recent_closes[i] <= recent_closes[i-1] * 1.001 
                                      for i in range(1, len(recent_closes)))
            
            return (strong_uptrend or strong_downtrend) and trend_consistency
            
        except Exception as e:
            logger.warning(f"Ошибка в фильтре тренда: {e}")
            return False  # При ошибке не торгуем
    
    async def _check_volume_filter(self, symbol: str) -> bool:
        """Улучшенный фильтр объемов - проверяет достаточность и качество объема"""
        try:
            # Получаем данные объемов за последние 20 свечей
            data_collector = BinanceDataCollector()
            volume_data = await data_collector.get_historical_data(symbol, '1m', days=1)  # 1 день данных
            
            if volume_data is None or len(volume_data) < 20:
                return False  # Без данных не торгуем
            
            # Рассчитываем средний объем за 20 периодов
            avg_volume_20 = volume_data['volume'].tail(20).mean()
            current_volume = volume_data['volume'].iloc[-1]
            
            # Проверяем объем за последние 3 свечи
            recent_volumes = volume_data['volume'].tail(3)
            avg_recent_volume = recent_volumes.mean()
            
            # Ужесточенные требования к объему для скальпинга
            volume_threshold_current = avg_volume_20 * 2.0  # Текущий объем должен быть в 2 раза выше среднего
            volume_threshold_recent = avg_volume_20 * 1.5   # Средний объем за 3 свечи в 1.5 раза выше
            
            # Дополнительная проверка - объем не должен быть экстремально высоким (возможная манипуляция)
            max_volume_threshold = avg_volume_20 * 10  # Не более чем в 10 раз выше среднего
            
            volume_ok = (current_volume >= volume_threshold_current and 
                        avg_recent_volume >= volume_threshold_recent and
                        current_volume <= max_volume_threshold)
            
            return volume_ok
            
        except Exception as e:
            logger.warning(f"Ошибка в фильтре объемов: {e}")
            return False  # При ошибке не торгуем
    
    async def _check_volatility_filter(self, symbol: str, current_price: float) -> bool:
        """Фильтр волатильности - избегает торговли в экстремальных условиях"""
        try:
            # Получаем данные для расчета волатильности
            data_collector = BinanceDataCollector()
            volatility_data = await data_collector.get_historical_data(symbol, '1m', days=1)
            
            if volatility_data is None or len(volatility_data) < 20:
                return False  # Без данных не торгуем
            
            # Рассчитываем ATR для оценки волатильности
            volatility_data['high_low'] = volatility_data['high'] - volatility_data['low']
            volatility_data['high_close'] = abs(volatility_data['high'] - volatility_data['close'].shift(1))
            volatility_data['low_close'] = abs(volatility_data['low'] - volatility_data['close'].shift(1))
            volatility_data['true_range'] = volatility_data[['high_low', 'high_close', 'low_close']].max(axis=1)
            
            # ATR за 14 и 5 периодов
            atr_14 = volatility_data['true_range'].tail(14).mean()
            atr_5 = volatility_data['true_range'].tail(5).mean()
            
            # Волатильность как процент от цены
            volatility_14_pct = (atr_14 / current_price) * 100
            volatility_5_pct = (atr_5 / current_price) * 100
            
            # Определяем приемлемые границы волатильности для скальпинга
            min_volatility = 0.02  # 0.02% - минимальная волатильность
            max_volatility = 0.25  # 0.25% - максимальная волатильность
            
            # Проверяем, что волатильность в приемлемых границах
            volatility_ok = (min_volatility <= volatility_14_pct <= max_volatility and
                           volatility_5_pct <= max_volatility * 1.5)  # Краткосрочная может быть чуть выше
            
            # Дополнительная проверка - резкие скачки волатильности
            volatility_spike = volatility_5_pct > volatility_14_pct * 3  # Скачок в 3 раза
            
            return volatility_ok and not volatility_spike
            
        except Exception as e:
            logger.warning(f"Ошибка в фильтре волатильности: {e}")
            return False  # При ошибке не торгуем
    
    async def _calculate_dynamic_stop_loss(self, symbol: str, current_price: float) -> float:
        """Рассчитывает оптимизированный динамический стоп-лосс для скальпинга"""
        try:
            # Получаем данные для расчета ATR
            async with self.data_collector as data_collector:
                atr_data = await data_collector.get_historical_data(symbol, '1m', days=1)  # 1 день данных
            
            if atr_data is None or len(atr_data) < 14:
                # Если нет данных, используем консервативный стоп-лосс
                return self.scalping_config['max_risk_pips'] / 10000
            
            # Рассчитываем ATR (Average True Range)
            atr_data['high_low'] = atr_data['high'] - atr_data['low']
            atr_data['high_close'] = abs(atr_data['high'] - atr_data['close'].shift(1))
            atr_data['low_close'] = abs(atr_data['low'] - atr_data['close'].shift(1))
            atr_data['true_range'] = atr_data[['high_low', 'high_close', 'low_close']].max(axis=1)
            
            # ATR за 14 периодов
            atr_14 = atr_data['true_range'].tail(14).mean()
            
            # Рассчитываем волатильность как процент от цены
            volatility_pct = (atr_14 / current_price) * 100
            
            # Базовый стоп-лосс - более консервативный для скальпинга
            base_stop_loss_pct = self.scalping_config['max_risk_pips'] / 10000
            
            # Адаптируем стоп-лосс к волатильности - более агрессивная оптимизация
            if volatility_pct > 0.2:  # Очень высокая волатильность - избегаем торговли
                return 0.004  # 0.4% - слишком рискованно
            elif volatility_pct > 0.12:  # Высокая волатильность
                dynamic_stop_loss_pct = base_stop_loss_pct * 1.2  # Немного увеличиваем
            elif volatility_pct < 0.03:  # Очень низкая волатильность
                dynamic_stop_loss_pct = base_stop_loss_pct * 0.5  # Сильно уменьшаем
            elif volatility_pct < 0.06:  # Низкая волатильность
                dynamic_stop_loss_pct = base_stop_loss_pct * 0.7  # Уменьшаем
            else:  # Нормальная волатильность
                dynamic_stop_loss_pct = base_stop_loss_pct * 0.8  # Немного уменьшаем
            
            # Жесткие ограничения для скальпинга - более узкие стопы
            min_stop_loss = 0.0003  # 0.03% - минимальный стоп
            max_stop_loss = 0.002   # 0.2% - максимальный стоп для скальпинга
            
            return max(min_stop_loss, min(max_stop_loss, dynamic_stop_loss_pct))
            
        except Exception as e:
            logger.warning(f"Ошибка в расчете динамического стоп-лосса: {e}")
            return self.scalping_config['max_risk_pips'] / 10000
    
    async def _calculate_signal_quality(self, symbol: str, signal_strength: float, confidence: float) -> float:
        """Рассчитывает качество сигнала на основе различных факторов"""
        try:
            # Базовое качество на основе силы сигнала и уверенности
            base_quality = (abs(signal_strength) + confidence) / 2
            
            # Получаем данные для дополнительных проверок
            data_collector = BinanceDataCollector()
            quality_data = await data_collector.get_historical_data(symbol, '1m', days=1)  # 1 день данных
            
            if quality_data is None or len(quality_data) < 20:
                return base_quality
            
            # Фактор стабильности цены (меньше шума = выше качество)
            price_stability = 1 - (quality_data['close'].pct_change().std() * 10)
            price_stability = max(0, min(1, price_stability))
            
            # Фактор объема (стабильный объем = выше качество)
            volume_stability = 1 - (quality_data['volume'].pct_change().std() * 5)
            volume_stability = max(0, min(1, volume_stability))
            
            # Фактор времени (избегаем торговли в неликвидные часы)
            current_hour = datetime.now().hour
            time_factor = 1.0
            if 22 <= current_hour or current_hour <= 6:  # Ночные часы UTC
                time_factor = 0.7
            elif 6 <= current_hour <= 8 or 20 <= current_hour <= 22:  # Переходные часы
                time_factor = 0.85
            
            # Итоговое качество сигнала
            final_quality = (
                base_quality * 0.5 +
                price_stability * 0.2 +
                volume_stability * 0.2 +
                time_factor * 0.1
            )
            
            return max(0, min(1, final_quality))
            
        except Exception as e:
            logger.warning(f"Ошибка в расчете качества сигнала: {e}")
            return confidence * 0.8  # Консервативная оценка при ошибке

# Пример использования
async def main():
    """Демонстрация работы скальпинг системы"""
    
    print("🚀 ГИБРИДНАЯ СКАЛЬПИНГ СИСТЕМА - КОМАНДНАЯ РАБОТА AI")
    print("="*60)
    
    # Создаем систему
    scalping_system = ScalpingEnsembleSystem()
    
    # Тестируем на BTCUSDT
    symbol = "BTCUSDT"
    
    try:
        # Генерируем сигнал
        print(f"📊 Анализ {symbol}...")
        signal = await scalping_system.generate_scalping_signal(symbol)
        
        if signal:
            print(f"\n✅ СИГНАЛ ПОЛУЧЕН:")
            print(f"   Действие: {signal.action}")
            print(f"   Уверенность: {signal.confidence:.1%}")
            print(f"   Цена входа: ${signal.entry_price:.2f}")
            print(f"   Стоп-лосс: ${signal.stop_loss:.2f}")
            print(f"   Тейк-профит: ${signal.take_profit:.2f}")
            print(f"   Ожидаемая прибыль: {signal.expected_profit_after_fees:.2f} пипсов")
            print(f"   Фаза рынка: {signal.market_phase.value}")
            print(f"   Обоснование: {signal.reasoning}")
        else:
            print("❌ Сигнал не сгенерирован - условия не подходят")
        
        # Запускаем бэктест
        print(f"\n🔄 Запуск бэктеста...")
        backtest_results = await scalping_system.backtest_scalping_system(symbol, days=1)
        
        if 'error' not in backtest_results:
            print(f"\n📈 РЕЗУЛЬТАТЫ БЭКТЕСТА:")
            print(f"   Всего сделок: {backtest_results['total_trades']}")
            print(f"   Прибыльных: {backtest_results['winning_trades']}")
            print(f"   Винрейт: {backtest_results['winrate']:.1%}")
            print(f"   Средняя прибыль: {backtest_results['avg_profit_per_trade']:.3%}")
            print(f"   Общая доходность: {backtest_results['total_return']:.1%}")
            print(f"   Финальный баланс: ${backtest_results['final_balance']:.2f}")
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")

if __name__ == "__main__":
    asyncio.run(main())