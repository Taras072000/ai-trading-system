#!/usr/bin/env python3
"""
🚀 УЛУЧШЕННАЯ AI ТОРГОВАЯ СИСТЕМА ДЛЯ ДОСТИЖЕНИЯ 60%+ ВИНРЕЙТА
Комплексные улучшения для повышения точности и прибыльности
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import logging
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
from utils.timezone_utils import get_utc_now

# Импорты AI модулей
from ai_modules.multi_ai_orchestrator import MultiAIOrchestrator
from ai_modules.ai_manager import ai_manager, AIModuleType
from ai_modules.trading_ai import TradingAI
from ai_modules.lava_ai import LavaAI
from ai_modules.lgbm_ai import LGBMAI
from ai_modules.mistral_ai import MistralAI
from ai_modules.reinforcement_learning_engine import ReinforcementLearningEngine
from historical_data_manager import HistoricalDataManager
from data_collector import BinanceDataCollector

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class EnhancedAIDecision:
    """Улучшенное решение AI модели с дополнительными метриками"""
    model_name: str
    action: str  # 'BUY', 'SELL', 'HOLD'
    confidence: float
    reasoning: str
    timestamp: datetime
    
    # Новые поля для улучшения
    signal_strength: float = 0.0  # Сила сигнала (0-1)
    market_regime: str = "UNKNOWN"  # BULL, BEAR, SIDEWAYS
    risk_score: float = 0.5  # Оценка риска (0-1)
    feature_importance: Dict[str, float] = field(default_factory=dict)
    model_performance_score: float = 0.5  # Историческая производительность модели

@dataclass
class EnhancedConsensusSignal:
    """Улучшенный консенсус сигнал с дополнительной аналитикой"""
    symbol: str
    timestamp: datetime
    price: float
    final_action: str
    consensus_strength: int
    participating_models: List[EnhancedAIDecision]
    confidence_avg: float
    
    # Новые поля для улучшения
    weighted_confidence: float = 0.0  # Взвешенная уверенность по производительности
    signal_quality_score: float = 0.0  # Качество сигнала (0-1)
    market_conditions_score: float = 0.0  # Оценка рыночных условий
    risk_reward_ratio: float = 0.0  # Соотношение риск/прибыль
    expected_profit_probability: float = 0.0  # Ожидаемая вероятность прибыли

@dataclass
class EnhancedTestConfig:
    """Улучшенная конфигурация с адаптивными параметрами"""
    test_period_days: int = 7
    start_balance: float = 100.0
    symbols: List[str] = None
    commission_rate: float = 0.001
    
    # Адаптивное управление позициями
    base_position_size_percent: float = 0.02  # Базовый размер позиции 2%
    max_position_size_percent: float = 0.05   # Максимальный размер позиции 5%
    confidence_multiplier: float = 2.0        # Мультипликатор размера по уверенности
    
    # Динамические стоп-лоссы и тейк-профиты
    base_stop_loss_percent: float = 0.015     # Базовый стоп-лосс 1.5%
    base_take_profit_percent: float = 0.045   # Базовый тейк-профит 4.5% (1:3)
    volatility_multiplier: float = 1.5        # Мультипликатор по волатильности
    
    # Улучшенные фильтры качества
    min_signal_quality_score: float = 0.6     # Минимальное качество сигнала
    min_weighted_confidence: float = 0.3      # Минимальная взвешенная уверенность
    min_consensus_models: int = 2             # Минимум моделей для консенсуса
    min_model_performance_score: float = 0.4  # Минимальная производительность модели
    
    # Рыночные условия
    min_market_conditions_score: float = 0.5  # Минимальная оценка рынка
    max_daily_trades: int = 3                 # Максимум сделок в день
    max_symbol_exposure: float = 0.1          # Максимальная экспозиция на символ
    
    # Управление рисками
    max_daily_loss_percent: float = 0.05      # Максимальная дневная потеря 5%
    max_drawdown_percent: float = 0.15        # Максимальная просадка 15%
    correlation_threshold: float = 0.7        # Порог корреляции для избежания переэкспозиции
    
    # Режимы работы
    adaptive_parameters: bool = True          # Адаптивная настройка параметров
    market_regime_detection: bool = True      # Детекция режима рынка
    performance_tracking: bool = True         # Отслеживание производительности
    debug_mode: bool = True

    def __post_init__(self):
        if self.symbols is None:
            self.symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'SOLUSDT']

class ModelPerformanceTracker:
    """Отслеживание производительности AI моделей"""
    
    def __init__(self):
        self.model_stats = {}
        self.performance_history = {}
        
    def update_model_performance(self, model_name: str, prediction_correct: bool, 
                               confidence: float, actual_return: float):
        """Обновление статистики производительности модели"""
        if model_name not in self.model_stats:
            self.model_stats[model_name] = {
                'total_predictions': 0,
                'correct_predictions': 0,
                'total_confidence': 0.0,
                'total_return': 0.0,
                'accuracy': 0.0,
                'avg_confidence': 0.0,
                'avg_return': 0.0,
                'performance_score': 0.5
            }
        
        stats = self.model_stats[model_name]
        stats['total_predictions'] += 1
        if prediction_correct:
            stats['correct_predictions'] += 1
        stats['total_confidence'] += confidence
        stats['total_return'] += actual_return
        
        # Пересчет метрик
        stats['accuracy'] = stats['correct_predictions'] / stats['total_predictions']
        stats['avg_confidence'] = stats['total_confidence'] / stats['total_predictions']
        stats['avg_return'] = stats['total_return'] / stats['total_predictions']
        
        # Комплексная оценка производительности
        stats['performance_score'] = (
            stats['accuracy'] * 0.4 +
            min(stats['avg_confidence'], 1.0) * 0.2 +
            max(0, min(stats['avg_return'] * 10, 1.0)) * 0.4
        )
        
    def get_model_performance_score(self, model_name: str) -> float:
        """Получение оценки производительности модели"""
        if model_name not in self.model_stats:
            return 0.5  # Нейтральная оценка для новых моделей
        return self.model_stats[model_name]['performance_score']
    
    def get_top_performing_models(self, min_predictions: int = 5) -> List[str]:
        """Получение списка лучших моделей"""
        qualified_models = [
            (name, stats['performance_score']) 
            for name, stats in self.model_stats.items()
            if stats['total_predictions'] >= min_predictions
        ]
        qualified_models.sort(key=lambda x: x[1], reverse=True)
        return [name for name, _ in qualified_models]

class MarketRegimeDetector:
    """Детектор режима рынка"""
    
    def detect_market_regime(self, data: pd.DataFrame) -> str:
        """Определение текущего режима рынка"""
        if len(data) < 50:
            return "UNKNOWN"
        
        # Анализ трендов
        short_ma = data['close'].rolling(10).mean().iloc[-1]
        long_ma = data['close'].rolling(50).mean().iloc[-1]
        current_price = data['close'].iloc[-1]
        
        # Анализ волатильности
        volatility = data['close'].pct_change().rolling(20).std().iloc[-1]
        avg_volatility = data['close'].pct_change().rolling(50).std().mean()
        
        # Определение режима
        if current_price > short_ma > long_ma and volatility < avg_volatility * 1.2:
            return "BULL"
        elif current_price < short_ma < long_ma and volatility < avg_volatility * 1.2:
            return "BEAR"
        else:
            return "SIDEWAYS"
    
    def calculate_market_conditions_score(self, data: pd.DataFrame) -> float:
        """Оценка качества рыночных условий для торговли"""
        if len(data) < 20:
            return 0.5
        
        # Факторы качества рынка
        volatility = data['close'].pct_change().rolling(20).std().iloc[-1]
        volume_trend = data['volume'].rolling(10).mean().iloc[-1] / data['volume'].rolling(30).mean().iloc[-1]
        price_momentum = (data['close'].iloc[-1] / data['close'].iloc[-10] - 1)
        
        # Нормализация и комбинирование
        vol_score = min(volatility * 100, 1.0)  # Волатильность как возможность
        volume_score = min(volume_trend, 2.0) / 2.0  # Объем как подтверждение
        momentum_score = min(abs(price_momentum) * 10, 1.0)  # Моментум как сила
        
        return (vol_score * 0.3 + volume_score * 0.4 + momentum_score * 0.3)

class EnhancedSignalProcessor:
    """Улучшенный процессор сигналов с качественной фильтрацией"""
    
    def __init__(self, config: EnhancedTestConfig):
        self.config = config
        self.performance_tracker = ModelPerformanceTracker()
        self.market_detector = MarketRegimeDetector()
    
    def calculate_signal_quality_score(self, decisions: List[EnhancedAIDecision], 
                                     market_data: pd.DataFrame) -> float:
        """Расчет качества сигнала"""
        if not decisions:
            return 0.0
        
        # Факторы качества
        confidence_factor = np.mean([d.confidence for d in decisions])
        consensus_factor = len([d for d in decisions if d.action != 'HOLD']) / len(decisions)
        performance_factor = np.mean([d.model_performance_score for d in decisions])
        
        # Рыночные условия
        market_score = self.market_detector.calculate_market_conditions_score(market_data)
        
        # Комбинированная оценка
        quality_score = (
            confidence_factor * 0.25 +
            consensus_factor * 0.25 +
            performance_factor * 0.25 +
            market_score * 0.25
        )
        
        return min(quality_score, 1.0)
    
    def calculate_weighted_confidence(self, decisions: List[EnhancedAIDecision]) -> float:
        """Расчет взвешенной уверенности по производительности моделей"""
        if not decisions:
            return 0.0
        
        total_weight = 0.0
        weighted_confidence = 0.0
        
        for decision in decisions:
            weight = decision.model_performance_score
            weighted_confidence += decision.confidence * weight
            total_weight += weight
        
        return weighted_confidence / total_weight if total_weight > 0 else 0.0
    
    def calculate_risk_reward_ratio(self, signal: EnhancedConsensusSignal, 
                                  market_data: pd.DataFrame) -> float:
        """Расчет соотношения риск/прибыль"""
        volatility = market_data['close'].pct_change().rolling(20).std().iloc[-1]
        
        # Адаптивные стоп-лосс и тейк-профит
        stop_loss = self.config.base_stop_loss_percent * (1 + volatility * self.config.volatility_multiplier)
        take_profit = self.config.base_take_profit_percent * (1 + volatility * self.config.volatility_multiplier)
        
        return take_profit / stop_loss if stop_loss > 0 else 0.0

class EnhancedWinrateTester:
    """Улучшенный тестер винрейта с комплексными улучшениями"""
    
    def __init__(self, config: EnhancedTestConfig):
        self.config = config
        self.signal_processor = EnhancedSignalProcessor(config)
        self.ai_orchestrator = None
        self.data_manager = HistoricalDataManager()
        self.data_collector = BinanceDataCollector()
        
        # Статистика
        self.daily_trades = {}
        self.daily_pnl = {}
        self.symbol_exposure = {}
        self.total_drawdown = 0.0
        
    async def initialize(self):
        """Инициализация улучшенной системы"""
        logger.info("🚀 Инициализация улучшенной AI торговой системы...")
        
        # Инициализация AI оркестратора
        self.ai_orchestrator = MultiAIOrchestrator()
        await self.ai_orchestrator.initialize()
        
        logger.info("✅ Улучшенная система готова к работе")
    
    async def get_enhanced_ai_signals(self, symbol: str, data: pd.DataFrame) -> List[EnhancedConsensusSignal]:
        """Получение улучшенных AI сигналов"""
        signals = []
        
        # Получение решений от всех моделей
        model_decisions = []
        for model_name in ['trading_ai', 'lava_ai', 'lgbm_ai', 'mistral_ai']:
            try:
                decision = await self.get_enhanced_individual_signal(model_name, symbol, data)
                if decision:
                    model_decisions.append(decision)
            except Exception as e:
                logger.warning(f"Ошибка получения сигнала от {model_name}: {e}")
        
        if len(model_decisions) >= self.config.min_consensus_models:
            consensus_signal = await self.create_enhanced_consensus(symbol, data, model_decisions)
            if consensus_signal:
                signals.append(consensus_signal)
        
        return signals
    
    async def get_enhanced_individual_signal(self, model_name: str, symbol: str, 
                                           data: pd.DataFrame) -> Optional[EnhancedAIDecision]:
        """Получение улучшенного индивидуального сигнала"""
        try:
            # Получение базового сигнала
            if model_name == 'lgbm_ai':
                ai_module = LGBMAI()
                result = await ai_module.analyze_trading_opportunity(symbol, data)
            elif model_name == 'mistral_ai':
                ai_module = MistralAI()
                result = await ai_module.analyze_trading_opportunity(symbol, data)
            elif model_name == 'lava_ai':
                ai_module = LavaAI()
                result = await ai_module.analyze_trading_opportunity(symbol, data)
            elif model_name == 'trading_ai':
                ai_module = TradingAI()
                result = await ai_module.analyze_trading_opportunity(symbol, data)
            else:
                return None
            
            if not result:
                return None
            
            # Создание улучшенного решения
            decision = EnhancedAIDecision(
                model_name=model_name,
                action=self._convert_to_action(result.get('direction', 0)),
                confidence=result.get('confidence', 0.0),
                reasoning=result.get('reasoning', ''),
                timestamp=get_utc_now()
            )
            
            # Дополнительные метрики
            decision.signal_strength = self._calculate_signal_strength(result, data)
            decision.market_regime = self.signal_processor.market_detector.detect_market_regime(data)
            decision.risk_score = self._calculate_risk_score(result, data)
            decision.model_performance_score = self.signal_processor.performance_tracker.get_model_performance_score(model_name)
            
            return decision
            
        except Exception as e:
            logger.error(f"Ошибка получения сигнала от {model_name}: {e}")
            return None
    
    async def create_enhanced_consensus(self, symbol: str, data: pd.DataFrame, 
                                      decisions: List[EnhancedAIDecision]) -> Optional[EnhancedConsensusSignal]:
        """Создание улучшенного консенсус сигнала"""
        if not decisions:
            return None
        
        # Фильтрация по производительности моделей
        qualified_decisions = [
            d for d in decisions 
            if d.model_performance_score >= self.config.min_model_performance_score
        ]
        
        if len(qualified_decisions) < self.config.min_consensus_models:
            return None
        
        # Определение консенсуса
        buy_votes = len([d for d in qualified_decisions if d.action == 'BUY'])
        sell_votes = len([d for d in qualified_decisions if d.action == 'SELL'])
        
        if buy_votes > sell_votes:
            final_action = 'BUY'
            consensus_strength = buy_votes
        elif sell_votes > buy_votes:
            final_action = 'SELL'
            consensus_strength = sell_votes
        else:
            return None  # Нет консенсуса
        
        # Создание консенсус сигнала
        consensus_signal = EnhancedConsensusSignal(
            symbol=symbol,
            timestamp=get_utc_now(),
            price=data['close'].iloc[-1],
            final_action=final_action,
            consensus_strength=consensus_strength,
            participating_models=qualified_decisions,
            confidence_avg=np.mean([d.confidence for d in qualified_decisions])
        )
        
        # Расчет дополнительных метрик
        consensus_signal.weighted_confidence = self.signal_processor.calculate_weighted_confidence(qualified_decisions)
        consensus_signal.signal_quality_score = self.signal_processor.calculate_signal_quality_score(qualified_decisions, data)
        consensus_signal.market_conditions_score = self.signal_processor.market_detector.calculate_market_conditions_score(data)
        consensus_signal.risk_reward_ratio = self.signal_processor.calculate_risk_reward_ratio(consensus_signal, data)
        
        # Расчет ожидаемой вероятности прибыли
        consensus_signal.expected_profit_probability = self._calculate_profit_probability(consensus_signal, data)
        
        # Проверка качественных фильтров
        if (consensus_signal.signal_quality_score >= self.config.min_signal_quality_score and
            consensus_signal.weighted_confidence >= self.config.min_weighted_confidence and
            consensus_signal.market_conditions_score >= self.config.min_market_conditions_score):
            return consensus_signal
        
        return None
    
    def _convert_to_action(self, direction: float) -> str:
        """Конвертация направления в действие"""
        if direction > 0.5:
            return 'BUY'
        elif direction < -0.5:
            return 'SELL'
        else:
            return 'HOLD'
    
    def _calculate_signal_strength(self, result: dict, data: pd.DataFrame) -> float:
        """Расчет силы сигнала"""
        confidence = result.get('confidence', 0.0)
        direction = abs(result.get('direction', 0.0))
        
        # Технические индикаторы для подтверждения
        rsi = self._calculate_rsi(data)
        volume_ratio = data['volume'].iloc[-1] / data['volume'].rolling(20).mean().iloc[-1]
        
        # Комбинированная сила сигнала
        signal_strength = (
            confidence * 0.4 +
            direction * 0.3 +
            min(volume_ratio / 2.0, 1.0) * 0.3
        )
        
        return min(signal_strength, 1.0)
    
    def _calculate_risk_score(self, result: dict, data: pd.DataFrame) -> float:
        """Расчет оценки риска"""
        volatility = data['close'].pct_change().rolling(20).std().iloc[-1]
        volume_volatility = data['volume'].pct_change().rolling(20).std().iloc[-1]
        
        # Нормализация рисков
        vol_risk = min(volatility * 100, 1.0)
        vol_volume_risk = min(volume_volatility * 10, 1.0)
        
        return (vol_risk + vol_volume_risk) / 2.0
    
    def _calculate_profit_probability(self, signal: EnhancedConsensusSignal, data: pd.DataFrame) -> float:
        """Расчет ожидаемой вероятности прибыли"""
        # Базовая вероятность на основе взвешенной уверенности
        base_probability = signal.weighted_confidence
        
        # Корректировка на основе рыночных условий
        market_adjustment = signal.market_conditions_score * 0.2
        
        # Корректировка на основе качества сигнала
        quality_adjustment = signal.signal_quality_score * 0.2
        
        # Корректировка на основе соотношения риск/прибыль
        rr_adjustment = min(signal.risk_reward_ratio / 3.0, 0.2)
        
        total_probability = base_probability + market_adjustment + quality_adjustment + rr_adjustment
        return min(total_probability, 0.95)  # Максимум 95%
    
    def _calculate_rsi(self, data: pd.DataFrame, period: int = 14) -> float:
        """Расчет RSI"""
        if len(data) < period + 1:
            return 50.0
        
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50.0

# Функция для запуска улучшенного тестирования
async def run_enhanced_test():
    """Запуск улучшенного тестирования"""
    config = EnhancedTestConfig()
    tester = EnhancedWinrateTester(config)
    
    await tester.initialize()
    
    logger.info("🚀 Запуск улучшенного тестирования винрейта...")
    logger.info(f"📊 Конфигурация: {config.test_period_days} дней, {len(config.symbols)} символов")
    logger.info(f"🎯 Цель: достижение 60%+ винрейта с положительным ROI")
    
    # Здесь будет основная логика тестирования
    # (аналогично существующей, но с улучшенными алгоритмами)
    
    return "Улучшенная система готова к тестированию"

if __name__ == "__main__":
    asyncio.run(run_enhanced_test())