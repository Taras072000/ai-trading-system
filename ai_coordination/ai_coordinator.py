"""
Координатор AI модулей для Peper Binance v4
Архитектурные улучшения - Фаза 2
"""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np
from enum import Enum
import threading
from collections import deque

from config.unified_config_manager import get_config_manager

logger = logging.getLogger(__name__)

class SignalType(Enum):
    """Типы торговых сигналов"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"

class ConflictResolutionStrategy(Enum):
    """Стратегии разрешения конфликтов"""
    WEIGHTED_AVERAGE = "weighted_average"
    MAJORITY_VOTE = "majority_vote"
    CONFIDENCE_BASED = "confidence_based"

@dataclass
class AISignal:
    """Сигнал от AI модуля"""
    module_name: str
    signal_type: SignalType
    confidence: float
    timestamp: datetime
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class ConsolidatedSignal:
    """Консолидированный сигнал"""
    signal_type: SignalType
    confidence: float
    contributing_modules: List[str]
    weights_used: Dict[str, float]
    consensus_level: float
    timestamp: datetime
    metadata: Dict[str, Any] = None

class PerformanceTracker:
    """Трекер производительности AI модулей"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.performance_history: Dict[str, deque] = {}
        self.trade_outcomes: Dict[str, deque] = {}
        self.lock = threading.Lock()
    
    def record_performance(self, module_name: str, performance: float) -> None:
        """Запись производительности модуля"""
        with self.lock:
            if module_name not in self.performance_history:
                self.performance_history[module_name] = deque(maxlen=self.window_size)
            
            self.performance_history[module_name].append(performance)
    
    def record_trade_outcome(self, module_name: str, profit_loss: float, was_correct: bool) -> None:
        """Запись результата сделки"""
        with self.lock:
            if module_name not in self.trade_outcomes:
                self.trade_outcomes[module_name] = deque(maxlen=self.window_size)
            
            self.trade_outcomes[module_name].append({
                'profit_loss': profit_loss,
                'was_correct': was_correct,
                'timestamp': datetime.now()
            })
    
    def get_recent_performance(self, module_name: str, lookback: int = 20) -> float:
        """Получение недавней производительности"""
        with self.lock:
            if module_name not in self.performance_history:
                return 0.5  # Нейтральная производительность
            
            history = list(self.performance_history[module_name])
            if not history:
                return 0.5
            
            recent = history[-lookback:] if len(history) >= lookback else history
            return sum(recent) / len(recent)
    
    def get_win_rate(self, module_name: str, lookback: int = 50) -> float:
        """Получение винрейта модуля"""
        with self.lock:
            if module_name not in self.trade_outcomes:
                return 0.5
            
            outcomes = list(self.trade_outcomes[module_name])
            if not outcomes:
                return 0.5
            
            recent = outcomes[-lookback:] if len(outcomes) >= lookback else outcomes
            wins = sum(1 for outcome in recent if outcome['was_correct'])
            return wins / len(recent) if recent else 0.5
    
    def get_average_profit(self, module_name: str, lookback: int = 50) -> float:
        """Получение средней прибыли модуля"""
        with self.lock:
            if module_name not in self.trade_outcomes:
                return 0.0
            
            outcomes = list(self.trade_outcomes[module_name])
            if not outcomes:
                return 0.0
            
            recent = outcomes[-lookback:] if len(outcomes) >= lookback else outcomes
            profits = [outcome['profit_loss'] for outcome in recent]
            return sum(profits) / len(profits) if profits else 0.0

class AICoordinator:
    """Координатор AI модулей"""
    
    def __init__(self):
        self.config_manager = get_config_manager()
        self.performance_tracker = PerformanceTracker()
        
        # Загрузка конфигурации координации
        self.coordination_config = self.config_manager.get_system_config('ai_coordination')
        
        # Настройки
        self.weighting_algorithm = self.coordination_config.get('weighting_algorithm', 'adaptive_performance')
        self.conflict_threshold = self.coordination_config.get('conflict_resolution', {}).get('threshold', 0.3)
        self.min_agreement = self.coordination_config.get('consensus', {}).get('min_agreement', 0.6)
        self.confidence_boost = self.coordination_config.get('consensus', {}).get('confidence_boost', 0.1)
        
        # Стратегия разрешения конфликтов
        strategy_name = self.coordination_config.get('conflict_resolution', {}).get('strategy', 'weighted_average')
        self.conflict_strategy = ConflictResolutionStrategy(strategy_name)
        
        # История сигналов
        self.signal_history: List[ConsolidatedSignal] = []
        self.max_history_size = 1000
        
        # Блокировка для потокобезопасности
        self.lock = threading.Lock()
        
        logger.info("AI Координатор инициализирован")
    
    async def process_signals(self, signals: List[AISignal]) -> Optional[ConsolidatedSignal]:
        """Обработка сигналов от AI модулей"""
        try:
            if not signals:
                return None
            
            # Фильтрация активных модулей
            active_signals = [
                signal for signal in signals 
                if self.config_manager.is_module_enabled(signal.module_name)
            ]
            
            if not active_signals:
                return None
            
            # Получение текущих весов
            current_weights = self.config_manager.get_current_weights()
            
            # Обнаружение конфликтов
            has_conflict = self._detect_conflicts(active_signals)
            
            if has_conflict:
                logger.info("Обнаружен конфликт сигналов, применяется стратегия разрешения")
                consolidated = await self._resolve_conflicts(active_signals, current_weights)
            else:
                consolidated = await self._consolidate_signals(active_signals, current_weights)
            
            # Проверка консенсуса
            if consolidated and consolidated.consensus_level >= self.min_agreement:
                # Бонус к уверенности при консенсусе
                consolidated.confidence = min(1.0, consolidated.confidence + self.confidence_boost)
                
                # Сохранение в историю
                with self.lock:
                    self.signal_history.append(consolidated)
                    if len(self.signal_history) > self.max_history_size:
                        self.signal_history = self.signal_history[-self.max_history_size:]
                
                logger.info(f"Консолидированный сигнал: {consolidated.signal_type.value}, "
                           f"уверенность: {consolidated.confidence:.3f}, "
                           f"консенсус: {consolidated.consensus_level:.3f}")
                
                return consolidated
            
            logger.info("Недостаточный консенсус для генерации сигнала")
            return None
            
        except Exception as e:
            logger.error(f"Ошибка обработки сигналов: {e}")
            return None
    
    def _detect_conflicts(self, signals: List[AISignal]) -> bool:
        """Обнаружение конфликтов между сигналами"""
        if len(signals) < 2:
            return False
        
        # Группировка по типам сигналов
        signal_groups = {}
        for signal in signals:
            signal_type = signal.signal_type
            if signal_type not in signal_groups:
                signal_groups[signal_type] = []
            signal_groups[signal_type].append(signal)
        
        # Конфликт если есть противоположные сигналы
        has_buy = SignalType.BUY in signal_groups
        has_sell = SignalType.SELL in signal_groups
        
        if has_buy and has_sell:
            # Проверка силы конфликтующих сигналов
            buy_strength = sum(s.confidence for s in signal_groups[SignalType.BUY])
            sell_strength = sum(s.confidence for s in signal_groups[SignalType.SELL])
            
            # Конфликт если разница меньше порога
            strength_diff = abs(buy_strength - sell_strength) / max(buy_strength + sell_strength, 1e-6)
            return strength_diff < self.conflict_threshold
        
        return False
    
    async def _resolve_conflicts(self, signals: List[AISignal], weights: Dict[str, float]) -> Optional[ConsolidatedSignal]:
        """Разрешение конфликтов между сигналами"""
        if self.conflict_strategy == ConflictResolutionStrategy.WEIGHTED_AVERAGE:
            return await self._weighted_average_resolution(signals, weights)
        elif self.conflict_strategy == ConflictResolutionStrategy.MAJORITY_VOTE:
            return await self._majority_vote_resolution(signals, weights)
        elif self.conflict_strategy == ConflictResolutionStrategy.CONFIDENCE_BASED:
            return await self._confidence_based_resolution(signals, weights)
        else:
            return await self._weighted_average_resolution(signals, weights)
    
    async def _weighted_average_resolution(self, signals: List[AISignal], weights: Dict[str, float]) -> Optional[ConsolidatedSignal]:
        """Разрешение конфликтов через взвешенное усреднение"""
        # Преобразование сигналов в числовые значения
        signal_values = []
        signal_weights = []
        contributing_modules = []
        
        for signal in signals:
            module_weight = weights.get(signal.module_name, 0.0)
            if module_weight > 0:
                # BUY = 1, HOLD = 0, SELL = -1
                if signal.signal_type == SignalType.BUY:
                    value = 1.0
                elif signal.signal_type == SignalType.SELL:
                    value = -1.0
                else:
                    value = 0.0
                
                # Взвешивание по уверенности и весу модуля
                weighted_value = value * signal.confidence * module_weight
                signal_values.append(weighted_value)
                signal_weights.append(module_weight * signal.confidence)
                contributing_modules.append(signal.module_name)
        
        if not signal_values:
            return None
        
        # Вычисление взвешенного среднего
        total_weight = sum(signal_weights)
        if total_weight == 0:
            return None
        
        weighted_average = sum(signal_values) / total_weight
        
        # Определение итогового сигнала
        if weighted_average > 0.1:
            final_signal = SignalType.BUY
        elif weighted_average < -0.1:
            final_signal = SignalType.SELL
        else:
            final_signal = SignalType.HOLD
        
        # Вычисление уверенности и консенсуса
        confidence = min(1.0, abs(weighted_average))
        consensus_level = self._calculate_consensus(signals, final_signal)
        
        # Нормализация весов
        weights_used = {}
        for i, module in enumerate(contributing_modules):
            weights_used[module] = signal_weights[i] / total_weight
        
        return ConsolidatedSignal(
            signal_type=final_signal,
            confidence=confidence,
            contributing_modules=contributing_modules,
            weights_used=weights_used,
            consensus_level=consensus_level,
            timestamp=datetime.now(),
            metadata={'resolution_method': 'weighted_average'}
        )
    
    async def _majority_vote_resolution(self, signals: List[AISignal], weights: Dict[str, float]) -> Optional[ConsolidatedSignal]:
        """Разрешение конфликтов через голосование большинства"""
        # Подсчет голосов с учетом весов
        votes = {SignalType.BUY: 0, SignalType.SELL: 0, SignalType.HOLD: 0}
        contributing_modules = []
        weights_used = {}
        
        for signal in signals:
            module_weight = weights.get(signal.module_name, 0.0)
            if module_weight > 0:
                vote_weight = module_weight * signal.confidence
                votes[signal.signal_type] += vote_weight
                contributing_modules.append(signal.module_name)
                weights_used[signal.module_name] = vote_weight
        
        if not contributing_modules:
            return None
        
        # Определение победителя
        winning_signal = max(votes, key=votes.get)
        winning_votes = votes[winning_signal]
        total_votes = sum(votes.values())
        
        if total_votes == 0:
            return None
        
        # Нормализация весов
        total_weight = sum(weights_used.values())
        if total_weight > 0:
            for module in weights_used:
                weights_used[module] /= total_weight
        
        confidence = winning_votes / total_votes
        consensus_level = self._calculate_consensus(signals, winning_signal)
        
        return ConsolidatedSignal(
            signal_type=winning_signal,
            confidence=confidence,
            contributing_modules=contributing_modules,
            weights_used=weights_used,
            consensus_level=consensus_level,
            timestamp=datetime.now(),
            metadata={'resolution_method': 'majority_vote'}
        )
    
    async def _confidence_based_resolution(self, signals: List[AISignal], weights: Dict[str, float]) -> Optional[ConsolidatedSignal]:
        """Разрешение конфликтов на основе уверенности"""
        # Поиск сигнала с максимальной взвешенной уверенностью
        best_signal = None
        best_score = 0
        
        for signal in signals:
            module_weight = weights.get(signal.module_name, 0.0)
            score = signal.confidence * module_weight
            
            if score > best_score:
                best_score = score
                best_signal = signal
        
        if not best_signal:
            return None
        
        # Поиск поддерживающих модулей
        contributing_modules = [best_signal.module_name]
        weights_used = {best_signal.module_name: weights.get(best_signal.module_name, 0.0)}
        
        for signal in signals:
            if signal != best_signal and signal.signal_type == best_signal.signal_type:
                contributing_modules.append(signal.module_name)
                weights_used[signal.module_name] = weights.get(signal.module_name, 0.0)
        
        # Нормализация весов
        total_weight = sum(weights_used.values())
        if total_weight > 0:
            for module in weights_used:
                weights_used[module] /= total_weight
        
        consensus_level = self._calculate_consensus(signals, best_signal.signal_type)
        
        return ConsolidatedSignal(
            signal_type=best_signal.signal_type,
            confidence=best_signal.confidence,
            contributing_modules=contributing_modules,
            weights_used=weights_used,
            consensus_level=consensus_level,
            timestamp=datetime.now(),
            metadata={'resolution_method': 'confidence_based'}
        )
    
    async def _consolidate_signals(self, signals: List[AISignal], weights: Dict[str, float]) -> Optional[ConsolidatedSignal]:
        """Консолидация сигналов без конфликтов"""
        # Группировка по типам сигналов
        signal_groups = {}
        for signal in signals:
            signal_type = signal.signal_type
            if signal_type not in signal_groups:
                signal_groups[signal_type] = []
            signal_groups[signal_type].append(signal)
        
        # Поиск доминирующего сигнала
        best_group = None
        best_weight = 0
        
        for signal_type, group_signals in signal_groups.items():
            total_weight = sum(
                weights.get(s.module_name, 0.0) * s.confidence 
                for s in group_signals
            )
            
            if total_weight > best_weight:
                best_weight = total_weight
                best_group = (signal_type, group_signals)
        
        if not best_group:
            return None
        
        signal_type, group_signals = best_group
        
        # Вычисление консолидированной уверенности
        total_confidence = 0
        total_weight = 0
        contributing_modules = []
        weights_used = {}
        
        for signal in group_signals:
            module_weight = weights.get(signal.module_name, 0.0)
            weighted_confidence = signal.confidence * module_weight
            
            total_confidence += weighted_confidence
            total_weight += module_weight
            contributing_modules.append(signal.module_name)
            weights_used[signal.module_name] = module_weight
        
        if total_weight == 0:
            return None
        
        # Нормализация весов
        for module in weights_used:
            weights_used[module] /= total_weight
        
        final_confidence = total_confidence / total_weight
        consensus_level = len(group_signals) / len(signals)
        
        return ConsolidatedSignal(
            signal_type=signal_type,
            confidence=final_confidence,
            contributing_modules=contributing_modules,
            weights_used=weights_used,
            consensus_level=consensus_level,
            timestamp=datetime.now(),
            metadata={'resolution_method': 'consolidation'}
        )
    
    def _calculate_consensus(self, signals: List[AISignal], target_signal: SignalType) -> float:
        """Вычисление уровня консенсуса"""
        if not signals:
            return 0.0
        
        agreeing_signals = [s for s in signals if s.signal_type == target_signal]
        return len(agreeing_signals) / len(signals)
    
    def update_performance(self, module_name: str, performance: float) -> None:
        """Обновление производительности модуля"""
        self.performance_tracker.record_performance(module_name, performance)
        
        # Обновление адаптивных весов
        performance_data = {}
        for module in self.config_manager.adaptive_weights.keys():
            performance_data[module] = self.performance_tracker.get_recent_performance(module)
        
        self.config_manager.update_adaptive_weights(performance_data)
    
    def record_trade_outcome(self, signal: ConsolidatedSignal, profit_loss: float, was_correct: bool) -> None:
        """Запись результата сделки"""
        for module_name in signal.contributing_modules:
            self.performance_tracker.record_trade_outcome(module_name, profit_loss, was_correct)
    
    def get_module_statistics(self, module_name: str) -> Dict[str, float]:
        """Получение статистики модуля"""
        return {
            'recent_performance': self.performance_tracker.get_recent_performance(module_name),
            'win_rate': self.performance_tracker.get_win_rate(module_name),
            'average_profit': self.performance_tracker.get_average_profit(module_name),
            'current_weight': self.config_manager.adaptive_weights.get(module_name, 0.0)
        }
    
    def get_coordination_status(self) -> Dict[str, Any]:
        """Получение статуса координации"""
        return {
            'active_modules': [
                name for name, config in self.config_manager.ai_modules.items() 
                if config.enabled
            ],
            'current_weights': self.config_manager.get_current_weights(),
            'weighting_algorithm': self.weighting_algorithm,
            'conflict_strategy': self.conflict_strategy.value,
            'recent_signals_count': len(self.signal_history),
            'last_weight_update': self.config_manager.last_weight_update.isoformat() if self.config_manager.last_weight_update else None
        }