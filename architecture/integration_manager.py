"""
Менеджер интеграции архитектурных улучшений для Peper Binance v4
Архитектурные улучшения - Фаза 2
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from config.unified_config_manager import get_config_manager
from ai_coordination.ai_coordinator import AICoordinator, AISignal, SignalType
from market_analysis.market_phase_detector import MarketPhaseDetector, MarketPhase
from risk_management.advanced_risk_manager import AdvancedRiskManager, Position, PositionType
from performance.performance_optimizer import get_performance_optimizer
from monitoring.architecture_monitor import get_architecture_monitor

logger = logging.getLogger(__name__)

class SystemState(Enum):
    """Состояния системы"""
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"
    SHUTDOWN = "shutdown"

@dataclass
class TradingDecision:
    """Торговое решение"""
    symbol: str
    action: str  # "BUY", "SELL", "HOLD"
    quantity: float
    price: Optional[float]
    stop_loss: Optional[float]
    take_profit: Optional[float]
    confidence: float
    reasoning: str
    timestamp: datetime
    market_phase: MarketPhase
    risk_level: str
    ai_signals: List[AISignal]

@dataclass
class SystemMetrics:
    """Метрики системы"""
    total_decisions: int
    successful_decisions: int
    failed_decisions: int
    avg_confidence: float
    avg_processing_time: float
    memory_usage: float
    cpu_usage: float
    cache_hit_ratio: float
    timestamp: datetime

class IntegrationManager:
    """Менеджер интеграции всех архитектурных компонентов"""
    
    def __init__(self):
        # Конфигурация
        self.config_manager = get_config_manager()
        
        # Основные компоненты
        self.ai_coordinator: Optional[AICoordinator] = None
        self.market_detector: Optional[MarketPhaseDetector] = None
        self.risk_manager: Optional[AdvancedRiskManager] = None
        self.performance_optimizer = get_performance_optimizer()
        self.architecture_monitor = get_architecture_monitor()
        
        # Состояние системы
        self.state = SystemState.INITIALIZING
        self.last_decision_time: Optional[datetime] = None
        self.decision_history: List[TradingDecision] = []
        
        # Метрики
        self.metrics = SystemMetrics(
            total_decisions=0,
            successful_decisions=0,
            failed_decisions=0,
            avg_confidence=0.0,
            avg_processing_time=0.0,
            memory_usage=0.0,
            cpu_usage=0.0,
            cache_hit_ratio=0.0,
            timestamp=datetime.now()
        )
        
        # Настройки
        self.min_confidence_threshold = 0.6
        self.max_processing_time = 5.0  # секунд
        
        logger.info("Менеджер интеграции инициализирован")
    
    async def initialize(self) -> bool:
        """Инициализация всех компонентов"""
        try:
            logger.info("Начало инициализации архитектурных компонентов...")
            
            # Инициализация AI координатора
            self.ai_coordinator = AICoordinator()
            await self.ai_coordinator.initialize()
            logger.info("AI координатор инициализирован")
            
            # Инициализация детектора рыночных фаз
            self.market_detector = MarketPhaseDetector()
            await self.market_detector.initialize()
            logger.info("Детектор рыночных фаз инициализирован")
            
            # Инициализация риск-менеджера
            self.risk_manager = AdvancedRiskManager()
            await self.risk_manager.initialize()
            logger.info("Риск-менеджер инициализирован")
            
            # Запуск оптимизатора производительности
            await self.performance_optimizer.start()
            logger.info("Оптимизатор производительности запущен")
            
            # Настройка мониторинга
            self.architecture_monitor.set_components(
                self.ai_coordinator,
                self.market_detector,
                self.risk_manager
            )
            await self.architecture_monitor.start_monitoring()
            logger.info("Мониторинг архитектуры запущен")
            
            # Проверка готовности всех компонентов
            if await self._verify_components():
                self.state = SystemState.RUNNING
                logger.info("Все архитектурные компоненты успешно инициализированы")
                return True
            else:
                self.state = SystemState.ERROR
                logger.error("Ошибка при проверке компонентов")
                return False
                
        except Exception as e:
            logger.error(f"Ошибка инициализации: {e}")
            self.state = SystemState.ERROR
            return False
    
    async def _verify_components(self) -> bool:
        """Проверка готовности всех компонентов"""
        try:
            # Проверка AI координатора
            if not self.ai_coordinator or not self.ai_coordinator.is_initialized:
                logger.error("AI координатор не готов")
                return False
            
            # Проверка детектора рыночных фаз
            if not self.market_detector:
                logger.error("Детектор рыночных фаз не готов")
                return False
            
            # Проверка риск-менеджера
            if not self.risk_manager:
                logger.error("Риск-менеджер не готов")
                return False
            
            # Проверка конфигурации
            config = self.config_manager.get_config()
            if not config:
                logger.error("Конфигурация не загружена")
                return False
            
            logger.info("Все компоненты прошли проверку готовности")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка проверки компонентов: {e}")
            return False
    
    async def process_market_data(self, market_data: Dict[str, Any]) -> Optional[TradingDecision]:
        """Обработка рыночных данных и принятие торгового решения"""
        if self.state != SystemState.RUNNING:
            logger.warning(f"Система не готова к обработке данных. Состояние: {self.state}")
            return None
        
        start_time = time.time()
        
        try:
            # 1. Анализ рыночной фазы
            market_condition = await self.market_detector.analyze_market_data(market_data)
            if not market_condition:
                logger.warning("Не удалось определить рыночную фазу")
                return None
            
            logger.debug(f"Рыночная фаза: {market_condition.primary_phase}, "
                        f"уверенность: {market_condition.confidence:.2f}")
            
            # 2. Получение сигналов от AI модулей
            ai_signals = await self.ai_coordinator.process_market_data(market_data)
            if not ai_signals:
                logger.debug("AI модули не сгенерировали сигналы")
                return None
            
            # 3. Консолидация сигналов
            consolidated_signal = await self.ai_coordinator.consolidate_signals(ai_signals)
            if not consolidated_signal:
                logger.debug("Не удалось консолидировать сигналы")
                return None
            
            logger.debug(f"Консолидированный сигнал: {consolidated_signal.signal_type}, "
                        f"уверенность: {consolidated_signal.confidence:.2f}")
            
            # 4. Проверка минимального порога уверенности
            if consolidated_signal.confidence < self.min_confidence_threshold:
                logger.debug(f"Уверенность сигнала слишком низкая: {consolidated_signal.confidence:.2f}")
                return None
            
            # 5. Адаптация к рыночной фазе
            adapted_signal = await self._adapt_signal_to_market_phase(
                consolidated_signal, market_condition
            )
            
            # 6. Риск-анализ и валидация
            risk_assessment = await self.risk_manager.assess_trade_risk(
                symbol=market_data.get('symbol', 'BTCUSDT'),
                signal_type=adapted_signal.signal_type,
                market_condition=market_condition,
                current_price=market_data.get('price', 0)
            )
            
            if not risk_assessment['approved']:
                logger.info(f"Сделка отклонена риск-менеджером: {risk_assessment['reason']}")
                return None
            
            # 7. Формирование торгового решения
            decision = await self._create_trading_decision(
                market_data=market_data,
                signal=adapted_signal,
                market_condition=market_condition,
                risk_assessment=risk_assessment,
                ai_signals=ai_signals
            )
            
            # 8. Обновление метрик
            processing_time = time.time() - start_time
            await self._update_metrics(decision, processing_time)
            
            # 9. Сохранение в историю
            self.decision_history.append(decision)
            self.last_decision_time = datetime.now()
            
            logger.info(f"Торговое решение принято: {decision.action} {decision.symbol} "
                       f"(уверенность: {decision.confidence:.2f}, время: {processing_time:.2f}с)")
            
            return decision
            
        except Exception as e:
            logger.error(f"Ошибка обработки рыночных данных: {e}")
            self.metrics.failed_decisions += 1
            return None
    
    async def _adapt_signal_to_market_phase(self, signal: Any, market_condition: Any) -> Any:
        """Адаптация сигнала к рыночной фазе"""
        try:
            # Получение адаптивных параметров для текущей фазы
            adaptive_params = self.market_detector.get_adaptive_parameters(
                market_condition.primary_phase
            )
            
            # Корректировка уверенности на основе рыночной фазы
            phase_confidence_multiplier = adaptive_params.get('confidence_multiplier', 1.0)
            adapted_confidence = signal.confidence * phase_confidence_multiplier
            
            # Ограничение уверенности
            adapted_confidence = max(0.0, min(1.0, adapted_confidence))
            
            # Создание адаптированного сигнала
            adapted_signal = type(signal)(
                signal_type=signal.signal_type,
                confidence=adapted_confidence,
                price=signal.price,
                timestamp=signal.timestamp,
                metadata={
                    **signal.metadata,
                    'market_phase': market_condition.primary_phase.value,
                    'phase_confidence': market_condition.confidence,
                    'adapted': True,
                    'original_confidence': signal.confidence
                }
            )
            
            logger.debug(f"Сигнал адаптирован к фазе {market_condition.primary_phase}: "
                        f"{signal.confidence:.2f} -> {adapted_confidence:.2f}")
            
            return adapted_signal
            
        except Exception as e:
            logger.error(f"Ошибка адаптации сигнала: {e}")
            return signal
    
    async def _create_trading_decision(self, market_data: Dict[str, Any], 
                                     signal: Any, market_condition: Any,
                                     risk_assessment: Dict[str, Any],
                                     ai_signals: List[AISignal]) -> TradingDecision:
        """Создание торгового решения"""
        try:
            symbol = market_data.get('symbol', 'BTCUSDT')
            current_price = market_data.get('price', 0)
            
            # Определение действия
            if signal.signal_type == SignalType.BUY:
                action = "BUY"
            elif signal.signal_type == SignalType.SELL:
                action = "SELL"
            else:
                action = "HOLD"
            
            # Расчет размера позиции
            position_size = risk_assessment.get('position_size', 0.0)
            
            # Расчет стоп-лосса и тейк-профита
            stop_loss = risk_assessment.get('stop_loss')
            take_profit = risk_assessment.get('take_profit')
            
            # Формирование обоснования
            reasoning_parts = [
                f"AI консенсус: {signal.signal_type.value} (уверенность: {signal.confidence:.2f})",
                f"Рыночная фаза: {market_condition.primary_phase.value}",
                f"Риск-оценка: {risk_assessment.get('risk_level', 'unknown')}"
            ]
            
            if market_condition.secondary_phase:
                reasoning_parts.append(f"Вторичная фаза: {market_condition.secondary_phase.value}")
            
            reasoning = "; ".join(reasoning_parts)
            
            decision = TradingDecision(
                symbol=symbol,
                action=action,
                quantity=position_size,
                price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                confidence=signal.confidence,
                reasoning=reasoning,
                timestamp=datetime.now(),
                market_phase=market_condition.primary_phase,
                risk_level=risk_assessment.get('risk_level', 'medium'),
                ai_signals=ai_signals
            )
            
            return decision
            
        except Exception as e:
            logger.error(f"Ошибка создания торгового решения: {e}")
            raise
    
    async def _update_metrics(self, decision: TradingDecision, processing_time: float) -> None:
        """Обновление метрик системы"""
        try:
            self.metrics.total_decisions += 1
            
            # Обновление времени обработки
            if self.metrics.total_decisions == 1:
                self.metrics.avg_processing_time = processing_time
            else:
                self.metrics.avg_processing_time = (
                    (self.metrics.avg_processing_time * (self.metrics.total_decisions - 1) + processing_time) /
                    self.metrics.total_decisions
                )
            
            # Обновление средней уверенности
            if self.metrics.total_decisions == 1:
                self.metrics.avg_confidence = decision.confidence
            else:
                self.metrics.avg_confidence = (
                    (self.metrics.avg_confidence * (self.metrics.total_decisions - 1) + decision.confidence) /
                    self.metrics.total_decisions
                )
            
            # Получение системных метрик
            perf_metrics = self.performance_optimizer.get_performance_metrics()
            if perf_metrics:
                self.metrics.memory_usage = perf_metrics.memory_usage
                self.metrics.cpu_usage = perf_metrics.cpu_usage
                self.metrics.cache_hit_ratio = perf_metrics.cache_hit_ratio
            
            self.metrics.timestamp = datetime.now()
            
            # Проверка производительности
            if processing_time > self.max_processing_time:
                logger.warning(f"Время обработки превышено: {processing_time:.2f}с > {self.max_processing_time}с")
            
        except Exception as e:
            logger.error(f"Ошибка обновления метрик: {e}")
    
    async def execute_decision(self, decision: TradingDecision) -> Dict[str, Any]:
        """Выполнение торгового решения"""
        try:
            logger.info(f"Выполнение решения: {decision.action} {decision.quantity} {decision.symbol}")
            
            # В реальной системе здесь будет интеграция с биржей
            # Пока что симулируем выполнение
            
            execution_result = {
                'success': True,
                'order_id': f"order_{int(time.time())}",
                'executed_price': decision.price,
                'executed_quantity': decision.quantity,
                'timestamp': datetime.now(),
                'fees': decision.quantity * decision.price * 0.001 if decision.price else 0  # 0.1% комиссия
            }
            
            # Создание позиции для риск-менеджера
            if decision.action in ["BUY", "SELL"]:
                position = Position(
                    symbol=decision.symbol,
                    position_type=PositionType.LONG if decision.action == "BUY" else PositionType.SHORT,
                    size=decision.quantity,
                    entry_price=decision.price or 0,
                    stop_loss=decision.stop_loss,
                    take_profit=decision.take_profit,
                    timestamp=decision.timestamp
                )
                
                await self.risk_manager.add_position(position)
            
            # Обновление метрик успешности
            self.metrics.successful_decisions += 1
            
            # Добавление результата в мониторинг
            self.architecture_monitor.add_trade_result(
                symbol=decision.symbol,
                pnl=0.0,  # PnL будет рассчитан позже при закрытии позиции
                entry_time=decision.timestamp,
                exit_time=decision.timestamp,  # Временно
                trade_type=decision.action
            )
            
            logger.info(f"Решение выполнено успешно: {execution_result['order_id']}")
            return execution_result
            
        except Exception as e:
            logger.error(f"Ошибка выполнения решения: {e}")
            self.metrics.failed_decisions += 1
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now()
            }
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Получение статуса системы"""
        try:
            # Статус компонентов
            component_status = {
                'ai_coordinator': self.ai_coordinator.is_initialized if self.ai_coordinator else False,
                'market_detector': self.market_detector is not None,
                'risk_manager': self.risk_manager is not None,
                'performance_optimizer': True,  # Всегда доступен
                'architecture_monitor': self.architecture_monitor is not None
            }
            
            # Текущее здоровье системы
            current_health = self.architecture_monitor.get_current_health()
            
            # Активные алерты
            active_alerts = self.architecture_monitor.get_active_alerts()
            
            # Статус мониторинга
            monitor_status = await self.architecture_monitor.get_system_status()
            
            return {
                'system_state': self.state.value,
                'components': component_status,
                'metrics': {
                    'total_decisions': self.metrics.total_decisions,
                    'successful_decisions': self.metrics.successful_decisions,
                    'failed_decisions': self.metrics.failed_decisions,
                    'success_rate': (self.metrics.successful_decisions / max(1, self.metrics.total_decisions)) * 100,
                    'avg_confidence': self.metrics.avg_confidence,
                    'avg_processing_time': self.metrics.avg_processing_time,
                    'memory_usage': self.metrics.memory_usage,
                    'cpu_usage': self.metrics.cpu_usage,
                    'cache_hit_ratio': self.metrics.cache_hit_ratio
                },
                'health': {
                    'overall_score': current_health.overall_score if current_health else 0,
                    'status': current_health.status if current_health else 'unknown',
                    'performance_score': current_health.performance_score if current_health else 0,
                    'accuracy_score': current_health.accuracy_score if current_health else 0,
                    'risk_score': current_health.risk_score if current_health else 0,
                    'system_score': current_health.system_score if current_health else 0
                },
                'alerts': {
                    'total_active': len(active_alerts),
                    'critical': len([a for a in active_alerts if a.level.value == 'critical']),
                    'warning': len([a for a in active_alerts if a.level.value == 'warning'])
                },
                'monitoring': monitor_status,
                'last_decision': self.last_decision_time.isoformat() if self.last_decision_time else None,
                'uptime': (datetime.now() - self.metrics.timestamp).total_seconds() if self.metrics.timestamp else 0
            }
            
        except Exception as e:
            logger.error(f"Ошибка получения статуса системы: {e}")
            return {
                'system_state': self.state.value,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def shutdown(self) -> None:
        """Корректное завершение работы системы"""
        try:
            logger.info("Начало завершения работы системы...")
            self.state = SystemState.SHUTDOWN
            
            # Остановка мониторинга
            if self.architecture_monitor:
                await self.architecture_monitor.stop_monitoring()
                logger.info("Мониторинг остановлен")
            
            # Остановка оптимизатора производительности
            await self.performance_optimizer.stop()
            logger.info("Оптимизатор производительности остановлен")
            
            # Сохранение финального отчета
            await self._save_final_report()
            
            logger.info("Система корректно завершила работу")
            
        except Exception as e:
            logger.error(f"Ошибка при завершении работы: {e}")
    
    async def _save_final_report(self) -> None:
        """Сохранение финального отчета"""
        try:
            import os
            import json
            
            report_data = {
                'session_summary': {
                    'start_time': self.metrics.timestamp.isoformat(),
                    'end_time': datetime.now().isoformat(),
                    'total_decisions': self.metrics.total_decisions,
                    'successful_decisions': self.metrics.successful_decisions,
                    'failed_decisions': self.metrics.failed_decisions,
                    'success_rate': (self.metrics.successful_decisions / max(1, self.metrics.total_decisions)) * 100,
                    'avg_confidence': self.metrics.avg_confidence,
                    'avg_processing_time': self.metrics.avg_processing_time
                },
                'final_status': await self.get_system_status(),
                'decision_history': [
                    {
                        'symbol': d.symbol,
                        'action': d.action,
                        'confidence': d.confidence,
                        'market_phase': d.market_phase.value,
                        'timestamp': d.timestamp.isoformat()
                    }
                    for d in self.decision_history[-100:]  # Последние 100 решений
                ]
            }
            
            # Сохранение отчета
            reports_dir = "reports"
            os.makedirs(reports_dir, exist_ok=True)
            
            filename = f"integration_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            filepath = os.path.join(reports_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"Финальный отчет сохранен: {filepath}")
            
        except Exception as e:
            logger.error(f"Ошибка сохранения финального отчета: {e}")

# Глобальный экземпляр менеджера интеграции
_integration_manager: Optional[IntegrationManager] = None

def get_integration_manager() -> IntegrationManager:
    """Получение глобального экземпляра менеджера интеграции"""
    global _integration_manager
    if _integration_manager is None:
        _integration_manager = IntegrationManager()
    return _integration_manager