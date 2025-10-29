"""
Система мониторинга архитектурных изменений для Peper Binance v4
Архитектурные улучшения - Фаза 2
"""

import asyncio
import logging
import time
import json
import os
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque
from enum import Enum
import numpy as np
import pandas as pd

from config.unified_config_manager import get_config_manager
from ai_coordination.ai_coordinator import AICoordinator
from market_analysis.market_phase_detector import MarketPhaseDetector
from risk_management.advanced_risk_manager import AdvancedRiskManager
from performance.performance_optimizer import get_performance_optimizer

logger = logging.getLogger(__name__)

class MetricType(Enum):
    """Типы метрик"""
    PERFORMANCE = "performance"
    ACCURACY = "accuracy"
    RISK = "risk"
    SYSTEM = "system"
    BUSINESS = "business"

class AlertLevel(Enum):
    """Уровни алертов"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class Metric:
    """Метрика системы"""
    name: str
    value: float
    metric_type: MetricType
    timestamp: datetime
    target_value: Optional[float] = None
    threshold_warning: Optional[float] = None
    threshold_critical: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Alert:
    """Алерт системы"""
    id: str
    level: AlertLevel
    message: str
    metric_name: str
    current_value: float
    threshold_value: float
    timestamp: datetime
    resolved: bool = False
    resolution_time: Optional[datetime] = None

@dataclass
class SystemHealth:
    """Здоровье системы"""
    overall_score: float  # 0-100
    performance_score: float
    accuracy_score: float
    risk_score: float
    system_score: float
    timestamp: datetime
    status: str  # "healthy", "warning", "critical"

@dataclass
class TradingMetrics:
    """Торговые метрики"""
    win_rate: float
    total_trades: int
    profitable_trades: int
    losing_trades: int
    total_pnl: float
    max_drawdown: float
    sharpe_ratio: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    consecutive_wins: int
    consecutive_losses: int
    timestamp: datetime

class MetricsCollector:
    """Сборщик метрик"""
    
    def __init__(self):
        self.ai_coordinator: Optional[AICoordinator] = None
        self.market_detector: Optional[MarketPhaseDetector] = None
        self.risk_manager: Optional[AdvancedRiskManager] = None
        self.performance_optimizer = get_performance_optimizer()
        
        # История метрик
        self.metrics_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.trading_history: deque = deque(maxlen=1000)
        
        # Базовые метрики
        self.baseline_metrics: Dict[str, float] = {
            'win_rate': 37.0,  # Текущий показатель
            'roi': -99.6,      # Текущий ROI
            'max_drawdown': 15.0,
            'sharpe_ratio': -2.5,
            'profit_factor': 0.3
        }
        
        # Целевые метрики (Фаза 2)
        self.target_metrics: Dict[str, float] = {
            'win_rate': 65.0,
            'roi': 5.0,
            'max_drawdown': 6.0,
            'sharpe_ratio': 1.0,
            'profit_factor': 1.5
        }
    
    def set_components(self, ai_coordinator: AICoordinator, 
                      market_detector: MarketPhaseDetector,
                      risk_manager: AdvancedRiskManager) -> None:
        """Установка компонентов системы"""
        self.ai_coordinator = ai_coordinator
        self.market_detector = market_detector
        self.risk_manager = risk_manager
    
    async def collect_all_metrics(self) -> List[Metric]:
        """Сбор всех метрик системы"""
        metrics = []
        
        # Метрики производительности
        metrics.extend(await self._collect_performance_metrics())
        
        # Метрики точности
        metrics.extend(await self._collect_accuracy_metrics())
        
        # Метрики риска
        metrics.extend(await self._collect_risk_metrics())
        
        # Системные метрики
        metrics.extend(await self._collect_system_metrics())
        
        # Бизнес-метрики
        metrics.extend(await self._collect_business_metrics())
        
        # Сохранение в историю
        for metric in metrics:
            self.metrics_history[metric.name].append(metric)
        
        return metrics
    
    async def _collect_performance_metrics(self) -> List[Metric]:
        """Сбор метрик производительности"""
        metrics = []
        timestamp = datetime.now()
        
        try:
            # Метрики оптимизатора производительности
            perf_metrics = self.performance_optimizer.get_performance_metrics()
            if perf_metrics:
                metrics.extend([
                    Metric(
                        name="cpu_usage",
                        value=perf_metrics.cpu_usage,
                        metric_type=MetricType.PERFORMANCE,
                        timestamp=timestamp,
                        threshold_warning=80.0,
                        threshold_critical=95.0
                    ),
                    Metric(
                        name="memory_usage",
                        value=perf_metrics.memory_usage,
                        metric_type=MetricType.PERFORMANCE,
                        timestamp=timestamp,
                        threshold_warning=85.0,
                        threshold_critical=95.0
                    ),
                    Metric(
                        name="cache_hit_ratio",
                        value=perf_metrics.cache_hit_ratio * 100,
                        metric_type=MetricType.PERFORMANCE,
                        timestamp=timestamp,
                        target_value=80.0,
                        threshold_warning=50.0,
                        threshold_critical=30.0
                    ),
                    Metric(
                        name="avg_response_time",
                        value=perf_metrics.avg_response_time * 1000,  # в миллисекундах
                        metric_type=MetricType.PERFORMANCE,
                        timestamp=timestamp,
                        target_value=100.0,
                        threshold_warning=500.0,
                        threshold_critical=1000.0
                    )
                ])
            
            # Метрики задач
            task_stats = self.performance_optimizer.get_task_stats()
            if task_stats:
                metrics.append(
                    Metric(
                        name="active_tasks",
                        value=task_stats['active_tasks'],
                        metric_type=MetricType.PERFORMANCE,
                        timestamp=timestamp,
                        threshold_warning=task_stats['max_workers'] * 0.8,
                        threshold_critical=task_stats['max_workers'] * 0.95
                    )
                )
                
                if task_stats['completed_tasks'] + task_stats['failed_tasks'] > 0:
                    success_rate = (task_stats['completed_tasks'] / 
                                  (task_stats['completed_tasks'] + task_stats['failed_tasks']) * 100)
                    metrics.append(
                        Metric(
                            name="task_success_rate",
                            value=success_rate,
                            metric_type=MetricType.PERFORMANCE,
                            timestamp=timestamp,
                            target_value=95.0,
                            threshold_warning=90.0,
                            threshold_critical=80.0
                        )
                    )
        
        except Exception as e:
            logger.error(f"Ошибка сбора метрик производительности: {e}")
        
        return metrics
    
    async def _collect_accuracy_metrics(self) -> List[Metric]:
        """Сбор метрик точности"""
        metrics = []
        timestamp = datetime.now()
        
        try:
            if self.ai_coordinator:
                # Метрики AI координатора
                performance_tracker = self.ai_coordinator.performance_tracker
                
                for module_name, performance in performance_tracker.module_performance.items():
                    if performance['total_signals'] > 0:
                        accuracy = performance['correct_signals'] / performance['total_signals'] * 100
                        metrics.append(
                            Metric(
                                name=f"{module_name}_accuracy",
                                value=accuracy,
                                metric_type=MetricType.ACCURACY,
                                timestamp=timestamp,
                                target_value=70.0,
                                threshold_warning=50.0,
                                threshold_critical=30.0,
                                metadata={'module': module_name}
                            )
                        )
                
                # Общая точность консенсуса
                total_signals = sum(p['total_signals'] for p in performance_tracker.module_performance.values())
                total_correct = sum(p['correct_signals'] for p in performance_tracker.module_performance.values())
                
                if total_signals > 0:
                    consensus_accuracy = total_correct / total_signals * 100
                    metrics.append(
                        Metric(
                            name="consensus_accuracy",
                            value=consensus_accuracy,
                            metric_type=MetricType.ACCURACY,
                            timestamp=timestamp,
                            target_value=75.0,
                            threshold_warning=60.0,
                            threshold_critical=40.0
                        )
                    )
            
            if self.market_detector:
                # Точность определения фаз рынка
                market_condition = self.market_detector.get_current_condition()
                if market_condition:
                    metrics.append(
                        Metric(
                            name="market_phase_confidence",
                            value=market_condition.confidence * 100,
                            metric_type=MetricType.ACCURACY,
                            timestamp=timestamp,
                            target_value=80.0,
                            threshold_warning=60.0,
                            threshold_critical=40.0
                        )
                    )
        
        except Exception as e:
            logger.error(f"Ошибка сбора метрик точности: {e}")
        
        return metrics
    
    async def _collect_risk_metrics(self) -> List[Metric]:
        """Сбор метрик риска"""
        metrics = []
        timestamp = datetime.now()
        
        try:
            if self.risk_manager:
                risk_metrics = await self.risk_manager.calculate_risk_metrics()
                
                metrics.extend([
                    Metric(
                        name="portfolio_risk",
                        value=risk_metrics.portfolio_risk * 100,
                        metric_type=MetricType.RISK,
                        timestamp=timestamp,
                        target_value=10.0,
                        threshold_warning=15.0,
                        threshold_critical=20.0
                    ),
                    Metric(
                        name="correlation_risk",
                        value=risk_metrics.correlation_risk * 100,
                        metric_type=MetricType.RISK,
                        timestamp=timestamp,
                        target_value=50.0,
                        threshold_warning=70.0,
                        threshold_critical=85.0
                    ),
                    Metric(
                        name="volatility_risk",
                        value=risk_metrics.volatility_risk * 100,
                        metric_type=MetricType.RISK,
                        timestamp=timestamp,
                        target_value=50.0,
                        threshold_warning=75.0,
                        threshold_critical=90.0
                    ),
                    Metric(
                        name="max_drawdown",
                        value=risk_metrics.max_drawdown * 100,
                        metric_type=MetricType.RISK,
                        timestamp=timestamp,
                        target_value=6.0,
                        threshold_warning=10.0,
                        threshold_critical=15.0
                    ),
                    Metric(
                        name="sharpe_ratio",
                        value=risk_metrics.sharpe_ratio,
                        metric_type=MetricType.RISK,
                        timestamp=timestamp,
                        target_value=1.0,
                        threshold_warning=0.5,
                        threshold_critical=0.0
                    )
                ])
                
                # Количество открытых позиций
                positions = self.risk_manager.get_current_positions()
                metrics.append(
                    Metric(
                        name="open_positions",
                        value=len(positions),
                        metric_type=MetricType.RISK,
                        timestamp=timestamp,
                        threshold_warning=8,
                        threshold_critical=10
                    )
                )
        
        except Exception as e:
            logger.error(f"Ошибка сбора метрик риска: {e}")
        
        return metrics
    
    async def _collect_system_metrics(self) -> List[Metric]:
        """Сбор системных метрик"""
        metrics = []
        timestamp = datetime.now()
        
        try:
            # Системная информация
            system_info = self.performance_optimizer.get_system_info()
            
            metrics.extend([
                Metric(
                    name="system_memory_usage",
                    value=system_info['memory_percent'],
                    metric_type=MetricType.SYSTEM,
                    timestamp=timestamp,
                    threshold_warning=80.0,
                    threshold_critical=90.0
                ),
                Metric(
                    name="system_cpu_usage",
                    value=system_info['cpu_percent'],
                    metric_type=MetricType.SYSTEM,
                    timestamp=timestamp,
                    threshold_warning=80.0,
                    threshold_critical=95.0
                )
            ])
            
            # Статистика кэша
            cache_stats = self.performance_optimizer.get_cache_stats()
            if cache_stats.hits + cache_stats.misses > 0:
                metrics.append(
                    Metric(
                        name="cache_efficiency",
                        value=cache_stats.hit_ratio * 100,
                        metric_type=MetricType.SYSTEM,
                        timestamp=timestamp,
                        target_value=80.0,
                        threshold_warning=60.0,
                        threshold_critical=40.0
                    )
                )
        
        except Exception as e:
            logger.error(f"Ошибка сбора системных метрик: {e}")
        
        return metrics
    
    async def _collect_business_metrics(self) -> List[Metric]:
        """Сбор бизнес-метрик"""
        metrics = []
        timestamp = datetime.now()
        
        try:
            # Расчет торговых метрик на основе истории
            trading_metrics = await self._calculate_trading_metrics()
            
            if trading_metrics:
                metrics.extend([
                    Metric(
                        name="win_rate",
                        value=trading_metrics.win_rate,
                        metric_type=MetricType.BUSINESS,
                        timestamp=timestamp,
                        target_value=self.target_metrics['win_rate'],
                        threshold_warning=50.0,
                        threshold_critical=40.0
                    ),
                    Metric(
                        name="profit_factor",
                        value=trading_metrics.profit_factor,
                        metric_type=MetricType.BUSINESS,
                        timestamp=timestamp,
                        target_value=self.target_metrics['profit_factor'],
                        threshold_warning=1.0,
                        threshold_critical=0.8
                    ),
                    Metric(
                        name="total_pnl",
                        value=trading_metrics.total_pnl,
                        metric_type=MetricType.BUSINESS,
                        timestamp=timestamp,
                        target_value=500.0,  # $500 прибыли
                        threshold_warning=-200.0,
                        threshold_critical=-500.0
                    )
                ])
        
        except Exception as e:
            logger.error(f"Ошибка сбора бизнес-метрик: {e}")
        
        return metrics
    
    async def _calculate_trading_metrics(self) -> Optional[TradingMetrics]:
        """Расчет торговых метрик"""
        if not self.trading_history:
            return None
        
        try:
            # Симуляция торговых данных (в реальной системе данные из базы)
            trades = list(self.trading_history)
            
            if not trades:
                return None
            
            total_trades = len(trades)
            profitable_trades = sum(1 for trade in trades if trade.get('pnl', 0) > 0)
            losing_trades = total_trades - profitable_trades
            
            win_rate = (profitable_trades / total_trades * 100) if total_trades > 0 else 0
            
            total_pnl = sum(trade.get('pnl', 0) for trade in trades)
            
            profits = [trade['pnl'] for trade in trades if trade.get('pnl', 0) > 0]
            losses = [abs(trade['pnl']) for trade in trades if trade.get('pnl', 0) < 0]
            
            avg_win = np.mean(profits) if profits else 0
            avg_loss = np.mean(losses) if losses else 0
            
            profit_factor = (sum(profits) / sum(losses)) if losses and sum(losses) > 0 else 0
            
            # Максимальная просадка (упрощенный расчет)
            cumulative_pnl = np.cumsum([trade.get('pnl', 0) for trade in trades])
            peak = np.maximum.accumulate(cumulative_pnl)
            drawdown = (peak - cumulative_pnl) / np.maximum(peak, 1)
            max_drawdown = np.max(drawdown) * 100
            
            # Коэффициент Шарпа (упрощенный)
            returns = [trade.get('pnl', 0) for trade in trades]
            if len(returns) > 1:
                sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
            else:
                sharpe_ratio = 0
            
            return TradingMetrics(
                win_rate=win_rate,
                total_trades=total_trades,
                profitable_trades=profitable_trades,
                losing_trades=losing_trades,
                total_pnl=total_pnl,
                max_drawdown=max_drawdown,
                sharpe_ratio=sharpe_ratio,
                profit_factor=profit_factor,
                avg_win=avg_win,
                avg_loss=avg_loss,
                largest_win=max(profits) if profits else 0,
                largest_loss=max(losses) if losses else 0,
                consecutive_wins=0,  # Требует более сложной логики
                consecutive_losses=0,  # Требует более сложной логики
                timestamp=datetime.now()
            )
        
        except Exception as e:
            logger.error(f"Ошибка расчета торговых метрик: {e}")
            return None
    
    def add_trade_result(self, symbol: str, pnl: float, entry_time: datetime, 
                        exit_time: datetime, trade_type: str) -> None:
        """Добавление результата сделки"""
        trade_data = {
            'symbol': symbol,
            'pnl': pnl,
            'entry_time': entry_time,
            'exit_time': exit_time,
            'trade_type': trade_type,
            'timestamp': datetime.now()
        }
        
        self.trading_history.append(trade_data)
    
    def get_metric_history(self, metric_name: str, hours: int = 24) -> List[Metric]:
        """Получение истории метрики"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        if metric_name in self.metrics_history:
            return [
                metric for metric in self.metrics_history[metric_name]
                if metric.timestamp >= cutoff_time
            ]
        
        return []

class AlertManager:
    """Менеджер алертов"""
    
    def __init__(self):
        self.alerts: Dict[str, Alert] = {}
        self.alert_history: deque = deque(maxlen=1000)
        self.alert_rules: Dict[str, Dict[str, Any]] = {}
        
        # Настройка правил алертов
        self._setup_alert_rules()
    
    def _setup_alert_rules(self) -> None:
        """Настройка правил алертов"""
        self.alert_rules = {
            'win_rate_critical': {
                'metric': 'win_rate',
                'condition': 'less_than',
                'threshold': 40.0,
                'level': AlertLevel.CRITICAL,
                'message': 'Критически низкий винрейт: {value:.1f}%'
            },
            'memory_warning': {
                'metric': 'memory_usage',
                'condition': 'greater_than',
                'threshold': 85.0,
                'level': AlertLevel.WARNING,
                'message': 'Высокое использование памяти: {value:.1f}%'
            },
            'drawdown_critical': {
                'metric': 'max_drawdown',
                'condition': 'greater_than',
                'threshold': 15.0,
                'level': AlertLevel.CRITICAL,
                'message': 'Критическая просадка: {value:.1f}%'
            },
            'cache_efficiency_warning': {
                'metric': 'cache_hit_ratio',
                'condition': 'less_than',
                'threshold': 50.0,
                'level': AlertLevel.WARNING,
                'message': 'Низкая эффективность кэша: {value:.1f}%'
            }
        }
    
    def check_metrics(self, metrics: List[Metric]) -> List[Alert]:
        """Проверка метрик на алерты"""
        new_alerts = []
        
        for metric in metrics:
            # Проверка пороговых значений
            alerts = self._check_thresholds(metric)
            new_alerts.extend(alerts)
            
            # Проверка правил алертов
            alerts = self._check_rules(metric)
            new_alerts.extend(alerts)
        
        # Сохранение алертов
        for alert in new_alerts:
            self.alerts[alert.id] = alert
            self.alert_history.append(alert)
        
        return new_alerts
    
    def _check_thresholds(self, metric: Metric) -> List[Alert]:
        """Проверка пороговых значений метрики"""
        alerts = []
        
        # Критический порог
        if (metric.threshold_critical is not None and 
            metric.value >= metric.threshold_critical):
            
            alert_id = f"{metric.name}_critical_{int(time.time())}"
            alert = Alert(
                id=alert_id,
                level=AlertLevel.CRITICAL,
                message=f"Критическое значение {metric.name}: {metric.value:.2f}",
                metric_name=metric.name,
                current_value=metric.value,
                threshold_value=metric.threshold_critical,
                timestamp=metric.timestamp
            )
            alerts.append(alert)
        
        # Предупреждающий порог
        elif (metric.threshold_warning is not None and 
              metric.value >= metric.threshold_warning):
            
            alert_id = f"{metric.name}_warning_{int(time.time())}"
            alert = Alert(
                id=alert_id,
                level=AlertLevel.WARNING,
                message=f"Предупреждение {metric.name}: {metric.value:.2f}",
                metric_name=metric.name,
                current_value=metric.value,
                threshold_value=metric.threshold_warning,
                timestamp=metric.timestamp
            )
            alerts.append(alert)
        
        return alerts
    
    def _check_rules(self, metric: Metric) -> List[Alert]:
        """Проверка правил алертов"""
        alerts = []
        
        for rule_name, rule in self.alert_rules.items():
            if rule['metric'] == metric.name:
                condition_met = False
                
                if rule['condition'] == 'greater_than':
                    condition_met = metric.value > rule['threshold']
                elif rule['condition'] == 'less_than':
                    condition_met = metric.value < rule['threshold']
                
                if condition_met:
                    alert_id = f"{rule_name}_{int(time.time())}"
                    alert = Alert(
                        id=alert_id,
                        level=rule['level'],
                        message=rule['message'].format(value=metric.value),
                        metric_name=metric.name,
                        current_value=metric.value,
                        threshold_value=rule['threshold'],
                        timestamp=metric.timestamp
                    )
                    alerts.append(alert)
        
        return alerts
    
    def get_active_alerts(self) -> List[Alert]:
        """Получение активных алертов"""
        return [alert for alert in self.alerts.values() if not alert.resolved]
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Разрешение алерта"""
        if alert_id in self.alerts:
            self.alerts[alert_id].resolved = True
            self.alerts[alert_id].resolution_time = datetime.now()
            return True
        return False

class ArchitectureMonitor:
    """Основной класс мониторинга архитектуры"""
    
    def __init__(self):
        self.config_manager = get_config_manager()
        
        # Компоненты
        self.metrics_collector = MetricsCollector()
        self.alert_manager = AlertManager()
        
        # Состояние мониторинга
        self.is_monitoring = False
        self.monitoring_task: Optional[asyncio.Task] = None
        
        # История здоровья системы
        self.health_history: deque = deque(maxlen=1000)
        
        # Интервал мониторинга
        self.monitoring_interval = 60  # 1 минута
        
        logger.info("Монитор архитектуры инициализирован")
    
    def set_components(self, ai_coordinator: AICoordinator, 
                      market_detector: MarketPhaseDetector,
                      risk_manager: AdvancedRiskManager) -> None:
        """Установка компонентов системы"""
        self.metrics_collector.set_components(ai_coordinator, market_detector, risk_manager)
    
    async def start_monitoring(self) -> None:
        """Запуск мониторинга"""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Мониторинг архитектуры запущен")
    
    async def stop_monitoring(self) -> None:
        """Остановка мониторинга"""
        self.is_monitoring = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Мониторинг архитектуры остановлен")
    
    async def _monitoring_loop(self) -> None:
        """Цикл мониторинга"""
        while self.is_monitoring:
            try:
                # Сбор метрик
                metrics = await self.metrics_collector.collect_all_metrics()
                
                # Проверка алертов
                new_alerts = self.alert_manager.check_metrics(metrics)
                
                # Расчет здоровья системы
                health = await self._calculate_system_health(metrics)
                self.health_history.append(health)
                
                # Логирование новых алертов
                for alert in new_alerts:
                    if alert.level in [AlertLevel.ERROR, AlertLevel.CRITICAL]:
                        logger.error(f"АЛЕРТ {alert.level.value.upper()}: {alert.message}")
                    elif alert.level == AlertLevel.WARNING:
                        logger.warning(f"АЛЕРТ {alert.level.value.upper()}: {alert.message}")
                    else:
                        logger.info(f"АЛЕРТ {alert.level.value.upper()}: {alert.message}")
                
                # Сохранение отчета
                await self._save_monitoring_report(metrics, health, new_alerts)
                
                # Ожидание
                await asyncio.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Ошибка в цикле мониторинга: {e}")
                await asyncio.sleep(5)
    
    async def _calculate_system_health(self, metrics: List[Metric]) -> SystemHealth:
        """Расчет здоровья системы"""
        try:
            # Группировка метрик по типам
            metric_groups = defaultdict(list)
            for metric in metrics:
                metric_groups[metric.metric_type].append(metric)
            
            # Расчет оценок по группам
            performance_score = self._calculate_group_score(metric_groups[MetricType.PERFORMANCE])
            accuracy_score = self._calculate_group_score(metric_groups[MetricType.ACCURACY])
            risk_score = self._calculate_group_score(metric_groups[MetricType.RISK])
            system_score = self._calculate_group_score(metric_groups[MetricType.SYSTEM])
            
            # Общая оценка (взвешенная)
            weights = {
                'performance': 0.2,
                'accuracy': 0.3,
                'risk': 0.3,
                'system': 0.2
            }
            
            overall_score = (
                performance_score * weights['performance'] +
                accuracy_score * weights['accuracy'] +
                risk_score * weights['risk'] +
                system_score * weights['system']
            )
            
            # Определение статуса
            if overall_score >= 80:
                status = "healthy"
            elif overall_score >= 60:
                status = "warning"
            else:
                status = "critical"
            
            return SystemHealth(
                overall_score=overall_score,
                performance_score=performance_score,
                accuracy_score=accuracy_score,
                risk_score=risk_score,
                system_score=system_score,
                timestamp=datetime.now(),
                status=status
            )
        
        except Exception as e:
            logger.error(f"Ошибка расчета здоровья системы: {e}")
            return SystemHealth(
                overall_score=50.0,
                performance_score=50.0,
                accuracy_score=50.0,
                risk_score=50.0,
                system_score=50.0,
                timestamp=datetime.now(),
                status="unknown"
            )
    
    def _calculate_group_score(self, metrics: List[Metric]) -> float:
        """Расчет оценки группы метрик"""
        if not metrics:
            return 50.0  # Нейтральная оценка
        
        scores = []
        
        for metric in metrics:
            score = 50.0  # Базовая оценка
            
            # Если есть целевое значение
            if metric.target_value is not None:
                if metric.value >= metric.target_value:
                    score = 100.0
                else:
                    score = (metric.value / metric.target_value) * 100
                    score = max(0, min(100, score))
            
            # Корректировка на основе порогов
            elif metric.threshold_critical is not None:
                if metric.value >= metric.threshold_critical:
                    score = 0.0
                elif metric.threshold_warning is not None and metric.value >= metric.threshold_warning:
                    score = 30.0
                else:
                    score = 80.0
            
            scores.append(score)
        
        return np.mean(scores)
    
    async def _save_monitoring_report(self, metrics: List[Metric], 
                                    health: SystemHealth, alerts: List[Alert]) -> None:
        """Сохранение отчета мониторинга"""
        try:
            report_data = {
                'timestamp': datetime.now().isoformat(),
                'system_health': asdict(health),
                'metrics': [asdict(metric) for metric in metrics],
                'alerts': [asdict(alert) for alert in alerts],
                'summary': {
                    'total_metrics': len(metrics),
                    'active_alerts': len(self.alert_manager.get_active_alerts()),
                    'new_alerts': len(alerts),
                    'health_status': health.status
                }
            }
            
            # Сохранение в файл
            reports_dir = "monitoring/reports"
            os.makedirs(reports_dir, exist_ok=True)
            
            filename = f"monitoring_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            filepath = os.path.join(reports_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False, default=str)
        
        except Exception as e:
            logger.error(f"Ошибка сохранения отчета: {e}")
    
    # Методы для получения данных
    def get_current_health(self) -> Optional[SystemHealth]:
        """Получение текущего здоровья системы"""
        return self.health_history[-1] if self.health_history else None
    
    def get_health_history(self, hours: int = 24) -> List[SystemHealth]:
        """Получение истории здоровья системы"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [
            health for health in self.health_history
            if health.timestamp >= cutoff_time
        ]
    
    def get_active_alerts(self) -> List[Alert]:
        """Получение активных алертов"""
        return self.alert_manager.get_active_alerts()
    
    def get_metric_history(self, metric_name: str, hours: int = 24) -> List[Metric]:
        """Получение истории метрики"""
        return self.metrics_collector.get_metric_history(metric_name, hours)
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Получение статуса системы"""
        current_health = self.get_current_health()
        active_alerts = self.get_active_alerts()
        
        return {
            'monitoring_active': self.is_monitoring,
            'current_health': asdict(current_health) if current_health else None,
            'active_alerts_count': len(active_alerts),
            'critical_alerts': len([a for a in active_alerts if a.level == AlertLevel.CRITICAL]),
            'warning_alerts': len([a for a in active_alerts if a.level == AlertLevel.WARNING]),
            'last_update': datetime.now().isoformat()
        }
    
    def add_trade_result(self, symbol: str, pnl: float, entry_time: datetime, 
                        exit_time: datetime, trade_type: str) -> None:
        """Добавление результата сделки"""
        self.metrics_collector.add_trade_result(symbol, pnl, entry_time, exit_time, trade_type)

# Глобальный экземпляр монитора
_monitor_instance: Optional[ArchitectureMonitor] = None

def get_architecture_monitor() -> ArchitectureMonitor:
    """Получение глобального экземпляра монитора"""
    global _monitor_instance
    if _monitor_instance is None:
        _monitor_instance = ArchitectureMonitor()
    return _monitor_instance