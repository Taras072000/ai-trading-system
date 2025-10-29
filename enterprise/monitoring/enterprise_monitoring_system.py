"""
Enterprise Monitoring & Auto-Recovery System - Система мониторинга и автовосстановления
Обеспечивает комплексный мониторинг системы, алерты и автоматическое восстановление
"""

import asyncio
import json
import time
import psutil
import socket
import subprocess
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import logging
import redis.asyncio as redis
from collections import defaultdict, deque
import uuid
import aiohttp
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Мониторинг и метрики
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, generate_latest
import numpy as np
from scipy import stats

# Машинное обучение для аномалий
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib

class AlertSeverity(Enum):
    """Уровни серьезности алертов"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class MonitoringMetricType(Enum):
    """Типы метрик мониторинга"""
    SYSTEM_CPU = "system_cpu"
    SYSTEM_MEMORY = "system_memory"
    SYSTEM_DISK = "system_disk"
    SYSTEM_NETWORK = "system_network"
    APPLICATION_LATENCY = "app_latency"
    APPLICATION_THROUGHPUT = "app_throughput"
    APPLICATION_ERROR_RATE = "app_error_rate"
    TRADING_PNL = "trading_pnl"
    TRADING_WIN_RATE = "trading_win_rate"
    TRADING_DRAWDOWN = "trading_drawdown"
    DATABASE_CONNECTIONS = "db_connections"
    REDIS_MEMORY = "redis_memory"

class RecoveryActionType(Enum):
    """Типы действий восстановления"""
    RESTART_SERVICE = "restart_service"
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    CLEAR_CACHE = "clear_cache"
    RESTART_CONTAINER = "restart_container"
    FAILOVER = "failover"
    CIRCUIT_BREAKER = "circuit_breaker"
    THROTTLE_REQUESTS = "throttle_requests"

@dataclass
class MonitoringMetric:
    """Метрика мониторинга"""
    name: str
    type: MonitoringMetricType
    value: float
    timestamp: datetime
    tags: Dict[str, str] = None
    threshold_warning: Optional[float] = None
    threshold_critical: Optional[float] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = {}

@dataclass
class Alert:
    """Алерт"""
    id: str
    metric_name: str
    severity: AlertSeverity
    message: str
    value: float
    threshold: float
    timestamp: datetime
    resolved_at: Optional[datetime] = None
    is_resolved: bool = False
    recovery_actions: List[str] = None
    
    def __post_init__(self):
        if self.recovery_actions is None:
            self.recovery_actions = []

@dataclass
class RecoveryAction:
    """Действие восстановления"""
    id: str
    type: RecoveryActionType
    target: str
    parameters: Dict[str, Any]
    triggered_by_alert: str
    executed_at: Optional[datetime] = None
    success: Optional[bool] = None
    error_message: Optional[str] = None

@dataclass
class HealthCheck:
    """Проверка здоровья сервиса"""
    service_name: str
    endpoint: str
    method: str = "GET"
    expected_status: int = 200
    timeout: int = 30
    headers: Dict[str, str] = None
    
    def __post_init__(self):
        if self.headers is None:
            self.headers = {}

@dataclass
class ServiceStatus:
    """Статус сервиса"""
    name: str
    is_healthy: bool
    response_time: float
    last_check: datetime
    error_message: Optional[str] = None
    uptime_percentage: float = 100.0

# Метрики Prometheus
MONITORING_METRICS = Gauge('monitoring_metric_value', 'Monitoring metric value', ['metric_name', 'metric_type'])
ALERTS_TOTAL = Counter('alerts_total', 'Total alerts generated', ['severity', 'metric_name'])
RECOVERY_ACTIONS_TOTAL = Counter('recovery_actions_total', 'Total recovery actions executed', ['action_type', 'success'])
HEALTH_CHECKS_TOTAL = Counter('health_checks_total', 'Total health checks performed', ['service', 'status'])
SYSTEM_UPTIME = Gauge('system_uptime_seconds', 'System uptime in seconds')

class AnomalyDetector:
    """Детектор аномалий на основе машинного обучения"""
    
    def __init__(self, contamination: float = 0.1):
        self.contamination = contamination
        self.models: Dict[str, IsolationForest] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.metric_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
    def add_metric_value(self, metric_name: str, value: float, timestamp: datetime):
        """Добавление значения метрики"""
        self.metric_history[metric_name].append({
            'value': value,
            'timestamp': timestamp,
            'hour': timestamp.hour,
            'day_of_week': timestamp.weekday(),
            'minute': timestamp.minute
        })
        
    def train_model(self, metric_name: str) -> bool:
        """Обучение модели для метрики"""
        if len(self.metric_history[metric_name]) < 100:
            return False
            
        # Подготовка данных
        data = []
        for record in self.metric_history[metric_name]:
            features = [
                record['value'],
                record['hour'],
                record['day_of_week'],
                record['minute']
            ]
            data.append(features)
            
        X = np.array(data)
        
        # Нормализация
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Обучение модели
        model = IsolationForest(contamination=self.contamination, random_state=42)
        model.fit(X_scaled)
        
        self.models[metric_name] = model
        self.scalers[metric_name] = scaler
        
        return True
        
    def detect_anomaly(self, metric_name: str, value: float, timestamp: datetime) -> Tuple[bool, float]:
        """Обнаружение аномалии"""
        if metric_name not in self.models:
            return False, 0.0
            
        # Подготовка данных
        features = np.array([[
            value,
            timestamp.hour,
            timestamp.weekday(),
            timestamp.minute
        ]])
        
        # Нормализация
        features_scaled = self.scalers[metric_name].transform(features)
        
        # Предсказание
        anomaly_score = self.models[metric_name].decision_function(features_scaled)[0]
        is_anomaly = self.models[metric_name].predict(features_scaled)[0] == -1
        
        return is_anomaly, anomaly_score

class SystemMonitor:
    """Монитор системных ресурсов"""
    
    def __init__(self):
        self.start_time = time.time()
        
    def get_cpu_usage(self) -> float:
        """Получение использования CPU"""
        return psutil.cpu_percent(interval=1)
        
    def get_memory_usage(self) -> Dict[str, float]:
        """Получение использования памяти"""
        memory = psutil.virtual_memory()
        return {
            'percent': memory.percent,
            'available_gb': memory.available / (1024**3),
            'used_gb': memory.used / (1024**3),
            'total_gb': memory.total / (1024**3)
        }
        
    def get_disk_usage(self, path: str = '/') -> Dict[str, float]:
        """Получение использования диска"""
        disk = psutil.disk_usage(path)
        return {
            'percent': (disk.used / disk.total) * 100,
            'free_gb': disk.free / (1024**3),
            'used_gb': disk.used / (1024**3),
            'total_gb': disk.total / (1024**3)
        }
        
    def get_network_stats(self) -> Dict[str, float]:
        """Получение статистики сети"""
        net_io = psutil.net_io_counters()
        return {
            'bytes_sent': net_io.bytes_sent,
            'bytes_recv': net_io.bytes_recv,
            'packets_sent': net_io.packets_sent,
            'packets_recv': net_io.packets_recv
        }
        
    def get_process_info(self, process_name: str) -> Optional[Dict[str, Any]]:
        """Получение информации о процессе"""
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
            if process_name.lower() in proc.info['name'].lower():
                return {
                    'pid': proc.info['pid'],
                    'name': proc.info['name'],
                    'cpu_percent': proc.info['cpu_percent'],
                    'memory_percent': proc.info['memory_percent']
                }
        return None
        
    def get_uptime(self) -> float:
        """Получение времени работы системы"""
        return time.time() - self.start_time

class HealthChecker:
    """Проверка здоровья сервисов"""
    
    def __init__(self):
        self.health_checks: List[HealthCheck] = []
        self.service_statuses: Dict[str, ServiceStatus] = {}
        
    def add_health_check(self, health_check: HealthCheck):
        """Добавление проверки здоровья"""
        self.health_checks.append(health_check)
        
    async def check_service_health(self, health_check: HealthCheck) -> ServiceStatus:
        """Проверка здоровья сервиса"""
        start_time = time.time()
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.request(
                    health_check.method,
                    health_check.endpoint,
                    headers=health_check.headers,
                    timeout=aiohttp.ClientTimeout(total=health_check.timeout)
                ) as response:
                    response_time = time.time() - start_time
                    is_healthy = response.status == health_check.expected_status
                    
                    status = ServiceStatus(
                        name=health_check.service_name,
                        is_healthy=is_healthy,
                        response_time=response_time,
                        last_check=datetime.now()
                    )
                    
                    if not is_healthy:
                        status.error_message = f"Expected status {health_check.expected_status}, got {response.status}"
                        
                    HEALTH_CHECKS_TOTAL.labels(
                        service=health_check.service_name,
                        status='healthy' if is_healthy else 'unhealthy'
                    ).inc()
                    
                    return status
                    
        except Exception as e:
            response_time = time.time() - start_time
            status = ServiceStatus(
                name=health_check.service_name,
                is_healthy=False,
                response_time=response_time,
                last_check=datetime.now(),
                error_message=str(e)
            )
            
            HEALTH_CHECKS_TOTAL.labels(
                service=health_check.service_name,
                status='unhealthy'
            ).inc()
            
            return status
            
    async def check_all_services(self) -> Dict[str, ServiceStatus]:
        """Проверка всех сервисов"""
        tasks = []
        for health_check in self.health_checks:
            task = asyncio.create_task(self.check_service_health(health_check))
            tasks.append(task)
            
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, result in enumerate(results):
            if isinstance(result, ServiceStatus):
                service_name = self.health_checks[i].service_name
                self.service_statuses[service_name] = result
                
        return self.service_statuses

class AlertManager:
    """Менеджер алертов"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.notification_channels: List[str] = config.get('notification_channels', ['email'])
        
    def create_alert(self, metric: MonitoringMetric, threshold: float, severity: AlertSeverity) -> Alert:
        """Создание алерта"""
        alert_id = f"{metric.name}_{int(time.time())}"
        
        alert = Alert(
            id=alert_id,
            metric_name=metric.name,
            severity=severity,
            message=f"{metric.name} value {metric.value} exceeded threshold {threshold}",
            value=metric.value,
            threshold=threshold,
            timestamp=datetime.now()
        )
        
        self.active_alerts[alert_id] = alert
        self.alert_history.append(alert)
        
        ALERTS_TOTAL.labels(severity=severity.value, metric_name=metric.name).inc()
        
        return alert
        
    def resolve_alert(self, alert_id: str, resolution_message: str = ""):
        """Разрешение алерта"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.is_resolved = True
            alert.resolved_at = datetime.now()
            del self.active_alerts[alert_id]
            
    async def send_alert_notification(self, alert: Alert):
        """Отправка уведомления об алерте"""
        for channel in self.notification_channels:
            if channel == 'email':
                await self._send_email_notification(alert)
            elif channel == 'slack':
                await self._send_slack_notification(alert)
            elif channel == 'webhook':
                await self._send_webhook_notification(alert)
                
    async def _send_email_notification(self, alert: Alert):
        """Отправка email уведомления"""
        try:
            smtp_config = self.config.get('email', {})
            if not smtp_config:
                return
                
            msg = MIMEMultipart()
            msg['From'] = smtp_config.get('from_email')
            msg['To'] = smtp_config.get('to_email')
            msg['Subject'] = f"[{alert.severity.value.upper()}] Alert: {alert.metric_name}"
            
            body = f"""
Alert Details:
- Metric: {alert.metric_name}
- Severity: {alert.severity.value}
- Value: {alert.value}
- Threshold: {alert.threshold}
- Time: {alert.timestamp}
- Message: {alert.message}
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(smtp_config.get('smtp_server'), smtp_config.get('smtp_port'))
            server.starttls()
            server.login(smtp_config.get('username'), smtp_config.get('password'))
            server.send_message(msg)
            server.quit()
            
        except Exception as e:
            logging.error(f"Failed to send email notification: {e}")
            
    async def _send_slack_notification(self, alert: Alert):
        """Отправка Slack уведомления"""
        try:
            slack_config = self.config.get('slack', {})
            webhook_url = slack_config.get('webhook_url')
            
            if not webhook_url:
                return
                
            color_map = {
                AlertSeverity.INFO: 'good',
                AlertSeverity.WARNING: 'warning',
                AlertSeverity.ERROR: 'danger',
                AlertSeverity.CRITICAL: 'danger'
            }
            
            payload = {
                'attachments': [{
                    'color': color_map.get(alert.severity, 'warning'),
                    'title': f"Alert: {alert.metric_name}",
                    'text': alert.message,
                    'fields': [
                        {'title': 'Severity', 'value': alert.severity.value, 'short': True},
                        {'title': 'Value', 'value': str(alert.value), 'short': True},
                        {'title': 'Threshold', 'value': str(alert.threshold), 'short': True},
                        {'title': 'Time', 'value': alert.timestamp.isoformat(), 'short': True}
                    ]
                }]
            }
            
            async with aiohttp.ClientSession() as session:
                await session.post(webhook_url, json=payload)
                
        except Exception as e:
            logging.error(f"Failed to send Slack notification: {e}")
            
    async def _send_webhook_notification(self, alert: Alert):
        """Отправка webhook уведомления"""
        try:
            webhook_config = self.config.get('webhook', {})
            webhook_url = webhook_config.get('url')
            
            if not webhook_url:
                return
                
            payload = {
                'alert_id': alert.id,
                'metric_name': alert.metric_name,
                'severity': alert.severity.value,
                'message': alert.message,
                'value': alert.value,
                'threshold': alert.threshold,
                'timestamp': alert.timestamp.isoformat()
            }
            
            async with aiohttp.ClientSession() as session:
                await session.post(webhook_url, json=payload)
                
        except Exception as e:
            logging.error(f"Failed to send webhook notification: {e}")

class AutoRecoveryManager:
    """Менеджер автоматического восстановления"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.recovery_actions: List[RecoveryAction] = []
        self.recovery_rules: Dict[str, List[RecoveryActionType]] = {}
        
    def add_recovery_rule(self, metric_name: str, actions: List[RecoveryActionType]):
        """Добавление правила восстановления"""
        self.recovery_rules[metric_name] = actions
        
    async def execute_recovery_actions(self, alert: Alert) -> List[RecoveryAction]:
        """Выполнение действий восстановления"""
        executed_actions = []
        
        if alert.metric_name in self.recovery_rules:
            for action_type in self.recovery_rules[alert.metric_name]:
                action = RecoveryAction(
                    id=str(uuid.uuid4()),
                    type=action_type,
                    target=alert.metric_name,
                    parameters={},
                    triggered_by_alert=alert.id
                )
                
                success = await self._execute_action(action)
                action.executed_at = datetime.now()
                action.success = success
                
                executed_actions.append(action)
                self.recovery_actions.append(action)
                
                RECOVERY_ACTIONS_TOTAL.labels(
                    action_type=action_type.value,
                    success=str(success)
                ).inc()
                
        return executed_actions
        
    async def _execute_action(self, action: RecoveryAction) -> bool:
        """Выполнение конкретного действия"""
        try:
            if action.type == RecoveryActionType.RESTART_SERVICE:
                return await self._restart_service(action.target)
            elif action.type == RecoveryActionType.CLEAR_CACHE:
                return await self._clear_cache()
            elif action.type == RecoveryActionType.SCALE_UP:
                return await self._scale_service(action.target, "up")
            elif action.type == RecoveryActionType.SCALE_DOWN:
                return await self._scale_service(action.target, "down")
            elif action.type == RecoveryActionType.CIRCUIT_BREAKER:
                return await self._activate_circuit_breaker(action.target)
            else:
                action.error_message = f"Unknown action type: {action.type}"
                return False
                
        except Exception as e:
            action.error_message = str(e)
            return False
            
    async def _restart_service(self, service_name: str) -> bool:
        """Перезапуск сервиса"""
        try:
            # Пример для systemd сервиса
            result = subprocess.run(
                ['sudo', 'systemctl', 'restart', service_name],
                capture_output=True,
                text=True,
                timeout=60
            )
            return result.returncode == 0
        except Exception:
            return False
            
    async def _clear_cache(self) -> bool:
        """Очистка кеша"""
        try:
            # Очистка Redis кеша
            redis_client = redis.Redis(
                host=self.config.get('redis_host', 'localhost'),
                port=self.config.get('redis_port', 6379)
            )
            await redis_client.flushdb()
            await redis_client.close()
            return True
        except Exception:
            return False
            
    async def _scale_service(self, service_name: str, direction: str) -> bool:
        """Масштабирование сервиса"""
        try:
            # Пример для Docker Compose
            scale_factor = 2 if direction == "up" else 1
            result = subprocess.run(
                ['docker-compose', 'scale', f'{service_name}={scale_factor}'],
                capture_output=True,
                text=True,
                timeout=120
            )
            return result.returncode == 0
        except Exception:
            return False
            
    async def _activate_circuit_breaker(self, service_name: str) -> bool:
        """Активация circuit breaker"""
        try:
            # Установка флага в Redis
            redis_client = redis.Redis(
                host=self.config.get('redis_host', 'localhost'),
                port=self.config.get('redis_port', 6379)
            )
            await redis_client.set(f"circuit_breaker_{service_name}", "open", ex=300)
            await redis_client.close()
            return True
        except Exception:
            return False

class EnterpriseMonitoringSystem:
    """Enterprise система мониторинга и автовосстановления"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.redis_client = None
        self.logger = self._setup_logging()
        
        # Компоненты системы
        self.system_monitor = SystemMonitor()
        self.health_checker = HealthChecker()
        self.alert_manager = AlertManager(config)
        self.recovery_manager = AutoRecoveryManager(config)
        self.anomaly_detector = AnomalyDetector()
        
        # Метрики и состояние
        self.metrics: Dict[str, MonitoringMetric] = {}
        self.metric_thresholds: Dict[str, Dict[str, float]] = {}
        
        # Инициализация проверок здоровья и правил восстановления
        self._initialize_health_checks()
        self._initialize_recovery_rules()
        self._initialize_metric_thresholds()
        
    def _setup_logging(self) -> logging.Logger:
        """Настройка логирования"""
        logger = logging.getLogger('enterprise_monitoring')
        logger.setLevel(logging.INFO)
        
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
        
    def _initialize_health_checks(self):
        """Инициализация проверок здоровья"""
        # Основные сервисы
        self.health_checker.add_health_check(HealthCheck(
            service_name="trading_api",
            endpoint="http://localhost:8000/health",
            timeout=10
        ))
        
        self.health_checker.add_health_check(HealthCheck(
            service_name="web_dashboard",
            endpoint="http://localhost:3000/health",
            timeout=5
        ))
        
        self.health_checker.add_health_check(HealthCheck(
            service_name="redis",
            endpoint="http://localhost:6379/ping",
            timeout=5
        ))
        
    def _initialize_recovery_rules(self):
        """Инициализация правил восстановления"""
        # CPU перегрузка
        self.recovery_manager.add_recovery_rule(
            "system_cpu",
            [RecoveryActionType.SCALE_UP, RecoveryActionType.THROTTLE_REQUESTS]
        )
        
        # Память
        self.recovery_manager.add_recovery_rule(
            "system_memory",
            [RecoveryActionType.CLEAR_CACHE, RecoveryActionType.RESTART_SERVICE]
        )
        
        # Ошибки приложения
        self.recovery_manager.add_recovery_rule(
            "app_error_rate",
            [RecoveryActionType.CIRCUIT_BREAKER, RecoveryActionType.RESTART_SERVICE]
        )
        
    def _initialize_metric_thresholds(self):
        """Инициализация пороговых значений метрик"""
        self.metric_thresholds = {
            "system_cpu": {"warning": 70.0, "critical": 90.0},
            "system_memory": {"warning": 80.0, "critical": 95.0},
            "system_disk": {"warning": 85.0, "critical": 95.0},
            "app_latency": {"warning": 1000.0, "critical": 5000.0},  # мс
            "app_error_rate": {"warning": 5.0, "critical": 10.0},  # %
            "trading_drawdown": {"warning": 3.0, "critical": 5.0},  # %
            "trading_win_rate": {"warning": 70.0, "critical": 60.0}  # % (обратная логика)
        }
        
    async def start(self):
        """Запуск системы мониторинга"""
        # Подключение к Redis
        self.redis_client = redis.Redis(
            host=self.config.get('redis_host', 'localhost'),
            port=self.config.get('redis_port', 6379),
            decode_responses=True
        )
        
        # Запуск фоновых задач
        asyncio.create_task(self._collect_system_metrics())
        asyncio.create_task(self._collect_application_metrics())
        asyncio.create_task(self._health_check_loop())
        asyncio.create_task(self._anomaly_detection_loop())
        asyncio.create_task(self._alert_processing_loop())
        asyncio.create_task(self._train_anomaly_models())
        
        self.logger.info("Enterprise Monitoring System started")
        
    async def stop(self):
        """Остановка системы"""
        if self.redis_client:
            await self.redis_client.close()
            
    async def _collect_system_metrics(self):
        """Сбор системных метрик"""
        while True:
            try:
                timestamp = datetime.now()
                
                # CPU
                cpu_usage = self.system_monitor.get_cpu_usage()
                cpu_metric = MonitoringMetric(
                    name="system_cpu",
                    type=MonitoringMetricType.SYSTEM_CPU,
                    value=cpu_usage,
                    timestamp=timestamp,
                    threshold_warning=self.metric_thresholds["system_cpu"]["warning"],
                    threshold_critical=self.metric_thresholds["system_cpu"]["critical"]
                )
                await self._process_metric(cpu_metric)
                
                # Память
                memory_stats = self.system_monitor.get_memory_usage()
                memory_metric = MonitoringMetric(
                    name="system_memory",
                    type=MonitoringMetricType.SYSTEM_MEMORY,
                    value=memory_stats['percent'],
                    timestamp=timestamp,
                    threshold_warning=self.metric_thresholds["system_memory"]["warning"],
                    threshold_critical=self.metric_thresholds["system_memory"]["critical"]
                )
                await self._process_metric(memory_metric)
                
                # Диск
                disk_stats = self.system_monitor.get_disk_usage()
                disk_metric = MonitoringMetric(
                    name="system_disk",
                    type=MonitoringMetricType.SYSTEM_DISK,
                    value=disk_stats['percent'],
                    timestamp=timestamp,
                    threshold_warning=self.metric_thresholds["system_disk"]["warning"],
                    threshold_critical=self.metric_thresholds["system_disk"]["critical"]
                )
                await self._process_metric(disk_metric)
                
                # Время работы системы
                uptime = self.system_monitor.get_uptime()
                SYSTEM_UPTIME.set(uptime)
                
                await asyncio.sleep(30)  # Сбор каждые 30 секунд
                
            except Exception as e:
                self.logger.error(f"System metrics collection error: {e}")
                await asyncio.sleep(60)
                
    async def _collect_application_metrics(self):
        """Сбор метрик приложения"""
        while True:
            try:
                timestamp = datetime.now()
                
                # Получение метрик из Redis
                trading_data_json = await self.redis_client.get("current_trading_metrics")
                if trading_data_json:
                    trading_data = json.loads(trading_data_json)
                    
                    # Метрики торговли
                    if 'win_rate' in trading_data:
                        win_rate_metric = MonitoringMetric(
                            name="trading_win_rate",
                            type=MonitoringMetricType.TRADING_WIN_RATE,
                            value=trading_data['win_rate'],
                            timestamp=timestamp,
                            threshold_warning=self.metric_thresholds["trading_win_rate"]["warning"],
                            threshold_critical=self.metric_thresholds["trading_win_rate"]["critical"]
                        )
                        await self._process_metric(win_rate_metric)
                        
                    if 'max_drawdown' in trading_data:
                        drawdown_metric = MonitoringMetric(
                            name="trading_drawdown",
                            type=MonitoringMetricType.TRADING_DRAWDOWN,
                            value=abs(trading_data['max_drawdown']),
                            timestamp=timestamp,
                            threshold_warning=self.metric_thresholds["trading_drawdown"]["warning"],
                            threshold_critical=self.metric_thresholds["trading_drawdown"]["critical"]
                        )
                        await self._process_metric(drawdown_metric)
                        
                # Метрики производительности приложения
                app_metrics_json = await self.redis_client.get("app_performance_metrics")
                if app_metrics_json:
                    app_metrics = json.loads(app_metrics_json)
                    
                    if 'avg_latency' in app_metrics:
                        latency_metric = MonitoringMetric(
                            name="app_latency",
                            type=MonitoringMetricType.APPLICATION_LATENCY,
                            value=app_metrics['avg_latency'],
                            timestamp=timestamp,
                            threshold_warning=self.metric_thresholds["app_latency"]["warning"],
                            threshold_critical=self.metric_thresholds["app_latency"]["critical"]
                        )
                        await self._process_metric(latency_metric)
                        
                    if 'error_rate' in app_metrics:
                        error_rate_metric = MonitoringMetric(
                            name="app_error_rate",
                            type=MonitoringMetricType.APPLICATION_ERROR_RATE,
                            value=app_metrics['error_rate'],
                            timestamp=timestamp,
                            threshold_warning=self.metric_thresholds["app_error_rate"]["warning"],
                            threshold_critical=self.metric_thresholds["app_error_rate"]["critical"]
                        )
                        await self._process_metric(error_rate_metric)
                        
                await asyncio.sleep(60)  # Сбор каждую минуту
                
            except Exception as e:
                self.logger.error(f"Application metrics collection error: {e}")
                await asyncio.sleep(120)
                
    async def _process_metric(self, metric: MonitoringMetric):
        """Обработка метрики"""
        # Сохранение метрики
        self.metrics[metric.name] = metric
        
        # Обновление Prometheus метрик
        MONITORING_METRICS.labels(
            metric_name=metric.name,
            metric_type=metric.type.value
        ).set(metric.value)
        
        # Добавление в детектор аномалий
        self.anomaly_detector.add_metric_value(metric.name, metric.value, metric.timestamp)
        
        # Проверка пороговых значений
        await self._check_thresholds(metric)
        
        # Сохранение в Redis
        await self.redis_client.set(
            f"metric_{metric.name}",
            json.dumps(asdict(metric), default=str),
            ex=3600
        )
        
    async def _check_thresholds(self, metric: MonitoringMetric):
        """Проверка пороговых значений"""
        if metric.threshold_critical and metric.value >= metric.threshold_critical:
            alert = self.alert_manager.create_alert(
                metric, metric.threshold_critical, AlertSeverity.CRITICAL
            )
            await self._handle_alert(alert)
            
        elif metric.threshold_warning and metric.value >= metric.threshold_warning:
            # Проверка для обратной логики (например, win_rate)
            if metric.name == "trading_win_rate" and metric.value <= metric.threshold_warning:
                alert = self.alert_manager.create_alert(
                    metric, metric.threshold_warning, AlertSeverity.WARNING
                )
                await self._handle_alert(alert)
            elif metric.name != "trading_win_rate":
                alert = self.alert_manager.create_alert(
                    metric, metric.threshold_warning, AlertSeverity.WARNING
                )
                await self._handle_alert(alert)
                
    async def _handle_alert(self, alert: Alert):
        """Обработка алерта"""
        self.logger.warning(f"Alert generated: {alert.message}")
        
        # Отправка уведомления
        await self.alert_manager.send_alert_notification(alert)
        
        # Выполнение действий восстановления для критических алертов
        if alert.severity == AlertSeverity.CRITICAL:
            recovery_actions = await self.recovery_manager.execute_recovery_actions(alert)
            alert.recovery_actions = [action.id for action in recovery_actions]
            
            for action in recovery_actions:
                if action.success:
                    self.logger.info(f"Recovery action {action.type.value} executed successfully")
                else:
                    self.logger.error(f"Recovery action {action.type.value} failed: {action.error_message}")
                    
    async def _health_check_loop(self):
        """Цикл проверки здоровья сервисов"""
        while True:
            try:
                service_statuses = await self.health_checker.check_all_services()
                
                for service_name, status in service_statuses.items():
                    if not status.is_healthy:
                        # Создание алерта для нездорового сервиса
                        metric = MonitoringMetric(
                            name=f"service_{service_name}_health",
                            type=MonitoringMetricType.APPLICATION_ERROR_RATE,
                            value=0.0 if status.is_healthy else 100.0,
                            timestamp=datetime.now()
                        )
                        
                        alert = self.alert_manager.create_alert(
                            metric, 50.0, AlertSeverity.CRITICAL
                        )
                        alert.message = f"Service {service_name} is unhealthy: {status.error_message}"
                        await self._handle_alert(alert)
                        
                    # Сохранение статуса в Redis
                    await self.redis_client.set(
                        f"service_status_{service_name}",
                        json.dumps(asdict(status), default=str),
                        ex=300
                    )
                    
                await asyncio.sleep(60)  # Проверка каждую минуту
                
            except Exception as e:
                self.logger.error(f"Health check loop error: {e}")
                await asyncio.sleep(120)
                
    async def _anomaly_detection_loop(self):
        """Цикл обнаружения аномалий"""
        while True:
            try:
                for metric_name, metric in self.metrics.items():
                    is_anomaly, anomaly_score = self.anomaly_detector.detect_anomaly(
                        metric_name, metric.value, metric.timestamp
                    )
                    
                    if is_anomaly:
                        alert = Alert(
                            id=f"anomaly_{metric_name}_{int(time.time())}",
                            metric_name=metric_name,
                            severity=AlertSeverity.WARNING,
                            message=f"Anomaly detected in {metric_name}: value {metric.value}, score {anomaly_score:.3f}",
                            value=metric.value,
                            threshold=anomaly_score,
                            timestamp=datetime.now()
                        )
                        
                        self.alert_manager.active_alerts[alert.id] = alert
                        await self.alert_manager.send_alert_notification(alert)
                        
                await asyncio.sleep(300)  # Проверка каждые 5 минут
                
            except Exception as e:
                self.logger.error(f"Anomaly detection loop error: {e}")
                await asyncio.sleep(600)
                
    async def _train_anomaly_models(self):
        """Обучение моделей обнаружения аномалий"""
        while True:
            try:
                await asyncio.sleep(3600)  # Обучение каждый час
                
                for metric_name in self.metrics.keys():
                    success = self.anomaly_detector.train_model(metric_name)
                    if success:
                        self.logger.info(f"Anomaly detection model trained for {metric_name}")
                        
            except Exception as e:
                self.logger.error(f"Model training error: {e}")
                await asyncio.sleep(7200)  # Пауза при ошибке
                
    async def _alert_processing_loop(self):
        """Цикл обработки алертов"""
        while True:
            try:
                # Автоматическое разрешение алертов
                current_time = datetime.now()
                alerts_to_resolve = []
                
                for alert_id, alert in self.alert_manager.active_alerts.items():
                    # Разрешение алертов старше 1 часа, если метрика в норме
                    if (current_time - alert.timestamp).total_seconds() > 3600:
                        current_metric = self.metrics.get(alert.metric_name)
                        if current_metric:
                            if alert.severity == AlertSeverity.CRITICAL:
                                if current_metric.value < alert.threshold * 0.8:  # 20% буфер
                                    alerts_to_resolve.append(alert_id)
                            elif alert.severity == AlertSeverity.WARNING:
                                if current_metric.value < alert.threshold * 0.9:  # 10% буфер
                                    alerts_to_resolve.append(alert_id)
                                    
                for alert_id in alerts_to_resolve:
                    self.alert_manager.resolve_alert(alert_id, "Auto-resolved: metric returned to normal")
                    
                await asyncio.sleep(300)  # Проверка каждые 5 минут
                
            except Exception as e:
                self.logger.error(f"Alert processing loop error: {e}")
                await asyncio.sleep(600)
                
    async def get_system_status(self) -> Dict[str, Any]:
        """Получение статуса системы"""
        return {
            'timestamp': datetime.now().isoformat(),
            'metrics': {name: asdict(metric) for name, metric in self.metrics.items()},
            'active_alerts': {alert_id: asdict(alert) for alert_id, alert in self.alert_manager.active_alerts.items()},
            'service_statuses': {name: asdict(status) for name, status in self.health_checker.service_statuses.items()},
            'uptime': self.system_monitor.get_uptime()
        }

async def main():
    """Основная функция запуска"""
    config = {
        'redis_host': 'localhost',
        'redis_port': 6379,
        'notification_channels': ['email', 'slack'],
        'email': {
            'smtp_server': 'smtp.gmail.com',
            'smtp_port': 587,
            'from_email': 'monitoring@company.com',
            'to_email': 'admin@company.com',
            'username': 'monitoring@company.com',
            'password': 'app_password'
        },
        'slack': {
            'webhook_url': 'https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK'
        },
        'webhook': {
            'url': 'https://your-webhook-endpoint.com/alerts'
        }
    }
    
    monitoring_system = EnterpriseMonitoringSystem(config)
    await monitoring_system.start()
    
    print("Enterprise Monitoring & Auto-Recovery System started")
    
    try:
        # Симуляция работы системы
        await asyncio.sleep(10)
        
        # Получение статуса системы
        status = await monitoring_system.get_system_status()
        print(f"System status: {len(status['metrics'])} metrics, {len(status['active_alerts'])} active alerts")
        
        # Непрерывная работа
        print("System running... Press Ctrl+C to stop")
        await asyncio.Future()  # Бесконечное ожидание
        
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        await monitoring_system.stop()

if __name__ == '__main__':
    asyncio.run(main())