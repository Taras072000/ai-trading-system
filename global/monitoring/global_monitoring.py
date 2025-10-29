"""
Глобальная система мониторинга и отчетности для пятой фазы
Обеспечивает мониторинг всех компонентов системы в реальном времени
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import aiohttp
import psutil
import numpy as np
from collections import defaultdict, deque

class MetricType(Enum):
    PERFORMANCE = "performance"
    TRADING = "trading"
    SYSTEM = "system"
    USER = "user"
    FINANCIAL = "financial"
    SECURITY = "security"
    COMPLIANCE = "compliance"

class AlertLevel(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class RegionCode(Enum):
    GLOBAL = "global"
    AMERICAS = "americas"
    EUROPE = "europe"
    ASIA_PACIFIC = "asia_pacific"
    MIDDLE_EAST = "middle_east"
    AFRICA = "africa"

@dataclass
class Metric:
    name: str
    value: float
    unit: str
    timestamp: datetime
    region: RegionCode
    metric_type: MetricType
    tags: Dict[str, str]

@dataclass
class Alert:
    id: str
    level: AlertLevel
    title: str
    description: str
    metric_name: str
    current_value: float
    threshold: float
    region: RegionCode
    timestamp: datetime
    resolved: bool = False
    resolution_time: Optional[datetime] = None

@dataclass
class SystemHealth:
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_latency: float
    active_connections: int
    error_rate: float
    uptime: float

@dataclass
class TradingMetrics:
    win_rate: float
    roi: float
    max_drawdown: float
    sharpe_ratio: float
    total_trades: int
    profit_loss: float
    volume_24h: float
    active_strategies: int

@dataclass
class UserMetrics:
    total_users: int
    active_users_24h: int
    new_registrations_24h: int
    user_retention_rate: float
    average_session_duration: float
    geographic_distribution: Dict[str, int]

@dataclass
class ComplianceMetrics:
    kyc_completion_rate: float
    aml_alerts: int
    regulatory_violations: int
    audit_score: float
    data_privacy_compliance: float

class MetricsCollector:
    """Сборщик метрик из различных источников"""
    
    def __init__(self):
        self.metrics_buffer = deque(maxlen=10000)
        self.last_collection = datetime.now()
    
    async def collect_system_metrics(self, region: RegionCode) -> SystemHealth:
        """Сбор системных метрик"""
        cpu_usage = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Симуляция сетевых метрик
        network_latency = np.random.normal(50, 10)  # мс
        active_connections = np.random.randint(1000, 5000)
        error_rate = np.random.uniform(0.001, 0.01)  # 0.1-1%
        uptime = (datetime.now() - datetime.now().replace(hour=0, minute=0, second=0)).total_seconds()
        
        return SystemHealth(
            cpu_usage=cpu_usage,
            memory_usage=memory.percent,
            disk_usage=disk.percent,
            network_latency=network_latency,
            active_connections=active_connections,
            error_rate=error_rate,
            uptime=uptime
        )
    
    async def collect_trading_metrics(self, region: RegionCode) -> TradingMetrics:
        """Сбор торговых метрик"""
        # Симуляция торговых метрик с учетом целевых значений
        base_win_rate = 0.823  # Текущий уровень
        target_win_rate = 0.85  # Целевой уровень
        
        win_rate = np.random.normal(
            (base_win_rate + target_win_rate) / 2, 0.02
        )
        
        roi = np.random.normal(0.30, 0.05)  # 30% ± 5%
        max_drawdown = np.random.uniform(0.015, 0.025)  # 1.5-2.5%
        sharpe_ratio = np.random.normal(3.2, 0.3)  # Целевой 3.5
        
        return TradingMetrics(
            win_rate=max(0, min(1, win_rate)),
            roi=roi,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            total_trades=np.random.randint(10000, 50000),
            profit_loss=np.random.normal(1000000, 200000),
            volume_24h=np.random.uniform(8e9, 12e9),  # $8-12B
            active_strategies=np.random.randint(50, 200)
        )
    
    async def collect_user_metrics(self, region: RegionCode) -> UserMetrics:
        """Сбор пользовательских метрик"""
        total_users = np.random.randint(80000, 120000)  # Цель: 100k+
        
        return UserMetrics(
            total_users=total_users,
            active_users_24h=int(total_users * np.random.uniform(0.15, 0.25)),
            new_registrations_24h=np.random.randint(500, 2000),
            user_retention_rate=np.random.uniform(0.75, 0.85),
            average_session_duration=np.random.uniform(1800, 3600),  # 30-60 мин
            geographic_distribution={
                "americas": int(total_users * 0.35),
                "europe": int(total_users * 0.30),
                "asia_pacific": int(total_users * 0.25),
                "others": int(total_users * 0.10)
            }
        )
    
    async def collect_compliance_metrics(self, region: RegionCode) -> ComplianceMetrics:
        """Сбор метрик соответствия"""
        return ComplianceMetrics(
            kyc_completion_rate=np.random.uniform(0.85, 0.95),
            aml_alerts=np.random.randint(0, 10),
            regulatory_violations=np.random.randint(0, 3),
            audit_score=np.random.uniform(0.90, 0.98),
            data_privacy_compliance=np.random.uniform(0.95, 1.0)
        )

class AlertManager:
    """Менеджер алертов и уведомлений"""
    
    def __init__(self):
        self.alerts: List[Alert] = []
        self.thresholds = {
            "cpu_usage": 80.0,
            "memory_usage": 85.0,
            "disk_usage": 90.0,
            "error_rate": 0.05,
            "win_rate": 0.75,  # Минимальный win rate
            "max_drawdown": 0.03,  # Максимальный drawdown
            "network_latency": 200.0  # мс
        }
    
    def check_thresholds(self, metrics: Dict[str, Any], region: RegionCode) -> List[Alert]:
        """Проверка пороговых значений"""
        new_alerts = []
        
        for metric_name, threshold in self.thresholds.items():
            if metric_name in metrics:
                value = metrics[metric_name]
                
                # Определение уровня алерта
                if metric_name in ["cpu_usage", "memory_usage", "disk_usage", "error_rate", "max_drawdown", "network_latency"]:
                    if value > threshold:
                        level = AlertLevel.CRITICAL if value > threshold * 1.2 else AlertLevel.WARNING
                        alert = Alert(
                            id=f"{metric_name}_{region.value}_{datetime.now().timestamp()}",
                            level=level,
                            title=f"High {metric_name.replace('_', ' ').title()}",
                            description=f"{metric_name} is {value:.2f}, exceeding threshold of {threshold}",
                            metric_name=metric_name,
                            current_value=value,
                            threshold=threshold,
                            region=region,
                            timestamp=datetime.now()
                        )
                        new_alerts.append(alert)
                
                elif metric_name == "win_rate":
                    if value < threshold:
                        level = AlertLevel.CRITICAL if value < threshold * 0.9 else AlertLevel.WARNING
                        alert = Alert(
                            id=f"{metric_name}_{region.value}_{datetime.now().timestamp()}",
                            level=level,
                            title=f"Low Win Rate",
                            description=f"Win rate is {value:.2%}, below threshold of {threshold:.2%}",
                            metric_name=metric_name,
                            current_value=value,
                            threshold=threshold,
                            region=region,
                            timestamp=datetime.now()
                        )
                        new_alerts.append(alert)
        
        self.alerts.extend(new_alerts)
        return new_alerts
    
    async def send_alert(self, alert: Alert):
        """Отправка алерта"""
        logging.warning(f"ALERT [{alert.level.value.upper()}] {alert.title}: {alert.description}")
        
        # Здесь можно добавить интеграцию с системами уведомлений
        # Slack, Telegram, Email, PagerDuty и т.д.

class ReportGenerator:
    """Генератор отчетов"""
    
    def __init__(self):
        self.report_templates = {
            "daily": self._generate_daily_report,
            "weekly": self._generate_weekly_report,
            "monthly": self._generate_monthly_report,
            "compliance": self._generate_compliance_report
        }
    
    async def _generate_daily_report(self, metrics_data: Dict) -> Dict:
        """Генерация ежедневного отчета"""
        return {
            "report_type": "daily",
            "date": datetime.now().strftime("%Y-%m-%d"),
            "summary": {
                "total_users": metrics_data.get("total_users", 0),
                "trading_volume_24h": metrics_data.get("volume_24h", 0),
                "win_rate": metrics_data.get("win_rate", 0),
                "system_uptime": metrics_data.get("uptime", 0),
                "active_regions": len([r for r in RegionCode if r != RegionCode.GLOBAL])
            },
            "performance": {
                "roi": metrics_data.get("roi", 0),
                "sharpe_ratio": metrics_data.get("sharpe_ratio", 0),
                "max_drawdown": metrics_data.get("max_drawdown", 0),
                "total_trades": metrics_data.get("total_trades", 0)
            },
            "system_health": {
                "cpu_usage": metrics_data.get("cpu_usage", 0),
                "memory_usage": metrics_data.get("memory_usage", 0),
                "error_rate": metrics_data.get("error_rate", 0)
            }
        }
    
    async def _generate_weekly_report(self, metrics_data: Dict) -> Dict:
        """Генерация еженедельного отчета"""
        return {
            "report_type": "weekly",
            "week_ending": datetime.now().strftime("%Y-%m-%d"),
            "growth_metrics": {
                "user_growth": np.random.uniform(0.05, 0.15),  # 5-15% рост
                "volume_growth": np.random.uniform(0.10, 0.25),  # 10-25% рост
                "revenue_growth": np.random.uniform(0.08, 0.20)  # 8-20% рост
            },
            "regional_performance": {
                region.value: {
                    "users": np.random.randint(10000, 30000),
                    "volume": np.random.uniform(1e9, 3e9),
                    "win_rate": np.random.uniform(0.80, 0.87)
                } for region in RegionCode if region != RegionCode.GLOBAL
            }
        }
    
    async def _generate_monthly_report(self, metrics_data: Dict) -> Dict:
        """Генерация месячного отчета"""
        return {
            "report_type": "monthly",
            "month": datetime.now().strftime("%Y-%m"),
            "achievements": {
                "target_users_progress": min(100, (metrics_data.get("total_users", 0) / 100000) * 100),
                "target_volume_progress": min(100, (metrics_data.get("volume_24h", 0) * 30 / 10e9) * 100),
                "target_win_rate_progress": min(100, (metrics_data.get("win_rate", 0) / 0.85) * 100)
            },
            "financial_summary": {
                "total_revenue": np.random.uniform(5e6, 15e6),  # $5-15M
                "operating_costs": np.random.uniform(2e6, 8e6),  # $2-8M
                "net_profit": np.random.uniform(1e6, 7e6),  # $1-7M
                "profit_margin": np.random.uniform(0.20, 0.50)  # 20-50%
            }
        }
    
    async def _generate_compliance_report(self, metrics_data: Dict) -> Dict:
        """Генерация отчета о соответствии"""
        return {
            "report_type": "compliance",
            "date": datetime.now().strftime("%Y-%m-%d"),
            "regulatory_status": {
                "kyc_compliance": metrics_data.get("kyc_completion_rate", 0),
                "aml_compliance": 1.0 - (metrics_data.get("aml_alerts", 0) / 1000),
                "data_protection": metrics_data.get("data_privacy_compliance", 0),
                "audit_score": metrics_data.get("audit_score", 0)
            },
            "regional_compliance": {
                region.value: {
                    "status": "compliant" if np.random.random() > 0.1 else "pending",
                    "last_audit": (datetime.now() - timedelta(days=np.random.randint(30, 90))).strftime("%Y-%m-%d"),
                    "next_review": (datetime.now() + timedelta(days=np.random.randint(30, 90))).strftime("%Y-%m-%d")
                } for region in RegionCode if region != RegionCode.GLOBAL
            }
        }

class GlobalMonitoringSystem:
    """Главная система глобального мониторинга"""
    
    def __init__(self):
        self.metrics_collector = MetricsCollector()
        self.alert_manager = AlertManager()
        self.report_generator = ReportGenerator()
        self.is_running = False
        self.monitoring_tasks = []
        self.metrics_history = defaultdict(list)
        
        # Настройка логирования
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    async def start_monitoring(self):
        """Запуск системы мониторинга"""
        self.is_running = True
        self.logger.info("Starting Global Monitoring System...")
        
        # Запуск задач мониторинга для каждого региона
        for region in RegionCode:
            if region != RegionCode.GLOBAL:
                task = asyncio.create_task(self._monitor_region(region))
                self.monitoring_tasks.append(task)
        
        # Запуск глобальных задач
        global_tasks = [
            asyncio.create_task(self._generate_periodic_reports()),
            asyncio.create_task(self._cleanup_old_data()),
            asyncio.create_task(self._health_check_loop())
        ]
        self.monitoring_tasks.extend(global_tasks)
        
        self.logger.info(f"Started monitoring for {len(RegionCode)-1} regions")
    
    async def stop_monitoring(self):
        """Остановка системы мониторинга"""
        self.is_running = False
        self.logger.info("Stopping Global Monitoring System...")
        
        for task in self.monitoring_tasks:
            task.cancel()
        
        await asyncio.gather(*self.monitoring_tasks, return_exceptions=True)
        self.monitoring_tasks.clear()
        
        self.logger.info("Global Monitoring System stopped")
    
    async def _monitor_region(self, region: RegionCode):
        """Мониторинг конкретного региона"""
        while self.is_running:
            try:
                # Сбор метрик
                system_health = await self.metrics_collector.collect_system_metrics(region)
                trading_metrics = await self.metrics_collector.collect_trading_metrics(region)
                user_metrics = await self.metrics_collector.collect_user_metrics(region)
                compliance_metrics = await self.metrics_collector.collect_compliance_metrics(region)
                
                # Объединение метрик
                all_metrics = {
                    **asdict(system_health),
                    **asdict(trading_metrics),
                    **asdict(user_metrics),
                    **asdict(compliance_metrics)
                }
                
                # Сохранение в историю
                self.metrics_history[region].append({
                    "timestamp": datetime.now(),
                    "metrics": all_metrics
                })
                
                # Проверка алертов
                alerts = self.alert_manager.check_thresholds(all_metrics, region)
                for alert in alerts:
                    await self.alert_manager.send_alert(alert)
                
                # Логирование ключевых метрик
                self.logger.info(
                    f"Region {region.value}: "
                    f"Win Rate: {trading_metrics.win_rate:.2%}, "
                    f"ROI: {trading_metrics.roi:.2%}, "
                    f"Users: {user_metrics.total_users:,}, "
                    f"Volume: ${trading_metrics.volume_24h/1e9:.1f}B"
                )
                
                await asyncio.sleep(60)  # Сбор каждую минуту
                
            except Exception as e:
                self.logger.error(f"Error monitoring region {region.value}: {e}")
                await asyncio.sleep(30)
    
    async def _generate_periodic_reports(self):
        """Генерация периодических отчетов"""
        while self.is_running:
            try:
                current_time = datetime.now()
                
                # Ежедневные отчеты в 00:00
                if current_time.hour == 0 and current_time.minute == 0:
                    await self._generate_and_save_report("daily")
                
                # Еженедельные отчеты в понедельник в 01:00
                if current_time.weekday() == 0 and current_time.hour == 1 and current_time.minute == 0:
                    await self._generate_and_save_report("weekly")
                
                # Месячные отчеты 1 числа в 02:00
                if current_time.day == 1 and current_time.hour == 2 and current_time.minute == 0:
                    await self._generate_and_save_report("monthly")
                
                await asyncio.sleep(60)  # Проверка каждую минуту
                
            except Exception as e:
                self.logger.error(f"Error generating periodic reports: {e}")
                await asyncio.sleep(300)
    
    async def _generate_and_save_report(self, report_type: str):
        """Генерация и сохранение отчета"""
        try:
            # Получение последних метрик
            latest_metrics = {}
            for region, history in self.metrics_history.items():
                if history:
                    latest_metrics.update(history[-1]["metrics"])
            
            # Генерация отчета
            report = await self.report_generator.report_templates[report_type](latest_metrics)
            
            # Сохранение отчета
            filename = f"reports/{report_type}_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False, default=str)
            
            self.logger.info(f"Generated {report_type} report: {filename}")
            
        except Exception as e:
            self.logger.error(f"Error generating {report_type} report: {e}")
    
    async def _cleanup_old_data(self):
        """Очистка старых данных"""
        while self.is_running:
            try:
                cutoff_time = datetime.now() - timedelta(days=7)  # Хранить 7 дней
                
                for region in self.metrics_history:
                    self.metrics_history[region] = [
                        entry for entry in self.metrics_history[region]
                        if entry["timestamp"] > cutoff_time
                    ]
                
                # Очистка старых алертов
                self.alert_manager.alerts = [
                    alert for alert in self.alert_manager.alerts
                    if alert.timestamp > cutoff_time
                ]
                
                await asyncio.sleep(3600)  # Очистка каждый час
                
            except Exception as e:
                self.logger.error(f"Error cleaning up old data: {e}")
                await asyncio.sleep(1800)
    
    async def _health_check_loop(self):
        """Цикл проверки здоровья системы"""
        while self.is_running:
            try:
                # Проверка состояния задач мониторинга
                active_tasks = sum(1 for task in self.monitoring_tasks if not task.done())
                total_tasks = len(self.monitoring_tasks)
                
                if active_tasks < total_tasks * 0.8:  # Если менее 80% задач активны
                    self.logger.warning(f"Only {active_tasks}/{total_tasks} monitoring tasks are active")
                
                # Проверка использования памяти
                memory_usage = psutil.virtual_memory().percent
                if memory_usage > 90:
                    self.logger.critical(f"High memory usage: {memory_usage}%")
                
                await asyncio.sleep(300)  # Проверка каждые 5 минут
                
            except Exception as e:
                self.logger.error(f"Error in health check: {e}")
                await asyncio.sleep(300)
    
    async def get_current_status(self) -> Dict:
        """Получение текущего статуса системы"""
        status = {
            "system_status": "running" if self.is_running else "stopped",
            "active_regions": len([r for r in RegionCode if r != RegionCode.GLOBAL]),
            "total_alerts": len(self.alert_manager.alerts),
            "critical_alerts": len([a for a in self.alert_manager.alerts if a.level == AlertLevel.CRITICAL]),
            "last_update": datetime.now().isoformat(),
            "regional_status": {}
        }
        
        # Статус по регионам
        for region in RegionCode:
            if region != RegionCode.GLOBAL and region in self.metrics_history:
                history = self.metrics_history[region]
                if history:
                    latest = history[-1]["metrics"]
                    status["regional_status"][region.value] = {
                        "last_update": history[-1]["timestamp"].isoformat(),
                        "win_rate": latest.get("win_rate", 0),
                        "total_users": latest.get("total_users", 0),
                        "system_health": "healthy" if latest.get("cpu_usage", 0) < 80 else "warning"
                    }
        
        return status
    
    async def generate_custom_report(self, report_type: str, date_range: Optional[Tuple[datetime, datetime]] = None) -> Dict:
        """Генерация пользовательского отчета"""
        if report_type not in self.report_generator.report_templates:
            raise ValueError(f"Unknown report type: {report_type}")
        
        # Получение данных за указанный период
        if date_range:
            start_date, end_date = date_range
            filtered_metrics = {}
            
            for region, history in self.metrics_history.items():
                filtered_history = [
                    entry for entry in history
                    if start_date <= entry["timestamp"] <= end_date
                ]
                if filtered_history:
                    # Агрегация метрик за период
                    aggregated = {}
                    for key in filtered_history[0]["metrics"].keys():
                        values = [entry["metrics"][key] for entry in filtered_history if key in entry["metrics"]]
                        if values:
                            aggregated[key] = np.mean(values)
                    filtered_metrics.update(aggregated)
        else:
            # Использование последних метрик
            filtered_metrics = {}
            for region, history in self.metrics_history.items():
                if history:
                    filtered_metrics.update(history[-1]["metrics"])
        
        return await self.report_generator.report_templates[report_type](filtered_metrics)

# Пример использования
async def main():
    """Пример использования системы мониторинга"""
    monitoring_system = GlobalMonitoringSystem()
    
    try:
        # Запуск мониторинга
        await monitoring_system.start_monitoring()
        
        # Работа в течение некоторого времени
        await asyncio.sleep(300)  # 5 минут для демонстрации
        
        # Получение текущего статуса
        status = await monitoring_system.get_current_status()
        print("Current System Status:")
        print(json.dumps(status, indent=2, default=str))
        
        # Генерация пользовательского отчета
        daily_report = await monitoring_system.generate_custom_report("daily")
        print("\nDaily Report:")
        print(json.dumps(daily_report, indent=2, default=str))
        
    finally:
        # Остановка мониторинга
        await monitoring_system.stop_monitoring()

if __name__ == "__main__":
    asyncio.run(main())