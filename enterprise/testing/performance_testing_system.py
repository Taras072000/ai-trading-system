"""
Enterprise Performance Testing System - Система тестирования производительности
Обеспечивает комплексное тестирование производительности и достижение целевых метрик
"""

import asyncio
import json
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import logging
import redis.asyncio as redis
from collections import deque, defaultdict
import statistics
import concurrent.futures
import threading
import multiprocessing
import psutil
import gc
import tracemalloc

# Тестирование
import pytest
import unittest
from unittest.mock import Mock, patch
import requests
import websocket
import aiohttp
import asyncio_mqtt

# Метрики и мониторинг
from prometheus_client import Counter, Histogram, Gauge, Summary
import matplotlib.pyplot as plt
import seaborn as sns

# Математические библиотеки
from scipy import stats
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class TestType(Enum):
    """Типы тестов"""
    UNIT = "unit"
    INTEGRATION = "integration"
    LOAD = "load"
    STRESS = "stress"
    PERFORMANCE = "performance"
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    SCALABILITY = "scalability"
    RELIABILITY = "reliability"
    SECURITY = "security"

class MetricType(Enum):
    """Типы метрик"""
    WIN_RATE = "win_rate"
    ROI = "roi"
    MAX_DRAWDOWN = "max_drawdown"
    SHARPE_RATIO = "sharpe_ratio"
    UPTIME = "uptime"
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    MEMORY_USAGE = "memory_usage"
    CPU_USAGE = "cpu_usage"
    ERROR_RATE = "error_rate"

class TestStatus(Enum):
    """Статусы тестов"""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"

@dataclass
class PerformanceMetric:
    """Метрика производительности"""
    name: str
    value: float
    target: float
    unit: str
    timestamp: datetime
    test_id: str
    passed: bool = False
    
    def __post_init__(self):
        self.passed = self.value >= self.target if self.name in ['win_rate', 'roi', 'sharpe_ratio', 'uptime', 'throughput'] else self.value <= self.target

@dataclass
class TestResult:
    """Результат теста"""
    test_id: str
    test_type: TestType
    test_name: str
    status: TestStatus
    duration: float
    metrics: List[PerformanceMetric]
    error_message: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

@dataclass
class LoadTestConfig:
    """Конфигурация нагрузочного тестирования"""
    concurrent_users: int = 100
    duration_seconds: int = 300
    ramp_up_seconds: int = 60
    target_rps: int = 1000
    max_response_time: float = 0.1
    endpoints: List[str] = None
    
    def __post_init__(self):
        if self.endpoints is None:
            self.endpoints = ['/api/v1/trading/order', '/api/v1/portfolio/balance', '/api/v1/market/data']

@dataclass
class TradingPerformanceMetrics:
    """Метрики торговой производительности"""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    win_rate: float = 0.0
    roi: float = 0.0
    avg_trade_duration: float = 0.0
    max_trade_duration: float = 0.0
    
    def calculate_metrics(self, trades: List[Dict[str, Any]]):
        """Расчет метрик на основе сделок"""
        if not trades:
            return
            
        self.total_trades = len(trades)
        
        pnl_values = [trade.get('pnl', 0) for trade in trades]
        self.total_pnl = sum(pnl_values)
        
        self.winning_trades = len([pnl for pnl in pnl_values if pnl > 0])
        self.losing_trades = len([pnl for pnl in pnl_values if pnl < 0])
        
        self.win_rate = (self.winning_trades / self.total_trades) * 100 if self.total_trades > 0 else 0
        
        # ROI (предполагаем начальный капитал 10000)
        initial_capital = 10000
        self.roi = (self.total_pnl / initial_capital) * 100
        
        # Max Drawdown
        cumulative_pnl = np.cumsum(pnl_values)
        running_max = np.maximum.accumulate(cumulative_pnl)
        drawdown = (cumulative_pnl - running_max) / running_max * 100
        self.max_drawdown = abs(np.min(drawdown)) if len(drawdown) > 0 else 0
        
        # Sharpe Ratio
        if len(pnl_values) > 1:
            returns = np.array(pnl_values) / initial_capital
            self.sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        
        # Длительность сделок
        durations = [trade.get('duration', 0) for trade in trades]
        self.avg_trade_duration = np.mean(durations) if durations else 0
        self.max_trade_duration = np.max(durations) if durations else 0

# Метрики Prometheus
TEST_COUNTER = Counter('tests_total', 'Total tests executed', ['test_type', 'status'])
TEST_DURATION = Histogram('test_duration_seconds', 'Test execution time', ['test_type'])
METRIC_GAUGE = Gauge('performance_metric', 'Performance metric value', ['metric_name', 'test_id'])
SYSTEM_METRICS = Gauge('system_metric', 'System performance metric', ['metric_type'])

class SystemMonitor:
    """Монитор системных ресурсов"""
    
    def __init__(self):
        self.cpu_usage = []
        self.memory_usage = []
        self.disk_usage = []
        self.network_io = []
        self.monitoring = False
        
    def start_monitoring(self):
        """Запуск мониторинга"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.start()
        
    def stop_monitoring(self):
        """Остановка мониторинга"""
        self.monitoring = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join()
            
    def _monitor_loop(self):
        """Цикл мониторинга"""
        while self.monitoring:
            try:
                # CPU
                cpu_percent = psutil.cpu_percent(interval=1)
                self.cpu_usage.append(cpu_percent)
                SYSTEM_METRICS.labels(metric_type='cpu_usage').set(cpu_percent)
                
                # Memory
                memory = psutil.virtual_memory()
                memory_percent = memory.percent
                self.memory_usage.append(memory_percent)
                SYSTEM_METRICS.labels(metric_type='memory_usage').set(memory_percent)
                
                # Disk
                disk = psutil.disk_usage('/')
                disk_percent = disk.percent
                self.disk_usage.append(disk_percent)
                SYSTEM_METRICS.labels(metric_type='disk_usage').set(disk_percent)
                
                # Network
                network = psutil.net_io_counters()
                self.network_io.append({
                    'bytes_sent': network.bytes_sent,
                    'bytes_recv': network.bytes_recv,
                    'timestamp': time.time()
                })
                
                time.sleep(1)
                
            except Exception as e:
                print(f"Monitoring error: {e}")
                
    def get_stats(self) -> Dict[str, Any]:
        """Получение статистики"""
        return {
            'cpu': {
                'avg': np.mean(self.cpu_usage) if self.cpu_usage else 0,
                'max': np.max(self.cpu_usage) if self.cpu_usage else 0,
                'min': np.min(self.cpu_usage) if self.cpu_usage else 0
            },
            'memory': {
                'avg': np.mean(self.memory_usage) if self.memory_usage else 0,
                'max': np.max(self.memory_usage) if self.memory_usage else 0,
                'min': np.min(self.memory_usage) if self.memory_usage else 0
            },
            'disk': {
                'avg': np.mean(self.disk_usage) if self.disk_usage else 0,
                'max': np.max(self.disk_usage) if self.disk_usage else 0,
                'min': np.min(self.disk_usage) if self.disk_usage else 0
            }
        }

class LoadTester:
    """Нагрузочное тестирование"""
    
    def __init__(self, config: LoadTestConfig):
        self.config = config
        self.results = []
        self.session = None
        
    async def run_load_test(self, base_url: str) -> List[TestResult]:
        """Запуск нагрузочного тестирования"""
        results = []
        
        # Создание сессии
        connector = aiohttp.TCPConnector(limit=self.config.concurrent_users * 2)
        timeout = aiohttp.ClientTimeout(total=30)
        self.session = aiohttp.ClientSession(connector=connector, timeout=timeout)
        
        try:
            # Тест латентности
            latency_result = await self._test_latency(base_url)
            results.append(latency_result)
            
            # Тест пропускной способности
            throughput_result = await self._test_throughput(base_url)
            results.append(throughput_result)
            
            # Стресс-тест
            stress_result = await self._test_stress(base_url)
            results.append(stress_result)
            
        finally:
            await self.session.close()
            
        return results
        
    async def _test_latency(self, base_url: str) -> TestResult:
        """Тест латентности"""
        start_time = time.time()
        latencies = []
        errors = 0
        
        # Выполнение запросов
        for _ in range(100):
            for endpoint in self.config.endpoints:
                try:
                    request_start = time.time()
                    async with self.session.get(f"{base_url}{endpoint}") as response:
                        await response.read()
                        latency = (time.time() - request_start) * 1000  # мс
                        latencies.append(latency)
                        
                except Exception:
                    errors += 1
                    
        duration = time.time() - start_time
        
        # Расчет метрик
        avg_latency = np.mean(latencies) if latencies else 0
        p95_latency = np.percentile(latencies, 95) if latencies else 0
        p99_latency = np.percentile(latencies, 99) if latencies else 0
        
        metrics = [
            PerformanceMetric(
                name="avg_latency",
                value=avg_latency,
                target=self.config.max_response_time * 1000,
                unit="ms",
                timestamp=datetime.now(),
                test_id="latency_test"
            ),
            PerformanceMetric(
                name="p95_latency",
                value=p95_latency,
                target=self.config.max_response_time * 1000 * 2,
                unit="ms",
                timestamp=datetime.now(),
                test_id="latency_test"
            ),
            PerformanceMetric(
                name="error_rate",
                value=(errors / (len(self.config.endpoints) * 100)) * 100,
                target=1.0,  # Максимум 1% ошибок
                unit="%",
                timestamp=datetime.now(),
                test_id="latency_test"
            )
        ]
        
        status = TestStatus.PASSED if all(m.passed for m in metrics) else TestStatus.FAILED
        
        return TestResult(
            test_id="latency_test",
            test_type=TestType.LATENCY,
            test_name="API Latency Test",
            status=status,
            duration=duration,
            metrics=metrics,
            details={
                'total_requests': len(latencies),
                'errors': errors,
                'latencies': latencies[-10:]  # Последние 10 для примера
            }
        )
        
    async def _test_throughput(self, base_url: str) -> TestResult:
        """Тест пропускной способности"""
        start_time = time.time()
        successful_requests = 0
        failed_requests = 0
        
        # Создание задач
        tasks = []
        for _ in range(self.config.concurrent_users):
            task = asyncio.create_task(self._throughput_worker(base_url))
            tasks.append(task)
            
        # Ожидание завершения
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Подсчет результатов
        for result in results:
            if isinstance(result, Exception):
                failed_requests += 1
            else:
                successful_requests += result.get('successful', 0)
                failed_requests += result.get('failed', 0)
                
        duration = time.time() - start_time
        total_requests = successful_requests + failed_requests
        rps = total_requests / duration if duration > 0 else 0
        
        metrics = [
            PerformanceMetric(
                name="throughput",
                value=rps,
                target=self.config.target_rps,
                unit="rps",
                timestamp=datetime.now(),
                test_id="throughput_test"
            ),
            PerformanceMetric(
                name="success_rate",
                value=(successful_requests / total_requests) * 100 if total_requests > 0 else 0,
                target=99.0,  # Минимум 99% успешных запросов
                unit="%",
                timestamp=datetime.now(),
                test_id="throughput_test"
            )
        ]
        
        status = TestStatus.PASSED if all(m.passed for m in metrics) else TestStatus.FAILED
        
        return TestResult(
            test_id="throughput_test",
            test_type=TestType.THROUGHPUT,
            test_name="API Throughput Test",
            status=status,
            duration=duration,
            metrics=metrics,
            details={
                'total_requests': total_requests,
                'successful_requests': successful_requests,
                'failed_requests': failed_requests,
                'concurrent_users': self.config.concurrent_users
            }
        )
        
    async def _throughput_worker(self, base_url: str) -> Dict[str, int]:
        """Воркер для теста пропускной способности"""
        successful = 0
        failed = 0
        
        end_time = time.time() + self.config.duration_seconds
        
        while time.time() < end_time:
            for endpoint in self.config.endpoints:
                try:
                    async with self.session.get(f"{base_url}{endpoint}") as response:
                        if response.status == 200:
                            successful += 1
                        else:
                            failed += 1
                except Exception:
                    failed += 1
                    
                # Небольшая пауза для контроля нагрузки
                await asyncio.sleep(0.01)
                
        return {'successful': successful, 'failed': failed}
        
    async def _test_stress(self, base_url: str) -> TestResult:
        """Стресс-тест"""
        start_time = time.time()
        
        # Постепенное увеличение нагрузки
        stress_levels = [50, 100, 200, 500, 1000]
        results = []
        
        for level in stress_levels:
            level_start = time.time()
            successful = 0
            failed = 0
            
            # Создание задач для текущего уровня
            tasks = []
            for _ in range(level):
                task = asyncio.create_task(self._stress_worker(base_url, 30))  # 30 секунд на уровень
                tasks.append(task)
                
            # Ожидание завершения
            level_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in level_results:
                if isinstance(result, Exception):
                    failed += 1
                else:
                    successful += result.get('successful', 0)
                    failed += result.get('failed', 0)
                    
            level_duration = time.time() - level_start
            level_rps = (successful + failed) / level_duration if level_duration > 0 else 0
            
            results.append({
                'level': level,
                'rps': level_rps,
                'success_rate': (successful / (successful + failed)) * 100 if (successful + failed) > 0 else 0,
                'duration': level_duration
            })
            
        duration = time.time() - start_time
        
        # Анализ результатов
        max_rps = max([r['rps'] for r in results])
        min_success_rate = min([r['success_rate'] for r in results])
        
        metrics = [
            PerformanceMetric(
                name="max_throughput",
                value=max_rps,
                target=self.config.target_rps * 0.8,  # 80% от целевого RPS
                unit="rps",
                timestamp=datetime.now(),
                test_id="stress_test"
            ),
            PerformanceMetric(
                name="min_success_rate",
                value=min_success_rate,
                target=95.0,  # Минимум 95% даже под нагрузкой
                unit="%",
                timestamp=datetime.now(),
                test_id="stress_test"
            )
        ]
        
        status = TestStatus.PASSED if all(m.passed for m in metrics) else TestStatus.FAILED
        
        return TestResult(
            test_id="stress_test",
            test_type=TestType.STRESS,
            test_name="API Stress Test",
            status=status,
            duration=duration,
            metrics=metrics,
            details={
                'stress_levels': stress_levels,
                'results': results
            }
        )
        
    async def _stress_worker(self, base_url: str, duration: int) -> Dict[str, int]:
        """Воркер для стресс-теста"""
        successful = 0
        failed = 0
        
        end_time = time.time() + duration
        
        while time.time() < end_time:
            endpoint = np.random.choice(self.config.endpoints)
            try:
                async with self.session.get(f"{base_url}{endpoint}") as response:
                    if response.status == 200:
                        successful += 1
                    else:
                        failed += 1
            except Exception:
                failed += 1
                
        return {'successful': successful, 'failed': failed}

class TradingPerformanceTester:
    """Тестер торговой производительности"""
    
    def __init__(self):
        self.trades_history = []
        
    def generate_test_trades(self, count: int = 1000) -> List[Dict[str, Any]]:
        """Генерация тестовых сделок"""
        trades = []
        
        for i in range(count):
            # Симуляция торговой стратегии с заданными параметрами
            win_probability = 0.65  # 65% выигрышных сделок
            
            if np.random.random() < win_probability:
                # Выигрышная сделка
                pnl = np.random.uniform(10, 100)
            else:
                # Проигрышная сделка
                pnl = -np.random.uniform(5, 80)
                
            trade = {
                'id': f"trade_{i}",
                'symbol': np.random.choice(['BTCUSDT', 'ETHUSDT', 'ADAUSDT']),
                'side': np.random.choice(['buy', 'sell']),
                'quantity': np.random.uniform(0.1, 10),
                'price': np.random.uniform(1000, 50000),
                'pnl': pnl,
                'duration': np.random.uniform(60, 3600),  # секунды
                'timestamp': datetime.now() - timedelta(seconds=i*60)
            }
            
            trades.append(trade)
            
        return trades
        
    def test_trading_performance(self, trades: List[Dict[str, Any]]) -> TestResult:
        """Тестирование торговой производительности"""
        start_time = time.time()
        
        # Расчет метрик
        metrics_calculator = TradingPerformanceMetrics()
        metrics_calculator.calculate_metrics(trades)
        
        # Создание метрик производительности
        metrics = [
            PerformanceMetric(
                name="win_rate",
                value=metrics_calculator.win_rate,
                target=80.0,  # Целевой Win Rate ≥ 80%
                unit="%",
                timestamp=datetime.now(),
                test_id="trading_performance"
            ),
            PerformanceMetric(
                name="roi",
                value=metrics_calculator.roi,
                target=20.0,  # Целевой ROI ≥ 20% годовых
                unit="%",
                timestamp=datetime.now(),
                test_id="trading_performance"
            ),
            PerformanceMetric(
                name="max_drawdown",
                value=metrics_calculator.max_drawdown,
                target=3.0,  # Целевой Max Drawdown ≤ 3%
                unit="%",
                timestamp=datetime.now(),
                test_id="trading_performance"
            ),
            PerformanceMetric(
                name="sharpe_ratio",
                value=metrics_calculator.sharpe_ratio,
                target=2.5,  # Целевой Sharpe Ratio ≥ 2.5
                unit="ratio",
                timestamp=datetime.now(),
                test_id="trading_performance"
            )
        ]
        
        duration = time.time() - start_time
        status = TestStatus.PASSED if all(m.passed for m in metrics) else TestStatus.FAILED
        
        return TestResult(
            test_id="trading_performance",
            test_type=TestType.PERFORMANCE,
            test_name="Trading Performance Test",
            status=status,
            duration=duration,
            metrics=metrics,
            details={
                'total_trades': metrics_calculator.total_trades,
                'winning_trades': metrics_calculator.winning_trades,
                'losing_trades': metrics_calculator.losing_trades,
                'total_pnl': metrics_calculator.total_pnl,
                'avg_trade_duration': metrics_calculator.avg_trade_duration
            }
        )

class ReliabilityTester:
    """Тестер надежности"""
    
    def __init__(self):
        self.uptime_start = time.time()
        self.downtime_periods = []
        
    def test_uptime(self, monitoring_duration: int = 3600) -> TestResult:
        """Тест времени безотказной работы"""
        start_time = time.time()
        
        # Симуляция мониторинга uptime
        total_time = monitoring_duration
        downtime = sum([period['duration'] for period in self.downtime_periods])
        uptime_percentage = ((total_time - downtime) / total_time) * 100
        
        metrics = [
            PerformanceMetric(
                name="uptime",
                value=uptime_percentage,
                target=99.9,  # Целевой Uptime ≥ 99.9%
                unit="%",
                timestamp=datetime.now(),
                test_id="uptime_test"
            )
        ]
        
        duration = time.time() - start_time
        status = TestStatus.PASSED if all(m.passed for m in metrics) else TestStatus.FAILED
        
        return TestResult(
            test_id="uptime_test",
            test_type=TestType.RELIABILITY,
            test_name="System Uptime Test",
            status=status,
            duration=duration,
            metrics=metrics,
            details={
                'monitoring_duration': monitoring_duration,
                'downtime_periods': len(self.downtime_periods),
                'total_downtime': downtime
            }
        )
        
    def simulate_downtime(self, duration: float, reason: str = "maintenance"):
        """Симуляция простоя"""
        self.downtime_periods.append({
            'start': time.time(),
            'duration': duration,
            'reason': reason
        })

class EnterprisePerformanceTestingSystem:
    """Enterprise система тестирования производительности"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.redis_client = None
        self.logger = self._setup_logging()
        
        # Компоненты тестирования
        self.system_monitor = SystemMonitor()
        self.load_tester = LoadTester(LoadTestConfig(**config.get('load_test', {})))
        self.trading_tester = TradingPerformanceTester()
        self.reliability_tester = ReliabilityTester()
        
        # Результаты тестов
        self.test_results: List[TestResult] = []
        self.performance_history: Dict[str, List[float]] = defaultdict(list)
        
        # Целевые метрики
        self.target_metrics = {
            'win_rate': 80.0,
            'roi': 20.0,
            'max_drawdown': 3.0,
            'sharpe_ratio': 2.5,
            'uptime': 99.9,
            'latency': 10.0,  # мс
            'throughput': 1000  # rps
        }
        
    def _setup_logging(self) -> logging.Logger:
        """Настройка логирования"""
        logger = logging.getLogger('enterprise_performance_testing')
        logger.setLevel(logging.INFO)
        
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
        
    async def start(self):
        """Запуск системы тестирования"""
        # Подключение к Redis
        self.redis_client = redis.Redis(
            host=self.config.get('redis_host', 'localhost'),
            port=self.config.get('redis_port', 6379),
            decode_responses=True
        )
        
        # Запуск мониторинга системы
        self.system_monitor.start_monitoring()
        
        # Запуск фоновых задач
        asyncio.create_task(self._continuous_testing())
        asyncio.create_task(self._metrics_collector())
        asyncio.create_task(self._report_generator())
        
        self.logger.info("Enterprise Performance Testing System started")
        
    async def stop(self):
        """Остановка системы"""
        self.system_monitor.stop_monitoring()
        
        if self.redis_client:
            await self.redis_client.close()
            
    async def run_full_test_suite(self) -> Dict[str, Any]:
        """Запуск полного набора тестов"""
        self.logger.info("Starting full test suite")
        
        all_results = []
        
        try:
            # 1. Тестирование торговой производительности
            self.logger.info("Running trading performance tests")
            trading_results = await self._run_trading_tests()
            all_results.extend(trading_results)
            
            # 2. Нагрузочное тестирование
            self.logger.info("Running load tests")
            load_results = await self._run_load_tests()
            all_results.extend(load_results)
            
            # 3. Тестирование надежности
            self.logger.info("Running reliability tests")
            reliability_results = await self._run_reliability_tests()
            all_results.extend(reliability_results)
            
            # 4. Системное тестирование
            self.logger.info("Running system tests")
            system_results = await self._run_system_tests()
            all_results.extend(system_results)
            
            # Сохранение результатов
            self.test_results.extend(all_results)
            
            # Анализ результатов
            summary = self._analyze_results(all_results)
            
            self.logger.info(f"Test suite completed. Total tests: {len(all_results)}")
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Test suite error: {e}")
            raise
            
    async def _run_trading_tests(self) -> List[TestResult]:
        """Запуск тестов торговой производительности"""
        results = []
        
        # Генерация тестовых данных
        test_trades = self.trading_tester.generate_test_trades(1000)
        
        # Тест производительности торговли
        trading_result = self.trading_tester.test_trading_performance(test_trades)
        results.append(trading_result)
        
        # Обновление метрик
        for metric in trading_result.metrics:
            METRIC_GAUGE.labels(metric_name=metric.name, test_id=trading_result.test_id).set(metric.value)
            
        TEST_COUNTER.labels(test_type=trading_result.test_type.value, status=trading_result.status.value).inc()
        TEST_DURATION.labels(test_type=trading_result.test_type.value).observe(trading_result.duration)
        
        return results
        
    async def _run_load_tests(self) -> List[TestResult]:
        """Запуск нагрузочных тестов"""
        results = []
        
        base_url = self.config.get('test_base_url', 'http://localhost:8000')
        
        try:
            # Запуск нагрузочных тестов
            load_results = await self.load_tester.run_load_test(base_url)
            results.extend(load_results)
            
            # Обновление метрик
            for result in load_results:
                for metric in result.metrics:
                    METRIC_GAUGE.labels(metric_name=metric.name, test_id=result.test_id).set(metric.value)
                    
                TEST_COUNTER.labels(test_type=result.test_type.value, status=result.status.value).inc()
                TEST_DURATION.labels(test_type=result.test_type.value).observe(result.duration)
                
        except Exception as e:
            self.logger.error(f"Load test error: {e}")
            
            # Создание результата с ошибкой
            error_result = TestResult(
                test_id="load_test_error",
                test_type=TestType.LOAD,
                test_name="Load Test Error",
                status=TestStatus.ERROR,
                duration=0,
                metrics=[],
                error_message=str(e)
            )
            results.append(error_result)
            
        return results
        
    async def _run_reliability_tests(self) -> List[TestResult]:
        """Запуск тестов надежности"""
        results = []
        
        # Тест uptime
        uptime_result = self.reliability_tester.test_uptime(3600)  # 1 час мониторинга
        results.append(uptime_result)
        
        # Обновление метрик
        for metric in uptime_result.metrics:
            METRIC_GAUGE.labels(metric_name=metric.name, test_id=uptime_result.test_id).set(metric.value)
            
        TEST_COUNTER.labels(test_type=uptime_result.test_type.value, status=uptime_result.status.value).inc()
        TEST_DURATION.labels(test_type=uptime_result.test_type.value).observe(uptime_result.duration)
        
        return results
        
    async def _run_system_tests(self) -> List[TestResult]:
        """Запуск системных тестов"""
        results = []
        
        # Получение статистики системы
        system_stats = self.system_monitor.get_stats()
        
        # Тест использования CPU
        cpu_metric = PerformanceMetric(
            name="cpu_usage",
            value=system_stats['cpu']['avg'],
            target=80.0,  # Максимум 80% CPU
            unit="%",
            timestamp=datetime.now(),
            test_id="system_test"
        )
        
        # Тест использования памяти
        memory_metric = PerformanceMetric(
            name="memory_usage",
            value=system_stats['memory']['avg'],
            target=85.0,  # Максимум 85% памяти
            unit="%",
            timestamp=datetime.now(),
            test_id="system_test"
        )
        
        metrics = [cpu_metric, memory_metric]
        status = TestStatus.PASSED if all(m.passed for m in metrics) else TestStatus.FAILED
        
        system_result = TestResult(
            test_id="system_test",
            test_type=TestType.PERFORMANCE,
            test_name="System Resource Test",
            status=status,
            duration=1.0,
            metrics=metrics,
            details=system_stats
        )
        
        results.append(system_result)
        
        # Обновление метрик
        for metric in metrics:
            METRIC_GAUGE.labels(metric_name=metric.name, test_id=system_result.test_id).set(metric.value)
            
        TEST_COUNTER.labels(test_type=system_result.test_type.value, status=system_result.status.value).inc()
        
        return results
        
    def _analyze_results(self, results: List[TestResult]) -> Dict[str, Any]:
        """Анализ результатов тестирования"""
        total_tests = len(results)
        passed_tests = len([r for r in results if r.status == TestStatus.PASSED])
        failed_tests = len([r for r in results if r.status == TestStatus.FAILED])
        error_tests = len([r for r in results if r.status == TestStatus.ERROR])
        
        # Сбор всех метрик
        all_metrics = {}
        for result in results:
            for metric in result.metrics:
                if metric.name not in all_metrics:
                    all_metrics[metric.name] = []
                all_metrics[metric.name].append(metric.value)
                
        # Расчет средних значений
        avg_metrics = {}
        for name, values in all_metrics.items():
            avg_metrics[name] = {
                'avg': np.mean(values),
                'min': np.min(values),
                'max': np.max(values),
                'target': self.target_metrics.get(name, 0),
                'achieved': np.mean(values) >= self.target_metrics.get(name, 0) if name in ['win_rate', 'roi', 'sharpe_ratio', 'uptime', 'throughput'] else np.mean(values) <= self.target_metrics.get(name, float('inf'))
            }
            
        # Общая оценка
        target_achievement = sum([1 for metrics in avg_metrics.values() if metrics['achieved']]) / len(avg_metrics) * 100 if avg_metrics else 0
        
        summary = {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': failed_tests,
            'error_tests': error_tests,
            'success_rate': (passed_tests / total_tests) * 100 if total_tests > 0 else 0,
            'target_achievement': target_achievement,
            'metrics': avg_metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        return summary
        
    async def _continuous_testing(self):
        """Непрерывное тестирование"""
        while True:
            try:
                # Быстрые тесты каждые 5 минут
                await self._run_quick_tests()
                await asyncio.sleep(300)
                
            except Exception as e:
                self.logger.error(f"Continuous testing error: {e}")
                await asyncio.sleep(600)
                
    async def _run_quick_tests(self):
        """Быстрые тесты"""
        # Тест системных ресурсов
        system_results = await self._run_system_tests()
        self.test_results.extend(system_results)
        
        # Ограничение истории результатов
        if len(self.test_results) > 10000:
            self.test_results = self.test_results[-5000:]
            
    async def _metrics_collector(self):
        """Сборщик метрик"""
        while True:
            try:
                # Сбор метрик производительности
                for result in self.test_results[-100:]:  # Последние 100 результатов
                    for metric in result.metrics:
                        self.performance_history[metric.name].append(metric.value)
                        
                        # Ограничение истории
                        if len(self.performance_history[metric.name]) > 1000:
                            self.performance_history[metric.name] = self.performance_history[metric.name][-500:]
                            
                await asyncio.sleep(60)  # Каждую минуту
                
            except Exception as e:
                self.logger.error(f"Metrics collector error: {e}")
                await asyncio.sleep(300)
                
    async def _report_generator(self):
        """Генератор отчетов"""
        while True:
            try:
                # Генерация отчета каждый час
                if self.test_results:
                    report = await self._generate_performance_report()
                    
                    # Сохранение отчета в Redis
                    await self.redis_client.set(
                        "performance_report",
                        json.dumps(report, default=str),
                        ex=3600
                    )
                    
                await asyncio.sleep(3600)  # Каждый час
                
            except Exception as e:
                self.logger.error(f"Report generator error: {e}")
                await asyncio.sleep(1800)
                
    async def _generate_performance_report(self) -> Dict[str, Any]:
        """Генерация отчета о производительности"""
        recent_results = self.test_results[-100:]  # Последние 100 результатов
        
        if not recent_results:
            return {}
            
        summary = self._analyze_results(recent_results)
        
        # Добавление трендов
        trends = {}
        for metric_name, values in self.performance_history.items():
            if len(values) >= 10:
                # Простой тренд (последние 10 vs предыдущие 10)
                recent_avg = np.mean(values[-10:])
                previous_avg = np.mean(values[-20:-10]) if len(values) >= 20 else recent_avg
                
                trend = "improving" if recent_avg > previous_avg else "declining" if recent_avg < previous_avg else "stable"
                trends[metric_name] = {
                    'trend': trend,
                    'recent_avg': recent_avg,
                    'previous_avg': previous_avg,
                    'change_percent': ((recent_avg - previous_avg) / previous_avg * 100) if previous_avg != 0 else 0
                }
                
        report = {
            'summary': summary,
            'trends': trends,
            'system_stats': self.system_monitor.get_stats(),
            'generated_at': datetime.now().isoformat()
        }
        
        return report
        
    async def get_performance_report(self) -> Dict[str, Any]:
        """Получение отчета о производительности"""
        report_json = await self.redis_client.get("performance_report")
        
        if report_json:
            return json.loads(report_json)
        else:
            return await self._generate_performance_report()

async def main():
    """Основная функция запуска"""
    config = {
        'redis_host': 'localhost',
        'redis_port': 6379,
        'test_base_url': 'http://localhost:8000',
        'load_test': {
            'concurrent_users': 100,
            'duration_seconds': 300,
            'target_rps': 1000,
            'max_response_time': 0.01  # 10ms
        }
    }
    
    testing_system = EnterprisePerformanceTestingSystem(config)
    await testing_system.start()
    
    print("Enterprise Performance Testing System started")
    
    try:
        # Запуск полного набора тестов
        print("Running full test suite...")
        results = await testing_system.run_full_test_suite()
        
        print("\n=== TEST RESULTS ===")
        print(f"Total tests: {results['total_tests']}")
        print(f"Passed: {results['passed_tests']}")
        print(f"Failed: {results['failed_tests']}")
        print(f"Errors: {results['error_tests']}")
        print(f"Success rate: {results['success_rate']:.1f}%")
        print(f"Target achievement: {results['target_achievement']:.1f}%")
        
        print("\n=== METRICS ===")
        for metric_name, metric_data in results['metrics'].items():
            status = "✅ PASS" if metric_data['achieved'] else "❌ FAIL"
            print(f"{metric_name}: {metric_data['avg']:.2f} (target: {metric_data['target']}) {status}")
            
        # Генерация отчета
        report = await testing_system.get_performance_report()
        print(f"\nDetailed report generated at: {report.get('generated_at')}")
        
        # Непрерывное тестирование
        print("\nStarting continuous testing...")
        await asyncio.Future()  # Бесконечное ожидание
        
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        await testing_system.stop()

if __name__ == '__main__':
    asyncio.run(main())