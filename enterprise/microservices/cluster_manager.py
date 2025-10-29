"""
Enterprise Cluster Manager - Управление High Availability кластером
Обеспечивает автоматическое масштабирование, failover и восстановление
"""

import asyncio
import json
import time
import psutil
import docker
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import aiohttp
import logging
import redis.asyncio as redis
from prometheus_client import Counter, Histogram, Gauge
import yaml
import subprocess

class NodeStatus(Enum):
    """Статусы узлов кластера"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    OFFLINE = "offline"
    MAINTENANCE = "maintenance"

class ServiceStatus(Enum):
    """Статусы сервисов"""
    RUNNING = "running"
    STARTING = "starting"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"

@dataclass
class ClusterNode:
    """Узел кластера"""
    id: str
    hostname: str
    ip_address: str
    port: int
    status: NodeStatus
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    last_heartbeat: datetime
    services: List[str]
    capacity: int = 100
    current_load: int = 0

@dataclass
class ServiceInstance:
    """Экземпляр сервиса"""
    id: str
    name: str
    node_id: str
    status: ServiceStatus
    port: int
    health_endpoint: str
    cpu_limit: float
    memory_limit: int
    restart_count: int = 0
    last_restart: Optional[datetime] = None

@dataclass
class AutoScalingConfig:
    """Конфигурация автомасштабирования"""
    min_instances: int = 1
    max_instances: int = 10
    target_cpu_utilization: float = 70.0
    target_memory_utilization: float = 80.0
    scale_up_threshold: float = 80.0
    scale_down_threshold: float = 30.0
    cooldown_period: int = 300  # секунды

# Метрики
CLUSTER_NODES = Gauge('cluster_nodes_total', 'Total cluster nodes', ['status'])
SERVICE_INSTANCES = Gauge('cluster_service_instances_total', 'Total service instances', ['service', 'status'])
FAILOVER_EVENTS = Counter('cluster_failover_events_total', 'Failover events', ['service'])
AUTO_SCALING_EVENTS = Counter('cluster_autoscaling_events_total', 'Auto scaling events', ['service', 'action'])

class EnterpriseClusterManager:
    """Enterprise Cluster Manager для High Availability"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.nodes: Dict[str, ClusterNode] = {}
        self.services: Dict[str, ServiceInstance] = {}
        self.auto_scaling_configs: Dict[str, AutoScalingConfig] = {}
        self.redis_client = None
        self.docker_client = None
        self.logger = self._setup_logging()
        self.is_leader = False
        self.leader_election_key = "cluster_leader"
        
        # Инициализация
        self._load_cluster_config()
        
    def _setup_logging(self) -> logging.Logger:
        """Настройка логирования"""
        logger = logging.getLogger('enterprise_cluster_manager')
        logger.setLevel(logging.INFO)
        
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
        
    def _load_cluster_config(self):
        """Загрузка конфигурации кластера"""
        # Конфигурация автомасштабирования для разных сервисов
        self.auto_scaling_configs = {
            'trading': AutoScalingConfig(
                min_instances=2,
                max_instances=20,
                target_cpu_utilization=60.0,
                scale_up_threshold=75.0,
                scale_down_threshold=25.0
            ),
            'portfolio': AutoScalingConfig(
                min_instances=2,
                max_instances=10,
                target_cpu_utilization=70.0
            ),
            'ai': AutoScalingConfig(
                min_instances=1,
                max_instances=5,
                target_cpu_utilization=80.0,
                target_memory_utilization=85.0
            ),
            'compliance': AutoScalingConfig(
                min_instances=1,
                max_instances=3
            ),
            'risk': AutoScalingConfig(
                min_instances=2,
                max_instances=8
            )
        }
        
    async def start(self):
        """Запуск Cluster Manager"""
        # Подключение к Redis
        self.redis_client = redis.Redis(
            host=self.config.get('redis_host', 'localhost'),
            port=self.config.get('redis_port', 6379),
            decode_responses=True
        )
        
        # Подключение к Docker
        try:
            self.docker_client = docker.from_env()
        except Exception as e:
            self.logger.warning(f"Docker not available: {e}")
            
        # Запуск основных процессов
        asyncio.create_task(self._leader_election_loop())
        asyncio.create_task(self._cluster_monitoring_loop())
        asyncio.create_task(self._auto_scaling_loop())
        asyncio.create_task(self._health_check_loop())
        
        self.logger.info("Enterprise Cluster Manager started")
        
    async def stop(self):
        """Остановка Cluster Manager"""
        if self.redis_client:
            await self.redis_client.close()
            
    async def _leader_election_loop(self):
        """Выборы лидера кластера"""
        while True:
            try:
                # Попытка стать лидером
                result = await self.redis_client.set(
                    self.leader_election_key,
                    self.config.get('node_id', 'node-1'),
                    ex=30,  # TTL 30 секунд
                    nx=True  # Только если ключ не существует
                )
                
                if result:
                    if not self.is_leader:
                        self.logger.info("Became cluster leader")
                        self.is_leader = True
                else:
                    # Проверяем, кто лидер
                    current_leader = await self.redis_client.get(self.leader_election_key)
                    if current_leader != self.config.get('node_id', 'node-1'):
                        if self.is_leader:
                            self.logger.info("Lost cluster leadership")
                            self.is_leader = False
                            
                await asyncio.sleep(10)
                
            except Exception as e:
                self.logger.error(f"Leader election error: {e}")
                await asyncio.sleep(5)
                
    async def _cluster_monitoring_loop(self):
        """Мониторинг состояния кластера"""
        while True:
            try:
                if self.is_leader:
                    await self._update_cluster_state()
                    await self._detect_failures()
                    await self._update_metrics()
                    
                await asyncio.sleep(30)
                
            except Exception as e:
                self.logger.error(f"Cluster monitoring error: {e}")
                await asyncio.sleep(10)
                
    async def _update_cluster_state(self):
        """Обновление состояния кластера"""
        # Получение информации о текущем узле
        current_node = ClusterNode(
            id=self.config.get('node_id', 'node-1'),
            hostname=self.config.get('hostname', 'localhost'),
            ip_address=self.config.get('ip_address', '127.0.0.1'),
            port=self.config.get('port', 8000),
            status=NodeStatus.HEALTHY,
            cpu_usage=psutil.cpu_percent(),
            memory_usage=psutil.virtual_memory().percent,
            disk_usage=psutil.disk_usage('/').percent,
            last_heartbeat=datetime.now(),
            services=await self._get_running_services()
        )
        
        # Сохранение в Redis
        await self.redis_client.setex(
            f"cluster_node:{current_node.id}",
            60,
            json.dumps(asdict(current_node), default=str)
        )
        
        self.nodes[current_node.id] = current_node
        
    async def _get_running_services(self) -> List[str]:
        """Получение списка запущенных сервисов"""
        services = []
        
        if self.docker_client:
            try:
                containers = self.docker_client.containers.list()
                for container in containers:
                    if 'peper-binance' in container.name:
                        services.append(container.name)
            except Exception as e:
                self.logger.warning(f"Failed to get Docker containers: {e}")
                
        return services
        
    async def _detect_failures(self):
        """Обнаружение отказов в кластере"""
        current_time = datetime.now()
        
        # Проверка узлов
        node_keys = await self.redis_client.keys("cluster_node:*")
        for key in node_keys:
            try:
                node_data = await self.redis_client.get(key)
                if node_data:
                    node_dict = json.loads(node_data)
                    node = ClusterNode(**node_dict)
                    
                    # Проверка heartbeat
                    if isinstance(node.last_heartbeat, str):
                        node.last_heartbeat = datetime.fromisoformat(node.last_heartbeat)
                        
                    time_since_heartbeat = current_time - node.last_heartbeat
                    
                    if time_since_heartbeat > timedelta(minutes=2):
                        node.status = NodeStatus.OFFLINE
                        await self._handle_node_failure(node)
                    elif time_since_heartbeat > timedelta(minutes=1):
                        node.status = NodeStatus.DEGRADED
                        
                    self.nodes[node.id] = node
                    
            except Exception as e:
                self.logger.error(f"Error checking node {key}: {e}")
                
        # Проверка сервисов
        await self._check_service_health()
        
    async def _handle_node_failure(self, failed_node: ClusterNode):
        """Обработка отказа узла"""
        self.logger.warning(f"Node {failed_node.id} failed, initiating failover")
        
        # Перезапуск сервисов на других узлах
        for service_name in failed_node.services:
            await self._failover_service(service_name, failed_node.id)
            
        FAILOVER_EVENTS.labels(service='node').inc()
        
    async def _failover_service(self, service_name: str, failed_node_id: str):
        """Failover сервиса на другой узел"""
        try:
            # Поиск здорового узла с наименьшей нагрузкой
            target_node = self._find_best_node_for_service(service_name)
            
            if target_node:
                await self._start_service_on_node(service_name, target_node.id)
                self.logger.info(f"Service {service_name} failed over to node {target_node.id}")
            else:
                self.logger.error(f"No available nodes for failover of service {service_name}")
                
        except Exception as e:
            self.logger.error(f"Failover error for service {service_name}: {e}")
            
    def _find_best_node_for_service(self, service_name: str) -> Optional[ClusterNode]:
        """Поиск лучшего узла для размещения сервиса"""
        healthy_nodes = [
            node for node in self.nodes.values()
            if node.status == NodeStatus.HEALTHY and node.current_load < node.capacity
        ]
        
        if not healthy_nodes:
            return None
            
        # Сортировка по загрузке CPU и памяти
        healthy_nodes.sort(key=lambda n: (n.cpu_usage + n.memory_usage) / 2)
        return healthy_nodes[0]
        
    async def _start_service_on_node(self, service_name: str, node_id: str):
        """Запуск сервиса на узле"""
        if self.docker_client and node_id == self.config.get('node_id'):
            try:
                # Конфигурация контейнера
                container_config = self._get_service_container_config(service_name)
                
                container = self.docker_client.containers.run(
                    **container_config,
                    detach=True
                )
                
                self.logger.info(f"Started service {service_name} in container {container.id}")
                
            except Exception as e:
                self.logger.error(f"Failed to start service {service_name}: {e}")
                
    def _get_service_container_config(self, service_name: str) -> Dict[str, Any]:
        """Получение конфигурации контейнера для сервиса"""
        base_config = {
            'image': f'peper-binance-{service_name}:latest',
            'name': f'peper-binance-{service_name}-{int(time.time())}',
            'network_mode': 'bridge',
            'restart_policy': {'Name': 'unless-stopped'},
            'environment': {
                'SERVICE_NAME': service_name,
                'NODE_ID': self.config.get('node_id'),
                'REDIS_HOST': self.config.get('redis_host', 'localhost')
            }
        }
        
        # Специфичные настройки для разных сервисов
        service_configs = {
            'trading': {
                'ports': {'8001/tcp': 8001},
                'mem_limit': '1g',
                'cpu_quota': 50000
            },
            'portfolio': {
                'ports': {'8003/tcp': 8003},
                'mem_limit': '512m',
                'cpu_quota': 30000
            },
            'ai': {
                'ports': {'8005/tcp': 8005},
                'mem_limit': '2g',
                'cpu_quota': 100000
            }
        }
        
        if service_name in service_configs:
            base_config.update(service_configs[service_name])
            
        return base_config
        
    async def _auto_scaling_loop(self):
        """Автоматическое масштабирование сервисов"""
        while True:
            try:
                if self.is_leader:
                    for service_name, config in self.auto_scaling_configs.items():
                        await self._check_auto_scaling(service_name, config)
                        
                await asyncio.sleep(60)  # Проверка каждую минуту
                
            except Exception as e:
                self.logger.error(f"Auto scaling error: {e}")
                await asyncio.sleep(30)
                
    async def _check_auto_scaling(self, service_name: str, config: AutoScalingConfig):
        """Проверка необходимости автомасштабирования"""
        # Получение метрик сервиса
        service_metrics = await self._get_service_metrics(service_name)
        current_instances = len(service_metrics)
        
        if current_instances == 0:
            return
            
        # Расчет средних метрик
        avg_cpu = sum(m['cpu_usage'] for m in service_metrics) / len(service_metrics)
        avg_memory = sum(m['memory_usage'] for m in service_metrics) / len(service_metrics)
        
        # Решение о масштабировании
        should_scale_up = (
            (avg_cpu > config.scale_up_threshold or avg_memory > config.scale_up_threshold) and
            current_instances < config.max_instances
        )
        
        should_scale_down = (
            avg_cpu < config.scale_down_threshold and
            avg_memory < config.scale_down_threshold and
            current_instances > config.min_instances
        )
        
        if should_scale_up:
            await self._scale_up_service(service_name)
            AUTO_SCALING_EVENTS.labels(service=service_name, action='scale_up').inc()
        elif should_scale_down:
            await self._scale_down_service(service_name)
            AUTO_SCALING_EVENTS.labels(service=service_name, action='scale_down').inc()
            
    async def _get_service_metrics(self, service_name: str) -> List[Dict[str, Any]]:
        """Получение метрик сервиса"""
        metrics = []
        
        # Поиск всех экземпляров сервиса
        service_keys = await self.redis_client.keys(f"service_metrics:{service_name}:*")
        
        for key in service_keys:
            try:
                metric_data = await self.redis_client.get(key)
                if metric_data:
                    metrics.append(json.loads(metric_data))
            except Exception as e:
                self.logger.warning(f"Failed to get metrics for {key}: {e}")
                
        return metrics
        
    async def _scale_up_service(self, service_name: str):
        """Увеличение количества экземпляров сервиса"""
        target_node = self._find_best_node_for_service(service_name)
        
        if target_node:
            await self._start_service_on_node(service_name, target_node.id)
            self.logger.info(f"Scaled up service {service_name}")
        else:
            self.logger.warning(f"No available nodes to scale up service {service_name}")
            
    async def _scale_down_service(self, service_name: str):
        """Уменьшение количества экземпляров сервиса"""
        # Поиск экземпляра с наименьшей нагрузкой
        service_metrics = await self._get_service_metrics(service_name)
        
        if len(service_metrics) > 1:
            # Сортировка по нагрузке
            service_metrics.sort(key=lambda m: m.get('cpu_usage', 0))
            
            # Остановка экземпляра с наименьшей нагрузкой
            instance_to_stop = service_metrics[0]
            await self._stop_service_instance(instance_to_stop['instance_id'])
            
            self.logger.info(f"Scaled down service {service_name}")
            
    async def _stop_service_instance(self, instance_id: str):
        """Остановка экземпляра сервиса"""
        if self.docker_client:
            try:
                container = self.docker_client.containers.get(instance_id)
                container.stop()
                container.remove()
                
                self.logger.info(f"Stopped service instance {instance_id}")
                
            except Exception as e:
                self.logger.error(f"Failed to stop service instance {instance_id}: {e}")
                
    async def _health_check_loop(self):
        """Проверка здоровья сервисов"""
        while True:
            try:
                await self._check_service_health()
                await asyncio.sleep(30)
                
            except Exception as e:
                self.logger.error(f"Health check error: {e}")
                await asyncio.sleep(10)
                
    async def _check_service_health(self):
        """Проверка здоровья всех сервисов"""
        if self.docker_client:
            try:
                containers = self.docker_client.containers.list()
                
                for container in containers:
                    if 'peper-binance' in container.name:
                        # Проверка статуса контейнера
                        if container.status != 'running':
                            await self._restart_unhealthy_service(container)
                            
            except Exception as e:
                self.logger.error(f"Service health check error: {e}")
                
    async def _restart_unhealthy_service(self, container):
        """Перезапуск нездорового сервиса"""
        try:
            self.logger.warning(f"Restarting unhealthy service {container.name}")
            container.restart()
            
        except Exception as e:
            self.logger.error(f"Failed to restart service {container.name}: {e}")
            
    async def _update_metrics(self):
        """Обновление метрик Prometheus"""
        # Подсчет узлов по статусам
        status_counts = {}
        for node in self.nodes.values():
            status = node.status.value
            status_counts[status] = status_counts.get(status, 0) + 1
            
        for status, count in status_counts.items():
            CLUSTER_NODES.labels(status=status).set(count)
            
        # Подсчет сервисов
        if self.docker_client:
            try:
                containers = self.docker_client.containers.list(all=True)
                service_counts = {}
                
                for container in containers:
                    if 'peper-binance' in container.name:
                        service_name = container.name.split('-')[2]  # peper-binance-trading-xxx
                        status = container.status
                        
                        key = f"{service_name}_{status}"
                        service_counts[key] = service_counts.get(key, 0) + 1
                        
                for key, count in service_counts.items():
                    service_name, status = key.rsplit('_', 1)
                    SERVICE_INSTANCES.labels(service=service_name, status=status).set(count)
                    
            except Exception as e:
                self.logger.error(f"Failed to update service metrics: {e}")

async def main():
    """Основная функция запуска"""
    config = {
        'node_id': 'cluster-manager-1',
        'hostname': 'localhost',
        'ip_address': '127.0.0.1',
        'port': 9000,
        'redis_host': 'localhost',
        'redis_port': 6379
    }
    
    cluster_manager = EnterpriseClusterManager(config)
    await cluster_manager.start()
    
    print("Enterprise Cluster Manager started")
    
    try:
        await asyncio.Future()  # Бесконечное ожидание
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        await cluster_manager.stop()

if __name__ == '__main__':
    asyncio.run(main())