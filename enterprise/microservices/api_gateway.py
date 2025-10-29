"""
Enterprise API Gateway - Центральная точка входа для всех запросов
Обеспечивает маршрутизацию, аутентификацию, rate limiting и мониторинг
"""

import asyncio
import json
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import aiohttp
import jwt
from aiohttp import web, ClientSession
import logging
import redis.asyncio as redis
from prometheus_client import Counter, Histogram, Gauge, generate_latest

# Метрики Prometheus
REQUEST_COUNT = Counter('api_gateway_requests_total', 'Total API requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('api_gateway_request_duration_seconds', 'Request duration')
ACTIVE_CONNECTIONS = Gauge('api_gateway_active_connections', 'Active connections')
RATE_LIMIT_HITS = Counter('api_gateway_rate_limit_hits_total', 'Rate limit hits', ['client_id'])

@dataclass
class ServiceConfig:
    """Конфигурация микросервиса"""
    name: str
    host: str
    port: int
    health_endpoint: str
    weight: int = 1
    max_connections: int = 100
    timeout: int = 30

@dataclass
class RateLimitConfig:
    """Конфигурация rate limiting"""
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    burst_limit: int = 10

class EnterpriseAPIGateway:
    """Enterprise API Gateway с поддержкой High Availability"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.services: Dict[str, List[ServiceConfig]] = {}
        self.redis_client = None
        self.session: Optional[ClientSession] = None
        self.logger = self._setup_logging()
        self.jwt_secret = config.get('jwt_secret', 'enterprise-secret-key')
        self.rate_limits: Dict[str, RateLimitConfig] = {}
        
        # Инициализация сервисов
        self._load_service_configs()
        
    def _setup_logging(self) -> logging.Logger:
        """Настройка логирования"""
        logger = logging.getLogger('enterprise_api_gateway')
        logger.setLevel(logging.INFO)
        
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
        
    def _load_service_configs(self):
        """Загрузка конфигураций микросервисов"""
        services_config = {
            'trading': [
                ServiceConfig('trading-1', 'localhost', 8001, '/health'),
                ServiceConfig('trading-2', 'localhost', 8002, '/health')
            ],
            'portfolio': [
                ServiceConfig('portfolio-1', 'localhost', 8003, '/health'),
                ServiceConfig('portfolio-2', 'localhost', 8004, '/health')
            ],
            'ai': [
                ServiceConfig('ai-1', 'localhost', 8005, '/health'),
                ServiceConfig('ai-2', 'localhost', 8006, '/health')
            ],
            'compliance': [
                ServiceConfig('compliance-1', 'localhost', 8007, '/health')
            ],
            'risk': [
                ServiceConfig('risk-1', 'localhost', 8008, '/health')
            ],
            'notification': [
                ServiceConfig('notification-1', 'localhost', 8009, '/health')
            ]
        }
        
        self.services = services_config
        
        # Настройка rate limits для разных типов клиентов
        self.rate_limits = {
            'retail': RateLimitConfig(60, 1000, 10),
            'professional': RateLimitConfig(300, 5000, 50),
            'institutional': RateLimitConfig(1000, 20000, 200),
            'admin': RateLimitConfig(10000, 100000, 1000)
        }
        
    async def start(self):
        """Запуск API Gateway"""
        # Подключение к Redis
        self.redis_client = redis.Redis(
            host=self.config.get('redis_host', 'localhost'),
            port=self.config.get('redis_port', 6379),
            decode_responses=True
        )
        
        # Создание HTTP сессии
        self.session = ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            connector=aiohttp.TCPConnector(limit=1000)
        )
        
        # Запуск health check для сервисов
        asyncio.create_task(self._health_check_loop())
        
        self.logger.info("Enterprise API Gateway started successfully")
        
    async def stop(self):
        """Остановка API Gateway"""
        if self.session:
            await self.session.close()
        if self.redis_client:
            await self.redis_client.close()
            
    async def _health_check_loop(self):
        """Периодическая проверка здоровья сервисов"""
        while True:
            try:
                for service_type, instances in self.services.items():
                    for service in instances:
                        await self._check_service_health(service)
                await asyncio.sleep(30)  # Проверка каждые 30 секунд
            except Exception as e:
                self.logger.error(f"Health check error: {e}")
                await asyncio.sleep(10)
                
    async def _check_service_health(self, service: ServiceConfig) -> bool:
        """Проверка здоровья конкретного сервиса"""
        try:
            url = f"http://{service.host}:{service.port}{service.health_endpoint}"
            async with self.session.get(url, timeout=5) as response:
                if response.status == 200:
                    await self.redis_client.setex(
                        f"service_health:{service.name}", 
                        60, 
                        "healthy"
                    )
                    return True
                else:
                    await self.redis_client.setex(
                        f"service_health:{service.name}", 
                        60, 
                        "unhealthy"
                    )
                    return False
        except Exception as e:
            self.logger.warning(f"Health check failed for {service.name}: {e}")
            await self.redis_client.setex(
                f"service_health:{service.name}", 
                60, 
                "unhealthy"
            )
            return False
            
    async def _get_healthy_service(self, service_type: str) -> Optional[ServiceConfig]:
        """Получение здорового сервиса с балансировкой нагрузки"""
        if service_type not in self.services:
            return None
            
        healthy_services = []
        for service in self.services[service_type]:
            health_status = await self.redis_client.get(f"service_health:{service.name}")
            if health_status == "healthy":
                healthy_services.append(service)
                
        if not healthy_services:
            # Если нет здоровых сервисов, возвращаем первый доступный
            return self.services[service_type][0] if self.services[service_type] else None
            
        # Простая round-robin балансировка
        service_index = await self.redis_client.incr(f"lb_counter:{service_type}") % len(healthy_services)
        return healthy_services[service_index]
        
    async def _authenticate_request(self, request: web.Request) -> Optional[Dict[str, Any]]:
        """Аутентификация запроса"""
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return None
            
        token = auth_header[7:]  # Убираем 'Bearer '
        
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=['HS256'])
            return payload
        except jwt.InvalidTokenError:
            return None
            
    async def _check_rate_limit(self, client_id: str, client_type: str) -> bool:
        """Проверка rate limiting"""
        if client_type not in self.rate_limits:
            client_type = 'retail'  # По умолчанию
            
        rate_config = self.rate_limits[client_type]
        current_time = int(time.time())
        
        # Проверка лимита в минуту
        minute_key = f"rate_limit:{client_id}:minute:{current_time // 60}"
        minute_count = await self.redis_client.incr(minute_key)
        await self.redis_client.expire(minute_key, 60)
        
        if minute_count > rate_config.requests_per_minute:
            RATE_LIMIT_HITS.labels(client_id=client_id).inc()
            return False
            
        # Проверка лимита в час
        hour_key = f"rate_limit:{client_id}:hour:{current_time // 3600}"
        hour_count = await self.redis_client.incr(hour_key)
        await self.redis_client.expire(hour_key, 3600)
        
        if hour_count > rate_config.requests_per_hour:
            RATE_LIMIT_HITS.labels(client_id=client_id).inc()
            return False
            
        return True
        
    async def _proxy_request(self, request: web.Request, service: ServiceConfig, path: str) -> web.Response:
        """Проксирование запроса к микросервису"""
        url = f"http://{service.host}:{service.port}{path}"
        
        # Подготовка заголовков
        headers = dict(request.headers)
        headers.pop('Host', None)  # Убираем Host заголовок
        
        # Подготовка данных
        data = None
        if request.method in ['POST', 'PUT', 'PATCH']:
            data = await request.read()
            
        try:
            async with self.session.request(
                method=request.method,
                url=url,
                headers=headers,
                data=data,
                params=request.query,
                timeout=service.timeout
            ) as response:
                body = await response.read()
                
                # Создание ответа
                resp = web.Response(
                    body=body,
                    status=response.status,
                    headers=response.headers
                )
                
                return resp
                
        except asyncio.TimeoutError:
            return web.json_response(
                {'error': 'Service timeout'}, 
                status=504
            )
        except Exception as e:
            self.logger.error(f"Proxy error: {e}")
            return web.json_response(
                {'error': 'Internal server error'}, 
                status=500
            )
            
    async def handle_request(self, request: web.Request) -> web.Response:
        """Основной обработчик запросов"""
        start_time = time.time()
        ACTIVE_CONNECTIONS.inc()
        
        try:
            # Определение типа сервиса по пути
            path = request.path
            service_type = None
            
            if path.startswith('/api/v1/trading'):
                service_type = 'trading'
            elif path.startswith('/api/v1/portfolio'):
                service_type = 'portfolio'
            elif path.startswith('/api/v1/ai'):
                service_type = 'ai'
            elif path.startswith('/api/v1/compliance'):
                service_type = 'compliance'
            elif path.startswith('/api/v1/risk'):
                service_type = 'risk'
            elif path.startswith('/api/v1/notifications'):
                service_type = 'notification'
            else:
                REQUEST_COUNT.labels(
                    method=request.method, 
                    endpoint=path, 
                    status='404'
                ).inc()
                return web.json_response({'error': 'Service not found'}, status=404)
                
            # Аутентификация
            auth_data = await self._authenticate_request(request)
            if not auth_data and not path.startswith('/api/v1/auth'):
                REQUEST_COUNT.labels(
                    method=request.method, 
                    endpoint=path, 
                    status='401'
                ).inc()
                return web.json_response({'error': 'Unauthorized'}, status=401)
                
            # Rate limiting
            if auth_data:
                client_id = auth_data.get('user_id', 'anonymous')
                client_type = auth_data.get('user_type', 'retail')
                
                if not await self._check_rate_limit(client_id, client_type):
                    REQUEST_COUNT.labels(
                        method=request.method, 
                        endpoint=path, 
                        status='429'
                    ).inc()
                    return web.json_response({'error': 'Rate limit exceeded'}, status=429)
                    
            # Получение здорового сервиса
            service = await self._get_healthy_service(service_type)
            if not service:
                REQUEST_COUNT.labels(
                    method=request.method, 
                    endpoint=path, 
                    status='503'
                ).inc()
                return web.json_response({'error': 'Service unavailable'}, status=503)
                
            # Проксирование запроса
            response = await self._proxy_request(request, service, path)
            
            REQUEST_COUNT.labels(
                method=request.method, 
                endpoint=path, 
                status=str(response.status)
            ).inc()
            
            return response
            
        except Exception as e:
            self.logger.error(f"Request handling error: {e}")
            REQUEST_COUNT.labels(
                method=request.method, 
                endpoint=request.path, 
                status='500'
            ).inc()
            return web.json_response({'error': 'Internal server error'}, status=500)
            
        finally:
            ACTIVE_CONNECTIONS.dec()
            REQUEST_DURATION.observe(time.time() - start_time)
            
    async def metrics_handler(self, request: web.Request) -> web.Response:
        """Обработчик метрик Prometheus"""
        return web.Response(
            text=generate_latest().decode('utf-8'),
            content_type='text/plain'
        )
        
    def create_app(self) -> web.Application:
        """Создание aiohttp приложения"""
        app = web.Application()
        
        # Добавление маршрутов
        app.router.add_route('*', '/metrics', self.metrics_handler)
        app.router.add_route('*', '/{path:.*}', self.handle_request)
        
        return app

async def main():
    """Основная функция запуска"""
    config = {
        'jwt_secret': 'enterprise-secret-key-2024',
        'redis_host': 'localhost',
        'redis_port': 6379,
        'port': 8000
    }
    
    gateway = EnterpriseAPIGateway(config)
    await gateway.start()
    
    app = gateway.create_app()
    
    # Запуск сервера
    runner = web.AppRunner(app)
    await runner.setup()
    
    site = web.TCPSite(runner, '0.0.0.0', config['port'])
    await site.start()
    
    print(f"Enterprise API Gateway started on port {config['port']}")
    
    try:
        await asyncio.Future()  # Бесконечное ожидание
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        await gateway.stop()
        await runner.cleanup()

if __name__ == '__main__':
    asyncio.run(main())