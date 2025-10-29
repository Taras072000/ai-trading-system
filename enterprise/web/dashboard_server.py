"""
Enterprise Web Dashboard Server - Веб-сервер для Enterprise Dashboard
Обеспечивает веб-интерфейс управления торговой системой
"""

import asyncio
import json
import time
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import logging
import redis.asyncio as redis
from pathlib import Path

# FastAPI и веб-компоненты
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import uvicorn

# Аутентификация и безопасность
import jwt
from passlib.context import CryptContext
from datetime import timedelta
import secrets

# Метрики и мониторинг
from prometheus_client import Counter, Histogram, Gauge, generate_latest
import psutil

# Базы данных
import aiosqlite
import asyncpg

# Утилиты
import numpy as np
import pandas as pd
from enum import Enum

class UserRole(Enum):
    """Роли пользователей"""
    ADMIN = "admin"
    TRADER = "trader"
    ANALYST = "analyst"
    VIEWER = "viewer"

class DashboardSection(Enum):
    """Разделы дашборда"""
    OVERVIEW = "overview"
    TRADING = "trading"
    PORTFOLIO = "portfolio"
    ANALYTICS = "analytics"
    RISK = "risk"
    SETTINGS = "settings"
    MONITORING = "monitoring"
    REPORTS = "reports"

@dataclass
class User:
    """Пользователь системы"""
    id: str
    username: str
    email: str
    role: UserRole
    permissions: List[str]
    created_at: datetime
    last_login: Optional[datetime] = None
    is_active: bool = True

@dataclass
class DashboardWidget:
    """Виджет дашборда"""
    id: str
    title: str
    type: str
    section: DashboardSection
    config: Dict[str, Any]
    position: Dict[str, int]
    size: Dict[str, int]
    data_source: str
    refresh_interval: int = 30

@dataclass
class TradingSession:
    """Торговая сессия"""
    id: str
    user_id: str
    strategy: str
    status: str
    start_time: datetime
    end_time: Optional[datetime]
    pnl: float
    trades_count: int
    win_rate: float

# Метрики Prometheus
DASHBOARD_REQUESTS = Counter('dashboard_requests_total', 'Total dashboard requests', ['endpoint', 'method'])
WEBSOCKET_CONNECTIONS = Gauge('websocket_connections_active', 'Active WebSocket connections')
USER_SESSIONS = Gauge('user_sessions_active', 'Active user sessions')
API_RESPONSE_TIME = Histogram('api_response_time_seconds', 'API response time', ['endpoint'])

class WebSocketManager:
    """Менеджер WebSocket соединений"""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.user_connections: Dict[str, List[str]] = {}
        
    async def connect(self, websocket: WebSocket, connection_id: str, user_id: str):
        """Подключение WebSocket"""
        await websocket.accept()
        self.active_connections[connection_id] = websocket
        
        if user_id not in self.user_connections:
            self.user_connections[user_id] = []
        self.user_connections[user_id].append(connection_id)
        
        WEBSOCKET_CONNECTIONS.set(len(self.active_connections))
        
    def disconnect(self, connection_id: str, user_id: str):
        """Отключение WebSocket"""
        if connection_id in self.active_connections:
            del self.active_connections[connection_id]
            
        if user_id in self.user_connections:
            if connection_id in self.user_connections[user_id]:
                self.user_connections[user_id].remove(connection_id)
            if not self.user_connections[user_id]:
                del self.user_connections[user_id]
                
        WEBSOCKET_CONNECTIONS.set(len(self.active_connections))
        
    async def send_personal_message(self, message: str, connection_id: str):
        """Отправка персонального сообщения"""
        if connection_id in self.active_connections:
            websocket = self.active_connections[connection_id]
            await websocket.send_text(message)
            
    async def send_user_message(self, message: str, user_id: str):
        """Отправка сообщения всем соединениям пользователя"""
        if user_id in self.user_connections:
            for connection_id in self.user_connections[user_id]:
                await self.send_personal_message(message, connection_id)
                
    async def broadcast(self, message: str):
        """Широковещательная отправка"""
        for websocket in self.active_connections.values():
            try:
                await websocket.send_text(message)
            except:
                pass  # Соединение может быть закрыто

class AuthManager:
    """Менеджер аутентификации"""
    
    def __init__(self, secret_key: str):
        self.secret_key = secret_key
        self.algorithm = "HS256"
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        self.security = HTTPBearer()
        
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Проверка пароля"""
        return self.pwd_context.verify(plain_password, hashed_password)
        
    def get_password_hash(self, password: str) -> str:
        """Хеширование пароля"""
        return self.pwd_context.hash(password)
        
    def create_access_token(self, data: dict, expires_delta: Optional[timedelta] = None):
        """Создание токена доступа"""
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(hours=24)
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt
        
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Проверка токена"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.PyJWTError:
            return None

class DatabaseManager:
    """Менеджер базы данных"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        
    async def init_database(self):
        """Инициализация базы данных"""
        async with aiosqlite.connect(self.db_path) as db:
            # Таблица пользователей
            await db.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    username TEXT UNIQUE NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    role TEXT NOT NULL,
                    permissions TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_login TIMESTAMP,
                    is_active BOOLEAN DEFAULT TRUE
                )
            """)
            
            # Таблица виджетов
            await db.execute("""
                CREATE TABLE IF NOT EXISTS dashboard_widgets (
                    id TEXT PRIMARY KEY,
                    user_id TEXT,
                    title TEXT NOT NULL,
                    type TEXT NOT NULL,
                    section TEXT NOT NULL,
                    config TEXT,
                    position TEXT,
                    size TEXT,
                    data_source TEXT,
                    refresh_interval INTEGER DEFAULT 30,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            """)
            
            # Таблица торговых сессий
            await db.execute("""
                CREATE TABLE IF NOT EXISTS trading_sessions (
                    id TEXT PRIMARY KEY,
                    user_id TEXT,
                    strategy TEXT,
                    status TEXT,
                    start_time TIMESTAMP,
                    end_time TIMESTAMP,
                    pnl REAL,
                    trades_count INTEGER,
                    win_rate REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            """)
            
            await db.commit()
            
    async def create_user(self, user: User, password_hash: str) -> bool:
        """Создание пользователя"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    INSERT INTO users (id, username, email, password_hash, role, permissions, created_at, is_active)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    user.id, user.username, user.email, password_hash, 
                    user.role.value, json.dumps(user.permissions), 
                    user.created_at, user.is_active
                ))
                await db.commit()
                return True
        except Exception as e:
            print(f"Error creating user: {e}")
            return False
            
    async def get_user_by_username(self, username: str) -> Optional[User]:
        """Получение пользователя по имени"""
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute("""
                SELECT id, username, email, role, permissions, created_at, last_login, is_active
                FROM users WHERE username = ?
            """, (username,)) as cursor:
                row = await cursor.fetchone()
                if row:
                    return User(
                        id=row[0],
                        username=row[1],
                        email=row[2],
                        role=UserRole(row[3]),
                        permissions=json.loads(row[4]) if row[4] else [],
                        created_at=datetime.fromisoformat(row[5]),
                        last_login=datetime.fromisoformat(row[6]) if row[6] else None,
                        is_active=bool(row[7])
                    )
        return None
        
    async def get_user_password_hash(self, username: str) -> Optional[str]:
        """Получение хеша пароля пользователя"""
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute("""
                SELECT password_hash FROM users WHERE username = ?
            """, (username,)) as cursor:
                row = await cursor.fetchone()
                return row[0] if row else None

class DataProvider:
    """Провайдер данных для дашборда"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis_client = redis_client
        
    async def get_trading_overview(self) -> Dict[str, Any]:
        """Получение обзора торговли"""
        # Симуляция данных (в реальности - из Redis/БД)
        return {
            'total_pnl': 15420.50,
            'daily_pnl': 1250.30,
            'win_rate': 78.5,
            'total_trades': 1247,
            'active_positions': 12,
            'portfolio_value': 125000.00,
            'available_balance': 25000.00,
            'margin_used': 15000.00,
            'unrealized_pnl': 2340.50
        }
        
    async def get_portfolio_data(self) -> Dict[str, Any]:
        """Получение данных портфеля"""
        return {
            'positions': [
                {'symbol': 'BTCUSDT', 'size': 0.5, 'pnl': 1250.30, 'unrealized_pnl': 340.50},
                {'symbol': 'ETHUSDT', 'size': 2.0, 'pnl': -150.20, 'unrealized_pnl': 120.30},
                {'symbol': 'ADAUSDT', 'size': 1000, 'pnl': 450.80, 'unrealized_pnl': -50.20}
            ],
            'allocation': {
                'BTC': 45.2,
                'ETH': 30.5,
                'ADA': 15.3,
                'Cash': 9.0
            },
            'performance': {
                'daily': 2.1,
                'weekly': 8.5,
                'monthly': 15.2,
                'yearly': 45.8
            }
        }
        
    async def get_market_data(self) -> Dict[str, Any]:
        """Получение рыночных данных"""
        return {
            'prices': {
                'BTCUSDT': 45230.50,
                'ETHUSDT': 3120.30,
                'ADAUSDT': 1.25
            },
            'changes_24h': {
                'BTCUSDT': 2.5,
                'ETHUSDT': -1.2,
                'ADAUSDT': 5.8
            },
            'volumes_24h': {
                'BTCUSDT': 1250000000,
                'ETHUSDT': 850000000,
                'ADAUSDT': 120000000
            }
        }
        
    async def get_risk_metrics(self) -> Dict[str, Any]:
        """Получение метрик риска"""
        return {
            'var_1d': -2340.50,
            'var_1w': -8920.30,
            'max_drawdown': 2.8,
            'sharpe_ratio': 2.45,
            'sortino_ratio': 3.12,
            'beta': 0.85,
            'correlation_btc': 0.72,
            'risk_score': 6.5
        }
        
    async def get_system_metrics(self) -> Dict[str, Any]:
        """Получение системных метрик"""
        return {
            'cpu_usage': psutil.cpu_percent(),
            'memory_usage': psutil.virtual_memory().percent,
            'disk_usage': psutil.disk_usage('/').percent,
            'uptime': time.time() - psutil.boot_time(),
            'active_connections': len(websocket_manager.active_connections),
            'api_requests_per_minute': 150,
            'latency_avg': 12.5,
            'error_rate': 0.02
        }

class EnterpriseDashboardServer:
    """Enterprise Dashboard Server"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.app = FastAPI(title="Enterprise Trading Dashboard", version="1.0.0")
        
        # Компоненты
        self.auth_manager = AuthManager(config.get('secret_key', secrets.token_hex(32)))
        self.db_manager = DatabaseManager(config.get('db_path', 'dashboard.db'))
        self.redis_client = None
        self.data_provider = None
        
        # Настройка приложения
        self._setup_middleware()
        self._setup_routes()
        
        self.logger = self._setup_logging()
        
    def _setup_logging(self) -> logging.Logger:
        """Настройка логирования"""
        logger = logging.getLogger('enterprise_dashboard')
        logger.setLevel(logging.INFO)
        
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
        
    def _setup_middleware(self):
        """Настройка middleware"""
        # CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Статические файлы
        static_path = Path(__file__).parent / "static"
        static_path.mkdir(exist_ok=True)
        self.app.mount("/static", StaticFiles(directory=str(static_path)), name="static")
        
        # Шаблоны
        templates_path = Path(__file__).parent / "templates"
        templates_path.mkdir(exist_ok=True)
        self.templates = Jinja2Templates(directory=str(templates_path))
        
    def _setup_routes(self):
        """Настройка маршрутов"""
        
        @self.app.on_event("startup")
        async def startup():
            await self.startup()
            
        @self.app.on_event("shutdown")
        async def shutdown():
            await self.shutdown()
            
        # Главная страница
        @self.app.get("/", response_class=HTMLResponse)
        async def dashboard_home(request: Request):
            DASHBOARD_REQUESTS.labels(endpoint="/", method="GET").inc()
            return self.templates.TemplateResponse("dashboard.html", {"request": request})
            
        # Аутентификация
        @self.app.post("/api/auth/login")
        async def login(credentials: Dict[str, str]):
            DASHBOARD_REQUESTS.labels(endpoint="/api/auth/login", method="POST").inc()
            
            username = credentials.get("username")
            password = credentials.get("password")
            
            if not username or not password:
                raise HTTPException(status_code=400, detail="Username and password required")
                
            # Проверка пользователя
            user = await self.db_manager.get_user_by_username(username)
            if not user or not user.is_active:
                raise HTTPException(status_code=401, detail="Invalid credentials")
                
            # Проверка пароля
            password_hash = await self.db_manager.get_user_password_hash(username)
            if not self.auth_manager.verify_password(password, password_hash):
                raise HTTPException(status_code=401, detail="Invalid credentials")
                
            # Создание токена
            access_token = self.auth_manager.create_access_token(
                data={"sub": user.username, "user_id": user.id, "role": user.role.value}
            )
            
            return {
                "access_token": access_token,
                "token_type": "bearer",
                "user": asdict(user)
            }
            
        # Получение текущего пользователя
        @self.app.get("/api/auth/me")
        async def get_current_user(token: HTTPAuthorizationCredentials = Depends(self.auth_manager.security)):
            payload = self.auth_manager.verify_token(token.credentials)
            if not payload:
                raise HTTPException(status_code=401, detail="Invalid token")
                
            user = await self.db_manager.get_user_by_username(payload["sub"])
            if not user:
                raise HTTPException(status_code=404, detail="User not found")
                
            return asdict(user)
            
        # API данных
        @self.app.get("/api/data/overview")
        async def get_overview_data():
            DASHBOARD_REQUESTS.labels(endpoint="/api/data/overview", method="GET").inc()
            return await self.data_provider.get_trading_overview()
            
        @self.app.get("/api/data/portfolio")
        async def get_portfolio_data():
            DASHBOARD_REQUESTS.labels(endpoint="/api/data/portfolio", method="GET").inc()
            return await self.data_provider.get_portfolio_data()
            
        @self.app.get("/api/data/market")
        async def get_market_data():
            DASHBOARD_REQUESTS.labels(endpoint="/api/data/market", method="GET").inc()
            return await self.data_provider.get_market_data()
            
        @self.app.get("/api/data/risk")
        async def get_risk_data():
            DASHBOARD_REQUESTS.labels(endpoint="/api/data/risk", method="GET").inc()
            return await self.data_provider.get_risk_metrics()
            
        @self.app.get("/api/data/system")
        async def get_system_data():
            DASHBOARD_REQUESTS.labels(endpoint="/api/data/system", method="GET").inc()
            return await self.data_provider.get_system_metrics()
            
        # Метрики Prometheus
        @self.app.get("/metrics")
        async def get_metrics():
            return generate_latest()
            
        # WebSocket для реального времени
        @self.app.websocket("/ws/{user_id}")
        async def websocket_endpoint(websocket: WebSocket, user_id: str):
            connection_id = f"{user_id}_{int(time.time())}"
            await websocket_manager.connect(websocket, connection_id, user_id)
            
            try:
                while True:
                    # Отправка данных в реальном времени
                    data = {
                        'type': 'update',
                        'timestamp': datetime.now().isoformat(),
                        'data': {
                            'overview': await self.data_provider.get_trading_overview(),
                            'market': await self.data_provider.get_market_data(),
                            'system': await self.data_provider.get_system_metrics()
                        }
                    }
                    
                    await websocket_manager.send_personal_message(
                        json.dumps(data), connection_id
                    )
                    
                    await asyncio.sleep(5)  # Обновление каждые 5 секунд
                    
            except WebSocketDisconnect:
                websocket_manager.disconnect(connection_id, user_id)
                
    async def startup(self):
        """Запуск сервера"""
        # Инициализация базы данных
        await self.db_manager.init_database()
        
        # Подключение к Redis
        self.redis_client = redis.Redis(
            host=self.config.get('redis_host', 'localhost'),
            port=self.config.get('redis_port', 6379),
            decode_responses=True
        )
        
        # Инициализация провайдера данных
        self.data_provider = DataProvider(self.redis_client)
        
        # Создание администратора по умолчанию
        await self._create_default_admin()
        
        # Создание шаблонов
        await self._create_templates()
        
        self.logger.info("Enterprise Dashboard Server started")
        
    async def shutdown(self):
        """Остановка сервера"""
        if self.redis_client:
            await self.redis_client.close()
            
    async def _create_default_admin(self):
        """Создание администратора по умолчанию"""
        admin_user = User(
            id="admin_001",
            username="admin",
            email="admin@enterprise.com",
            role=UserRole.ADMIN,
            permissions=["all"],
            created_at=datetime.now()
        )
        
        password_hash = self.auth_manager.get_password_hash("admin123")
        await self.db_manager.create_user(admin_user, password_hash)
        
    async def _create_templates(self):
        """Создание HTML шаблонов"""
        templates_path = Path(__file__).parent / "templates"
        
        # Основной шаблон дашборда
        dashboard_html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Enterprise Trading Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        .widget { transition: all 0.3s ease; }
        .widget:hover { transform: translateY(-2px); box-shadow: 0 4px 12px rgba(0,0,0,0.15); }
    </style>
</head>
<body class="bg-gray-100">
    <div id="app" class="min-h-screen">
        <!-- Header -->
        <header class="bg-white shadow-sm border-b">
            <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                <div class="flex justify-between items-center py-4">
                    <h1 class="text-2xl font-bold text-gray-900">Enterprise Trading Dashboard</h1>
                    <div class="flex items-center space-x-4">
                        <span class="text-sm text-gray-500" id="last-update">Last update: --</span>
                        <div class="w-3 h-3 bg-green-500 rounded-full" id="status-indicator"></div>
                    </div>
                </div>
            </div>
        </header>

        <!-- Main Content -->
        <main class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
            <!-- Overview Cards -->
            <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
                <div class="widget bg-white rounded-lg shadow p-6">
                    <div class="flex items-center">
                        <div class="flex-1">
                            <p class="text-sm font-medium text-gray-500">Total P&L</p>
                            <p class="text-2xl font-bold text-green-600" id="total-pnl">$0.00</p>
                        </div>
                        <div class="w-12 h-12 bg-green-100 rounded-lg flex items-center justify-center">
                            <svg class="w-6 h-6 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6"></path>
                            </svg>
                        </div>
                    </div>
                </div>

                <div class="widget bg-white rounded-lg shadow p-6">
                    <div class="flex items-center">
                        <div class="flex-1">
                            <p class="text-sm font-medium text-gray-500">Win Rate</p>
                            <p class="text-2xl font-bold text-blue-600" id="win-rate">0%</p>
                        </div>
                        <div class="w-12 h-12 bg-blue-100 rounded-lg flex items-center justify-center">
                            <svg class="w-6 h-6 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"></path>
                            </svg>
                        </div>
                    </div>
                </div>

                <div class="widget bg-white rounded-lg shadow p-6">
                    <div class="flex items-center">
                        <div class="flex-1">
                            <p class="text-sm font-medium text-gray-500">Active Positions</p>
                            <p class="text-2xl font-bold text-purple-600" id="active-positions">0</p>
                        </div>
                        <div class="w-12 h-12 bg-purple-100 rounded-lg flex items-center justify-center">
                            <svg class="w-6 h-6 text-purple-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10"></path>
                            </svg>
                        </div>
                    </div>
                </div>

                <div class="widget bg-white rounded-lg shadow p-6">
                    <div class="flex items-center">
                        <div class="flex-1">
                            <p class="text-sm font-medium text-gray-500">Portfolio Value</p>
                            <p class="text-2xl font-bold text-indigo-600" id="portfolio-value">$0.00</p>
                        </div>
                        <div class="w-12 h-12 bg-indigo-100 rounded-lg flex items-center justify-center">
                            <svg class="w-6 h-6 text-indigo-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8c-1.657 0-3 .895-3 2s1.343 2 3 2 3 .895 3 2-1.343 2-3 2m0-8c1.11 0 2.08.402 2.599 1M12 8V7m0 1v8m0 0v1m0-1c-1.11 0-2.08-.402-2.599-1"></path>
                            </svg>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Charts Section -->
            <div class="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
                <div class="widget bg-white rounded-lg shadow p-6">
                    <h3 class="text-lg font-semibold text-gray-900 mb-4">P&L Chart</h3>
                    <canvas id="pnl-chart" width="400" height="200"></canvas>
                </div>

                <div class="widget bg-white rounded-lg shadow p-6">
                    <h3 class="text-lg font-semibold text-gray-900 mb-4">Portfolio Allocation</h3>
                    <canvas id="allocation-chart" width="400" height="200"></canvas>
                </div>
            </div>

            <!-- Positions Table -->
            <div class="widget bg-white rounded-lg shadow">
                <div class="px-6 py-4 border-b border-gray-200">
                    <h3 class="text-lg font-semibold text-gray-900">Active Positions</h3>
                </div>
                <div class="overflow-x-auto">
                    <table class="min-w-full divide-y divide-gray-200">
                        <thead class="bg-gray-50">
                            <tr>
                                <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Symbol</th>
                                <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Size</th>
                                <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">P&L</th>
                                <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Unrealized P&L</th>
                            </tr>
                        </thead>
                        <tbody id="positions-table" class="bg-white divide-y divide-gray-200">
                            <!-- Positions will be populated here -->
                        </tbody>
                    </table>
                </div>
            </div>
        </main>
    </div>

    <script>
        // WebSocket connection
        let ws = null;
        let pnlChart = null;
        let allocationChart = null;

        function connectWebSocket() {
            ws = new WebSocket(`ws://localhost:8000/ws/user_001`);
            
            ws.onopen = function(event) {
                console.log('WebSocket connected');
                document.getElementById('status-indicator').className = 'w-3 h-3 bg-green-500 rounded-full';
            };
            
            ws.onmessage = function(event) {
                const data = JSON.parse(event.data);
                updateDashboard(data.data);
                document.getElementById('last-update').textContent = `Last update: ${new Date().toLocaleTimeString()}`;
            };
            
            ws.onclose = function(event) {
                console.log('WebSocket disconnected');
                document.getElementById('status-indicator').className = 'w-3 h-3 bg-red-500 rounded-full';
                // Reconnect after 5 seconds
                setTimeout(connectWebSocket, 5000);
            };
        }

        function updateDashboard(data) {
            // Update overview cards
            if (data.overview) {
                document.getElementById('total-pnl').textContent = `$${data.overview.total_pnl.toFixed(2)}`;
                document.getElementById('win-rate').textContent = `${data.overview.win_rate.toFixed(1)}%`;
                document.getElementById('active-positions').textContent = data.overview.active_positions;
                document.getElementById('portfolio-value').textContent = `$${data.overview.portfolio_value.toFixed(2)}`;
            }

            // Update charts
            updateCharts(data);
        }

        function updateCharts(data) {
            // P&L Chart
            if (!pnlChart) {
                const ctx = document.getElementById('pnl-chart').getContext('2d');
                pnlChart = new Chart(ctx, {
                    type: 'line',
                    data: {
                        labels: [],
                        datasets: [{
                            label: 'P&L',
                            data: [],
                            borderColor: 'rgb(34, 197, 94)',
                            backgroundColor: 'rgba(34, 197, 94, 0.1)',
                            tension: 0.1
                        }]
                    },
                    options: {
                        responsive: true,
                        scales: {
                            y: {
                                beginAtZero: true
                            }
                        }
                    }
                });
            }

            // Portfolio Allocation Chart
            if (!allocationChart) {
                const ctx = document.getElementById('allocation-chart').getContext('2d');
                allocationChart = new Chart(ctx, {
                    type: 'doughnut',
                    data: {
                        labels: ['BTC', 'ETH', 'ADA', 'Cash'],
                        datasets: [{
                            data: [45.2, 30.5, 15.3, 9.0],
                            backgroundColor: [
                                'rgb(249, 115, 22)',
                                'rgb(59, 130, 246)',
                                'rgb(139, 69, 19)',
                                'rgb(34, 197, 94)'
                            ]
                        }]
                    },
                    options: {
                        responsive: true
                    }
                });
            }
        }

        // Initialize dashboard
        document.addEventListener('DOMContentLoaded', function() {
            connectWebSocket();
            
            // Load initial data
            fetch('/api/data/overview')
                .then(response => response.json())
                .then(data => updateDashboard({overview: data}));
        });
    </script>
</body>
</html>
        """
        
        with open(templates_path / "dashboard.html", "w") as f:
            f.write(dashboard_html)

# Глобальный менеджер WebSocket
websocket_manager = WebSocketManager()

async def main():
    """Основная функция запуска"""
    config = {
        'secret_key': 'enterprise_dashboard_secret_key_2024',
        'db_path': 'enterprise_dashboard.db',
        'redis_host': 'localhost',
        'redis_port': 6379,
        'host': '0.0.0.0',
        'port': 8000
    }
    
    dashboard_server = EnterpriseDashboardServer(config)
    
    # Запуск сервера
    uvicorn_config = uvicorn.Config(
        dashboard_server.app,
        host=config['host'],
        port=config['port'],
        log_level="info"
    )
    
    server = uvicorn.Server(uvicorn_config)
    await server.serve()

if __name__ == '__main__':
    asyncio.run(main())