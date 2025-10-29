"""
Система интеграций и API для торговой системы
RESTful API, WebSocket real-time данные, внешние источники данных, мобильные уведомления
"""

import asyncio
import json
import logging
import websockets
import aiohttp
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from fastapi import FastAPI, WebSocket, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np
import sqlite3
import hashlib
import jwt
import requests
import threading
import queue
import time
from abc import ABC, abstractmethod
import uvicorn

# Модели данных для API
class TradingSignal(BaseModel):
    """Торговый сигнал"""
    symbol: str
    action: str  # BUY, SELL, HOLD
    price: float
    confidence: float
    timestamp: datetime
    reasoning: List[str] = []
    metadata: Dict[str, Any] = {}

class MarketData(BaseModel):
    """Рыночные данные"""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    indicators: Dict[str, float] = {}

class SystemStatus(BaseModel):
    """Статус системы"""
    timestamp: datetime
    status: str  # RUNNING, STOPPED, ERROR
    uptime: float
    active_symbols: List[str]
    performance_metrics: Dict[str, float]
    last_signal: Optional[TradingSignal] = None

class NotificationRequest(BaseModel):
    """Запрос на уведомление"""
    type: str  # email, sms, push
    recipient: str
    subject: str
    message: str
    priority: str = "normal"  # low, normal, high, critical

class APIKey(BaseModel):
    """API ключ"""
    key_id: str
    name: str
    permissions: List[str]
    created_at: datetime
    expires_at: Optional[datetime] = None
    is_active: bool = True

@dataclass
class WebSocketConnection:
    """WebSocket соединение"""
    websocket: WebSocket
    client_id: str
    subscriptions: List[str]
    connected_at: datetime
    last_ping: datetime

class ExternalDataSource(ABC):
    """Абстрактный класс для внешних источников данных"""
    
    @abstractmethod
    async def fetch_data(self, symbol: str, timeframe: str = "1h") -> Dict[str, Any]:
        pass
    
    @abstractmethod
    async def get_market_status(self) -> Dict[str, Any]:
        pass

class BinanceDataSource(ExternalDataSource):
    """Источник данных Binance"""
    
    def __init__(self, api_key: str = None, api_secret: str = None):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = "https://api.binance.com"
        self.logger = logging.getLogger(__name__)
    
    async def fetch_data(self, symbol: str, timeframe: str = "1h") -> Dict[str, Any]:
        """Получение данных с Binance"""
        
        try:
            url = f"{self.base_url}/api/v3/klines"
            params = {
                'symbol': symbol.upper(),
                'interval': timeframe,
                'limit': 100
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Преобразование в удобный формат
                        formatted_data = []
                        for kline in data:
                            formatted_data.append({
                                'timestamp': datetime.fromtimestamp(kline[0] / 1000),
                                'open': float(kline[1]),
                                'high': float(kline[2]),
                                'low': float(kline[3]),
                                'close': float(kline[4]),
                                'volume': float(kline[5])
                            })
                        
                        return {
                            'symbol': symbol,
                            'timeframe': timeframe,
                            'data': formatted_data,
                            'source': 'binance'
                        }
                    else:
                        raise Exception(f"Binance API error: {response.status}")
                        
        except Exception as e:
            self.logger.error(f"Ошибка получения данных Binance: {e}")
            return {'error': str(e)}
    
    async def get_market_status(self) -> Dict[str, Any]:
        """Получение статуса рынка"""
        
        try:
            url = f"{self.base_url}/api/v3/exchangeInfo"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        return {
                            'status': data.get('status', 'UNKNOWN'),
                            'server_time': datetime.fromtimestamp(data.get('serverTime', 0) / 1000),
                            'symbols_count': len(data.get('symbols', [])),
                            'source': 'binance'
                        }
                    else:
                        raise Exception(f"Binance API error: {response.status}")
                        
        except Exception as e:
            self.logger.error(f"Ошибка получения статуса Binance: {e}")
            return {'error': str(e)}

class CoinGeckoDataSource(ExternalDataSource):
    """Источник данных CoinGecko"""
    
    def __init__(self):
        self.base_url = "https://api.coingecko.com/api/v3"
        self.logger = logging.getLogger(__name__)
    
    async def fetch_data(self, symbol: str, timeframe: str = "1h") -> Dict[str, Any]:
        """Получение данных с CoinGecko"""
        
        try:
            # Преобразование символа (например, BTCUSDT -> bitcoin)
            coin_id = self._symbol_to_coin_id(symbol)
            
            url = f"{self.base_url}/coins/{coin_id}/market_chart"
            params = {
                'vs_currency': 'usd',
                'days': '7',  # Последние 7 дней
                'interval': 'hourly' if timeframe == '1h' else 'daily'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Преобразование в удобный формат
                        prices = data.get('prices', [])
                        volumes = data.get('total_volumes', [])
                        
                        formatted_data = []
                        for i, (timestamp, price) in enumerate(prices):
                            volume = volumes[i][1] if i < len(volumes) else 0
                            formatted_data.append({
                                'timestamp': datetime.fromtimestamp(timestamp / 1000),
                                'close': price,
                                'volume': volume
                            })
                        
                        return {
                            'symbol': symbol,
                            'timeframe': timeframe,
                            'data': formatted_data,
                            'source': 'coingecko'
                        }
                    else:
                        raise Exception(f"CoinGecko API error: {response.status}")
                        
        except Exception as e:
            self.logger.error(f"Ошибка получения данных CoinGecko: {e}")
            return {'error': str(e)}
    
    async def get_market_status(self) -> Dict[str, Any]:
        """Получение статуса рынка"""
        
        try:
            url = f"{self.base_url}/global"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        global_data = data.get('data', {})
                        
                        return {
                            'total_market_cap': global_data.get('total_market_cap', {}).get('usd', 0),
                            'total_volume': global_data.get('total_volume', {}).get('usd', 0),
                            'market_cap_change_24h': global_data.get('market_cap_change_percentage_24h_usd', 0),
                            'active_cryptocurrencies': global_data.get('active_cryptocurrencies', 0),
                            'source': 'coingecko'
                        }
                    else:
                        raise Exception(f"CoinGecko API error: {response.status}")
                        
        except Exception as e:
            self.logger.error(f"Ошибка получения статуса CoinGecko: {e}")
            return {'error': str(e)}
    
    def _symbol_to_coin_id(self, symbol: str) -> str:
        """Преобразование символа в ID монеты CoinGecko"""
        
        symbol_map = {
            'BTCUSDT': 'bitcoin',
            'ETHUSDT': 'ethereum',
            'ADAUSDT': 'cardano',
            'DOTUSDT': 'polkadot',
            'LINKUSDT': 'chainlink',
            'LTCUSDT': 'litecoin',
            'BCHUSDT': 'bitcoin-cash',
            'XLMUSDT': 'stellar',
            'XRPUSDT': 'ripple',
            'EOSUSDT': 'eos'
        }
        
        return symbol_map.get(symbol.upper(), 'bitcoin')

class NotificationService:
    """Сервис уведомлений"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.email_config = {
            'smtp_server': 'smtp.gmail.com',
            'smtp_port': 587,
            'username': '',  # Настроить
            'password': ''   # Настроить
        }
        self.telegram_config = {
            'bot_token': '',  # Настроить
            'chat_id': ''     # Настроить
        }
    
    async def send_notification(self, notification: NotificationRequest) -> Dict[str, Any]:
        """Отправка уведомления"""
        
        try:
            if notification.type == 'email':
                return await self._send_email(notification)
            elif notification.type == 'telegram':
                return await self._send_telegram(notification)
            elif notification.type == 'push':
                return await self._send_push_notification(notification)
            else:
                raise ValueError(f"Неподдерживаемый тип уведомления: {notification.type}")
                
        except Exception as e:
            self.logger.error(f"Ошибка отправки уведомления: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _send_email(self, notification: NotificationRequest) -> Dict[str, Any]:
        """Отправка email уведомления"""
        
        try:
            if not self.email_config['username'] or not self.email_config['password']:
                return {'success': False, 'error': 'Email не настроен'}
            
            msg = MIMEMultipart()
            msg['From'] = self.email_config['username']
            msg['To'] = notification.recipient
            msg['Subject'] = notification.subject
            
            msg.attach(MIMEText(notification.message, 'plain'))
            
            server = smtplib.SMTP(self.email_config['smtp_server'], self.email_config['smtp_port'])
            server.starttls()
            server.login(self.email_config['username'], self.email_config['password'])
            
            text = msg.as_string()
            server.sendmail(self.email_config['username'], notification.recipient, text)
            server.quit()
            
            return {'success': True, 'message': 'Email отправлен успешно'}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def _send_telegram(self, notification: NotificationRequest) -> Dict[str, Any]:
        """Отправка Telegram уведомления"""
        
        try:
            if not self.telegram_config['bot_token']:
                return {'success': False, 'error': 'Telegram не настроен'}
            
            url = f"https://api.telegram.org/bot{self.telegram_config['bot_token']}/sendMessage"
            
            data = {
                'chat_id': notification.recipient or self.telegram_config['chat_id'],
                'text': f"*{notification.subject}*\n\n{notification.message}",
                'parse_mode': 'Markdown'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=data) as response:
                    if response.status == 200:
                        return {'success': True, 'message': 'Telegram сообщение отправлено'}
                    else:
                        return {'success': False, 'error': f'Telegram API error: {response.status}'}
                        
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def _send_push_notification(self, notification: NotificationRequest) -> Dict[str, Any]:
        """Отправка push уведомления (заглушка)"""
        
        # Здесь можно интегрировать с Firebase Cloud Messaging или другим сервисом
        self.logger.info(f"Push уведомление: {notification.subject} - {notification.message}")
        return {'success': True, 'message': 'Push уведомление отправлено (симуляция)'}

class APIKeyManager:
    """Менеджер API ключей"""
    
    def __init__(self, db_path: str = "api_keys.db"):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self._init_database()
    
    def _init_database(self):
        """Инициализация базы данных"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS api_keys (
                key_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                key_hash TEXT NOT NULL,
                permissions TEXT NOT NULL,
                created_at TIMESTAMP NOT NULL,
                expires_at TIMESTAMP,
                is_active BOOLEAN NOT NULL DEFAULT 1,
                last_used TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def generate_api_key(self, name: str, permissions: List[str], expires_days: int = None) -> Dict[str, str]:
        """Генерация нового API ключа"""
        
        # Генерация ключа
        key_id = f"ak_{int(time.time())}_{hashlib.md5(name.encode()).hexdigest()[:8]}"
        api_key = f"sk_{hashlib.sha256(f"{key_id}{time.time()}".encode()).hexdigest()}"
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        
        # Расчет срока действия
        expires_at = None
        if expires_days:
            expires_at = datetime.now() + timedelta(days=expires_days)
        
        # Сохранение в базу
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO api_keys (key_id, name, key_hash, permissions, created_at, expires_at, is_active)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            key_id,
            name,
            key_hash,
            json.dumps(permissions),
            datetime.now(),
            expires_at,
            True
        ))
        
        conn.commit()
        conn.close()
        
        self.logger.info(f"Создан новый API ключ: {key_id}")
        
        return {
            'key_id': key_id,
            'api_key': api_key,
            'permissions': permissions,
            'expires_at': expires_at.isoformat() if expires_at else None
        }
    
    def validate_api_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        """Валидация API ключа"""
        
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT key_id, name, permissions, expires_at, is_active
            FROM api_keys
            WHERE key_hash = ?
        ''', (key_hash,))
        
        result = cursor.fetchone()
        
        if result:
            key_id, name, permissions_json, expires_at, is_active = result
            
            # Проверка активности
            if not is_active:
                conn.close()
                return None
            
            # Проверка срока действия
            if expires_at:
                expires_datetime = datetime.fromisoformat(expires_at)
                if datetime.now() > expires_datetime:
                    conn.close()
                    return None
            
            # Обновление времени последнего использования
            cursor.execute('''
                UPDATE api_keys SET last_used = ? WHERE key_hash = ?
            ''', (datetime.now(), key_hash))
            
            conn.commit()
            conn.close()
            
            return {
                'key_id': key_id,
                'name': name,
                'permissions': json.loads(permissions_json),
                'expires_at': expires_at
            }
        
        conn.close()
        return None

class WebSocketManager:
    """Менеджер WebSocket соединений"""
    
    def __init__(self):
        self.connections: Dict[str, WebSocketConnection] = {}
        self.logger = logging.getLogger(__name__)
        self.message_queue = queue.Queue()
        
        # Запуск фонового процесса для отправки сообщений
        self._start_message_sender()
    
    def _start_message_sender(self):
        """Запуск фонового процесса отправки сообщений"""
        
        def message_sender():
            while True:
                try:
                    if not self.message_queue.empty():
                        message_data = self.message_queue.get(timeout=1)
                        asyncio.create_task(self._broadcast_message(message_data))
                    time.sleep(0.1)
                except queue.Empty:
                    continue
                except Exception as e:
                    self.logger.error(f"Ошибка отправки сообщений: {e}")
        
        thread = threading.Thread(target=message_sender, daemon=True)
        thread.start()
    
    async def connect(self, websocket: WebSocket, client_id: str) -> bool:
        """Подключение нового клиента"""
        
        try:
            await websocket.accept()
            
            connection = WebSocketConnection(
                websocket=websocket,
                client_id=client_id,
                subscriptions=[],
                connected_at=datetime.now(),
                last_ping=datetime.now()
            )
            
            self.connections[client_id] = connection
            self.logger.info(f"WebSocket клиент подключен: {client_id}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Ошибка подключения WebSocket: {e}")
            return False
    
    async def disconnect(self, client_id: str):
        """Отключение клиента"""
        
        if client_id in self.connections:
            try:
                connection = self.connections[client_id]
                await connection.websocket.close()
            except:
                pass
            
            del self.connections[client_id]
            self.logger.info(f"WebSocket клиент отключен: {client_id}")
    
    async def subscribe(self, client_id: str, channels: List[str]) -> bool:
        """Подписка на каналы"""
        
        if client_id not in self.connections:
            return False
        
        connection = self.connections[client_id]
        for channel in channels:
            if channel not in connection.subscriptions:
                connection.subscriptions.append(channel)
        
        self.logger.info(f"Клиент {client_id} подписан на каналы: {channels}")
        return True
    
    async def unsubscribe(self, client_id: str, channels: List[str]) -> bool:
        """Отписка от каналов"""
        
        if client_id not in self.connections:
            return False
        
        connection = self.connections[client_id]
        for channel in channels:
            if channel in connection.subscriptions:
                connection.subscriptions.remove(channel)
        
        self.logger.info(f"Клиент {client_id} отписан от каналов: {channels}")
        return True
    
    def broadcast_to_channel(self, channel: str, message: Dict[str, Any]):
        """Отправка сообщения в канал"""
        
        message_data = {
            'channel': channel,
            'message': message,
            'timestamp': datetime.now().isoformat()
        }
        
        self.message_queue.put(message_data)
    
    async def _broadcast_message(self, message_data: Dict[str, Any]):
        """Отправка сообщения подписанным клиентам"""
        
        channel = message_data['channel']
        message = message_data['message']
        
        disconnected_clients = []
        
        for client_id, connection in self.connections.items():
            if channel in connection.subscriptions:
                try:
                    await connection.websocket.send_text(json.dumps({
                        'channel': channel,
                        'data': message,
                        'timestamp': message_data['timestamp']
                    }))
                except Exception as e:
                    self.logger.error(f"Ошибка отправки сообщения клиенту {client_id}: {e}")
                    disconnected_clients.append(client_id)
        
        # Удаление отключенных клиентов
        for client_id in disconnected_clients:
            await self.disconnect(client_id)
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Получение статистики соединений"""
        
        return {
            'total_connections': len(self.connections),
            'connections': [
                {
                    'client_id': client_id,
                    'connected_at': conn.connected_at.isoformat(),
                    'subscriptions': conn.subscriptions,
                    'last_ping': conn.last_ping.isoformat()
                }
                for client_id, conn in self.connections.items()
            ]
        }

# Создание FastAPI приложения
app = FastAPI(
    title="Trading System API",
    description="RESTful API для торговой системы с WebSocket поддержкой",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Инициализация сервисов
api_key_manager = APIKeyManager()
notification_service = NotificationService()
websocket_manager = WebSocketManager()
data_sources = {
    'binance': BinanceDataSource(),
    'coingecko': CoinGeckoDataSource()
}

# Security
security = HTTPBearer()

async def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict[str, Any]:
    """Проверка API ключа"""
    
    api_key = credentials.credentials
    key_info = api_key_manager.validate_api_key(api_key)
    
    if not key_info:
        raise HTTPException(status_code=401, detail="Недействительный API ключ")
    
    return key_info

# REST API endpoints
@app.get("/")
async def root():
    """Корневой endpoint"""
    return {
        "message": "Trading System API",
        "version": "1.0.0",
        "status": "running",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/v1/status")
async def get_system_status(key_info: Dict = Depends(verify_api_key)) -> SystemStatus:
    """Получение статуса системы"""
    
    return SystemStatus(
        timestamp=datetime.now(),
        status="RUNNING",
        uptime=time.time(),  # Упрощенно
        active_symbols=["BTCUSDT", "ETHUSDT"],  # Заглушка
        performance_metrics={
            "win_rate": 0.75,
            "roi": 0.08,
            "max_drawdown": 0.05,
            "sharpe_ratio": 1.5
        }
    )

@app.get("/api/v1/market-data/{symbol}")
async def get_market_data(
    symbol: str,
    source: str = "binance",
    timeframe: str = "1h",
    key_info: Dict = Depends(verify_api_key)
) -> Dict[str, Any]:
    """Получение рыночных данных"""
    
    if 'read_market_data' not in key_info['permissions']:
        raise HTTPException(status_code=403, detail="Недостаточно прав")
    
    if source not in data_sources:
        raise HTTPException(status_code=400, detail=f"Неподдерживаемый источник: {source}")
    
    data_source = data_sources[source]
    result = await data_source.fetch_data(symbol, timeframe)
    
    if 'error' in result:
        raise HTTPException(status_code=500, detail=result['error'])
    
    return result

@app.post("/api/v1/notifications")
async def send_notification(
    notification: NotificationRequest,
    background_tasks: BackgroundTasks,
    key_info: Dict = Depends(verify_api_key)
) -> Dict[str, Any]:
    """Отправка уведомления"""
    
    if 'send_notifications' not in key_info['permissions']:
        raise HTTPException(status_code=403, detail="Недостаточно прав")
    
    # Отправка в фоне
    background_tasks.add_task(notification_service.send_notification, notification)
    
    return {
        "message": "Уведомление поставлено в очередь",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/v1/api-keys")
async def create_api_key(
    name: str,
    permissions: List[str],
    expires_days: Optional[int] = None,
    key_info: Dict = Depends(verify_api_key)
) -> Dict[str, Any]:
    """Создание нового API ключа"""
    
    if 'manage_api_keys' not in key_info['permissions']:
        raise HTTPException(status_code=403, detail="Недостаточно прав")
    
    result = api_key_manager.generate_api_key(name, permissions, expires_days)
    return result

@app.get("/api/v1/websocket/stats")
async def get_websocket_stats(key_info: Dict = Depends(verify_api_key)) -> Dict[str, Any]:
    """Получение статистики WebSocket соединений"""
    
    if 'read_system_stats' not in key_info['permissions']:
        raise HTTPException(status_code=403, detail="Недостаточно прав")
    
    return websocket_manager.get_connection_stats()

# WebSocket endpoint
@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """WebSocket endpoint для real-time данных"""
    
    if not await websocket_manager.connect(websocket, client_id):
        await websocket.close(code=1000)
        return
    
    try:
        while True:
            # Получение сообщений от клиента
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # Обработка команд
            if message.get('action') == 'subscribe':
                channels = message.get('channels', [])
                await websocket_manager.subscribe(client_id, channels)
                
                await websocket.send_text(json.dumps({
                    'type': 'subscription_success',
                    'channels': channels,
                    'timestamp': datetime.now().isoformat()
                }))
            
            elif message.get('action') == 'unsubscribe':
                channels = message.get('channels', [])
                await websocket_manager.unsubscribe(client_id, channels)
                
                await websocket.send_text(json.dumps({
                    'type': 'unsubscription_success',
                    'channels': channels,
                    'timestamp': datetime.now().isoformat()
                }))
            
            elif message.get('action') == 'ping':
                await websocket.send_text(json.dumps({
                    'type': 'pong',
                    'timestamp': datetime.now().isoformat()
                }))
                
                # Обновление времени последнего пинга
                if client_id in websocket_manager.connections:
                    websocket_manager.connections[client_id].last_ping = datetime.now()
    
    except Exception as e:
        logging.getLogger(__name__).error(f"WebSocket ошибка для клиента {client_id}: {e}")
    
    finally:
        await websocket_manager.disconnect(client_id)

# Функция для тестирования API
async def test_integrations_api():
    """Тестирование системы интеграций и API"""
    
    print("=== Тестирование системы интеграций и API ===")
    
    # Тестирование API ключей
    print("\n1. Тестирование API ключей...")
    key_manager = APIKeyManager()
    
    # Создание тестового ключа
    test_key = key_manager.generate_api_key(
        name="test_key",
        permissions=["read_market_data", "send_notifications"],
        expires_days=30
    )
    print(f"Создан тестовый API ключ: {test_key['key_id']}")
    
    # Валидация ключа
    validation_result = key_manager.validate_api_key(test_key['api_key'])
    print(f"Валидация ключа: {'Успешно' if validation_result else 'Ошибка'}")
    
    # Тестирование источников данных
    print("\n2. Тестирование источников данных...")
    
    # Binance
    binance_source = BinanceDataSource()
    binance_data = await binance_source.fetch_data("BTCUSDT", "1h")
    print(f"Binance данные: {'Получены' if 'data' in binance_data else 'Ошибка'}")
    
    binance_status = await binance_source.get_market_status()
    print(f"Binance статус: {'Получен' if 'status' in binance_status else 'Ошибка'}")
    
    # CoinGecko
    coingecko_source = CoinGeckoDataSource()
    coingecko_data = await coingecko_source.fetch_data("BTCUSDT", "1h")
    print(f"CoinGecko данные: {'Получены' if 'data' in coingecko_data else 'Ошибка'}")
    
    coingecko_status = await coingecko_source.get_market_status()
    print(f"CoinGecko статус: {'Получен' if 'total_market_cap' in coingecko_status else 'Ошибка'}")
    
    # Тестирование уведомлений
    print("\n3. Тестирование уведомлений...")
    notification_service = NotificationService()
    
    test_notification = NotificationRequest(
        type="push",  # Используем push для тестирования (симуляция)
        recipient="test@example.com",
        subject="Тестовое уведомление",
        message="Это тестовое сообщение от торговой системы",
        priority="normal"
    )
    
    notification_result = await notification_service.send_notification(test_notification)
    print(f"Уведомление: {'Отправлено' if notification_result.get('success') else 'Ошибка'}")
    
    # Тестирование WebSocket менеджера
    print("\n4. Тестирование WebSocket менеджера...")
    ws_manager = WebSocketManager()
    
    # Симуляция подключения (без реального WebSocket)
    print("WebSocket менеджер инициализирован")
    
    # Тестирование broadcast
    ws_manager.broadcast_to_channel("market_data", {
        "symbol": "BTCUSDT",
        "price": 45000,
        "change": 2.5
    })
    print("Тестовое сообщение отправлено в канал")
    
    stats = ws_manager.get_connection_stats()
    print(f"WebSocket статистика: {stats['total_connections']} соединений")
    
    print("\n=== Тестирование завершено ===")
    
    return {
        'api_keys': validation_result is not None,
        'binance_data': 'data' in binance_data,
        'coingecko_data': 'data' in coingecko_data,
        'notifications': notification_result.get('success', False),
        'websocket': True
    }

if __name__ == "__main__":
    # Настройка логирования
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Запуск тестирования
    import asyncio
    asyncio.run(test_integrations_api())
    
    # Запуск сервера (раскомментировать для продакшена)
    # uvicorn.run(app, host="0.0.0.0", port=8000)