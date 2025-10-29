"""
Enterprise Social Trading Web Interface - Веб-интерфейс для социального трейдинга
Предоставляет REST API и веб-интерфейс для платформы социального трейдинга
"""

import asyncio
import json
import time
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
import jwt
import bcrypt
import uuid
from pathlib import Path

# Импорт компонентов социального трейдинга
from social_trading_platform import (
    EnterpriseSocialTradingPlatform,
    TraderProfile, TradingSignal, SocialPost, CopyTradingSettings,
    TraderTier, SignalType, PostType, CopyMode, FollowType
)

# Модели Pydantic для API
class TraderProfileCreate(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    display_name: str = Field(..., min_length=1, max_length=100)
    bio: Optional[str] = Field(None, max_length=500)
    avatar_url: Optional[str] = None
    allow_copying: bool = True
    copy_fee_percentage: float = Field(0.0, ge=0.0, le=50.0)
    min_copy_amount: float = Field(100.0, ge=1.0)
    max_copiers: int = Field(1000, ge=1, le=10000)

class TradingSignalCreate(BaseModel):
    signal_type: SignalType
    symbol: str = Field(..., min_length=1, max_length=20)
    price: float = Field(..., gt=0)
    quantity: Optional[float] = Field(None, gt=0)
    stop_loss: Optional[float] = Field(None, gt=0)
    take_profit: Optional[float] = Field(None, gt=0)
    confidence: float = Field(50.0, ge=0.0, le=100.0)
    reasoning: Optional[str] = Field(None, max_length=1000)
    expires_in_hours: Optional[int] = Field(None, ge=1, le=168)  # До недели

class SocialPostCreate(BaseModel):
    post_type: PostType
    title: str = Field(..., min_length=1, max_length=200)
    content: str = Field(..., min_length=1, max_length=5000)
    tags: List[str] = Field(default_factory=list, max_items=10)
    related_symbols: List[str] = Field(default_factory=list, max_items=5)

class CopyTradingSettingsCreate(BaseModel):
    trader_id: str
    copy_mode: CopyMode
    copy_amount: Optional[float] = Field(None, gt=0)
    copy_percentage: Optional[float] = Field(None, gt=0, le=100)
    proportion_factor: float = Field(1.0, gt=0, le=10)
    max_daily_loss: Optional[float] = Field(None, gt=0)
    max_position_size: Optional[float] = Field(None, gt=0)
    allowed_symbols: Optional[List[str]] = None
    excluded_symbols: Optional[List[str]] = None
    stop_loss_percentage: Optional[float] = Field(None, gt=0, le=50)
    take_profit_percentage: Optional[float] = Field(None, gt=0, le=100)
    max_open_positions: int = Field(10, ge=1, le=100)

class UserLogin(BaseModel):
    username: str
    password: str

class UserRegister(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: str = Field(..., regex=r'^[^@]+@[^@]+\.[^@]+$')
    password: str = Field(..., min_length=8)
    display_name: str = Field(..., min_length=1, max_length=100)

class CommentCreate(BaseModel):
    content: str = Field(..., min_length=1, max_length=1000)
    parent_comment_id: Optional[str] = None

class AuthManager:
    """Менеджер аутентификации"""
    
    def __init__(self, secret_key: str):
        self.secret_key = secret_key
        self.algorithm = "HS256"
        self.users: Dict[str, Dict[str, Any]] = {}  # В реальности - база данных
        
    def hash_password(self, password: str) -> str:
        """Хеширование пароля"""
        return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        
    def verify_password(self, password: str, hashed: str) -> bool:
        """Проверка пароля"""
        return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))
        
    def create_token(self, user_id: str) -> str:
        """Создание JWT токена"""
        payload = {
            'user_id': user_id,
            'exp': datetime.utcnow() + timedelta(hours=24)
        }
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        
    def verify_token(self, token: str) -> Optional[str]:
        """Проверка JWT токена"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload.get('user_id')
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None
            
    def register_user(self, user_data: UserRegister) -> str:
        """Регистрация пользователя"""
        if user_data.username in self.users:
            raise ValueError("Username already exists")
            
        user_id = str(uuid.uuid4())
        self.users[user_data.username] = {
            'user_id': user_id,
            'username': user_data.username,
            'email': user_data.email,
            'password_hash': self.hash_password(user_data.password),
            'display_name': user_data.display_name,
            'created_at': datetime.now(),
            'is_active': True
        }
        
        return user_id
        
    def authenticate_user(self, username: str, password: str) -> Optional[str]:
        """Аутентификация пользователя"""
        user = self.users.get(username)
        if user and self.verify_password(password, user['password_hash']):
            return user['user_id']
        return None
        
    def get_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Получение пользователя по ID"""
        for user in self.users.values():
            if user['user_id'] == user_id:
                return user
        return None

class WebSocketManager:
    """Менеджер WebSocket соединений"""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        
    async def connect(self, websocket: WebSocket, user_id: str):
        """Подключение WebSocket"""
        await websocket.accept()
        self.active_connections[user_id] = websocket
        
    def disconnect(self, user_id: str):
        """Отключение WebSocket"""
        if user_id in self.active_connections:
            del self.active_connections[user_id]
            
    async def send_personal_message(self, message: Dict[str, Any], user_id: str):
        """Отправка персонального сообщения"""
        if user_id in self.active_connections:
            try:
                await self.active_connections[user_id].send_text(json.dumps(message))
            except:
                self.disconnect(user_id)
                
    async def broadcast(self, message: Dict[str, Any]):
        """Широковещательная отправка"""
        disconnected = []
        for user_id, connection in self.active_connections.items():
            try:
                await connection.send_text(json.dumps(message))
            except:
                disconnected.append(user_id)
                
        for user_id in disconnected:
            self.disconnect(user_id)

# Создание FastAPI приложения
app = FastAPI(
    title="Enterprise Social Trading Platform",
    description="API для платформы социального трейдинга",
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

# Глобальные переменные
social_platform: Optional[EnterpriseSocialTradingPlatform] = None
auth_manager = AuthManager("your-secret-key-here")
websocket_manager = WebSocketManager()
security = HTTPBearer()

# Статические файлы и шаблоны
templates_dir = Path(__file__).parent / "templates"
static_dir = Path(__file__).parent / "static"

templates_dir.mkdir(exist_ok=True)
static_dir.mkdir(exist_ok=True)

templates = Jinja2Templates(directory=str(templates_dir))
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Dependency для аутентификации
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """Получение текущего пользователя"""
    user_id = auth_manager.verify_token(credentials.credentials)
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user_id

# API Endpoints

@app.post("/api/auth/register")
async def register(user_data: UserRegister):
    """Регистрация пользователя"""
    try:
        user_id = auth_manager.register_user(user_data)
        token = auth_manager.create_token(user_id)
        
        return {
            "user_id": user_id,
            "token": token,
            "message": "User registered successfully"
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/auth/login")
async def login(user_data: UserLogin):
    """Вход пользователя"""
    user_id = auth_manager.authenticate_user(user_data.username, user_data.password)
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password"
        )
        
    token = auth_manager.create_token(user_id)
    user = auth_manager.get_user(user_id)
    
    return {
        "user_id": user_id,
        "token": token,
        "user": {
            "username": user["username"],
            "display_name": user["display_name"],
            "email": user["email"]
        }
    }

@app.post("/api/traders/profile")
async def create_trader_profile(
    profile_data: TraderProfileCreate,
    current_user: str = Depends(get_current_user)
):
    """Создание профиля трейдера"""
    if not social_platform:
        raise HTTPException(status_code=503, detail="Social platform not available")
        
    user = auth_manager.get_user(current_user)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
        
    profile = TraderProfile(
        user_id=current_user,
        username=profile_data.username,
        display_name=profile_data.display_name,
        tier=TraderTier.NOVICE,  # Начальный уровень
        bio=profile_data.bio,
        avatar_url=profile_data.avatar_url,
        allow_copying=profile_data.allow_copying,
        copy_fee_percentage=profile_data.copy_fee_percentage,
        min_copy_amount=profile_data.min_copy_amount,
        max_copiers=profile_data.max_copiers
    )
    
    await social_platform.create_trader_profile(profile)
    
    return {"message": "Trader profile created successfully", "profile_id": current_user}

@app.get("/api/traders/profile/{trader_id}")
async def get_trader_profile(trader_id: str):
    """Получение профиля трейдера"""
    if not social_platform:
        raise HTTPException(status_code=503, detail="Social platform not available")
        
    if trader_id not in social_platform.trader_profiles:
        raise HTTPException(status_code=404, detail="Trader not found")
        
    profile = social_platform.trader_profiles[trader_id]
    return {"profile": profile.__dict__}

@app.post("/api/signals")
async def create_signal(
    signal_data: TradingSignalCreate,
    current_user: str = Depends(get_current_user)
):
    """Создание торгового сигнала"""
    if not social_platform:
        raise HTTPException(status_code=503, detail="Social platform not available")
        
    # Проверка, что пользователь является трейдером
    if current_user not in social_platform.trader_profiles:
        raise HTTPException(status_code=403, detail="User is not a registered trader")
        
    expires_at = None
    if signal_data.expires_in_hours:
        expires_at = datetime.now() + timedelta(hours=signal_data.expires_in_hours)
        
    signal = TradingSignal(
        id=str(uuid.uuid4()),
        trader_id=current_user,
        signal_type=signal_data.signal_type,
        symbol=signal_data.symbol,
        price=signal_data.price,
        quantity=signal_data.quantity,
        stop_loss=signal_data.stop_loss,
        take_profit=signal_data.take_profit,
        confidence=signal_data.confidence,
        reasoning=signal_data.reasoning,
        expires_at=expires_at
    )
    
    signal_id = await social_platform.publish_signal(signal)
    
    return {"message": "Signal created successfully", "signal_id": signal_id}

@app.get("/api/signals")
async def get_signals(
    trader_id: Optional[str] = None,
    symbol: Optional[str] = None,
    limit: int = 50
):
    """Получение торговых сигналов"""
    if not social_platform:
        raise HTTPException(status_code=503, detail="Social platform not available")
        
    signals = list(social_platform.trading_signals.values())
    
    # Фильтрация
    if trader_id:
        signals = [s for s in signals if s.trader_id == trader_id]
    if symbol:
        signals = [s for s in signals if s.symbol == symbol]
        
    # Сортировка по времени создания
    signals.sort(key=lambda x: x.created_at, reverse=True)
    
    # Ограничение количества
    signals = signals[:limit]
    
    return {"signals": [s.__dict__ for s in signals]}

@app.post("/api/posts")
async def create_post(
    post_data: SocialPostCreate,
    current_user: str = Depends(get_current_user)
):
    """Создание социального поста"""
    if not social_platform:
        raise HTTPException(status_code=503, detail="Social platform not available")
        
    post = SocialPost(
        id=str(uuid.uuid4()),
        author_id=current_user,
        post_type=post_data.post_type,
        title=post_data.title,
        content=post_data.content,
        tags=post_data.tags,
        related_symbols=post_data.related_symbols
    )
    
    post_id = await social_platform.create_social_post(post)
    
    return {"message": "Post created successfully", "post_id": post_id}

@app.get("/api/feed")
async def get_social_feed(
    feed_type: str = "following",
    limit: int = 50,
    current_user: str = Depends(get_current_user)
):
    """Получение социальной ленты"""
    if not social_platform:
        raise HTTPException(status_code=503, detail="Social platform not available")
        
    feed = await social_platform.get_social_feed(current_user, feed_type, limit)
    
    return {"feed": feed}

@app.post("/api/posts/{post_id}/like")
async def like_post(
    post_id: str,
    current_user: str = Depends(get_current_user)
):
    """Лайк поста"""
    if not social_platform:
        raise HTTPException(status_code=503, detail="Social platform not available")
        
    social_platform.social_feed_manager.like_post(post_id, current_user)
    
    return {"message": "Post liked successfully"}

@app.post("/api/posts/{post_id}/comments")
async def add_comment(
    post_id: str,
    comment_data: CommentCreate,
    current_user: str = Depends(get_current_user)
):
    """Добавление комментария"""
    if not social_platform:
        raise HTTPException(status_code=503, detail="Social platform not available")
        
    from social_trading_platform import Comment
    
    comment = Comment(
        id=str(uuid.uuid4()),
        post_id=post_id,
        author_id=current_user,
        content=comment_data.content,
        parent_comment_id=comment_data.parent_comment_id
    )
    
    social_platform.social_feed_manager.add_comment(comment)
    
    return {"message": "Comment added successfully", "comment_id": comment.id}

@app.post("/api/follow/{trader_id}")
async def follow_trader(
    trader_id: str,
    follow_type: FollowType = FollowType.FOLLOW,
    current_user: str = Depends(get_current_user)
):
    """Подписка на трейдера"""
    if not social_platform:
        raise HTTPException(status_code=503, detail="Social platform not available")
        
    if trader_id == current_user:
        raise HTTPException(status_code=400, detail="Cannot follow yourself")
        
    if trader_id not in social_platform.trader_profiles:
        raise HTTPException(status_code=404, detail="Trader not found")
        
    follow_id = await social_platform.follow_trader(current_user, trader_id, follow_type)
    
    return {"message": "Successfully followed trader", "follow_id": follow_id}

@app.post("/api/copy-trading/setup")
async def setup_copy_trading(
    settings_data: CopyTradingSettingsCreate,
    current_user: str = Depends(get_current_user)
):
    """Настройка копи-трейдинга"""
    if not social_platform:
        raise HTTPException(status_code=503, detail="Social platform not available")
        
    if settings_data.trader_id not in social_platform.trader_profiles:
        raise HTTPException(status_code=404, detail="Trader not found")
        
    settings = CopyTradingSettings(
        id=str(uuid.uuid4()),
        follower_id=current_user,
        trader_id=settings_data.trader_id,
        copy_mode=settings_data.copy_mode,
        copy_amount=settings_data.copy_amount,
        copy_percentage=settings_data.copy_percentage,
        proportion_factor=settings_data.proportion_factor,
        max_daily_loss=settings_data.max_daily_loss,
        max_position_size=settings_data.max_position_size,
        allowed_symbols=settings_data.allowed_symbols,
        excluded_symbols=settings_data.excluded_symbols,
        stop_loss_percentage=settings_data.stop_loss_percentage,
        take_profit_percentage=settings_data.take_profit_percentage,
        max_open_positions=settings_data.max_open_positions
    )
    
    settings_id = await social_platform.setup_copy_trading(settings)
    
    return {"message": "Copy trading setup successfully", "settings_id": settings_id}

@app.get("/api/leaderboard")
async def get_leaderboard(
    period: str = "monthly",
    metric: str = "composite",
    limit: int = 50
):
    """Получение рейтинга"""
    if not social_platform:
        raise HTTPException(status_code=503, detail="Social platform not available")
        
    leaderboard = await social_platform.get_leaderboard(period, metric, limit)
    
    return {"leaderboard": leaderboard}

@app.get("/api/recommendations")
async def get_recommendations(
    limit: int = 10,
    current_user: str = Depends(get_current_user)
):
    """Получение рекомендаций"""
    if not social_platform:
        raise HTTPException(status_code=503, detail="Social platform not available")
        
    recommendations = await social_platform.get_recommendations(current_user, limit)
    
    return {"recommendations": recommendations}

# WebSocket endpoint
@app.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    """WebSocket соединение"""
    await websocket_manager.connect(websocket, user_id)
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # Обработка различных типов сообщений
            if message.get("type") == "ping":
                await websocket_manager.send_personal_message(
                    {"type": "pong", "timestamp": time.time()},
                    user_id
                )
            elif message.get("type") == "subscribe_trader":
                trader_id = message.get("trader_id")
                # Подписка на обновления трейдера
                await websocket_manager.send_personal_message(
                    {"type": "subscribed", "trader_id": trader_id},
                    user_id
                )
                
    except WebSocketDisconnect:
        websocket_manager.disconnect(user_id)

# HTML страницы
@app.get("/", response_class=HTMLResponse)
async def home_page():
    """Главная страница"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Enterprise Social Trading Platform</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <script src="https://cdn.tailwindcss.com"></script>
        <script src="https://unpkg.com/alpinejs@3.x.x/dist/cdn.min.js" defer></script>
    </head>
    <body class="bg-gray-100">
        <div class="min-h-screen">
            <!-- Header -->
            <header class="bg-blue-600 text-white shadow-lg">
                <div class="container mx-auto px-4 py-6">
                    <h1 class="text-3xl font-bold">Enterprise Social Trading Platform</h1>
                    <p class="text-blue-100 mt-2">Профессиональная платформа социального трейдинга</p>
                </div>
            </header>
            
            <!-- Main Content -->
            <main class="container mx-auto px-4 py-8">
                <div class="grid grid-cols-1 md:grid-cols-3 gap-8">
                    <!-- Features -->
                    <div class="md:col-span-2">
                        <h2 class="text-2xl font-bold mb-6">Возможности платформы</h2>
                        
                        <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                            <div class="bg-white p-6 rounded-lg shadow">
                                <h3 class="text-xl font-semibold mb-3 text-blue-600">Социальный трейдинг</h3>
                                <p class="text-gray-600">Следите за успешными трейдерами, изучайте их стратегии и взаимодействуйте с сообществом.</p>
                            </div>
                            
                            <div class="bg-white p-6 rounded-lg shadow">
                                <h3 class="text-xl font-semibold mb-3 text-green-600">Копи-трейдинг</h3>
                                <p class="text-gray-600">Автоматически копируйте сделки профессиональных трейдеров с настраиваемыми параметрами риска.</p>
                            </div>
                            
                            <div class="bg-white p-6 rounded-lg shadow">
                                <h3 class="text-xl font-semibold mb-3 text-purple-600">Торговые сигналы</h3>
                                <p class="text-gray-600">Получайте торговые сигналы в реальном времени от проверенных трейдеров.</p>
                            </div>
                            
                            <div class="bg-white p-6 rounded-lg shadow">
                                <h3 class="text-xl font-semibold mb-3 text-orange-600">Рейтинги и аналитика</h3>
                                <p class="text-gray-600">Изучайте рейтинги трейдеров и детальную аналитику их торговых результатов.</p>
                            </div>
                        </div>
                    </div>
                    
                    <!-- Quick Stats -->
                    <div>
                        <h2 class="text-2xl font-bold mb-6">Статистика платформы</h2>
                        
                        <div class="space-y-4">
                            <div class="bg-white p-4 rounded-lg shadow">
                                <div class="text-2xl font-bold text-blue-600">1,250+</div>
                                <div class="text-gray-600">Активных трейдеров</div>
                            </div>
                            
                            <div class="bg-white p-4 rounded-lg shadow">
                                <div class="text-2xl font-bold text-green-600">78.5%</div>
                                <div class="text-gray-600">Средний Win Rate</div>
                            </div>
                            
                            <div class="bg-white p-4 rounded-lg shadow">
                                <div class="text-2xl font-bold text-purple-600">$2.8M</div>
                                <div class="text-gray-600">Общий объем торгов</div>
                            </div>
                            
                            <div class="bg-white p-4 rounded-lg shadow">
                                <div class="text-2xl font-bold text-orange-600">24/7</div>
                                <div class="text-gray-600">Мониторинг рынка</div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- API Documentation -->
                <div class="mt-12">
                    <h2 class="text-2xl font-bold mb-6">API Endpoints</h2>
                    
                    <div class="bg-white rounded-lg shadow overflow-hidden">
                        <div class="p-6">
                            <h3 class="text-lg font-semibold mb-4">Основные API методы:</h3>
                            
                            <div class="space-y-3">
                                <div class="flex items-center justify-between p-3 bg-gray-50 rounded">
                                    <span class="font-mono text-sm">POST /api/auth/register</span>
                                    <span class="text-sm text-gray-600">Регистрация пользователя</span>
                                </div>
                                
                                <div class="flex items-center justify-between p-3 bg-gray-50 rounded">
                                    <span class="font-mono text-sm">POST /api/auth/login</span>
                                    <span class="text-sm text-gray-600">Вход в систему</span>
                                </div>
                                
                                <div class="flex items-center justify-between p-3 bg-gray-50 rounded">
                                    <span class="font-mono text-sm">POST /api/signals</span>
                                    <span class="text-sm text-gray-600">Создание торгового сигнала</span>
                                </div>
                                
                                <div class="flex items-center justify-between p-3 bg-gray-50 rounded">
                                    <span class="font-mono text-sm">GET /api/leaderboard</span>
                                    <span class="text-sm text-gray-600">Получение рейтинга трейдеров</span>
                                </div>
                                
                                <div class="flex items-center justify-between p-3 bg-gray-50 rounded">
                                    <span class="font-mono text-sm">POST /api/copy-trading/setup</span>
                                    <span class="text-sm text-gray-600">Настройка копи-трейдинга</span>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </main>
            
            <!-- Footer -->
            <footer class="bg-gray-800 text-white py-8 mt-12">
                <div class="container mx-auto px-4 text-center">
                    <p>&copy; 2024 Enterprise Social Trading Platform. Все права защищены.</p>
                    <p class="text-gray-400 mt-2">Профессиональная платформа для социального трейдинга</p>
                </div>
            </footer>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/docs-custom", response_class=HTMLResponse)
async def custom_docs():
    """Кастомная документация API"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Social Trading API Documentation</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <script src="https://cdn.tailwindcss.com"></script>
    </head>
    <body class="bg-gray-100">
        <div class="container mx-auto px-4 py-8">
            <h1 class="text-3xl font-bold mb-8">Social Trading API Documentation</h1>
            
            <div class="bg-white rounded-lg shadow p-6">
                <h2 class="text-2xl font-semibold mb-4">Аутентификация</h2>
                <p class="mb-4">Все защищенные endpoints требуют Bearer токен в заголовке Authorization.</p>
                
                <h3 class="text-lg font-semibold mb-2">Регистрация</h3>
                <div class="bg-gray-100 p-4 rounded mb-4">
                    <code>POST /api/auth/register</code>
                </div>
                
                <h3 class="text-lg font-semibold mb-2">Вход</h3>
                <div class="bg-gray-100 p-4 rounded mb-4">
                    <code>POST /api/auth/login</code>
                </div>
                
                <h2 class="text-2xl font-semibold mb-4 mt-8">Торговые сигналы</h2>
                
                <h3 class="text-lg font-semibold mb-2">Создание сигнала</h3>
                <div class="bg-gray-100 p-4 rounded mb-4">
                    <code>POST /api/signals</code>
                </div>
                
                <h3 class="text-lg font-semibold mb-2">Получение сигналов</h3>
                <div class="bg-gray-100 p-4 rounded mb-4">
                    <code>GET /api/signals?trader_id=&symbol=&limit=50</code>
                </div>
                
                <h2 class="text-2xl font-semibold mb-4 mt-8">Социальные функции</h2>
                
                <h3 class="text-lg font-semibold mb-2">Создание поста</h3>
                <div class="bg-gray-100 p-4 rounded mb-4">
                    <code>POST /api/posts</code>
                </div>
                
                <h3 class="text-lg font-semibold mb-2">Получение ленты</h3>
                <div class="bg-gray-100 p-4 rounded mb-4">
                    <code>GET /api/feed?feed_type=following&limit=50</code>
                </div>
                
                <h2 class="text-2xl font-semibold mb-4 mt-8">Копи-трейдинг</h2>
                
                <h3 class="text-lg font-semibold mb-2">Настройка копирования</h3>
                <div class="bg-gray-100 p-4 rounded mb-4">
                    <code>POST /api/copy-trading/setup</code>
                </div>
                
                <h3 class="text-lg font-semibold mb-2">Подписка на трейдера</h3>
                <div class="bg-gray-100 p-4 rounded mb-4">
                    <code>POST /api/follow/{trader_id}</code>
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

# Startup event
@app.on_event("startup")
async def startup_event():
    """Инициализация при запуске"""
    global social_platform
    
    config = {
        'redis_host': 'localhost',
        'redis_port': 6379,
        'websocket_host': 'localhost',
        'websocket_port': 8765
    }
    
    social_platform = EnterpriseSocialTradingPlatform(config)
    await social_platform.start()
    
    print("Social Trading Web Interface started")

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Очистка при остановке"""
    if social_platform:
        await social_platform.stop()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)