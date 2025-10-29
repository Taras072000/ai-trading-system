"""
Enterprise Federated Learning System - Система федеративного обучения
Обеспечивает распределенное обучение моделей без передачи данных
"""

import asyncio
import json
import time
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import logging
import redis.asyncio as redis
import hashlib
import pickle
import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import tensorflow as tf
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from prometheus_client import Counter, Histogram, Gauge
import websockets
import ssl
from collections import defaultdict, deque

class ClientType(Enum):
    """Типы клиентов федеративного обучения"""
    RETAIL = "retail"
    PROFESSIONAL = "professional"
    INSTITUTIONAL = "institutional"
    EXCHANGE = "exchange"
    DATA_PROVIDER = "data_provider"

class ModelType(Enum):
    """Типы моделей"""
    PRICE_PREDICTION = "price_prediction"
    RISK_ASSESSMENT = "risk_assessment"
    ANOMALY_DETECTION = "anomaly_detection"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    PORTFOLIO_OPTIMIZATION = "portfolio_optimization"

class AggregationMethod(Enum):
    """Методы агрегации"""
    FEDERATED_AVERAGING = "federated_averaging"
    WEIGHTED_AVERAGING = "weighted_averaging"
    SECURE_AGGREGATION = "secure_aggregation"
    DIFFERENTIAL_PRIVACY = "differential_privacy"

@dataclass
class FederatedClient:
    """Клиент федеративного обучения"""
    id: str
    name: str
    client_type: ClientType
    public_key: str
    last_seen: datetime
    data_samples: int
    model_version: int
    performance_metrics: Dict[str, float]
    reputation_score: float
    is_active: bool = True
    contribution_weight: float = 1.0
    privacy_budget: float = 1.0
    
@dataclass
class ModelUpdate:
    """Обновление модели"""
    client_id: str
    model_type: ModelType
    round_number: int
    parameters: Dict[str, Any]
    gradients: Optional[Dict[str, Any]] = None
    metrics: Dict[str, float] = None
    data_samples: int = 0
    timestamp: datetime = None
    signature: str = ""
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

@dataclass
class FederatedRound:
    """Раунд федеративного обучения"""
    round_number: int
    model_type: ModelType
    participants: List[str]
    start_time: datetime
    end_time: Optional[datetime] = None
    global_model: Optional[Dict[str, Any]] = None
    aggregated_metrics: Dict[str, float] = None
    convergence_score: float = 0.0
    
# Метрики
FL_ROUNDS_TOTAL = Counter('fl_rounds_total', 'Total federated learning rounds', ['model_type'])
FL_CLIENTS_ACTIVE = Gauge('fl_clients_active', 'Active federated learning clients', ['client_type'])
FL_MODEL_ACCURACY = Gauge('fl_model_accuracy', 'Federated model accuracy', ['model_type'])
FL_AGGREGATION_TIME = Histogram('fl_aggregation_time_seconds', 'Model aggregation time', ['method'])
FL_COMMUNICATION_OVERHEAD = Histogram('fl_communication_overhead_bytes', 'Communication overhead')

class SecureAggregator:
    """Безопасный агрегатор для федеративного обучения"""
    
    def __init__(self, privacy_budget: float = 1.0):
        self.privacy_budget = privacy_budget
        self.noise_multiplier = 1.0
        
    def federated_averaging(self, updates: List[ModelUpdate]) -> Dict[str, Any]:
        """Федеративное усреднение"""
        if not updates:
            return {}
            
        # Вычисление весов на основе количества данных
        total_samples = sum(update.data_samples for update in updates)
        weights = [update.data_samples / total_samples for update in updates]
        
        # Агрегация параметров
        aggregated_params = {}
        
        # Получение структуры параметров из первого обновления
        first_params = updates[0].parameters
        
        for param_name in first_params.keys():
            # Взвешенное усреднение
            weighted_sum = None
            for i, update in enumerate(updates):
                param_value = np.array(update.parameters[param_name])
                if weighted_sum is None:
                    weighted_sum = weights[i] * param_value
                else:
                    weighted_sum += weights[i] * param_value
                    
            aggregated_params[param_name] = weighted_sum.tolist()
            
        return aggregated_params
        
    def weighted_averaging(self, updates: List[ModelUpdate], 
                          client_weights: Dict[str, float]) -> Dict[str, Any]:
        """Взвешенное усреднение с учетом репутации клиентов"""
        if not updates:
            return {}
            
        # Нормализация весов
        total_weight = sum(client_weights.get(update.client_id, 1.0) for update in updates)
        normalized_weights = [
            client_weights.get(update.client_id, 1.0) / total_weight 
            for update in updates
        ]
        
        # Агрегация параметров
        aggregated_params = {}
        first_params = updates[0].parameters
        
        for param_name in first_params.keys():
            weighted_sum = None
            for i, update in enumerate(updates):
                param_value = np.array(update.parameters[param_name])
                if weighted_sum is None:
                    weighted_sum = normalized_weights[i] * param_value
                else:
                    weighted_sum += normalized_weights[i] * param_value
                    
            aggregated_params[param_name] = weighted_sum.tolist()
            
        return aggregated_params
        
    def differential_privacy_aggregation(self, updates: List[ModelUpdate],
                                       epsilon: float = 1.0) -> Dict[str, Any]:
        """Агрегация с дифференциальной приватностью"""
        if not updates:
            return {}
            
        # Стандартное федеративное усреднение
        aggregated_params = self.federated_averaging(updates)
        
        # Добавление шума для дифференциальной приватности
        sensitivity = self._calculate_sensitivity(updates)
        noise_scale = sensitivity / epsilon
        
        for param_name, param_value in aggregated_params.items():
            param_array = np.array(param_value)
            noise = np.random.laplace(0, noise_scale, param_array.shape)
            aggregated_params[param_name] = (param_array + noise).tolist()
            
        return aggregated_params
        
    def _calculate_sensitivity(self, updates: List[ModelUpdate]) -> float:
        """Расчет чувствительности для дифференциальной приватности"""
        # Упрощенный расчет чувствительности
        max_norm = 0.0
        
        for update in updates:
            param_norm = 0.0
            for param_value in update.parameters.values():
                param_array = np.array(param_value)
                param_norm += np.linalg.norm(param_array) ** 2
            param_norm = np.sqrt(param_norm)
            max_norm = max(max_norm, param_norm)
            
        return max_norm / len(updates)

class ModelManager:
    """Менеджер моделей федеративного обучения"""
    
    def __init__(self):
        self.models: Dict[ModelType, Any] = {}
        self.model_versions: Dict[ModelType, int] = {}
        
    def create_model(self, model_type: ModelType, config: Dict[str, Any]) -> Any:
        """Создание модели"""
        if model_type == ModelType.PRICE_PREDICTION:
            return self._create_price_prediction_model(config)
        elif model_type == ModelType.RISK_ASSESSMENT:
            return self._create_risk_assessment_model(config)
        elif model_type == ModelType.ANOMALY_DETECTION:
            return self._create_anomaly_detection_model(config)
        elif model_type == ModelType.SENTIMENT_ANALYSIS:
            return self._create_sentiment_analysis_model(config)
        elif model_type == ModelType.PORTFOLIO_OPTIMIZATION:
            return self._create_portfolio_optimization_model(config)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
            
    def _create_price_prediction_model(self, config: Dict[str, Any]) -> nn.Module:
        """Создание модели предсказания цен"""
        class PricePredictionModel(nn.Module):
            def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2):
                super().__init__()
                self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
                self.fc = nn.Linear(hidden_size, 1)
                self.dropout = nn.Dropout(0.2)
                
            def forward(self, x):
                lstm_out, _ = self.lstm(x)
                output = self.fc(self.dropout(lstm_out[:, -1, :]))
                return output
                
        return PricePredictionModel(
            input_size=config.get('input_size', 10),
            hidden_size=config.get('hidden_size', 128),
            num_layers=config.get('num_layers', 2)
        )
        
    def _create_risk_assessment_model(self, config: Dict[str, Any]) -> nn.Module:
        """Создание модели оценки рисков"""
        class RiskAssessmentModel(nn.Module):
            def __init__(self, input_size: int, hidden_size: int = 64):
                super().__init__()
                self.fc1 = nn.Linear(input_size, hidden_size)
                self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
                self.fc3 = nn.Linear(hidden_size // 2, 3)  # Low, Medium, High risk
                self.relu = nn.ReLU()
                self.dropout = nn.Dropout(0.3)
                
            def forward(self, x):
                x = self.relu(self.fc1(x))
                x = self.dropout(x)
                x = self.relu(self.fc2(x))
                x = self.dropout(x)
                x = self.fc3(x)
                return torch.softmax(x, dim=1)
                
        return RiskAssessmentModel(
            input_size=config.get('input_size', 20),
            hidden_size=config.get('hidden_size', 64)
        )
        
    def _create_anomaly_detection_model(self, config: Dict[str, Any]) -> nn.Module:
        """Создание модели обнаружения аномалий"""
        class AnomalyDetectionModel(nn.Module):
            def __init__(self, input_size: int, encoding_size: int = 32):
                super().__init__()
                # Encoder
                self.encoder = nn.Sequential(
                    nn.Linear(input_size, input_size // 2),
                    nn.ReLU(),
                    nn.Linear(input_size // 2, encoding_size),
                    nn.ReLU()
                )
                
                # Decoder
                self.decoder = nn.Sequential(
                    nn.Linear(encoding_size, input_size // 2),
                    nn.ReLU(),
                    nn.Linear(input_size // 2, input_size),
                    nn.Sigmoid()
                )
                
            def forward(self, x):
                encoded = self.encoder(x)
                decoded = self.decoder(encoded)
                return decoded
                
        return AnomalyDetectionModel(
            input_size=config.get('input_size', 50),
            encoding_size=config.get('encoding_size', 32)
        )
        
    def _create_sentiment_analysis_model(self, config: Dict[str, Any]) -> nn.Module:
        """Создание модели анализа настроений"""
        class SentimentAnalysisModel(nn.Module):
            def __init__(self, vocab_size: int, embedding_dim: int = 128, hidden_size: int = 64):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, embedding_dim)
                self.lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True)
                self.fc = nn.Linear(hidden_size, 3)  # Negative, Neutral, Positive
                
            def forward(self, x):
                embedded = self.embedding(x)
                lstm_out, _ = self.lstm(embedded)
                output = self.fc(lstm_out[:, -1, :])
                return torch.softmax(output, dim=1)
                
        return SentimentAnalysisModel(
            vocab_size=config.get('vocab_size', 10000),
            embedding_dim=config.get('embedding_dim', 128),
            hidden_size=config.get('hidden_size', 64)
        )
        
    def _create_portfolio_optimization_model(self, config: Dict[str, Any]) -> nn.Module:
        """Создание модели оптимизации портфеля"""
        class PortfolioOptimizationModel(nn.Module):
            def __init__(self, num_assets: int, hidden_size: int = 128):
                super().__init__()
                self.fc1 = nn.Linear(num_assets * 3, hidden_size)  # price, volume, volatility
                self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
                self.fc3 = nn.Linear(hidden_size // 2, num_assets)
                self.relu = nn.ReLU()
                self.dropout = nn.Dropout(0.2)
                
            def forward(self, x):
                x = self.relu(self.fc1(x))
                x = self.dropout(x)
                x = self.relu(self.fc2(x))
                x = self.dropout(x)
                x = self.fc3(x)
                return torch.softmax(x, dim=1)  # Portfolio weights
                
        return PortfolioOptimizationModel(
            num_assets=config.get('num_assets', 10),
            hidden_size=config.get('hidden_size', 128)
        )
        
    def get_model_parameters(self, model: nn.Module) -> Dict[str, Any]:
        """Получение параметров модели"""
        return {name: param.data.numpy().tolist() for name, param in model.named_parameters()}
        
    def set_model_parameters(self, model: nn.Module, parameters: Dict[str, Any]):
        """Установка параметров модели"""
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in parameters:
                    param.data = torch.tensor(parameters[name])

class EnterpriseFederatedLearningSystem:
    """Enterprise система федеративного обучения"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.redis_client = None
        self.logger = self._setup_logging()
        
        # Компоненты системы
        self.secure_aggregator = SecureAggregator(
            privacy_budget=config.get('privacy_budget', 1.0)
        )
        self.model_manager = ModelManager()
        
        # Состояние системы
        self.clients: Dict[str, FederatedClient] = {}
        self.active_rounds: Dict[str, FederatedRound] = {}
        self.model_updates: Dict[str, List[ModelUpdate]] = defaultdict(list)
        self.global_models: Dict[ModelType, Any] = {}
        
        # Конфигурация обучения
        self.min_clients_per_round = config.get('min_clients_per_round', 3)
        self.max_clients_per_round = config.get('max_clients_per_round', 100)
        self.round_timeout = config.get('round_timeout', 3600)  # 1 час
        self.convergence_threshold = config.get('convergence_threshold', 0.001)
        
        # Безопасность
        self.encryption_key = self._generate_encryption_key()
        
        # WebSocket сервер для клиентов
        self.websocket_server = None
        self.connected_clients: Dict[str, Any] = {}
        
    def _setup_logging(self) -> logging.Logger:
        """Настройка логирования"""
        logger = logging.getLogger('enterprise_federated_learning')
        logger.setLevel(logging.INFO)
        
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
        
    def _generate_encryption_key(self) -> bytes:
        """Генерация ключа шифрования"""
        password = self.config.get('encryption_password', 'default_password').encode()
        salt = b'federated_learning_salt'
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        return base64.urlsafe_b64encode(kdf.derive(password))
        
    async def start(self):
        """Запуск системы федеративного обучения"""
        # Подключение к Redis
        self.redis_client = redis.Redis(
            host=self.config.get('redis_host', 'localhost'),
            port=self.config.get('redis_port', 6379),
            decode_responses=True
        )
        
        # Инициализация глобальных моделей
        await self._initialize_global_models()
        
        # Запуск WebSocket сервера
        await self._start_websocket_server()
        
        # Запуск фоновых задач
        asyncio.create_task(self._round_coordinator())
        asyncio.create_task(self._client_monitor())
        asyncio.create_task(self._model_evaluator())
        asyncio.create_task(self._reputation_updater())
        
        self.logger.info("Enterprise Federated Learning System started")
        
    async def stop(self):
        """Остановка системы"""
        if self.websocket_server:
            self.websocket_server.close()
            await self.websocket_server.wait_closed()
            
        if self.redis_client:
            await self.redis_client.close()
            
    async def _initialize_global_models(self):
        """Инициализация глобальных моделей"""
        model_configs = self.config.get('model_configs', {})
        
        for model_type_str, config in model_configs.items():
            model_type = ModelType(model_type_str)
            model = self.model_manager.create_model(model_type, config)
            self.global_models[model_type] = model
            self.model_manager.model_versions[model_type] = 1
            
            # Сохранение начальной модели
            await self._save_global_model(model_type, model)
            
        self.logger.info(f"Initialized {len(self.global_models)} global models")
        
    async def _start_websocket_server(self):
        """Запуск WebSocket сервера"""
        async def handle_client(websocket, path):
            try:
                await self._handle_client_connection(websocket, path)
            except Exception as e:
                self.logger.error(f"WebSocket client error: {e}")
                
        # SSL контекст для безопасного соединения
        ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        # ssl_context.load_cert_chain('path/to/cert.pem', 'path/to/key.pem')
        
        self.websocket_server = await websockets.serve(
            handle_client,
            self.config.get('websocket_host', 'localhost'),
            self.config.get('websocket_port', 8765),
            # ssl=ssl_context  # Раскомментировать для HTTPS
        )
        
        self.logger.info("WebSocket server started")
        
    async def _handle_client_connection(self, websocket, path):
        """Обработка подключения клиента"""
        client_id = None
        
        try:
            # Аутентификация клиента
            auth_message = await websocket.recv()
            auth_data = json.loads(auth_message)
            
            client_id = await self._authenticate_client(auth_data)
            if not client_id:
                await websocket.send(json.dumps({'error': 'Authentication failed'}))
                return
                
            self.connected_clients[client_id] = websocket
            
            # Отправка подтверждения
            await websocket.send(json.dumps({
                'type': 'auth_success',
                'client_id': client_id
            }))
            
            # Обработка сообщений
            async for message in websocket:
                await self._handle_client_message(client_id, json.loads(message))
                
        except websockets.exceptions.ConnectionClosed:
            self.logger.info(f"Client {client_id} disconnected")
        except Exception as e:
            self.logger.error(f"Client handling error: {e}")
        finally:
            if client_id and client_id in self.connected_clients:
                del self.connected_clients[client_id]
                
    async def _authenticate_client(self, auth_data: Dict[str, Any]) -> Optional[str]:
        """Аутентификация клиента"""
        client_id = auth_data.get('client_id')
        signature = auth_data.get('signature')
        
        # Проверка подписи (упрощенная реализация)
        if client_id and signature:
            # Здесь должна быть проверка цифровой подписи
            return client_id
            
        return None
        
    async def _handle_client_message(self, client_id: str, message: Dict[str, Any]):
        """Обработка сообщения от клиента"""
        message_type = message.get('type')
        
        if message_type == 'register':
            await self._register_client(client_id, message)
        elif message_type == 'model_update':
            await self._receive_model_update(client_id, message)
        elif message_type == 'request_model':
            await self._send_global_model(client_id, message)
        elif message_type == 'heartbeat':
            await self._update_client_heartbeat(client_id)
        else:
            self.logger.warning(f"Unknown message type: {message_type}")
            
    async def _register_client(self, client_id: str, message: Dict[str, Any]):
        """Регистрация клиента"""
        client_data = message.get('client_data', {})
        
        client = FederatedClient(
            id=client_id,
            name=client_data.get('name', f'Client_{client_id}'),
            client_type=ClientType(client_data.get('client_type', 'retail')),
            public_key=client_data.get('public_key', ''),
            last_seen=datetime.now(),
            data_samples=client_data.get('data_samples', 0),
            model_version=0,
            performance_metrics={},
            reputation_score=1.0
        )
        
        self.clients[client_id] = client
        await self._save_client(client)
        
        # Отправка подтверждения
        if client_id in self.connected_clients:
            await self.connected_clients[client_id].send(json.dumps({
                'type': 'registration_success',
                'client_id': client_id
            }))
            
        self.logger.info(f"Client registered: {client_id}")
        
    async def _receive_model_update(self, client_id: str, message: Dict[str, Any]):
        """Получение обновления модели от клиента"""
        update_data = message.get('update_data', {})
        
        model_update = ModelUpdate(
            client_id=client_id,
            model_type=ModelType(update_data.get('model_type')),
            round_number=update_data.get('round_number'),
            parameters=update_data.get('parameters', {}),
            gradients=update_data.get('gradients'),
            metrics=update_data.get('metrics', {}),
            data_samples=update_data.get('data_samples', 0),
            signature=update_data.get('signature', '')
        )
        
        # Проверка подписи
        if not self._verify_update_signature(model_update):
            self.logger.warning(f"Invalid signature from client {client_id}")
            return
            
        # Сохранение обновления
        round_key = f"{model_update.model_type.value}_{model_update.round_number}"
        self.model_updates[round_key].append(model_update)
        
        await self._save_model_update(model_update)
        
        # Обновление клиента
        if client_id in self.clients:
            self.clients[client_id].last_seen = datetime.now()
            self.clients[client_id].model_version = model_update.round_number
            
        self.logger.info(f"Received model update from {client_id}")
        
    async def _send_global_model(self, client_id: str, message: Dict[str, Any]):
        """Отправка глобальной модели клиенту"""
        model_type = ModelType(message.get('model_type'))
        
        if model_type in self.global_models:
            model = self.global_models[model_type]
            parameters = self.model_manager.get_model_parameters(model)
            
            response = {
                'type': 'global_model',
                'model_type': model_type.value,
                'parameters': parameters,
                'version': self.model_manager.model_versions.get(model_type, 1)
            }
            
            if client_id in self.connected_clients:
                await self.connected_clients[client_id].send(json.dumps(response))
                
    async def _update_client_heartbeat(self, client_id: str):
        """Обновление heartbeat клиента"""
        if client_id in self.clients:
            self.clients[client_id].last_seen = datetime.now()
            
    def _verify_update_signature(self, update: ModelUpdate) -> bool:
        """Проверка подписи обновления модели"""
        # Упрощенная проверка подписи
        # В реальной реализации здесь должна быть криптографическая проверка
        return len(update.signature) > 0
        
    # === Координация раундов ===
    
    async def _round_coordinator(self):
        """Координатор раундов обучения"""
        while True:
            try:
                for model_type in self.global_models.keys():
                    await self._coordinate_round(model_type)
                    
                await asyncio.sleep(self.config.get('round_interval', 300))  # 5 минут
                
            except Exception as e:
                self.logger.error(f"Round coordinator error: {e}")
                await asyncio.sleep(60)
                
    async def _coordinate_round(self, model_type: ModelType):
        """Координация раунда обучения для модели"""
        # Проверка активных клиентов
        active_clients = [
            client for client in self.clients.values()
            if client.is_active and 
            (datetime.now() - client.last_seen).total_seconds() < 3600
        ]
        
        if len(active_clients) < self.min_clients_per_round:
            return
            
        # Выбор участников раунда
        participants = self._select_round_participants(active_clients, model_type)
        
        if len(participants) < self.min_clients_per_round:
            return
            
        # Создание нового раунда
        round_number = self.model_manager.model_versions.get(model_type, 1)
        round_key = f"{model_type.value}_{round_number}"
        
        federated_round = FederatedRound(
            round_number=round_number,
            model_type=model_type,
            participants=[p.id for p in participants],
            start_time=datetime.now()
        )
        
        self.active_rounds[round_key] = federated_round
        
        # Отправка приглашений участникам
        await self._send_round_invitations(participants, federated_round)
        
        # Ожидание обновлений
        await self._wait_for_round_completion(round_key)
        
        # Агрегация и обновление глобальной модели
        await self._aggregate_and_update_model(round_key)
        
    def _select_round_participants(self, active_clients: List[FederatedClient],
                                 model_type: ModelType) -> List[FederatedClient]:
        """Выбор участников раунда"""
        # Сортировка по репутации и количеству данных
        scored_clients = []
        for client in active_clients:
            score = (
                client.reputation_score * 0.5 +
                min(client.data_samples / 1000, 1.0) * 0.3 +
                (1.0 if client.client_type in [ClientType.INSTITUTIONAL, ClientType.PROFESSIONAL] else 0.5) * 0.2
            )
            scored_clients.append((client, score))
            
        # Сортировка по убыванию счета
        scored_clients.sort(key=lambda x: x[1], reverse=True)
        
        # Выбор топ клиентов
        max_participants = min(self.max_clients_per_round, len(scored_clients))
        selected = [client for client, _ in scored_clients[:max_participants]]
        
        return selected
        
    async def _send_round_invitations(self, participants: List[FederatedClient],
                                    federated_round: FederatedRound):
        """Отправка приглашений на участие в раунде"""
        invitation = {
            'type': 'round_invitation',
            'round_number': federated_round.round_number,
            'model_type': federated_round.model_type.value,
            'deadline': (federated_round.start_time + timedelta(seconds=self.round_timeout)).isoformat()
        }
        
        for participant in participants:
            if participant.id in self.connected_clients:
                try:
                    await self.connected_clients[participant.id].send(json.dumps(invitation))
                except Exception as e:
                    self.logger.error(f"Failed to send invitation to {participant.id}: {e}")
                    
    async def _wait_for_round_completion(self, round_key: str):
        """Ожидание завершения раунда"""
        start_time = time.time()
        
        while time.time() - start_time < self.round_timeout:
            round_updates = self.model_updates.get(round_key, [])
            federated_round = self.active_rounds[round_key]
            
            # Проверка получения обновлений от всех участников
            received_clients = {update.client_id for update in round_updates}
            expected_clients = set(federated_round.participants)
            
            if received_clients >= expected_clients:
                break
                
            await asyncio.sleep(10)
            
        # Завершение раунда
        self.active_rounds[round_key].end_time = datetime.now()
        
    async def _aggregate_and_update_model(self, round_key: str):
        """Агрегация обновлений и обновление глобальной модели"""
        start_time = time.time()
        
        try:
            round_updates = self.model_updates.get(round_key, [])
            federated_round = self.active_rounds[round_key]
            
            if not round_updates:
                self.logger.warning(f"No updates received for round {round_key}")
                return
                
            # Выбор метода агрегации
            aggregation_method = self.config.get('aggregation_method', 'federated_averaging')
            
            if aggregation_method == 'federated_averaging':
                aggregated_params = self.secure_aggregator.federated_averaging(round_updates)
            elif aggregation_method == 'weighted_averaging':
                client_weights = {
                    client_id: self.clients[client_id].reputation_score
                    for client_id in self.clients
                }
                aggregated_params = self.secure_aggregator.weighted_averaging(
                    round_updates, client_weights
                )
            elif aggregation_method == 'differential_privacy':
                aggregated_params = self.secure_aggregator.differential_privacy_aggregation(
                    round_updates, epsilon=self.config.get('privacy_epsilon', 1.0)
                )
            else:
                aggregated_params = self.secure_aggregator.federated_averaging(round_updates)
                
            # Обновление глобальной модели
            model_type = federated_round.model_type
            global_model = self.global_models[model_type]
            
            self.model_manager.set_model_parameters(global_model, aggregated_params)
            
            # Увеличение версии модели
            self.model_manager.model_versions[model_type] += 1
            
            # Сохранение обновленной модели
            await self._save_global_model(model_type, global_model)
            
            # Расчет метрик агрегации
            aggregated_metrics = self._calculate_aggregated_metrics(round_updates)
            federated_round.aggregated_metrics = aggregated_metrics
            federated_round.global_model = aggregated_params
            
            # Метрики
            FL_ROUNDS_TOTAL.labels(model_type=model_type.value).inc()
            FL_AGGREGATION_TIME.labels(method=aggregation_method).observe(time.time() - start_time)
            
            if 'accuracy' in aggregated_metrics:
                FL_MODEL_ACCURACY.labels(model_type=model_type.value).set(aggregated_metrics['accuracy'])
                
            # Уведомление участников об обновлении
            await self._notify_round_completion(federated_round)
            
            self.logger.info(f"Round {round_key} completed successfully")
            
        except Exception as e:
            self.logger.error(f"Model aggregation error: {e}")
            
        finally:
            # Очистка данных раунда
            if round_key in self.model_updates:
                del self.model_updates[round_key]
            if round_key in self.active_rounds:
                del self.active_rounds[round_key]
                
    def _calculate_aggregated_metrics(self, updates: List[ModelUpdate]) -> Dict[str, float]:
        """Расчет агрегированных метрик"""
        if not updates:
            return {}
            
        metrics = {}
        metric_names = set()
        
        # Сбор всех имен метрик
        for update in updates:
            if update.metrics:
                metric_names.update(update.metrics.keys())
                
        # Вычисление взвешенных средних
        total_samples = sum(update.data_samples for update in updates)
        
        for metric_name in metric_names:
            weighted_sum = 0.0
            for update in updates:
                if update.metrics and metric_name in update.metrics:
                    weight = update.data_samples / total_samples
                    weighted_sum += weight * update.metrics[metric_name]
                    
            metrics[metric_name] = weighted_sum
            
        return metrics
        
    async def _notify_round_completion(self, federated_round: FederatedRound):
        """Уведомление о завершении раунда"""
        notification = {
            'type': 'round_completed',
            'round_number': federated_round.round_number,
            'model_type': federated_round.model_type.value,
            'metrics': federated_round.aggregated_metrics,
            'new_version': self.model_manager.model_versions[federated_round.model_type]
        }
        
        for participant_id in federated_round.participants:
            if participant_id in self.connected_clients:
                try:
                    await self.connected_clients[participant_id].send(json.dumps(notification))
                except Exception as e:
                    self.logger.error(f"Failed to notify {participant_id}: {e}")
                    
    # === Мониторинг и оценка ===
    
    async def _client_monitor(self):
        """Мониторинг клиентов"""
        while True:
            try:
                current_time = datetime.now()
                
                for client in self.clients.values():
                    # Проверка активности
                    time_since_last_seen = (current_time - client.last_seen).total_seconds()
                    
                    if time_since_last_seen > 7200:  # 2 часа
                        client.is_active = False
                    elif time_since_last_seen < 3600:  # 1 час
                        client.is_active = True
                        
                # Обновление метрик
                for client_type in ClientType:
                    active_count = sum(
                        1 for client in self.clients.values()
                        if client.client_type == client_type and client.is_active
                    )
                    FL_CLIENTS_ACTIVE.labels(client_type=client_type.value).set(active_count)
                    
                await asyncio.sleep(300)  # 5 минут
                
            except Exception as e:
                self.logger.error(f"Client monitor error: {e}")
                await asyncio.sleep(60)
                
    async def _model_evaluator(self):
        """Оценщик моделей"""
        while True:
            try:
                # Периодическая оценка глобальных моделей
                for model_type, model in self.global_models.items():
                    await self._evaluate_global_model(model_type, model)
                    
                await asyncio.sleep(1800)  # 30 минут
                
            except Exception as e:
                self.logger.error(f"Model evaluator error: {e}")
                await asyncio.sleep(600)
                
    async def _evaluate_global_model(self, model_type: ModelType, model: nn.Module):
        """Оценка глобальной модели"""
        # Здесь должна быть логика оценки модели на тестовых данных
        # Пока что заглушка
        pass
        
    async def _reputation_updater(self):
        """Обновление репутации клиентов"""
        while True:
            try:
                for client in self.clients.values():
                    await self._update_client_reputation(client)
                    
                await asyncio.sleep(3600)  # 1 час
                
            except Exception as e:
                self.logger.error(f"Reputation updater error: {e}")
                await asyncio.sleep(1800)
                
    async def _update_client_reputation(self, client: FederatedClient):
        """Обновление репутации клиента"""
        # Факторы репутации:
        # 1. Участие в раундах
        # 2. Качество обновлений
        # 3. Стабильность подключения
        # 4. Количество данных
        
        base_score = 1.0
        
        # Бонус за активность
        if client.is_active:
            base_score += 0.1
            
        # Бонус за тип клиента
        if client.client_type == ClientType.INSTITUTIONAL:
            base_score += 0.2
        elif client.client_type == ClientType.PROFESSIONAL:
            base_score += 0.1
            
        # Бонус за количество данных
        data_bonus = min(client.data_samples / 10000, 0.3)
        base_score += data_bonus
        
        # Ограничение диапазона
        client.reputation_score = max(0.1, min(2.0, base_score))
        
        await self._save_client(client)
        
    # === Сохранение и загрузка ===
    
    async def _save_global_model(self, model_type: ModelType, model: nn.Module):
        """Сохранение глобальной модели"""
        parameters = self.model_manager.get_model_parameters(model)
        
        model_data = {
            'model_type': model_type.value,
            'parameters': parameters,
            'version': self.model_manager.model_versions[model_type],
            'timestamp': datetime.now().isoformat()
        }
        
        await self.redis_client.setex(
            f"global_model:{model_type.value}",
            86400 * 30,  # 30 дней
            json.dumps(model_data)
        )
        
    async def _save_client(self, client: FederatedClient):
        """Сохранение данных клиента"""
        await self.redis_client.setex(
            f"fl_client:{client.id}",
            86400 * 7,  # 7 дней
            json.dumps(asdict(client), default=str)
        )
        
    async def _save_model_update(self, update: ModelUpdate):
        """Сохранение обновления модели"""
        await self.redis_client.setex(
            f"model_update:{update.client_id}:{update.round_number}",
            86400 * 3,  # 3 дня
            json.dumps(asdict(update), default=str)
        )

async def main():
    """Основная функция запуска"""
    config = {
        'redis_host': 'localhost',
        'redis_port': 6379,
        'websocket_host': 'localhost',
        'websocket_port': 8765,
        'min_clients_per_round': 3,
        'max_clients_per_round': 50,
        'round_timeout': 3600,
        'round_interval': 300,
        'privacy_budget': 1.0,
        'privacy_epsilon': 1.0,
        'aggregation_method': 'federated_averaging',
        'model_configs': {
            'price_prediction': {
                'input_size': 10,
                'hidden_size': 128,
                'num_layers': 2
            },
            'risk_assessment': {
                'input_size': 20,
                'hidden_size': 64
            }
        }
    }
    
    fl_system = EnterpriseFederatedLearningSystem(config)
    await fl_system.start()
    
    print("Enterprise Federated Learning System started")
    print(f"WebSocket server listening on ws://{config['websocket_host']}:{config['websocket_port']}")
    
    try:
        await asyncio.Future()  # Бесконечное ожидание
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        await fl_system.stop()

if __name__ == '__main__':
    asyncio.run(main())