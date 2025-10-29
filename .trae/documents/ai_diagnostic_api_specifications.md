# API Спецификации системы диагностики AI моделей

## 1. Обзор API

Система диагностики AI моделей предоставляет RESTful API для программного доступа к функциям диагностики, мониторинга и анализа производительности торговых AI моделей.

### 1.1 Базовый URL
```
http://localhost:8080/api/v1/diagnostic
```

### 1.2 Аутентификация
```http
Authorization: Bearer <api_token>
Content-Type: application/json
```

## 2. Основные эндпоинты

### 2.1 Управление диагностическими сессиями

#### Создание новой диагностической сессии
```http
POST /api/v1/diagnostic/sessions
```

**Тело запроса:**
```json
{
  "session_name": "daily_diagnostic_20241028",
  "test_type": "full",
  "models": ["lava", "mistral", "trading", "lgbm"],
  "config": {
    "test_duration_minutes": 30,
    "test_pairs": ["BTCUSDT", "ETHUSDT"],
    "parallel_tests": true,
    "max_parallel_models": 2
  },
  "scheduled_start": "2024-10-28T10:00:00Z"
}
```

**Ответ:**
```json
{
  "session_id": "diag_20241028_001",
  "status": "created",
  "estimated_duration": "30 minutes",
  "created_at": "2024-10-28T09:45:00Z",
  "scheduled_start": "2024-10-28T10:00:00Z"
}
```

#### Получение статуса сессии
```http
GET /api/v1/diagnostic/sessions/{session_id}
```

**Ответ:**
```json
{
  "session_id": "diag_20241028_001",
  "status": "running",
  "progress": 65,
  "current_test": "isolated_mistral",
  "started_at": "2024-10-28T10:00:00Z",
  "estimated_completion": "2024-10-28T10:25:00Z",
  "tests_completed": 3,
  "tests_total": 7
}
```

#### Остановка диагностической сессии
```http
DELETE /api/v1/diagnostic/sessions/{session_id}
```

**Ответ:**
```json
{
  "session_id": "diag_20241028_001",
  "status": "stopped",
  "stopped_at": "2024-10-28T10:15:00Z",
  "reason": "user_request"
}
```

### 2.2 Изолированное тестирование моделей

#### Запуск изолированного теста
```http
POST /api/v1/diagnostic/isolated-test
```

**Тело запроса:**
```json
{
  "model_name": "lava",
  "test_config": {
    "duration_minutes": 15,
    "test_pairs": ["BTCUSDT"],
    "metrics": ["accuracy", "response_time", "memory_usage", "cpu_usage"],
    "sampling_rate": 10
  }
}
```

**Ответ:**
```json
{
  "test_id": "isolated_lava_20241028_001",
  "model_name": "lava",
  "status": "running",
  "started_at": "2024-10-28T10:00:00Z",
  "estimated_completion": "2024-10-28T10:15:00Z"
}
```

#### Получение результатов изолированного теста
```http
GET /api/v1/diagnostic/isolated-test/{test_id}/results
```

**Ответ:**
```json
{
  "test_id": "isolated_lava_20241028_001",
  "model_name": "lava",
  "status": "completed",
  "duration": "15 minutes",
  "results": {
    "accuracy": {
      "value": 0.67,
      "status": "good",
      "threshold": 0.6
    },
    "response_time": {
      "avg_ms": 450,
      "max_ms": 890,
      "min_ms": 120,
      "status": "excellent"
    },
    "memory_usage": {
      "avg_mb": 180,
      "max_mb": 220,
      "status": "optimal"
    },
    "cpu_usage": {
      "avg_percent": 35,
      "max_percent": 65,
      "status": "good"
    },
    "stability_score": 0.85,
    "error_rate": 0.02
  },
  "recommendations": [
    "Model performance is within acceptable ranges",
    "Consider optimizing memory allocation for peak loads"
  ]
}
```

### 2.3 Комбинированное тестирование

#### Запуск комбинированного теста
```http
POST /api/v1/diagnostic/combined-test
```

**Тело запроса:**
```json
{
  "models": ["lava", "mistral"],
  "test_config": {
    "duration_minutes": 20,
    "test_pairs": ["BTCUSDT", "ETHUSDT"],
    "interaction_analysis": true,
    "conflict_detection": true,
    "load_testing": false
  }
}
```

**Ответ:**
```json
{
  "test_id": "combined_lava_mistral_20241028_001",
  "models": ["lava", "mistral"],
  "status": "running",
  "started_at": "2024-10-28T10:30:00Z",
  "estimated_completion": "2024-10-28T10:50:00Z"
}
```

#### Получение результатов комбинированного теста
```http
GET /api/v1/diagnostic/combined-test/{test_id}/results
```

**Ответ:**
```json
{
  "test_id": "combined_lava_mistral_20241028_001",
  "models": ["lava", "mistral"],
  "status": "completed",
  "results": {
    "synergy_score": 0.78,
    "conflict_rate": 0.12,
    "combined_accuracy": 0.72,
    "individual_performance": {
      "lava": {
        "accuracy": 0.68,
        "response_time_ms": 420
      },
      "mistral": {
        "accuracy": 0.71,
        "response_time_ms": 380
      }
    },
    "interaction_analysis": {
      "agreement_rate": 0.88,
      "complementary_decisions": 0.65,
      "conflicting_decisions": 0.12
    }
  },
  "recommendations": [
    "Models show good synergy",
    "Consider adjusting weights to reduce conflicts"
  ]
}
```

### 2.4 Мониторинг ресурсов

#### Получение текущих метрик ресурсов
```http
GET /api/v1/diagnostic/resources/current
```

**Ответ:**
```json
{
  "timestamp": "2024-10-28T10:45:00Z",
  "system_resources": {
    "cpu_usage_percent": 45,
    "memory_usage_mb": 2048,
    "memory_total_mb": 8192,
    "disk_usage_percent": 65,
    "gpu_usage_percent": 30
  },
  "model_resources": {
    "lava": {
      "memory_mb": 180,
      "cpu_percent": 12,
      "gpu_percent": 8
    },
    "mistral": {
      "memory_mb": 220,
      "cpu_percent": 15,
      "gpu_percent": 12
    }
  }
}
```

#### Получение исторических данных ресурсов
```http
GET /api/v1/diagnostic/resources/history?from=2024-10-28T09:00:00Z&to=2024-10-28T11:00:00Z&interval=5m
```

**Ответ:**
```json
{
  "from": "2024-10-28T09:00:00Z",
  "to": "2024-10-28T11:00:00Z",
  "interval": "5m",
  "data": [
    {
      "timestamp": "2024-10-28T09:00:00Z",
      "cpu_usage": 35,
      "memory_usage": 1800,
      "models": {
        "lava": {"memory": 160, "cpu": 10},
        "mistral": {"memory": 200, "cpu": 12}
      }
    }
  ]
}
```

### 2.5 Отчеты и аналитика

#### Генерация отчета
```http
POST /api/v1/diagnostic/reports
```

**Тело запроса:**
```json
{
  "session_id": "diag_20241028_001",
  "report_type": "comprehensive",
  "format": "html",
  "include_charts": true,
  "include_recommendations": true,
  "email_recipients": ["admin@trading.com"]
}
```

**Ответ:**
```json
{
  "report_id": "report_20241028_001",
  "status": "generating",
  "estimated_completion": "2024-10-28T11:05:00Z",
  "download_url": "/api/v1/diagnostic/reports/report_20241028_001/download"
}
```

#### Получение списка отчетов
```http
GET /api/v1/diagnostic/reports?limit=10&offset=0
```

**Ответ:**
```json
{
  "reports": [
    {
      "report_id": "report_20241028_001",
      "session_id": "diag_20241028_001",
      "created_at": "2024-10-28T11:00:00Z",
      "format": "html",
      "size_mb": 2.5,
      "status": "completed"
    }
  ],
  "total": 25,
  "limit": 10,
  "offset": 0
}
```

#### Скачивание отчета
```http
GET /api/v1/diagnostic/reports/{report_id}/download
```

**Ответ:** Файл отчета (HTML/PDF/CSV)

### 2.6 Рекомендации и аналитика

#### Получение рекомендаций по системе
```http
GET /api/v1/diagnostic/recommendations?session_id=diag_20241028_001
```

**Ответ:**
```json
{
  "session_id": "diag_20241028_001",
  "generated_at": "2024-10-28T11:00:00Z",
  "overall_health": "good",
  "critical_issues": 0,
  "warnings": 2,
  "recommendations": [
    {
      "priority": "high",
      "category": "performance",
      "model": "mistral",
      "issue": "High memory usage during peak loads",
      "recommendation": "Optimize model parameters or increase memory allocation",
      "estimated_impact": "15% performance improvement"
    },
    {
      "priority": "medium",
      "category": "interaction",
      "models": ["lava", "trading"],
      "issue": "Moderate decision conflicts",
      "recommendation": "Adjust model weights in ensemble",
      "estimated_impact": "8% accuracy improvement"
    }
  ],
  "action_plan": [
    {
      "step": 1,
      "action": "Optimize Mistral AI memory usage",
      "estimated_time": "2 hours",
      "risk": "low"
    },
    {
      "step": 2,
      "action": "Retrain ensemble weights",
      "estimated_time": "4 hours",
      "risk": "medium"
    }
  ]
}
```

#### Получение трендов производительности
```http
GET /api/v1/diagnostic/trends?period=7d&models=lava,mistral
```

**Ответ:**
```json
{
  "period": "7d",
  "models": ["lava", "mistral"],
  "trends": {
    "accuracy": {
      "lava": {
        "current": 0.67,
        "trend": "stable",
        "change_percent": 1.2
      },
      "mistral": {
        "current": 0.71,
        "trend": "improving",
        "change_percent": 3.5
      }
    },
    "response_time": {
      "lava": {
        "current_ms": 450,
        "trend": "degrading",
        "change_percent": -8.2
      },
      "mistral": {
        "current_ms": 380,
        "trend": "stable",
        "change_percent": 0.5
      }
    }
  }
}
```

## 3. WebSocket API для реального времени

### 3.1 Подключение к WebSocket
```javascript
const ws = new WebSocket('ws://localhost:8080/ws/diagnostic');

ws.onopen = function() {
    // Подписка на обновления сессии
    ws.send(JSON.stringify({
        type: 'subscribe',
        channel: 'session_updates',
        session_id: 'diag_20241028_001'
    }));
};
```

### 3.2 Типы сообщений WebSocket

#### Обновления статуса сессии
```json
{
  "type": "session_update",
  "session_id": "diag_20241028_001",
  "status": "running",
  "progress": 75,
  "current_test": "combined_lava_mistral",
  "timestamp": "2024-10-28T10:45:00Z"
}
```

#### Метрики в реальном времени
```json
{
  "type": "metrics_update",
  "timestamp": "2024-10-28T10:45:00Z",
  "model": "lava",
  "metrics": {
    "response_time_ms": 420,
    "memory_usage_mb": 185,
    "cpu_usage_percent": 12,
    "accuracy": 0.68
  }
}
```

#### Уведомления об ошибках
```json
{
  "type": "error_notification",
  "severity": "warning",
  "model": "mistral",
  "message": "High memory usage detected",
  "timestamp": "2024-10-28T10:45:00Z",
  "details": {
    "current_usage_mb": 480,
    "threshold_mb": 400
  }
}
```

## 4. Коды ошибок

### 4.1 HTTP коды состояния

| Код | Описание | Пример использования |
|-----|----------|---------------------|
| 200 | OK | Успешный запрос |
| 201 | Created | Сессия создана |
| 400 | Bad Request | Неверные параметры |
| 401 | Unauthorized | Неверный токен |
| 404 | Not Found | Сессия не найдена |
| 409 | Conflict | Сессия уже запущена |
| 429 | Too Many Requests | Превышен лимит запросов |
| 500 | Internal Server Error | Внутренняя ошибка |

### 4.2 Специфичные коды ошибок

```json
{
  "error": {
    "code": "MODEL_NOT_AVAILABLE",
    "message": "Requested AI model is not available",
    "details": {
      "model": "lava",
      "reason": "Model is currently being updated"
    }
  }
}
```

**Коды ошибок:**
- `MODEL_NOT_AVAILABLE` - Модель недоступна
- `INSUFFICIENT_RESOURCES` - Недостаточно ресурсов
- `INVALID_CONFIG` - Неверная конфигурация
- `SESSION_LIMIT_EXCEEDED` - Превышен лимит сессий
- `TEST_TIMEOUT` - Тайм-аут теста
- `DATA_CORRUPTION` - Повреждение данных

## 5. Ограничения и квоты

### 5.1 Лимиты API

| Ресурс | Лимит | Период |
|--------|-------|--------|
| Запросы | 1000 | 1 час |
| Сессии | 5 | Одновременно |
| Отчеты | 50 | 1 день |
| WebSocket соединения | 10 | Одновременно |

### 5.2 Лимиты ресурсов

| Ресурс | Лимит |
|--------|-------|
| Максимальная длительность теста | 120 минут |
| Максимальное количество моделей в тесте | 5 |
| Максимальный размер отчета | 100 MB |
| Максимальное время хранения результатов | 30 дней |

## 6. Примеры интеграции

### 6.1 Python клиент

```python
import requests
import json
from typing import Dict, List

class DiagnosticAPIClient:
    def __init__(self, base_url: str, api_token: str):
        self.base_url = base_url
        self.headers = {
            'Authorization': f'Bearer {api_token}',
            'Content-Type': 'application/json'
        }
    
    def create_session(self, session_name: str, models: List[str], 
                      config: Dict) -> Dict:
        """Создание новой диагностической сессии"""
        payload = {
            'session_name': session_name,
            'test_type': 'full',
            'models': models,
            'config': config
        }
        
        response = requests.post(
            f'{self.base_url}/sessions',
            headers=self.headers,
            json=payload
        )
        
        return response.json()
    
    def get_session_status(self, session_id: str) -> Dict:
        """Получение статуса сессии"""
        response = requests.get(
            f'{self.base_url}/sessions/{session_id}',
            headers=self.headers
        )
        
        return response.json()
    
    def run_isolated_test(self, model_name: str, config: Dict) -> Dict:
        """Запуск изолированного теста"""
        payload = {
            'model_name': model_name,
            'test_config': config
        }
        
        response = requests.post(
            f'{self.base_url}/isolated-test',
            headers=self.headers,
            json=payload
        )
        
        return response.json()

# Пример использования
client = DiagnosticAPIClient(
    'http://localhost:8080/api/v1/diagnostic',
    'your_api_token_here'
)

# Создание сессии
session = client.create_session(
    'daily_check',
    ['lava', 'mistral'],
    {'test_duration_minutes': 30}
)

print(f"Сессия создана: {session['session_id']}")
```

### 6.2 JavaScript клиент

```javascript
class DiagnosticAPIClient {
    constructor(baseUrl, apiToken) {
        this.baseUrl = baseUrl;
        this.headers = {
            'Authorization': `Bearer ${apiToken}`,
            'Content-Type': 'application/json'
        };
    }
    
    async createSession(sessionName, models, config) {
        const response = await fetch(`${this.baseUrl}/sessions`, {
            method: 'POST',
            headers: this.headers,
            body: JSON.stringify({
                session_name: sessionName,
                test_type: 'full',
                models: models,
                config: config
            })
        });
        
        return await response.json();
    }
    
    async getSessionStatus(sessionId) {
        const response = await fetch(`${this.baseUrl}/sessions/${sessionId}`, {
            headers: this.headers
        });
        
        return await response.json();
    }
    
    // WebSocket подключение для реального времени
    connectWebSocket(sessionId) {
        const ws = new WebSocket('ws://localhost:8080/ws/diagnostic');
        
        ws.onopen = () => {
            ws.send(JSON.stringify({
                type: 'subscribe',
                channel: 'session_updates',
                session_id: sessionId
            }));
        };
        
        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            this.handleRealtimeUpdate(data);
        };
        
        return ws;
    }
    
    handleRealtimeUpdate(data) {
        switch(data.type) {
            case 'session_update':
                console.log(`Прогресс: ${data.progress}%`);
                break;
            case 'metrics_update':
                console.log(`Метрики ${data.model}:`, data.metrics);
                break;
            case 'error_notification':
                console.warn(`Предупреждение: ${data.message}`);
                break;
        }
    }
}

// Пример использования
const client = new DiagnosticAPIClient(
    'http://localhost:8080/api/v1/diagnostic',
    'your_api_token_here'
);

// Создание и мониторинг сессии
async function runDiagnostic() {
    const session = await client.createSession(
        'web_diagnostic',
        ['lava', 'mistral'],
        { test_duration_minutes: 15 }
    );
    
    console.log(`Сессия создана: ${session.session_id}`);
    
    // Подключение WebSocket для мониторинга
    const ws = client.connectWebSocket(session.session_id);
    
    // Периодическая проверка статуса
    const checkStatus = setInterval(async () => {
        const status = await client.getSessionStatus(session.session_id);
        
        if (status.status === 'completed') {
            console.log('Диагностика завершена!');
            clearInterval(checkStatus);
            ws.close();
        }
    }, 5000);
}

runDiagnostic();
```

## 7. Безопасность

### 7.1 Аутентификация
- Используйте HTTPS для всех запросов в продакшене
- API токены должны храниться в безопасном месте
- Токены имеют ограниченный срок действия (24 часа)

### 7.2 Авторизация
- Разные уровни доступа: read-only, operator, admin
- Ограничения по IP адресам
- Аудит всех действий

### 7.3 Защита данных
- Шифрование чувствительных данных
- Автоматическое удаление старых отчетов
- Маскирование API ключей в логах