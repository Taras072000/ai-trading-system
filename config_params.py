"""
Конфигурация системных требований для Peper Binance v4
Этот файл содержит минимальные и рекомендуемые требования к системе
"""

import os
from dataclasses import dataclass
from typing import Dict, Any, Optional

@dataclass
class SystemRequirements:
    """Класс для хранения системных требований"""
    
    # Минимальные требования
    min_ram_gb: float
    min_cpu_cores: int
    min_disk_space_gb: float
    min_python_version: tuple
    
    # Рекомендуемые требования
    recommended_ram_gb: float
    recommended_cpu_cores: int
    recommended_disk_space_gb: float
    
    # Требования для AI модулей
    ai_min_ram_gb: float
    ai_recommended_ram_gb: float
    
    # Поддерживаемые операционные системы
    supported_os: list
    
    # Требования к GPU (опционально)
    gpu_memory_gb: Optional[float] = None
    cuda_support: bool = False

# Конфигурация требований для Peper Binance v4
SYSTEM_REQUIREMENTS = SystemRequirements(
    # Минимальные требования для базовой работы
    min_ram_gb=4.0,
    min_cpu_cores=2,
    min_disk_space_gb=2.0,
    min_python_version=(3, 8),
    
    # Рекомендуемые требования для оптимальной работы
    recommended_ram_gb=8.0,
    recommended_cpu_cores=4,
    recommended_disk_space_gb=5.0,
    
    # Требования для AI модулей (включая Mistral 7B)
    ai_min_ram_gb=8.0,          # Минимум для работы AI
    ai_recommended_ram_gb=16.0,  # Рекомендуемо для AI
    
    # Поддерживаемые ОС
    supported_os=['Darwin', 'Linux', 'Windows'],
    
    # GPU требования (опционально для ускорения AI)
    gpu_memory_gb=4.0,
    cuda_support=True
)

# Дополнительные параметры конфигурации
CONFIG_PARAMS = {
    # Настройки производительности
    'performance': {
        'max_workers': int(os.getenv('MAX_WORKERS', '2')),  # Ограничиваем количество воркеров
        'memory_limit_mb': int(os.getenv('MEMORY_LIMIT_MB', '512')),  # Лимит памяти в МБ
        'cpu_usage_limit': float(os.getenv('CPU_USAGE_LIMIT', '0.8')),  # Лимит CPU (80%)
        'max_concurrent_trades': 10,
        'api_request_timeout': 30,
        'data_update_interval': 1,  # секунды
    },
    
    # Настройки мониторинга ресурсов
    'monitoring': {
        'check_interval': 30,  # Проверка каждые 30 секунд
        'memory_threshold': 0.85,  # Предупреждение при 85% использования памяти
        'cpu_threshold': 0.90,  # Предупреждение при 90% использования CPU
        'auto_cleanup': True  # Автоматическая очистка памяти
    },
    
    # Настройки кэширования
    'cache': {
        'max_size': 100,  # Максимальный размер кэша
        'ttl': 3600,  # Время жизни кэша в секундах
        'cleanup_interval': 300  # Очистка кэша каждые 5 минут
    },
    
    # Настройки AI модулей
    'ai_modules': {
        'trading_ai': {
            'enabled': True,
            'priority': 'high',
            'memory_limit_mb': 512
        },
        'lava_ai': {
            'enabled': True,
            'priority': 'medium',
            'memory_limit_mb': 256
        },
        'mistral_ai': {
            'enabled': True,
            'priority': 'low',
            'memory_limit_mb': 4096,  # Требует больше памяти
            'fallback_mode': True,    # Может работать в режиме заглушки
            'min_ram_gb': 8.0
        },
        'lgbm_ai': {
            'enabled': True,
            'priority': 'high',
            'memory_limit_mb': 1024
        }
    },
    
    # Настройки безопасности
    'security': {
        'max_position_size': 0.1,  # 10% от депозита
        'stop_loss_percent': 2.0,
        'daily_loss_limit': 5.0
    },
    
    # Настройки логирования
    'logging': {
        'level': 'INFO',
        'max_file_size_mb': 100,
        'backup_count': 5
    }
}

# Сообщения для пользователя
MESSAGES = {
    'system_check_passed': "✅ Система соответствует всем требованиям. Все AI модули будут активны.",
    'system_check_warning': "⚠️  Система соответствует минимальным требованиям, но производительность может быть снижена.",
    'mistral_disabled': "🔄 Mistral AI отключен из-за недостатка ресурсов. Винрейт может быть ниже заявленного.",
    'system_check_failed': "❌ Система не соответствует минимальным требованиям. Запуск невозможен.",
    'performance_warning': "⚠️  Внимание: При текущих параметрах системы винрейт может быть ниже заявленного."
}

def get_requirement_description() -> str:
    """Возвращает описание системных требований"""
    req = SYSTEM_REQUIREMENTS
    
    description = f"""
=== Системные требования Peper Binance v4 ===

МИНИМАЛЬНЫЕ ТРЕБОВАНИЯ:
• ОЗУ: {req.min_ram_gb} ГБ
• Процессор: {req.min_cpu_cores} ядра
• Свободное место: {req.min_disk_space_gb} ГБ
• Python: {req.min_python_version[0]}.{req.min_python_version[1]}+

РЕКОМЕНДУЕМЫЕ ТРЕБОВАНИЯ:
• ОЗУ: {req.recommended_ram_gb} ГБ
• Процессор: {req.recommended_cpu_cores} ядра
• Свободное место: {req.recommended_disk_space_gb} ГБ

ДЛЯ AI МОДУЛЕЙ:
• Минимум ОЗУ: {req.ai_min_ram_gb} ГБ
• Рекомендуемо ОЗУ: {req.ai_recommended_ram_gb} ГБ

ОПЦИОНАЛЬНО (для ускорения):
• GPU память: {req.gpu_memory_gb} ГБ
• CUDA поддержка: {'Да' if req.cuda_support else 'Нет'}

Поддерживаемые ОС: {', '.join(req.supported_os)}
"""
    return description