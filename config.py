"""
Конфигурация для Peper Binance v4
Базовые системные настройки
"""

import os
from pathlib import Path
from typing import Dict, Any
import yaml
from dotenv import load_dotenv

# Загружаем переменные окружения
load_dotenv()

class Config:
    """Класс базовой конфигурации системы"""
    
    # Базовые настройки
    BASE_DIR = Path(__file__).parent
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    
    # Пути к директориям
    DATA_DIR = BASE_DIR / 'data'
    LOGS_DIR = BASE_DIR / 'logs'
    MODELS_DIR = BASE_DIR / 'models'
    
    # Настройки логирования
    LOGGING_CONFIG = {
        'level': LOG_LEVEL,
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        'file_path': LOGS_DIR / 'app.log',
        'max_file_size': 10 * 1024 * 1024,  # 10MB
        'backup_count': 5
    }
    
    # API настройки
    API_CONFIG = {
        'timeout': 30,
        'max_retries': 3,
        'retry_delay': 1,
        'connection_pool_size': 10,
        'base_url': os.getenv('API_BASE_URL', 'https://api.binance.com'),
        'api_key': os.getenv('BINANCE_API_KEY'),
        'api_secret': os.getenv('BINANCE_API_SECRET')
    }
    
    # Настройки базы данных
    DATABASE_CONFIG = {
        'type': os.getenv('DB_TYPE', 'sqlite'),
        'path': BASE_DIR / 'data' / 'trading.db',
        'backup_interval': 3600,  # Резервное копирование каждый час
        'max_connections': 5
    }
    
    @classmethod
    def load_from_file(cls, config_path: str = None) -> Dict[str, Any]:
        """Загрузка конфигурации из YAML файла"""
        if config_path is None:
            config_path = cls.BASE_DIR / 'config.yaml'
        
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        return {}
    
    @classmethod
    def get_api_config(cls) -> Dict[str, Any]:
        """Получение API конфигурации"""
        return cls.API_CONFIG
    
    @classmethod
    def get_database_config(cls) -> Dict[str, Any]:
        """Получение конфигурации базы данных"""
        return cls.DATABASE_CONFIG

# Глобальный экземпляр конфигурации
config = Config()