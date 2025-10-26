#!/usr/bin/env python3
"""
Configuration Manager for Reinforcement Learning System
Менеджер конфигурации для системы обучения с подкреплением
"""

import json
import logging
import os
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from datetime import datetime

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ReinforcementConfig:
    """Конфигурация обучения с подкреплением"""
    enabled: bool = True
    learning_rate: float = 0.01
    reward_multiplier: float = 1.5
    punishment_multiplier: float = 0.8
    weight_decay: float = 0.001
    min_weight: float = 0.05
    max_weight: float = 0.8
    confidence_threshold: float = 0.6
    session_auto_save: bool = True
    session_save_interval: int = 10
    max_sessions_history: int = 50

@dataclass
class AIModulesConfig:
    """Конфигурация AI модулей"""
    initial_weights: Dict[str, float] = None
    weight_constraints: Dict[str, Dict[str, float]] = None
    
    def __post_init__(self):
        if self.initial_weights is None:
            self.initial_weights = {
                "trading_ai": 0.25,
                "lava_ai": 0.35,
                "lgbm_ai": 0.40,
                "mistral_ai": 0.0
            }
        
        if self.weight_constraints is None:
            self.weight_constraints = {
                "trading_ai": {"min": 0.1, "max": 0.6},
                "lava_ai": {"min": 0.1, "max": 0.6},
                "lgbm_ai": {"min": 0.1, "max": 0.7},
                "mistral_ai": {"min": 0.0, "max": 0.5}
            }

@dataclass
class MistralServerConfig:
    """Конфигурация Mistral сервера"""
    auto_start: bool = True
    auto_stop: bool = True
    model_name: str = "mistral:latest"
    host: str = "localhost"
    port: int = 11434
    timeout: int = 30
    max_retries: int = 3
    health_check_interval: int = 60
    startup_timeout: int = 120
    max_startup_attempts: int = 3
    server_command: str = "ollama serve"
    model_pull_command: str = "ollama pull mistral:latest"

@dataclass
class TestingConfig:
    """Конфигурация тестирования"""
    default_symbols: List[str] = None
    default_timeframe: str = "1h"
    default_test_days: int = 30
    min_trades_for_learning: int = 5
    confidence_boost_threshold: float = 0.7
    confidence_penalty_threshold: float = 0.4
    
    def __post_init__(self):
        if self.default_symbols is None:
            self.default_symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "SOLUSDT"]

@dataclass
class AnalyticsConfig:
    """Конфигурация аналитики"""
    auto_generate_reports: bool = True
    save_detailed_trades: bool = True
    plot_formats: List[str] = None
    report_frequency: str = "after_each_session"
    comparison_sessions_limit: int = 5
    
    def __post_init__(self):
        if self.plot_formats is None:
            self.plot_formats = ["html", "png"]

@dataclass
class PersistenceConfig:
    """Конфигурация персистентности"""
    database_path: str = "data/reinforcement_learning.db"
    backup_frequency: str = "daily"
    max_backups: int = 30
    compress_old_data: bool = True

@dataclass
class LoggingConfig:
    """Конфигурация логирования"""
    level: str = "INFO"
    file_path: str = "logs/reinforcement_learning.log"
    max_file_size: str = "10MB"
    backup_count: int = 5
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

class ConfigManager:
    """
    Менеджер конфигурации для системы обучения с подкреплением
    """
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = config_dir
        self.main_config_file = os.path.join(config_dir, "reinforcement_learning_config.json")
        self.profiles_config_file = os.path.join(config_dir, "reinforcement_learning_profiles.json")
        
        # Конфигурации
        self.reinforcement_config = ReinforcementConfig()
        self.ai_modules_config = AIModulesConfig()
        self.mistral_server_config = MistralServerConfig()
        self.testing_config = TestingConfig()
        self.analytics_config = AnalyticsConfig()
        self.persistence_config = PersistenceConfig()
        self.logging_config = LoggingConfig()
        
        # Профили и специальные конфигурации
        self.profiles: Dict[str, Dict] = {}
        self.symbol_specific_configs: Dict[str, Dict] = {}
        self.time_based_configs: Dict[str, Dict] = {}
        
        # Текущий профиль
        self.current_profile: Optional[str] = None
        
        # Загружаем конфигурации
        self.load_configurations()
    
    def load_configurations(self) -> bool:
        """Загрузка всех конфигураций"""
        try:
            # Загружаем основную конфигурацию
            if os.path.exists(self.main_config_file):
                self._load_main_config()
            else:
                logger.warning(f"Основной файл конфигурации не найден: {self.main_config_file}")
                self._create_default_main_config()
            
            # Загружаем профили
            if os.path.exists(self.profiles_config_file):
                self._load_profiles_config()
            else:
                logger.warning(f"Файл профилей не найден: {self.profiles_config_file}")
                self._create_default_profiles_config()
            
            logger.info("✅ Конфигурации успешно загружены")
            return True
            
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки конфигураций: {e}")
            return False
    
    def _load_main_config(self):
        """Загрузка основной конфигурации"""
        try:
            with open(self.main_config_file, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            # Загружаем каждую секцию
            if 'reinforcement_learning' in config_data:
                self.reinforcement_config = ReinforcementConfig(**config_data['reinforcement_learning'])
            
            if 'ai_modules' in config_data:
                self.ai_modules_config = AIModulesConfig(**config_data['ai_modules'])
            
            if 'mistral_server' in config_data:
                self.mistral_server_config = MistralServerConfig(**config_data['mistral_server'])
            
            if 'testing' in config_data:
                self.testing_config = TestingConfig(**config_data['testing'])
            
            if 'analytics' in config_data:
                self.analytics_config = AnalyticsConfig(**config_data['analytics'])
            
            if 'persistence' in config_data:
                self.persistence_config = PersistenceConfig(**config_data['persistence'])
            
            if 'logging' in config_data:
                self.logging_config = LoggingConfig(**config_data['logging'])
            
            logger.info("✅ Основная конфигурация загружена")
            
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки основной конфигурации: {e}")
    
    def _load_profiles_config(self):
        """Загрузка профилей и специальных конфигураций"""
        try:
            with open(self.profiles_config_file, 'r', encoding='utf-8') as f:
                profiles_data = json.load(f)
            
            self.profiles = profiles_data.get('profiles', {})
            self.symbol_specific_configs = profiles_data.get('symbol_specific_configs', {})
            self.time_based_configs = profiles_data.get('time_based_configs', {})
            
            logger.info(f"✅ Загружено профилей: {len(self.profiles)}")
            logger.info(f"✅ Загружено конфигураций по символам: {len(self.symbol_specific_configs)}")
            logger.info(f"✅ Загружено временных конфигураций: {len(self.time_based_configs)}")
            
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки профилей: {e}")
    
    def _create_default_main_config(self):
        """Создание конфигурации по умолчанию"""
        try:
            os.makedirs(self.config_dir, exist_ok=True)
            
            default_config = {
                "reinforcement_learning": asdict(self.reinforcement_config),
                "ai_modules": asdict(self.ai_modules_config),
                "mistral_server": asdict(self.mistral_server_config),
                "testing": asdict(self.testing_config),
                "analytics": asdict(self.analytics_config),
                "persistence": asdict(self.persistence_config),
                "logging": asdict(self.logging_config)
            }
            
            with open(self.main_config_file, 'w', encoding='utf-8') as f:
                json.dump(default_config, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ Создана конфигурация по умолчанию: {self.main_config_file}")
            
        except Exception as e:
            logger.error(f"❌ Ошибка создания конфигурации по умолчанию: {e}")
    
    def _create_default_profiles_config(self):
        """Создание профилей по умолчанию"""
        try:
            default_profiles = {
                "profiles": {
                    "balanced": {
                        "description": "Сбалансированный профиль обучения (по умолчанию)",
                        "reinforcement_learning": asdict(self.reinforcement_config),
                        "ai_modules": asdict(self.ai_modules_config),
                        "testing": asdict(self.testing_config)
                    }
                },
                "symbol_specific_configs": {},
                "time_based_configs": {}
            }
            
            with open(self.profiles_config_file, 'w', encoding='utf-8') as f:
                json.dump(default_profiles, f, indent=2, ensure_ascii=False)
            
            self.profiles = default_profiles["profiles"]
            
            logger.info(f"✅ Созданы профили по умолчанию: {self.profiles_config_file}")
            
        except Exception as e:
            logger.error(f"❌ Ошибка создания профилей по умолчанию: {e}")
    
    def apply_profile(self, profile_name: str) -> bool:
        """Применение профиля"""
        try:
            if profile_name not in self.profiles:
                logger.error(f"❌ Профиль '{profile_name}' не найден")
                return False
            
            profile = self.profiles[profile_name]
            
            # Применяем настройки профиля
            if 'reinforcement_learning' in profile:
                for key, value in profile['reinforcement_learning'].items():
                    if hasattr(self.reinforcement_config, key):
                        setattr(self.reinforcement_config, key, value)
            
            if 'ai_modules' in profile:
                for key, value in profile['ai_modules'].items():
                    if hasattr(self.ai_modules_config, key):
                        setattr(self.ai_modules_config, key, value)
            
            if 'testing' in profile:
                for key, value in profile['testing'].items():
                    if hasattr(self.testing_config, key):
                        setattr(self.testing_config, key, value)
            
            if 'mistral_server' in profile:
                for key, value in profile['mistral_server'].items():
                    if hasattr(self.mistral_server_config, key):
                        setattr(self.mistral_server_config, key, value)
            
            self.current_profile = profile_name
            logger.info(f"✅ Применен профиль: {profile_name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Ошибка применения профиля {profile_name}: {e}")
            return False
    
    def apply_symbol_config(self, symbol: str) -> bool:
        """Применение конфигурации для конкретного символа"""
        try:
            if symbol not in self.symbol_specific_configs:
                logger.info(f"Специальная конфигурация для {symbol} не найдена, используется общая")
                return True
            
            symbol_config = self.symbol_specific_configs[symbol]
            
            # Применяем настройки для символа
            if 'reinforcement_learning' in symbol_config:
                for key, value in symbol_config['reinforcement_learning'].items():
                    if hasattr(self.reinforcement_config, key):
                        setattr(self.reinforcement_config, key, value)
            
            if 'ai_modules' in symbol_config:
                for key, value in symbol_config['ai_modules'].items():
                    if hasattr(self.ai_modules_config, key):
                        setattr(self.ai_modules_config, key, value)
            
            if 'testing' in symbol_config:
                for key, value in symbol_config['testing'].items():
                    if hasattr(self.testing_config, key):
                        setattr(self.testing_config, key, value)
            
            logger.info(f"✅ Применена конфигурация для символа: {symbol}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Ошибка применения конфигурации для символа {symbol}: {e}")
            return False
    
    def get_config_for_symbol(self, symbol: str) -> Dict[str, Any]:
        """Получение полной конфигурации для символа"""
        try:
            # Базовая конфигурация
            config = {
                "reinforcement_learning": asdict(self.reinforcement_config),
                "ai_modules": asdict(self.ai_modules_config),
                "mistral_server": asdict(self.mistral_server_config),
                "testing": asdict(self.testing_config),
                "analytics": asdict(self.analytics_config),
                "persistence": asdict(self.persistence_config),
                "logging": asdict(self.logging_config)
            }
            
            # Применяем специфичные для символа настройки
            if symbol in self.symbol_specific_configs:
                symbol_config = self.symbol_specific_configs[symbol]
                
                for section, settings in symbol_config.items():
                    if section in config and isinstance(settings, dict):
                        config[section].update(settings)
            
            return config
            
        except Exception as e:
            logger.error(f"❌ Ошибка получения конфигурации для символа {symbol}: {e}")
            return {}
    
    def save_current_config(self) -> bool:
        """Сохранение текущей конфигурации"""
        try:
            current_config = {
                "reinforcement_learning": asdict(self.reinforcement_config),
                "ai_modules": asdict(self.ai_modules_config),
                "mistral_server": asdict(self.mistral_server_config),
                "testing": asdict(self.testing_config),
                "analytics": asdict(self.analytics_config),
                "persistence": asdict(self.persistence_config),
                "logging": asdict(self.logging_config)
            }
            
            # Создаем резервную копию
            backup_file = f"{self.main_config_file}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            if os.path.exists(self.main_config_file):
                os.rename(self.main_config_file, backup_file)
            
            # Сохраняем новую конфигурацию
            with open(self.main_config_file, 'w', encoding='utf-8') as f:
                json.dump(current_config, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ Конфигурация сохранена: {self.main_config_file}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Ошибка сохранения конфигурации: {e}")
            return False
    
    def list_available_profiles(self) -> List[str]:
        """Получение списка доступных профилей"""
        return list(self.profiles.keys())
    
    def get_profile_description(self, profile_name: str) -> str:
        """Получение описания профиля"""
        if profile_name in self.profiles:
            return self.profiles[profile_name].get('description', 'Описание отсутствует')
        return "Профиль не найден"
    
    def create_custom_profile(self, profile_name: str, description: str, 
                            custom_settings: Dict[str, Any]) -> bool:
        """Создание пользовательского профиля"""
        try:
            # Базовые настройки
            base_profile = {
                "description": description,
                "reinforcement_learning": asdict(self.reinforcement_config),
                "ai_modules": asdict(self.ai_modules_config),
                "testing": asdict(self.testing_config)
            }
            
            # Применяем пользовательские настройки
            for section, settings in custom_settings.items():
                if section in base_profile and isinstance(settings, dict):
                    base_profile[section].update(settings)
                else:
                    base_profile[section] = settings
            
            self.profiles[profile_name] = base_profile
            
            # Сохраняем обновленные профили
            profiles_data = {
                "profiles": self.profiles,
                "symbol_specific_configs": self.symbol_specific_configs,
                "time_based_configs": self.time_based_configs
            }
            
            with open(self.profiles_config_file, 'w', encoding='utf-8') as f:
                json.dump(profiles_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ Создан пользовательский профиль: {profile_name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Ошибка создания профиля {profile_name}: {e}")
            return False
    
    def validate_config(self) -> List[str]:
        """Валидация конфигурации"""
        errors = []
        
        try:
            # Проверяем learning_rate
            if not 0.001 <= self.reinforcement_config.learning_rate <= 0.1:
                errors.append("learning_rate должен быть между 0.001 и 0.1")
            
            # Проверяем веса
            total_weight = sum(self.ai_modules_config.initial_weights.values())
            if abs(total_weight - 1.0) > 0.01:
                errors.append(f"Сумма весов AI модулей должна быть 1.0, текущая: {total_weight}")
            
            # Проверяем ограничения весов
            for ai_name, weights in self.ai_modules_config.initial_weights.items():
                if ai_name in self.ai_modules_config.weight_constraints:
                    constraints = self.ai_modules_config.weight_constraints[ai_name]
                    if weights < constraints['min'] or weights > constraints['max']:
                        errors.append(f"Вес {ai_name} ({weights}) выходит за ограничения {constraints}")
            
            # Проверяем пороги уверенности
            if not 0.1 <= self.reinforcement_config.confidence_threshold <= 0.9:
                errors.append("confidence_threshold должен быть между 0.1 и 0.9")
            
            # Проверяем пути
            if not os.path.exists(os.path.dirname(self.persistence_config.database_path)):
                errors.append(f"Директория для базы данных не существует: {os.path.dirname(self.persistence_config.database_path)}")
            
            if errors:
                logger.warning(f"⚠️ Найдены ошибки в конфигурации: {len(errors)}")
                for error in errors:
                    logger.warning(f"  - {error}")
            else:
                logger.info("✅ Конфигурация валидна")
            
        except Exception as e:
            errors.append(f"Ошибка валидации: {e}")
            logger.error(f"❌ Ошибка валидации конфигурации: {e}")
        
        return errors

# Пример использования
def main():
    """Пример использования менеджера конфигурации"""
    config_manager = ConfigManager()
    
    # Проверяем доступные профили
    profiles = config_manager.list_available_profiles()
    print(f"Доступные профили: {profiles}")
    
    # Применяем профиль
    if profiles:
        profile_name = profiles[0]
        if config_manager.apply_profile(profile_name):
            print(f"✅ Применен профиль: {profile_name}")
    
    # Валидируем конфигурацию
    errors = config_manager.validate_config()
    if not errors:
        print("✅ Конфигурация валидна")
    else:
        print(f"❌ Ошибки конфигурации: {errors}")

if __name__ == "__main__":
    main()