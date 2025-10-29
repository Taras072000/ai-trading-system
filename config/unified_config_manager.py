"""
Унифицированный менеджер конфигурации для Peper Binance v4
Архитектурные улучшения - Фаза 2
"""

import yaml
import os
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from pathlib import Path
import threading
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

@dataclass
class AIModuleConfig:
    """Конфигурация AI модуля"""
    enabled: bool = True
    priority: str = "medium"
    memory_limit_mb: int = 512
    timeout: int = 30
    max_retries: int = 3
    weight: float = 0.25
    min_weight: float = 0.1
    max_weight: float = 0.5
    cache_enabled: bool = True
    parameters: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MarketPhaseConfig:
    """Конфигурация фазы рынка"""
    ai_weights: Dict[str, float] = field(default_factory=dict)
    risk_multiplier: float = 1.0
    parameters: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RiskManagementConfig:
    """Конфигурация риск-менеджмента"""
    enabled: bool = True
    base_position_size: float = 0.02
    max_position_size: float = 0.05
    stop_loss_atr_multiplier: float = 2.5
    take_profit_ratio: float = 2.0
    daily_loss_limit: float = 0.03
    max_open_positions: int = 5
    max_correlation: float = 0.7

@dataclass
class PerformanceConfig:
    """Конфигурация производительности"""
    async_enabled: bool = True
    max_concurrent_tasks: int = 10
    caching_enabled: bool = True
    vectorization: bool = True
    parallel_processing: bool = True

class UnifiedConfigManager:
    """Унифицированный менеджер конфигурации"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized'):
            return
            
        self._initialized = True
        self.config_path = Path(__file__).parent / "unified_config.yaml"
        self.config: Dict[str, Any] = {}
        self.ai_modules: Dict[str, AIModuleConfig] = {}
        self.market_phases: Dict[str, MarketPhaseConfig] = {}
        self.risk_management: RiskManagementConfig = RiskManagementConfig()
        self.performance: PerformanceConfig = PerformanceConfig()
        
        # Адаптивные веса
        self.adaptive_weights: Dict[str, float] = {}
        self.performance_history: Dict[str, List[float]] = {}
        self.last_weight_update = datetime.now()
        
        self.load_config()
    
    def load_config(self) -> None:
        """Загрузка конфигурации из YAML файла"""
        try:
            if not self.config_path.exists():
                logger.warning(f"Конфигурационный файл не найден: {self.config_path}")
                self._create_default_config()
                return
            
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
            
            self._parse_ai_modules()
            self._parse_market_phases()
            self._parse_risk_management()
            self._parse_performance()
            
            logger.info("Конфигурация успешно загружена")
            
        except Exception as e:
            logger.error(f"Ошибка загрузки конфигурации: {e}")
            self._create_default_config()
    
    def _parse_ai_modules(self) -> None:
        """Парсинг конфигурации AI модулей"""
        ai_config = self.config.get('ai_modules', {})
        global_config = ai_config.get('global', {})
        weights = ai_config.get('weights', {})
        constraints = ai_config.get('weight_constraints', {})
        
        # Инициализация AI модулей
        for module_name in ['trading_ai', 'lava_ai', 'lgbm_ai', 'mistral_ai']:
            module_config = ai_config.get(module_name, {})
            constraint = constraints.get(module_name, {})
            
            self.ai_modules[module_name] = AIModuleConfig(
                enabled=module_config.get('enabled', global_config.get('enabled', True)),
                priority=module_config.get('priority', 'medium'),
                memory_limit_mb=module_config.get('memory_limit_mb', 512),
                timeout=module_config.get('timeout', global_config.get('timeout', 30)),
                max_retries=module_config.get('max_retries', global_config.get('max_retries', 3)),
                weight=weights.get(module_name, 0.25),
                min_weight=constraint.get('min', 0.1),
                max_weight=constraint.get('max', 0.5),
                cache_enabled=module_config.get('cache', {}).get('enabled', True),
                parameters=module_config
            )
            
            # Инициализация адаптивных весов
            self.adaptive_weights[module_name] = weights.get(module_name, 0.25)
            self.performance_history[module_name] = []
    
    def _parse_market_phases(self) -> None:
        """Парсинг конфигурации фаз рынка"""
        phases_config = self.config.get('market_phases', {}).get('phase_parameters', {})
        
        for phase_name, phase_data in phases_config.items():
            self.market_phases[phase_name] = MarketPhaseConfig(
                ai_weights=phase_data.get('ai_weights', {}),
                risk_multiplier=phase_data.get('risk_multiplier', 1.0),
                parameters=phase_data
            )
    
    def _parse_risk_management(self) -> None:
        """Парсинг конфигурации риск-менеджмента"""
        risk_config = self.config.get('risk_management', {})
        
        self.risk_management = RiskManagementConfig(
            enabled=risk_config.get('dynamic_positioning', {}).get('enabled', True),
            base_position_size=risk_config.get('dynamic_positioning', {}).get('base_position_size', 0.02),
            max_position_size=risk_config.get('dynamic_positioning', {}).get('max_position_size', 0.05),
            stop_loss_atr_multiplier=risk_config.get('stop_loss', {}).get('atr_multiplier', 2.5),
            take_profit_ratio=risk_config.get('take_profit', {}).get('risk_reward_ratio', 2.0),
            daily_loss_limit=risk_config.get('limits', {}).get('daily_loss_limit', 0.03),
            max_open_positions=risk_config.get('limits', {}).get('max_open_positions', 5),
            max_correlation=risk_config.get('limits', {}).get('max_correlation', 0.7)
        )
    
    def _parse_performance(self) -> None:
        """Парсинг конфигурации производительности"""
        perf_config = self.config.get('performance_optimization', {})
        
        self.performance = PerformanceConfig(
            async_enabled=perf_config.get('async_processing', {}).get('enabled', True),
            max_concurrent_tasks=perf_config.get('async_processing', {}).get('max_concurrent_tasks', 10),
            caching_enabled=perf_config.get('caching', {}).get('enabled', True),
            vectorization=perf_config.get('algorithms', {}).get('vectorization', True),
            parallel_processing=perf_config.get('algorithms', {}).get('parallel_processing', True)
        )
    
    def get_ai_module_config(self, module_name: str) -> Optional[AIModuleConfig]:
        """Получение конфигурации AI модуля"""
        return self.ai_modules.get(module_name)
    
    def get_current_weights(self) -> Dict[str, float]:
        """Получение текущих весов AI модулей"""
        return self.adaptive_weights.copy()
    
    def update_adaptive_weights(self, performance_data: Dict[str, float]) -> None:
        """Обновление адаптивных весов на основе производительности"""
        try:
            coordination_config = self.config.get('ai_coordination', {}).get('adaptive_weights', {})
            
            if not coordination_config.get('enabled', True):
                return
            
            learning_rate = coordination_config.get('learning_rate', 0.01)
            window_size = coordination_config.get('performance_window', 100)
            
            # Обновление истории производительности
            for module_name, performance in performance_data.items():
                if module_name in self.performance_history:
                    self.performance_history[module_name].append(performance)
                    
                    # Ограничение размера истории
                    if len(self.performance_history[module_name]) > window_size:
                        self.performance_history[module_name] = self.performance_history[module_name][-window_size:]
            
            # Вычисление новых весов
            total_performance = sum(
                sum(history[-20:]) / len(history[-20:]) if history else 0.5
                for history in self.performance_history.values()
            )
            
            if total_performance > 0:
                for module_name in self.adaptive_weights:
                    if module_name in self.performance_history and self.performance_history[module_name]:
                        recent_performance = sum(self.performance_history[module_name][-20:]) / len(self.performance_history[module_name][-20:])
                        target_weight = recent_performance / total_performance
                        
                        # Применение ограничений
                        module_config = self.ai_modules.get(module_name)
                        if module_config:
                            target_weight = max(module_config.min_weight, min(module_config.max_weight, target_weight))
                        
                        # Плавное обновление весов
                        current_weight = self.adaptive_weights[module_name]
                        new_weight = current_weight + learning_rate * (target_weight - current_weight)
                        self.adaptive_weights[module_name] = new_weight
                
                # Нормализация весов
                total_weight = sum(self.adaptive_weights.values())
                if total_weight > 0:
                    for module_name in self.adaptive_weights:
                        self.adaptive_weights[module_name] /= total_weight
                
                self.last_weight_update = datetime.now()
                logger.info(f"Адаптивные веса обновлены: {self.adaptive_weights}")
        
        except Exception as e:
            logger.error(f"Ошибка обновления адаптивных весов: {e}")
    
    def get_market_phase_config(self, phase: str) -> Optional[MarketPhaseConfig]:
        """Получение конфигурации для фазы рынка"""
        return self.market_phases.get(phase)
    
    def get_phase_weights(self, phase: str) -> Dict[str, float]:
        """Получение весов AI модулей для конкретной фазы рынка"""
        phase_config = self.get_market_phase_config(phase)
        if phase_config and phase_config.ai_weights:
            return phase_config.ai_weights
        return self.get_current_weights()
    
    def get_risk_config(self) -> RiskManagementConfig:
        """Получение конфигурации риск-менеджмента"""
        return self.risk_management
    
    def get_performance_config(self) -> PerformanceConfig:
        """Получение конфигурации производительности"""
        return self.performance
    
    def get_system_config(self, section: str) -> Dict[str, Any]:
        """Получение конфигурации системы"""
        return self.config.get(section, {})
    
    def is_module_enabled(self, module_name: str) -> bool:
        """Проверка, включен ли модуль"""
        module_config = self.get_ai_module_config(module_name)
        return module_config.enabled if module_config else False
    
    def get_cache_config(self, cache_type: str = 'indicators') -> Dict[str, Any]:
        """Получение конфигурации кэширования"""
        caching_config = self.config.get('performance_optimization', {}).get('caching', {})
        return caching_config.get(cache_type, {})
    
    def get_monitoring_config(self) -> Dict[str, Any]:
        """Получение конфигурации мониторинга"""
        return self.config.get('monitoring', {})
    
    def get_target_metrics(self) -> Dict[str, float]:
        """Получение целевых метрик"""
        return self.config.get('monitoring', {}).get('performance_metrics', {}).get('targets', {})
    
    def save_config(self) -> None:
        """Сохранение текущей конфигурации"""
        try:
            # Обновление весов в конфигурации
            if 'ai_modules' in self.config and 'weights' in self.config['ai_modules']:
                self.config['ai_modules']['weights'] = self.adaptive_weights
            
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
            
            logger.info("Конфигурация сохранена")
            
        except Exception as e:
            logger.error(f"Ошибка сохранения конфигурации: {e}")
    
    def _create_default_config(self) -> None:
        """Создание конфигурации по умолчанию"""
        logger.info("Создание конфигурации по умолчанию")
        
        # Базовая конфигурация AI модулей
        self.ai_modules = {
            'trading_ai': AIModuleConfig(weight=0.25, min_weight=0.15, max_weight=0.40),
            'lava_ai': AIModuleConfig(weight=0.30, min_weight=0.20, max_weight=0.45),
            'lgbm_ai': AIModuleConfig(weight=0.35, min_weight=0.25, max_weight=0.50),
            'mistral_ai': AIModuleConfig(weight=0.10, min_weight=0.05, max_weight=0.25)
        }
        
        # Инициализация адаптивных весов
        for module_name, config in self.ai_modules.items():
            self.adaptive_weights[module_name] = config.weight
            self.performance_history[module_name] = []
    
    def reload_config(self) -> None:
        """Перезагрузка конфигурации"""
        logger.info("Перезагрузка конфигурации")
        self.load_config()
    
    def validate_config(self) -> List[str]:
        """Валидация конфигурации"""
        errors = []
        
        # Проверка весов
        total_weight = sum(self.adaptive_weights.values())
        if abs(total_weight - 1.0) > 0.01:
            errors.append(f"Сумма весов AI модулей не равна 1.0: {total_weight}")
        
        # Проверка ограничений весов
        for module_name, weight in self.adaptive_weights.items():
            module_config = self.ai_modules.get(module_name)
            if module_config:
                if weight < module_config.min_weight or weight > module_config.max_weight:
                    errors.append(f"Вес модуля {module_name} вне допустимых границ: {weight}")
        
        # Проверка риск-менеджмента
        if self.risk_management.base_position_size > self.risk_management.max_position_size:
            errors.append("Базовый размер позиции больше максимального")
        
        return errors

# Глобальный экземпляр менеджера конфигурации
config_manager = UnifiedConfigManager()

def get_config_manager() -> UnifiedConfigManager:
    """Получение экземпляра менеджера конфигурации"""
    return config_manager