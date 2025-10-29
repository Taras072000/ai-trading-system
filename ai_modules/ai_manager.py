"""
AI Manager - Главный интерфейс для управления всеми AI модулями
Peper Binance v4 - Оптимизированная версия для минимального потребления ресурсов
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import gc
import psutil
import os
import config
from config_params import CONFIG_PARAMS

# Ленивые импорты AI модулей
_trading_ai = None
_lava_ai = None
_mistral_ai = None
_lgbm_ai = None

logger = logging.getLogger(__name__)

class AIModuleType(Enum):
    """Типы AI модулей"""
    TRADING = "trading"
    LAVA = "lava"
    MISTRAL = "mistral"
    LGBM = "lgbm"

class AIModuleStatus(Enum):
    """Статусы AI модулей"""
    UNLOADED = "unloaded"
    LOADING = "loading"
    READY = "ready"
    BUSY = "busy"
    ERROR = "error"

@dataclass
class AIModuleInfo:
    """Информация о AI модуле"""
    name: str
    type: AIModuleType
    status: AIModuleStatus
    memory_usage_mb: float
    last_used: datetime
    initialization_time: Optional[datetime]
    error_message: Optional[str] = None

@dataclass
class AIResponse:
    """Универсальный ответ от AI модуля"""
    module_type: AIModuleType
    success: bool
    data: Any
    error: Optional[str] = None
    execution_time_ms: float = 0.0
    memory_used_mb: float = 0.0
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

class AIManager:
    """
    Главный менеджер AI модулей с оптимизацией ресурсов
    Управляет загрузкой, выгрузкой и использованием AI модулей
    """
    
    def __init__(self):
        # Получаем конфигурацию AI модулей из CONFIG_PARAMS
        ai_config = CONFIG_PARAMS.get('ai_modules', {})
        manager_config = ai_config.get('manager', {})
        
        self.config = manager_config
        self.modules = {}
        self.module_status = {}
        self.last_cleanup = datetime.now()
        self.max_memory_mb = manager_config.get('memory_limit_mb', 2048)  # 2GB лимит по умолчанию
        self.auto_unload_timeout = manager_config.get('auto_unload_timeout', 1800)  # 30 минут
        
        # Инициализируем статусы модулей
        for module_type in AIModuleType:
            self.module_status[module_type] = AIModuleInfo(
                name=module_type.value,
                type=module_type,
                status=AIModuleStatus.UNLOADED,
                memory_usage_mb=0.0,
                last_used=datetime.now(),
                initialization_time=None
            )
        
        logger.info("AI Manager инициализирован с оптимизацией ресурсов")
    
    async def initialize(self):
        """Инициализация менеджера"""
        logger.info("Инициализация AI Manager...")
        
        # Проверяем доступную память
        available_memory = self._get_available_memory_mb()
        if available_memory < 512:  # Минимум 512MB
            logger.warning(f"Мало доступной памяти: {available_memory}MB")
        
        logger.info("AI Manager готов к работе")
        return True
    
    async def load_module(self, module_type: AIModuleType, force: bool = False) -> bool:
        """
        Загрузка AI модуля с проверкой ресурсов
        """
        try:
            module_info = self.module_status[module_type]
            
            # Проверяем, не загружен ли уже модуль
            if module_info.status == AIModuleStatus.READY and not force:
                logger.info(f"Модуль {module_type.value} уже загружен")
                return True
            
            # Проверяем доступную память
            if not await self._check_memory_availability():
                logger.warning("Недостаточно памяти для загрузки модуля")
                await self._free_memory()
            
            # Обновляем статус
            module_info.status = AIModuleStatus.LOADING
            start_time = datetime.now()
            
            logger.info(f"Загрузка модуля {module_type.value}...")
            
            # Ленивая загрузка модуля
            module = await self._lazy_load_module(module_type)
            if module is None:
                module_info.status = AIModuleStatus.ERROR
                module_info.error_message = "Ошибка загрузки модуля"
                return False
            
            # Инициализируем модуль
            if hasattr(module, 'initialize'):
                success = await module.initialize()
                if not success:
                    module_info.status = AIModuleStatus.ERROR
                    module_info.error_message = "Ошибка инициализации модуля"
                    return False
            
            # Сохраняем модуль
            self.modules[module_type] = module
            
            # Обновляем информацию о модуле
            module_info.status = AIModuleStatus.READY
            module_info.initialization_time = datetime.now()
            module_info.last_used = datetime.now()
            module_info.memory_usage_mb = self._estimate_module_memory(module)
            module_info.error_message = None
            
            load_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"Модуль {module_type.value} загружен за {load_time:.2f}с")
            
            return True
            
        except Exception as e:
            logger.error(f"Ошибка загрузки модуля {module_type.value}: {e}")
            self.module_status[module_type].status = AIModuleStatus.ERROR
            self.module_status[module_type].error_message = str(e)
            return False
    
    async def _lazy_load_module(self, module_type: AIModuleType):
        """Ленивая загрузка модуля"""
        global _trading_ai, _lava_ai, _mistral_ai, _lgbm_ai
        
        try:
            if module_type == AIModuleType.TRADING:
                if _trading_ai is None:
                    from .trading_ai import TradingAI
                    _trading_ai = TradingAI()
                return _trading_ai
            
            elif module_type == AIModuleType.LAVA:
                if _lava_ai is None:
                    from .lava_ai import LavaAI
                    _lava_ai = LavaAI()
                return _lava_ai
            
            elif module_type == AIModuleType.MISTRAL:
                if _mistral_ai is None:
                    from .mistral_ai import MistralAI
                    _mistral_ai = MistralAI()
                return _mistral_ai
            
            elif module_type == AIModuleType.LGBM:
                if _lgbm_ai is None:
                    from .lgbm_ai import LGBMAI
                    _lgbm_ai = LGBMAI()
                return _lgbm_ai
            
            return None
            
        except Exception as e:
            logger.error(f"Ошибка ленивой загрузки {module_type.value}: {e}")
            return None
    
    async def unload_module(self, module_type: AIModuleType) -> bool:
        """Выгрузка AI модуля"""
        try:
            if module_type not in self.modules:
                logger.info(f"Модуль {module_type.value} не загружен")
                return True
            
            logger.info(f"Выгрузка модуля {module_type.value}...")
            
            # Очищаем ресурсы модуля
            module = self.modules[module_type]
            if hasattr(module, 'cleanup'):
                await module.cleanup()
            
            # Удаляем модуль
            del self.modules[module_type]
            
            # Обновляем статус
            module_info = self.module_status[module_type]
            module_info.status = AIModuleStatus.UNLOADED
            module_info.memory_usage_mb = 0.0
            module_info.initialization_time = None
            module_info.error_message = None
            
            # Принудительная сборка мусора
            gc.collect()
            
            logger.info(f"Модуль {module_type.value} выгружен")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка выгрузки модуля {module_type.value}: {e}")
            return False
    
    async def execute_ai_task(self, module_type: AIModuleType, method: str, 
                             *args, **kwargs) -> AIResponse:
        """
        Выполнение задачи AI модуля с автоматической загрузкой
        """
        start_time = datetime.now()
        start_memory = self._get_memory_usage_mb()
        
        try:
            # Загружаем модуль если необходимо
            if module_type not in self.modules:
                success = await self.load_module(module_type)
                if not success:
                    return AIResponse(
                        module_type=module_type,
                        success=False,
                        data=None,
                        error="Не удалось загрузить модуль"
                    )
            
            # Обновляем статус и время использования
            module_info = self.module_status[module_type]
            module_info.status = AIModuleStatus.BUSY
            module_info.last_used = datetime.now()
            
            # Выполняем задачу
            module = self.modules[module_type]
            if not hasattr(module, method):
                return AIResponse(
                    module_type=module_type,
                    success=False,
                    data=None,
                    error=f"Метод {method} не найден в модуле"
                )
            
            result = await getattr(module, method)(*args, **kwargs)
            
            # Обновляем статус
            module_info.status = AIModuleStatus.READY
            
            # Вычисляем метрики
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            memory_used = self._get_memory_usage_mb() - start_memory
            
            # Периодическая очистка
            await self._periodic_maintenance()
            
            return AIResponse(
                module_type=module_type,
                success=True,
                data=result,
                execution_time_ms=execution_time,
                memory_used_mb=memory_used
            )
            
        except Exception as e:
            logger.error(f"Ошибка выполнения {method} в {module_type.value}: {e}")
            
            # Восстанавливаем статус
            if module_type in self.module_status:
                self.module_status[module_type].status = AIModuleStatus.READY
            
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return AIResponse(
                module_type=module_type,
                success=False,
                data=None,
                error=str(e),
                execution_time_ms=execution_time
            )
    
    async def get_trading_analysis(self, market_data: Dict[str, Any]) -> AIResponse:
        """Получение торгового анализа"""
        return await self.execute_ai_task(
            AIModuleType.TRADING, 
            'analyze_market', 
            market_data
        )
    
    async def get_lava_analysis(self, data: Dict[str, Any]) -> AIResponse:
        """Получение Lava анализа"""
        return await self.execute_ai_task(
            AIModuleType.LAVA, 
            'analyze_patterns', 
            data
        )
    
    async def get_mistral_response(self, prompt: str) -> AIResponse:
        """Получение ответа от Mistral"""
        return await self.execute_ai_task(
            AIModuleType.MISTRAL, 
            'generate_text', 
            prompt
        )
    
    async def get_lgbm_prediction(self, features: Dict[str, Any]) -> AIResponse:
        """Получение предсказания LGBM"""
        return await self.execute_ai_task(
            AIModuleType.LGBM, 
            'predict_price_movement', 
            features
        )
    
    async def _check_memory_availability(self) -> bool:
        """Проверка доступности памяти"""
        current_usage = self._get_memory_usage_mb()
        available = self._get_available_memory_mb()
        
        # Проверяем лимиты
        if current_usage > self.max_memory_mb:
            return False
        
        if available < 256:  # Минимум 256MB свободной памяти
            return False
        
        return True
    
    async def _free_memory(self):
        """Освобождение памяти путем выгрузки неиспользуемых модулей"""
        logger.info("Освобождение памяти...")
        
        # Находим модули для выгрузки (давно не использовались)
        now = datetime.now()
        modules_to_unload = []
        
        for module_type, info in self.module_status.items():
            if info.status == AIModuleStatus.READY:
                time_since_use = (now - info.last_used).total_seconds()
                if time_since_use > self.auto_unload_timeout:
                    modules_to_unload.append(module_type)
        
        # Выгружаем модули
        for module_type in modules_to_unload:
            await self.unload_module(module_type)
        
        # Принудительная сборка мусора
        gc.collect()
        
        logger.info(f"Выгружено модулей: {len(modules_to_unload)}")
    
    async def _periodic_maintenance(self):
        """Периодическое обслуживание"""
        now = datetime.now()
        if (now - self.last_cleanup).seconds > 300:  # Каждые 5 минут
            
            # Проверяем использование памяти
            if not await self._check_memory_availability():
                await self._free_memory()
            
            # Обновляем статистику модулей
            for module_type, module in self.modules.items():
                if hasattr(module, 'get_memory_usage'):
                    try:
                        usage = await module.get_memory_usage()
                        self.module_status[module_type].memory_usage_mb = usage.get('rss_mb', 0)
                    except:
                        pass
            
            self.last_cleanup = now
            logger.debug("Выполнено периодическое обслуживание AI Manager")
    
    def _estimate_module_memory(self, module) -> float:
        """Оценка использования памяти модулем"""
        try:
            if hasattr(module, 'get_memory_usage'):
                usage = module.get_memory_usage()
                return usage.get('rss_mb', 50.0)  # Базовая оценка
            return 50.0  # Базовая оценка для модуля
        except:
            return 50.0
    
    def _get_memory_usage_mb(self) -> float:
        """Получение текущего использования памяти процессом"""
        try:
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024
        except:
            return 0.0
    
    def _get_available_memory_mb(self) -> float:
        """Получение доступной памяти системы"""
        try:
            return psutil.virtual_memory().available / 1024 / 1024
        except:
            return 1024.0  # Базовая оценка
    
    def get_modules_status(self) -> Dict[str, AIModuleInfo]:
        """Получение статуса всех модулей"""
        return {
            module_type.value: info 
            for module_type, info in self.module_status.items()
        }
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Получение системной статистики"""
        loaded_modules = len(self.modules)
        total_memory = sum(
            info.memory_usage_mb 
            for info in self.module_status.values()
        )
        
        return {
            'loaded_modules': loaded_modules,
            'total_modules': len(AIModuleType),
            'total_memory_mb': total_memory,
            'max_memory_mb': self.max_memory_mb,
            'system_memory_mb': self._get_memory_usage_mb(),
            'available_memory_mb': self._get_available_memory_mb(),
            'auto_unload_timeout': self.auto_unload_timeout
        }
    
    async def preload_essential_modules(self):
        """Предварительная загрузка основных модулей"""
        logger.info("Предварительная загрузка основных модулей...")
        
        # Загружаем только Trading AI по умолчанию
        essential_modules = [AIModuleType.TRADING]
        
        for module_type in essential_modules:
            try:
                await self.load_module(module_type)
            except Exception as e:
                logger.error(f"Ошибка предварительной загрузки {module_type.value}: {e}")
    
    def clear_cache(self):
        """Очистка кешей всех загруженных AI модулей"""
        try:
            cleared_modules = []
            
            # Очищаем кеши всех загруженных модулей
            for module_type, module in self.modules.items():
                if hasattr(module, 'clear_all_cache'):
                    module.clear_all_cache()
                    cleared_modules.append(module_type.value)
                elif hasattr(module, 'clear_cache'):
                    module.clear_cache()
                    cleared_modules.append(module_type.value)
            
            # Принудительная сборка мусора
            gc.collect()
            
            if cleared_modules:
                logger.info(f"AIManager: Кеши очищены для модулей: {', '.join(cleared_modules)}")
            else:
                logger.info("AIManager: Нет загруженных модулей для очистки кеша")
                
        except Exception as e:
            logger.error(f"Ошибка очистки кешей в AIManager: {e}")

    async def shutdown(self):
        """Корректное завершение работы менеджера"""
        logger.info("Завершение работы AI Manager...")
        
        # Выгружаем все модули
        for module_type in list(self.modules.keys()):
            await self.unload_module(module_type)
        
        # Очищаем глобальные переменные
        global _trading_ai, _lava_ai, _mistral_ai, _lgbm_ai
        _trading_ai = None
        _lava_ai = None
        _mistral_ai = None
        _lgbm_ai = None
        
        # Финальная сборка мусора
        gc.collect()
        
        logger.info("AI Manager завершил работу")

# Глобальный экземпляр менеджера
ai_manager = AIManager()

# Удобные функции для быстрого доступа
async def get_trading_analysis(market_data: Dict[str, Any]) -> AIResponse:
    """Быстрый доступ к торговому анализу"""
    return await ai_manager.get_trading_analysis(market_data)

async def get_lava_analysis(data: Dict[str, Any]) -> AIResponse:
    """Быстрый доступ к Lava анализу"""
    return await ai_manager.get_lava_analysis(data)

async def get_mistral_response(prompt: str) -> AIResponse:
    """Быстрый доступ к Mistral"""
    return await ai_manager.get_mistral_response(prompt)

async def get_lgbm_prediction(features: Dict[str, Any]) -> AIResponse:
    """Быстрый доступ к LGBM предсказанию"""
    return await ai_manager.get_lgbm_prediction(features)