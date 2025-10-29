"""
Система оптимизации производительности для Peper Binance v4
Архитектурные улучшения - Фаза 2
"""

import asyncio
import threading
import time
import logging
import psutil
import gc
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import numpy as np
import pandas as pd
from functools import wraps, lru_cache
import weakref
import pickle
import hashlib
import os

from config.unified_config_manager import get_config_manager

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Метрики производительности"""
    cpu_usage: float
    memory_usage: float
    memory_available: float
    cache_hit_ratio: float
    avg_response_time: float
    active_tasks: int
    completed_tasks: int
    failed_tasks: int
    timestamp: datetime

@dataclass
class CacheStats:
    """Статистика кэша"""
    hits: int = 0
    misses: int = 0
    size: int = 0
    max_size: int = 1000
    
    @property
    def hit_ratio(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

class AsyncCache:
    """Асинхронный кэш с TTL и LRU"""
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 300):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.access_times: Dict[str, float] = {}
        self.stats = CacheStats(max_size=max_size)
        self.lock = asyncio.Lock()
    
    async def get(self, key: str) -> Optional[Any]:
        """Получение значения из кэша"""
        async with self.lock:
            if key in self.cache:
                entry = self.cache[key]
                
                # Проверка TTL
                if time.time() - entry['timestamp'] > entry['ttl']:
                    await self._remove_key(key)
                    self.stats.misses += 1
                    return None
                
                # Обновление времени доступа для LRU
                self.access_times[key] = time.time()
                self.stats.hits += 1
                return entry['value']
            
            self.stats.misses += 1
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Установка значения в кэш"""
        async with self.lock:
            # Проверка размера кэша
            if len(self.cache) >= self.max_size and key not in self.cache:
                await self._evict_lru()
            
            ttl = ttl or self.default_ttl
            self.cache[key] = {
                'value': value,
                'timestamp': time.time(),
                'ttl': ttl
            }
            self.access_times[key] = time.time()
            self.stats.size = len(self.cache)
    
    async def invalidate(self, pattern: Optional[str] = None) -> None:
        """Инвалидация кэша"""
        async with self.lock:
            if pattern:
                keys_to_remove = [k for k in self.cache.keys() if pattern in k]
                for key in keys_to_remove:
                    await self._remove_key(key)
            else:
                self.cache.clear()
                self.access_times.clear()
                self.stats.size = 0
    
    async def _remove_key(self, key: str) -> None:
        """Удаление ключа"""
        if key in self.cache:
            del self.cache[key]
        if key in self.access_times:
            del self.access_times[key]
        self.stats.size = len(self.cache)
    
    async def _evict_lru(self) -> None:
        """Удаление наименее используемого элемента"""
        if not self.access_times:
            return
        
        lru_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        await self._remove_key(lru_key)
    
    def get_stats(self) -> CacheStats:
        """Получение статистики кэша"""
        return self.stats

class TaskPool:
    """Пул задач для асинхронного выполнения"""
    
    def __init__(self, max_workers: int = 10):
        self.max_workers = max_workers
        self.thread_executor = ThreadPoolExecutor(max_workers=max_workers)
        self.process_executor = ProcessPoolExecutor(max_workers=min(4, max_workers))
        self.active_tasks: Dict[str, asyncio.Task] = {}
        self.completed_tasks = 0
        self.failed_tasks = 0
        self.response_times: deque = deque(maxlen=100)
        self.lock = asyncio.Lock()
    
    async def submit_async(self, coro, task_id: Optional[str] = None) -> Any:
        """Отправка асинхронной задачи"""
        task_id = task_id or f"task_{int(time.time() * 1000)}"
        start_time = time.time()
        
        try:
            async with self.lock:
                if len(self.active_tasks) >= self.max_workers:
                    # Ожидание освобождения слота
                    await asyncio.sleep(0.1)
            
            task = asyncio.create_task(coro)
            
            async with self.lock:
                self.active_tasks[task_id] = task
            
            result = await task
            
            # Статистика
            response_time = time.time() - start_time
            self.response_times.append(response_time)
            self.completed_tasks += 1
            
            return result
            
        except Exception as e:
            self.failed_tasks += 1
            logger.error(f"Ошибка выполнения задачи {task_id}: {e}")
            raise
        finally:
            async with self.lock:
                if task_id in self.active_tasks:
                    del self.active_tasks[task_id]
    
    async def submit_thread(self, func: Callable, *args, **kwargs) -> Any:
        """Отправка задачи в пул потоков"""
        loop = asyncio.get_event_loop()
        start_time = time.time()
        
        try:
            result = await loop.run_in_executor(self.thread_executor, func, *args, **kwargs)
            
            response_time = time.time() - start_time
            self.response_times.append(response_time)
            self.completed_tasks += 1
            
            return result
            
        except Exception as e:
            self.failed_tasks += 1
            logger.error(f"Ошибка выполнения задачи в потоке: {e}")
            raise
    
    async def submit_process(self, func: Callable, *args, **kwargs) -> Any:
        """Отправка задачи в пул процессов"""
        loop = asyncio.get_event_loop()
        start_time = time.time()
        
        try:
            result = await loop.run_in_executor(self.process_executor, func, *args, **kwargs)
            
            response_time = time.time() - start_time
            self.response_times.append(response_time)
            self.completed_tasks += 1
            
            return result
            
        except Exception as e:
            self.failed_tasks += 1
            logger.error(f"Ошибка выполнения задачи в процессе: {e}")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """Получение статистики пула"""
        avg_response_time = np.mean(self.response_times) if self.response_times else 0.0
        
        return {
            'active_tasks': len(self.active_tasks),
            'completed_tasks': self.completed_tasks,
            'failed_tasks': self.failed_tasks,
            'avg_response_time': avg_response_time,
            'max_workers': self.max_workers
        }
    
    async def shutdown(self) -> None:
        """Завершение работы пула"""
        # Ожидание завершения активных задач
        if self.active_tasks:
            await asyncio.gather(*self.active_tasks.values(), return_exceptions=True)
        
        # Завершение исполнителей
        self.thread_executor.shutdown(wait=True)
        self.process_executor.shutdown(wait=True)

class MemoryManager:
    """Менеджер памяти"""
    
    def __init__(self, max_memory_percent: float = 80.0):
        self.max_memory_percent = max_memory_percent
        self.weak_refs: List[weakref.ref] = []
        self.last_gc_time = time.time()
        self.gc_interval = 60  # Сборка мусора каждую минуту
    
    def register_object(self, obj: Any) -> None:
        """Регистрация объекта для отслеживания"""
        self.weak_refs.append(weakref.ref(obj))
    
    def check_memory_usage(self) -> Tuple[float, bool]:
        """Проверка использования памяти"""
        memory = psutil.virtual_memory()
        usage_percent = memory.percent
        
        needs_cleanup = usage_percent > self.max_memory_percent
        
        return usage_percent, needs_cleanup
    
    async def cleanup_if_needed(self) -> bool:
        """Очистка памяти при необходимости"""
        usage_percent, needs_cleanup = self.check_memory_usage()
        
        if needs_cleanup or (time.time() - self.last_gc_time > self.gc_interval):
            await self._perform_cleanup()
            self.last_gc_time = time.time()
            return True
        
        return False
    
    async def _perform_cleanup(self) -> None:
        """Выполнение очистки памяти"""
        # Удаление мертвых ссылок
        self.weak_refs = [ref for ref in self.weak_refs if ref() is not None]
        
        # Принудительная сборка мусора
        collected = gc.collect()
        
        logger.info(f"Очистка памяти: собрано {collected} объектов")

class DataProcessor:
    """Процессор данных с оптимизацией"""
    
    def __init__(self, cache: AsyncCache, task_pool: TaskPool):
        self.cache = cache
        self.task_pool = task_pool
        self.batch_size = 1000
        self.processing_stats = defaultdict(int)
    
    async def process_market_data(self, data: List[Dict[str, Any]], 
                                symbol: str) -> List[Dict[str, Any]]:
        """Обработка рыночных данных"""
        cache_key = f"market_data_{symbol}_{len(data)}"
        
        # Проверка кэша
        cached_result = await self.cache.get(cache_key)
        if cached_result is not None:
            return cached_result
        
        # Обработка данных батчами
        if len(data) > self.batch_size:
            result = await self._process_in_batches(data, symbol)
        else:
            result = await self._process_single_batch(data, symbol)
        
        # Кэширование результата
        await self.cache.set(cache_key, result, ttl=60)  # 1 минута
        
        self.processing_stats[f'processed_{symbol}'] += len(data)
        
        return result
    
    async def _process_in_batches(self, data: List[Dict[str, Any]], 
                                symbol: str) -> List[Dict[str, Any]]:
        """Обработка данных батчами"""
        batches = [data[i:i + self.batch_size] 
                  for i in range(0, len(data), self.batch_size)]
        
        # Параллельная обработка батчей
        tasks = [
            self.task_pool.submit_async(
                self._process_single_batch(batch, symbol),
                f"batch_{symbol}_{i}"
            )
            for i, batch in enumerate(batches)
        ]
        
        results = await asyncio.gather(*tasks)
        
        # Объединение результатов
        combined_result = []
        for result in results:
            combined_result.extend(result)
        
        return combined_result
    
    async def _process_single_batch(self, data: List[Dict[str, Any]], 
                                  symbol: str) -> List[Dict[str, Any]]:
        """Обработка одного батча"""
        # Симуляция обработки данных
        processed_data = []
        
        for item in data:
            # Добавление вычисленных полей
            processed_item = item.copy()
            processed_item['processed_timestamp'] = datetime.now().isoformat()
            processed_item['symbol'] = symbol
            
            # Простые технические индикаторы
            if 'price' in item:
                processed_item['price_change'] = item.get('price', 0) - item.get('prev_price', 0)
                processed_item['price_change_percent'] = (
                    processed_item['price_change'] / item.get('prev_price', 1) * 100
                    if item.get('prev_price', 0) != 0 else 0
                )
            
            processed_data.append(processed_item)
        
        return processed_data
    
    def get_processing_stats(self) -> Dict[str, int]:
        """Получение статистики обработки"""
        return dict(self.processing_stats)

class PerformanceOptimizer:
    """Основной класс оптимизации производительности"""
    
    def __init__(self):
        self.config_manager = get_config_manager()
        self.performance_config = self.config_manager.get_performance_config()
        
        # Компоненты
        self.cache = AsyncCache(
            max_size=self.performance_config.cache_size,
            default_ttl=self.performance_config.cache_ttl
        )
        
        self.task_pool = TaskPool(
            max_workers=self.performance_config.max_workers
        )
        
        self.memory_manager = MemoryManager(
            max_memory_percent=self.performance_config.memory_threshold
        )
        
        self.data_processor = DataProcessor(self.cache, self.task_pool)
        
        # Метрики
        self.metrics_history: deque = deque(maxlen=1000)
        self.last_metrics_time = time.time()
        
        # Мониторинг
        self.monitoring_task: Optional[asyncio.Task] = None
        self.is_monitoring = False
        
        logger.info("Оптимизатор производительности инициализирован")
    
    async def start_monitoring(self) -> None:
        """Запуск мониторинга производительности"""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Мониторинг производительности запущен")
    
    async def stop_monitoring(self) -> None:
        """Остановка мониторинга"""
        self.is_monitoring = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        await self.task_pool.shutdown()
        logger.info("Мониторинг производительности остановлен")
    
    async def _monitoring_loop(self) -> None:
        """Цикл мониторинга"""
        while self.is_monitoring:
            try:
                # Сбор метрик
                metrics = await self._collect_metrics()
                self.metrics_history.append(metrics)
                
                # Проверка памяти
                await self.memory_manager.cleanup_if_needed()
                
                # Оптимизация кэша
                await self._optimize_cache()
                
                # Ожидание
                await asyncio.sleep(self.performance_config.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Ошибка в цикле мониторинга: {e}")
                await asyncio.sleep(5)
    
    async def _collect_metrics(self) -> PerformanceMetrics:
        """Сбор метрик производительности"""
        # Системные метрики
        cpu_usage = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        memory_usage = memory.percent
        memory_available = memory.available / (1024 * 1024 * 1024)  # GB
        
        # Метрики кэша
        cache_stats = self.cache.get_stats()
        cache_hit_ratio = cache_stats.hit_ratio
        
        # Метрики задач
        task_stats = self.task_pool.get_stats()
        
        metrics = PerformanceMetrics(
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            memory_available=memory_available,
            cache_hit_ratio=cache_hit_ratio,
            avg_response_time=task_stats['avg_response_time'],
            active_tasks=task_stats['active_tasks'],
            completed_tasks=task_stats['completed_tasks'],
            failed_tasks=task_stats['failed_tasks'],
            timestamp=datetime.now()
        )
        
        return metrics
    
    async def _optimize_cache(self) -> None:
        """Оптимизация кэша"""
        cache_stats = self.cache.get_stats()
        
        # Если коэффициент попаданий низкий, увеличиваем TTL
        if cache_stats.hit_ratio < 0.5 and cache_stats.hits + cache_stats.misses > 100:
            self.cache.default_ttl = min(self.cache.default_ttl * 1.1, 600)  # Максимум 10 минут
        
        # Если кэш переполнен, очищаем старые записи
        if cache_stats.size > cache_stats.max_size * 0.9:
            await self.cache.invalidate()
    
    # Декораторы для оптимизации
    def cached(self, ttl: int = 300, key_func: Optional[Callable] = None):
        """Декоратор для кэширования результатов функций"""
        def decorator(func: Callable):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # Генерация ключа кэша
                if key_func:
                    cache_key = key_func(*args, **kwargs)
                else:
                    cache_key = f"{func.__name__}_{hash(str(args) + str(kwargs))}"
                
                # Проверка кэша
                cached_result = await self.cache.get(cache_key)
                if cached_result is not None:
                    return cached_result
                
                # Выполнение функции
                result = await func(*args, **kwargs)
                
                # Кэширование результата
                await self.cache.set(cache_key, result, ttl)
                
                return result
            
            return wrapper
        return decorator
    
    def async_task(self, task_id: Optional[str] = None):
        """Декоратор для асинхронного выполнения задач"""
        def decorator(func: Callable):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                return await self.task_pool.submit_async(
                    func(*args, **kwargs),
                    task_id or f"{func.__name__}_{int(time.time() * 1000)}"
                )
            
            return wrapper
        return decorator
    
    def thread_task(self):
        """Декоратор для выполнения в пуле потоков"""
        def decorator(func: Callable):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                return await self.task_pool.submit_thread(func, *args, **kwargs)
            
            return wrapper
        return decorator
    
    # Методы для работы с данными
    async def process_market_data(self, data: List[Dict[str, Any]], 
                                symbol: str) -> List[Dict[str, Any]]:
        """Обработка рыночных данных"""
        return await self.data_processor.process_market_data(data, symbol)
    
    async def batch_process(self, items: List[Any], 
                          processor: Callable, batch_size: int = 100) -> List[Any]:
        """Батчевая обработка элементов"""
        if len(items) <= batch_size:
            return await processor(items)
        
        batches = [items[i:i + batch_size] for i in range(0, len(items), batch_size)]
        
        tasks = [
            self.task_pool.submit_async(
                processor(batch),
                f"batch_process_{i}"
            )
            for i, batch in enumerate(batches)
        ]
        
        results = await asyncio.gather(*tasks)
        
        # Объединение результатов
        combined_result = []
        for result in results:
            if isinstance(result, list):
                combined_result.extend(result)
            else:
                combined_result.append(result)
        
        return combined_result
    
    # Методы для получения статистики
    def get_performance_metrics(self) -> Optional[PerformanceMetrics]:
        """Получение текущих метрик производительности"""
        return self.metrics_history[-1] if self.metrics_history else None
    
    def get_performance_history(self, minutes: int = 60) -> List[PerformanceMetrics]:
        """Получение истории метрик"""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        return [
            metrics for metrics in self.metrics_history
            if metrics.timestamp >= cutoff_time
        ]
    
    def get_cache_stats(self) -> CacheStats:
        """Получение статистики кэша"""
        return self.cache.get_stats()
    
    def get_task_stats(self) -> Dict[str, Any]:
        """Получение статистики задач"""
        return self.task_pool.get_stats()
    
    def get_processing_stats(self) -> Dict[str, int]:
        """Получение статистики обработки данных"""
        return self.data_processor.get_processing_stats()
    
    def get_system_info(self) -> Dict[str, Any]:
        """Получение информации о системе"""
        memory = psutil.virtual_memory()
        cpu_count = psutil.cpu_count()
        
        return {
            'cpu_count': cpu_count,
            'memory_total_gb': memory.total / (1024 * 1024 * 1024),
            'memory_available_gb': memory.available / (1024 * 1024 * 1024),
            'memory_percent': memory.percent,
            'cpu_percent': psutil.cpu_percent(interval=1)
        }
    
    async def invalidate_cache(self, pattern: Optional[str] = None) -> None:
        """Инвалидация кэша"""
        await self.cache.invalidate(pattern)
    
    async def force_cleanup(self) -> None:
        """Принудительная очистка ресурсов"""
        await self.memory_manager._perform_cleanup()
        await self.cache.invalidate()
        
        logger.info("Принудительная очистка ресурсов выполнена")

# Глобальный экземпляр оптимизатора
_optimizer_instance: Optional[PerformanceOptimizer] = None

def get_performance_optimizer() -> PerformanceOptimizer:
    """Получение глобального экземпляра оптимизатора"""
    global _optimizer_instance
    if _optimizer_instance is None:
        _optimizer_instance = PerformanceOptimizer()
    return _optimizer_instance

# Удобные функции
async def start_performance_monitoring():
    """Запуск мониторинга производительности"""
    optimizer = get_performance_optimizer()
    await optimizer.start_monitoring()

async def stop_performance_monitoring():
    """Остановка мониторинга производительности"""
    optimizer = get_performance_optimizer()
    await optimizer.stop_monitoring()

def performance_cached(ttl: int = 300, key_func: Optional[Callable] = None):
    """Декоратор для кэширования"""
    optimizer = get_performance_optimizer()
    return optimizer.cached(ttl, key_func)

def performance_async_task(task_id: Optional[str] = None):
    """Декоратор для асинхронных задач"""
    optimizer = get_performance_optimizer()
    return optimizer.async_task(task_id)

def performance_thread_task():
    """Декоратор для задач в потоках"""
    optimizer = get_performance_optimizer()
    return optimizer.thread_task()