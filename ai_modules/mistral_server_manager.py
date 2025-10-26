#!/usr/bin/env python3
"""
Менеджер автоматического управления Mistral сервером для системы Peper Binance v4
Автоматический запуск, остановка и мониторинг Mistral сервера
"""

import asyncio
import logging
import subprocess
import time
import requests
import psutil
import signal
import os
from typing import Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class MistralServerConfig:
    """Конфигурация Mistral сервера"""
    host: str = "localhost"
    port: int = 11434
    model_name: str = "mistral"
    startup_timeout: int = 120  # секунды
    health_check_interval: int = 10  # секунды
    max_startup_attempts: int = 3
    server_command: str = "ollama serve"
    model_pull_command: str = "ollama pull mistral"

class MistralServerManager:
    """
    Менеджер автоматического управления Mistral сервером
    Обеспечивает автоматический запуск, остановку и мониторинг сервера
    """
    
    def __init__(self, config: MistralServerConfig = None):
        self.config = config or MistralServerConfig()
        self.server_process: Optional[subprocess.Popen] = None
        self.is_running = False
        self.startup_time: Optional[datetime] = None
        self.last_health_check: Optional[datetime] = None
        
        logger.info("🤖 MistralServerManager инициализирован")
        logger.info(f"🔧 Конфигурация: {self.config.host}:{self.config.port}")
    
    async def start_server(self, force_restart: bool = False) -> bool:
        """
        Запуск Mistral сервера
        
        Args:
            force_restart: Принудительный перезапуск если сервер уже запущен
        
        Returns:
            bool: True если сервер успешно запущен
        """
        try:
            # Проверяем, не запущен ли уже сервер
            if await self.is_server_running() and not force_restart:
                logger.info("✅ Mistral сервер уже запущен")
                return True
            
            # Останавливаем существующий сервер если нужен перезапуск
            if force_restart:
                await self.stop_server()
                await asyncio.sleep(2)
            
            logger.info("🚀 Запуск Mistral сервера...")
            
            # Проверяем доступность ollama
            if not await self._check_ollama_available():
                logger.error("❌ Ollama не найден в системе")
                return False
            
            # Проверяем наличие модели Mistral
            if not await self._check_model_available():
                logger.info("📥 Загрузка модели Mistral...")
                if not await self._pull_model():
                    logger.error("❌ Не удалось загрузить модель Mistral")
                    return False
            
            # Запускаем сервер
            for attempt in range(self.config.max_startup_attempts):
                logger.info(f"🔄 Попытка запуска {attempt + 1}/{self.config.max_startup_attempts}")
                
                if await self._start_server_process():
                    # Ждем готовности сервера
                    if await self._wait_for_server_ready():
                        self.is_running = True
                        self.startup_time = datetime.now()
                        logger.info("✅ Mistral сервер успешно запущен")
                        return True
                    else:
                        logger.warning(f"⚠️ Сервер не готов после попытки {attempt + 1}")
                        await self._cleanup_failed_start()
                else:
                    logger.warning(f"⚠️ Не удалось запустить процесс сервера, попытка {attempt + 1}")
                
                if attempt < self.config.max_startup_attempts - 1:
                    await asyncio.sleep(5)
            
            logger.error("❌ Не удалось запустить Mistral сервер после всех попыток")
            return False
            
        except Exception as e:
            logger.error(f"❌ Ошибка запуска Mistral сервера: {e}")
            return False
    
    async def stop_server(self) -> bool:
        """
        Остановка Mistral сервера
        
        Returns:
            bool: True если сервер успешно остановлен
        """
        try:
            logger.info("🛑 Остановка Mistral сервера...")
            
            # Останавливаем наш процесс
            if self.server_process:
                try:
                    # Сначала пытаемся мягко завершить
                    self.server_process.terminate()
                    
                    # Ждем завершения
                    try:
                        await asyncio.wait_for(
                            asyncio.create_task(self._wait_process_termination()),
                            timeout=10
                        )
                    except asyncio.TimeoutError:
                        # Принудительное завершение
                        logger.warning("⚠️ Принудительное завершение процесса сервера")
                        self.server_process.kill()
                        await asyncio.sleep(2)
                    
                    self.server_process = None
                    
                except Exception as e:
                    logger.warning(f"⚠️ Ошибка при остановке процесса: {e}")
            
            # Дополнительно ищем и останавливаем все процессы ollama
            await self._kill_ollama_processes()
            
            # Проверяем что сервер действительно остановлен
            for _ in range(10):
                if not await self.is_server_running():
                    break
                await asyncio.sleep(1)
            
            self.is_running = False
            self.startup_time = None
            self.last_health_check = None
            
            if await self.is_server_running():
                logger.warning("⚠️ Сервер все еще отвечает после остановки")
                return False
            else:
                logger.info("✅ Mistral сервер успешно остановлен")
                return True
                
        except Exception as e:
            logger.error(f"❌ Ошибка остановки Mistral сервера: {e}")
            return False
    
    async def is_server_running(self) -> bool:
        """
        Проверка работает ли Mistral сервер
        
        Returns:
            bool: True если сервер работает
        """
        try:
            url = f"http://{self.config.host}:{self.config.port}/api/tags"
            
            # Используем короткий таймаут для быстрой проверки
            response = requests.get(url, timeout=5)
            
            if response.status_code == 200:
                self.last_health_check = datetime.now()
                return True
            else:
                return False
                
        except Exception:
            return False
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Полная проверка здоровья Mistral сервера
        
        Returns:
            Dict с информацией о состоянии сервера
        """
        health_info = {
            'is_running': False,
            'response_time': None,
            'model_available': False,
            'last_check': datetime.now(),
            'uptime': None,
            'error': None
        }
        
        try:
            # Проверяем доступность сервера
            start_time = time.time()
            
            url = f"http://{self.config.host}:{self.config.port}/api/tags"
            response = requests.get(url, timeout=10)
            
            response_time = time.time() - start_time
            health_info['response_time'] = response_time
            
            if response.status_code == 200:
                health_info['is_running'] = True
                
                # Проверяем доступность модели
                models_data = response.json()
                model_names = [model.get('name', '') for model in models_data.get('models', [])]
                health_info['model_available'] = any(self.config.model_name in name for name in model_names)
                
                # Рассчитываем uptime
                if self.startup_time:
                    uptime = datetime.now() - self.startup_time
                    health_info['uptime'] = uptime.total_seconds()
                
                self.last_health_check = datetime.now()
            
        except Exception as e:
            health_info['error'] = str(e)
            logger.debug(f"Health check error: {e}")
        
        return health_info
    
    async def test_model_inference(self, test_prompt: str = "Hello, how are you?") -> Dict[str, Any]:
        """
        Тестирование инференса модели
        
        Args:
            test_prompt: Тестовый промпт
        
        Returns:
            Dict с результатами теста
        """
        test_result = {
            'success': False,
            'response_time': None,
            'response_text': None,
            'error': None,
            'timestamp': datetime.now()
        }
        
        try:
            if not await self.is_server_running():
                test_result['error'] = "Сервер не запущен"
                return test_result
            
            start_time = time.time()
            
            url = f"http://{self.config.host}:{self.config.port}/api/generate"
            payload = {
                "model": self.config.model_name,
                "prompt": test_prompt,
                "stream": False
            }
            
            response = requests.post(url, json=payload, timeout=30)
            response_time = time.time() - start_time
            
            test_result['response_time'] = response_time
            
            if response.status_code == 200:
                response_data = response.json()
                test_result['response_text'] = response_data.get('response', '')
                test_result['success'] = True
                logger.info(f"✅ Тест инференса успешен (время: {response_time:.2f}с)")
            else:
                test_result['error'] = f"HTTP {response.status_code}: {response.text}"
                
        except Exception as e:
            test_result['error'] = str(e)
            logger.error(f"❌ Ошибка теста инференса: {e}")
        
        return test_result
    
    async def get_server_info(self) -> Dict[str, Any]:
        """Получение информации о сервере"""
        info = {
            'config': {
                'host': self.config.host,
                'port': self.config.port,
                'model_name': self.config.model_name
            },
            'status': {
                'is_running': self.is_running,
                'startup_time': self.startup_time,
                'last_health_check': self.last_health_check,
                'process_id': self.server_process.pid if self.server_process else None
            },
            'health': await self.health_check()
        }
        
        return info
    
    def get_server_status(self) -> Dict[str, Any]:
        """
        Получение статуса сервера (синхронная версия для совместимости)
        
        Returns:
            Dict с информацией о статусе сервера
        """
        try:
            # Синхронная проверка статуса
            is_running = False
            try:
                url = f"http://{self.config.host}:{self.config.port}/api/tags"
                response = requests.get(url, timeout=5)
                is_running = response.status_code == 200
            except:
                is_running = False
            
            return {
                'is_running': is_running,
                'host': self.config.host,
                'port': self.config.port,
                'model_name': self.config.model_name,
                'startup_time': self.startup_time,
                'last_health_check': self.last_health_check,
                'process_id': self.server_process.pid if self.server_process else None
            }
        except Exception as e:
            logger.error(f"❌ Ошибка получения статуса сервера: {e}")
            return {
                'is_running': False,
                'error': str(e)
            }
    
    async def _check_ollama_available(self) -> bool:
        """Проверка доступности ollama в системе"""
        try:
            result = subprocess.run(['which', 'ollama'], capture_output=True, text=True)
            return result.returncode == 0
        except Exception:
            return False
    
    async def _check_model_available(self) -> bool:
        """Проверка наличия модели Mistral"""
        try:
            result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
            if result.returncode == 0:
                return self.config.model_name in result.stdout
            return False
        except Exception:
            return False
    
    async def _pull_model(self) -> bool:
        """Загрузка модели Mistral"""
        try:
            logger.info(f"📥 Загрузка модели {self.config.model_name}...")
            
            process = subprocess.Popen(
                self.config.model_pull_command.split(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Ждем завершения загрузки (может занять много времени)
            stdout, stderr = process.communicate(timeout=600)  # 10 минут таймаут
            
            if process.returncode == 0:
                logger.info("✅ Модель успешно загружена")
                return True
            else:
                logger.error(f"❌ Ошибка загрузки модели: {stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("❌ Таймаут загрузки модели")
            process.kill()
            return False
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки модели: {e}")
            return False
    
    async def _start_server_process(self) -> bool:
        """Запуск процесса сервера"""
        try:
            # Устанавливаем переменные окружения
            env = os.environ.copy()
            env['OLLAMA_HOST'] = f"{self.config.host}:{self.config.port}"
            
            self.server_process = subprocess.Popen(
                self.config.server_command.split(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                preexec_fn=os.setsid  # Создаем новую группу процессов
            )
            
            # Даем процессу время на запуск
            await asyncio.sleep(3)
            
            # Проверяем что процесс не завершился сразу
            if self.server_process.poll() is not None:
                logger.error("❌ Процесс сервера завершился сразу после запуска")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Ошибка запуска процесса сервера: {e}")
            return False
    
    async def _wait_for_server_ready(self) -> bool:
        """Ожидание готовности сервера"""
        logger.info("⏳ Ожидание готовности сервера...")
        
        start_time = time.time()
        
        while time.time() - start_time < self.config.startup_timeout:
            try:
                if await self.is_server_running():
                    # Дополнительно проверяем что модель доступна
                    health = await self.health_check()
                    if health['model_available']:
                        logger.info("✅ Сервер готов к работе")
                        return True
                    else:
                        logger.debug("⏳ Сервер запущен, но модель еще не готова")
                
                await asyncio.sleep(self.config.health_check_interval)
                
            except Exception as e:
                logger.debug(f"Ошибка проверки готовности: {e}")
                await asyncio.sleep(2)
        
        logger.error(f"❌ Сервер не готов после {self.config.startup_timeout} секунд")
        return False
    
    async def _wait_process_termination(self):
        """Ожидание завершения процесса"""
        while self.server_process and self.server_process.poll() is None:
            await asyncio.sleep(0.1)
    
    async def _cleanup_failed_start(self):
        """Очистка после неудачного запуска"""
        if self.server_process:
            try:
                self.server_process.kill()
                await asyncio.sleep(1)
            except:
                pass
            self.server_process = None
    
    async def _kill_ollama_processes(self):
        """Принудительная остановка всех процессов ollama"""
        try:
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    if 'ollama' in proc.info['name'].lower():
                        logger.debug(f"Останавливаем процесс ollama: {proc.info['pid']}")
                        proc.terminate()
                        
                        # Ждем завершения
                        try:
                            proc.wait(timeout=5)
                        except psutil.TimeoutExpired:
                            proc.kill()
                            
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
                    
        except Exception as e:
            logger.debug(f"Ошибка при остановке процессов ollama: {e}")
    
    async def __aenter__(self):
        """Контекстный менеджер - вход"""
        await self.start_server()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Контекстный менеджер - выход"""
        await self.stop_server()

# Глобальный экземпляр менеджера
_global_manager: Optional[MistralServerManager] = None

def get_global_manager() -> MistralServerManager:
    """Получить глобальный экземпляр менеджера"""
    global _global_manager
    if _global_manager is None:
        _global_manager = MistralServerManager()
    return _global_manager

async def ensure_mistral_server_running() -> bool:
    """Убедиться что Mistral сервер запущен"""
    manager = get_global_manager()
    return await manager.start_server()

async def stop_mistral_server() -> bool:
    """Остановить Mistral сервер"""
    manager = get_global_manager()
    return await manager.stop_server()