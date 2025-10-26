#!/usr/bin/env python3
"""
Peper Binance v4 - Легковесный AI софт для трейдинга
Оптимизирован для минимального потребления ресурсов
"""

import asyncio
import logging
import os
import sys
import signal
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional
from system_checker import SystemChecker

# Настройка логирования для минимального потребления ресурсов
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log', mode='a', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

class PeperBinanceApp:
    """Главный класс приложения с оптимизацией ресурсов"""
    
    def __init__(self):
        self.ai_modules = {}
        self.is_running = False
        self.mistral_ai_enabled = True  # По умолчанию включен
        self.system_check_passed = False
        self.mistral_server_process = None  # Процесс Mistral сервера
        
    async def check_system_requirements(self) -> bool:
        """Проверка системных требований перед запуском"""
        logger.info("Проверка системных требований...")
        
        try:
            checker = SystemChecker()
            checker.get_system_info()
            checker.print_system_info()
            
            result = checker.check_system_requirements()
            checker.print_check_results(result)
            
            # Сохраняем результаты проверки
            self.system_check_passed = result.passed
            self.mistral_ai_enabled = result.mistral_ai_enabled
            
            if not result.passed:
                logger.error("Система не соответствует минимальным требованиям!")
                return False
            
            if not result.mistral_ai_enabled:
                logger.warning("Mistral AI будет отключен из-за недостатка ресурсов")
            
            logger.info("Проверка системных требований завершена успешно")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка при проверке системных требований: {e}")
            return False
    
    async def start_mistral_server(self):
        """Запуск Mistral сервера в отдельном терминале"""
        if not self.mistral_ai_enabled:
            logger.info("Mistral AI отключен, сервер не запускается")
            return False
            
        try:
            logger.info("Запуск Mistral сервера в отдельном терминале...")
            
            # Проверяем наличие модели
            model_path = Path("models/mistral-7b-instruct-v0.1.Q4_K_M.gguf")
            if not model_path.exists():
                logger.error(f"Модель Mistral не найдена: {model_path}")
                return False
            
            # Команда для запуска сервера в новом терминале
            # Используем osascript для macOS для открытия нового терминала
            script = f'''
            tell application "Terminal"
                do script "cd '{os.getcwd()}' && python3 -m llama_cpp.server --model {model_path} --host 127.0.0.1 --port 8000 --n_ctx 2048"
                activate
            end tell
            '''
            
            # Запускаем AppleScript для открытия нового терминала
            process = subprocess.Popen([
                'osascript', '-e', script
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Ждем немного для запуска сервера
            await asyncio.sleep(3)
            
            logger.info("Mistral сервер запущен в отдельном терминале на порту 8000")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка запуска Mistral сервера: {e}")
            return False
        
    async def initialize_ai_modules(self):
        """Ленивая инициализация AI модулей для экономии памяти"""
        try:
            # Импортируем модули только при необходимости
            from ai_modules.trading_ai import TradingAI
            from ai_modules.lava_ai import LavaAI
            from ai_modules.lgbm_ai import LGBMAI
            
            logger.info("Инициализация AI модулей...")
            
            # Создаем экземпляры с минимальным потреблением памяти
            self.ai_modules = {
                'trading': TradingAI(),
                'lava': LavaAI(),
                'lgbm': LGBMAI()
            }
            
            # Mistral AI инициализируем только если система позволяет
            if self.mistral_ai_enabled:
                from ai_modules.mistral_ai import MistralAI
                self.ai_modules['mistral'] = MistralAI()
                logger.info("Mistral AI модуль включен")
            else:
                logger.warning("Mistral AI модуль отключен из-за недостатка ресурсов")
            
            # Инициализируем все модули
            for name, module in self.ai_modules.items():
                try:
                    if hasattr(module, 'initialize'):
                        await module.initialize()
                    logger.info(f"AI модуль {name} инициализирован")
                except Exception as e:
                    logger.error(f"Ошибка инициализации модуля {name}: {e}")
            
            logger.info("Все AI модули успешно инициализированы")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка инициализации AI модулей: {e}")
            return False
    
    async def start(self):
        """Запуск приложения"""
        logger.info("Запуск Peper Binance v4...")
        
        # Сначала проверяем системные требования
        if not await self.check_system_requirements():
            logger.error("Запуск невозможен: система не соответствует требованиям")
            return False
        
        # Запускаем Mistral сервер если система прошла проверку
        if self.mistral_ai_enabled:
            await self.start_mistral_server()
        
        if not await self.initialize_ai_modules():
            logger.error("Не удалось инициализировать AI модули")
            return False
            
        self.is_running = True
        logger.info("Приложение успешно запущено")
        
        # Основной цикл приложения
        try:
            while self.is_running:
                await self.main_loop()
                await asyncio.sleep(0.1)  # Минимальная задержка для экономии CPU
                
        except KeyboardInterrupt:
            logger.info("Получен сигнал остановки")
        finally:
            await self.shutdown()
    
    async def main_loop(self):
        """Основной цикл обработки"""
        # Здесь будет логика работы с AI модулями
        pass
    
    async def shutdown(self):
        """Корректное завершение работы"""
        logger.info("Завершение работы приложения...")
        self.is_running = False
        
        # Останавливаем Mistral сервер если он был запущен
        if self.mistral_server_process:
            try:
                logger.info("Остановка Mistral сервера...")
                self.mistral_server_process.terminate()
                await asyncio.sleep(1)
                if self.mistral_server_process.poll() is None:
                    self.mistral_server_process.kill()
                logger.info("Mistral сервер остановлен")
            except Exception as e:
                logger.error(f"Ошибка при остановке Mistral сервера: {e}")
        
        # Освобождаем ресурсы AI модулей
        for name, module in self.ai_modules.items():
            try:
                if hasattr(module, 'cleanup'):
                    await module.cleanup()
                logger.info(f"AI модуль {name} корректно завершен")
            except Exception as e:
                logger.error(f"Ошибка при завершении модуля {name}: {e}")
        
        logger.info("Приложение завершено")

async def main():
    """Точка входа в приложение"""
    app = PeperBinanceApp()
    await app.start()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        logger.error(f"Критическая ошибка: {e}")
        sys.exit(1)