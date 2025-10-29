#!/usr/bin/env python3
"""
System Checker - Проверка системных требований для Peper Binance v4
"""

import platform
import psutil
import os
import sys
from pathlib import Path
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class SystemCheckResult:
    """Результат проверки системных требований"""
    
    def __init__(self):
        self.passed = False
        self.mistral_ai_enabled = False
        self.warnings = []
        self.errors = []
        self.system_info = {}

class SystemChecker:
    """Класс для проверки системных требований"""
    
    def __init__(self):
        self.system_info = {}
        self.min_ram_gb = 4  # Минимум 4GB RAM
        self.recommended_ram_gb = 8  # Рекомендуется 8GB RAM
        self.min_disk_space_gb = 2  # Минимум 2GB свободного места
        self.mistral_ram_requirement_gb = 6  # Для Mistral AI нужно 6GB+
        
    def get_system_info(self) -> Dict[str, Any]:
        """Получение информации о системе"""
        try:
            # Основная информация о системе
            self.system_info = {
                'platform': platform.system(),
                'platform_release': platform.release(),
                'platform_version': platform.version(),
                'architecture': platform.machine(),
                'processor': platform.processor(),
                'python_version': sys.version,
                'cpu_count': psutil.cpu_count(),
                'cpu_count_logical': psutil.cpu_count(logical=True),
                'memory_total_gb': round(psutil.virtual_memory().total / (1024**3), 2),
                'memory_available_gb': round(psutil.virtual_memory().available / (1024**3), 2),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_total_gb': round(psutil.disk_usage('.').total / (1024**3), 2),
                'disk_free_gb': round(psutil.disk_usage('.').free / (1024**3), 2),
                'disk_used_percent': round((psutil.disk_usage('.').used / psutil.disk_usage('.').total) * 100, 2)
            }
            
            # Проверка наличия GPU (опционально)
            try:
                import GPUtil
                gpus = GPUtil.getGPUs()
                if gpus:
                    self.system_info['gpu_count'] = len(gpus)
                    self.system_info['gpu_info'] = [{'name': gpu.name, 'memory_total': gpu.memoryTotal} for gpu in gpus]
                else:
                    self.system_info['gpu_count'] = 0
            except ImportError:
                self.system_info['gpu_count'] = 0
                
        except Exception as e:
            logger.error(f"Ошибка при получении информации о системе: {e}")
            
        return self.system_info
    
    def print_system_info(self):
        """Вывод информации о системе"""
        print("\n" + "="*60)
        print("           ИНФОРМАЦИЯ О СИСТЕМЕ")
        print("="*60)
        
        if not self.system_info:
            self.get_system_info()
            
        print(f"Операционная система: {self.system_info.get('platform', 'Unknown')} {self.system_info.get('platform_release', '')}")
        print(f"Архитектура: {self.system_info.get('architecture', 'Unknown')}")
        print(f"Процессор: {self.system_info.get('processor', 'Unknown')}")
        print(f"Количество ядер: {self.system_info.get('cpu_count', 'Unknown')} физических, {self.system_info.get('cpu_count_logical', 'Unknown')} логических")
        print(f"Оперативная память: {self.system_info.get('memory_total_gb', 0)} GB (доступно: {self.system_info.get('memory_available_gb', 0)} GB)")
        print(f"Использование памяти: {self.system_info.get('memory_percent', 0)}%")
        print(f"Дисковое пространство: {self.system_info.get('disk_total_gb', 0)} GB (свободно: {self.system_info.get('disk_free_gb', 0)} GB)")
        print(f"Использование диска: {self.system_info.get('disk_used_percent', 0)}%")
        
        if self.system_info.get('gpu_count', 0) > 0:
            print(f"GPU: {self.system_info['gpu_count']} устройств")
            for i, gpu in enumerate(self.system_info.get('gpu_info', [])):
                print(f"  GPU {i+1}: {gpu['name']} ({gpu['memory_total']} MB)")
        else:
            print("GPU: Не обнаружено")
            
        print(f"Python версия: {sys.version.split()[0]}")
        print("="*60)
    
    def check_system_requirements(self) -> SystemCheckResult:
        """Проверка системных требований"""
        result = SystemCheckResult()
        result.system_info = self.system_info.copy()
        
        if not self.system_info:
            self.get_system_info()
        
        # Проверка RAM
        total_ram = self.system_info.get('memory_total_gb', 0)
        available_ram = self.system_info.get('memory_available_gb', 0)
        
        if total_ram < self.min_ram_gb:
            result.errors.append(f"Недостаточно оперативной памяти: {total_ram} GB (минимум {self.min_ram_gb} GB)")
        elif total_ram < self.recommended_ram_gb:
            result.warnings.append(f"Рекомендуется больше оперативной памяти: {total_ram} GB (рекомендуется {self.recommended_ram_gb} GB)")
        
        # Проверка свободного места на диске
        free_disk = self.system_info.get('disk_free_gb', 0)
        if free_disk < self.min_disk_space_gb:
            result.errors.append(f"Недостаточно свободного места на диске: {free_disk} GB (минимум {self.min_disk_space_gb} GB)")
        
        # Проверка возможности запуска Mistral AI
        if total_ram >= self.mistral_ram_requirement_gb and available_ram >= 4:
            result.mistral_ai_enabled = True
        else:
            result.mistral_ai_enabled = False
            result.warnings.append(f"Mistral AI будет отключен: недостаточно RAM ({total_ram} GB, требуется {self.mistral_ram_requirement_gb} GB)")
        
        # Проверка Python версии
        python_version = sys.version_info
        if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
            result.errors.append(f"Неподдерживаемая версия Python: {python_version.major}.{python_version.minor} (требуется Python 3.8+)")
        
        # Проверка наличия критических файлов
        critical_files = ['main.py', 'requirements.txt', 'config.py']
        for file in critical_files:
            if not Path(file).exists():
                result.warnings.append(f"Отсутствует критический файл: {file}")
        
        # Проверка наличия директорий
        critical_dirs = ['ai_modules', 'config', 'utils']
        for dir_name in critical_dirs:
            if not Path(dir_name).exists():
                result.warnings.append(f"Отсутствует критическая директория: {dir_name}")
        
        # Определение общего результата
        result.passed = len(result.errors) == 0
        
        return result
    
    def print_check_results(self, result: SystemCheckResult):
        """Вывод результатов проверки"""
        print("\n" + "="*60)
        print("         РЕЗУЛЬТАТЫ ПРОВЕРКИ СИСТЕМЫ")
        print("="*60)
        
        if result.passed:
            print("✅ СИСТЕМА СООТВЕТСТВУЕТ ТРЕБОВАНИЯМ")
        else:
            print("❌ СИСТЕМА НЕ СООТВЕТСТВУЕТ МИНИМАЛЬНЫМ ТРЕБОВАНИЯМ")
        
        if result.mistral_ai_enabled:
            print("✅ Mistral AI: ВКЛЮЧЕН")
        else:
            print("⚠️  Mistral AI: ОТКЛЮЧЕН")
        
        if result.warnings:
            print("\n⚠️  ПРЕДУПРЕЖДЕНИЯ:")
            for warning in result.warnings:
                print(f"   • {warning}")
        
        if result.errors:
            print("\n❌ ОШИБКИ:")
            for error in result.errors:
                print(f"   • {error}")
        
        print("\n" + "="*60)
        
        if not result.passed:
            print("Рекомендации:")
            print("• Увеличьте объем оперативной памяти")
            print("• Освободите место на диске")
            print("• Обновите Python до версии 3.8+")
            print("• Установите недостающие зависимости")
            print("="*60)

def main():
    """Основная функция для прямого запуска проверки"""
    checker = SystemChecker()
    checker.get_system_info()
    checker.print_system_info()
    
    result = checker.check_system_requirements()
    checker.print_check_results(result)
    
    if not result.passed:
        sys.exit(1)

if __name__ == "__main__":
    main()