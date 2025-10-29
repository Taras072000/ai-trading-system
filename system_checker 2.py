"""
Модуль проверки системных требований для Peper Binance v4
Проверяет соответствие системы минимальным и рекомендуемым требованиям
"""

import os
import sys
import platform
import psutil
import shutil
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from config_params import SYSTEM_REQUIREMENTS, CONFIG_PARAMS, MESSAGES

logger = logging.getLogger(__name__)

@dataclass
class SystemInfo:
    """Информация о текущей системе"""
    os_name: str
    os_version: str
    cpu_cores: int
    cpu_freq: float
    total_ram_gb: float
    available_ram_gb: float
    disk_space_gb: float
    python_version: tuple
    gpu_info: Optional[Dict] = None

@dataclass
class CheckResult:
    """Результат проверки системы"""
    passed: bool
    warnings: List[str]
    errors: List[str]
    recommendations: List[str]
    mistral_ai_enabled: bool
    performance_level: str  # 'optimal', 'good', 'minimal', 'insufficient'

class SystemChecker:
    """Класс для проверки системных требований"""
    
    def __init__(self):
        self.requirements = SYSTEM_REQUIREMENTS
        self.config = CONFIG_PARAMS
        self.system_info = None
        
    def get_system_info(self) -> SystemInfo:
        """Получает информацию о текущей системе"""
        try:
            # Основная информация о системе
            os_name = platform.system()
            os_version = platform.release()
            cpu_cores = psutil.cpu_count(logical=False) or psutil.cpu_count()
            
            # Частота процессора
            try:
                cpu_freq = psutil.cpu_freq().max / 1000 if psutil.cpu_freq() else 0.0
            except:
                cpu_freq = 0.0
            
            # Информация о памяти
            memory = psutil.virtual_memory()
            total_ram_gb = memory.total / (1024**3)
            available_ram_gb = memory.available / (1024**3)
            
            # Свободное место на диске
            disk_usage = shutil.disk_usage(Path.cwd())
            disk_space_gb = disk_usage.free / (1024**3)
            
            # Версия Python
            python_version = sys.version_info[:2]
            
            # Информация о GPU (опционально)
            gpu_info = self._get_gpu_info()
            
            self.system_info = SystemInfo(
                os_name=os_name,
                os_version=os_version,
                cpu_cores=cpu_cores,
                cpu_freq=cpu_freq,
                total_ram_gb=total_ram_gb,
                available_ram_gb=available_ram_gb,
                disk_space_gb=disk_space_gb,
                python_version=python_version,
                gpu_info=gpu_info
            )
            
            return self.system_info
            
        except Exception as e:
            logger.error(f"Ошибка получения информации о системе: {e}")
            raise
    
    def _get_gpu_info(self) -> Optional[Dict]:
        """Получает информацию о GPU (если доступно)"""
        try:
            # Попытка получить информацию о NVIDIA GPU
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]  # Берем первую GPU
                return {
                    'name': gpu.name,
                    'memory_total': gpu.memoryTotal / 1024,  # В ГБ
                    'memory_free': gpu.memoryFree / 1024,
                    'driver_version': gpu.driver
                }
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"Не удалось получить информацию о GPU: {e}")
        
        return None
    
    def check_system_requirements(self) -> CheckResult:
        """Проверяет соответствие системы требованиям"""
        if not self.system_info:
            self.get_system_info()
        
        warnings = []
        errors = []
        recommendations = []
        mistral_ai_enabled = True
        
        # Проверка операционной системы
        if self.system_info.os_name not in self.requirements.supported_os:
            errors.append(f"Неподдерживаемая ОС: {self.system_info.os_name}")
        
        # Проверка версии Python
        if self.system_info.python_version < self.requirements.min_python_version:
            errors.append(f"Требуется Python {self.requirements.min_python_version[0]}.{self.requirements.min_python_version[1]}+, "
                         f"текущая версия: {self.system_info.python_version[0]}.{self.system_info.python_version[1]}")
        
        # Проверка ОЗУ
        if self.system_info.total_ram_gb < self.requirements.min_ram_gb:
            errors.append(f"Недостаточно ОЗУ: {self.system_info.total_ram_gb:.1f} ГБ "
                         f"(минимум: {self.requirements.min_ram_gb} ГБ)")
        elif self.system_info.total_ram_gb < self.requirements.recommended_ram_gb:
            warnings.append(f"ОЗУ ниже рекомендуемого: {self.system_info.total_ram_gb:.1f} ГБ "
                           f"(рекомендуемо: {self.requirements.recommended_ram_gb} ГБ)")
        
        # Проверка процессора
        if self.system_info.cpu_cores < self.requirements.min_cpu_cores:
            errors.append(f"Недостаточно ядер процессора: {self.system_info.cpu_cores} "
                         f"(минимум: {self.requirements.min_cpu_cores})")
        elif self.system_info.cpu_cores < self.requirements.recommended_cpu_cores:
            warnings.append(f"Количество ядер ниже рекомендуемого: {self.system_info.cpu_cores} "
                           f"(рекомендуемо: {self.requirements.recommended_cpu_cores})")
        
        # Проверка свободного места
        if self.system_info.disk_space_gb < self.requirements.min_disk_space_gb:
            errors.append(f"Недостаточно свободного места: {self.system_info.disk_space_gb:.1f} ГБ "
                         f"(минимум: {self.requirements.min_disk_space_gb} ГБ)")
        elif self.system_info.disk_space_gb < self.requirements.recommended_disk_space_gb:
            warnings.append(f"Свободное место ниже рекомендуемого: {self.system_info.disk_space_gb:.1f} ГБ "
                           f"(рекомендуемо: {self.requirements.recommended_disk_space_gb} ГБ)")
        
        # Проверка требований для AI модулей (особенно Mistral)
        if self.system_info.total_ram_gb < self.requirements.ai_min_ram_gb:
            mistral_ai_enabled = False
            warnings.append(f"Mistral AI будет отключен: недостаточно ОЗУ для AI модулей "
                           f"({self.system_info.total_ram_gb:.1f} ГБ < {self.requirements.ai_min_ram_gb} ГБ)")
            recommendations.append("Для включения Mistral AI увеличьте ОЗУ до 8+ ГБ")
        elif self.system_info.total_ram_gb < self.requirements.ai_recommended_ram_gb:
            warnings.append(f"ОЗУ для AI модулей ниже рекомендуемого: {self.system_info.total_ram_gb:.1f} ГБ "
                           f"(рекомендуемо: {self.requirements.ai_recommended_ram_gb} ГБ)")
            recommendations.append("Для оптимальной работы AI модулей рекомендуется 16+ ГБ ОЗУ")
        
        # Определение уровня производительности
        performance_level = self._determine_performance_level(errors, warnings, mistral_ai_enabled)
        
        # Дополнительные рекомендации
        if self.system_info.gpu_info is None and self.requirements.gpu_memory_gb:
            recommendations.append("Для ускорения AI модулей рекомендуется GPU с поддержкой CUDA")
        
        # Проверка успешности
        passed = len(errors) == 0
        
        return CheckResult(
            passed=passed,
            warnings=warnings,
            errors=errors,
            recommendations=recommendations,
            mistral_ai_enabled=mistral_ai_enabled,
            performance_level=performance_level
        )
    
    def _determine_performance_level(self, errors: List[str], warnings: List[str], mistral_enabled: bool) -> str:
        """Определяет уровень производительности системы"""
        if errors:
            return 'insufficient'
        elif not mistral_enabled or len(warnings) > 2:
            return 'minimal'
        elif len(warnings) > 0:
            return 'good'
        else:
            return 'optimal'
    
    def print_system_info(self):
        """Выводит информацию о системе"""
        if not self.system_info:
            self.get_system_info()
        
        info = self.system_info
        print("\n" + "="*50)
        print("ИНФОРМАЦИЯ О СИСТЕМЕ")
        print("="*50)
        print(f"ОС: {info.os_name} {info.os_version}")
        print(f"Python: {info.python_version[0]}.{info.python_version[1]}")
        print(f"Процессор: {info.cpu_cores} ядер @ {info.cpu_freq:.1f} ГГц")
        print(f"ОЗУ: {info.total_ram_gb:.1f} ГБ (доступно: {info.available_ram_gb:.1f} ГБ)")
        print(f"Свободное место: {info.disk_space_gb:.1f} ГБ")
        
        if info.gpu_info:
            gpu = info.gpu_info
            print(f"GPU: {gpu['name']} ({gpu['memory_total']:.1f} ГБ)")
        else:
            print("GPU: Не обнаружен")
        print("="*50)
    
    def print_check_results(self, result: CheckResult):
        """Выводит результаты проверки"""
        print("\n" + "="*50)
        print("РЕЗУЛЬТАТЫ ПРОВЕРКИ СИСТЕМЫ")
        print("="*50)
        
        # Статус проверки
        if result.passed:
            if result.performance_level == 'optimal':
                print(MESSAGES['system_check_passed'])
            else:
                print(MESSAGES['system_check_warning'])
        else:
            print(MESSAGES['system_check_failed'])
        
        # Ошибки
        if result.errors:
            print("\n❌ КРИТИЧЕСКИЕ ОШИБКИ:")
            for error in result.errors:
                print(f"  • {error}")
        
        # Предупреждения
        if result.warnings:
            print("\n⚠️  ПРЕДУПРЕЖДЕНИЯ:")
            for warning in result.warnings:
                print(f"  • {warning}")
        
        # Статус Mistral AI
        if not result.mistral_ai_enabled:
            print(f"\n{MESSAGES['mistral_disabled']}")
            print(f"{MESSAGES['performance_warning']}")
        
        # Рекомендации
        if result.recommendations:
            print("\n💡 РЕКОМЕНДАЦИИ:")
            for rec in result.recommendations:
                print(f"  • {rec}")
        
        print(f"\nУровень производительности: {result.performance_level.upper()}")
        print("="*50)

def main():
    """Основная функция для тестирования"""
    checker = SystemChecker()
    
    print("Проверка системных требований Peper Binance v4...")
    
    # Получение информации о системе
    checker.get_system_info()
    checker.print_system_info()
    
    # Проверка требований
    result = checker.check_system_requirements()
    checker.print_check_results(result)
    
    return result.passed, result.mistral_ai_enabled

if __name__ == "__main__":
    main()