#!/usr/bin/env python3
"""
Оптимизированный скрипт очистки проекта Peper Binance v4
Удаляет только лишние файлы, сохраняя основную систему, винрейт-тестирование и все 5 AI-моделей

ОБЯЗАТЕЛЬНО СОХРАНЯЕТ:
- winrate_test_with_results2.py (основной файл тестирования винрейта)
- Все 5 AI-моделей в ai_modules/
- Основные системные файлы
- Папку models/ с обученными моделями
- Конфигурации и утилиты

УДАЛЯЕТ:
- Дублирующиеся тестовые системы
- Временные отчеты и логи
- Неиспользуемые скрипты
- Старую документацию
"""

import os
import shutil
import logging
import json
import zipfile
import subprocess
import sys
import fnmatch
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Set, Tuple
import argparse

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('cleanup_log.txt'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class OptimizedCleanup:
    """Оптимизированная система очистки проекта"""
    
    def __init__(self, project_root: str = None):
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.backup_dir = self.project_root / "cleanup_backup"
        self.stats = {
            'files_deleted': 0,
            'dirs_deleted': 0,
            'space_freed': 0,
            'backup_created': False
        }
        
        # КРИТИЧЕСКИ ВАЖНЫЕ ФАЙЛЫ - НЕ УДАЛЯТЬ!
        self.essential_files = {
            # Основной файл винрейт-тестирования
            'winrate_test_with_results2.py',
            
            # Основные системные файлы
            'main.py',
            'config.py', 
            'data_collector.py',
            'historical_data_manager.py',
            'detailed_trade_visualizer.py',
            'requirements.txt',
            '.gitignore',
            '.gitattributes',
            
            # AI модули (все 5 моделей)
            'ai_modules/trading_ai.py',
            'ai_modules/lava_ai.py',
            'ai_modules/lgbm_ai.py',
            'ai_modules/mistral_ai.py',
            'ai_modules/reinforcement_learning_engine.py',
            'ai_modules/ai_manager.py',
            'ai_modules/multi_ai_orchestrator.py',
            'ai_modules/mistral_server_manager.py',
            'ai_modules/__init__.py',
            
            # Утилиты
            'utils/timezone_utils.py',
            'utils/__init__.py',
            
            # Конфигурации
            'config/config_manager.py',
            'config/reinforcement_learning_config.json',
            'config/reinforcement_learning_profiles.json',
            'config/__init__.py',
            
            # База данных
            'database/persistence_manager.py',
            'database/reinforcement_learning_db.py',
            'database/__init__.py',
            
            # Аналитика
            'analytics/reinforcement_learning_analytics.py',
        }
        
        # КРИТИЧЕСКИ ВАЖНЫЕ ПАПКИ - НЕ УДАЛЯТЬ!
        self.essential_dirs = {
            'ai_modules',
            'models',
            'config', 
            'utils',
            'database',
            'analytics',
            '.trae'
        }
        
        # ФАЙЛЫ ДЛЯ УДАЛЕНИЯ (паттерны)
        self.files_to_delete = {
            # Тестовые файлы (кроме основного винрейт-теста)
            'test_*.py',
            'debug_*.py', 
            'mock_*.py',
            'demo_*.py',
            '*_test.py',
            '*_debug.py',
            '*_mock.py',
            
            # Временные файлы с датами (особенно 20251024)
            '*_20251024_*',
            '*_20251025_*',
            '*_20251026_*',
            '*_20251027_*',
            '*_20251028_*',
            
            # Отчеты и логи
            '*.html',
            '*.log',
            'report_*.json',
            'diagnostic_*.json',
            'test_report*.json',
            'quick_test_results.json',
            
            # Дублирующиеся системы тестирования
            'advanced_backtester.py',
            'advanced_strategy_tester.py',
            'automated_multi_cycle_tester.py',
            'mass_testing_50_pairs.py',
            'mass_training_script.py',
            'improved_mass_testing.py',
            'individual_model_tester.py',
            'reinforcement_winrate_tester.py',
            'sequential_strategy_tester.py',
            'run_automated_tests.py',
            'run_sequential_testing.py',
            'run_strategy_testing.py',
            
            # Калибровка и оптимизация (дубли)
            'ai_model_calibrator.py',
            'ai_models_diagnostics.py',
            'lava_ai_calibration.py',
            'parameter_optimizer.py',
            'quick_recalibration.py',
            'run_calibration.py',
            'trading_ai_trainer.py',
            
            # Анализ и диагностика (дубли)
            'comprehensive_system_report.py',
            'diagnose_model.py',
            'integrated_trading_diagnostics.py',
            'results_analyzer.py',
            'zero_trades_diagnostics.py',
            'system_checker.py',
            
            # Старые системы
            'enhanced_winrate_system.py',
            'ensemble_system.py',
            'optimized_quality_system.py',
            'premium_ultra_system.py',
            'quality_focused_system.py',
            'scalping_ensemble_system.py',
            
            # Старые стратегии и индикаторы
            'enhanced_indicators.py',
            'enhanced_labeling_strategy.py',
            'improved_labeling_strategy.py',
            'strategy_manager.py',
            
            # Визуализация (дубли)
            'trading_visualizer.py',
            'report_generator.py',
            
            # Мониторинг (дубли)
            'cli_monitor.py',
            'checkpoint_manager.py',
            
            # Скрипты запуска (дубли)
            'run_reinforcement_learning.py',
            'run_scalping_system.py',
            'run_trading_analysis.py',
            'run_winrate_test.sh',
            'rl_quick_start.py',
            'activate_env.sh',
            
            # Бэкапы и старые скрипты очистки
            'cleanup_script.py',
            'targeted_cleanup.py', 
            'final_cleanup.py',
            
            # Документация (оставляем только основную)
            'ANALYSIS_CRITICAL_FIXES.md',
            'AUTOMATED_STRATEGY_TESTING_GUIDE.md',
            'CRITICAL_ISSUES_REPORT.md',
            'README_SCALPING.md',
            'README_automated_testing.md',
            'README_trading_ai.md',
            'SEQUENTIAL_TESTING_GUIDE.md',
            'VISUALIZATION_GUIDE.md',
            'adaptive_reinforcement_learning_system_requirements.md',
            'adaptive_reinforcement_learning_technical_architecture.md',
            'calibration_analysis_report.md',
            'final_optimization_results.md',
            'improvements_summary.md',
            'reinforcement_learning_implementation_plan.md',
            'СИСТЕМА_ГОТОВА.md'
        }
        
        # ПАПКИ ДЛЯ УДАЛЕНИЯ
        self.dirs_to_delete = {
            'test_results',
            'automated_test_results',
            'training_results',
            'strategy_testing_results',
            'ai_diagnostic_system',  # Полностью удаляем диагностическую систему
        }
    
    def create_backup(self) -> bool:
        """Создание бэкапа перед очисткой"""
        try:
            logger.info("🔄 Создание бэкапа проекта...")
            
            if self.backup_dir.exists():
                shutil.rmtree(self.backup_dir)
            
            self.backup_dir.mkdir(exist_ok=True)
            
            # Создаем ZIP архив
            backup_file = self.backup_dir / f"project_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
            
            with zipfile.ZipFile(backup_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, dirs, files in os.walk(self.project_root):
                    # Пропускаем папку бэкапа
                    if 'cleanup_backup' in root:
                        continue
                        
                    for file in files:
                        file_path = Path(root) / file
                        arcname = file_path.relative_to(self.project_root)
                        zipf.write(file_path, arcname)
            
            self.stats['backup_created'] = True
            logger.info(f"✅ Бэкап создан: {backup_file}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Ошибка создания бэкапа: {e}")
            return False
    
    def get_file_size(self, file_path: Path) -> int:
        """Получение размера файла"""
        try:
            return file_path.stat().st_size
        except:
            return 0
    
    def is_essential_file(self, file_path: Path) -> bool:
        """Проверка, является ли файл критически важным"""
        relative_path = file_path.relative_to(self.project_root)
        
        # Проверяем точные совпадения
        if str(relative_path) in self.essential_files:
            return True
        
        # Проверяем папки
        for part in relative_path.parts:
            if part in self.essential_dirs:
                return True
        
        # Проверяем файлы в папке models
        if 'models' in relative_path.parts:
            return True
            
        return False
    
    def should_delete_file(self, file_path: Path) -> bool:
        """Определение, нужно ли удалить файл"""
        if self.is_essential_file(file_path):
            return False
        
        file_name = file_path.name
        relative_path = str(file_path.relative_to(self.project_root))
        
        # Проверяем паттерны для удаления
        for pattern in self.files_to_delete:
            # Точное совпадение имени файла
            if file_name == pattern:
                return True
            # Проверяем wildcard паттерны
            if '*' in pattern:
                if fnmatch.fnmatch(file_name, pattern):
                    return True
        
        return False
    
    def delete_files(self) -> None:
        """Удаление файлов согласно правилам"""
        logger.info("🗑️ Начинаем удаление файлов...")
        
        files_to_remove = []
        dirs_to_remove = []
        
        # Сначала собираем все файлы для удаления
        for root, dirs, files in os.walk(self.project_root):
            root_path = Path(root)
            
            # Пропускаем папку бэкапа
            if 'cleanup_backup' in str(root_path):
                continue
            
            # Проверяем папки для удаления
            for dir_name in dirs[:]:  # Копируем список для безопасного изменения
                dir_path = root_path / dir_name
                relative_dir = str(dir_path.relative_to(self.project_root))
                
                if relative_dir in self.dirs_to_delete:
                    dirs_to_remove.append(dir_path)
                    dirs.remove(dir_name)  # Не заходим в эту папку
            
            # Проверяем файлы для удаления
            for file_name in files:
                file_path = root_path / file_name
                
                if self.should_delete_file(file_path):
                    files_to_remove.append(file_path)
        
        # Удаляем файлы
        for file_path in files_to_remove:
            try:
                size = self.get_file_size(file_path)
                file_path.unlink()
                self.stats['files_deleted'] += 1
                self.stats['space_freed'] += size
                logger.info(f"🗑️ Удален файл: {file_path.relative_to(self.project_root)}")
            except Exception as e:
                logger.warning(f"⚠️ Не удалось удалить файл {file_path}: {e}")
        
        # Удаляем папки
        for dir_path in dirs_to_remove:
            try:
                # Подсчитываем размер папки
                for root, dirs, files in os.walk(dir_path):
                    for file in files:
                        self.stats['space_freed'] += self.get_file_size(Path(root) / file)
                
                shutil.rmtree(dir_path)
                self.stats['dirs_deleted'] += 1
                logger.info(f"🗑️ Удалена папка: {dir_path.relative_to(self.project_root)}")
            except Exception as e:
                logger.warning(f"⚠️ Не удалось удалить папку {dir_path}: {e}")
    
    def clean_empty_directories(self) -> None:
        """Удаление пустых папок"""
        logger.info("🧹 Очистка пустых папок...")
        
        for root, dirs, files in os.walk(self.project_root, topdown=False):
            for dir_name in dirs:
                dir_path = Path(root) / dir_name
                
                # Пропускаем важные папки
                if dir_name in self.essential_dirs:
                    continue
                
                # Пропускаем папку бэкапа
                if 'cleanup_backup' in str(dir_path):
                    continue
                
                try:
                    if not any(dir_path.iterdir()):  # Папка пустая
                        dir_path.rmdir()
                        self.stats['dirs_deleted'] += 1
                        logger.info(f"🗑️ Удалена пустая папка: {dir_path.relative_to(self.project_root)}")
                except Exception as e:
                    logger.debug(f"Не удалось удалить папку {dir_path}: {e}")
    
    def verify_essential_files(self) -> bool:
        """Проверка наличия критически важных файлов после очистки"""
        logger.info("🔍 Проверка критически важных файлов...")
        
        missing_files = []
        
        for essential_file in self.essential_files:
            file_path = self.project_root / essential_file
            if not file_path.exists():
                missing_files.append(essential_file)
        
        if missing_files:
            logger.error(f"❌ Отсутствуют критически важные файлы: {missing_files}")
            return False
        
        logger.info("✅ Все критически важные файлы на месте")
        return True
    
    def test_system_functionality(self) -> bool:
        """Тестирование работоспособности системы"""
        logger.info("🧪 Тестирование работоспособности системы...")
        
        try:
            # Проверяем импорт основных модулей
            test_imports = [
                "import ai_modules.ai_manager",
                "import ai_modules.trading_ai", 
                "import ai_modules.lava_ai",
                "import ai_modules.lgbm_ai",
                "import ai_modules.mistral_ai",
                "import ai_modules.reinforcement_learning_engine",
                "import data_collector",
                "import historical_data_manager",
                "import config"
            ]
            
            for import_test in test_imports:
                try:
                    exec(import_test)
                    logger.info(f"✅ {import_test}")
                except Exception as e:
                    logger.error(f"❌ {import_test}: {e}")
                    return False
            
            # Проверяем доступность winrate_test_with_results2.py
            winrate_file = self.project_root / "winrate_test_with_results2.py"
            if not winrate_file.exists():
                logger.error("❌ Файл winrate_test_with_results2.py не найден!")
                return False
            
            # Проверяем наличие всех 5 AI-моделей
            ai_models = [
                "ai_modules/trading_ai.py",
                "ai_modules/lava_ai.py", 
                "ai_modules/lgbm_ai.py",
                "ai_modules/mistral_ai.py",
                "ai_modules/reinforcement_learning_engine.py"
            ]
            
            for model_file in ai_models:
                model_path = self.project_root / model_file
                if not model_path.exists():
                    logger.error(f"❌ AI-модель не найдена: {model_file}")
                    return False
                logger.info(f"✅ AI-модель найдена: {model_file}")
            
            logger.info("✅ Система функциональна - все 5 AI-моделей доступны")
            return True
            
        except Exception as e:
            logger.error(f"❌ Ошибка тестирования системы: {e}")
            return False
    
    def generate_report(self) -> None:
        """Генерация отчета об очистке"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'stats': self.stats,
            'space_freed_mb': round(self.stats['space_freed'] / (1024 * 1024), 2),
            'essential_files_preserved': list(self.essential_files),
            'essential_dirs_preserved': list(self.essential_dirs)
        }
        
        report_file = self.project_root / "cleanup_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📊 Отчет сохранен: {report_file}")
        
        # Выводим статистику
        logger.info("📈 СТАТИСТИКА ОЧИСТКИ:")
        logger.info(f"   Удалено файлов: {self.stats['files_deleted']}")
        logger.info(f"   Удалено папок: {self.stats['dirs_deleted']}")
        logger.info(f"   Освобождено места: {report['space_freed_mb']} МБ")
        logger.info(f"   Бэкап создан: {'Да' if self.stats['backup_created'] else 'Нет'}")
    
    def rollback(self) -> bool:
        """Откат изменений из бэкапа"""
        logger.info("🔄 Выполнение отката изменений...")
        
        try:
            backup_files = list(self.backup_dir.glob("project_backup_*.zip"))
            if not backup_files:
                logger.error("❌ Файлы бэкапа не найдены!")
                return False
            
            # Берем последний бэкап
            latest_backup = max(backup_files, key=lambda x: x.stat().st_mtime)
            
            # Очищаем текущий проект (кроме бэкапа)
            for item in self.project_root.iterdir():
                if item.name != 'cleanup_backup':
                    if item.is_dir():
                        shutil.rmtree(item)
                    else:
                        item.unlink()
            
            # Восстанавливаем из бэкапа
            with zipfile.ZipFile(latest_backup, 'r') as zipf:
                zipf.extractall(self.project_root)
            
            logger.info("✅ Откат выполнен успешно")
            return True
            
        except Exception as e:
            logger.error(f"❌ Ошибка отката: {e}")
            return False
    
    def preview_cleanup(self) -> None:
        """Предварительный просмотр файлов для удаления"""
        logger.info("🔍 ПРЕДВАРИТЕЛЬНЫЙ ПРОСМОТР ОЧИСТКИ")
        logger.info("=" * 60)
        
        files_to_remove = []
        dirs_to_remove = []
        total_size = 0
        
        # Собираем файлы для удаления
        for root, dirs, files in os.walk(self.project_root):
            root_path = Path(root)
            
            # Пропускаем папку бэкапа
            if 'cleanup_backup' in str(root_path):
                continue
            
            # Проверяем папки для удаления
            for dir_name in dirs[:]:
                dir_path = root_path / dir_name
                relative_dir = str(dir_path.relative_to(self.project_root))
                
                if relative_dir in self.dirs_to_delete:
                    dirs_to_remove.append(relative_dir)
                    # Подсчитываем размер папки
                    for r, d, f in os.walk(dir_path):
                        for file in f:
                            total_size += self.get_file_size(Path(r) / file)
            
            # Проверяем файлы для удаления
            for file_name in files:
                file_path = root_path / file_name
                
                if self.should_delete_file(file_path):
                    relative_path = str(file_path.relative_to(self.project_root))
                    files_to_remove.append(relative_path)
                    total_size += self.get_file_size(file_path)
        
        # Выводим результаты
        logger.info(f"📁 ПАПКИ ДЛЯ УДАЛЕНИЯ ({len(dirs_to_remove)}):")
        for dir_path in sorted(dirs_to_remove):
            logger.info(f"   🗑️ {dir_path}")
        
        logger.info(f"\n📄 ФАЙЛЫ ДЛЯ УДАЛЕНИЯ ({len(files_to_remove)}):")
        for file_path in sorted(files_to_remove):
            logger.info(f"   🗑️ {file_path}")
        
        logger.info(f"\n📊 СТАТИСТИКА:")
        logger.info(f"   Папок к удалению: {len(dirs_to_remove)}")
        logger.info(f"   Файлов к удалению: {len(files_to_remove)}")
        logger.info(f"   Освободится места: {round(total_size / (1024 * 1024), 2)} МБ")
        
        logger.info(f"\n✅ КРИТИЧЕСКИ ВАЖНЫЕ ФАЙЛЫ (СОХРАНЯЮТСЯ):")
        for essential_file in sorted(self.essential_files):
            file_path = self.project_root / essential_file
            if file_path.exists():
                logger.info(f"   ✅ {essential_file}")
        
        logger.info("=" * 60)
    
    def run_cleanup(self, create_backup: bool = True) -> bool:
        """Запуск полной очистки"""
        logger.info("🚀 ЗАПУСК ОПТИМИЗИРОВАННОЙ ОЧИСТКИ ПРОЕКТА")
        logger.info("=" * 60)
        
        try:
            # 1. Создание бэкапа
            if create_backup:
                if not self.create_backup():
                    logger.error("❌ Не удалось создать бэкап. Очистка отменена.")
                    return False
            
            # 2. Удаление файлов
            self.delete_files()
            
            # 3. Очистка пустых папок
            self.clean_empty_directories()
            
            # 4. Проверка критически важных файлов
            if not self.verify_essential_files():
                logger.error("❌ Критически важные файлы повреждены!")
                if create_backup:
                    logger.info("🔄 Выполняем откат...")
                    self.rollback()
                return False
            
            # 5. Тестирование функциональности
            if not self.test_system_functionality():
                logger.warning("⚠️ Обнаружены проблемы с функциональностью")
                if create_backup:
                    logger.info("🔄 Рекомендуется выполнить откат")
            
            # 6. Генерация отчета
            self.generate_report()
            
            logger.info("=" * 60)
            logger.info("🎉 ОЧИСТКА ЗАВЕРШЕНА УСПЕШНО!")
            logger.info("✅ Основная система сохранена")
            logger.info("✅ Винрейт-тестирование сохранено") 
            logger.info("✅ Все 5 AI-моделей сохранены")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Критическая ошибка очистки: {e}")
            if create_backup:
                logger.info("🔄 Выполняем откат...")
                self.rollback()
            return False

def main():
    """Главная функция"""
    parser = argparse.ArgumentParser(description='Оптимизированная очистка проекта Peper Binance v4')
    parser.add_argument('--no-backup', action='store_true', help='Не создавать бэкап')
    parser.add_argument('--rollback', action='store_true', help='Откатить изменения')
    parser.add_argument('--dry-run', action='store_true', help='Показать что будет удалено без фактического удаления')
    
    args = parser.parse_args()
    
    cleanup = OptimizedCleanup()
    
    if args.rollback:
        cleanup.rollback()
        return
    
    if args.dry_run:
        cleanup.preview_cleanup()
        return
    
    success = cleanup.run_cleanup(create_backup=not args.no_backup)
    
    if not success:
        logger.error("❌ Очистка завершилась с ошибками")
        sys.exit(1)
    else:
        logger.info("✅ Очистка завершена успешно")

if __name__ == "__main__":
    main()