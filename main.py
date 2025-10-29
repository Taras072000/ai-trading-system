#!/usr/bin/env python3
"""
Peper Binance v4 - Главный файл запуска
Поддерживает три режима работы:
1. Винрейт тестирование
2. Проверка системы  
3. Очистка проекта

Использование:
    python main.py --winrate          # Запуск винрейт тестирования
    python main.py --check-system     # Проверка системных требований
    python main.py --cleanup          # Очистка проекта
    python main.py --interactive      # Интерактивное меню
    python main.py --help             # Справка
"""

import sys
import os
import subprocess
import argparse
import logging
from pathlib import Path
from typing import Optional

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('main.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PeperBinanceManager:
    """Главный менеджер для управления режимами работы Peper Binance v4"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.scripts = {
            'winrate': self.project_root / 'winrate_test_with_results2.py',
            'system_check': self.project_root / 'system_checker.py',
            'cleanup': self.project_root / 'optimized_cleanup.py'
        }
        
    def validate_scripts(self) -> bool:
        """Проверка существования всех необходимых скриптов"""
        missing_scripts = []
        
        for name, script_path in self.scripts.items():
            if not script_path.exists():
                missing_scripts.append(f"{name}: {script_path}")
        
        if missing_scripts:
            logger.error("❌ Отсутствуют необходимые скрипты:")
            for script in missing_scripts:
                logger.error(f"   • {script}")
            return False
        
        logger.info("✅ Все необходимые скрипты найдены")
        return True
    
    def run_script(self, script_path: Path, script_name: str, args: list = None) -> bool:
        """Запуск скрипта с обработкой ошибок"""
        try:
            logger.info(f"🚀 Запуск {script_name}...")
            logger.info(f"📁 Скрипт: {script_path}")
            
            # Формируем команду
            cmd = [sys.executable, str(script_path)]
            if args:
                cmd.extend(args)
            
            # Запускаем скрипт
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=False,  # Показываем вывод в реальном времени
                text=True
            )
            
            if result.returncode == 0:
                logger.info(f"✅ {script_name} завершен успешно")
                return True
            else:
                logger.error(f"❌ {script_name} завершен с ошибкой (код: {result.returncode})")
                return False
                
        except FileNotFoundError:
            logger.error(f"❌ Не удалось найти Python интерпретатор или скрипт: {script_path}")
            return False
        except Exception as e:
            logger.error(f"❌ Ошибка при запуске {script_name}: {e}")
            return False
    
    def run_winrate_test(self, args: list = None) -> bool:
        """Запуск винрейт тестирования"""
        logger.info("🎯 Режим: Винрейт тестирование")
        return self.run_script(
            self.scripts['winrate'], 
            "Винрейт тестирование", 
            args
        )
    
    def run_system_check(self) -> bool:
        """Запуск проверки системы"""
        logger.info("🔍 Режим: Проверка системы")
        return self.run_script(
            self.scripts['system_check'], 
            "Проверка системы"
        )
    
    def run_cleanup(self, args: list = None) -> bool:
        """Запуск очистки проекта"""
        logger.info("🧹 Режим: Очистка проекта")
        
        # Предупреждение пользователя
        print("\n" + "="*60)
        print("⚠️  ВНИМАНИЕ: ОЧИСТКА ПРОЕКТА")
        print("="*60)
        print("Этот режим удалит ненужные файлы из проекта.")
        print("Перед удалением будет создан бэкап.")
        print("="*60)
        
        response = input("Продолжить? (y/N): ").strip().lower()
        if response not in ['y', 'yes', 'да']:
            logger.info("❌ Очистка отменена пользователем")
            return False
        
        return self.run_script(
            self.scripts['cleanup'], 
            "Очистка проекта", 
            args
        )
    
    def show_interactive_menu(self) -> None:
        """Интерактивное меню выбора режима"""
        while True:
            print("\n" + "="*60)
            print("           PEPER BINANCE V4 - ГЛАВНОЕ МЕНЮ")
            print("="*60)
            print("Выберите режим работы:")
            print()
            print("1. 🎯 Винрейт тестирование")
            print("2. 🔍 Проверка системы")
            print("3. 🧹 Очистка проекта")
            print("4. ❓ Справка")
            print("0. 🚪 Выход")
            print("="*60)
            
            try:
                choice = input("Ваш выбор (0-4): ").strip()
                
                if choice == '0':
                    logger.info("👋 Выход из программы")
                    break
                elif choice == '1':
                    self.run_winrate_test()
                elif choice == '2':
                    self.run_system_check()
                elif choice == '3':
                    self.run_cleanup()
                elif choice == '4':
                    self.show_help()
                else:
                    print("❌ Неверный выбор. Попробуйте снова.")
                    
            except KeyboardInterrupt:
                print("\n👋 Выход из программы")
                break
            except Exception as e:
                logger.error(f"❌ Ошибка в интерактивном меню: {e}")
    
    def show_help(self) -> None:
        """Показать справку"""
        print("\n" + "="*60)
        print("                    СПРАВКА")
        print("="*60)
        print("Peper Binance v4 - AI торговая система")
        print()
        print("Режимы работы:")
        print()
        print("🎯 ВИНРЕЙТ ТЕСТИРОВАНИЕ")
        print("   Запускает полное тестирование AI моделей")
        print("   на исторических данных для оценки винрейта")
        print()
        print("🔍 ПРОВЕРКА СИСТЕМЫ")
        print("   Проверяет системные требования:")
        print("   • Объем оперативной памяти")
        print("   • Свободное место на диске")
        print("   • Версия Python")
        print("   • Наличие необходимых файлов")
        print()
        print("🧹 ОЧИСТКА ПРОЕКТА")
        print("   Удаляет ненужные файлы:")
        print("   • Временные файлы и логи")
        print("   • Дублирующиеся тестовые системы")
        print("   • Старые отчеты")
        print("   • Создает бэкап перед удалением")
        print()
        print("Использование из командной строки:")
        print("   python main.py --winrate")
        print("   python main.py --check-system")
        print("   python main.py --cleanup")
        print("   python main.py --interactive")
        print("="*60)

def create_argument_parser() -> argparse.ArgumentParser:
    """Создание парсера аргументов командной строки"""
    parser = argparse.ArgumentParser(
        description='Peper Binance v4 - AI торговая система',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python main.py --winrate              # Запуск винрейт тестирования
  python main.py --check-system         # Проверка системы
  python main.py --cleanup              # Очистка проекта
  python main.py --interactive          # Интерактивное меню
  
Для винрейт тестирования можно передать дополнительные аргументы:
  python main.py --winrate --symbol BTCUSDT --timeframe 1h
        """
    )
    
    # Основные режимы (взаимоисключающие)
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        '--winrate', 
        action='store_true',
        help='Запуск винрейт тестирования'
    )
    mode_group.add_argument(
        '--check-system', 
        action='store_true',
        help='Проверка системных требований'
    )
    mode_group.add_argument(
        '--cleanup', 
        action='store_true',
        help='Очистка проекта от ненужных файлов'
    )
    mode_group.add_argument(
        '--interactive', 
        action='store_true',
        help='Интерактивное меню выбора режима'
    )
    
    # Дополнительные аргументы для винрейт тестирования
    parser.add_argument(
        '--symbol',
        type=str,
        help='Торговая пара для тестирования (например: BTCUSDT)'
    )
    parser.add_argument(
        '--timeframe',
        type=str,
        help='Таймфрейм для тестирования (например: 1h, 4h, 1d)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Режим предварительного просмотра (для cleanup)'
    )
    
    return parser

def main():
    """Главная функция"""
    try:
        # Создаем менеджер
        manager = PeperBinanceManager()
        
        # Проверяем наличие скриптов
        if not manager.validate_scripts():
            logger.error("❌ Не удалось найти необходимые скрипты")
            sys.exit(1)
        
        # Парсим аргументы
        parser = create_argument_parser()
        args = parser.parse_args()
        
        # Определяем режим работы
        if args.winrate:
            # Формируем дополнительные аргументы для винрейт теста
            winrate_args = []
            if args.symbol:
                winrate_args.extend(['--symbol', args.symbol])
            if args.timeframe:
                winrate_args.extend(['--timeframe', args.timeframe])
            
            success = manager.run_winrate_test(winrate_args)
            
        elif args.check_system:
            success = manager.run_system_check()
            
        elif args.cleanup:
            cleanup_args = []
            if args.dry_run:
                cleanup_args.append('--dry-run')
            
            success = manager.run_cleanup(cleanup_args)
            
        elif args.interactive:
            manager.show_interactive_menu()
            success = True
            
        else:
            # Если аргументы не указаны, показываем интерактивное меню
            logger.info("🎮 Запуск в интерактивном режиме")
            manager.show_interactive_menu()
            success = True
        
        # Завершение
        if 'success' in locals():
            sys.exit(0 if success else 1)
        else:
            sys.exit(0)
            
    except KeyboardInterrupt:
        logger.info("\n👋 Программа прервана пользователем")
        sys.exit(0)
    except Exception as e:
        logger.error(f"❌ Критическая ошибка: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()