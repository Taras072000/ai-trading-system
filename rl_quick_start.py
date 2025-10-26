#!/usr/bin/env python3
"""
Quick Start Script for Reinforcement Learning System
Скрипт быстрого запуска системы обучения с подкреплением

Простой интерфейс для запуска и мониторинга системы обучения
"""

import asyncio
import os
import sys
import json
from datetime import datetime
from typing import Dict, Any

# Добавляем путь к проекту
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from run_reinforcement_learning import ReinforcementLearningRunner

class QuickStartInterface:
    """Интерфейс быстрого запуска"""
    
    def __init__(self):
        self.runner = None
    
    def print_banner(self):
        """Вывод баннера"""
        print("\n" + "="*80)
        print("🚀 PEPER BINANCE V4 - REINFORCEMENT LEARNING SYSTEM")
        print("   Система адаптивного обучения с подкреплением")
        print("="*80)
        print("📅 Время запуска:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        print("="*80 + "\n")
    
    def print_menu(self):
        """Вывод главного меню"""
        print("📋 ГЛАВНОЕ МЕНЮ:")
        print("1. 🎯 Запустить одиночный цикл обучения")
        print("2. 🔄 Запустить непрерывное обучение")
        print("3. 📊 Показать статус системы")
        print("4. 📈 Показать статистику производительности")
        print("5. 🗂️ Показать последние сессии")
        print("6. ⚙️ Настройки профилей")
        print("7. 🧹 Очистка и обслуживание")
        print("0. 🚪 Выход")
        print("-" * 50)
    
    def print_profiles_menu(self):
        """Меню выбора профилей"""
        print("\n📋 ДОСТУПНЫЕ ПРОФИЛИ ОБУЧЕНИЯ:")
        print("1. 🐌 Conservative - Консервативное обучение (медленное, стабильное)")
        print("2. ⚖️ Balanced - Сбалансированное обучение (рекомендуется)")
        print("3. 🚀 Aggressive - Агрессивное обучение (быстрое, рискованное)")
        print("4. 🧪 Experimental - Экспериментальное обучение")
        print("5. ⚡ Quick Learning - Быстрое обучение")
        print("0. ⬅️ Назад")
        print("-" * 50)
    
    async def initialize_system(self):
        """Инициализация системы"""
        try:
            print("🔄 Инициализация системы...")
            self.runner = ReinforcementLearningRunner()
            
            success = await self.runner.initialize_components()
            if success:
                print("✅ Система успешно инициализирована")
                return True
            else:
                print("❌ Ошибка инициализации системы")
                return False
                
        except Exception as e:
            print(f"❌ Критическая ошибка инициализации: {e}")
            return False
    
    async def run_single_cycle(self):
        """Запуск одиночного цикла"""
        try:
            self.print_profiles_menu()
            choice = input("Выберите профиль (1-5): ").strip()
            
            profile_map = {
                '1': 'conservative',
                '2': 'balanced', 
                '3': 'aggressive',
                '4': 'experimental',
                '5': 'quick_learning'
            }
            
            if choice not in profile_map:
                print("❌ Неверный выбор профиля")
                return
            
            profile = profile_map[choice]
            session_name = input(f"Название сессии (Enter для автогенерации): ").strip()
            
            if not session_name:
                session_name = None
            
            print(f"\n🎯 Запуск цикла обучения с профилем: {profile}")
            print("⏳ Это может занять несколько минут...")
            
            success = await self.runner.run_learning_cycle(profile, session_name)
            
            if success:
                print("✅ Цикл обучения завершен успешно!")
            else:
                print("❌ Цикл обучения завершился с ошибкой")
                
        except Exception as e:
            print(f"❌ Ошибка запуска цикла: {e}")
    
    async def run_continuous_learning(self):
        """Запуск непрерывного обучения"""
        try:
            print("\n🔄 НАСТРОЙКА НЕПРЕРЫВНОГО ОБУЧЕНИЯ")
            
            # Выбор профилей
            print("Выберите профили для обучения (через пробел, например: 1 2 3):")
            self.print_profiles_menu()
            
            choices = input("Профили: ").strip().split()
            profile_map = {
                '1': 'conservative',
                '2': 'balanced', 
                '3': 'aggressive',
                '4': 'experimental',
                '5': 'quick_learning'
            }
            
            profiles = []
            for choice in choices:
                if choice in profile_map:
                    profiles.append(profile_map[choice])
            
            if not profiles:
                print("❌ Не выбрано ни одного профиля")
                return
            
            # Количество циклов
            try:
                cycles = int(input("Количество циклов на профиль (по умолчанию 3): ") or "3")
            except ValueError:
                cycles = 3
            
            # Задержка между циклами
            try:
                delay = int(input("Задержка между циклами в секундах (по умолчанию 300): ") or "300")
            except ValueError:
                delay = 300
            
            print(f"\n🔄 Запуск непрерывного обучения:")
            print(f"   📋 Профили: {', '.join(profiles)}")
            print(f"   🔢 Циклов на профиль: {cycles}")
            print(f"   ⏱️ Задержка: {delay} сек")
            print("⏳ Это может занять несколько часов...")
            print("💡 Нажмите Ctrl+C для корректного завершения")
            
            await self.runner.run_continuous_learning(profiles, cycles, delay)
            print("✅ Непрерывное обучение завершено!")
            
        except KeyboardInterrupt:
            print("\n🛑 Получен сигнал прерывания, завершаем...")
        except Exception as e:
            print(f"❌ Ошибка непрерывного обучения: {e}")
    
    async def show_system_status(self):
        """Показать статус системы"""
        try:
            print("\n🔍 СТАТУС СИСТЕМЫ")
            print("-" * 50)
            
            status = await self.runner.get_system_status()
            
            print(f"⏰ Время: {status.get('timestamp', 'N/A')}")
            print(f"🏃 Работает: {'Да' if status.get('is_running') else 'Нет'}")
            print(f"🆔 Текущая сессия: {status.get('session_id', 'Нет')}")
            
            # Статус Mistral сервера
            mistral_status = status.get('mistral_server', {})
            if mistral_status:
                print(f"\n🤖 MISTRAL СЕРВЕР:")
                print(f"   📡 Статус: {'Запущен' if mistral_status.get('is_running') else 'Остановлен'}")
                print(f"   🌐 URL: {mistral_status.get('url', 'N/A')}")
                print(f"   📊 Модель: {mistral_status.get('model_name', 'N/A')}")
            
            # Статус базы данных
            db_status = status.get('database', {})
            if db_status:
                print(f"\n💾 БАЗА ДАННЫХ:")
                print(f"   📁 Путь: {db_status.get('database_path', 'N/A')}")
                print(f"   📏 Размер: {db_status.get('database_size_mb', 0):.2f} MB")
                print(f"   📊 Сессий: {db_status.get('sessions_count', 0)}")
                print(f"   💰 Сделок: {db_status.get('trades_count', 0)}")
            
            print("-" * 50)
            
        except Exception as e:
            print(f"❌ Ошибка получения статуса: {e}")
    
    async def show_performance_stats(self):
        """Показать статистику производительности"""
        try:
            print("\n📈 СТАТИСТИКА ПРОИЗВОДИТЕЛЬНОСТИ")
            print("-" * 50)
            
            if not self.runner.persistence_manager:
                print("❌ Менеджер персистентности не инициализирован")
                return
            
            # Выбор периода
            print("Выберите период для анализа:")
            print("1. 📅 Последние 7 дней")
            print("2. 📅 Последние 30 дней")
            print("3. 📅 Последние 90 дней")
            
            choice = input("Период (1-3): ").strip()
            days_map = {'1': 7, '2': 30, '3': 90}
            days = days_map.get(choice, 30)
            
            stats = self.runner.persistence_manager.get_performance_statistics(days)
            
            if not stats:
                print("❌ Нет данных для анализа")
                return
            
            session_stats = stats.get('session_statistics', {})
            trade_stats = stats.get('trade_statistics', {})
            symbol_stats = stats.get('symbol_statistics', [])
            
            print(f"\n📊 СТАТИСТИКА ЗА {days} ДНЕЙ:")
            print(f"   🎯 Всего сессий: {session_stats.get('total_sessions', 0)}")
            print(f"   ✅ Завершенных: {session_stats.get('completed_sessions', 0)}")
            print(f"   📈 Средний винрейт: {session_stats.get('avg_win_rate', 0):.2f}%")
            print(f"   💰 Общий PnL: {session_stats.get('total_pnl', 0):.2f}")
            print(f"   📊 Средний PnL на сессию: {session_stats.get('avg_pnl_per_session', 0):.2f}")
            
            print(f"\n💼 СТАТИСТИКА СДЕЛОК:")
            print(f"   🔢 Всего сделок: {trade_stats.get('total_trades', 0)}")
            print(f"   💚 Прибыльных: {trade_stats.get('profitable_trades', 0)}")
            print(f"   💰 Средний PnL: {trade_stats.get('avg_pnl_per_trade', 0):.2f}")
            print(f"   🎯 Средняя уверенность: {trade_stats.get('avg_confidence', 0):.3f}")
            print(f"   ⏱️ Средняя длительность: {trade_stats.get('avg_duration', 0):.1f} мин")
            
            if symbol_stats:
                print(f"\n📊 ТОП СИМВОЛЫ:")
                for i, symbol in enumerate(symbol_stats[:5], 1):
                    win_rate = (symbol['profitable_count'] / symbol['trades_count'] * 100) if symbol['trades_count'] > 0 else 0
                    print(f"   {i}. {symbol['symbol']}: {symbol['trades_count']} сделок, {win_rate:.1f}% винрейт, {symbol['total_pnl']:.2f} PnL")
            
            print("-" * 50)
            
        except Exception as e:
            print(f"❌ Ошибка получения статистики: {e}")
    
    async def show_recent_sessions(self):
        """Показать последние сессии"""
        try:
            print("\n🗂️ ПОСЛЕДНИЕ СЕССИИ")
            print("-" * 80)
            
            if not self.runner.persistence_manager:
                print("❌ Менеджер персистентности не инициализирован")
                return
            
            sessions = self.runner.persistence_manager.get_recent_sessions(10)
            
            if not sessions:
                print("📭 Нет сохраненных сессий")
                return
            
            print(f"{'№':<3} {'ID сессии':<25} {'Название':<20} {'Винрейт':<8} {'PnL':<10} {'Статус':<10}")
            print("-" * 80)
            
            for i, session in enumerate(sessions, 1):
                session_id_short = session.session_id[-20:] if len(session.session_id) > 20 else session.session_id
                name_short = session.session_name[:18] + "..." if len(session.session_name) > 20 else session.session_name
                
                print(f"{i:<3} {session_id_short:<25} {name_short:<20} {session.win_rate:<7.2f}% {session.total_pnl:<9.2f} {session.status:<10}")
            
            print("-" * 80)
            
            # Опция экспорта
            choice = input("\nЭкспортировать сессию? (введите номер или Enter для пропуска): ").strip()
            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(sessions):
                    session = sessions[idx]
                    export_path = f"exports/session_{session.session_id}.json"
                    os.makedirs("exports", exist_ok=True)
                    
                    if self.runner.persistence_manager.export_session(session.session_id, export_path):
                        print(f"✅ Сессия экспортирована: {export_path}")
                    else:
                        print("❌ Ошибка экспорта сессии")
            
        except Exception as e:
            print(f"❌ Ошибка получения сессий: {e}")
    
    async def profiles_settings(self):
        """Настройки профилей"""
        try:
            print("\n⚙️ НАСТРОЙКИ ПРОФИЛЕЙ")
            print("-" * 50)
            print("1. 📋 Показать доступные профили")
            print("2. 🔧 Создать новый профиль")
            print("3. ✏️ Редактировать профиль")
            print("0. ⬅️ Назад")
            
            choice = input("Выберите действие: ").strip()
            
            if choice == '1':
                # Показать профили
                config_manager = self.runner.config_manager
                profiles = config_manager.get_available_profiles()
                
                print(f"\n📋 ДОСТУПНЫЕ ПРОФИЛИ ({len(profiles)}):")
                for name, profile in profiles.items():
                    rl_config = profile.get('reinforcement_learning', {})
                    print(f"\n🏷️ {name}:")
                    print(f"   📈 Скорость обучения: {rl_config.get('learning_rate', 0)}")
                    print(f"   🎁 Множитель награды: {rl_config.get('reward_multiplier', 0)}")
                    print(f"   ⚠️ Множитель наказания: {rl_config.get('punishment_multiplier', 0)}")
                    print(f"   📉 Затухание весов: {rl_config.get('weight_decay', 0)}")
            
            elif choice == '2':
                print("🔧 Создание нового профиля (функция в разработке)")
            
            elif choice == '3':
                print("✏️ Редактирование профиля (функция в разработке)")
            
        except Exception as e:
            print(f"❌ Ошибка настроек профилей: {e}")
    
    async def maintenance(self):
        """Обслуживание системы"""
        try:
            print("\n🧹 ОБСЛУЖИВАНИЕ СИСТЕМЫ")
            print("-" * 50)
            print("1. 💾 Создать резервную копию")
            print("2. 🗑️ Очистить старые данные")
            print("3. 📊 Информация о базе данных")
            print("4. 🔧 Проверка целостности")
            print("0. ⬅️ Назад")
            
            choice = input("Выберите действие: ").strip()
            
            if choice == '1':
                # Резервная копия
                if self.runner.persistence_manager:
                    backup_path = self.runner.persistence_manager.db.create_backup()
                    if backup_path:
                        print(f"✅ Резервная копия создана: {backup_path}")
                    else:
                        print("❌ Ошибка создания резервной копии")
                else:
                    print("❌ Менеджер персистентности не доступен")
            
            elif choice == '2':
                # Очистка данных
                days = input("Удалить данные старше (дней, по умолчанию 90): ").strip()
                try:
                    days = int(days) if days else 90
                except ValueError:
                    days = 90
                
                print(f"🗑️ Очистка данных старше {days} дней...")
                if self.runner.persistence_manager:
                    success = self.runner.persistence_manager.cleanup_old_data(days)
                    if success:
                        print("✅ Очистка завершена")
                    else:
                        print("❌ Ошибка очистки")
                else:
                    print("❌ Менеджер персистентности не доступен")
            
            elif choice == '3':
                # Информация о БД
                if self.runner.persistence_manager:
                    info = self.runner.persistence_manager.get_database_info()
                    print(f"\n💾 ИНФОРМАЦИЯ О БАЗЕ ДАННЫХ:")
                    print(f"   📁 Путь: {info.get('database_path', 'N/A')}")
                    print(f"   📏 Размер: {info.get('database_size_mb', 0):.2f} MB")
                    print(f"   📊 Сессий: {info.get('sessions_count', 0)}")
                    print(f"   💰 Сделок: {info.get('trades_count', 0)}")
                    print(f"   📈 Записей эволюции: {info.get('weight_evolution_count', 0)}")
                    print(f"   📅 Последняя активность: {info.get('last_session_created', 'N/A')}")
                else:
                    print("❌ Менеджер персистентности не доступен")
            
            elif choice == '4':
                print("🔧 Проверка целостности (функция в разработке)")
            
        except Exception as e:
            print(f"❌ Ошибка обслуживания: {e}")
    
    async def run(self):
        """Основной цикл интерфейса"""
        self.print_banner()
        
        # Инициализация системы
        if not await self.initialize_system():
            print("❌ Не удалось инициализировать систему. Завершение работы.")
            return
        
        try:
            while True:
                self.print_menu()
                choice = input("Выберите действие (0-7): ").strip()
                
                if choice == '0':
                    print("👋 До свидания!")
                    break
                elif choice == '1':
                    await self.run_single_cycle()
                elif choice == '2':
                    await self.run_continuous_learning()
                elif choice == '3':
                    await self.show_system_status()
                elif choice == '4':
                    await self.show_performance_stats()
                elif choice == '5':
                    await self.show_recent_sessions()
                elif choice == '6':
                    await self.profiles_settings()
                elif choice == '7':
                    await self.maintenance()
                else:
                    print("❌ Неверный выбор. Попробуйте снова.")
                
                if choice != '0':
                    input("\nНажмите Enter для продолжения...")
                    print("\n" * 2)  # Очистка экрана
        
        except KeyboardInterrupt:
            print("\n🛑 Получен сигнал прерывания")
        except Exception as e:
            print(f"❌ Критическая ошибка интерфейса: {e}")
        finally:
            if self.runner:
                await self.runner.cleanup()

async def main():
    """Точка входа"""
    interface = QuickStartInterface()
    await interface.run()

if __name__ == "__main__":
    # Создаем необходимые директории
    os.makedirs("logs", exist_ok=True)
    os.makedirs("exports", exist_ok=True)
    
    # Запускаем интерфейс
    asyncio.run(main())