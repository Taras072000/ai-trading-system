#!/usr/bin/env python3
"""
Automated Reinforcement Learning Script
Автоматизированный скрипт для обучения с подкреплением

Этот скрипт автоматически:
1. Запускает Mistral сервер
2. Инициализирует систему обучения с подкреплением
3. Проводит winrate тесты с адаптивным обучением
4. Сохраняет результаты и генерирует отчеты
5. Останавливает Mistral сервер
"""

import asyncio
import argparse
import logging
import sys
import os
import signal
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import json

# Добавляем путь к проекту
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ai_modules.reinforcement_learning_engine import ReinforcementLearningEngine, ReinforcementConfig
from ai_modules.mistral_server_manager import MistralServerManager
from ai_modules.multi_ai_orchestrator import MultiAIOrchestrator
from reinforcement_winrate_tester import ReinforcementWinrateTester, ReinforcementTestConfig
from analytics.reinforcement_learning_analytics import ReinforcementLearningAnalytics, AnalyticsConfig
from database.persistence_manager import PersistenceManager, PersistenceConfig
from config.config_manager import ConfigManager

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/reinforcement_learning.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class ReinforcementLearningRunner:
    """
    Основной класс для запуска системы обучения с подкреплением
    """
    
    def __init__(self, config_path: str = "config"):
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager
        
        # Инициализируем компоненты
        self.persistence_manager = None
        self.mistral_manager = None
        self.orchestrator = None
        self.tester = None
        self.analytics = None
        
        # Состояние
        self.session_id = None
        self.is_running = False
        self.shutdown_requested = False
        
        # Настройка обработчика сигналов
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info("✅ ReinforcementLearningRunner инициализирован")
    
    def _signal_handler(self, signum, frame):
        """Обработчик сигналов для корректного завершения"""
        logger.info(f"🛑 Получен сигнал {signum}, начинаем корректное завершение...")
        self.shutdown_requested = True
    
    async def initialize_components(self):
        """Инициализация всех компонентов системы"""
        try:
            logger.info("🚀 Инициализация компонентов системы...")
            
            # Создаем необходимые директории
            os.makedirs("data", exist_ok=True)
            os.makedirs("logs", exist_ok=True)
            os.makedirs("reports", exist_ok=True)
            
            # Инициализируем менеджер персистентности
            persistence_config = PersistenceConfig(
                db_path=self.config.persistence_config.database_path,
                auto_backup=True,
                backup_interval_hours=24,
                max_backups=self.config.persistence_config.max_backups,
                evolution_snapshot_interval=10,
                enable_compression=self.config.persistence_config.compress_old_data
            )
            self.persistence_manager = PersistenceManager(persistence_config)
            
            # Инициализируем Mistral менеджер
            self.mistral_manager = MistralServerManager(self.config.mistral_server_config)
            
            # Запускаем Mistral сервер
            if self.config.mistral_server_config.auto_start:
                logger.info("🔄 Запуск Mistral сервера...")
                if await self.mistral_manager.start_server():
                    logger.info("✅ Mistral сервер запущен")
                else:
                    logger.error("❌ Не удалось запустить Mistral сервер")
                    return False
            
            # Инициализируем оркестратор
            self.orchestrator = MultiAIOrchestrator(
                backtest_mode=True,
                reinforcement_learning=True
            )
            await self.orchestrator.initialize()
            
            # Инициализируем тестер
            test_config = ReinforcementTestConfig(
                symbols=self.config.testing_config.default_symbols,
                start_date="2024-01-01",
                end_date="2024-12-31",
                initial_balance=10000
            )
            
            self.tester = ReinforcementWinrateTester(test_config)
            await self.tester.initialize()
            
            # Инициализируем аналитику
            analytics_config = AnalyticsConfig(
                reports_dir="reports",
                plots_dir="plots",
                results_dir="results"
            )
            self.analytics = ReinforcementLearningAnalytics(analytics_config)
            
            logger.info("✅ Все компоненты инициализированы")
            return True
            
        except Exception as e:
            logger.error(f"❌ Ошибка инициализации компонентов: {e}")
            return False
    
    async def run_learning_cycle(self, profile_name: str, session_name: Optional[str] = None) -> bool:
        """Запуск цикла обучения с подкреплением"""
        try:
            if not session_name:
                session_name = f"RL_Session_{profile_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            logger.info(f"🎯 Начинаем цикл обучения: {session_name} (профиль: {profile_name})")
            
            # Применяем профиль
            if not self.config_manager.apply_profile(profile_name):
                logger.error(f"❌ Не удалось применить профиль: {profile_name}")
                return False
            
            # Начинаем сессию в базе данных
            initial_weights = self.config.ai_modules_config.initial_weights
            config_snapshot = {}
            
            self.session_id = self.persistence_manager.start_session(
                session_name=session_name,
                profile_name=profile_name,
                initial_weights=initial_weights,
                config_snapshot=config_snapshot
            )
            
            if not self.session_id:
                logger.error("❌ Не удалось создать сессию в базе данных")
                return False
            
            # Настраиваем оркестратор для обучения
            if self.orchestrator.rl_engine:
                # Загружаем лучшие веса если доступны
                best_weights = self.persistence_manager.get_best_performing_weights(30)
                if best_weights:
                    logger.info("📈 Загружены лучшие веса из предыдущих сессий")
                    self.orchestrator.rl_engine.load_weights(best_weights)
                
                # Применяем конфигурацию обучения
                rl_config = ReinforcementConfig(
                    learning_rate=self.config.reinforcement_config.learning_rate,
                    reward_multiplier=self.config.reinforcement_config.reward_multiplier,
                    punishment_multiplier=self.config.reinforcement_config.punishment_multiplier,
                    weight_decay=self.config.reinforcement_config.weight_decay,
                    min_weight=self.config.reinforcement_config.min_weight,
                    max_weight=self.config.reinforcement_config.max_weight,
                    confidence_threshold=self.config.reinforcement_config.confidence_threshold
                )
                self.orchestrator.rl_engine.update_config(rl_config)
            
            # Запускаем тестирование с обучением
            self.is_running = True
            results = await self.tester.run_test_with_learning(
                orchestrator=self.orchestrator,
                persistence_manager=self.persistence_manager,
                session_id=self.session_id
            )
            
            if not results:
                logger.error("❌ Тестирование завершилось с ошибкой")
                return False
            
            # Завершаем сессию
            final_weights = self.orchestrator.rl_engine.get_weights() if self.orchestrator.rl_engine else {}
            
            success = self.persistence_manager.end_session(
                final_weights=final_weights,
                total_trades=results.total_trades,
                profitable_trades=results.profitable_trades,
                total_pnl=results.total_pnl,
                win_rate=results.win_rate
            )
            
            if success:
                logger.info(f"✅ Сессия завершена успешно: {self.session_id}")
                
                # Генерируем отчет
                await self.generate_session_report()
                
                # Выводим результаты
                self._print_session_results(results)
                
                return True
            else:
                logger.error("❌ Ошибка завершения сессии")
                return False
            
        except Exception as e:
            logger.error(f"❌ Ошибка цикла обучения: {e}")
            return False
        finally:
            self.is_running = False
    
    async def generate_session_report(self):
        """Генерация отчета по сессии"""
        try:
            if not self.session_id or not self.analytics:
                return
            
            logger.info("📊 Генерация отчета по сессии...")
            
            # Получаем данные сессии
            session_data = self.persistence_manager.get_session_data(self.session_id)
            if not session_data:
                logger.error("❌ Не удалось получить данные сессии")
                return
            
            # Генерируем HTML отчет
            report_path = f"reports/session_report_{self.session_id}.html"
            success = await self.analytics.generate_session_report(
                session_data=session_data,
                output_path=report_path
            )
            
            if success:
                logger.info(f"✅ Отчет сохранен: {report_path}")
            else:
                logger.error("❌ Ошибка генерации отчета")
            
        except Exception as e:
            logger.error(f"❌ Ошибка генерации отчета: {e}")
    
    def _print_session_results(self, results):
        """Вывод результатов сессии"""
        print("\n" + "="*80)
        print("📊 РЕЗУЛЬТАТЫ СЕССИИ ОБУЧЕНИЯ С ПОДКРЕПЛЕНИЕМ")
        print("="*80)
        print(f"🆔 ID сессии: {self.session_id}")
        print(f"📈 Общее количество сделок: {results.total_trades}")
        print(f"💰 Прибыльные сделки: {results.profitable_trades}")
        print(f"📊 Винрейт: {results.win_rate:.2f}%")
        print(f"💵 Общий PnL: {results.total_pnl:.2f}")
        print(f"⚡ Средняя уверенность: {results.avg_confidence:.3f}")
        print(f"⏱️ Средняя длительность: {results.avg_duration:.1f} мин")
        
        if hasattr(results, 'learning_stats') and results.learning_stats:
            print(f"\n🧠 СТАТИСТИКА ОБУЧЕНИЯ:")
            stats = results.learning_stats
            print(f"🔄 Применено наград: {stats.get('total_rewards', 0)}")
            print(f"⚠️ Применено наказаний: {stats.get('total_punishments', 0)}")
            print(f"📊 Изменений весов: {stats.get('weight_changes', 0)}")
            
            if 'final_weights' in stats:
                print(f"\n⚖️ ФИНАЛЬНЫЕ ВЕСА AI МОДУЛЕЙ:")
                for ai_name, weight in stats['final_weights'].items():
                    print(f"   {ai_name}: {weight:.3f}")
        
        print("="*80)
        print(f"📁 Отчет сохранен в: reports/session_report_{self.session_id}.html")
        print("="*80 + "\n")
    
    async def run_continuous_learning(self, 
                                    profiles: List[str], 
                                    cycles_per_profile: int = 3,
                                    delay_between_cycles: int = 300):
        """Непрерывное обучение с несколькими профилями"""
        try:
            logger.info(f"🔄 Запуск непрерывного обучения: {len(profiles)} профилей, {cycles_per_profile} циклов каждый")
            
            total_cycles = len(profiles) * cycles_per_profile
            current_cycle = 0
            
            for profile in profiles:
                if self.shutdown_requested:
                    break
                
                logger.info(f"📋 Переключение на профиль: {profile}")
                
                for cycle in range(cycles_per_profile):
                    if self.shutdown_requested:
                        break
                    
                    current_cycle += 1
                    session_name = f"Continuous_{profile}_Cycle_{cycle+1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    
                    logger.info(f"🎯 Цикл {current_cycle}/{total_cycles}: {session_name}")
                    
                    success = await self.run_learning_cycle(profile, session_name)
                    
                    if not success:
                        logger.error(f"❌ Цикл {current_cycle} завершился с ошибкой")
                        continue
                    
                    # Пауза между циклами (кроме последнего)
                    if current_cycle < total_cycles and not self.shutdown_requested:
                        logger.info(f"⏸️ Пауза {delay_between_cycles} секунд перед следующим циклом...")
                        await asyncio.sleep(delay_between_cycles)
            
            logger.info("✅ Непрерывное обучение завершено")
            
        except Exception as e:
            logger.error(f"❌ Ошибка непрерывного обучения: {e}")
    
    async def cleanup(self):
        """Очистка ресурсов"""
        try:
            logger.info("🧹 Очистка ресурсов...")
            
            # Завершаем текущую сессию если активна
            if self.persistence_manager and self.persistence_manager.is_session_active():
                logger.info("⏹️ Завершение активной сессии...")
                # Получаем текущие данные для завершения
                if self.orchestrator and self.orchestrator.rl_engine:
                    final_weights = self.orchestrator.rl_engine.get_weights()
                    stats = self.orchestrator.rl_engine.get_performance_metrics()
                    
                    self.persistence_manager.end_session(
                        final_weights=final_weights,
                        total_trades=stats.get('total_trades', 0),
                        profitable_trades=stats.get('profitable_trades', 0),
                        total_pnl=stats.get('total_pnl', 0.0),
                        win_rate=stats.get('win_rate', 0.0)
                    )
            
            # Останавливаем Mistral сервер
            if self.mistral_manager and self.config.mistral_server_config.auto_stop:
                logger.info("🛑 Остановка Mistral сервера...")
                await self.mistral_manager.stop_server()
            
            # Создаем резервную копию
            if self.persistence_manager:
                self.persistence_manager._create_backup_if_needed()
            
            logger.info("✅ Очистка завершена")
            
        except Exception as e:
            logger.error(f"❌ Ошибка очистки: {e}")
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Получение статуса системы"""
        try:
            status = {
                'timestamp': datetime.now().isoformat(),
                'is_running': self.is_running,
                'session_id': self.session_id,
                'mistral_server': None,
                'database': None,
                'recent_sessions': []
            }
            
            # Статус Mistral сервера
            if self.mistral_manager:
                status['mistral_server'] = self.mistral_manager.get_server_status()
            
            # Статус базы данных
            if self.persistence_manager:
                status['database'] = self.persistence_manager.get_database_info()
                status['recent_sessions'] = [
                    {
                        'session_id': session.session_id,
                        'session_name': session.session_name,
                        'start_time': session.start_time.isoformat(),
                        'win_rate': session.win_rate,
                        'total_pnl': session.total_pnl,
                        'status': session.status
                    }
                    for session in self.persistence_manager.get_recent_sessions(5)
                ]
            
            return status
            
        except Exception as e:
            logger.error(f"❌ Ошибка получения статуса: {e}")
            return {'error': str(e)}

async def main():
    """Основная функция"""
    parser = argparse.ArgumentParser(description='Reinforcement Learning System for Peper Binance v4')
    parser.add_argument('--config', default='config/reinforcement_learning_config.json', 
                       help='Путь к файлу конфигурации')
    parser.add_argument('--profile', default='balanced', 
                       help='Профиль обучения (conservative, balanced, aggressive, experimental)')
    parser.add_argument('--session-name', 
                       help='Название сессии (по умолчанию генерируется автоматически)')
    parser.add_argument('--continuous', action='store_true', 
                       help='Режим непрерывного обучения')
    parser.add_argument('--profiles', nargs='+', default=['conservative', 'balanced', 'aggressive'],
                       help='Профили для непрерывного обучения')
    parser.add_argument('--cycles', type=int, default=3,
                       help='Количество циклов на профиль в непрерывном режиме')
    parser.add_argument('--delay', type=int, default=300,
                       help='Задержка между циклами в секундах')
    parser.add_argument('--status', action='store_true',
                       help='Показать статус системы и выйти')
    
    args = parser.parse_args()
    
    # Создаем и инициализируем runner
    runner = ReinforcementLearningRunner(args.config)
    
    try:
        # Инициализируем компоненты
        if not await runner.initialize_components():
            logger.error("❌ Не удалось инициализировать систему")
            return 1
        
        # Режим статуса
        if args.status:
            status = await runner.get_system_status()
            print(json.dumps(status, indent=2, ensure_ascii=False))
            return 0
        
        # Режим непрерывного обучения
        if args.continuous:
            await runner.run_continuous_learning(
                profiles=args.profiles,
                cycles_per_profile=args.cycles,
                delay_between_cycles=args.delay
            )
        else:
            # Одиночный цикл обучения
            success = await runner.run_learning_cycle(args.profile, args.session_name)
            if not success:
                return 1
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("🛑 Получен сигнал прерывания")
        return 0
    except Exception as e:
        logger.error(f"❌ Критическая ошибка: {e}")
        return 1
    finally:
        await runner.cleanup()

if __name__ == "__main__":
    # Создаем директории если не существуют
    os.makedirs("logs", exist_ok=True)
    
    # Запускаем основную функцию
    exit_code = asyncio.run(main())
    sys.exit(exit_code)