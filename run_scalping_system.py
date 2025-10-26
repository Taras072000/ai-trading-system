#!/usr/bin/env python3
"""
Запуск гибридной скальпинг системы ансамбля AI моделей
Peper Binance v4 - Командная работа для достижения минимального профита с учетом комиссий
"""

import asyncio
import logging
import sys
import json
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
from typing import Dict, List, Optional
from utils.timezone_utils import get_utc_now

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'scalping_system_{get_utc_now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# Импорт системы
try:
    from scalping_ensemble_system import ScalpingEnsembleSystem, ScalpingSignal, MarketPhase
except ImportError as e:
    logger.error(f"Ошибка импорта ScalpingEnsembleSystem: {e}")
    sys.exit(1)

class ScalpingSystemRunner:
    """Запускатель скальпинг системы"""
    
    def __init__(self, config_path: str = "scalping_config.json"):
        self.config_path = config_path
        self.config = self._load_config()
        self.scalping_system = ScalpingEnsembleSystem()
        self.active_trades = {}
        self.daily_stats = {
            'trades': 0,
            'wins': 0,
            'losses': 0,
            'total_profit': 0.0,
            'start_time': get_utc_now()
        }
        
    def _load_config(self) -> Dict:
        """Загрузка конфигурации"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Ошибка загрузки конфигурации: {e}")
            return {}
    
    async def run_live_scalping(self, symbols: List[str], duration_hours: int = 1):
        """Запуск живого скальпинга"""
        
        print("🚀 ЗАПУСК ГИБРИДНОЙ СКАЛЬПИНГ СИСТЕМЫ")
        print("="*60)
        print(f"🎯 Цель: Минимальный профит с учетом комиссий Binance")
        print(f"🤖 AI Команда: LGBM + Lava + Mistral + Trading AI")
        print(f"📊 Символы: {', '.join(symbols)}")
        print(f"⏰ Длительность: {duration_hours} час(ов)")
        print("="*60)
        
        start_time = get_utc_now()
        end_time = start_time + timedelta(hours=duration_hours)
        
        # Основной цикл скальпинга
        while get_utc_now() < end_time:
            try:
                # Проверяем каждый символ
                for symbol in symbols:
                    await self._process_symbol(symbol)
                
                # Проверяем активные сделки
                await self._check_active_trades()
                
                # Выводим статистику каждые 10 минут
                if get_utc_now().minute % 10 == 0:
                    self._print_stats()
                
                # Пауза между циклами (30 секунд)
                await asyncio.sleep(30)
                
            except KeyboardInterrupt:
                print("\n⏹️  Остановка системы пользователем...")
                break
            except Exception as e:
                logger.error(f"Ошибка в основном цикле: {e}")
                await asyncio.sleep(60)  # Пауза при ошибке
        
        # Финальная статистика
        await self._print_final_stats()
    
    async def _process_symbol(self, symbol: str):
        """Обработка одного символа"""
        
        try:
            # Проверяем лимиты
            if not self._check_trading_limits():
                return
            
            # Генерируем сигнал
            signal = await self.scalping_system.generate_scalping_signal(symbol, '1m')
            
            if signal and signal.action in ['BUY', 'SELL']:
                logger.info(f"🎯 Сигнал для {symbol}: {signal.action} "
                          f"(уверенность: {signal.confidence:.1%}, "
                          f"прибыль: {signal.expected_profit_after_fees:.2f} пипсов)")
                
                # В реальной торговле здесь был бы вызов API биржи
                await self._simulate_trade(symbol, signal)
        
        except Exception as e:
            logger.error(f"Ошибка обработки {symbol}: {e}")
    
    async def _simulate_trade(self, symbol: str, signal: ScalpingSignal):
        """Симуляция сделки (в реальности - вызов API биржи)"""
        
        trade_id = f"{symbol}_{get_utc_now().strftime('%H%M%S')}"
        
        # Добавляем в активные сделки
        self.active_trades[trade_id] = {
            'symbol': symbol,
            'signal': signal,
            'entry_time': get_utc_now(),
            'status': 'active'
        }
        
        print(f"📈 НОВАЯ СДЕЛКА: {trade_id}")
        print(f"   Символ: {symbol}")
        print(f"   Действие: {signal.action}")
        print(f"   Цена входа: ${signal.entry_price:.4f}")
        print(f"   Стоп-лосс: ${signal.stop_loss:.4f}")
        print(f"   Тейк-профит: ${signal.take_profit:.4f}")
        print(f"   Фаза рынка: {signal.market_phase.value}")
        print(f"   AI Консенсус: {signal.ai_consensus}")
        print()
    
    async def _check_active_trades(self):
        """Проверка активных сделок"""
        
        completed_trades = []
        
        for trade_id, trade_info in self.active_trades.items():
            # Проверяем время жизни сделки (максимум 5 минут для скальпинга)
            if get_utc_now() - trade_info['entry_time'] > timedelta(minutes=5):
                # Закрываем сделку
                await self._close_trade(trade_id, 'timeout')
                completed_trades.append(trade_id)
        
        # Удаляем завершенные сделки
        for trade_id in completed_trades:
            del self.active_trades[trade_id]
    
    async def _close_trade(self, trade_id: str, reason: str):
        """Закрытие сделки"""
        
        trade_info = self.active_trades[trade_id]
        signal = trade_info['signal']
        
        # Симулируем результат (в реальности - получаем с биржи)
        import random
        
        # Простая симуляция: 70% вероятность прибыли для демонстрации
        is_profitable = random.random() < 0.7
        
        if is_profitable:
            profit_pips = signal.expected_profit_after_fees
            self.daily_stats['wins'] += 1
            result_emoji = "✅"
        else:
            profit_pips = -self.config['risk_management']['stop_loss_pips']
            self.daily_stats['losses'] += 1
            result_emoji = "❌"
        
        self.daily_stats['trades'] += 1
        self.daily_stats['total_profit'] += profit_pips
        
        print(f"{result_emoji} СДЕЛКА ЗАКРЫТА: {trade_id}")
        print(f"   Причина: {reason}")
        print(f"   Результат: {profit_pips:+.2f} пипсов")
        print(f"   Время: {get_utc_now() - trade_info['entry_time']}")
        print()
    
    def _check_trading_limits(self) -> bool:
        """Проверка лимитов торговли"""
        
        # Проверяем количество сделок в час
        trades_per_hour = self.config['trading_parameters']['max_trades_per_hour']
        if len(self.active_trades) >= trades_per_hour:
            return False
        
        # Проверяем дневные лимиты
        if self.daily_stats['trades'] >= self.config['trading_parameters']['max_trades_per_day']:
            return False
        
        # Проверяем максимальные потери
        max_daily_loss = self.config['risk_management']['max_daily_loss_pct'] * 100
        if self.daily_stats['total_profit'] < -max_daily_loss:
            logger.warning("Достигнут лимит дневных потерь!")
            return False
        
        return True
    
    def _print_stats(self):
        """Вывод текущей статистики"""
        
        if self.daily_stats['trades'] == 0:
            return
        
        winrate = self.daily_stats['wins'] / self.daily_stats['trades']
        runtime = get_utc_now() - self.daily_stats['start_time']
        
        print(f"📊 СТАТИСТИКА ({runtime}):")
        print(f"   Сделок: {self.daily_stats['trades']}")
        print(f"   Прибыльных: {self.daily_stats['wins']}")
        print(f"   Убыточных: {self.daily_stats['losses']}")
        print(f"   Винрейт: {winrate:.1%}")
        print(f"   Общая прибыль: {self.daily_stats['total_profit']:+.2f} пипсов")
        print(f"   Активных сделок: {len(self.active_trades)}")
        print("-" * 40)
    
    async def _print_final_stats(self):
        """Финальная статистика"""
        
        print("\n" + "="*60)
        print("🏁 ФИНАЛЬНЫЕ РЕЗУЛЬТАТЫ СКАЛЬПИНГ СИСТЕМЫ")
        print("="*60)
        
        if self.daily_stats['trades'] > 0:
            winrate = self.daily_stats['wins'] / self.daily_stats['trades']
            avg_profit = self.daily_stats['total_profit'] / self.daily_stats['trades']
            
            print(f"📈 Всего сделок: {self.daily_stats['trades']}")
            print(f"✅ Прибыльных: {self.daily_stats['wins']}")
            print(f"❌ Убыточных: {self.daily_stats['losses']}")
            print(f"🎯 Винрейт: {winrate:.1%}")
            print(f"💰 Общая прибыль: {self.daily_stats['total_profit']:+.2f} пипсов")
            print(f"📊 Средняя прибыль: {avg_profit:+.2f} пипсов/сделка")
            
            # Оценка с учетом комиссий
            total_fees = self.daily_stats['trades'] * 2 * 0.1  # 0.1% * 2 (вход/выход)
            net_profit = self.daily_stats['total_profit'] - total_fees
            print(f"💸 Комиссии: -{total_fees:.2f} пипсов")
            print(f"💎 Чистая прибыль: {net_profit:+.2f} пипсов")
            
            if winrate >= 0.75:
                print("🎉 ЦЕЛЬ ДОСТИГНУТА! Винрейт 75%+")
            else:
                print("⚠️  Цель не достигнута, требуется оптимизация")
        else:
            print("❌ Сделок не было")
        
        print("="*60)
    
    async def run_training_mode(self):
        """Режим обучения AI моделей"""
        
        print("\n🎓 РЕЖИМ ОБУЧЕНИЯ AI МОДЕЛЕЙ")
        print("="*50)
        print("🤖 Доступные AI модели для обучения:")
        print("1. 🧠 Trading AI - Быстрые торговые решения")
        print("2. 🌟 LGBM AI - Градиентный бустинг")
        print("3. 🔥 Lava AI - Распознавание паттернов")
        print("4. 🎯 Mistral AI - Гибридный анализ")
        print("5. 🚀 Все модели одновременно")
        print("="*50)
        
        try:
            choice = input("\nВыберите модель для обучения (1-5): ").strip()
            
            if choice == "1":
                await self._train_trading_ai()
            elif choice == "2":
                await self._train_lgbm_ai()
            elif choice == "3":
                await self._train_lava_ai()
            elif choice == "4":
                await self._train_mistral_ai()
            elif choice == "5":
                await self._train_all_models()
            else:
                print("❌ Неверный выбор")
                
        except KeyboardInterrupt:
            print("\n⏹️  Обучение остановлено пользователем")
        except Exception as e:
            logger.error(f"Ошибка в режиме обучения: {e}")
            print(f"❌ Ошибка: {e}")
    
    async def _train_trading_ai(self):
        """Обучение Trading AI"""
        print("\n🧠 ОБУЧЕНИЕ TRADING AI")
        print("-" * 30)
        
        try:
            # Импорт тренера
            from trading_ai_trainer import TradingAITrainer
            
            # Параметры обучения
            symbol = input("Символ для обучения (BTCUSDT): ").strip() or "BTCUSDT"
            days = int(input("Дней данных для обучения (365): ") or "365")
            
            print(f"🔄 Начинаю обучение Trading AI для {symbol} за {days} дней...")
            
            trainer = TradingAITrainer(symbol)
            
            # Загружаем данные
            print("📊 Загрузка рыночных данных...")
            data = await trainer.load_market_data(days)
            
            # Подготавливаем признаки
            print("🔧 Подготовка признаков...")
            features_df = trainer.prepare_features(data)
            
            # Создаем метки
            print("🏷️  Создание меток...")
            labels, features_df = trainer.create_labels(data, features_df)
            
            # Обучаем модели
            print("🤖 Обучение моделей...")
            results = trainer.train_models(features_df, labels)
            
            # Оцениваем модели
            print("📈 Оценка моделей...")
            evaluation = trainer.evaluate_model(results)
            
            # Сохраняем лучшую модель
            print("💾 Сохранение модели...")
            trainer.save_model()
            
            print("✅ Обучение Trading AI завершено!")
            print(f"📊 Лучшая модель: {evaluation.get('best_model', 'N/A')}")
            print(f"🎯 Точность: {evaluation.get('best_score', 0):.4f}")
            print(f"💾 Модель сохранена в: models/trading_ai/{symbol}_trading_model.joblib")
            
        except ImportError as e:
            print(f"❌ Ошибка импорта: {e}")
            print("💡 Убедитесь, что файл trading_ai_trainer.py находится в корневой папке")
        except Exception as e:
            print(f"❌ Ошибка обучения Trading AI: {e}")
            logger.error(f"Ошибка в _train_trading_ai: {e}", exc_info=True)
    
    async def _train_lgbm_ai(self):
        """Обучение LGBM AI"""
        print("\n🌟 ОБУЧЕНИЕ LGBM AI")
        print("-" * 30)
        
        try:
            # Получаем AI модуль
            lgbm_ai = self.scalping_system.ai_manager.models.get('lgbm_ai')
            if not lgbm_ai:
                print("❌ LGBM AI не найден в системе")
                return
            
            # Параметры обучения
            symbols = input("Символы для обучения (BTCUSDT,ETHUSDT): ").strip() or "BTCUSDT,ETHUSDT"
            symbols = [s.strip() for s in symbols.split(",")]
            
            days = int(input("Дней данных для обучения (90): ") or "90")
            
            print(f"🔄 Начинаю обучение LGBM AI на {symbols} за {days} дней...")
            
            # Подготовка данных (упрощенная версия)
            from data_collector import BinanceDataCollector
            
            all_data = []
            async with BinanceDataCollector() as collector:
                for symbol in symbols:
                    data = await collector.get_historical_data(symbol, "1h", days)
                    if data is not None and len(data) > 0:
                        all_data.append(data)
            
            if not all_data:
                print("❌ Не удалось получить данные для обучения")
                return
            
            # Объединяем данные
            combined_data = pd.concat(all_data, ignore_index=True)
            
            # Создаем простые признаки и метки
            X = combined_data[['open', 'high', 'low', 'close', 'volume']].copy()
            y = (combined_data['close'].shift(-1) > combined_data['close']).astype(int)
            
            # Удаляем последнюю строку (нет будущей цены)
            X = X[:-1]
            y = y[:-1]
            
            # Обучение
            results = await lgbm_ai.train_model("scalping_model", X, y, "classification")
            
            print("✅ Обучение LGBM AI завершено!")
            print(f"📊 Результаты: {results}")
            
        except Exception as e:
            print(f"❌ Ошибка обучения LGBM AI: {e}")
    
    async def _train_lava_ai(self):
        """Обучение Lava AI"""
        print("\n🔥 ОБУЧЕНИЕ LAVA AI")
        print("-" * 30)
        
        try:
            # Получаем AI модуль
            lava_ai = self.scalping_system.ai_manager.models.get('lava_ai')
            if not lava_ai:
                print("❌ Lava AI не найден в системе")
                return
            
            # Параметры обучения
            symbols = input("Символы для обучения (BTCUSDT,ETHUSDT): ").strip() or "BTCUSDT,ETHUSDT"
            symbols = [s.strip() for s in symbols.split(",")]
            
            days = int(input("Дней данных для обучения (90): ") or "90")
            
            print(f"🔄 Начинаю обучение Lava AI на {symbols} за {days} дней...")
            
            # Подготовка данных
            from data_collector import BinanceDataCollector
            
            all_data = []
            async with BinanceDataCollector() as collector:
                for symbol in symbols:
                    print(f"📊 Загружаю данные для {symbol}...")
                    data = await collector.get_historical_data(symbol, "1h", days)
                    if data is not None and len(data) > 0:
                        print(f"✅ {symbol}: {len(data)} свечей загружено")
                        all_data.append(data)
            
            if not all_data:
                print("❌ Не удалось получить данные для обучения")
                return
            
            # Объединяем данные
            combined_data = pd.concat(all_data, ignore_index=True)
            print(f"📈 Общий объем данных: {len(combined_data)} свечей")
            
            # Создаем признаки и метки для анализа паттернов
            X = combined_data[['open', 'high', 'low', 'close', 'volume']].copy()
            y = (combined_data['close'].shift(-1) > combined_data['close']).astype(int)
            
            # Удаляем последнюю строку (нет будущей цены)
            X = X[:-1]
            y = y[:-1]
            
            print("🧠 Инициализация Lava AI модуля...")
            await lava_ai.initialize()
            
            print("🔥 Обучение модели на основе анализа паттернов...")
            # Обучение
            results = await lava_ai.train_model("scalping_pattern_model", X, y, "pattern_analysis")
            
            print("✅ Обучение Lava AI завершено!")
            print(f"📊 Результаты:")
            print(f"   • Модель: {results['model_name']}")
            print(f"   • Обучающая выборка: {results['training_samples']} образцов")
            print(f"   • Тестовая выборка: {results['test_samples']} образцов")
            print(f"   • Точность: {results['accuracy']:.3f}")
            print(f"   • Уверенность: {results['confidence']:.3f}")
            print(f"   • Резюме: {results['analysis_summary']}")
            
            # Показываем важность признаков
            print(f"\n🎯 Важность признаков:")
            for feature, importance in results['feature_importance'].items():
                print(f"   • {feature}: {importance:.1%}")
            
        except Exception as e:
            print(f"❌ Ошибка обучения Lava AI: {e}")
            logger.error(f"Ошибка обучения Lava AI: {e}")
    
    async def _train_mistral_ai(self):
        """Обучение Mistral AI"""
        print("\n🎯 ОБУЧЕНИЕ MISTRAL AI")
        print("-" * 30)
        
        try:
            # Получение модуля Mistral AI
            mistral_ai = self.scalping_system.ai_manager.models.get('mistral_ai')
            if not mistral_ai:
                print("❌ Ошибка: Mistral AI модуль не найден")
                return
            
            # Ввод параметров обучения
            print("\n📝 Настройка параметров обучения:")
            symbols_input = input("Введите символы через запятую (по умолчанию BTCUSDT,ETHUSDT): ").strip()
            symbols = [s.strip().upper() for s in symbols_input.split(',')] if symbols_input else ['BTCUSDT', 'ETHUSDT']
            
            days_input = input("Количество дней данных (по умолчанию 90): ").strip()
            days = int(days_input) if days_input.isdigit() else 90
            
            print(f"\n🔄 Загрузка данных для символов: {', '.join(symbols)}")
            print(f"📅 Период: {days} дней")
            
            # Подготовка данных
            all_data = []
            end_date = get_utc_now()
            start_date = end_date - timedelta(days=days)
            
            for symbol in symbols:
                print(f"📊 Загружаю данные для {symbol}...")
                
                # Попробуем загрузить данные из HistoricalDataManager
                data = await self.scalping_system.data_manager.load_data(
                    symbol=symbol, 
                    interval='1h',  # Используем часовой интервал для обучения
                    start_date=start_date,
                    end_date=end_date
                )
                
                if data is not None and not data.empty:
                    print(f"✅ Загружено {len(data)} свечей для {symbol}")
                    all_data.append(data)
                else:
                    print(f"⚠️ Нет данных для {symbol}, попробуем загрузить через data_collector...")
                    
                    # Если нет данных в HistoricalDataManager, используем data_collector
                    try:
                        async with self.scalping_system.data_collector as collector:
                            collector_data = await collector.get_historical_data(
                                symbol=symbol,
                                interval='1h',
                                days=days
                            )
                            if collector_data is not None and not collector_data.empty:
                                print(f"✅ Загружено {len(collector_data)} свечей для {symbol} через data_collector")
                                all_data.append(collector_data)
                            else:
                                print(f"❌ Не удалось загрузить данные для {symbol}")
                    except Exception as e:
                        print(f"❌ Ошибка загрузки данных для {symbol}: {e}")
            
            if not all_data:
                print("❌ Не удалось загрузить данные для обучения")
                return
            
            # Объединение данных
            combined_data = pd.concat(all_data, ignore_index=True)
            print(f"📈 Общий объем данных: {len(combined_data)} записей")
            
            # Создание признаков для обучения
            features = combined_data[['open', 'high', 'low', 'close', 'volume']].copy()
            target = (combined_data['close'].shift(-1) > combined_data['close']).astype(int)
            
            # Удаление последней строки (нет целевого значения)
            features = features[:-1]
            target = target[:-1]
            
            print(f"🧠 Инициализация Mistral AI...")
            await mistral_ai.initialize()
            
            print(f"🎯 Начинаю обучение модели...")
            
            # Обучение модели
            results = await mistral_ai.train_model(
                model_name='mistral_hybrid_model',
                X=features,
                y=target
            )
            
            # Отображение результатов
            print(f"\n🎉 РЕЗУЛЬТАТЫ ОБУЧЕНИЯ MISTRAL AI:")
            print("=" * 50)
            print(f"📊 Модель: {results.get('model_name', 'N/A')}")
            print(f"📈 Обучающих образцов: {results.get('training_samples', 0)}")
            print(f"🎯 Точность: {results.get('accuracy', 0):.3f}")
            print(f"💪 Уверенность: {results.get('confidence', 0):.3f}")
            print(f"⏱️ Время обучения: {results.get('training_time', 0):.2f}с")
            print(f"🔤 Токенов использовано: {results.get('tokens_used', 0)}")
            
            if 'analysis' in results:
                print(f"\n🤖 Анализ Mistral AI:")
                print(f"   {results['analysis']}")
            
            if 'recommendations' in results:
                print(f"\n💡 Рекомендации:")
                for i, rec in enumerate(results['recommendations'], 1):
                    print(f"   {i}. {rec}")
            
            print("\n" + "=" * 50)
                
        except Exception as e:
            print(f"❌ Ошибка при обучении Mistral AI: {e}")
            import traceback
            traceback.print_exc()
    
    async def _train_all_models(self):
        """Обучение всех моделей"""
        print("\n🚀 ОБУЧЕНИЕ ВСЕХ AI МОДЕЛЕЙ")
        print("="*40)
        
        try:
            # Импорт главного тренера
            from multi_ai_trainer import MultiAITrainer
            
            trainer = MultiAITrainer()
            
            # Параметры обучения по умолчанию для автоматического запуска
            symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
            days = 180  # Полгода данных для комплексного тестирования
            
            print(f"🔄 Начинаю обучение всех AI моделей на {symbols} за {days} дней...")
            print("⚠️  Это может занять значительное время...")
            
            # Запуск полного обучения
            results = await trainer.train_all_models(symbols)
            
            print("\n✅ ОБУЧЕНИЕ ВСЕХ МОДЕЛЕЙ ЗАВЕРШЕНО!")
            print("="*40)
            
            for model_name, result in results.items():
                print(f"🤖 {model_name.upper()}:")
                print(f"   📊 Точность: {result.accuracy:.2%}")
                print(f"   🎯 Винрейт: {result.winrate:.2%}")
                print(f"   💰 Прибыльных сделок: {result.profitable_trades}/{result.total_trades}")
                print(f"   ⏱️  Время обучения: {result.training_time:.1f}с")
                print()
            
            return results
            
        except ImportError as e:
            print(f"❌ Модуль multi_ai_trainer не найден: {e}")
            print("🔧 Проверяю наличие файла...")
            import os
            if os.path.exists("multi_ai_trainer.py"):
                print("✅ Файл multi_ai_trainer.py найден")
            else:
                print("❌ Файл multi_ai_trainer.py отсутствует")
        except Exception as e:
            print(f"❌ Ошибка обучения всех моделей: {e}")
            import traceback
            traceback.print_exc()
    
    async def run_backtest(self, symbol: str, days: int = 7):
        """Запуск бэктеста"""
        
        print(f"🔄 БЭКТЕСТ СКАЛЬПИНГ СИСТЕМЫ")
        print("="*50)
        print(f"📊 Символ: {symbol}")
        print(f"📅 Период: {days} дней")
        print("="*50)
        
        try:
            results = await self.scalping_system.backtest_scalping_system(symbol, days)
            
            if 'error' not in results:
                print(f"\n📈 РЕЗУЛЬТАТЫ БЭКТЕСТА:")
                print(f"   Всего сделок: {results['total_trades']}")
                print(f"   Прибыльных: {results['winning_trades']}")
                print(f"   Винрейт: {results['winrate']:.1%}")
                print(f"   Средняя прибыль: {results['avg_profit_per_trade']:.3%}")
                print(f"   Общая доходность: {results['total_return']:.1%}")
                print(f"   Финальный баланс: ${results['final_balance']:.2f}")
                
                # Анализ с учетом комиссий
                total_fees = results['total_trades'] * 2 * 0.001  # 0.1% * 2
                net_return = results['total_return'] - total_fees
                print(f"   Чистая доходность: {net_return:.1%}")
                
                if results['winrate'] >= 0.75:
                    print("\n✅ Система показывает целевой винрейт 75%+")
                else:
                    print("\n⚠️  Требуется дополнительная оптимизация")
                    
                # Показываем примеры сделок
                if 'trades' in results and results['trades']:
                    print(f"\n📋 ПРИМЕРЫ СДЕЛОК:")
                    for i, trade in enumerate(results['trades'][:5]):
                        profit_emoji = "✅" if trade['profit_pct'] > 0 else "❌"
                        print(f"   {i+1}. {profit_emoji} {trade['action']} "
                              f"${trade['entry_price']:.4f} → ${trade['exit_price']:.4f} "
                              f"({trade['profit_pct']:+.3%})")
            else:
                print(f"❌ Ошибка бэктеста: {results['error']}")
                
        except Exception as e:
            print(f"❌ Ошибка бэктеста: {e}")

async def main():
    """Главная функция"""
    
    print("🚀 PEPER BINANCE V4 - ГИБРИДНАЯ СКАЛЬПИНГ СИСТЕМА")
    print("="*70)
    print("🎯 Командная работа 4 AI для достижения минимального профита")
    print("💰 С учетом комиссий Binance (0.1%)")
    print("="*70)
    
    # Создаем систему
    runner = ScalpingSystemRunner()
    
    # Меню выбора режимов
    print("\nДоступные режимы:")
    print("1. 🎓 Обучение AI моделей")
    print("2. 📊 Бэктест ансамбля AI")
    print("3. 🚀 Live торговля")
    
    try:
        choice = input("\nВыберите режим (1-3): ").strip()
        
        if choice == "1":
            # Запуск режима обучения
            await runner.run_training_mode()
        elif choice == "2":
            # Запуск бэктеста ансамбля
            symbol = input("Символ для бэктеста (BTCUSDT): ").strip() or "BTCUSDT"
            days = int(input("Дней для бэктеста (180): ") or "180")
            await runner.run_backtest(symbol, days)
        elif choice == "3":
            # Запуск live торговли
            symbols_input = input("Символы для торговли (BTCUSDT,ETHUSDT): ").strip() or "BTCUSDT,ETHUSDT"
            symbols = [s.strip() for s in symbols_input.split(",")]
            hours = int(input("Часов торговли (1): ") or "1")
            await runner.run_live_scalping(symbols, hours)
        else:
            print("❌ Неверный выбор")
            
    except KeyboardInterrupt:
        print("\n👋 Программа остановлена пользователем")
    except Exception as e:
        print(f"❌ Ошибка: {e}")

if __name__ == "__main__":
    asyncio.run(main())