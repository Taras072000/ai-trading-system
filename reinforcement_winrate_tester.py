#!/usr/bin/env python3
"""
Reinforcement Learning Winrate Tester
Расширенный тестер винрейта с поддержкой обучения с подкреплением и автозапуском Mistral
"""

import asyncio
import logging
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np

from ai_modules.multi_ai_orchestrator import MultiAIOrchestrator
from ai_modules.reinforcement_learning_engine import ReinforcementLearningEngine, ReinforcementConfig
from ai_modules.mistral_server_manager import MistralServerManager
from historical_data_manager import HistoricalDataManager
from data_collector import BinanceDataCollector

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ReinforcementTestConfig:
    """Конфигурация для тестирования с обучением с подкреплением"""
    # Основные параметры
    symbols: List[str]
    start_date: str
    end_date: str
    initial_balance: float = 10000.0
    
    # Параметры обучения с подкреплением
    enable_reinforcement_learning: bool = True
    learning_rate: float = 0.01
    reward_multiplier: float = 1.0
    punishment_multiplier: float = 1.5
    weight_decay: float = 0.001
    
    # Параметры тестирования
    test_intervals: int = 10  # Количество интервалов для обучения
    trades_per_interval: int = 50  # Количество сделок в интервале
    
    # Параметры Mistral
    auto_start_mistral: bool = True
    mistral_model: str = "mistral:latest"
    mistral_timeout: int = 300
    
    # Параметры сохранения
    save_results: bool = True
    results_dir: str = "results/reinforcement_learning"
    session_name: Optional[str] = None

@dataclass
class ReinforcementTradeResult:
    """Результат сделки с дополнительной информацией для RL"""
    symbol: str
    action: str
    entry_price: float
    exit_price: float
    pnl: float
    pnl_percent: float
    confidence: float
    entry_time: datetime
    exit_time: datetime
    duration_minutes: int
    
    # RL специфичные поля
    ai_weights_before: Dict[str, float]
    ai_weights_after: Dict[str, float]
    reward_applied: float
    punishment_applied: float
    
    # Дополнительная информация
    market_conditions: Dict
    reasoning: str

@dataclass
class ReinforcementTestResult:
    """Результат тестирования с обучением с подкреплением"""
    config: ReinforcementTestConfig
    trades: List[ReinforcementTradeResult]
    
    # Общая статистика
    total_trades: int
    profitable_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    total_pnl_percent: float
    
    # Статистика по интервалам
    interval_stats: List[Dict]
    
    # RL статистика
    initial_weights: Dict[str, float]
    final_weights: Dict[str, float]
    weight_evolution: List[Dict]
    learning_progress: Dict
    
    # Производительность
    start_time: datetime
    end_time: datetime
    duration_seconds: float
    
    # Mistral статистика
    mistral_server_stats: Dict

class ReinforcementWinrateTester:
    """
    Расширенный тестер винрейта с поддержкой обучения с подкреплением
    """
    
    def __init__(self, config: ReinforcementTestConfig):
        self.config = config
        self.orchestrator = None
        self.historical_data_manager = None
        self.binance_collector = None
        self.mistral_manager = None
        
        # Результаты
        self.trades: List[ReinforcementTradeResult] = []
        self.interval_stats: List[Dict] = []
        self.weight_evolution: List[Dict] = []
        
        # Статистика
        self.start_time = None
        self.end_time = None
        
    async def initialize(self) -> bool:
        """
        Инициализация всех компонентов
        """
        try:
            logger.info("🚀 Инициализация ReinforcementWinrateTester...")
            
            # Инициализация Mistral сервера
            if self.config.auto_start_mistral:
                await self._initialize_mistral_server()
            
            # Инициализация AI оркестратора с RL
            await self._initialize_ai_orchestrator()
            
            # Инициализация менеджера исторических данных
            await self._initialize_data_managers()
            
            logger.info("✅ ReinforcementWinrateTester инициализирован успешно")
            return True
            
        except Exception as e:
            logger.error(f"❌ Ошибка инициализации: {e}")
            return False
    
    async def _initialize_mistral_server(self):
        """Инициализация и запуск Mistral сервера"""
        try:
            logger.info("🔄 Инициализация Mistral сервера...")
            
            # Создаем конфигурацию для Mistral сервера
            from config.config_manager import MistralServerConfig
            mistral_config = MistralServerConfig(
                model_name=self.config.mistral_model,
                timeout=self.config.mistral_timeout
            )
            
            self.mistral_manager = MistralServerManager(mistral_config)
            
            # Проверяем статус сервера
            status = self.mistral_manager.get_server_status()
            if not status.get('is_running', False):
                logger.info("🚀 Запуск Mistral сервера...")
                success = await self.mistral_manager.start_server()
                if not success:
                    raise Exception("Не удалось запустить Mistral сервер")
                
                # Ждем запуска
                await asyncio.sleep(10)
                
                # Проверяем готовность
                health = await self.mistral_manager.health_check()
                if not health.get('is_running', False):
                    raise Exception("Mistral сервер не прошел проверку здоровья")
            
            logger.info("✅ Mistral сервер готов к работе")
            
        except Exception as e:
            logger.error(f"❌ Ошибка инициализации Mistral сервера: {e}")
            raise
    
    async def _initialize_ai_orchestrator(self):
        """Инициализация AI оркестратора с поддержкой RL"""
        try:
            logger.info("🤖 Инициализация AI оркестратора с RL...")
            
            # Создаем конфигурацию RL
            rl_config = ReinforcementConfig(
                learning_rate=self.config.learning_rate,
                reward_multiplier=self.config.reward_multiplier,
                punishment_multiplier=self.config.punishment_multiplier,
                weight_decay=self.config.weight_decay
            )
            
            # Инициализируем оркестратор с RL
            self.orchestrator = MultiAIOrchestrator(
                backtest_mode=True,
                reinforcement_learning=True
            )
            
            await self.orchestrator.initialize()
            
            # Сохраняем начальные веса
            initial_weights = self.orchestrator.get_reinforcement_learning_stats()
            self.weight_evolution.append({
                'timestamp': datetime.now(),
                'weights': initial_weights.get('current_weights', {}),
                'trade_count': 0,
                'win_rate': 0.0
            })
            
            logger.info("✅ AI оркестратор с RL инициализирован")
            
        except Exception as e:
            logger.error(f"❌ Ошибка инициализации AI оркестратора: {e}")
            raise
    
    async def _initialize_data_managers(self):
        """Инициализация менеджеров данных"""
        try:
            logger.info("📊 Инициализация менеджеров данных...")
            
            self.historical_data_manager = HistoricalDataManager()
            self.binance_collector = BinanceDataCollector()
            
            logger.info("✅ Менеджеры данных инициализированы")
            
        except Exception as e:
            logger.error(f"❌ Ошибка инициализации менеджеров данных: {e}")
            raise
    
    async def run_reinforcement_learning_test(self) -> ReinforcementTestResult:
        """
        Запуск тестирования с обучением с подкреплением
        """
        try:
            self.start_time = datetime.now()
            logger.info(f"🎯 Начало тестирования с RL: {self.start_time}")
            
            # Загружаем исторические данные
            await self._load_historical_data()
            
            # Разбиваем данные на интервалы для обучения
            intervals = self._create_learning_intervals()
            
            # Проходим по каждому интервалу
            for i, interval in enumerate(intervals):
                logger.info(f"📈 Интервал {i+1}/{len(intervals)}: {interval['start']} - {interval['end']}")
                
                # Тестируем на интервале
                interval_result = await self._test_interval(interval, i)
                self.interval_stats.append(interval_result)
                
                # Обновляем эволюцию весов
                current_stats = self.orchestrator.get_reinforcement_learning_stats()
                self.weight_evolution.append({
                    'timestamp': datetime.now(),
                    'weights': current_stats.get('current_weights', {}),
                    'trade_count': len(self.trades),
                    'win_rate': current_stats.get('win_rate', 0.0),
                    'interval': i + 1
                })
                
                logger.info(f"✅ Интервал {i+1} завершен. Винрейт: {interval_result['win_rate']:.2%}")
            
            # Создаем итоговый результат
            result = await self._create_final_result()
            
            # Сохраняем результаты
            if self.config.save_results:
                await self._save_results(result)
            
            self.end_time = datetime.now()
            logger.info(f"🏁 Тестирование завершено: {self.end_time}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Ошибка тестирования с RL: {e}")
            raise
        finally:
            # Останавливаем Mistral сервер если запускали
            if self.config.auto_start_mistral and self.mistral_manager:
                self.mistral_manager.stop_server()
    
    async def _load_historical_data(self):
        """Загрузка исторических данных"""
        try:
            logger.info("📥 Загрузка исторических данных...")
            
            for symbol in self.config.symbols:
                logger.info(f"📊 Загрузка данных для {symbol}...")
                
                # Пытаемся загрузить из кэша
                data = await self.historical_data_manager.get_historical_data(
                    symbol=symbol,
                    start_date=self.config.start_date,
                    end_date=self.config.end_date,
                    interval='1h'
                )
                
                if data is None or len(data) == 0:
                    # Загружаем с Binance
                    logger.info(f"🌐 Загрузка {symbol} с Binance...")
                    data = await self.binance_collector.get_historical_klines(
                        symbol=symbol,
                        interval='1h',
                        start_str=self.config.start_date,
                        end_str=self.config.end_date
                    )
                    
                    if data is not None:
                        await self.historical_data_manager.save_historical_data(symbol, data)
                
                logger.info(f"✅ Загружено {len(data) if data is not None else 0} записей для {symbol}")
            
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки данных: {e}")
            raise
    
    def _create_learning_intervals(self) -> List[Dict]:
        """Создание интервалов для обучения"""
        try:
            start_date = datetime.strptime(self.config.start_date, '%Y-%m-%d')
            end_date = datetime.strptime(self.config.end_date, '%Y-%m-%d')
            
            total_days = (end_date - start_date).days
            days_per_interval = total_days // self.config.test_intervals
            
            intervals = []
            current_start = start_date
            
            for i in range(self.config.test_intervals):
                current_end = current_start + timedelta(days=days_per_interval)
                if i == self.config.test_intervals - 1:  # Последний интервал
                    current_end = end_date
                
                intervals.append({
                    'start': current_start.strftime('%Y-%m-%d'),
                    'end': current_end.strftime('%Y-%m-%d'),
                    'interval_id': i
                })
                
                current_start = current_end
            
            logger.info(f"📅 Создано {len(intervals)} интервалов для обучения")
            return intervals
            
        except Exception as e:
            logger.error(f"❌ Ошибка создания интервалов: {e}")
            raise
    
    async def _test_interval(self, interval: Dict, interval_id: int) -> Dict:
        """Тестирование одного интервала"""
        try:
            interval_trades = []
            interval_start_time = datetime.now()
            
            # Получаем веса до интервала
            weights_before = self.orchestrator.get_reinforcement_learning_stats().get('current_weights', {})
            
            # Генерируем сигналы для интервала
            signals = await self._generate_signals_for_interval(interval)
            
            # Ограничиваем количество сделок
            if len(signals) > self.config.trades_per_interval:
                signals = signals[:self.config.trades_per_interval]
            
            # Обрабатываем каждый сигнал
            for signal_data in signals:
                trade_result = await self._process_signal(signal_data, weights_before)
                if trade_result:
                    interval_trades.append(trade_result)
                    self.trades.append(trade_result)
            
            # Получаем веса после интервала
            weights_after = self.orchestrator.get_reinforcement_learning_stats().get('current_weights', {})
            
            # Статистика интервала
            profitable_trades = len([t for t in interval_trades if t.pnl > 0])
            total_pnl = sum(t.pnl for t in interval_trades)
            win_rate = profitable_trades / len(interval_trades) if interval_trades else 0
            
            interval_result = {
                'interval_id': interval_id,
                'start_date': interval['start'],
                'end_date': interval['end'],
                'total_trades': len(interval_trades),
                'profitable_trades': profitable_trades,
                'losing_trades': len(interval_trades) - profitable_trades,
                'win_rate': win_rate,
                'total_pnl': total_pnl,
                'weights_before': weights_before,
                'weights_after': weights_after,
                'duration_seconds': (datetime.now() - interval_start_time).total_seconds()
            }
            
            return interval_result
            
        except Exception as e:
            logger.error(f"❌ Ошибка тестирования интервала {interval_id}: {e}")
            raise
    
    async def _generate_signals_for_interval(self, interval: Dict) -> List[Dict]:
        """Генерация сигналов для интервала"""
        try:
            signals = []
            
            for symbol in self.config.symbols:
                # Получаем данные для интервала
                data = await self.historical_data_manager.get_historical_data(
                    symbol=symbol,
                    start_date=interval['start'],
                    end_date=interval['end'],
                    interval='1h'
                )
                
                if data is None or len(data) < 24:  # Минимум 24 часа данных
                    continue
                
                # Генерируем сигналы каждые 4 часа
                for i in range(0, len(data), 4):
                    if i + 24 > len(data):  # Нужно минимум 24 часа для анализа
                        break
                    
                    current_data = data.iloc[i:i+24]
                    current_price = current_data.iloc[-1]['close']
                    timestamp = current_data.iloc[-1]['timestamp']
                    
                    # Создаем market_data для анализа
                    market_data = {
                        'price': current_price,
                        'volume': current_data['volume'].mean(),
                        'high_24h': current_data['high'].max(),
                        'low_24h': current_data['low'].min(),
                        'price_change_24h': ((current_price - current_data.iloc[0]['close']) / current_data.iloc[0]['close']) * 100,
                        'timestamp': timestamp
                    }
                    
                    signals.append({
                        'symbol': symbol,
                        'market_data': market_data,
                        'historical_data': current_data,
                        'timestamp': timestamp
                    })
            
            return signals
            
        except Exception as e:
            logger.error(f"❌ Ошибка генерации сигналов: {e}")
            return []
    
    async def _process_signal(self, signal_data: Dict, weights_before: Dict) -> Optional[ReinforcementTradeResult]:
        """Обработка одного сигнала"""
        try:
            symbol = signal_data['symbol']
            market_data = signal_data['market_data']
            
            # Получаем решение от AI
            decision = await self.orchestrator.analyze_and_decide(symbol, market_data)
            
            if decision.action == 'HOLD':
                return None
            
            # Симулируем сделку
            entry_price = decision.entry_price
            entry_time = datetime.fromtimestamp(market_data['timestamp'] / 1000)
            
            # Симулируем выход (через 4-8 часов)
            exit_hours = np.random.randint(4, 9)
            exit_time = entry_time + timedelta(hours=exit_hours)
            
            # Симулируем цену выхода (случайное изменение ±5%)
            price_change = np.random.uniform(-0.05, 0.05)
            if decision.action == 'LONG':
                exit_price = entry_price * (1 + price_change)
                pnl = (exit_price - entry_price) / entry_price
            else:  # SHORT
                exit_price = entry_price * (1 - price_change)
                pnl = (entry_price - exit_price) / entry_price
            
            pnl_absolute = pnl * 1000  # Предполагаем позицию $1000
            
            # Применяем результат к RL
            self.orchestrator.apply_trade_result(
                symbol=symbol,
                action=decision.action,
                pnl=pnl_absolute,
                confidence=decision.confidence,
                entry_price=entry_price,
                exit_price=exit_price,
                duration_minutes=exit_hours * 60
            )
            
            # Получаем веса после применения результата
            weights_after = self.orchestrator.get_reinforcement_learning_stats().get('current_weights', {})
            
            # Создаем результат сделки
            trade_result = ReinforcementTradeResult(
                symbol=symbol,
                action=decision.action,
                entry_price=entry_price,
                exit_price=exit_price,
                pnl=pnl_absolute,
                pnl_percent=pnl * 100,
                confidence=decision.confidence,
                entry_time=entry_time,
                exit_time=exit_time,
                duration_minutes=exit_hours * 60,
                ai_weights_before=weights_before.copy(),
                ai_weights_after=weights_after.copy(),
                reward_applied=pnl_absolute if pnl_absolute > 0 else 0,
                punishment_applied=abs(pnl_absolute) if pnl_absolute <= 0 else 0,
                market_conditions=market_data.copy(),
                reasoning=decision.reasoning
            )
            
            return trade_result
            
        except Exception as e:
            logger.error(f"❌ Ошибка обработки сигнала: {e}")
            return None
    
    async def _create_final_result(self) -> ReinforcementTestResult:
        """Создание итогового результата"""
        try:
            # Общая статистика
            total_trades = len(self.trades)
            profitable_trades = len([t for t in self.trades if t.pnl > 0])
            losing_trades = total_trades - profitable_trades
            win_rate = profitable_trades / total_trades if total_trades > 0 else 0
            total_pnl = sum(t.pnl for t in self.trades)
            total_pnl_percent = (total_pnl / self.config.initial_balance) * 100
            
            # RL статистика
            initial_weights = self.weight_evolution[0]['weights'] if self.weight_evolution else {}
            final_weights = self.weight_evolution[-1]['weights'] if self.weight_evolution else {}
            
            rl_stats = self.orchestrator.get_reinforcement_learning_stats()
            learning_progress = {
                'total_rewards': sum(t.reward_applied for t in self.trades),
                'total_punishments': sum(t.punishment_applied for t in self.trades),
                'weight_changes': len([w for w in self.weight_evolution if w['weights'] != initial_weights]),
                'performance_improvement': win_rate - (self.interval_stats[0]['win_rate'] if self.interval_stats else 0)
            }
            
            # Mistral статистика
            mistral_stats = {}
            if self.mistral_manager:
                mistral_stats = self.mistral_manager.get_server_status()
            
            result = ReinforcementTestResult(
                config=self.config,
                trades=self.trades,
                total_trades=total_trades,
                profitable_trades=profitable_trades,
                losing_trades=losing_trades,
                win_rate=win_rate,
                total_pnl=total_pnl,
                total_pnl_percent=total_pnl_percent,
                interval_stats=self.interval_stats,
                initial_weights=initial_weights,
                final_weights=final_weights,
                weight_evolution=self.weight_evolution,
                learning_progress=learning_progress,
                start_time=self.start_time,
                end_time=self.end_time,
                duration_seconds=(self.end_time - self.start_time).total_seconds(),
                mistral_server_stats=mistral_stats
            )
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Ошибка создания итогового результата: {e}")
            raise
    
    async def _save_results(self, result: ReinforcementTestResult):
        """Сохранение результатов"""
        try:
            import os
            
            # Создаем директорию если не существует
            os.makedirs(self.config.results_dir, exist_ok=True)
            
            # Имя файла
            session_name = self.config.session_name or f"rl_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            filename = f"{session_name}.json"
            filepath = os.path.join(self.config.results_dir, filename)
            
            # Конвертируем результат в словарь
            result_dict = asdict(result)
            
            # Конвертируем datetime объекты в строки
            def convert_datetime(obj):
                if isinstance(obj, datetime):
                    return obj.isoformat()
                elif isinstance(obj, dict):
                    return {k: convert_datetime(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_datetime(item) for item in obj]
                return obj
            
            result_dict = convert_datetime(result_dict)
            
            # Сохраняем
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(result_dict, f, indent=2, ensure_ascii=False)
            
            logger.info(f"💾 Результаты сохранены: {filepath}")
            
            # Сохраняем сессию RL
            if self.orchestrator:
                self.orchestrator.save_reinforcement_learning_session(session_name)
            
        except Exception as e:
            logger.error(f"❌ Ошибка сохранения результатов: {e}")

# Пример использования
async def main():
    """Пример запуска тестирования"""
    config = ReinforcementTestConfig(
        symbols=['BTCUSDT', 'ETHUSDT'],
        start_date='2024-01-01',
        end_date='2024-01-31',
        initial_balance=10000.0,
        test_intervals=5,
        trades_per_interval=20,
        learning_rate=0.01,
        session_name='test_session_btc_eth'
    )
    
    tester = ReinforcementWinrateTester(config)
    
    if await tester.initialize():
        result = await tester.run_reinforcement_learning_test()
        
        print(f"\n🎯 Результаты тестирования:")
        print(f"📊 Всего сделок: {result.total_trades}")
        print(f"✅ Прибыльных: {result.profitable_trades}")
        print(f"❌ Убыточных: {result.losing_trades}")
        print(f"📈 Винрейт: {result.win_rate:.2%}")
        print(f"💰 Общий PnL: ${result.total_pnl:.2f} ({result.total_pnl_percent:.2f}%)")
        print(f"⏱️ Время тестирования: {result.duration_seconds:.0f} секунд")
        
        print(f"\n🧠 Эволюция весов AI:")
        print(f"🔸 Начальные веса: {result.initial_weights}")
        print(f"🔸 Финальные веса: {result.final_weights}")

if __name__ == "__main__":
    asyncio.run(main())