#!/usr/bin/env python3
"""
Multi-AI Backtester
Бэктестер для тестирования MultiAIOrchestrator на исторических данных
"""

import asyncio
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
import os

# Импорты модулей системы
from ai_modules.multi_ai_orchestrator import MultiAIOrchestrator
from data_collector import BinanceDataCollector, DataManager
from historical_data_manager import HistoricalDataManager

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class BacktestConfig:
    """Конфигурация бэктеста"""
    initial_balance: float = 100.0  # Стартовый баланс в USD
    take_profit_levels: int = 5     # Количество тейк-профитов
    stop_loss_enabled: bool = True  # Использовать стоп-лосс
    position_size_percent: float = 0.1  # Размер позиции в % от баланса
    commission_rate: float = 0.001  # Комиссия 0.1%
    
@dataclass
class Trade:
    """Структура сделки"""
    symbol: str
    entry_time: datetime
    entry_price: float
    direction: str  # 'LONG' или 'SHORT'
    size: float
    stop_loss: float
    take_profits: List[float]
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    pnl: Optional[float] = None
    
@dataclass
class BacktestResult:
    """Результаты бэктеста"""
    symbol: str
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    max_drawdown: float
    sharpe_ratio: float
    final_balance: float
    trades: List[Trade]

class MultiAIBacktester:
    """Бэктестер для MultiAIOrchestrator"""
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.orchestrator = None
        self.data_collector = BinanceDataCollector()
        self.data_manager = DataManager()
        self.historical_manager = HistoricalDataManager()
        
        # Топ-5 валютных пар для тестирования
        self.symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'SOLUSDT']
        
    async def initialize(self):
        """Инициализация всех компонентов"""
        try:
            logger.info("🔧 Инициализация бэктестера...")
            
            # Инициализируем AI оркестратор с режимом бэктестинга
            self.orchestrator = MultiAIOrchestrator(backtest_mode=True)
            await self.orchestrator.initialize()
            
            # Инициализируем менеджер данных
            self.data_manager = DataManager()
            
            logger.info("✅ Бэктестер инициализирован успешно")
            
        except Exception as e:
            logger.error(f"❌ Ошибка инициализации бэктестера: {e}")
            raise
        
    async def load_historical_data(self, symbol: str, days: int = 7) -> pd.DataFrame:
        """Загрузка недельных исторических данных"""
        logger.info(f"📊 Загрузка данных для {symbol} за {days} дней...")
        
        try:
            # Получение данных через DataManager
            async with self.data_collector:
                data = await self.data_collector.get_historical_data(
                    symbol=symbol,
                    interval='1h',
                    days=days
                )
            
            if data is None or len(data) == 0:
                logger.error(f"❌ Не удалось загрузить данные для {symbol}")
                return pd.DataFrame()
                
            logger.info(f"✅ Загружено {len(data)} свечей для {symbol}")
            return data
            
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки данных для {symbol}: {e}")
            return pd.DataFrame()
    
    def calculate_position_size(self, balance: float, price: float) -> float:
        """Расчет размера позиции"""
        position_value = balance * self.config.position_size_percent
        return position_value / price
    
    def calculate_take_profits(self, entry_price: float, direction: str) -> List[float]:
        """Расчет уровней тейк-профитов"""
        take_profits = []
        
        # Оптимизированные уровни для лучшего соотношения риск/прибыль
        tp_percentages = [0.8, 1.2, 2.0, 3.0, 5.0]  # 5 уровней
        
        for i, tp_pct in enumerate(tp_percentages[:self.config.take_profit_levels]):
            if direction == 'LONG':
                tp_price = entry_price * (1 + tp_pct / 100)
            else:  # SHORT
                tp_price = entry_price * (1 - tp_pct / 100)
            take_profits.append(tp_price)
            
        return take_profits
    
    def calculate_stop_loss(self, entry_price: float, direction: str) -> float:
        """Расчет стоп-лосса"""
        sl_percentage = 1.5  # Уменьшен до 1.5% для лучшего соотношения риск/прибыль
        
        if direction == 'LONG':
            return entry_price * (1 - sl_percentage / 100)
        else:  # SHORT
            return entry_price * (1 + sl_percentage / 100)
    
    async def simulate_trade(self, trade: Trade, data: pd.DataFrame, start_idx: int) -> Trade:
        """Симуляция выполнения сделки"""
        
        # Поиск точки выхода из сделки
        for i in range(start_idx + 1, len(data)):
            current_candle = data.iloc[i]
            high = current_candle['high']
            low = current_candle['low']
            close = current_candle['close']
            timestamp = current_candle['timestamp']
            
            # Проверка стоп-лосса
            if self.config.stop_loss_enabled:
                if trade.direction == 'LONG' and low <= trade.stop_loss:
                    trade.exit_time = timestamp
                    trade.exit_price = trade.stop_loss
                    trade.exit_reason = 'Stop Loss'
                    break
                elif trade.direction == 'SHORT' and high >= trade.stop_loss:
                    trade.exit_time = timestamp
                    trade.exit_price = trade.stop_loss
                    trade.exit_reason = 'Stop Loss'
                    break
            
            # Проверка тейк-профитов
            for j, tp_price in enumerate(trade.take_profits):
                if trade.direction == 'LONG' and high >= tp_price:
                    trade.exit_time = timestamp
                    trade.exit_price = tp_price
                    trade.exit_reason = f'Take Profit {j+1}'
                    break
                elif trade.direction == 'SHORT' and low <= tp_price:
                    trade.exit_time = timestamp
                    trade.exit_price = tp_price
                    trade.exit_reason = f'Take Profit {j+1}'
                    break
            
            if trade.exit_time:
                break
        
        # Если сделка не закрылась, закрываем по последней цене
        if not trade.exit_time:
            last_candle = data.iloc[-1]
            trade.exit_time = last_candle['timestamp']
            trade.exit_price = last_candle['close']
            trade.exit_reason = 'End of Data'
        
        # Расчет PnL
        if trade.direction == 'LONG':
            trade.pnl = (trade.exit_price - trade.entry_price) * trade.size
        else:  # SHORT
            trade.pnl = (trade.entry_price - trade.exit_price) * trade.size
        
        # Учет комиссии
        commission = (trade.entry_price + trade.exit_price) * trade.size * self.config.commission_rate
        trade.pnl -= commission
        
        return trade
    
    async def backtest_symbol(self, symbol: str) -> BacktestResult:
        """Бэктест для одной валютной пары"""
        logger.info(f"🔄 Начинаю бэктест для {symbol}...")
        
        # Загрузка данных
        data = await self.load_historical_data(symbol, days=30)  # Изменено на месячный период
        if data.empty:
            return BacktestResult(
                symbol=symbol, total_trades=0, winning_trades=0, losing_trades=0,
                win_rate=0, total_pnl=0, max_drawdown=0, sharpe_ratio=0,
                final_balance=self.config.initial_balance, trades=[]
            )
        
        trades = []
        balance = self.config.initial_balance
        peak_balance = balance
        max_drawdown = 0
        
        # Проход по данным с шагом в 4 часа для поиска сигналов
        step = 4  # Анализируем каждые 4 часа
        
        for i in range(100, len(data) - 24, step):  # Оставляем место для выхода из сделки
            try:
                # Подготовка данных для анализа
                current_data = data.iloc[:i+1].copy()
                
                # Получение сигнала от AI системы
                decision = await self.orchestrator.analyze_and_decide(symbol, current_data)
                
                if not decision or decision.action in ['HOLD', 'WAIT']:
                    continue
                
                action = decision.action
                confidence = decision.confidence
                
                # Фильтр по уверенности (минимум 45%)
                if confidence < 0.45:
                    continue
                
                # Создание сделки
                current_candle = data.iloc[i]
                entry_price = current_candle['close']
                
                # Расчет размера позиции
                position_size = self.calculate_position_size(balance, entry_price)
                
                # Создание объекта сделки
                trade = Trade(
                    symbol=symbol,
                    entry_time=current_candle['timestamp'],
                    entry_price=entry_price,
                    direction=action,
                    size=position_size,
                    stop_loss=self.calculate_stop_loss(entry_price, action),
                    take_profits=self.calculate_take_profits(entry_price, action)
                )
                
                # Симуляция выполнения сделки
                completed_trade = await self.simulate_trade(trade, data, i)
                trades.append(completed_trade)
                
                # Обновление баланса
                balance += completed_trade.pnl
                
                # Отслеживание максимальной просадки
                if balance > peak_balance:
                    peak_balance = balance
                
                current_drawdown = (peak_balance - balance) / peak_balance
                if current_drawdown > max_drawdown:
                    max_drawdown = current_drawdown
                
                logger.info(f"💰 {symbol}: Сделка {len(trades)} - {completed_trade.exit_reason}, PnL: ${completed_trade.pnl:.2f}, Баланс: ${balance:.2f}")
                
            except Exception as e:
                logger.error(f"❌ Ошибка при анализе {symbol} на индексе {i}: {e}")
                continue
        
        # Расчет статистики
        winning_trades = len([t for t in trades if t.pnl > 0])
        losing_trades = len([t for t in trades if t.pnl <= 0])
        win_rate = winning_trades / len(trades) if trades else 0
        total_pnl = sum(t.pnl for t in trades)
        
        # Расчет Sharpe ratio (упрощенный)
        if trades:
            returns = [t.pnl / self.config.initial_balance for t in trades]
            sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
        else:
            sharpe_ratio = 0
        
        result = BacktestResult(
            symbol=symbol,
            total_trades=len(trades),
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            total_pnl=total_pnl,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            final_balance=balance,
            trades=trades
        )
        
        logger.info(f"✅ Бэктест {symbol} завершен: {len(trades)} сделок, Win Rate: {win_rate:.1%}, PnL: ${total_pnl:.2f}")
        
        return result
    
    async def run_full_backtest(self) -> Dict[str, BacktestResult]:
        """Запуск полного бэктеста по всем парам"""
        logger.info("🚀 Запуск полного бэктеста Multi-AI системы...")
        
        await self.initialize()
        
        results = {}
        
        for symbol in self.symbols:
            try:
                result = await self.backtest_symbol(symbol)
                results[symbol] = result
            except Exception as e:
                logger.error(f"❌ Ошибка бэктеста для {symbol}: {e}")
                continue
        
        return results
    
    def generate_report(self, results: Dict[str, BacktestResult]) -> str:
        """Генерация отчета по результатам"""
        report = []
        report.append("=" * 80)
        report.append("📊 ОТЧЕТ ПО БЭКТЕСТУ MULTI-AI СИСТЕМЫ")
        report.append("=" * 80)
        report.append(f"Период тестирования: 7 дней")
        report.append(f"Стартовый баланс: ${self.config.initial_balance}")
        report.append(f"Валютные пары: {', '.join(self.symbols)}")
        report.append("")
        
        total_pnl = 0
        total_trades = 0
        total_winning = 0
        
        for symbol, result in results.items():
            if result.total_trades == 0:
                continue
                
            report.append(f"🔸 {symbol}:")
            report.append(f"   Сделок: {result.total_trades}")
            report.append(f"   Win Rate: {result.win_rate:.1%}")
            report.append(f"   PnL: ${result.total_pnl:.2f}")
            report.append(f"   Финальный баланс: ${result.final_balance:.2f}")
            report.append(f"   Макс. просадка: {result.max_drawdown:.1%}")
            report.append(f"   Sharpe Ratio: {result.sharpe_ratio:.2f}")
            report.append("")
            
            total_pnl += result.total_pnl
            total_trades += result.total_trades
            total_winning += result.winning_trades
        
        # Общая статистика
        overall_win_rate = total_winning / total_trades if total_trades > 0 else 0
        
        report.append("=" * 50)
        report.append("📈 ОБЩАЯ СТАТИСТИКА:")
        report.append(f"Всего сделок: {total_trades}")
        report.append(f"Общий Win Rate: {overall_win_rate:.1%}")
        report.append(f"Общий PnL: ${total_pnl:.2f}")
        report.append(f"ROI: {(total_pnl / self.config.initial_balance) * 100:.1f}%")
        report.append("=" * 50)
        
        return "\n".join(report)

async def main():
    """Главная функция"""
    
    # Конфигурация бэктеста
    config = BacktestConfig(
        initial_balance=100.0,
        take_profit_levels=5,
        stop_loss_enabled=True,
        position_size_percent=0.1,
        commission_rate=0.001
    )
    
    # Создание и запуск бэктестера
    backtester = MultiAIBacktester(config)
    
    try:
        # Запуск бэктеста
        results = await backtester.run_full_backtest()
        
        # Генерация отчета
        report = backtester.generate_report(results)
        print(report)
        
        # Сохранение результатов
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Сохранение детального отчета
        with open(f"backtest_results_{timestamp}.txt", "w", encoding="utf-8") as f:
            f.write(report)
        
        # Сохранение данных в JSON
        json_results = {}
        for symbol, result in results.items():
            json_results[symbol] = {
                'total_trades': result.total_trades,
                'winning_trades': result.winning_trades,
                'losing_trades': result.losing_trades,
                'win_rate': result.win_rate,
                'total_pnl': result.total_pnl,
                'max_drawdown': result.max_drawdown,
                'sharpe_ratio': result.sharpe_ratio,
                'final_balance': result.final_balance,
                'trades': [
                    {
                        'entry_time': trade.entry_time.isoformat(),
                        'entry_price': trade.entry_price,
                        'direction': trade.direction,
                        'exit_time': trade.exit_time.isoformat() if trade.exit_time else None,
                        'exit_price': trade.exit_price,
                        'exit_reason': trade.exit_reason,
                        'pnl': trade.pnl
                    }
                    for trade in result.trades
                ]
            }
        
        with open(f"backtest_data_{timestamp}.json", "w", encoding="utf-8") as f:
            json.dump(json_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Результаты сохранены в backtest_results_{timestamp}.txt и backtest_data_{timestamp}.json")
        
    except Exception as e:
        logger.error(f"❌ Ошибка выполнения бэктеста: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())