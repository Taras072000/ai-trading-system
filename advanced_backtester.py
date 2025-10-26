"""
Продвинутая система бэктестинга для Trading AI
Включает детальную аналитику, визуализацию и оценку рисков
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)

class TradeType(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"

class TradeStatus(Enum):
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    CANCELLED = "CANCELLED"

@dataclass
class Trade:
    """Класс для представления сделки"""
    id: int
    symbol: str
    trade_type: TradeType
    entry_price: float
    entry_time: datetime
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    quantity: float = 0.0
    status: TradeStatus = TradeStatus.OPEN
    confidence: float = 0.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    commission: float = 0.0
    pnl: float = 0.0
    pnl_pct: float = 0.0
    duration_hours: float = 0.0
    max_drawdown: float = 0.0
    max_profit: float = 0.0
    
    def close_trade(self, exit_price: float, exit_time: datetime, commission: float = 0.0):
        """Закрытие сделки"""
        self.exit_price = exit_price
        self.exit_time = exit_time
        self.status = TradeStatus.CLOSED
        self.commission = commission
        
        # Расчет P&L
        if self.trade_type == TradeType.BUY:
            self.pnl = (exit_price - self.entry_price) * self.quantity - commission
        else:  # SELL
            self.pnl = (self.entry_price - exit_price) * self.quantity - commission
        
        self.pnl_pct = (self.pnl / (self.entry_price * self.quantity)) * 100
        
        # Расчет длительности
        if self.exit_time and self.entry_time:
            self.duration_hours = (self.exit_time - self.entry_time).total_seconds() / 3600
    
    def is_profitable(self) -> bool:
        """Проверка прибыльности сделки"""
        return self.pnl > 0
    
    def to_dict(self) -> Dict:
        """Конвертация в словарь"""
        result = asdict(self)
        result['trade_type'] = self.trade_type.value
        result['status'] = self.status.value
        result['entry_time'] = self.entry_time.isoformat() if self.entry_time else None
        result['exit_time'] = self.exit_time.isoformat() if self.exit_time else None
        return result

@dataclass
class BacktestResults:
    """Результаты бэктестинга"""
    # Основные метрики
    initial_balance: float
    final_balance: float
    total_return: float
    total_return_pct: float
    
    # Торговые метрики
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    
    # Финансовые метрики
    profit_factor: float
    sharpe_ratio: float
    max_drawdown: float
    max_drawdown_pct: float
    
    # Временные метрики
    avg_trade_duration: float
    avg_winning_trade: float
    avg_losing_trade: float
    
    # Дополнительные метрики
    largest_win: float
    largest_loss: float
    consecutive_wins: int
    consecutive_losses: int
    
    # Данные для анализа
    trades: List[Trade]
    equity_curve: pd.DataFrame
    monthly_returns: pd.DataFrame

class AdvancedBacktester:
    """Продвинутая система бэктестинга"""
    
    def __init__(self, initial_balance: float = 10000, commission: float = 0.001):
        self.initial_balance = initial_balance
        self.commission = commission  # Комиссия в долях (0.001 = 0.1%)
        self.trades: List[Trade] = []
        self.equity_curve: List[Dict] = []
        self.current_balance = initial_balance
        self.current_position = 0.0
        self.trade_counter = 0
        
        # Настройки риск-менеджмента
        self.max_position_size = 0.95  # Максимальный размер позиции (95% от баланса)
        self.stop_loss_pct = 0.02  # Стоп-лосс 2%
        self.take_profit_pct = 0.04  # Тейк-профит 4%
        
    def run_backtest(self, data: pd.DataFrame, signals: pd.DataFrame, 
                    symbol: str = "BTCUSDT") -> BacktestResults:
        """
        Запуск бэктестинга
        
        Args:
            data: Исторические данные (OHLCV)
            signals: Торговые сигналы с колонками ['action', 'confidence']
            symbol: Торговый символ
        """
        logger.info(f"Запуск бэктестинга для {symbol}")
        logger.info(f"Данные: {len(data)} свечей, сигналы: {len(signals)} записей")
        
        # Сброс состояния
        self._reset_state()
        
        # Объединяем данные и сигналы
        combined_data = self._prepare_data(data, signals)
        
        # Основной цикл бэктестинга
        for i, row in combined_data.iterrows():
            self._process_bar(row, symbol)
        
        # Закрываем оставшиеся позиции
        self._close_remaining_positions(combined_data.iloc[-1], symbol)
        
        # Анализируем результаты
        results = self._analyze_results(symbol)
        
        logger.info(f"Бэктестинг завершен. Винрейт: {results.win_rate:.2%}")
        
        return results
    
    def _reset_state(self):
        """Сброс состояния бэктестера"""
        self.trades = []
        self.equity_curve = []
        self.current_balance = self.initial_balance
        self.current_position = 0.0
        self.trade_counter = 0
    
    def _prepare_data(self, data: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
        """Подготовка данных для бэктестинга"""
        # Убеждаемся, что индексы совпадают
        min_len = min(len(data), len(signals))
        data_subset = data.iloc[:min_len].copy()
        signals_subset = signals.iloc[:min_len].copy()
        
        # Объединяем данные
        combined = pd.concat([data_subset.reset_index(drop=True), 
                             signals_subset.reset_index(drop=True)], axis=1)
        
        return combined
    
    def _process_bar(self, row: pd.Series, symbol: str):
        """Обработка одной свечи"""
        current_time = row['timestamp'] if 'timestamp' in row else datetime.now()
        current_price = row['close']
        action = row.get('action', 'HOLD')
        confidence = row.get('confidence', 0.0)
        
        # Обновляем equity curve
        self._update_equity_curve(current_time, current_price)
        
        # Проверяем стоп-лоссы и тейк-профиты
        self._check_exit_conditions(row, symbol)
        
        # Обрабатываем новые сигналы
        if action in ['BUY', 'SELL'] and confidence > 0.6:
            self._execute_signal(row, symbol, action, confidence)
    
    def _execute_signal(self, row: pd.Series, symbol: str, action: str, confidence: float):
        """Выполнение торгового сигнала"""
        current_price = row['close']
        current_time = row['timestamp'] if 'timestamp' in row else datetime.now()
        
        # Закрываем противоположную позицию если есть
        if self.current_position != 0:
            if (action == 'BUY' and self.current_position < 0) or \
               (action == 'SELL' and self.current_position > 0):
                self._close_current_position(current_price, current_time, symbol)
        
        # Открываем новую позицию
        if self.current_position == 0:
            self._open_position(row, symbol, action, confidence)
    
    def _open_position(self, row: pd.Series, symbol: str, action: str, confidence: float):
        """Открытие новой позиции"""
        current_price = row['close']
        current_time = row['timestamp'] if 'timestamp' in row else datetime.now()
        
        # Рассчитываем размер позиции
        available_balance = self.current_balance * self.max_position_size
        commission_cost = available_balance * self.commission
        position_value = available_balance - commission_cost
        quantity = position_value / current_price
        
        if action == 'SELL':
            quantity = -quantity
        
        # Создаем сделку
        trade = Trade(
            id=self.trade_counter,
            symbol=symbol,
            trade_type=TradeType.BUY if action == 'BUY' else TradeType.SELL,
            entry_price=current_price,
            entry_time=current_time,
            quantity=abs(quantity),
            confidence=confidence,
            stop_loss=self._calculate_stop_loss(current_price, action),
            take_profit=self._calculate_take_profit(current_price, action)
        )
        
        self.trades.append(trade)
        self.current_position = quantity
        self.current_balance -= commission_cost
        self.trade_counter += 1
        
        logger.debug(f"Открыта позиция: {action} {quantity:.6f} по цене {current_price}")
    
    def _close_current_position(self, exit_price: float, exit_time: datetime, symbol: str):
        """Закрытие текущей позиции"""
        if self.current_position == 0:
            return
        
        # Находим последнюю открытую сделку
        open_trades = [t for t in self.trades if t.status == TradeStatus.OPEN]
        if not open_trades:
            return
        
        trade = open_trades[-1]
        commission_cost = abs(self.current_position) * exit_price * self.commission
        
        # Закрываем сделку
        trade.close_trade(exit_price, exit_time, commission_cost)
        
        # Обновляем баланс
        self.current_balance += trade.pnl
        self.current_position = 0.0
        
        logger.debug(f"Закрыта позиция: P&L = {trade.pnl:.2f} ({trade.pnl_pct:.2f}%)")
    
    def _check_exit_conditions(self, row: pd.Series, symbol: str):
        """Проверка условий выхода (стоп-лосс, тейк-профит)"""
        if self.current_position == 0:
            return
        
        current_price = row['close']
        current_time = row['timestamp'] if 'timestamp' in row else datetime.now()
        
        # Находим текущую открытую сделку
        open_trades = [t for t in self.trades if t.status == TradeStatus.OPEN]
        if not open_trades:
            return
        
        trade = open_trades[-1]
        
        # Проверяем стоп-лосс
        if trade.stop_loss:
            if (trade.trade_type == TradeType.BUY and current_price <= trade.stop_loss) or \
               (trade.trade_type == TradeType.SELL and current_price >= trade.stop_loss):
                self._close_current_position(current_price, current_time, symbol)
                return
        
        # Проверяем тейк-профит
        if trade.take_profit:
            if (trade.trade_type == TradeType.BUY and current_price >= trade.take_profit) or \
               (trade.trade_type == TradeType.SELL and current_price <= trade.take_profit):
                self._close_current_position(current_price, current_time, symbol)
                return
    
    def _calculate_stop_loss(self, entry_price: float, action: str) -> float:
        """Расчет стоп-лосса"""
        if action == 'BUY':
            return entry_price * (1 - self.stop_loss_pct)
        else:  # SELL
            return entry_price * (1 + self.stop_loss_pct)
    
    def _calculate_take_profit(self, entry_price: float, action: str) -> float:
        """Расчет тейк-профита"""
        if action == 'BUY':
            return entry_price * (1 + self.take_profit_pct)
        else:  # SELL
            return entry_price * (1 - self.take_profit_pct)
    
    def _update_equity_curve(self, timestamp: datetime, current_price: float):
        """Обновление кривой капитала"""
        # Рассчитываем текущую стоимость портфеля
        portfolio_value = self.current_balance
        
        if self.current_position != 0:
            position_value = self.current_position * current_price
            portfolio_value += position_value
        
        self.equity_curve.append({
            'timestamp': timestamp,
            'balance': self.current_balance,
            'position_value': self.current_position * current_price if self.current_position != 0 else 0,
            'total_value': portfolio_value,
            'drawdown': (portfolio_value - self.initial_balance) / self.initial_balance
        })
    
    def _close_remaining_positions(self, last_row: pd.Series, symbol: str):
        """Закрытие оставшихся позиций в конце бэктестинга"""
        if self.current_position != 0:
            exit_price = last_row['close']
            exit_time = last_row['timestamp'] if 'timestamp' in last_row else datetime.now()
            self._close_current_position(exit_price, exit_time, symbol)
    
    def _analyze_results(self, symbol: str) -> BacktestResults:
        """Анализ результатов бэктестинга"""
        # Базовые метрики
        final_balance = self.current_balance
        total_return = final_balance - self.initial_balance
        total_return_pct = (total_return / self.initial_balance) * 100
        
        # Торговые метрики
        closed_trades = [t for t in self.trades if t.status == TradeStatus.CLOSED]
        total_trades = len(closed_trades)
        winning_trades = len([t for t in closed_trades if t.is_profitable()])
        losing_trades = total_trades - winning_trades
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Финансовые метрики
        profits = [t.pnl for t in closed_trades if t.is_profitable()]
        losses = [abs(t.pnl) for t in closed_trades if not t.is_profitable()]
        
        profit_factor = sum(profits) / sum(losses) if losses else float('inf')
        
        # Sharpe ratio (упрощенный расчет)
        returns = [t.pnl_pct for t in closed_trades]
        sharpe_ratio = np.mean(returns) / np.std(returns) if len(returns) > 1 and np.std(returns) > 0 else 0
        
        # Максимальная просадка
        equity_df = pd.DataFrame(self.equity_curve)
        if not equity_df.empty:
            equity_df['cummax'] = equity_df['total_value'].cummax()
            equity_df['drawdown'] = (equity_df['total_value'] - equity_df['cummax']) / equity_df['cummax']
            max_drawdown = equity_df['drawdown'].min()
            max_drawdown_pct = max_drawdown * 100
        else:
            max_drawdown = 0
            max_drawdown_pct = 0
        
        # Временные метрики
        durations = [t.duration_hours for t in closed_trades if t.duration_hours > 0]
        avg_trade_duration = np.mean(durations) if durations else 0
        
        avg_winning_trade = np.mean(profits) if profits else 0
        avg_losing_trade = np.mean(losses) if losses else 0
        
        # Дополнительные метрики
        largest_win = max(profits) if profits else 0
        largest_loss = max(losses) if losses else 0
        
        # Подсчет последовательных побед/поражений
        consecutive_wins, consecutive_losses = self._calculate_consecutive_trades(closed_trades)
        
        # Месячная доходность
        monthly_returns = self._calculate_monthly_returns(equity_df)
        
        return BacktestResults(
            initial_balance=self.initial_balance,
            final_balance=final_balance,
            total_return=total_return,
            total_return_pct=total_return_pct,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            profit_factor=profit_factor,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            max_drawdown_pct=max_drawdown_pct,
            avg_trade_duration=avg_trade_duration,
            avg_winning_trade=avg_winning_trade,
            avg_losing_trade=avg_losing_trade,
            largest_win=largest_win,
            largest_loss=largest_loss,
            consecutive_wins=consecutive_wins,
            consecutive_losses=consecutive_losses,
            trades=closed_trades,
            equity_curve=equity_df,
            monthly_returns=monthly_returns
        )
    
    def _calculate_consecutive_trades(self, trades: List[Trade]) -> Tuple[int, int]:
        """Расчет максимальных последовательных побед и поражений"""
        if not trades:
            return 0, 0
        
        max_wins = 0
        max_losses = 0
        current_wins = 0
        current_losses = 0
        
        for trade in trades:
            if trade.is_profitable():
                current_wins += 1
                current_losses = 0
                max_wins = max(max_wins, current_wins)
            else:
                current_losses += 1
                current_wins = 0
                max_losses = max(max_losses, current_losses)
        
        return max_wins, max_losses
    
    def _calculate_monthly_returns(self, equity_df: pd.DataFrame) -> pd.DataFrame:
        """Расчет месячной доходности"""
        if equity_df.empty:
            return pd.DataFrame()
        
        equity_df = equity_df.copy()
        equity_df['timestamp'] = pd.to_datetime(equity_df['timestamp'])
        equity_df.set_index('timestamp', inplace=True)
        
        # Ресемплируем по месяцам
        monthly = equity_df['total_value'].resample('M').last()
        monthly_returns = monthly.pct_change().dropna()
        
        return pd.DataFrame({
            'month': monthly_returns.index,
            'return_pct': monthly_returns.values * 100
        })
    
    def generate_report(self, results: BacktestResults, save_path: Optional[str] = None) -> str:
        """Генерация детального отчета"""
        report = f"""
=== ОТЧЕТ ПО БЭКТЕСТИНГУ ===

ОСНОВНЫЕ РЕЗУЛЬТАТЫ:
- Начальный баланс: ${results.initial_balance:,.2f}
- Конечный баланс: ${results.final_balance:,.2f}
- Общая прибыль: ${results.total_return:,.2f} ({results.total_return_pct:.2f}%)

ТОРГОВЫЕ МЕТРИКИ:
- Всего сделок: {results.total_trades}
- Прибыльных сделок: {results.winning_trades}
- Убыточных сделок: {results.losing_trades}
- Винрейт: {results.win_rate:.2%}

ФИНАНСОВЫЕ МЕТРИКИ:
- Profit Factor: {results.profit_factor:.2f}
- Sharpe Ratio: {results.sharpe_ratio:.2f}
- Максимальная просадка: {results.max_drawdown_pct:.2f}%

ВРЕМЕННЫЕ МЕТРИКИ:
- Средняя длительность сделки: {results.avg_trade_duration:.1f} часов
- Средняя прибыльная сделка: ${results.avg_winning_trade:.2f}
- Средняя убыточная сделка: ${results.avg_losing_trade:.2f}

ЭКСТРЕМАЛЬНЫЕ ЗНАЧЕНИЯ:
- Наибольшая прибыль: ${results.largest_win:.2f}
- Наибольший убыток: ${results.largest_loss:.2f}
- Максимум побед подряд: {results.consecutive_wins}
- Максимум поражений подряд: {results.consecutive_losses}

ОЦЕНКА КАЧЕСТВА:
- Достигает целевой винрейт (75%): {'ДА' if results.win_rate >= 0.75 else 'НЕТ'}
- Profit Factor > 1.5: {'ДА' if results.profit_factor > 1.5 else 'НЕТ'}
- Sharpe Ratio > 1.0: {'ДА' if results.sharpe_ratio > 1.0 else 'НЕТ'}
"""
        
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report)
            logger.info(f"Отчет сохранен в {save_path}")
        
        return report
    
    def plot_results(self, results: BacktestResults, save_path: Optional[str] = None):
        """Создание графиков результатов"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # График кривой капитала
        if not results.equity_curve.empty:
            axes[0, 0].plot(results.equity_curve['timestamp'], results.equity_curve['total_value'])
            axes[0, 0].set_title('Кривая капитала')
            axes[0, 0].set_ylabel('Стоимость портфеля ($)')
            axes[0, 0].grid(True)
        
        # График просадки
        if not results.equity_curve.empty:
            axes[0, 1].fill_between(results.equity_curve['timestamp'], 
                                   results.equity_curve['drawdown'] * 100, 0, alpha=0.3, color='red')
            axes[0, 1].set_title('Просадка')
            axes[0, 1].set_ylabel('Просадка (%)')
            axes[0, 1].grid(True)
        
        # Распределение P&L
        pnl_values = [t.pnl for t in results.trades]
        if pnl_values:
            axes[1, 0].hist(pnl_values, bins=20, alpha=0.7, edgecolor='black')
            axes[1, 0].set_title('Распределение P&L')
            axes[1, 0].set_xlabel('P&L ($)')
            axes[1, 0].set_ylabel('Количество сделок')
            axes[1, 0].grid(True)
        
        # Месячная доходность
        if not results.monthly_returns.empty:
            axes[1, 1].bar(range(len(results.monthly_returns)), results.monthly_returns['return_pct'])
            axes[1, 1].set_title('Месячная доходность')
            axes[1, 1].set_xlabel('Месяц')
            axes[1, 1].set_ylabel('Доходность (%)')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Графики сохранены в {save_path}")
        
        plt.show()

# Пример использования
async def example_usage():
    """Пример использования бэктестера"""
    
    # Создаем синтетические данные
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='1H')
    np.random.seed(42)
    
    prices = 50000 * np.cumprod(1 + np.random.normal(0, 0.02, len(dates)))
    
    data = pd.DataFrame({
        'timestamp': dates,
        'open': prices * 0.999,
        'high': prices * 1.001,
        'low': prices * 0.998,
        'close': prices,
        'volume': np.random.lognormal(10, 1, len(dates))
    })
    
    # Создаем случайные сигналы
    signals = pd.DataFrame({
        'action': np.random.choice(['BUY', 'SELL', 'HOLD'], len(dates), p=[0.1, 0.1, 0.8]),
        'confidence': np.random.uniform(0.5, 1.0, len(dates))
    })
    
    # Запускаем бэктестинг
    backtester = AdvancedBacktester(initial_balance=10000)
    results = backtester.run_backtest(data, signals, "BTCUSDT")
    
    # Генерируем отчет
    report = backtester.generate_report(results)
    print(report)
    
    # Создаем графики
    backtester.plot_results(results)

if __name__ == "__main__":
    import asyncio
    asyncio.run(example_usage())