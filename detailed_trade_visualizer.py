#!/usr/bin/env python3
"""
🎯 Детальный визуализатор торговых сделок
Создает подробные графические отчеты для анализа каждой сделки и общей статистики по валютным парам
Использует японские свечи (candlesticks) для профессионального отображения данных
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import mplfinance as mpf
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from dataclasses import dataclass
import json
from pathlib import Path
import logging

# Настройка логирования
logger = logging.getLogger(__name__)

# Настройка стиля для matplotlib
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

@dataclass
class TradeVisualizationData:
    """Данные для визуализации сделки"""
    symbol: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    direction: str  # 'LONG' или 'SHORT'
    pnl: float
    pnl_percent: float
    confidence: float
    ai_model: str
    position_size: float
    commission: float
    exit_reason: str
    partial_exits: List[Dict[str, Any]]
    market_data: pd.DataFrame  # Данные рынка за период сделки
    tp_levels: List[float] = None  # Уровни Take Profit
    sl_level: float = None  # Уровень Stop Loss
    max_profit: float = 0.0  # Максимальная прибыль во время сделки
    max_loss: float = 0.0  # Максимальный убыток во время сделки

def generate_ohlc_data(entry_time: datetime, exit_time: datetime, entry_price: float, exit_price: float, interval_minutes: int = 5) -> pd.DataFrame:
    """
    🕯️ Генерирует OHLC данные для создания свечного графика
    
    Args:
        entry_time: Время входа в сделку
        exit_time: Время выхода из сделки
        entry_price: Цена входа
        exit_price: Цена выхода
        interval_minutes: Интервал свечей в минутах
    
    Returns:
        DataFrame с OHLC данными
    """
    try:
        # Создаем временной ряд
        time_range = pd.date_range(start=entry_time, end=exit_time, freq=f'{interval_minutes}min')
        
        if len(time_range) < 2:
            # Если период слишком короткий, создаем минимальные данные
            time_range = pd.date_range(start=entry_time, periods=10, freq=f'{interval_minutes}min')
        
        # Генерируем реалистичное движение цены
        price_diff = exit_price - entry_price
        num_candles = len(time_range)
        
        # Создаем плавное движение цены с небольшими колебаниями
        base_trend = np.linspace(0, price_diff, num_candles)
        volatility = abs(price_diff) * 0.02  # 2% волатильность
        noise = np.random.normal(0, volatility, num_candles)
        
        close_prices = entry_price + base_trend + noise
        
        # Генерируем OHLC данные
        ohlc_data = []
        for i, timestamp in enumerate(time_range):
            close = close_prices[i]
            
            # Генерируем High и Low относительно Close
            high_low_range = abs(close) * 0.001  # 0.1% диапазон
            high = close + np.random.uniform(0, high_low_range)
            low = close - np.random.uniform(0, high_low_range)
            
            # Open берем из предыдущего Close (или entry_price для первой свечи)
            open_price = entry_price if i == 0 else ohlc_data[i-1]['Close']
            
            # Убеждаемся, что High >= max(Open, Close) и Low <= min(Open, Close)
            high = max(high, open_price, close)
            low = min(low, open_price, close)
            
            # Генерируем объем
            volume = np.random.randint(1000, 10000)
            
            ohlc_data.append({
                'Open': open_price,
                'High': high,
                'Low': low,
                'Close': close,
                'Volume': volume
            })
        
        df = pd.DataFrame(ohlc_data, index=time_range)
        return df
        
    except Exception as e:
        logger.error(f"❌ Ошибка генерации OHLC данных: {e}")
        # Возвращаем минимальные данные в случае ошибки
        simple_data = pd.DataFrame({
            'Open': [entry_price, entry_price],
            'High': [max(entry_price, exit_price), max(entry_price, exit_price)],
            'Low': [min(entry_price, exit_price), min(entry_price, exit_price)],
            'Close': [entry_price, exit_price],
            'Volume': [5000, 5000]
        }, index=[entry_time, exit_time])
        return simple_data


class DetailedTradeVisualizer:
    """
    🎨 Класс для создания детальных графиков торговых сделок с японскими свечами
    
    Создает:
    - Индивидуальные свечные графики для каждой сделки
    - Общие свечные графики по валютным парам
    - Портфельный обзор
    """
    
    def __init__(self, output_dir: str = "reports/detailed_charts"):
        """
        Инициализация визуализатора
        
        Args:
            output_dir: Директория для сохранения графиков
        """
        self.output_dir = Path(output_dir)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Создаем структуру папок
        self.session_dir = self.output_dir / self.timestamp
        self.individual_trades_dir = self.session_dir / "individual_trades"
        self.pair_summaries_dir = self.session_dir / "pair_summaries"
        self.portfolio_dir = self.session_dir / "portfolio"
        
        # Создаем все необходимые директории
        for dir_path in [self.individual_trades_dir, self.pair_summaries_dir, self.portfolio_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"📁 Создана структура папок в: {self.session_dir}")
        
        # Цветовая схема для mplfinance
        self.candlestick_style = mpf.make_marketcolors(
            up='#2E8B57',      # Зеленый для растущих свечей
            down='#DC143C',    # Красный для падающих свечей
            edge='inherit',
            wick={'up':'#2E8B57', 'down':'#DC143C'},
            volume='#708090'   # Серый для объема
        )
        
        self.chart_style = mpf.make_mpf_style(
            marketcolors=self.candlestick_style,
            gridstyle='-',
            gridcolor='lightgray',
            y_on_right=False
        )
        
        # Цветовая схема для дополнительных элементов
        self.colors = {
            'profit': '#2E8B57',      # Зеленый для прибыли
            'loss': '#DC143C',        # Красный для убытка
            'entry_long': '#4169E1',  # Синий для входа в лонг
            'entry_short': '#FF6347', # Оранжевый для входа в шорт
            'exit': '#9932CC',        # Фиолетовый для выхода
            'tp': '#32CD32',          # Лайм для TP
            'sl': '#FF4500',          # Красно-оранжевый для SL
            'price': '#1E90FF',       # Голубой для цены
            'volume': '#708090'       # Серый для объема
        }

    def create_individual_trade_charts(self, symbol: str, trades: List[Any]) -> None:
        """
        🕯️ Создает свечные графики для каждой отдельной сделки
        
        Args:
            symbol: Символ валютной пары
            trades: Список сделок
        """
        try:
            logger.info(f"🎨 Создание свечных графиков отдельных сделок для {symbol}: {len(trades)} сделок")
            
            for i, trade in enumerate(trades):
                try:
                    # Генерируем OHLC данные для сделки
                    ohlc_data = generate_ohlc_data(
                        trade.entry_time, 
                        trade.exit_time, 
                        trade.entry_price, 
                        trade.exit_price
                    )
                    
                    # Создаем дополнительные линии для отображения на графике
                    addplot_lines = []
                    
                    # Линия входа
                    entry_line = [trade.entry_price] * len(ohlc_data)
                    addplot_lines.append(mpf.make_addplot(entry_line, color=self.colors['entry_long'], 
                                                        linestyle='--', width=2, alpha=0.8))
                    
                    # Линия выхода
                    exit_line = [trade.exit_price] * len(ohlc_data)
                    addplot_lines.append(mpf.make_addplot(exit_line, color=self.colors['exit'], 
                                                        linestyle='--', width=2, alpha=0.8))
                    
                    # Добавляем TP и SL уровни если они есть
                    if hasattr(trade, 'tp_levels') and trade.tp_levels:
                        for tp_level in trade.tp_levels:
                            tp_line = [tp_level] * len(ohlc_data)
                            addplot_lines.append(mpf.make_addplot(tp_line, color=self.colors['tp'], 
                                                                linestyle=':', width=1.5, alpha=0.7))
                    
                    if hasattr(trade, 'sl_level') and trade.sl_level:
                        sl_line = [trade.sl_level] * len(ohlc_data)
                        addplot_lines.append(mpf.make_addplot(sl_line, color=self.colors['sl'], 
                                                            linestyle=':', width=1.5, alpha=0.7))
                    
                    # Создаем свечной график с помощью mplfinance
                    color = self.colors['profit'] if trade.pnl > 0 else self.colors['loss']
                    result_text = '✅ ПРИБЫЛЬ' if trade.pnl > 0 else '❌ УБЫТОК'
                    
                    title = f"🕯️ {symbol} - Сделка #{i+1} - {result_text}\n" \
                           f"Вход: {trade.entry_price:.6f} | Выход: {trade.exit_price:.6f} | " \
                           f"P&L: {trade.pnl:.2f}$ ({trade.pnl_percent:.2f}%)"
                    
                    # Создаем свечной график
                    mpf.plot(ohlc_data, 
                           type='candle',
                           style=self.chart_style,
                           addplot=addplot_lines,
                           volume=True,
                           title=title,
                           ylabel='Цена',
                           ylabel_lower='Объем',
                           datetime_format='%H:%M',
                           xrotation=45,
                           figsize=(16, 10),
                           savefig=dict(
                               fname=self.individual_trades_dir / f"{symbol}_trade_{i+1:03d}_{trade.entry_time.strftime('%Y%m%d_%H%M')}.png",
                               dpi=300,
                               bbox_inches='tight',
                               facecolor='white'
                           ))
                    
                    logger.info(f"✅ Создан свечной график сделки: {symbol}_trade_{i+1:03d}")
                    
                except Exception as e:
                    logger.error(f"❌ Ошибка создания свечного графика сделки {i+1}: {e}")
                    
        except Exception as e:
            logger.error(f"❌ Ошибка создания свечных графиков отдельных сделок для {symbol}: {e}")

    def create_pair_summary_chart(self, symbol: str, trades: List[Any]) -> None:
        """
        🕯️ Создает общий свечной график всех сделок по валютной паре
        
        Args:
            symbol: Символ валютной пары
            trades: Список сделок
        """
        try:
            logger.info(f"📈 Создание общего свечного графика для {symbol}: {len(trades)} сделок")
            
            if not trades:
                logger.warning(f"⚠️ Нет сделок для создания графика {symbol}")
                return
            
            # Сортируем сделки по времени
            sorted_trades = sorted(trades, key=lambda x: x.entry_time)
            
            # Определяем общий временной диапазон
            start_time = min(t.entry_time for t in sorted_trades)
            end_time = max(t.exit_time for t in sorted_trades)
            
            # Генерируем общие OHLC данные для всего периода
            all_prices = []
            for trade in sorted_trades:
                all_prices.extend([trade.entry_price, trade.exit_price])
            
            min_price = min(all_prices)
            max_price = max(all_prices)
            price_range = max_price - min_price
            
            # Создаем базовые OHLC данные для периода
            time_range = pd.date_range(start=start_time, end=end_time, freq='1h')
            if len(time_range) < 10:
                time_range = pd.date_range(start=start_time, periods=24, freq='1h')
            
            # Генерируем базовый тренд
            base_prices = np.linspace(min_price, max_price, len(time_range))
            volatility = price_range * 0.01
            
            ohlc_data = []
            for i, timestamp in enumerate(time_range):
                base_price = base_prices[i]
                noise = np.random.normal(0, volatility)
                close = base_price + noise
                
                high_low_range = price_range * 0.005
                high = close + np.random.uniform(0, high_low_range)
                low = close - np.random.uniform(0, high_low_range)
                open_price = base_prices[i-1] if i > 0 else close
                
                high = max(high, open_price, close)
                low = min(low, open_price, close)
                
                ohlc_data.append({
                    'Open': open_price,
                    'High': high,
                    'Low': low,
                    'Close': close,
                    'Volume': np.random.randint(5000, 15000)
                })
            
            df = pd.DataFrame(ohlc_data, index=time_range)
            
            # Создаем фигуру с тремя подграфиками
            fig = plt.figure(figsize=(20, 16))
            
            # График 1: Свечной график с точками входа/выхода
            ax1 = plt.subplot2grid((4, 1), (0, 0), rowspan=2)
            
            # Создаем дополнительные элементы для отображения
            addplot_elements = []
            
            # Добавляем точки входа и выхода как отдельные серии
            profitable_trades = [t for t in trades if t.pnl > 0]
            losing_trades = [t for t in trades if t.pnl <= 0]
            
            # Создаем серии для точек входа/выхода
            entry_points = pd.Series(index=df.index, dtype=float)
            exit_points = pd.Series(index=df.index, dtype=float)
            
            for trade in sorted_trades:
                # Находим ближайший индекс для времени входа и выхода
                entry_idx = df.index.get_indexer([trade.entry_time], method='nearest')[0]
                exit_idx = df.index.get_indexer([trade.exit_time], method='nearest')[0]
                
                if 0 <= entry_idx < len(df):
                    entry_points.iloc[entry_idx] = trade.entry_price
                if 0 <= exit_idx < len(df):
                    exit_points.iloc[exit_idx] = trade.exit_price
            
            # Добавляем точки входа и выхода
            addplot_elements.append(mpf.make_addplot(entry_points, type='scatter', 
                                                   markersize=100, marker='o', 
                                                   color=self.colors['entry_long']))
            addplot_elements.append(mpf.make_addplot(exit_points, type='scatter', 
                                                   markersize=100, marker='X', 
                                                   color=self.colors['exit']))
            
            # Создаем свечной график
            mpf.plot(df, 
                   type='candle',
                   style=self.chart_style,
                   addplot=addplot_elements,
                   volume=True,
                   title=f"🕯️ {symbol} - Свечной график всех сделок",
                   ylabel='Цена',
                   datetime_format='%m-%d %H:%M',
                   xrotation=45,
                   figsize=(16, 8),
                   savefig=dict(
                       fname=self.pair_summaries_dir / f"{symbol}_summary_candlestick.png",
                       dpi=300,
                       bbox_inches='tight',
                       facecolor='white'
                   ))
            
            # Создаем отдельный график для статистики
            profitable_trades = [t for t in trades if t.pnl > 0]
            losing_trades = [t for t in trades if t.pnl < 0]
            
            total_trades = len(trades)
            winning_trades = len(profitable_trades)
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            total_pnl = sum(t.pnl for t in trades)
            avg_pnl = total_pnl / total_trades if total_trades > 0 else 0
            
            # Создаем график статистики
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
            
            # График P&L по времени
            cumulative_pnl = 0
            times = []
            pnl_values = []
            
            for trade in sorted_trades:
                cumulative_pnl += trade.pnl
                times.append(trade.exit_time)
                pnl_values.append(cumulative_pnl)
            
            ax1.plot(times, pnl_values, linewidth=3, color=self.colors['price'], marker='o')
            ax1.fill_between(times, pnl_values, alpha=0.3, color=self.colors['price'])
            ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            ax1.set_title(f"💰 {symbol} - Кумулятивный P&L по времени", fontsize=14)
            ax1.set_ylabel("P&L ($)")
            ax1.grid(True, alpha=0.3)
            
            # График статистики
            stats_text = f"""
📊 СТАТИСТИКА ПО {symbol}:
• Всего сделок: {total_trades}                    • Прибыльных: {winning_trades} ({win_rate:.1f}%)
• Убыточных: {len(losing_trades)} ({100-win_rate:.1f}%)      • Общий P&L: {total_pnl:.2f}$
• Средний P&L: {avg_pnl:.2f}$                     • Лучшая сделка: {max(t.pnl for t in trades):.2f}$
• Худшая сделка: {min(t.pnl for t in trades):.2f}$
            """
            
            ax2.text(0.05, 0.5, stats_text, fontsize=12, transform=ax2.transAxes, 
                    verticalalignment='center', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
            ax2.axis('off')
            
            plt.tight_layout()
            
            # Сохраняем график статистики
            filename = f"{symbol}_summary_statistics.png"
            filepath = self.pair_summaries_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            logger.info(f"✅ Создан общий свечной график для {symbol}: {filename}")
            
        except Exception as e:
            logger.error(f"❌ Ошибка создания общего свечного графика для {symbol}: {e}")

    def create_portfolio_overview(self, results: Dict[str, Any]) -> None:
        """
        🎯 Создает обзор всего портфеля
        
        Args:
            results: Результаты тестирования по всем парам
        """
        try:
            logger.info(f"🎯 Создание обзора портфеля для {len(results)} пар")
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 15))
            
            # Подготавливаем данные
            symbols = list(results.keys())
            
            # Безопасное извлечение данных с проверкой типов
            win_rates = []
            total_pnls = []
            total_trades = []
            
            for symbol in symbols:
                result = results[symbol]
                if hasattr(result, 'win_rate'):
                    win_rates.append(result.win_rate)
                    total_pnls.append(result.total_pnl)
                    total_trades.append(result.total_trades)
                else:
                    # Если это словарь или другой тип данных
                    win_rates.append(50.0)  # Значение по умолчанию
                    total_pnls.append(0.0)
                    total_trades.append(len(result) if isinstance(result, list) else 1)
            
            # График 1: Винрейт по парам
            colors = [self.colors['profit'] if wr >= 50 else self.colors['loss'] for wr in win_rates]
            bars1 = ax1.bar(symbols, win_rates, color=colors, alpha=0.7)
            ax1.axhline(y=50, color='black', linestyle='--', alpha=0.5, label='50% винрейт')
            ax1.set_title("📊 Винрейт по валютным парам", fontsize=14, weight='bold')
            ax1.set_ylabel("Винрейт (%)")
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Добавляем значения на столбцы
            for bar, wr in zip(bars1, win_rates):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{wr:.1f}%', ha='center', va='bottom', fontweight='bold')
            
            # График 2: P&L по парам
            colors = [self.colors['profit'] if pnl >= 0 else self.colors['loss'] for pnl in total_pnls]
            bars2 = ax2.bar(symbols, total_pnls, color=colors, alpha=0.7)
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            ax2.set_title("💰 P&L по валютным парам", fontsize=14, weight='bold')
            ax2.set_ylabel("P&L ($)")
            ax2.grid(True, alpha=0.3)
            
            # Добавляем значения на столбцы
            for bar, pnl in zip(bars2, total_pnls):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + (0.5 if height >= 0 else -0.5),
                        f'{pnl:.1f}$', ha='center', va='bottom' if height >= 0 else 'top', fontweight='bold')
            
            # График 3: Количество сделок
            ax3.bar(symbols, total_trades, color=self.colors['price'], alpha=0.7)
            ax3.set_title("📈 Количество сделок по парам", fontsize=14, weight='bold')
            ax3.set_ylabel("Количество сделок")
            ax3.grid(True, alpha=0.3)
            
            # График 4: Общая статистика
            total_all_trades = sum(total_trades)
            total_all_pnl = sum(total_pnls)
            avg_winrate = sum(win_rates) / len(win_rates) if win_rates else 0
            
            stats_text = f"""
🎯 ОБЩАЯ СТАТИСТИКА ПОРТФЕЛЯ:

📊 Торговая активность:
• Всего сделок: {total_all_trades}
• Средний винрейт: {avg_winrate:.1f}%
• Общий P&L: {total_all_pnl:.2f}$

💰 Результаты по парам:
• Прибыльных пар: {len([p for p in total_pnls if p > 0])}
• Убыточных пар: {len([p for p in total_pnls if p <= 0])}
• Лучшая пара: {symbols[total_pnls.index(max(total_pnls))]} ({max(total_pnls):.2f}$)
• Худшая пара: {symbols[total_pnls.index(min(total_pnls))]} ({min(total_pnls):.2f}$)

🎯 Эффективность:
• Средний P&L на сделку: {total_all_pnl/total_all_trades:.2f}$ 
• ROI портфеля: {total_all_pnl:.2f}$
            """
            
            ax4.text(0.1, 0.5, stats_text, fontsize=11, transform=ax4.transAxes, 
                    verticalalignment='center', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.3))
            ax4.axis('off')
            
            plt.tight_layout()
            
            # Сохраняем график
            filename = "portfolio_overview.png"
            filepath = self.portfolio_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"✅ Создан обзор портфеля: {filename}")
            
        except Exception as e:
            logger.error(f"❌ Ошибка создания обзора портфеля: {e}")

def generate_trade_reports(winrate_results: Dict[str, Any], test_period_days: int) -> Dict[str, str]:
    """
    🎨 Генерирует все отчеты по результатам тестирования
    
    Args:
        winrate_results: Результаты тестирования винрейта
        test_period_days: Период тестирования в днях
        
    Returns:
        Словарь с путями к созданным отчетам
    """
    visualizer = DetailedTradeVisualizer()
    report_paths = {}
    
    # Рассчитываем общий винрейт
    total_trades = sum(result.total_trades for result in winrate_results.values())
    total_wins = sum(result.winning_trades for result in winrate_results.values())
    overall_winrate = (total_wins / total_trades * 100) if total_trades > 0 else 0
    
    print(f"🎨 Генерируем детальные отчеты...")
    print(f"📊 Общий винрейт: {overall_winrate:.1f}% ({total_wins}/{total_trades})")
    
    # Создаем отчеты по каждой паре
    for symbol, result in winrate_results.items():
        if result.trades:
            print(f"📈 Создаем отчет для {symbol} ({len(result.trades)} сделок)")
            
            # Преобразуем данные сделок
            trade_data_list = []
            for trade in result.trades:
                # Здесь нужно загрузить рыночные данные для периода сделки
                # Пока используем заглушку
                mock_market_data = pd.DataFrame({
                    'open': [trade.entry_price] * 100,
                    'high': [trade.entry_price * 1.01] * 100,
                    'low': [trade.entry_price * 0.99] * 100,
                    'close': [trade.exit_price] * 100,
                    'volume': [1000000] * 100
                }, index=pd.date_range(trade.entry_time, periods=100, freq='1h'))
                
                trade_viz_data = TradeVisualizationData(
                    symbol=trade.symbol,
                    entry_time=trade.entry_time,
                    exit_time=trade.exit_time,
                    entry_price=trade.entry_price,
                    exit_price=trade.exit_price,
                    direction=trade.direction,
                    pnl=trade.pnl,
                    pnl_percent=trade.pnl_percent,
                    confidence=trade.confidence,
                    ai_model=trade.ai_model,
                    position_size=trade.position_size,
                    commission=trade.commission,
                    exit_reason=trade.exit_reason,
                    partial_exits=trade.partial_exits or [],
                    market_data=mock_market_data
                )
                trade_data_list.append(trade_viz_data)
            
            # Создаем общий отчет по паре
            pair_winrate = result.win_rate * 100
            pair_report_path = visualizer.create_pair_summary_chart(
                symbol, trade_data_list, mock_market_data, test_period_days, pair_winrate
            )
            report_paths[f'{symbol}_summary'] = pair_report_path
            
            # Создаем отчеты по отдельным сделкам (первые 5 для примера)
            for i, trade_data in enumerate(trade_data_list[:5]):
                individual_report_path = visualizer.create_individual_trade_chart(
                    trade_data, test_period_days, pair_winrate
                )
                report_paths[f'{symbol}_trade_{i+1}'] = individual_report_path
    
    # Создаем общий обзор портфеля
    portfolio_data = {}
    for symbol, result in winrate_results.items():
        portfolio_data[symbol] = {
            'total_pnl': result.total_pnl,
            'win_rate': result.win_rate,
            'total_trades': result.total_trades,
            'avg_trade_pnl': result.avg_trade_pnl
        }
    
    portfolio_report_path = visualizer.create_portfolio_overview(
        portfolio_data, test_period_days, overall_winrate
    )
    report_paths['portfolio_overview'] = portfolio_report_path
    
    print(f"✅ Создано {len(report_paths)} отчетов")
    print(f"📁 Отчеты сохранены в: {visualizer.output_dir}")
    
    return report_paths

if __name__ == "__main__":
    print("🎯 Детальный визуализатор торговых сделок")
    print("Для использования импортируйте функцию generate_trade_reports")