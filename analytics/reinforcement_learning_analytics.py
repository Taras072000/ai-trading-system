#!/usr/bin/env python3
"""
Reinforcement Learning Analytics System
Система аналитики для обучения с подкреплением - отчеты и визуализация
"""

import json
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Настройка стиля графиков
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

@dataclass
class AnalyticsConfig:
    """Конфигурация для системы аналитики"""
    results_dir: str = "results/reinforcement_learning"
    reports_dir: str = "reports/reinforcement_learning"
    plots_dir: str = "plots/reinforcement_learning"
    
    # Настройки графиков
    figure_size: Tuple[int, int] = (12, 8)
    dpi: int = 300
    save_format: str = "png"
    
    # Настройки отчетов
    include_detailed_trades: bool = True
    include_weight_evolution: bool = True
    include_performance_metrics: bool = True

class ReinforcementLearningAnalytics:
    """
    Система аналитики для обучения с подкреплением
    """
    
    def __init__(self, config: AnalyticsConfig = None):
        self.config = config or AnalyticsConfig()
        
        # Создаем директории
        os.makedirs(self.config.reports_dir, exist_ok=True)
        os.makedirs(self.config.plots_dir, exist_ok=True)
        
        # Данные
        self.sessions_data: List[Dict] = []
        self.combined_data: Optional[pd.DataFrame] = None
    
    def load_session_results(self, session_files: List[str] = None) -> bool:
        """
        Загрузка результатов сессий
        """
        try:
            if session_files is None:
                # Загружаем все файлы из директории результатов
                session_files = []
                if os.path.exists(self.config.results_dir):
                    for file in os.listdir(self.config.results_dir):
                        if file.endswith('.json'):
                            session_files.append(os.path.join(self.config.results_dir, file))
            
            self.sessions_data = []
            
            for file_path in session_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        session_data = json.load(f)
                        session_data['file_path'] = file_path
                        session_data['session_name'] = os.path.basename(file_path).replace('.json', '')
                        self.sessions_data.append(session_data)
                        
                except Exception as e:
                    logger.error(f"Ошибка загрузки файла {file_path}: {e}")
            
            logger.info(f"✅ Загружено {len(self.sessions_data)} сессий")
            
            # Создаем объединенный DataFrame
            self._create_combined_dataframe()
            
            return len(self.sessions_data) > 0
            
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки результатов: {e}")
            return False
    
    def _create_combined_dataframe(self):
        """Создание объединенного DataFrame для анализа"""
        try:
            all_trades = []
            
            for session in self.sessions_data:
                session_name = session['session_name']
                trades = session.get('trades', [])
                
                for trade in trades:
                    trade_data = {
                        'session_name': session_name,
                        'symbol': trade['symbol'],
                        'action': trade['action'],
                        'entry_price': trade['entry_price'],
                        'exit_price': trade['exit_price'],
                        'pnl': trade['pnl'],
                        'pnl_percent': trade['pnl_percent'],
                        'confidence': trade['confidence'],
                        'entry_time': pd.to_datetime(trade['entry_time']),
                        'exit_time': pd.to_datetime(trade['exit_time']),
                        'duration_minutes': trade['duration_minutes'],
                        'reward_applied': trade['reward_applied'],
                        'punishment_applied': trade['punishment_applied'],
                        'is_profitable': trade['pnl'] > 0
                    }
                    
                    # Добавляем веса AI
                    for ai_name, weight in trade.get('ai_weights_after', {}).items():
                        trade_data[f'weight_{ai_name}'] = weight
                    
                    all_trades.append(trade_data)
            
            self.combined_data = pd.DataFrame(all_trades)
            
            if not self.combined_data.empty:
                # Добавляем дополнительные колонки
                self.combined_data['trade_number'] = self.combined_data.groupby('session_name').cumcount() + 1
                self.combined_data['cumulative_pnl'] = self.combined_data.groupby('session_name')['pnl'].cumsum()
                self.combined_data['rolling_winrate'] = self.combined_data.groupby('session_name')['is_profitable'].rolling(window=10, min_periods=1).mean().reset_index(0, drop=True)
                
                logger.info(f"✅ Создан объединенный DataFrame с {len(self.combined_data)} сделками")
            
        except Exception as e:
            logger.error(f"❌ Ошибка создания DataFrame: {e}")
    
    def generate_comprehensive_report(self, session_name: str = None) -> str:
        """
        Генерация комплексного отчета
        """
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            if session_name:
                report_name = f"rl_report_{session_name}_{timestamp}.html"
                title = f"Отчет по обучению с подкреплением - {session_name}"
            else:
                report_name = f"rl_comprehensive_report_{timestamp}.html"
                title = "Комплексный отчет по обучению с подкреплением"
            
            report_path = os.path.join(self.config.reports_dir, report_name)
            
            # Генерируем все графики
            plots = self._generate_all_plots(session_name)
            
            # Создаем HTML отчет
            html_content = self._create_html_report(title, plots, session_name)
            
            # Сохраняем отчет
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logger.info(f"✅ Отчет сохранен: {report_path}")
            return report_path
            
        except Exception as e:
            logger.error(f"❌ Ошибка генерации отчета: {e}")
            return ""
    
    def _generate_all_plots(self, session_name: str = None) -> Dict[str, str]:
        """Генерация всех графиков"""
        plots = {}
        
        try:
            # Фильтруем данные по сессии если указана
            if session_name and not self.combined_data.empty:
                data = self.combined_data[self.combined_data['session_name'] == session_name]
            else:
                data = self.combined_data
            
            if data.empty:
                logger.warning("Нет данных для генерации графиков")
                return plots
            
            # 1. График эволюции винрейта
            plots['winrate_evolution'] = self._plot_winrate_evolution(data, session_name)
            
            # 2. График кумулятивного PnL
            plots['cumulative_pnl'] = self._plot_cumulative_pnl(data, session_name)
            
            # 3. График эволюции весов AI
            plots['weights_evolution'] = self._plot_weights_evolution(session_name)
            
            # 4. Распределение PnL
            plots['pnl_distribution'] = self._plot_pnl_distribution(data, session_name)
            
            # 5. Анализ по символам
            plots['symbol_analysis'] = self._plot_symbol_analysis(data, session_name)
            
            # 6. Корреляция уверенности и результата
            plots['confidence_correlation'] = self._plot_confidence_correlation(data, session_name)
            
            # 7. Тепловая карта производительности
            plots['performance_heatmap'] = self._plot_performance_heatmap(data, session_name)
            
            # 8. Анализ длительности сделок
            plots['duration_analysis'] = self._plot_duration_analysis(data, session_name)
            
            logger.info(f"✅ Сгенерировано {len(plots)} графиков")
            
        except Exception as e:
            logger.error(f"❌ Ошибка генерации графиков: {e}")
        
        return plots
    
    def _plot_winrate_evolution(self, data: pd.DataFrame, session_name: str = None) -> str:
        """График эволюции винрейта"""
        try:
            fig = go.Figure()
            
            if session_name:
                session_data = data[data['session_name'] == session_name]
                fig.add_trace(go.Scatter(
                    x=session_data['trade_number'],
                    y=session_data['rolling_winrate'] * 100,
                    mode='lines+markers',
                    name=f'Винрейт {session_name}',
                    line=dict(width=2)
                ))
            else:
                for session in data['session_name'].unique():
                    session_data = data[data['session_name'] == session]
                    fig.add_trace(go.Scatter(
                        x=session_data['trade_number'],
                        y=session_data['rolling_winrate'] * 100,
                        mode='lines',
                        name=f'Винрейт {session}',
                        line=dict(width=2)
                    ))
            
            fig.update_layout(
                title='Эволюция винрейта (скользящее среднее 10 сделок)',
                xaxis_title='Номер сделки',
                yaxis_title='Винрейт (%)',
                hovermode='x unified',
                height=500
            )
            
            filename = f"winrate_evolution_{session_name or 'all'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            filepath = os.path.join(self.config.plots_dir, filename)
            pyo.plot(fig, filename=filepath, auto_open=False)
            
            return filepath
            
        except Exception as e:
            logger.error(f"❌ Ошибка создания графика винрейта: {e}")
            return ""
    
    def _plot_cumulative_pnl(self, data: pd.DataFrame, session_name: str = None) -> str:
        """График кумулятивного PnL"""
        try:
            fig = go.Figure()
            
            if session_name:
                session_data = data[data['session_name'] == session_name]
                fig.add_trace(go.Scatter(
                    x=session_data['trade_number'],
                    y=session_data['cumulative_pnl'],
                    mode='lines+markers',
                    name=f'Кумулятивный PnL {session_name}',
                    line=dict(width=2),
                    fill='tonexty' if session_data['cumulative_pnl'].iloc[-1] > 0 else None,
                    fillcolor='rgba(0,255,0,0.1)' if session_data['cumulative_pnl'].iloc[-1] > 0 else 'rgba(255,0,0,0.1)'
                ))
            else:
                for session in data['session_name'].unique():
                    session_data = data[data['session_name'] == session]
                    fig.add_trace(go.Scatter(
                        x=session_data['trade_number'],
                        y=session_data['cumulative_pnl'],
                        mode='lines',
                        name=f'PnL {session}',
                        line=dict(width=2)
                    ))
            
            fig.add_hline(y=0, line_dash="dash", line_color="gray")
            
            fig.update_layout(
                title='Кумулятивный PnL',
                xaxis_title='Номер сделки',
                yaxis_title='Кумулятивный PnL ($)',
                hovermode='x unified',
                height=500
            )
            
            filename = f"cumulative_pnl_{session_name or 'all'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            filepath = os.path.join(self.config.plots_dir, filename)
            pyo.plot(fig, filename=filepath, auto_open=False)
            
            return filepath
            
        except Exception as e:
            logger.error(f"❌ Ошибка создания графика PnL: {e}")
            return ""
    
    def _plot_weights_evolution(self, session_name: str = None) -> str:
        """График эволюции весов AI"""
        try:
            # Получаем данные об эволюции весов из сессий
            weight_data = []
            
            sessions_to_analyze = [s for s in self.sessions_data if not session_name or s['session_name'] == session_name]
            
            for session in sessions_to_analyze:
                weight_evolution = session.get('weight_evolution', [])
                for point in weight_evolution:
                    for ai_name, weight in point.get('weights', {}).items():
                        weight_data.append({
                            'session_name': session['session_name'],
                            'timestamp': pd.to_datetime(point['timestamp']),
                            'trade_count': point.get('trade_count', 0),
                            'ai_name': ai_name,
                            'weight': weight,
                            'win_rate': point.get('win_rate', 0)
                        })
            
            if not weight_data:
                logger.warning("Нет данных об эволюции весов")
                return ""
            
            weight_df = pd.DataFrame(weight_data)
            
            fig = go.Figure()
            
            colors = px.colors.qualitative.Set1
            
            for i, ai_name in enumerate(weight_df['ai_name'].unique()):
                ai_data = weight_df[weight_df['ai_name'] == ai_name]
                
                if session_name:
                    session_data = ai_data[ai_data['session_name'] == session_name]
                    fig.add_trace(go.Scatter(
                        x=session_data['trade_count'],
                        y=session_data['weight'],
                        mode='lines+markers',
                        name=ai_name,
                        line=dict(width=2, color=colors[i % len(colors)])
                    ))
                else:
                    for session in ai_data['session_name'].unique():
                        session_data = ai_data[ai_data['session_name'] == session]
                        fig.add_trace(go.Scatter(
                            x=session_data['trade_count'],
                            y=session_data['weight'],
                            mode='lines',
                            name=f'{ai_name} ({session})',
                            line=dict(width=2, color=colors[i % len(colors)])
                        ))
            
            fig.update_layout(
                title='Эволюция весов AI модулей',
                xaxis_title='Количество сделок',
                yaxis_title='Вес модуля',
                hovermode='x unified',
                height=500
            )
            
            filename = f"weights_evolution_{session_name or 'all'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            filepath = os.path.join(self.config.plots_dir, filename)
            pyo.plot(fig, filename=filepath, auto_open=False)
            
            return filepath
            
        except Exception as e:
            logger.error(f"❌ Ошибка создания графика весов: {e}")
            return ""
    
    def _plot_pnl_distribution(self, data: pd.DataFrame, session_name: str = None) -> str:
        """Распределение PnL"""
        try:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Гистограмма PnL', 'Box Plot PnL', 'PnL по действиям', 'PnL по символам'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # Гистограмма PnL
            fig.add_trace(
                go.Histogram(x=data['pnl'], nbinsx=30, name='PnL Distribution'),
                row=1, col=1
            )
            
            # Box Plot PnL
            fig.add_trace(
                go.Box(y=data['pnl'], name='PnL Box Plot'),
                row=1, col=2
            )
            
            # PnL по действиям
            for action in data['action'].unique():
                action_data = data[data['action'] == action]
                fig.add_trace(
                    go.Box(y=action_data['pnl'], name=f'{action} PnL'),
                    row=2, col=1
                )
            
            # PnL по символам
            for symbol in data['symbol'].unique():
                symbol_data = data[data['symbol'] == symbol]
                fig.add_trace(
                    go.Box(y=symbol_data['pnl'], name=f'{symbol} PnL'),
                    row=2, col=2
                )
            
            fig.update_layout(
                title='Анализ распределения PnL',
                height=800,
                showlegend=False
            )
            
            filename = f"pnl_distribution_{session_name or 'all'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            filepath = os.path.join(self.config.plots_dir, filename)
            pyo.plot(fig, filename=filepath, auto_open=False)
            
            return filepath
            
        except Exception as e:
            logger.error(f"❌ Ошибка создания графика распределения PnL: {e}")
            return ""
    
    def _plot_symbol_analysis(self, data: pd.DataFrame, session_name: str = None) -> str:
        """Анализ по символам"""
        try:
            symbol_stats = data.groupby('symbol').agg({
                'pnl': ['count', 'sum', 'mean'],
                'is_profitable': 'mean',
                'confidence': 'mean'
            }).round(4)
            
            symbol_stats.columns = ['Trades', 'Total_PnL', 'Avg_PnL', 'Win_Rate', 'Avg_Confidence']
            symbol_stats = symbol_stats.reset_index()
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Винрейт по символам', 'Общий PnL по символам', 'Средний PnL по символам', 'Средняя уверенность'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # Винрейт по символам
            fig.add_trace(
                go.Bar(x=symbol_stats['symbol'], y=symbol_stats['Win_Rate'] * 100, name='Win Rate'),
                row=1, col=1
            )
            
            # Общий PnL по символам
            fig.add_trace(
                go.Bar(x=symbol_stats['symbol'], y=symbol_stats['Total_PnL'], name='Total PnL'),
                row=1, col=2
            )
            
            # Средний PnL по символам
            fig.add_trace(
                go.Bar(x=symbol_stats['symbol'], y=symbol_stats['Avg_PnL'], name='Avg PnL'),
                row=2, col=1
            )
            
            # Средняя уверенность
            fig.add_trace(
                go.Bar(x=symbol_stats['symbol'], y=symbol_stats['Avg_Confidence'] * 100, name='Avg Confidence'),
                row=2, col=2
            )
            
            fig.update_layout(
                title='Анализ производительности по символам',
                height=800,
                showlegend=False
            )
            
            filename = f"symbol_analysis_{session_name or 'all'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            filepath = os.path.join(self.config.plots_dir, filename)
            pyo.plot(fig, filename=filepath, auto_open=False)
            
            return filepath
            
        except Exception as e:
            logger.error(f"❌ Ошибка создания анализа по символам: {e}")
            return ""
    
    def _plot_confidence_correlation(self, data: pd.DataFrame, session_name: str = None) -> str:
        """Корреляция уверенности и результата"""
        try:
            fig = go.Figure()
            
            # Scatter plot уверенности vs PnL
            colors = ['green' if profitable else 'red' for profitable in data['is_profitable']]
            
            fig.add_trace(go.Scatter(
                x=data['confidence'],
                y=data['pnl'],
                mode='markers',
                marker=dict(
                    color=colors,
                    size=8,
                    opacity=0.6
                ),
                text=data['symbol'],
                hovertemplate='<b>%{text}</b><br>Confidence: %{x:.2f}<br>PnL: $%{y:.2f}<extra></extra>',
                name='Trades'
            ))
            
            # Добавляем линию тренда
            z = np.polyfit(data['confidence'], data['pnl'], 1)
            p = np.poly1d(z)
            x_trend = np.linspace(data['confidence'].min(), data['confidence'].max(), 100)
            y_trend = p(x_trend)
            
            fig.add_trace(go.Scatter(
                x=x_trend,
                y=y_trend,
                mode='lines',
                name='Trend Line',
                line=dict(color='blue', dash='dash')
            ))
            
            fig.add_hline(y=0, line_dash="dash", line_color="gray")
            
            fig.update_layout(
                title='Корреляция между уверенностью AI и результатом сделки',
                xaxis_title='Уверенность AI',
                yaxis_title='PnL ($)',
                height=500
            )
            
            filename = f"confidence_correlation_{session_name or 'all'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            filepath = os.path.join(self.config.plots_dir, filename)
            pyo.plot(fig, filename=filepath, auto_open=False)
            
            return filepath
            
        except Exception as e:
            logger.error(f"❌ Ошибка создания графика корреляции: {e}")
            return ""
    
    def _plot_performance_heatmap(self, data: pd.DataFrame, session_name: str = None) -> str:
        """Тепловая карта производительности"""
        try:
            # Создаем сводную таблицу по символам и действиям
            pivot_winrate = data.pivot_table(
                values='is_profitable',
                index='symbol',
                columns='action',
                aggfunc='mean'
            ).fillna(0)
            
            pivot_pnl = data.pivot_table(
                values='pnl',
                index='symbol',
                columns='action',
                aggfunc='mean'
            ).fillna(0)
            
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Винрейт по символам и действиям', 'Средний PnL по символам и действиям')
            )
            
            # Тепловая карта винрейта
            fig.add_trace(
                go.Heatmap(
                    z=pivot_winrate.values * 100,
                    x=pivot_winrate.columns,
                    y=pivot_winrate.index,
                    colorscale='RdYlGn',
                    text=np.round(pivot_winrate.values * 100, 1),
                    texttemplate="%{text}%",
                    textfont={"size": 10},
                    colorbar=dict(title="Win Rate (%)", x=0.45)
                ),
                row=1, col=1
            )
            
            # Тепловая карта PnL
            fig.add_trace(
                go.Heatmap(
                    z=pivot_pnl.values,
                    x=pivot_pnl.columns,
                    y=pivot_pnl.index,
                    colorscale='RdYlGn',
                    text=np.round(pivot_pnl.values, 2),
                    texttemplate="$%{text}",
                    textfont={"size": 10},
                    colorbar=dict(title="Avg PnL ($)", x=1.02)
                ),
                row=1, col=2
            )
            
            fig.update_layout(
                title='Тепловая карта производительности',
                height=500
            )
            
            filename = f"performance_heatmap_{session_name or 'all'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            filepath = os.path.join(self.config.plots_dir, filename)
            pyo.plot(fig, filename=filepath, auto_open=False)
            
            return filepath
            
        except Exception as e:
            logger.error(f"❌ Ошибка создания тепловой карты: {e}")
            return ""
    
    def _plot_duration_analysis(self, data: pd.DataFrame, session_name: str = None) -> str:
        """Анализ длительности сделок"""
        try:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Распределение длительности', 'Длительность vs PnL', 'Длительность по символам', 'Длительность по действиям')
            )
            
            # Распределение длительности
            fig.add_trace(
                go.Histogram(x=data['duration_minutes'], nbinsx=20, name='Duration Distribution'),
                row=1, col=1
            )
            
            # Длительность vs PnL
            colors = ['green' if profitable else 'red' for profitable in data['is_profitable']]
            fig.add_trace(
                go.Scatter(
                    x=data['duration_minutes'],
                    y=data['pnl'],
                    mode='markers',
                    marker=dict(color=colors, size=6, opacity=0.6),
                    name='Duration vs PnL'
                ),
                row=1, col=2
            )
            
            # Длительность по символам
            for symbol in data['symbol'].unique():
                symbol_data = data[data['symbol'] == symbol]
                fig.add_trace(
                    go.Box(y=symbol_data['duration_minutes'], name=f'{symbol}'),
                    row=2, col=1
                )
            
            # Длительность по действиям
            for action in data['action'].unique():
                action_data = data[data['action'] == action]
                fig.add_trace(
                    go.Box(y=action_data['duration_minutes'], name=f'{action}'),
                    row=2, col=2
                )
            
            fig.update_layout(
                title='Анализ длительности сделок',
                height=800,
                showlegend=False
            )
            
            filename = f"duration_analysis_{session_name or 'all'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            filepath = os.path.join(self.config.plots_dir, filename)
            pyo.plot(fig, filename=filepath, auto_open=False)
            
            return filepath
            
        except Exception as e:
            logger.error(f"❌ Ошибка создания анализа длительности: {e}")
            return ""
    
    def _create_html_report(self, title: str, plots: Dict[str, str], session_name: str = None) -> str:
        """Создание HTML отчета"""
        try:
            # Получаем статистику
            stats = self._calculate_statistics(session_name)
            
            html_content = f"""
            <!DOCTYPE html>
            <html lang="ru">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>{title}</title>
                <style>
                    body {{
                        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                        margin: 0;
                        padding: 20px;
                        background-color: #f5f5f5;
                    }}
                    .container {{
                        max-width: 1200px;
                        margin: 0 auto;
                        background-color: white;
                        padding: 30px;
                        border-radius: 10px;
                        box-shadow: 0 0 20px rgba(0,0,0,0.1);
                    }}
                    h1 {{
                        color: #2c3e50;
                        text-align: center;
                        margin-bottom: 30px;
                        border-bottom: 3px solid #3498db;
                        padding-bottom: 10px;
                    }}
                    h2 {{
                        color: #34495e;
                        margin-top: 40px;
                        margin-bottom: 20px;
                        border-left: 4px solid #3498db;
                        padding-left: 15px;
                    }}
                    .stats-grid {{
                        display: grid;
                        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                        gap: 20px;
                        margin-bottom: 30px;
                    }}
                    .stat-card {{
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        color: white;
                        padding: 20px;
                        border-radius: 10px;
                        text-align: center;
                        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
                    }}
                    .stat-value {{
                        font-size: 2em;
                        font-weight: bold;
                        margin-bottom: 5px;
                    }}
                    .stat-label {{
                        font-size: 0.9em;
                        opacity: 0.9;
                    }}
                    .plot-container {{
                        margin: 30px 0;
                        text-align: center;
                    }}
                    .plot-frame {{
                        width: 100%;
                        height: 600px;
                        border: none;
                        border-radius: 10px;
                        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
                    }}
                    .timestamp {{
                        text-align: center;
                        color: #7f8c8d;
                        margin-top: 30px;
                        font-style: italic;
                    }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>{title}</h1>
                    
                    <h2>📊 Основная статистика</h2>
                    <div class="stats-grid">
                        <div class="stat-card">
                            <div class="stat-value">{stats['total_trades']}</div>
                            <div class="stat-label">Всего сделок</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value">{stats['win_rate']:.1f}%</div>
                            <div class="stat-label">Винрейт</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value">${stats['total_pnl']:.2f}</div>
                            <div class="stat-label">Общий PnL</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value">${stats['avg_pnl']:.2f}</div>
                            <div class="stat-label">Средний PnL</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value">{stats['avg_confidence']:.1f}%</div>
                            <div class="stat-label">Средняя уверенность</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value">{stats['avg_duration']:.0f} мин</div>
                            <div class="stat-label">Средняя длительность</div>
                        </div>
                    </div>
            """
            
            # Добавляем графики
            for plot_name, plot_path in plots.items():
                if plot_path and os.path.exists(plot_path):
                    plot_title = plot_name.replace('_', ' ').title()
                    html_content += f"""
                    <h2>📈 {plot_title}</h2>
                    <div class="plot-container">
                        <iframe src="{os.path.basename(plot_path)}" class="plot-frame"></iframe>
                    </div>
                    """
            
            html_content += f"""
                    <div class="timestamp">
                        Отчет сгенерирован: {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}
                    </div>
                </div>
            </body>
            </html>
            """
            
            return html_content
            
        except Exception as e:
            logger.error(f"❌ Ошибка создания HTML отчета: {e}")
            return ""
    
    def _calculate_statistics(self, session_name: str = None) -> Dict:
        """Расчет статистики"""
        try:
            if self.combined_data.empty:
                return {
                    'total_trades': 0,
                    'win_rate': 0,
                    'total_pnl': 0,
                    'avg_pnl': 0,
                    'avg_confidence': 0,
                    'avg_duration': 0
                }
            
            # Фильтруем данные по сессии если указана
            if session_name:
                data = self.combined_data[self.combined_data['session_name'] == session_name]
            else:
                data = self.combined_data
            
            stats = {
                'total_trades': len(data),
                'win_rate': (data['is_profitable'].sum() / len(data)) * 100 if len(data) > 0 else 0,
                'total_pnl': data['pnl'].sum(),
                'avg_pnl': data['pnl'].mean(),
                'avg_confidence': data['confidence'].mean() * 100,
                'avg_duration': data['duration_minutes'].mean()
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"❌ Ошибка расчета статистики: {e}")
            return {}
    
    def compare_sessions(self, session_names: List[str]) -> str:
        """Сравнение нескольких сессий"""
        try:
            if not session_names or len(session_names) < 2:
                logger.error("Для сравнения нужно минимум 2 сессии")
                return ""
            
            comparison_data = []
            
            for session_name in session_names:
                stats = self._calculate_statistics(session_name)
                stats['session_name'] = session_name
                comparison_data.append(stats)
            
            # Создаем сравнительные графики
            comparison_plots = self._create_comparison_plots(comparison_data)
            
            # Создаем отчет сравнения
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_name = f"sessions_comparison_{timestamp}.html"
            report_path = os.path.join(self.config.reports_dir, report_name)
            
            html_content = self._create_comparison_html_report(comparison_data, comparison_plots)
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logger.info(f"✅ Отчет сравнения сохранен: {report_path}")
            return report_path
            
        except Exception as e:
            logger.error(f"❌ Ошибка сравнения сессий: {e}")
            return ""
    
    def _create_comparison_plots(self, comparison_data: List[Dict]) -> Dict[str, str]:
        """Создание графиков сравнения"""
        plots = {}
        
        try:
            df = pd.DataFrame(comparison_data)
            
            # График сравнения винрейта
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=df['session_name'],
                y=df['win_rate'],
                name='Win Rate',
                marker_color='lightblue'
            ))
            fig.update_layout(
                title='Сравнение винрейта по сессиям',
                xaxis_title='Сессия',
                yaxis_title='Винрейт (%)',
                height=400
            )
            
            filename = f"comparison_winrate_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            filepath = os.path.join(self.config.plots_dir, filename)
            pyo.plot(fig, filename=filepath, auto_open=False)
            plots['winrate_comparison'] = filepath
            
            # График сравнения PnL
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=df['session_name'],
                y=df['total_pnl'],
                name='Total PnL',
                marker_color='lightgreen'
            ))
            fig.update_layout(
                title='Сравнение общего PnL по сессиям',
                xaxis_title='Сессия',
                yaxis_title='Общий PnL ($)',
                height=400
            )
            
            filename = f"comparison_pnl_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            filepath = os.path.join(self.config.plots_dir, filename)
            pyo.plot(fig, filename=filepath, auto_open=False)
            plots['pnl_comparison'] = filepath
            
        except Exception as e:
            logger.error(f"❌ Ошибка создания графиков сравнения: {e}")
        
        return plots
    
    def _create_comparison_html_report(self, comparison_data: List[Dict], plots: Dict[str, str]) -> str:
        """Создание HTML отчета сравнения"""
        try:
            html_content = f"""
            <!DOCTYPE html>
            <html lang="ru">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Сравнение сессий обучения с подкреплением</title>
                <style>
                    body {{
                        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                        margin: 0;
                        padding: 20px;
                        background-color: #f5f5f5;
                    }}
                    .container {{
                        max-width: 1200px;
                        margin: 0 auto;
                        background-color: white;
                        padding: 30px;
                        border-radius: 10px;
                        box-shadow: 0 0 20px rgba(0,0,0,0.1);
                    }}
                    h1 {{
                        color: #2c3e50;
                        text-align: center;
                        margin-bottom: 30px;
                    }}
                    table {{
                        width: 100%;
                        border-collapse: collapse;
                        margin: 20px 0;
                    }}
                    th, td {{
                        padding: 12px;
                        text-align: center;
                        border-bottom: 1px solid #ddd;
                    }}
                    th {{
                        background-color: #3498db;
                        color: white;
                    }}
                    tr:nth-child(even) {{
                        background-color: #f2f2f2;
                    }}
                    .plot-frame {{
                        width: 100%;
                        height: 500px;
                        border: none;
                        border-radius: 10px;
                        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
                        margin: 20px 0;
                    }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>🔄 Сравнение сессий обучения с подкреплением</h1>
                    
                    <h2>📊 Сравнительная таблица</h2>
                    <table>
                        <tr>
                            <th>Сессия</th>
                            <th>Всего сделок</th>
                            <th>Винрейт (%)</th>
                            <th>Общий PnL ($)</th>
                            <th>Средний PnL ($)</th>
                            <th>Средняя уверенность (%)</th>
                        </tr>
            """
            
            for data in comparison_data:
                html_content += f"""
                        <tr>
                            <td>{data['session_name']}</td>
                            <td>{data['total_trades']}</td>
                            <td>{data['win_rate']:.1f}%</td>
                            <td>${data['total_pnl']:.2f}</td>
                            <td>${data['avg_pnl']:.2f}</td>
                            <td>{data['avg_confidence']:.1f}%</td>
                        </tr>
                """
            
            html_content += """
                    </table>
            """
            
            # Добавляем графики сравнения
            for plot_name, plot_path in plots.items():
                if plot_path and os.path.exists(plot_path):
                    html_content += f"""
                    <iframe src="{os.path.basename(plot_path)}" class="plot-frame"></iframe>
                    """
            
            html_content += f"""
                    <div style="text-align: center; color: #7f8c8d; margin-top: 30px; font-style: italic;">
                        Отчет сгенерирован: {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}
                    </div>
                </div>
            </body>
            </html>
            """
            
            return html_content
            
        except Exception as e:
            logger.error(f"❌ Ошибка создания HTML отчета сравнения: {e}")
            return ""

# Пример использования
def main():
    """Пример использования системы аналитики"""
    config = AnalyticsConfig()
    analytics = ReinforcementLearningAnalytics(config)
    
    # Загружаем результаты
    if analytics.load_session_results():
        # Генерируем комплексный отчет
        report_path = analytics.generate_comprehensive_report()
        print(f"✅ Отчет сохранен: {report_path}")
        
        # Сравниваем сессии (если есть несколько)
        if len(analytics.sessions_data) > 1:
            session_names = [s['session_name'] for s in analytics.sessions_data[:3]]  # Первые 3 сессии
            comparison_report = analytics.compare_sessions(session_names)
            print(f"✅ Отчет сравнения сохранен: {comparison_report}")
    else:
        print("❌ Не удалось загрузить результаты сессий")

if __name__ == "__main__":
    main()