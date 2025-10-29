"""
Улучшенная система отчетности с real-time дашбордами
Автоматические отчеты производительности, детальная аналитика решений, система алертов
"""

import asyncio
import logging
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio
from pathlib import Path
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
import websockets
import threading
import time

class AlertLevel(Enum):
    """Уровни алертов"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    SUCCESS = "success"

class ReportType(Enum):
    """Типы отчетов"""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    REAL_TIME = "real_time"
    CUSTOM = "custom"

@dataclass
class Alert:
    """Структура алерта"""
    id: str
    timestamp: datetime
    level: AlertLevel
    title: str
    message: str
    source: str
    data: Dict[str, Any] = None
    acknowledged: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat(),
            'level': self.level.value,
            'title': self.title,
            'message': self.message,
            'source': self.source,
            'data': self.data or {},
            'acknowledged': self.acknowledged
        }

@dataclass
class PerformanceSnapshot:
    """Снимок производительности"""
    timestamp: datetime
    win_rate: float
    roi: float
    drawdown: float
    sharpe_ratio: float
    profit_factor: float
    total_trades: int
    active_positions: int
    portfolio_value: float
    daily_pnl: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class TradingDecision:
    """Торговое решение для аналитики"""
    timestamp: datetime
    symbol: str
    action: str  # BUY, SELL, HOLD
    confidence: float
    reasoning: str
    parameters: Dict[str, Any]
    market_conditions: Dict[str, Any]
    result: Optional[Dict[str, Any]] = None  # Заполняется после закрытия позиции

class RealTimeDashboard:
    """Real-time дашборд для мониторинга системы"""
    
    def __init__(self, update_interval: int = 5):
        self.update_interval = update_interval
        self.is_running = False
        self.websocket_server = None
        self.connected_clients = set()
        self.logger = logging.getLogger(__name__)
        
        # Данные для дашборда
        self.performance_history: List[PerformanceSnapshot] = []
        self.recent_decisions: List[TradingDecision] = []
        self.active_alerts: List[Alert] = []
        
        # Настройки дашборда
        self.dashboard_config = {
            'theme': 'plotly_dark',
            'update_interval': update_interval,
            'max_history_points': 1000,
            'max_recent_decisions': 100
        }
    
    async def start_dashboard(self, host: str = "localhost", port: int = 8765):
        """Запуск real-time дашборда"""
        
        self.logger.info(f"Запуск real-time дашборда на {host}:{port}")
        self.is_running = True
        
        # Запуск WebSocket сервера
        start_server = websockets.serve(
            self._handle_websocket_connection, 
            host, 
            port
        )
        
        # Запуск обновления данных
        update_task = asyncio.create_task(self._update_dashboard_data())
        
        await asyncio.gather(start_server, update_task)
    
    async def _handle_websocket_connection(self, websocket, path):
        """Обработка WebSocket подключений"""
        
        self.connected_clients.add(websocket)
        self.logger.info(f"Новое подключение к дашборду: {websocket.remote_address}")
        
        try:
            # Отправляем начальные данные
            await self._send_initial_data(websocket)
            
            # Ожидаем сообщения от клиента
            async for message in websocket:
                await self._handle_client_message(websocket, message)
                
        except websockets.exceptions.ConnectionClosed:
            self.logger.info(f"Подключение закрыто: {websocket.remote_address}")
        finally:
            self.connected_clients.discard(websocket)
    
    async def _send_initial_data(self, websocket):
        """Отправка начальных данных клиенту"""
        
        initial_data = {
            'type': 'initial_data',
            'config': self.dashboard_config,
            'performance_history': [snap.to_dict() for snap in self.performance_history[-100:]],
            'recent_decisions': [asdict(decision) for decision in self.recent_decisions[-20:]],
            'active_alerts': [alert.to_dict() for alert in self.active_alerts]
        }
        
        await websocket.send(json.dumps(initial_data, default=str))
    
    async def _handle_client_message(self, websocket, message):
        """Обработка сообщений от клиента"""
        
        try:
            data = json.loads(message)
            message_type = data.get('type')
            
            if message_type == 'acknowledge_alert':
                alert_id = data.get('alert_id')
                await self._acknowledge_alert(alert_id)
            
            elif message_type == 'request_detailed_report':
                report_type = data.get('report_type', 'daily')
                await self._send_detailed_report(websocket, report_type)
            
            elif message_type == 'update_config':
                new_config = data.get('config', {})
                self.dashboard_config.update(new_config)
                
        except json.JSONDecodeError:
            self.logger.error(f"Некорректное JSON сообщение: {message}")
    
    async def _update_dashboard_data(self):
        """Периодическое обновление данных дашборда"""
        
        while self.is_running:
            try:
                # Генерируем новый снимок производительности
                new_snapshot = await self._generate_performance_snapshot()
                self.performance_history.append(new_snapshot)
                
                # Ограничиваем историю
                if len(self.performance_history) > self.dashboard_config['max_history_points']:
                    self.performance_history = self.performance_history[-self.dashboard_config['max_history_points']:]
                
                # Отправляем обновления всем подключенным клиентам
                if self.connected_clients:
                    update_data = {
                        'type': 'performance_update',
                        'snapshot': new_snapshot.to_dict(),
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    # Отправляем всем клиентам
                    disconnected_clients = set()
                    for client in self.connected_clients:
                        try:
                            await client.send(json.dumps(update_data, default=str))
                        except websockets.exceptions.ConnectionClosed:
                            disconnected_clients.add(client)
                    
                    # Удаляем отключенных клиентов
                    self.connected_clients -= disconnected_clients
                
                await asyncio.sleep(self.update_interval)
                
            except Exception as e:
                self.logger.error(f"Ошибка обновления дашборда: {e}")
                await asyncio.sleep(self.update_interval)
    
    async def _generate_performance_snapshot(self) -> PerformanceSnapshot:
        """Генерация снимка производительности"""
        
        # Симуляция данных (в реальной системе здесь будет получение актуальных данных)
        now = datetime.now()
        
        # Базовые метрики с небольшими колебаниями
        base_win_rate = 0.78
        base_roi = 0.085
        base_drawdown = 0.045
        
        # Добавляем реалистичные колебания
        win_rate = base_win_rate + np.random.normal(0, 0.02)
        roi = base_roi + np.random.normal(0, 0.005)
        drawdown = base_drawdown + np.random.normal(0, 0.01)
        
        # Ограничиваем значения
        win_rate = max(0.5, min(1.0, win_rate))
        roi = max(-0.1, roi)
        drawdown = max(0, min(0.2, drawdown))
        
        return PerformanceSnapshot(
            timestamp=now,
            win_rate=win_rate,
            roi=roi,
            drawdown=drawdown,
            sharpe_ratio=1.5 + np.random.normal(0, 0.1),
            profit_factor=2.2 + np.random.normal(0, 0.2),
            total_trades=len(self.performance_history) * 5 + np.random.randint(0, 10),
            active_positions=np.random.randint(0, 8),
            portfolio_value=100000 * (1 + roi),
            daily_pnl=np.random.normal(500, 200)
        )
    
    async def _acknowledge_alert(self, alert_id: str):
        """Подтверждение алерта"""
        
        for alert in self.active_alerts:
            if alert.id == alert_id:
                alert.acknowledged = True
                self.logger.info(f"Алерт {alert_id} подтвержден")
                break
    
    async def _send_detailed_report(self, websocket, report_type: str):
        """Отправка детального отчета"""
        
        report_generator = DetailedReportGenerator()
        report = await report_generator.generate_report(
            report_type, 
            self.performance_history,
            self.recent_decisions
        )
        
        response = {
            'type': 'detailed_report',
            'report_type': report_type,
            'report': report
        }
        
        await websocket.send(json.dumps(response, default=str))
    
    def add_trading_decision(self, decision: TradingDecision):
        """Добавление торгового решения"""
        
        self.recent_decisions.append(decision)
        
        # Ограничиваем количество решений
        if len(self.recent_decisions) > self.dashboard_config['max_recent_decisions']:
            self.recent_decisions = self.recent_decisions[-self.dashboard_config['max_recent_decisions']:]
    
    def add_alert(self, alert: Alert):
        """Добавление алерта"""
        
        self.active_alerts.append(alert)
        self.logger.warning(f"Новый алерт: {alert.title}")
        
        # Отправляем алерт всем подключенным клиентам
        if self.connected_clients:
            alert_data = {
                'type': 'new_alert',
                'alert': alert.to_dict()
            }
            
            asyncio.create_task(self._broadcast_alert(alert_data))
    
    async def _broadcast_alert(self, alert_data: Dict[str, Any]):
        """Рассылка алерта всем клиентам"""
        
        disconnected_clients = set()
        for client in self.connected_clients:
            try:
                await client.send(json.dumps(alert_data, default=str))
            except websockets.exceptions.ConnectionClosed:
                disconnected_clients.add(client)
        
        self.connected_clients -= disconnected_clients

class DetailedReportGenerator:
    """Генератор детальных отчетов"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    async def generate_report(
        self, 
        report_type: str,
        performance_history: List[PerformanceSnapshot],
        trading_decisions: List[TradingDecision]
    ) -> Dict[str, Any]:
        """Генерация детального отчета"""
        
        if report_type == ReportType.DAILY.value:
            return await self._generate_daily_report(performance_history, trading_decisions)
        elif report_type == ReportType.WEEKLY.value:
            return await self._generate_weekly_report(performance_history, trading_decisions)
        elif report_type == ReportType.MONTHLY.value:
            return await self._generate_monthly_report(performance_history, trading_decisions)
        else:
            return await self._generate_custom_report(performance_history, trading_decisions)
    
    async def _generate_daily_report(
        self, 
        performance_history: List[PerformanceSnapshot],
        trading_decisions: List[TradingDecision]
    ) -> Dict[str, Any]:
        """Генерация дневного отчета"""
        
        # Фильтруем данные за последний день
        yesterday = datetime.now() - timedelta(days=1)
        daily_snapshots = [
            snap for snap in performance_history 
            if snap.timestamp >= yesterday
        ]
        
        daily_decisions = [
            decision for decision in trading_decisions
            if decision.timestamp >= yesterday
        ]
        
        if not daily_snapshots:
            return {'error': 'Недостаточно данных для дневного отчета'}
        
        # Расчет метрик
        latest_snapshot = daily_snapshots[-1]
        first_snapshot = daily_snapshots[0]
        
        daily_roi = latest_snapshot.roi - first_snapshot.roi
        daily_trades = len(daily_decisions)
        
        # Анализ решений
        decision_analysis = self._analyze_trading_decisions(daily_decisions)
        
        # Создание графиков
        charts = await self._create_daily_charts(daily_snapshots, daily_decisions)
        
        return {
            'report_type': 'daily',
            'period': {
                'start': yesterday.isoformat(),
                'end': datetime.now().isoformat()
            },
            'summary': {
                'daily_roi': daily_roi,
                'daily_trades': daily_trades,
                'win_rate': latest_snapshot.win_rate,
                'current_drawdown': latest_snapshot.drawdown,
                'portfolio_value': latest_snapshot.portfolio_value,
                'daily_pnl': latest_snapshot.daily_pnl
            },
            'decision_analysis': decision_analysis,
            'charts': charts,
            'recommendations': self._generate_recommendations(daily_snapshots, daily_decisions)
        }
    
    async def _generate_weekly_report(
        self, 
        performance_history: List[PerformanceSnapshot],
        trading_decisions: List[TradingDecision]
    ) -> Dict[str, Any]:
        """Генерация недельного отчета"""
        
        # Фильтруем данные за последнюю неделю
        week_ago = datetime.now() - timedelta(days=7)
        weekly_snapshots = [
            snap for snap in performance_history 
            if snap.timestamp >= week_ago
        ]
        
        weekly_decisions = [
            decision for decision in trading_decisions
            if decision.timestamp >= week_ago
        ]
        
        if not weekly_snapshots:
            return {'error': 'Недостаточно данных для недельного отчета'}
        
        # Расчет недельных метрик
        latest_snapshot = weekly_snapshots[-1]
        first_snapshot = weekly_snapshots[0]
        
        weekly_roi = latest_snapshot.roi - first_snapshot.roi
        weekly_trades = len(weekly_decisions)
        
        # Анализ по дням недели
        daily_performance = self._analyze_daily_performance(weekly_snapshots)
        
        # Анализ решений
        decision_analysis = self._analyze_trading_decisions(weekly_decisions)
        
        # Создание графиков
        charts = await self._create_weekly_charts(weekly_snapshots, weekly_decisions)
        
        return {
            'report_type': 'weekly',
            'period': {
                'start': week_ago.isoformat(),
                'end': datetime.now().isoformat()
            },
            'summary': {
                'weekly_roi': weekly_roi,
                'weekly_trades': weekly_trades,
                'avg_daily_roi': weekly_roi / 7,
                'best_day': daily_performance['best_day'],
                'worst_day': daily_performance['worst_day'],
                'consistency_score': daily_performance['consistency_score']
            },
            'daily_breakdown': daily_performance['daily_breakdown'],
            'decision_analysis': decision_analysis,
            'charts': charts,
            'recommendations': self._generate_recommendations(weekly_snapshots, weekly_decisions)
        }
    
    async def _generate_monthly_report(
        self, 
        performance_history: List[PerformanceSnapshot],
        trading_decisions: List[TradingDecision]
    ) -> Dict[str, Any]:
        """Генерация месячного отчета"""
        
        # Фильтруем данные за последний месяц
        month_ago = datetime.now() - timedelta(days=30)
        monthly_snapshots = [
            snap for snap in performance_history 
            if snap.timestamp >= month_ago
        ]
        
        monthly_decisions = [
            decision for decision in trading_decisions
            if decision.timestamp >= month_ago
        ]
        
        if not monthly_snapshots:
            return {'error': 'Недостаточно данных для месячного отчета'}
        
        # Расчет месячных метрик
        latest_snapshot = monthly_snapshots[-1]
        first_snapshot = monthly_snapshots[0]
        
        monthly_roi = latest_snapshot.roi - first_snapshot.roi
        monthly_trades = len(monthly_decisions)
        
        # Анализ по неделям
        weekly_performance = self._analyze_weekly_performance(monthly_snapshots)
        
        # Анализ решений
        decision_analysis = self._analyze_trading_decisions(monthly_decisions)
        
        # Создание графиков
        charts = await self._create_monthly_charts(monthly_snapshots, monthly_decisions)
        
        return {
            'report_type': 'monthly',
            'period': {
                'start': month_ago.isoformat(),
                'end': datetime.now().isoformat()
            },
            'summary': {
                'monthly_roi': monthly_roi,
                'monthly_trades': monthly_trades,
                'avg_weekly_roi': monthly_roi / 4,
                'max_drawdown': max(snap.drawdown for snap in monthly_snapshots),
                'sharpe_ratio': latest_snapshot.sharpe_ratio,
                'profit_factor': latest_snapshot.profit_factor
            },
            'weekly_breakdown': weekly_performance,
            'decision_analysis': decision_analysis,
            'charts': charts,
            'recommendations': self._generate_recommendations(monthly_snapshots, monthly_decisions)
        }
    
    def _analyze_trading_decisions(self, decisions: List[TradingDecision]) -> Dict[str, Any]:
        """Анализ торговых решений"""
        
        if not decisions:
            return {'error': 'Нет решений для анализа'}
        
        # Группировка по действиям
        actions = {}
        for decision in decisions:
            action = decision.action
            if action not in actions:
                actions[action] = []
            actions[action].append(decision)
        
        # Анализ уверенности
        confidences = [d.confidence for d in decisions]
        avg_confidence = np.mean(confidences)
        
        # Анализ по символам
        symbols = {}
        for decision in decisions:
            symbol = decision.symbol
            if symbol not in symbols:
                symbols[symbol] = 0
            symbols[symbol] += 1
        
        # Анализ по времени
        hourly_distribution = {}
        for decision in decisions:
            hour = decision.timestamp.hour
            if hour not in hourly_distribution:
                hourly_distribution[hour] = 0
            hourly_distribution[hour] += 1
        
        return {
            'total_decisions': len(decisions),
            'action_distribution': {action: len(decisions) for action, decisions in actions.items()},
            'average_confidence': avg_confidence,
            'confidence_distribution': {
                'high': len([d for d in decisions if d.confidence >= 0.8]),
                'medium': len([d for d in decisions if 0.6 <= d.confidence < 0.8]),
                'low': len([d for d in decisions if d.confidence < 0.6])
            },
            'symbol_distribution': symbols,
            'hourly_distribution': hourly_distribution,
            'most_active_symbol': max(symbols.items(), key=lambda x: x[1])[0] if symbols else None,
            'most_active_hour': max(hourly_distribution.items(), key=lambda x: x[1])[0] if hourly_distribution else None
        }
    
    def _analyze_daily_performance(self, snapshots: List[PerformanceSnapshot]) -> Dict[str, Any]:
        """Анализ дневной производительности"""
        
        # Группировка по дням
        daily_data = {}
        for snapshot in snapshots:
            day = snapshot.timestamp.date()
            if day not in daily_data:
                daily_data[day] = []
            daily_data[day].append(snapshot)
        
        # Расчет дневных ROI
        daily_roi = {}
        for day, day_snapshots in daily_data.items():
            if len(day_snapshots) > 1:
                daily_roi[day] = day_snapshots[-1].roi - day_snapshots[0].roi
            else:
                daily_roi[day] = 0
        
        if not daily_roi:
            return {'error': 'Недостаточно данных для анализа'}
        
        best_day = max(daily_roi.items(), key=lambda x: x[1])
        worst_day = min(daily_roi.items(), key=lambda x: x[1])
        
        # Расчет консистентности
        roi_values = list(daily_roi.values())
        consistency_score = 1 - (np.std(roi_values) / np.mean(roi_values)) if np.mean(roi_values) != 0 else 0
        
        return {
            'daily_breakdown': {str(day): roi for day, roi in daily_roi.items()},
            'best_day': {'date': str(best_day[0]), 'roi': best_day[1]},
            'worst_day': {'date': str(worst_day[0]), 'roi': worst_day[1]},
            'consistency_score': max(0, min(1, consistency_score))
        }
    
    def _analyze_weekly_performance(self, snapshots: List[PerformanceSnapshot]) -> Dict[str, Any]:
        """Анализ недельной производительности"""
        
        # Группировка по неделям
        weekly_data = {}
        for snapshot in snapshots:
            week = snapshot.timestamp.isocalendar()[1]  # Номер недели
            if week not in weekly_data:
                weekly_data[week] = []
            weekly_data[week].append(snapshot)
        
        # Расчет недельных ROI
        weekly_roi = {}
        for week, week_snapshots in weekly_data.items():
            if len(week_snapshots) > 1:
                weekly_roi[week] = week_snapshots[-1].roi - week_snapshots[0].roi
            else:
                weekly_roi[week] = 0
        
        return {
            'weekly_breakdown': weekly_roi,
            'total_weeks': len(weekly_roi),
            'average_weekly_roi': np.mean(list(weekly_roi.values())) if weekly_roi else 0,
            'best_week': max(weekly_roi.items(), key=lambda x: x[1]) if weekly_roi else None,
            'worst_week': min(weekly_roi.items(), key=lambda x: x[1]) if weekly_roi else None
        }
    
    async def _create_daily_charts(
        self, 
        snapshots: List[PerformanceSnapshot],
        decisions: List[TradingDecision]
    ) -> Dict[str, str]:
        """Создание графиков для дневного отчета"""
        
        charts = {}
        
        # График производительности
        timestamps = [snap.timestamp for snap in snapshots]
        roi_values = [snap.roi for snap in snapshots]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=roi_values,
            mode='lines+markers',
            name='ROI',
            line=dict(color='green', width=2)
        ))
        
        fig.update_layout(
            title='Дневная производительность',
            xaxis_title='Время',
            yaxis_title='ROI (%)',
            template='plotly_dark'
        )
        
        charts['performance'] = fig.to_html(include_plotlyjs='cdn')
        
        # График решений
        if decisions:
            decision_times = [d.timestamp for d in decisions]
            decision_confidences = [d.confidence for d in decisions]
            decision_actions = [d.action for d in decisions]
            
            fig2 = go.Figure()
            
            # Разные цвета для разных действий
            colors = {'BUY': 'green', 'SELL': 'red', 'HOLD': 'yellow'}
            for action in set(decision_actions):
                action_times = [t for t, a in zip(decision_times, decision_actions) if a == action]
                action_confidences = [c for c, a in zip(decision_confidences, decision_actions) if a == action]
                
                fig2.add_trace(go.Scatter(
                    x=action_times,
                    y=action_confidences,
                    mode='markers',
                    name=action,
                    marker=dict(color=colors.get(action, 'blue'), size=8)
                ))
            
            fig2.update_layout(
                title='Торговые решения',
                xaxis_title='Время',
                yaxis_title='Уверенность',
                template='plotly_dark'
            )
            
            charts['decisions'] = fig2.to_html(include_plotlyjs='cdn')
        
        return charts
    
    async def _create_weekly_charts(
        self, 
        snapshots: List[PerformanceSnapshot],
        decisions: List[TradingDecision]
    ) -> Dict[str, str]:
        """Создание графиков для недельного отчета"""
        
        charts = {}
        
        # График недельной производительности
        timestamps = [snap.timestamp for snap in snapshots]
        roi_values = [snap.roi for snap in snapshots]
        drawdown_values = [snap.drawdown for snap in snapshots]
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('ROI', 'Drawdown'),
            vertical_spacing=0.1
        )
        
        fig.add_trace(
            go.Scatter(x=timestamps, y=roi_values, name='ROI', line=dict(color='green')),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=timestamps, y=drawdown_values, name='Drawdown', line=dict(color='red')),
            row=2, col=1
        )
        
        fig.update_layout(
            title='Недельная производительность',
            template='plotly_dark',
            height=600
        )
        
        charts['weekly_performance'] = fig.to_html(include_plotlyjs='cdn')
        
        return charts
    
    async def _create_monthly_charts(
        self, 
        snapshots: List[PerformanceSnapshot],
        decisions: List[TradingDecision]
    ) -> Dict[str, str]:
        """Создание графиков для месячного отчета"""
        
        charts = {}
        
        # Комплексный график месячной производительности
        timestamps = [snap.timestamp for snap in snapshots]
        roi_values = [snap.roi for snap in snapshots]
        sharpe_values = [snap.sharpe_ratio for snap in snapshots]
        win_rates = [snap.win_rate for snap in snapshots]
        
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('ROI', 'Sharpe Ratio', 'Win Rate'),
            vertical_spacing=0.08
        )
        
        fig.add_trace(
            go.Scatter(x=timestamps, y=roi_values, name='ROI', line=dict(color='green')),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=timestamps, y=sharpe_values, name='Sharpe Ratio', line=dict(color='blue')),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=timestamps, y=win_rates, name='Win Rate', line=dict(color='orange')),
            row=3, col=1
        )
        
        fig.update_layout(
            title='Месячная производительность',
            template='plotly_dark',
            height=800
        )
        
        charts['monthly_performance'] = fig.to_html(include_plotlyjs='cdn')
        
        return charts
    
    def _generate_recommendations(
        self, 
        snapshots: List[PerformanceSnapshot],
        decisions: List[TradingDecision]
    ) -> List[str]:
        """Генерация рекомендаций на основе анализа"""
        
        recommendations = []
        
        if not snapshots:
            return ["Недостаточно данных для генерации рекомендаций"]
        
        latest_snapshot = snapshots[-1]
        
        # Анализ производительности
        if latest_snapshot.win_rate < 0.7:
            recommendations.append("Рассмотрите пересмотр стратегии - винрейт ниже целевого")
        
        if latest_snapshot.drawdown > 0.1:
            recommendations.append("Усильте управление рисками - просадка превышает допустимый уровень")
        
        if latest_snapshot.sharpe_ratio < 1.0:
            recommendations.append("Оптимизируйте соотношение доходность/риск")
        
        # Анализ решений
        if decisions:
            avg_confidence = np.mean([d.confidence for d in decisions])
            if avg_confidence < 0.7:
                recommendations.append("Повысьте уверенность в торговых решениях через дополнительную валидацию")
        
        # Общие рекомендации
        if len(recommendations) == 0:
            recommendations.append("Система работает в пределах нормы")
            recommendations.append("Продолжайте мониторинг ключевых метрик")
        
        return recommendations

class AlertSystem:
    """Система алертов и уведомлений"""
    
    def __init__(self, dashboard: RealTimeDashboard):
        self.dashboard = dashboard
        self.logger = logging.getLogger(__name__)
        self.alert_rules = self._setup_default_alert_rules()
        self.notification_channels = []
    
    def _setup_default_alert_rules(self) -> List[Dict[str, Any]]:
        """Настройка правил алертов по умолчанию"""
        
        return [
            {
                'name': 'high_drawdown',
                'condition': lambda snapshot: snapshot.drawdown > 0.1,
                'level': AlertLevel.CRITICAL,
                'message': 'Критическая просадка: {drawdown:.1%}'
            },
            {
                'name': 'low_win_rate',
                'condition': lambda snapshot: snapshot.win_rate < 0.6,
                'level': AlertLevel.WARNING,
                'message': 'Низкий винрейт: {win_rate:.1%}'
            },
            {
                'name': 'negative_daily_pnl',
                'condition': lambda snapshot: snapshot.daily_pnl < -1000,
                'level': AlertLevel.WARNING,
                'message': 'Отрицательный дневной P&L: ${daily_pnl:.2f}'
            },
            {
                'name': 'excellent_performance',
                'condition': lambda snapshot: snapshot.roi > 0.1 and snapshot.win_rate > 0.8,
                'level': AlertLevel.SUCCESS,
                'message': 'Отличная производительность: ROI {roi:.1%}, Win Rate {win_rate:.1%}'
            }
        ]
    
    def check_alerts(self, snapshot: PerformanceSnapshot):
        """Проверка условий алертов"""
        
        for rule in self.alert_rules:
            try:
                if rule['condition'](snapshot):
                    alert = Alert(
                        id=f"{rule['name']}_{int(time.time())}",
                        timestamp=datetime.now(),
                        level=rule['level'],
                        title=rule['name'].replace('_', ' ').title(),
                        message=rule['message'].format(**snapshot.to_dict()),
                        source='alert_system',
                        data={'snapshot': snapshot.to_dict()}
                    )
                    
                    self.dashboard.add_alert(alert)
                    
            except Exception as e:
                self.logger.error(f"Ошибка проверки алерта {rule['name']}: {e}")
    
    def add_notification_channel(self, channel: Callable[[Alert], None]):
        """Добавление канала уведомлений"""
        self.notification_channels.append(channel)
    
    def send_notification(self, alert: Alert):
        """Отправка уведомления через все каналы"""
        
        for channel in self.notification_channels:
            try:
                channel(alert)
            except Exception as e:
                self.logger.error(f"Ошибка отправки уведомления: {e}")

class AdvancedReportingSystem:
    """Главный класс улучшенной системы отчетности"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.dashboard = RealTimeDashboard()
        self.report_generator = DetailedReportGenerator()
        self.alert_system = AlertSystem(self.dashboard)
        
        # Настройка уведомлений
        self._setup_notification_channels()
        
        # Планировщик отчетов
        self.scheduled_reports = []
        
    def _setup_notification_channels(self):
        """Настройка каналов уведомлений"""
        
        # Email уведомления (заглушка)
        def email_notification(alert: Alert):
            self.logger.info(f"Email уведомление: {alert.title} - {alert.message}")
        
        # Telegram уведомления (заглушка)
        def telegram_notification(alert: Alert):
            self.logger.info(f"Telegram уведомление: {alert.title} - {alert.message}")
        
        self.alert_system.add_notification_channel(email_notification)
        self.alert_system.add_notification_channel(telegram_notification)
    
    async def start_system(self, dashboard_host: str = "localhost", dashboard_port: int = 8765):
        """Запуск системы отчетности"""
        
        self.logger.info("Запуск улучшенной системы отчетности")
        
        # Запуск дашборда
        dashboard_task = asyncio.create_task(
            self.dashboard.start_dashboard(dashboard_host, dashboard_port)
        )
        
        # Запуск планировщика отчетов
        scheduler_task = asyncio.create_task(self._run_report_scheduler())
        
        await asyncio.gather(dashboard_task, scheduler_task)
    
    async def _run_report_scheduler(self):
        """Планировщик автоматических отчетов"""
        
        while True:
            try:
                current_time = datetime.now()
                
                # Проверяем запланированные отчеты
                for report_config in self.scheduled_reports:
                    if self._should_generate_report(report_config, current_time):
                        await self._generate_scheduled_report(report_config)
                
                # Проверяем алерты
                if self.dashboard.performance_history:
                    latest_snapshot = self.dashboard.performance_history[-1]
                    self.alert_system.check_alerts(latest_snapshot)
                
                await asyncio.sleep(60)  # Проверка каждую минуту
                
            except Exception as e:
                self.logger.error(f"Ошибка в планировщике отчетов: {e}")
                await asyncio.sleep(60)
    
    def _should_generate_report(self, report_config: Dict[str, Any], current_time: datetime) -> bool:
        """Проверка необходимости генерации отчета"""
        
        # Простая логика планирования (можно расширить)
        last_generated = report_config.get('last_generated')
        interval = report_config.get('interval_hours', 24)
        
        if not last_generated:
            return True
        
        time_diff = current_time - last_generated
        return time_diff.total_seconds() >= interval * 3600
    
    async def _generate_scheduled_report(self, report_config: Dict[str, Any]):
        """Генерация запланированного отчета"""
        
        try:
            report_type = report_config['type']
            report = await self.report_generator.generate_report(
                report_type,
                self.dashboard.performance_history,
                self.dashboard.recent_decisions
            )
            
            # Сохранение отчета
            report_file = f"reports/{report_type}_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            Path("reports").mkdir(exist_ok=True)
            
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2, default=str)
            
            report_config['last_generated'] = datetime.now()
            self.logger.info(f"Автоматический отчет {report_type} сохранен: {report_file}")
            
        except Exception as e:
            self.logger.error(f"Ошибка генерации запланированного отчета: {e}")
    
    def schedule_report(self, report_type: str, interval_hours: int = 24):
        """Планирование автоматического отчета"""
        
        report_config = {
            'type': report_type,
            'interval_hours': interval_hours,
            'last_generated': None
        }
        
        self.scheduled_reports.append(report_config)
        self.logger.info(f"Запланирован автоматический отчет {report_type} каждые {interval_hours} часов")
    
    def add_trading_decision(self, decision: TradingDecision):
        """Добавление торгового решения"""
        self.dashboard.add_trading_decision(decision)
    
    def get_dashboard_url(self, host: str = "localhost", port: int = 8765) -> str:
        """Получение URL дашборда"""
        return f"ws://{host}:{port}"

# Функция для демонстрации системы
async def demo_reporting_system():
    """Демонстрация работы системы отчетности"""
    
    # Настройка логирования
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Создание системы отчетности
    reporting_system = AdvancedReportingSystem()
    
    # Планирование автоматических отчетов
    reporting_system.schedule_report('daily', interval_hours=24)
    reporting_system.schedule_report('weekly', interval_hours=168)
    
    # Добавление тестовых торговых решений
    for i in range(10):
        decision = TradingDecision(
            timestamp=datetime.now() - timedelta(hours=i),
            symbol='BTCUSDT',
            action='BUY' if i % 2 == 0 else 'SELL',
            confidence=0.7 + np.random.random() * 0.3,
            reasoning=f"Тестовое решение {i}",
            parameters={'rsi': 30 + i, 'ema_fast': 12},
            market_conditions={'trend': 'bullish', 'volatility': 'medium'}
        )
        reporting_system.add_trading_decision(decision)
    
    print("Система отчетности запущена!")
    print(f"Dashboard URL: {reporting_system.get_dashboard_url()}")
    print("Нажмите Ctrl+C для остановки")
    
    try:
        await reporting_system.start_system()
    except KeyboardInterrupt:
        print("\nСистема отчетности остановлена")

if __name__ == "__main__":
    asyncio.run(demo_reporting_system())