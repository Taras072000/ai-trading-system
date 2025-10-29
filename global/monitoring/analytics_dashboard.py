"""
Система аналитики и дашборда для глобального мониторинга
Предоставляет веб-интерфейс для визуализации метрик и отчетов
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
import aiohttp
from aiohttp import web, web_ws
import aiofiles
import plotly.graph_objects as go
import plotly.express as px
from plotly.utils import PlotlyJSONEncoder
import pandas as pd
import numpy as np
from jinja2 import Environment, FileSystemLoader

class DashboardTheme(Enum):
    DARK = "dark"
    LIGHT = "light"
    AUTO = "auto"

class ChartType(Enum):
    LINE = "line"
    BAR = "bar"
    PIE = "pie"
    SCATTER = "scatter"
    HEATMAP = "heatmap"
    CANDLESTICK = "candlestick"

@dataclass
class DashboardConfig:
    theme: DashboardTheme
    refresh_interval: int  # секунды
    max_data_points: int
    enable_real_time: bool
    chart_height: int
    chart_width: int

@dataclass
class Widget:
    id: str
    title: str
    chart_type: ChartType
    data_source: str
    position: Dict[str, int]  # x, y, width, height
    config: Dict[str, Any]

@dataclass
class Dashboard:
    id: str
    name: str
    description: str
    widgets: List[Widget]
    layout: Dict[str, Any]
    permissions: List[str]

class DataProcessor:
    """Обработчик данных для дашборда"""
    
    def __init__(self):
        self.data_cache = {}
        self.cache_ttl = 300  # 5 минут
    
    async def process_trading_metrics(self, raw_data: Dict) -> Dict:
        """Обработка торговых метрик"""
        processed = {
            "win_rate": {
                "current": raw_data.get("win_rate", 0) * 100,
                "target": 85.0,
                "trend": "up" if raw_data.get("win_rate", 0) > 0.82 else "down"
            },
            "roi": {
                "current": raw_data.get("roi", 0) * 100,
                "target": 35.0,
                "trend": "up" if raw_data.get("roi", 0) > 0.25 else "down"
            },
            "sharpe_ratio": {
                "current": raw_data.get("sharpe_ratio", 0),
                "target": 3.5,
                "trend": "up" if raw_data.get("sharpe_ratio", 0) > 2.85 else "down"
            },
            "max_drawdown": {
                "current": raw_data.get("max_drawdown", 0) * 100,
                "target": 2.0,
                "trend": "down" if raw_data.get("max_drawdown", 0) < 0.021 else "up"
            }
        }
        return processed
    
    async def process_user_metrics(self, raw_data: Dict) -> Dict:
        """Обработка пользовательских метрик"""
        total_users = raw_data.get("total_users", 0)
        target_users = 100000
        
        processed = {
            "total_users": {
                "current": total_users,
                "target": target_users,
                "progress": min(100, (total_users / target_users) * 100),
                "trend": "up"
            },
            "active_users": {
                "current": raw_data.get("active_users_24h", 0),
                "percentage": (raw_data.get("active_users_24h", 0) / max(total_users, 1)) * 100
            },
            "retention_rate": {
                "current": raw_data.get("user_retention_rate", 0) * 100,
                "target": 80.0
            },
            "geographic_distribution": raw_data.get("geographic_distribution", {})
        }
        return processed
    
    async def process_system_metrics(self, raw_data: Dict) -> Dict:
        """Обработка системных метрик"""
        processed = {
            "cpu_usage": {
                "current": raw_data.get("cpu_usage", 0),
                "status": "healthy" if raw_data.get("cpu_usage", 0) < 70 else "warning"
            },
            "memory_usage": {
                "current": raw_data.get("memory_usage", 0),
                "status": "healthy" if raw_data.get("memory_usage", 0) < 80 else "warning"
            },
            "network_latency": {
                "current": raw_data.get("network_latency", 0),
                "status": "healthy" if raw_data.get("network_latency", 0) < 100 else "warning"
            },
            "uptime": {
                "current": raw_data.get("uptime", 0),
                "percentage": min(100, (raw_data.get("uptime", 0) / 86400) * 100)  # 24 часа
            }
        }
        return processed

class ChartGenerator:
    """Генератор графиков"""
    
    def __init__(self):
        self.color_schemes = {
            "trading": ["#00ff88", "#ff4444", "#ffaa00", "#4488ff"],
            "system": ["#44ff44", "#ffff44", "#ff8844", "#ff4444"],
            "users": ["#4488ff", "#88ff44", "#ff8888", "#ffaa88"]
        }
    
    def create_trading_performance_chart(self, data: Dict) -> str:
        """График торговой производительности"""
        metrics = ["win_rate", "roi", "sharpe_ratio"]
        current_values = [data[metric]["current"] for metric in metrics]
        target_values = [data[metric]["target"] for metric in metrics]
        
        fig = go.Figure()
        
        # Текущие значения
        fig.add_trace(go.Bar(
            name='Текущие',
            x=metrics,
            y=current_values,
            marker_color=self.color_schemes["trading"][0]
        ))
        
        # Целевые значения
        fig.add_trace(go.Bar(
            name='Целевые',
            x=metrics,
            y=target_values,
            marker_color=self.color_schemes["trading"][1]
        ))
        
        fig.update_layout(
            title="Торговая производительность",
            xaxis_title="Метрики",
            yaxis_title="Значения",
            barmode='group',
            template="plotly_dark"
        )
        
        return json.dumps(fig, cls=PlotlyJSONEncoder)
    
    def create_user_growth_chart(self, historical_data: List[Dict]) -> str:
        """График роста пользователей"""
        if not historical_data:
            return "{}"
        
        dates = [entry["timestamp"] for entry in historical_data]
        users = [entry["metrics"].get("total_users", 0) for entry in historical_data]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=dates,
            y=users,
            mode='lines+markers',
            name='Общее количество пользователей',
            line=dict(color=self.color_schemes["users"][0], width=3),
            marker=dict(size=6)
        ))
        
        # Добавление целевой линии
        target_line = [100000] * len(dates)
        fig.add_trace(go.Scatter(
            x=dates,
            y=target_line,
            mode='lines',
            name='Цель (100K)',
            line=dict(color=self.color_schemes["users"][1], dash='dash')
        ))
        
        fig.update_layout(
            title="Рост пользовательской базы",
            xaxis_title="Время",
            yaxis_title="Количество пользователей",
            template="plotly_dark"
        )
        
        return json.dumps(fig, cls=PlotlyJSONEncoder)
    
    def create_geographic_distribution_chart(self, geo_data: Dict) -> str:
        """График географического распределения"""
        if not geo_data:
            return "{}"
        
        regions = list(geo_data.keys())
        values = list(geo_data.values())
        
        fig = go.Figure(data=[go.Pie(
            labels=regions,
            values=values,
            hole=0.3,
            marker_colors=self.color_schemes["users"]
        )])
        
        fig.update_layout(
            title="Географическое распределение пользователей",
            template="plotly_dark"
        )
        
        return json.dumps(fig, cls=PlotlyJSONEncoder)
    
    def create_system_health_chart(self, data: Dict) -> str:
        """График состояния системы"""
        metrics = ["cpu_usage", "memory_usage", "network_latency"]
        values = [data[metric]["current"] for metric in metrics]
        colors = [
            self.color_schemes["system"][0] if data[metric]["status"] == "healthy" 
            else self.color_schemes["system"][3] 
            for metric in metrics
        ]
        
        fig = go.Figure(data=[go.Bar(
            x=metrics,
            y=values,
            marker_color=colors
        )])
        
        fig.update_layout(
            title="Состояние системы",
            xaxis_title="Метрики",
            yaxis_title="Значения (%/мс)",
            template="plotly_dark"
        )
        
        return json.dumps(fig, cls=PlotlyJSONEncoder)
    
    def create_volume_heatmap(self, volume_data: Dict) -> str:
        """Тепловая карта торгового объема"""
        # Симуляция данных по часам и дням недели
        hours = list(range(24))
        days = ['Пн', 'Вт', 'Ср', 'Чт', 'Пт', 'Сб', 'Вс']
        
        # Генерация случайных данных объема
        z_data = np.random.uniform(0.5, 2.0, (7, 24))
        
        fig = go.Figure(data=go.Heatmap(
            z=z_data,
            x=hours,
            y=days,
            colorscale='Viridis',
            hoverongaps=False
        ))
        
        fig.update_layout(
            title="Тепловая карта торгового объема",
            xaxis_title="Час дня",
            yaxis_title="День недели",
            template="plotly_dark"
        )
        
        return json.dumps(fig, cls=PlotlyJSONEncoder)

class WebSocketManager:
    """Менеджер WebSocket соединений"""
    
    def __init__(self):
        self.connections: List[web_ws.WebSocketResponse] = []
    
    async def add_connection(self, ws: web_ws.WebSocketResponse):
        """Добавление нового соединения"""
        self.connections.append(ws)
        logging.info(f"New WebSocket connection. Total: {len(self.connections)}")
    
    async def remove_connection(self, ws: web_ws.WebSocketResponse):
        """Удаление соединения"""
        if ws in self.connections:
            self.connections.remove(ws)
            logging.info(f"WebSocket connection removed. Total: {len(self.connections)}")
    
    async def broadcast(self, message: Dict):
        """Рассылка сообщения всем подключенным клиентам"""
        if not self.connections:
            return
        
        message_str = json.dumps(message, default=str)
        disconnected = []
        
        for ws in self.connections:
            try:
                await ws.send_str(message_str)
            except Exception as e:
                logging.warning(f"Failed to send message to WebSocket: {e}")
                disconnected.append(ws)
        
        # Удаление отключенных соединений
        for ws in disconnected:
            await self.remove_connection(ws)

class AnalyticsDashboard:
    """Главный класс аналитического дашборда"""
    
    def __init__(self, monitoring_system=None):
        self.monitoring_system = monitoring_system
        self.data_processor = DataProcessor()
        self.chart_generator = ChartGenerator()
        self.ws_manager = WebSocketManager()
        self.app = web.Application()
        self.config = DashboardConfig(
            theme=DashboardTheme.DARK,
            refresh_interval=30,
            max_data_points=1000,
            enable_real_time=True,
            chart_height=400,
            chart_width=600
        )
        
        # Настройка маршрутов
        self._setup_routes()
        
        # Настройка Jinja2
        self.jinja_env = Environment(
            loader=FileSystemLoader('templates'),
            enable_async=True
        )
    
    def _setup_routes(self):
        """Настройка маршрутов"""
        self.app.router.add_get('/', self.index_handler)
        self.app.router.add_get('/api/status', self.status_handler)
        self.app.router.add_get('/api/metrics', self.metrics_handler)
        self.app.router.add_get('/api/charts/{chart_type}', self.chart_handler)
        self.app.router.add_get('/api/reports/{report_type}', self.report_handler)
        self.app.router.add_get('/ws', self.websocket_handler)
        self.app.router.add_static('/static', 'static')
    
    async def index_handler(self, request):
        """Главная страница дашборда"""
        html_content = """
        <!DOCTYPE html>
        <html lang="ru">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Peper Binance v4 - Global Monitoring Dashboard</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
            <style>
                body {
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    margin: 0;
                    padding: 0;
                    background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
                    color: white;
                    min-height: 100vh;
                }
                .header {
                    background: rgba(0,0,0,0.3);
                    padding: 20px;
                    text-align: center;
                    border-bottom: 2px solid #00ff88;
                }
                .header h1 {
                    margin: 0;
                    font-size: 2.5em;
                    text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
                }
                .status-bar {
                    background: rgba(0,0,0,0.2);
                    padding: 10px 20px;
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                }
                .status-item {
                    display: flex;
                    align-items: center;
                    gap: 10px;
                }
                .status-indicator {
                    width: 12px;
                    height: 12px;
                    border-radius: 50%;
                    background: #00ff88;
                    animation: pulse 2s infinite;
                }
                @keyframes pulse {
                    0% { opacity: 1; }
                    50% { opacity: 0.5; }
                    100% { opacity: 1; }
                }
                .dashboard-grid {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
                    gap: 20px;
                    padding: 20px;
                }
                .chart-container {
                    background: rgba(255,255,255,0.1);
                    border-radius: 15px;
                    padding: 20px;
                    backdrop-filter: blur(10px);
                    border: 1px solid rgba(255,255,255,0.2);
                    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
                }
                .chart-title {
                    font-size: 1.2em;
                    margin-bottom: 15px;
                    text-align: center;
                    color: #00ff88;
                }
                .metrics-summary {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 15px;
                    margin-bottom: 20px;
                }
                .metric-card {
                    background: rgba(0,0,0,0.3);
                    padding: 15px;
                    border-radius: 10px;
                    text-align: center;
                    border-left: 4px solid #00ff88;
                }
                .metric-value {
                    font-size: 2em;
                    font-weight: bold;
                    color: #00ff88;
                }
                .metric-label {
                    font-size: 0.9em;
                    opacity: 0.8;
                    margin-top: 5px;
                }
                .loading {
                    text-align: center;
                    padding: 50px;
                    font-size: 1.2em;
                }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🚀 Peper Binance v4 - Phase 5 Global Monitoring</h1>
                <p>Глобальная система мониторинга и аналитики</p>
            </div>
            
            <div class="status-bar">
                <div class="status-item">
                    <div class="status-indicator"></div>
                    <span>Система активна</span>
                </div>
                <div class="status-item">
                    <span id="last-update">Последнее обновление: загрузка...</span>
                </div>
                <div class="status-item">
                    <span id="active-regions">Активные регионы: загрузка...</span>
                </div>
            </div>
            
            <div class="metrics-summary" id="metrics-summary">
                <div class="loading">Загрузка метрик...</div>
            </div>
            
            <div class="dashboard-grid">
                <div class="chart-container">
                    <div class="chart-title">Торговая производительность</div>
                    <div id="trading-performance-chart"></div>
                </div>
                
                <div class="chart-container">
                    <div class="chart-title">Рост пользователей</div>
                    <div id="user-growth-chart"></div>
                </div>
                
                <div class="chart-container">
                    <div class="chart-title">Географическое распределение</div>
                    <div id="geographic-chart"></div>
                </div>
                
                <div class="chart-container">
                    <div class="chart-title">Состояние системы</div>
                    <div id="system-health-chart"></div>
                </div>
                
                <div class="chart-container">
                    <div class="chart-title">Тепловая карта объемов</div>
                    <div id="volume-heatmap"></div>
                </div>
            </div>
            
            <script>
                // WebSocket соединение
                const ws = new WebSocket('ws://localhost:8080/ws');
                
                ws.onopen = function(event) {
                    console.log('WebSocket connected');
                };
                
                ws.onmessage = function(event) {
                    const data = JSON.parse(event.data);
                    updateDashboard(data);
                };
                
                // Функция обновления дашборда
                function updateDashboard(data) {
                    if (data.type === 'metrics_update') {
                        updateMetricsSummary(data.metrics);
                        updateCharts(data.charts);
                    }
                    
                    if (data.type === 'status_update') {
                        updateStatusBar(data.status);
                    }
                }
                
                function updateMetricsSummary(metrics) {
                    const summaryDiv = document.getElementById('metrics-summary');
                    summaryDiv.innerHTML = `
                        <div class="metric-card">
                            <div class="metric-value">${metrics.win_rate}%</div>
                            <div class="metric-label">Win Rate</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">${metrics.total_users}</div>
                            <div class="metric-label">Пользователи</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">$${(metrics.volume_24h / 1e9).toFixed(1)}B</div>
                            <div class="metric-label">Объем 24ч</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">${metrics.roi}%</div>
                            <div class="metric-label">ROI</div>
                        </div>
                    `;
                }
                
                function updateCharts(charts) {
                    if (charts.trading_performance) {
                        Plotly.newPlot('trading-performance-chart', JSON.parse(charts.trading_performance));
                    }
                    if (charts.user_growth) {
                        Plotly.newPlot('user-growth-chart', JSON.parse(charts.user_growth));
                    }
                    if (charts.geographic) {
                        Plotly.newPlot('geographic-chart', JSON.parse(charts.geographic));
                    }
                    if (charts.system_health) {
                        Plotly.newPlot('system-health-chart', JSON.parse(charts.system_health));
                    }
                    if (charts.volume_heatmap) {
                        Plotly.newPlot('volume-heatmap', JSON.parse(charts.volume_heatmap));
                    }
                }
                
                function updateStatusBar(status) {
                    document.getElementById('last-update').textContent = 
                        `Последнее обновление: ${new Date(status.last_update).toLocaleString()}`;
                    document.getElementById('active-regions').textContent = 
                        `Активные регионы: ${status.active_regions}`;
                }
                
                // Начальная загрузка данных
                async function loadInitialData() {
                    try {
                        const response = await fetch('/api/metrics');
                        const data = await response.json();
                        updateDashboard({
                            type: 'metrics_update',
                            metrics: data.summary,
                            charts: data.charts
                        });
                        
                        const statusResponse = await fetch('/api/status');
                        const statusData = await statusResponse.json();
                        updateStatusBar(statusData);
                    } catch (error) {
                        console.error('Error loading initial data:', error);
                    }
                }
                
                // Загрузка при старте
                loadInitialData();
                
                // Периодическое обновление
                setInterval(loadInitialData, 30000); // каждые 30 секунд
            </script>
        </body>
        </html>
        """
        return web.Response(text=html_content, content_type='text/html')
    
    async def status_handler(self, request):
        """API статуса системы"""
        if self.monitoring_system:
            status = await self.monitoring_system.get_current_status()
        else:
            status = {
                "system_status": "demo",
                "active_regions": 5,
                "total_alerts": 0,
                "critical_alerts": 0,
                "last_update": datetime.now().isoformat()
            }
        
        return web.json_response(status)
    
    async def metrics_handler(self, request):
        """API метрик"""
        # Симуляция данных для демонстрации
        raw_metrics = {
            "win_rate": np.random.uniform(0.82, 0.87),
            "roi": np.random.uniform(0.25, 0.35),
            "sharpe_ratio": np.random.uniform(2.8, 3.5),
            "max_drawdown": np.random.uniform(0.015, 0.025),
            "total_users": np.random.randint(85000, 105000),
            "active_users_24h": np.random.randint(15000, 25000),
            "user_retention_rate": np.random.uniform(0.75, 0.85),
            "volume_24h": np.random.uniform(8e9, 12e9),
            "cpu_usage": np.random.uniform(30, 70),
            "memory_usage": np.random.uniform(40, 80),
            "network_latency": np.random.uniform(20, 100),
            "uptime": 86400 * 0.99,
            "geographic_distribution": {
                "Americas": np.random.randint(25000, 35000),
                "Europe": np.random.randint(20000, 30000),
                "Asia Pacific": np.random.randint(15000, 25000),
                "Others": np.random.randint(5000, 15000)
            }
        }
        
        # Обработка данных
        trading_data = await self.data_processor.process_trading_metrics(raw_metrics)
        user_data = await self.data_processor.process_user_metrics(raw_metrics)
        system_data = await self.data_processor.process_system_metrics(raw_metrics)
        
        # Генерация графиков
        charts = {
            "trading_performance": self.chart_generator.create_trading_performance_chart(trading_data),
            "user_growth": self.chart_generator.create_user_growth_chart([]),  # Пустые исторические данные
            "geographic": self.chart_generator.create_geographic_distribution_chart(
                raw_metrics["geographic_distribution"]
            ),
            "system_health": self.chart_generator.create_system_health_chart(system_data),
            "volume_heatmap": self.chart_generator.create_volume_heatmap({})
        }
        
        response_data = {
            "summary": {
                "win_rate": f"{trading_data['win_rate']['current']:.1f}",
                "total_users": f"{user_data['total_users']['current']:,}",
                "volume_24h": raw_metrics["volume_24h"],
                "roi": f"{trading_data['roi']['current']:.1f}"
            },
            "charts": charts,
            "timestamp": datetime.now().isoformat()
        }
        
        return web.json_response(response_data)
    
    async def chart_handler(self, request):
        """API отдельных графиков"""
        chart_type = request.match_info['chart_type']
        
        # Здесь можно реализовать генерацию конкретных графиков
        return web.json_response({"chart_type": chart_type, "data": "chart_data"})
    
    async def report_handler(self, request):
        """API отчетов"""
        report_type = request.match_info['report_type']
        
        if self.monitoring_system:
            try:
                report = await self.monitoring_system.generate_custom_report(report_type)
                return web.json_response(report)
            except Exception as e:
                return web.json_response({"error": str(e)}, status=400)
        else:
            return web.json_response({"error": "Monitoring system not available"}, status=503)
    
    async def websocket_handler(self, request):
        """WebSocket обработчик"""
        ws = web_ws.WebSocketResponse()
        await ws.prepare(request)
        
        await self.ws_manager.add_connection(ws)
        
        try:
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    # Обработка сообщений от клиента
                    if data.get('type') == 'subscribe':
                        # Подписка на обновления
                        pass
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logging.error(f'WebSocket error: {ws.exception()}')
        except Exception as e:
            logging.error(f"WebSocket handler error: {e}")
        finally:
            await self.ws_manager.remove_connection(ws)
        
        return ws
    
    async def start_real_time_updates(self):
        """Запуск обновлений в реальном времени"""
        while True:
            try:
                if self.config.enable_real_time:
                    # Получение свежих данных
                    response = await self.metrics_handler(None)
                    if hasattr(response, 'text'):
                        data = json.loads(await response.text())
                        
                        # Отправка обновлений через WebSocket
                        await self.ws_manager.broadcast({
                            "type": "metrics_update",
                            "metrics": data["summary"],
                            "charts": data["charts"],
                            "timestamp": datetime.now().isoformat()
                        })
                
                await asyncio.sleep(self.config.refresh_interval)
                
            except Exception as e:
                logging.error(f"Error in real-time updates: {e}")
                await asyncio.sleep(60)
    
    async def start_server(self, host='localhost', port=8080):
        """Запуск веб-сервера"""
        # Запуск обновлений в реальном времени
        asyncio.create_task(self.start_real_time_updates())
        
        runner = web.AppRunner(self.app)
        await runner.setup()
        site = web.TCPSite(runner, host, port)
        await site.start()
        
        logging.info(f"Analytics Dashboard started at http://{host}:{port}")
        return runner

# Пример использования
async def main():
    """Пример запуска дашборда"""
    dashboard = AnalyticsDashboard()
    
    try:
        runner = await dashboard.start_server()
        
        # Работа сервера
        while True:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        logging.info("Shutting down dashboard...")
        await runner.cleanup()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())