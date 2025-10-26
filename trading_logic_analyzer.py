"""
🔍 АНАЛИЗАТОР ТОРГОВОЙ ЛОГИКИ
Система диагностики и калибровки торговых решений с детальной аналитикой каждого этапа

Автор: AI Trading System
Дата: 2024
"""

import asyncio
import pandas as pd
import numpy as np
import json
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import warnings
warnings.filterwarnings('ignore')

# Импортируем существующие компоненты
from winrate_test_with_results2 import RealWinrateTester, TestConfig, WinrateTestResult
from mock_ai_trading_system import MockTradingSystem, MockAIOrchestrator

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Настройка matplotlib для русского языка
plt.rcParams['font.family'] = ['Arial Unicode MS', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False

@dataclass
class SignalAnalysis:
    """Анализ качества сигналов AI модели"""
    model_name: str
    total_signals: int
    buy_signals: int
    sell_signals: int
    hold_signals: int
    avg_confidence: float
    confidence_distribution: Dict[str, int]  # Распределение по диапазонам уверенности
    signal_accuracy: float  # Точность сигналов (если есть данные)
    bias_score: float  # Склонность модели (-1 медвежья, 0 нейтральная, 1 бычья)

@dataclass
class ConsensusAnalysis:
    """Анализ логики консенсуса"""
    total_consensus_attempts: int
    successful_consensus: int
    consensus_rate: float
    avg_models_participating: float
    avg_consensus_confidence: float
    consensus_by_strength: Dict[int, int]  # Распределение по количеству участвующих моделей
    action_distribution: Dict[str, int]  # Распределение действий (BUY/SELL/HOLD)

@dataclass
class FilterAnalysis:
    """Анализ эффективности фильтров"""
    filter_name: str
    total_checks: int
    passed_checks: int
    blocked_checks: int
    pass_rate: float
    impact_on_performance: float  # Влияние на производительность
    false_positives: int  # Заблокированные хорошие сигналы
    true_positives: int   # Заблокированные плохие сигналы

@dataclass
class EntryExitAnalysis:
    """Анализ условий входа и выхода"""
    total_entries: int
    successful_entries: int  # Прибыльные входы
    entry_success_rate: float
    avg_holding_time: float
    exit_reasons: Dict[str, int]  # Распределение причин выхода
    avg_profit_per_exit_reason: Dict[str, float]
    timing_analysis: Dict[str, Any]  # Анализ времени входа/выхода

@dataclass
class RiskManagementAnalysis:
    """Анализ управления рисками"""
    total_trades: int
    stop_loss_triggered: int
    take_profit_triggered: int
    max_drawdown_reached: int
    risk_reward_ratio: float
    avg_risk_per_trade: float
    position_sizing_effectiveness: float

@dataclass
class ComponentAnalysisResult:
    """Результат анализа компонента торговой системы"""
    component_name: str
    analysis_data: Any
    performance_score: float  # 0-100
    issues_found: List[str]
    recommendations: List[str]
    charts_generated: List[str]

class TradingLogicAnalyzer:
    """
    🔍 АНАЛИЗАТОР ТОРГОВОЙ ЛОГИКИ
    
    Разбивает торговую систему на составляющие и анализирует каждый компонент:
    1. Генерация сигналов AI моделей
    2. Логика консенсуса
    3. Условия входа в сделку
    4. Условия выхода из сделки
    5. Управление рисками
    6. Эффективность фильтров
    """
    
    def __init__(self, config: TestConfig):
        self.config = config
        self.mock_system = MockTradingSystem(config)
        self.analysis_results = {}
        self.charts_dir = "trading_logic_analysis"
        
        # Создаем директорию для графиков
        import os
        os.makedirs(self.charts_dir, exist_ok=True)
        
        logger.info("🔍 Инициализирован анализатор торговой логики")
    
    async def run_full_analysis(self) -> Dict[str, ComponentAnalysisResult]:
        """
        🚀 ЗАПУСК ПОЛНОГО АНАЛИЗА ТОРГОВОЙ ЛОГИКИ
        
        Этапы:
        1. Сбор данных с mock системы
        2. Анализ каждого компонента
        3. Генерация визуализаций
        4. Формирование рекомендаций
        """
        logger.info("🚀 Запуск полного анализа торговой логики...")
        
        # Собираем данные для анализа
        analysis_data = await self._collect_analysis_data()
        
        # Анализируем каждый компонент
        results = {}
        
        # 1. Анализ генерации сигналов
        logger.info("📊 Анализ генерации сигналов AI моделей...")
        results['signal_generation'] = await self.analyze_signal_generation(analysis_data)
        
        # 2. Анализ логики консенсуса
        logger.info("🤝 Анализ логики консенсуса...")
        results['consensus_logic'] = await self.analyze_consensus_logic(analysis_data)
        
        # 3. Анализ условий входа
        logger.info("📈 Анализ условий входа...")
        results['entry_conditions'] = await self.analyze_entry_conditions(analysis_data)
        
        # 4. Анализ условий выхода
        logger.info("📉 Анализ условий выхода...")
        results['exit_conditions'] = await self.analyze_exit_conditions(analysis_data)
        
        # 5. Анализ управления рисками
        logger.info("⚖️ Анализ управления рисками...")
        results['risk_management'] = await self.analyze_risk_management(analysis_data)
        
        # 6. Анализ фильтров
        logger.info("🔧 Анализ эффективности фильтров...")
        results['filters'] = await self.analyze_filters(analysis_data)
        
        # Генерируем сводный отчет
        await self._generate_summary_report(results)
        
        logger.info("✅ Полный анализ торговой логики завершен!")
        
        return results
    
    async def _collect_analysis_data(self) -> Dict[str, Any]:
        """Сбор данных для анализа"""
        logger.info("📊 Сбор данных для анализа...")
        
        data = {
            'signals': [],
            'consensus_attempts': [],
            'trades': [],
            'filter_checks': [],
            'market_data': {}
        }
        
        # Запускаем mock систему для сбора данных
        for symbol in self.config.symbols:
            logger.info(f"📈 Сбор данных для {symbol}...")
            
            # Загружаем исторические данные
            market_data = await self.mock_system._load_mock_historical_data(symbol)
            data['market_data'][symbol] = market_data
            
            # Симулируем торговлю с детальным логированием
            symbol_data = await self._simulate_with_logging(symbol, market_data)
            
            data['signals'].extend(symbol_data['signals'])
            data['consensus_attempts'].extend(symbol_data['consensus_attempts'])
            data['trades'].extend(symbol_data['trades'])
            data['filter_checks'].extend(symbol_data['filter_checks'])
        
        logger.info(f"📊 Собрано данных: {len(data['signals'])} сигналов, "
                   f"{len(data['consensus_attempts'])} попыток консенсуса, "
                   f"{len(data['trades'])} сделок")
        
        return data
    
    async def _simulate_with_logging(self, symbol: str, data: pd.DataFrame) -> Dict[str, List]:
        """Симуляция торговли с детальным логированием"""
        
        simulation_data = {
            'signals': [],
            'consensus_attempts': [],
            'trades': [],
            'filter_checks': []
        }
        
        position = None
        
        # Проходим по данным с окном
        for i in range(20, len(data) - 1):
            current_data = data.iloc[:i+1]
            current_price = current_data['close'].iloc[-1]
            current_time = current_data['timestamp'].iloc[-1]
            
            # Получаем сигналы от AI моделей с логированием
            signals = await self.mock_system.ai_orchestrator.get_all_signals(symbol, current_data)
            
            # Логируем каждый сигнал
            for model_name, signal in signals.items():
                simulation_data['signals'].append({
                    'symbol': symbol,
                    'timestamp': current_time,
                    'model_name': model_name,
                    'action': signal.action,
                    'confidence': signal.confidence,
                    'reasoning': signal.reasoning,
                    'price': current_price
                })
            
            # Формируем консенсус с логированием
            consensus = self.mock_system._form_consensus(signals)
            
            # Логируем попытку консенсуса
            simulation_data['consensus_attempts'].append({
                'symbol': symbol,
                'timestamp': current_time,
                'signals_count': len(signals),
                'consensus_formed': consensus is not None,
                'consensus_action': consensus['action'] if consensus else None,
                'consensus_confidence': consensus['confidence'] if consensus else None,
                'participating_models': consensus['total_models'] if consensus else 0,
                'price': current_price
            })
            
            if consensus is None:
                continue
            
            # Проверяем фильтры с логированием
            filter_result = self._check_filters_with_logging(consensus, current_data, symbol, current_time)
            simulation_data['filter_checks'].extend(filter_result)
            
            # Логика входа и выхода (упрощенная для анализа)
            if position is None and consensus['action'] in ['BUY', 'SELL']:
                if self.mock_system._check_entry_conditions(consensus, current_data):
                    # Открываем позицию
                    position = {
                        'symbol': symbol,
                        'action': consensus['action'],
                        'entry_price': current_price,
                        'entry_time': current_time,
                        'size': self.config.position_size_percent / 100 * self.mock_system.balance / current_price,
                        'consensus': consensus
                    }
            
            elif position is not None:
                exit_reason = self.mock_system._check_exit_conditions(position, current_price, current_time)
                if exit_reason:
                    # Закрываем позицию
                    trade = self.mock_system._close_position(position, current_price, current_time, exit_reason)
                    simulation_data['trades'].append(trade)
                    position = None
        
        # Закрываем открытую позицию в конце
        if position is not None:
            final_price = data['close'].iloc[-1]
            final_time = data['timestamp'].iloc[-1]
            trade = self.mock_system._close_position(position, final_price, final_time, "Конец теста")
            simulation_data['trades'].append(trade)
        
        return simulation_data
    
    def _check_filters_with_logging(self, consensus: Dict[str, Any], data: pd.DataFrame, 
                                   symbol: str, timestamp: datetime) -> List[Dict[str, Any]]:
        """Проверка фильтров с детальным логированием"""
        
        filter_checks = []
        
        # Фильтр минимальной уверенности
        confidence_check = {
            'symbol': symbol,
            'timestamp': timestamp,
            'filter_name': 'min_confidence',
            'value': consensus['confidence'],
            'threshold': self.config.min_confidence,
            'passed': consensus['confidence'] >= self.config.min_confidence,
            'impact': 'entry_block' if consensus['confidence'] < self.config.min_confidence else 'entry_allow'
        }
        filter_checks.append(confidence_check)
        
        # Фильтр волатильности
        if self.config.min_volatility > 0:
            volatility = data['close'].pct_change().tail(10).std()
            volatility_check = {
                'symbol': symbol,
                'timestamp': timestamp,
                'filter_name': 'min_volatility',
                'value': volatility,
                'threshold': self.config.min_volatility,
                'passed': volatility >= self.config.min_volatility,
                'impact': 'entry_block' if volatility < self.config.min_volatility else 'entry_allow'
            }
            filter_checks.append(volatility_check)
        
        # Фильтр объема
        if self.config.min_volume_ratio > 0 and 'volume' in data.columns:
            recent_volume = data['volume'].tail(5).mean()
            avg_volume = data['volume'].tail(20).mean()
            volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 0
            volume_check = {
                'symbol': symbol,
                'timestamp': timestamp,
                'filter_name': 'min_volume_ratio',
                'value': volume_ratio,
                'threshold': self.config.min_volume_ratio,
                'passed': volume_ratio >= self.config.min_volume_ratio,
                'impact': 'entry_block' if volume_ratio < self.config.min_volume_ratio else 'entry_allow'
            }
            filter_checks.append(volume_check)
        
        return filter_checks
    
    async def analyze_signal_generation(self, data: Dict[str, Any]) -> ComponentAnalysisResult:
        """Анализ генерации сигналов AI моделей"""
        
        signals_df = pd.DataFrame(data['signals'])
        
        if signals_df.empty:
            return ComponentAnalysisResult(
                component_name="Signal Generation",
                analysis_data=None,
                performance_score=0,
                issues_found=["Нет сигналов для анализа"],
                recommendations=["Проверить работу AI моделей"],
                charts_generated=[]
            )
        
        # Анализ по моделям
        model_analyses = {}
        
        for model_name in signals_df['model_name'].unique():
            model_signals = signals_df[signals_df['model_name'] == model_name]
            
            # Распределение уверенности
            confidence_bins = pd.cut(model_signals['confidence'], 
                                   bins=[0, 0.3, 0.5, 0.7, 0.9, 1.0], 
                                   labels=['Низкая', 'Средняя-', 'Средняя', 'Высокая', 'Очень высокая'])
            confidence_dist = confidence_bins.value_counts().to_dict()
            
            # Склонность модели
            action_counts = model_signals['action'].value_counts()
            buy_ratio = action_counts.get('BUY', 0) / len(model_signals)
            sell_ratio = action_counts.get('SELL', 0) / len(model_signals)
            bias_score = buy_ratio - sell_ratio  # От -1 (медвежья) до 1 (бычья)
            
            analysis = SignalAnalysis(
                model_name=model_name,
                total_signals=len(model_signals),
                buy_signals=action_counts.get('BUY', 0),
                sell_signals=action_counts.get('SELL', 0),
                hold_signals=action_counts.get('HOLD', 0),
                avg_confidence=model_signals['confidence'].mean(),
                confidence_distribution=confidence_dist,
                signal_accuracy=0.0,  # Требует дополнительных данных
                bias_score=bias_score
            )
            
            model_analyses[model_name] = analysis
        
        # Создаем визуализации
        charts = await self._create_signal_charts(signals_df, model_analyses)
        
        # Оценка производительности
        performance_score = self._calculate_signal_performance_score(model_analyses)
        
        # Выявление проблем
        issues = self._identify_signal_issues(model_analyses)
        
        # Рекомендации
        recommendations = self._generate_signal_recommendations(model_analyses, issues)
        
        return ComponentAnalysisResult(
            component_name="Signal Generation",
            analysis_data=model_analyses,
            performance_score=performance_score,
            issues_found=issues,
            recommendations=recommendations,
            charts_generated=charts
        )
    
    async def analyze_consensus_logic(self, data: Dict[str, Any]) -> ComponentAnalysisResult:
        """Анализ логики консенсуса"""
        
        consensus_df = pd.DataFrame(data['consensus_attempts'])
        
        if consensus_df.empty:
            return ComponentAnalysisResult(
                component_name="Consensus Logic",
                analysis_data=None,
                performance_score=0,
                issues_found=["Нет данных о консенсусе"],
                recommendations=["Проверить логику формирования консенсуса"],
                charts_generated=[]
            )
        
        # Анализ консенсуса
        total_attempts = len(consensus_df)
        successful = len(consensus_df[consensus_df['consensus_formed'] == True])
        consensus_rate = successful / total_attempts if total_attempts > 0 else 0
        
        successful_consensus = consensus_df[consensus_df['consensus_formed'] == True]
        
        avg_models = successful_consensus['participating_models'].mean() if not successful_consensus.empty else 0
        avg_confidence = successful_consensus['consensus_confidence'].mean() if not successful_consensus.empty else 0
        
        # Распределение по силе консенсуса
        consensus_strength_dist = successful_consensus['participating_models'].value_counts().to_dict()
        
        # Распределение действий
        action_dist = successful_consensus['consensus_action'].value_counts().to_dict()
        
        analysis = ConsensusAnalysis(
            total_consensus_attempts=total_attempts,
            successful_consensus=successful,
            consensus_rate=consensus_rate,
            avg_models_participating=avg_models,
            avg_consensus_confidence=avg_confidence,
            consensus_by_strength=consensus_strength_dist,
            action_distribution=action_dist
        )
        
        # Создаем визуализации
        charts = await self._create_consensus_charts(consensus_df, analysis)
        
        # Оценка производительности
        performance_score = self._calculate_consensus_performance_score(analysis)
        
        # Выявление проблем
        issues = self._identify_consensus_issues(analysis)
        
        # Рекомендации
        recommendations = self._generate_consensus_recommendations(analysis, issues)
        
        return ComponentAnalysisResult(
            component_name="Consensus Logic",
            analysis_data=analysis,
            performance_score=performance_score,
            issues_found=issues,
            recommendations=recommendations,
            charts_generated=charts
        )
    
    async def analyze_entry_conditions(self, data: Dict[str, Any]) -> ComponentAnalysisResult:
        """Анализ условий входа в сделку"""
        
        trades_df = pd.DataFrame(data['trades'])
        
        if trades_df.empty:
            return ComponentAnalysisResult(
                component_name="Entry Conditions",
                analysis_data=None,
                performance_score=0,
                issues_found=["Нет сделок для анализа"],
                recommendations=["Проверить условия входа в сделку"],
                charts_generated=[]
            )
        
        # Анализ входов
        total_entries = len(trades_df)
        successful_entries = len(trades_df[trades_df['pnl'] > 0])
        entry_success_rate = successful_entries / total_entries if total_entries > 0 else 0
        
        # Время удержания
        avg_holding_time = trades_df['holding_time_hours'].mean()
        
        # Анализ времени входа
        trades_df['entry_hour'] = pd.to_datetime(trades_df['entry_time']).dt.hour
        hourly_performance = trades_df.groupby('entry_hour')['pnl'].agg(['count', 'mean', 'sum']).to_dict()
        
        analysis = EntryExitAnalysis(
            total_entries=total_entries,
            successful_entries=successful_entries,
            entry_success_rate=entry_success_rate,
            avg_holding_time=avg_holding_time,
            exit_reasons={},  # Будет заполнено в analyze_exit_conditions
            avg_profit_per_exit_reason={},
            timing_analysis=hourly_performance
        )
        
        # Создаем визуализации
        charts = await self._create_entry_charts(trades_df, analysis)
        
        # Оценка производительности
        performance_score = entry_success_rate * 100
        
        # Выявление проблем
        issues = []
        if entry_success_rate < 0.4:
            issues.append(f"Низкий процент успешных входов: {entry_success_rate:.1%}")
        if avg_holding_time > self.config.max_hold_hours * 0.8:
            issues.append(f"Слишком долгое удержание позиций: {avg_holding_time:.1f} часов")
        
        # Рекомендации
        recommendations = []
        if entry_success_rate < 0.4:
            recommendations.append("Ужесточить условия входа в сделку")
            recommendations.append("Добавить дополнительные фильтры качества сигналов")
        if avg_holding_time > self.config.max_hold_hours * 0.8:
            recommendations.append("Сократить максимальное время удержания")
            recommendations.append("Добавить более агрессивные условия выхода")
        
        return ComponentAnalysisResult(
            component_name="Entry Conditions",
            analysis_data=analysis,
            performance_score=performance_score,
            issues_found=issues,
            recommendations=recommendations,
            charts_generated=charts
        )
    
    async def analyze_exit_conditions(self, data: Dict[str, Any]) -> ComponentAnalysisResult:
        """Анализ условий выхода из сделки"""
        
        trades_df = pd.DataFrame(data['trades'])
        
        if trades_df.empty:
            return ComponentAnalysisResult(
                component_name="Exit Conditions",
                analysis_data=None,
                performance_score=0,
                issues_found=["Нет сделок для анализа"],
                recommendations=["Проверить условия выхода из сделки"],
                charts_generated=[]
            )
        
        # Анализ причин выхода
        exit_reasons = trades_df['exit_reason'].value_counts().to_dict()
        
        # Средняя прибыль по причинам выхода
        avg_profit_per_reason = trades_df.groupby('exit_reason')['pnl'].mean().to_dict()
        
        # Анализ времени выхода
        trades_df['exit_hour'] = pd.to_datetime(trades_df['exit_time']).dt.hour
        hourly_exit_performance = trades_df.groupby('exit_hour')['pnl'].agg(['count', 'mean']).to_dict()
        
        analysis = EntryExitAnalysis(
            total_entries=len(trades_df),
            successful_entries=len(trades_df[trades_df['pnl'] > 0]),
            entry_success_rate=len(trades_df[trades_df['pnl'] > 0]) / len(trades_df),
            avg_holding_time=trades_df['holding_time_hours'].mean(),
            exit_reasons=exit_reasons,
            avg_profit_per_exit_reason=avg_profit_per_reason,
            timing_analysis=hourly_exit_performance
        )
        
        # Создаем визуализации
        charts = await self._create_exit_charts(trades_df, analysis)
        
        # Оценка производительности
        tp_ratio = exit_reasons.get('Take Profit', 0) / len(trades_df) if len(trades_df) > 0 else 0
        sl_ratio = exit_reasons.get('Stop Loss', 0) / len(trades_df) if len(trades_df) > 0 else 0
        performance_score = tp_ratio * 100 - sl_ratio * 50  # TP хорошо, SL плохо
        
        # Выявление проблем
        issues = []
        if sl_ratio > 0.6:
            issues.append(f"Слишком много срабатываний Stop Loss: {sl_ratio:.1%}")
        if tp_ratio < 0.3:
            issues.append(f"Мало срабатываний Take Profit: {tp_ratio:.1%}")
        
        # Рекомендации
        recommendations = []
        if sl_ratio > 0.6:
            recommendations.append("Увеличить размер Stop Loss")
            recommendations.append("Улучшить качество сигналов входа")
        if tp_ratio < 0.3:
            recommendations.append("Уменьшить размер Take Profit")
            recommendations.append("Добавить частичное закрытие позиций")
        
        return ComponentAnalysisResult(
            component_name="Exit Conditions",
            analysis_data=analysis,
            performance_score=max(0, performance_score),
            issues_found=issues,
            recommendations=recommendations,
            charts_generated=charts
        )
    
    async def analyze_risk_management(self, data: Dict[str, Any]) -> ComponentAnalysisResult:
        """Анализ управления рисками"""
        
        trades_df = pd.DataFrame(data['trades'])
        
        if trades_df.empty:
            return ComponentAnalysisResult(
                component_name="Risk Management",
                analysis_data=None,
                performance_score=0,
                issues_found=["Нет сделок для анализа"],
                recommendations=["Проверить систему управления рисками"],
                charts_generated=[]
            )
        
        # Анализ рисков
        total_trades = len(trades_df)
        stop_loss_count = len(trades_df[trades_df['exit_reason'] == 'Stop Loss'])
        take_profit_count = len(trades_df[trades_df['exit_reason'] == 'Take Profit'])
        
        # Расчет риск/доходность
        wins = trades_df[trades_df['pnl'] > 0]['pnl']
        losses = trades_df[trades_df['pnl'] <= 0]['pnl']
        
        avg_win = wins.mean() if not wins.empty else 0
        avg_loss = abs(losses.mean()) if not losses.empty else 0
        risk_reward_ratio = avg_win / avg_loss if avg_loss > 0 else 0
        
        # Средний риск на сделку
        avg_risk = trades_df['size'].mean() * self.config.stop_loss_percent / 100
        
        # Эффективность размера позиций
        position_sizing_eff = trades_df['pnl'].sum() / trades_df['size'].sum() if trades_df['size'].sum() > 0 else 0
        
        analysis = RiskManagementAnalysis(
            total_trades=total_trades,
            stop_loss_triggered=stop_loss_count,
            take_profit_triggered=take_profit_count,
            max_drawdown_reached=0,  # Требует дополнительного расчета
            risk_reward_ratio=risk_reward_ratio,
            avg_risk_per_trade=avg_risk,
            position_sizing_effectiveness=position_sizing_eff
        )
        
        # Создаем визуализации
        charts = await self._create_risk_charts(trades_df, analysis)
        
        # Оценка производительности
        performance_score = min(100, risk_reward_ratio * 50) if risk_reward_ratio > 0 else 0
        
        # Выявление проблем
        issues = []
        if risk_reward_ratio < 1.5:
            issues.append(f"Низкое соотношение риск/доходность: {risk_reward_ratio:.2f}")
        if stop_loss_count / total_trades > 0.7:
            issues.append(f"Слишком частые срабатывания Stop Loss: {stop_loss_count/total_trades:.1%}")
        
        # Рекомендации
        recommendations = []
        if risk_reward_ratio < 1.5:
            recommendations.append("Увеличить размер Take Profit")
            recommendations.append("Уменьшить размер Stop Loss")
        if stop_loss_count / total_trades > 0.7:
            recommendations.append("Улучшить качество сигналов входа")
            recommendations.append("Добавить дополнительные фильтры")
        
        return ComponentAnalysisResult(
            component_name="Risk Management",
            analysis_data=analysis,
            performance_score=performance_score,
            issues_found=issues,
            recommendations=recommendations,
            charts_generated=charts
        )
    
    async def analyze_filters(self, data: Dict[str, Any]) -> ComponentAnalysisResult:
        """Анализ эффективности фильтров"""
        
        filter_checks_df = pd.DataFrame(data['filter_checks'])
        
        if filter_checks_df.empty:
            return ComponentAnalysisResult(
                component_name="Filters",
                analysis_data=None,
                performance_score=0,
                issues_found=["Нет данных о фильтрах"],
                recommendations=["Проверить работу фильтров"],
                charts_generated=[]
            )
        
        # Анализ по типам фильтров
        filter_analyses = {}
        
        for filter_name in filter_checks_df['filter_name'].unique():
            filter_data = filter_checks_df[filter_checks_df['filter_name'] == filter_name]
            
            total_checks = len(filter_data)
            passed_checks = len(filter_data[filter_data['passed'] == True])
            blocked_checks = total_checks - passed_checks
            pass_rate = passed_checks / total_checks if total_checks > 0 else 0
            
            analysis = FilterAnalysis(
                filter_name=filter_name,
                total_checks=total_checks,
                passed_checks=passed_checks,
                blocked_checks=blocked_checks,
                pass_rate=pass_rate,
                impact_on_performance=0.0,  # Требует дополнительного анализа
                false_positives=0,  # Требует дополнительного анализа
                true_positives=0    # Требует дополнительного анализа
            )
            
            filter_analyses[filter_name] = analysis
        
        # Создаем визуализации
        charts = await self._create_filter_charts(filter_checks_df, filter_analyses)
        
        # Оценка производительности
        avg_pass_rate = np.mean([f.pass_rate for f in filter_analyses.values()])
        performance_score = avg_pass_rate * 100
        
        # Выявление проблем
        issues = []
        for name, analysis in filter_analyses.items():
            if analysis.pass_rate < 0.1:
                issues.append(f"Фильтр {name} блокирует слишком много сигналов: {analysis.pass_rate:.1%}")
            elif analysis.pass_rate > 0.9:
                issues.append(f"Фильтр {name} пропускает слишком много сигналов: {analysis.pass_rate:.1%}")
        
        # Рекомендации
        recommendations = []
        for name, analysis in filter_analyses.items():
            if analysis.pass_rate < 0.1:
                recommendations.append(f"Ослабить фильтр {name}")
            elif analysis.pass_rate > 0.9:
                recommendations.append(f"Ужесточить фильтр {name}")
        
        return ComponentAnalysisResult(
            component_name="Filters",
            analysis_data=filter_analyses,
            performance_score=performance_score,
            issues_found=issues,
            recommendations=recommendations,
            charts_generated=charts
        )
    
    async def _create_signal_charts(self, signals_df: pd.DataFrame, 
                                   analyses: Dict[str, SignalAnalysis]) -> List[str]:
        """Создание графиков для анализа сигналов"""
        
        charts = []
        
        # 1. Распределение сигналов по моделям
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('📊 Анализ генерации сигналов AI моделей', fontsize=16, fontweight='bold')
        
        # График 1: Количество сигналов по моделям
        model_counts = signals_df['model_name'].value_counts()
        axes[0, 0].bar(model_counts.index, model_counts.values, color='skyblue')
        axes[0, 0].set_title('Количество сигналов по моделям')
        axes[0, 0].set_ylabel('Количество сигналов')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # График 2: Средняя уверенность по моделям
        avg_confidence = signals_df.groupby('model_name')['confidence'].mean()
        axes[0, 1].bar(avg_confidence.index, avg_confidence.values, color='lightgreen')
        axes[0, 1].set_title('Средняя уверенность по моделям')
        axes[0, 1].set_ylabel('Средняя уверенность')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # График 3: Распределение действий
        action_counts = signals_df['action'].value_counts()
        axes[1, 0].pie(action_counts.values, labels=action_counts.index, autopct='%1.1f%%')
        axes[1, 0].set_title('Распределение действий')
        
        # График 4: Склонность моделей
        bias_scores = [analyses[model].bias_score for model in analyses.keys()]
        model_names = list(analyses.keys())
        colors = ['red' if score < -0.1 else 'green' if score > 0.1 else 'gray' for score in bias_scores]
        axes[1, 1].bar(model_names, bias_scores, color=colors)
        axes[1, 1].set_title('Склонность моделей (Медвежья ← → Бычья)')
        axes[1, 1].set_ylabel('Склонность')
        axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        chart_path = f"{self.charts_dir}/signal_analysis.png"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        charts.append(chart_path)
        
        # 2. Временной анализ сигналов
        fig, ax = plt.subplots(figsize=(15, 8))
        
        # Группируем по часам
        signals_df['hour'] = pd.to_datetime(signals_df['timestamp']).dt.hour
        hourly_signals = signals_df.groupby(['hour', 'action']).size().unstack(fill_value=0)
        
        hourly_signals.plot(kind='bar', stacked=True, ax=ax, color=['red', 'green', 'gray'])
        ax.set_title('📈 Распределение сигналов по времени суток', fontsize=14, fontweight='bold')
        ax.set_xlabel('Час (UTC)')
        ax.set_ylabel('Количество сигналов')
        ax.legend(title='Действие')
        
        plt.tight_layout()
        chart_path = f"{self.charts_dir}/signal_timing.png"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        charts.append(chart_path)
        
        return charts
    
    async def _create_consensus_charts(self, consensus_df: pd.DataFrame, 
                                      analysis: ConsensusAnalysis) -> List[str]:
        """Создание графиков для анализа консенсуса"""
        
        charts = []
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('🤝 Анализ логики консенсуса', fontsize=16, fontweight='bold')
        
        # График 1: Успешность формирования консенсуса
        consensus_success = consensus_df['consensus_formed'].value_counts()
        axes[0, 0].pie(consensus_success.values, 
                      labels=['Не сформирован', 'Сформирован'], 
                      autopct='%1.1f%%',
                      colors=['lightcoral', 'lightgreen'])
        axes[0, 0].set_title('Успешность формирования консенсуса')
        
        # График 2: Распределение по силе консенсуса
        if analysis.consensus_by_strength:
            strength_data = list(analysis.consensus_by_strength.items())
            strength_data.sort()
            models_count, frequency = zip(*strength_data)
            axes[0, 1].bar(models_count, frequency, color='skyblue')
            axes[0, 1].set_title('Распределение по количеству участвующих моделей')
            axes[0, 1].set_xlabel('Количество моделей')
            axes[0, 1].set_ylabel('Частота')
        
        # График 3: Распределение действий консенсуса
        if analysis.action_distribution:
            actions = list(analysis.action_distribution.keys())
            counts = list(analysis.action_distribution.values())
            axes[1, 0].bar(actions, counts, color=['red', 'green', 'gray'])
            axes[1, 0].set_title('Распределение действий консенсуса')
            axes[1, 0].set_ylabel('Количество')
        
        # График 4: Временной анализ консенсуса
        consensus_df['hour'] = pd.to_datetime(consensus_df['timestamp']).dt.hour
        hourly_consensus = consensus_df.groupby('hour')['consensus_formed'].mean()
        axes[1, 1].plot(hourly_consensus.index, hourly_consensus.values, marker='o', color='blue')
        axes[1, 1].set_title('Успешность консенсуса по времени суток')
        axes[1, 1].set_xlabel('Час (UTC)')
        axes[1, 1].set_ylabel('Доля успешных консенсусов')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        chart_path = f"{self.charts_dir}/consensus_analysis.png"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        charts.append(chart_path)
        
        return charts
    
    async def _create_entry_charts(self, trades_df: pd.DataFrame, 
                                  analysis: EntryExitAnalysis) -> List[str]:
        """Создание графиков для анализа входов"""
        
        charts = []
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('📈 Анализ условий входа в сделку', fontsize=16, fontweight='bold')
        
        # График 1: Успешность входов
        success_data = ['Прибыльные', 'Убыточные']
        success_counts = [analysis.successful_entries, analysis.total_entries - analysis.successful_entries]
        axes[0, 0].pie(success_counts, labels=success_data, autopct='%1.1f%%', 
                      colors=['lightgreen', 'lightcoral'])
        axes[0, 0].set_title(f'Успешность входов ({analysis.entry_success_rate:.1%})')
        
        # График 2: Распределение времени удержания
        axes[0, 1].hist(trades_df['holding_time_hours'], bins=20, color='skyblue', alpha=0.7)
        axes[0, 1].axvline(analysis.avg_holding_time, color='red', linestyle='--', 
                          label=f'Среднее: {analysis.avg_holding_time:.1f}ч')
        axes[0, 1].set_title('Распределение времени удержания')
        axes[0, 1].set_xlabel('Время удержания (часы)')
        axes[0, 1].set_ylabel('Количество сделок')
        axes[0, 1].legend()
        
        # График 3: Производительность по времени входа
        trades_df['entry_hour'] = pd.to_datetime(trades_df['entry_time']).dt.hour
        hourly_performance = trades_df.groupby('entry_hour')['pnl'].mean()
        axes[1, 0].bar(hourly_performance.index, hourly_performance.values, 
                      color=['red' if x < 0 else 'green' for x in hourly_performance.values])
        axes[1, 0].set_title('Средняя прибыль по времени входа')
        axes[1, 0].set_xlabel('Час входа (UTC)')
        axes[1, 0].set_ylabel('Средняя прибыль (USDT)')
        axes[1, 0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # График 4: Размер позиций vs прибыль
        axes[1, 1].scatter(trades_df['size'], trades_df['pnl'], alpha=0.6, 
                          c=['red' if x < 0 else 'green' for x in trades_df['pnl']])
        axes[1, 1].set_title('Размер позиции vs Прибыль')
        axes[1, 1].set_xlabel('Размер позиции')
        axes[1, 1].set_ylabel('Прибыль (USDT)')
        axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        plt.tight_layout()
        chart_path = f"{self.charts_dir}/entry_analysis.png"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        charts.append(chart_path)
        
        return charts
    
    async def _create_exit_charts(self, trades_df: pd.DataFrame, 
                                 analysis: EntryExitAnalysis) -> List[str]:
        """Создание графиков для анализа выходов"""
        
        charts = []
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('📉 Анализ условий выхода из сделки', fontsize=16, fontweight='bold')
        
        # График 1: Распределение причин выхода
        if analysis.exit_reasons:
            reasons = list(analysis.exit_reasons.keys())
            counts = list(analysis.exit_reasons.values())
            axes[0, 0].pie(counts, labels=reasons, autopct='%1.1f%%')
            axes[0, 0].set_title('Распределение причин выхода')
        
        # График 2: Прибыль по причинам выхода
        if analysis.avg_profit_per_exit_reason:
            reasons = list(analysis.avg_profit_per_exit_reason.keys())
            profits = list(analysis.avg_profit_per_exit_reason.values())
            colors = ['red' if p < 0 else 'green' for p in profits]
            axes[0, 1].bar(reasons, profits, color=colors)
            axes[0, 1].set_title('Средняя прибыль по причинам выхода')
            axes[0, 1].set_ylabel('Средняя прибыль (USDT)')
            axes[0, 1].tick_params(axis='x', rotation=45)
            axes[0, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # График 3: Время выхода vs прибыль
        trades_df['exit_hour'] = pd.to_datetime(trades_df['exit_time']).dt.hour
        hourly_exit_performance = trades_df.groupby('exit_hour')['pnl'].mean()
        axes[1, 0].bar(hourly_exit_performance.index, hourly_exit_performance.values,
                      color=['red' if x < 0 else 'green' for x in hourly_exit_performance.values])
        axes[1, 0].set_title('Средняя прибыль по времени выхода')
        axes[1, 0].set_xlabel('Час выхода (UTC)')
        axes[1, 0].set_ylabel('Средняя прибыль (USDT)')
        axes[1, 0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # График 4: Кумулятивная прибыль
        trades_df_sorted = trades_df.sort_values('exit_time')
        cumulative_pnl = trades_df_sorted['pnl'].cumsum()
        axes[1, 1].plot(range(len(cumulative_pnl)), cumulative_pnl, color='blue', linewidth=2)
        axes[1, 1].set_title('Кумулятивная прибыль')
        axes[1, 1].set_xlabel('Номер сделки')
        axes[1, 1].set_ylabel('Кумулятивная прибыль (USDT)')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        plt.tight_layout()
        chart_path = f"{self.charts_dir}/exit_analysis.png"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        charts.append(chart_path)
        
        return charts
    
    async def _create_risk_charts(self, trades_df: pd.DataFrame, 
                                 analysis: RiskManagementAnalysis) -> List[str]:
        """Создание графиков для анализа рисков"""
        
        charts = []
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('⚖️ Анализ управления рисками', fontsize=16, fontweight='bold')
        
        # График 1: Соотношение TP/SL
        tp_sl_data = ['Take Profit', 'Stop Loss', 'Другие']
        tp_sl_counts = [
            analysis.take_profit_triggered,
            analysis.stop_loss_triggered,
            analysis.total_trades - analysis.take_profit_triggered - analysis.stop_loss_triggered
        ]
        axes[0, 0].pie(tp_sl_counts, labels=tp_sl_data, autopct='%1.1f%%',
                      colors=['lightgreen', 'lightcoral', 'lightgray'])
        axes[0, 0].set_title('Распределение срабатываний TP/SL')
        
        # График 2: Распределение P&L
        axes[0, 1].hist(trades_df['pnl'], bins=20, color='skyblue', alpha=0.7)
        axes[0, 1].axvline(trades_df['pnl'].mean(), color='red', linestyle='--',
                          label=f'Среднее: {trades_df["pnl"].mean():.2f}')
        axes[0, 1].set_title('Распределение прибыли/убытка')
        axes[0, 1].set_xlabel('P&L (USDT)')
        axes[0, 1].set_ylabel('Количество сделок')
        axes[0, 1].axvline(0, color='black', linestyle='-', alpha=0.3)
        axes[0, 1].legend()
        
        # График 3: Риск vs доходность
        wins = trades_df[trades_df['pnl'] > 0]['pnl']
        losses = trades_df[trades_df['pnl'] <= 0]['pnl'].abs()
        
        if not wins.empty and not losses.empty:
            axes[1, 0].scatter([1], [wins.mean()], s=100, color='green', label='Средняя прибыль')
            axes[1, 0].scatter([1], [losses.mean()], s=100, color='red', label='Средний убыток')
            axes[1, 0].plot([1, 1], [losses.mean(), wins.mean()], 'k--', alpha=0.5)
            axes[1, 0].set_title(f'Риск/Доходность (R/R: {analysis.risk_reward_ratio:.2f})')
            axes[1, 0].set_ylabel('Размер (USDT)')
            axes[1, 0].set_xlim(0.5, 1.5)
            axes[1, 0].set_xticks([1])
            axes[1, 0].set_xticklabels(['Сделки'])
            axes[1, 0].legend()
        
        # График 4: Размер позиций
        axes[1, 1].hist(trades_df['size'], bins=20, color='orange', alpha=0.7)
        axes[1, 1].axvline(trades_df['size'].mean(), color='red', linestyle='--',
                          label=f'Среднее: {trades_df["size"].mean():.4f}')
        axes[1, 1].set_title('Распределение размеров позиций')
        axes[1, 1].set_xlabel('Размер позиции')
        axes[1, 1].set_ylabel('Количество сделок')
        axes[1, 1].legend()
        
        plt.tight_layout()
        chart_path = f"{self.charts_dir}/risk_analysis.png"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        charts.append(chart_path)
        
        return charts
    
    async def _create_filter_charts(self, filter_checks_df: pd.DataFrame, 
                                   analyses: Dict[str, FilterAnalysis]) -> List[str]:
        """Создание графиков для анализа фильтров"""
        
        charts = []
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('🔧 Анализ эффективности фильтров', fontsize=16, fontweight='bold')
        
        # График 1: Процент прохождения фильтров
        filter_names = list(analyses.keys())
        pass_rates = [analyses[name].pass_rate * 100 for name in filter_names]
        
        axes[0, 0].bar(filter_names, pass_rates, color='lightblue')
        axes[0, 0].set_title('Процент прохождения фильтров')
        axes[0, 0].set_ylabel('Процент прохождения (%)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50%')
        axes[0, 0].legend()
        
        # График 2: Количество проверок по фильтрам
        total_checks = [analyses[name].total_checks for name in filter_names]
        axes[0, 1].bar(filter_names, total_checks, color='lightgreen')
        axes[0, 1].set_title('Количество проверок по фильтрам')
        axes[0, 1].set_ylabel('Количество проверок')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # График 3: Блокировки vs пропуски
        blocked_checks = [analyses[name].blocked_checks for name in filter_names]
        passed_checks = [analyses[name].passed_checks for name in filter_names]
        
        x = np.arange(len(filter_names))
        width = 0.35
        
        axes[1, 0].bar(x - width/2, blocked_checks, width, label='Заблокировано', color='red', alpha=0.7)
        axes[1, 0].bar(x + width/2, passed_checks, width, label='Пропущено', color='green', alpha=0.7)
        axes[1, 0].set_title('Блокировки vs Пропуски')
        axes[1, 0].set_ylabel('Количество')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(filter_names, rotation=45)
        axes[1, 0].legend()
        
        # График 4: Временной анализ фильтров
        if not filter_checks_df.empty:
            filter_checks_df['hour'] = pd.to_datetime(filter_checks_df['timestamp']).dt.hour
            hourly_pass_rate = filter_checks_df.groupby('hour')['passed'].mean()
            axes[1, 1].plot(hourly_pass_rate.index, hourly_pass_rate.values * 100, 
                           marker='o', color='blue')
            axes[1, 1].set_title('Процент прохождения фильтров по времени')
            axes[1, 1].set_xlabel('Час (UTC)')
            axes[1, 1].set_ylabel('Процент прохождения (%)')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        chart_path = f"{self.charts_dir}/filter_analysis.png"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        charts.append(chart_path)
        
        return charts
    
    def _calculate_signal_performance_score(self, analyses: Dict[str, SignalAnalysis]) -> float:
        """Расчет оценки производительности сигналов"""
        if not analyses:
            return 0
        
        # Средняя уверенность всех моделей
        avg_confidence = np.mean([a.avg_confidence for a in analyses.values()])
        
        # Баланс сигналов (не слишком много HOLD)
        total_signals = sum(a.total_signals for a in analyses.values())
        total_hold = sum(a.hold_signals for a in analyses.values())
        hold_ratio = total_hold / total_signals if total_signals > 0 else 1
        
        # Разнообразие сигналов
        diversity_score = 1 - abs(0.5 - hold_ratio)  # Оптимально около 50% HOLD
        
        # Итоговая оценка
        score = (avg_confidence * 50 + diversity_score * 50)
        return min(100, max(0, score))
    
    def _calculate_consensus_performance_score(self, analysis: ConsensusAnalysis) -> float:
        """Расчет оценки производительности консенсуса"""
        # Базовая оценка по проценту успешных консенсусов
        base_score = analysis.consensus_rate * 60
        
        # Бонус за качество консенсуса
        confidence_bonus = analysis.avg_consensus_confidence * 20 if analysis.avg_consensus_confidence > 0 else 0
        
        # Бонус за участие моделей
        participation_bonus = min(20, analysis.avg_models_participating * 5)
        
        score = base_score + confidence_bonus + participation_bonus
        return min(100, max(0, score))
    
    def _identify_signal_issues(self, analyses: Dict[str, SignalAnalysis]) -> List[str]:
        """Выявление проблем в генерации сигналов"""
        issues = []
        
        for model_name, analysis in analyses.items():
            # Низкая уверенность
            if analysis.avg_confidence < 0.4:
                issues.append(f"Модель {model_name}: низкая средняя уверенность ({analysis.avg_confidence:.2f})")
            
            # Слишком много HOLD сигналов
            hold_ratio = analysis.hold_signals / analysis.total_signals
            if hold_ratio > 0.8:
                issues.append(f"Модель {model_name}: слишком много HOLD сигналов ({hold_ratio:.1%})")
            
            # Сильная склонность
            if abs(analysis.bias_score) > 0.5:
                bias_type = "бычья" if analysis.bias_score > 0 else "медвежья"
                issues.append(f"Модель {model_name}: сильная {bias_type} склонность ({analysis.bias_score:.2f})")
        
        return issues
    
    def _identify_consensus_issues(self, analysis: ConsensusAnalysis) -> List[str]:
        """Выявление проблем в логике консенсуса"""
        issues = []
        
        # Низкий процент успешных консенсусов
        if analysis.consensus_rate < 0.3:
            issues.append(f"Низкий процент формирования консенсуса: {analysis.consensus_rate:.1%}")
        
        # Низкая уверенность консенсуса
        if analysis.avg_consensus_confidence < 0.5:
            issues.append(f"Низкая средняя уверенность консенсуса: {analysis.avg_consensus_confidence:.2f}")
        
        # Мало участвующих моделей
        if analysis.avg_models_participating < 2:
            issues.append(f"Мало участвующих моделей в консенсусе: {analysis.avg_models_participating:.1f}")
        
        return issues
    
    def _generate_signal_recommendations(self, analyses: Dict[str, SignalAnalysis], issues: List[str]) -> List[str]:
        """Генерация рекомендаций для улучшения сигналов"""
        recommendations = []
        
        for model_name, analysis in analyses.items():
            if analysis.avg_confidence < 0.4:
                recommendations.append(f"Модель {model_name}: увеличить порог минимальной уверенности")
            
            hold_ratio = analysis.hold_signals / analysis.total_signals
            if hold_ratio > 0.8:
                recommendations.append(f"Модель {model_name}: настроить параметры для более активной торговли")
            
            if abs(analysis.bias_score) > 0.5:
                recommendations.append(f"Модель {model_name}: сбалансировать склонность к покупке/продаже")
        
        return recommendations
    
    def _generate_consensus_recommendations(self, analysis: ConsensusAnalysis, issues: List[str]) -> List[str]:
        """Генерация рекомендаций для улучшения консенсуса"""
        recommendations = []
        
        if analysis.consensus_rate < 0.3:
            recommendations.append("Снизить требования к формированию консенсуса")
            recommendations.append("Добавить весовые коэффициенты для моделей")
        
        if analysis.avg_consensus_confidence < 0.5:
            recommendations.append("Увеличить минимальную уверенность для консенсуса")
            recommendations.append("Улучшить качество индивидуальных сигналов")
        
        if analysis.avg_models_participating < 2:
            recommendations.append("Проверить работу всех AI моделей")
            recommendations.append("Снизить требования к участию в консенсусе")
        
        return recommendations
    
    async def _generate_summary_report(self, results: Dict[str, ComponentAnalysisResult]):
        """Генерация сводного отчета анализа"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = f"trading_logic_analysis_report_{timestamp}.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("🔍 ОТЧЕТ АНАЛИЗА ТОРГОВОЙ ЛОГИКИ\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Дата анализа: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Общая оценка
            total_score = np.mean([r.performance_score for r in results.values()])
            f.write(f"📊 ОБЩАЯ ОЦЕНКА СИСТЕМЫ: {total_score:.1f}/100\n\n")
            
            # Анализ по компонентам
            for component_name, result in results.items():
                f.write(f"🔧 {result.component_name.upper()}\n")
                f.write("-" * 30 + "\n")
                f.write(f"Оценка производительности: {result.performance_score:.1f}/100\n")
                
                if result.issues_found:
                    f.write("\n❌ Выявленные проблемы:\n")
                    for issue in result.issues_found:
                        f.write(f"  • {issue}\n")
                
                if result.recommendations:
                    f.write("\n💡 Рекомендации:\n")
                    for rec in result.recommendations:
                        f.write(f"  • {rec}\n")
                
                if result.charts_generated:
                    f.write("\n📈 Созданные графики:\n")
                    for chart in result.charts_generated:
                        f.write(f"  • {chart}\n")
                
                f.write("\n" + "=" * 50 + "\n\n")
            
            # Общие рекомендации
            f.write("🎯 ОБЩИЕ РЕКОМЕНДАЦИИ ДЛЯ УЛУЧШЕНИЯ СИСТЕМЫ\n")
            f.write("=" * 50 + "\n")
            
            all_issues = []
            for result in results.values():
                all_issues.extend(result.issues_found)
            
            if len(all_issues) > 5:
                f.write("• Система требует комплексной оптимизации\n")
                f.write("• Рекомендуется поэтапная калибровка компонентов\n")
            elif len(all_issues) > 2:
                f.write("• Система требует точечных улучшений\n")
                f.write("• Сосредоточиться на компонентах с низкими оценками\n")
            else:
                f.write("• Система работает удовлетворительно\n")
                f.write("• Рекомендуется мониторинг и тонкая настройка\n")
        
        logger.info(f"📄 Сводный отчет сохранен: {report_path}")


class TradingCalibrator:
    """
    ⚙️ КАЛИБРАТОР ТОРГОВОЙ СИСТЕМЫ
    
    Система автоматической калибровки компонентов торговой логики
    на основе результатов анализа
    """
    
    def __init__(self, config: TestConfig, analysis_results: Dict[str, ComponentAnalysisResult]):
        self.config = config
        self.analysis_results = analysis_results
        self.calibration_history = []
        
        logger.info("⚙️ Инициализирован калибратор торговой системы")
    
    async def run_auto_calibration(self) -> TestConfig:
        """Автоматическая калибровка системы"""
        logger.info("🚀 Запуск автоматической калибровки...")
        
        new_config = self._create_config_copy()
        
        # Калибровка по компонентам
        new_config = await self.calibrate_confidence_thresholds(new_config)
        new_config = await self.calibrate_consensus_rules(new_config)
        new_config = await self.calibrate_filters(new_config)
        new_config = await self.optimize_entry_exit(new_config)
        
        logger.info("✅ Автоматическая калибровка завершена!")
        
        return new_config
    
    def _create_config_copy(self) -> TestConfig:
        """Создание копии конфигурации для калибровки"""
        return TestConfig(
            test_period_days=self.config.test_period_days,
            start_balance=self.config.start_balance,
            commission_rate=self.config.commission_rate,
            position_size_percent=self.config.position_size_percent,
            min_position_value_usdt=self.config.min_position_value_usdt,
            leverage_multiplier=self.config.leverage_multiplier,
            stop_loss_percent=self.config.stop_loss_percent,
            take_profit_percent=self.config.take_profit_percent,
            use_trailing_stop=self.config.use_trailing_stop,
            trailing_stop_activation_percent=self.config.trailing_stop_activation_percent,
            trailing_stop_distance_percent=self.config.trailing_stop_distance_percent,
            max_hold_hours=self.config.max_hold_hours,
            min_confidence=self.config.min_confidence,
            min_volatility=self.config.min_volatility,
            min_volume_ratio=self.config.min_volume_ratio,
            symbols=self.config.symbols.copy()
        )
    
    async def calibrate_confidence_thresholds(self, config: TestConfig) -> TestConfig:
        """Калибровка порогов уверенности"""
        
        signal_result = self.analysis_results.get('signal_generation')
        if signal_result and signal_result.performance_score < 60:
            # Увеличиваем минимальную уверенность
            config.min_confidence = min(0.8, config.min_confidence + 0.1)
            logger.info(f"⚙️ Увеличен порог уверенности до {config.min_confidence}")
        
        return config
    
    async def calibrate_consensus_rules(self, config: TestConfig) -> TestConfig:
        """Калибровка правил консенсуса"""
        
        consensus_result = self.analysis_results.get('consensus_logic')
        if consensus_result and consensus_result.performance_score < 50:
            # Снижаем требования к консенсусу
            config.min_confidence = max(0.3, config.min_confidence - 0.05)
            logger.info(f"⚙️ Снижен порог консенсуса до {config.min_confidence}")
        
        return config
    
    async def calibrate_filters(self, config: TestConfig) -> TestConfig:
        """Калибровка фильтров"""
        
        filter_result = self.analysis_results.get('filters')
        if filter_result and filter_result.performance_score < 40:
            # Ослабляем фильтры
            config.min_volatility = max(0, config.min_volatility * 0.8)
            config.min_volume_ratio = max(0, config.min_volume_ratio * 0.8)
            logger.info(f"⚙️ Ослаблены фильтры: волатильность={config.min_volatility}, объем={config.min_volume_ratio}")
        
        return config
    
    async def optimize_entry_exit(self, config: TestConfig) -> TestConfig:
        """Оптимизация точек входа/выхода"""
        
        risk_result = self.analysis_results.get('risk_management')
        if risk_result and hasattr(risk_result.analysis_data, 'risk_reward_ratio'):
            rr_ratio = risk_result.analysis_data.risk_reward_ratio
            
            if rr_ratio < 1.5:
                # Увеличиваем TP и уменьшаем SL
                config.take_profit_percent = min(10, config.take_profit_percent * 1.2)
                config.stop_loss_percent = max(1, config.stop_loss_percent * 0.8)
                logger.info(f"⚙️ Оптимизированы TP/SL: TP={config.take_profit_percent}%, SL={config.stop_loss_percent}%")
        
        return config


async def main():
    """Главная функция для запуска анализа торговой логики"""
    
    # Конфигурация для анализа
    config = TestConfig(
        test_period_days=7,
        start_balance=1000.0,
        commission_rate=0.001,
        position_size_percent=10.0,
        min_position_value_usdt=10.0,
        leverage_multiplier=1.0,
        stop_loss_percent=2.0,
        take_profit_percent=4.0,
        use_trailing_stop=True,
        trailing_stop_activation_percent=1.0,
        trailing_stop_distance_percent=0.5,
        max_hold_hours=24,
        min_confidence=0.6,
        min_volatility=0.01,
        min_volume_ratio=1.2,
        symbols=['BTCUSDT', 'ETHUSDT']
    )
    
    # Запуск анализа
    analyzer = TradingLogicAnalyzer(config)
    results = await analyzer.run_full_analysis()
    
    # Запуск калибровки
    calibrator = TradingCalibrator(config, results)
    optimized_config = await calibrator.run_auto_calibration()
    
    # Сохранение оптимизированной конфигурации
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_path = f"optimized_config_{timestamp}.json"
    
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(asdict(optimized_config), f, indent=2, ensure_ascii=False)
    
    logger.info(f"💾 Оптимизированная конфигурация сохранена: {config_path}")
    
    print("\n🎉 АНАЛИЗ ТОРГОВОЙ ЛОГИКИ ЗАВЕРШЕН!")
    print(f"📊 Общая оценка системы: {np.mean([r.performance_score for r in results.values()]):.1f}/100")
    print(f"📄 Отчеты и графики сохранены в директории: trading_logic_analysis/")
    print(f"⚙️ Оптимизированная конфигурация: {config_path}")


if __name__ == "__main__":
    asyncio.run(main())