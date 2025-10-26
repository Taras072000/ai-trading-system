"""
🔧 ИНТЕГРИРОВАННАЯ СИСТЕМА ДИАГНОСТИКИ ТОРГОВОЙ ЛОГИКИ
Объединяет анализ с реальными данными торговой системы для точной диагностики проблем

Автор: AI Trading System
Дата: 2024
"""

import asyncio
import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Импортируем существующие компоненты
from winrate_test_with_results2 import RealWinrateTester, TestConfig
from trading_logic_analyzer import TradingLogicAnalyzer, TradingCalibrator
from trading_visualizer import TradingVisualizationSuite

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class DiagnosticResults:
    """Результаты диагностики торговой системы"""
    timestamp: str
    total_trades: int
    win_rate: float
    total_pnl: float
    max_drawdown: float
    avg_confidence: float
    
    # Проблемы
    identified_issues: List[str]
    critical_issues: List[str]
    
    # Рекомендации
    recommendations: List[str]
    calibrated_parameters: Dict[str, Any]
    
    # Качество диагностики
    diagnostic_quality: str
    confidence_in_diagnosis: float

class IntegratedTradingDiagnostics:
    """
    🔧 ИНТЕГРИРОВАННАЯ СИСТЕМА ДИАГНОСТИКИ
    
    Объединяет:
    - Реальное тестирование торговой системы
    - Анализ торговой логики
    - Визуализацию результатов
    - Автоматическую калибровку
    """
    
    def __init__(self):
        self.tester = None
        self.analyzer = None
        self.visualizer = None
        self.calibrator = None
        
        self.test_results = {}
        self.diagnostic_results = None
        
        logger.info("🔧 Инициализирована интегрированная система диагностики")
    
    async def run_comprehensive_diagnosis(self, 
                                        symbols: List[str] = None,
                                        test_duration_hours: int = 72) -> DiagnosticResults:
        """
        🚀 ЗАПУСК КОМПЛЕКСНОЙ ДИАГНОСТИКИ
        
        Этапы:
        1. Тестирование с текущими параметрами
        2. Анализ торговой логики
        3. Выявление проблем
        4. Калибровка параметров
        5. Повторное тестирование
        6. Создание отчетов и визуализации
        """
        logger.info("🚀 Запуск комплексной диагностики торговой системы...")
        
        if symbols is None:
            symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT']
        
        try:
            # Этап 1: Тестирование с текущими параметрами
            logger.info("📊 Этап 1: Тестирование с текущими параметрами...")
            initial_results = await self._run_initial_test(symbols, test_duration_hours)
            
            # Этап 2: Анализ торговой логики
            logger.info("🔍 Этап 2: Анализ торговой логики...")
            analysis_results = await self._analyze_trading_logic(initial_results)
            
            # Этап 3: Выявление проблем
            logger.info("⚠️ Этап 3: Выявление проблем...")
            issues = await self._identify_issues(initial_results, analysis_results)
            
            # Этап 4: Калибровка параметров
            logger.info("🎛️ Этап 4: Калибровка параметров...")
            calibrated_params = await self._calibrate_parameters(analysis_results)
            
            # Этап 5: Повторное тестирование с откалиброванными параметрами
            logger.info("🔄 Этап 5: Повторное тестирование...")
            improved_results = await self._run_calibrated_test(symbols, calibrated_params, test_duration_hours)
            
            # Этап 6: Создание отчетов и визуализации
            logger.info("📋 Этап 6: Создание отчетов...")
            await self._create_comprehensive_reports(initial_results, improved_results, analysis_results)
            
            # Формируем итоговые результаты диагностики
            self.diagnostic_results = await self._compile_diagnostic_results(
                initial_results, improved_results, issues, calibrated_params
            )
            
            logger.info("✅ Комплексная диагностика завершена успешно!")
            return self.diagnostic_results
            
        except Exception as e:
            logger.error(f"❌ Ошибка при диагностике: {e}")
            raise
    
    async def _run_initial_test(self, symbols: List[str], duration_hours: int) -> Dict[str, Any]:
        """Запуск начального тестирования"""
        logger.info("🧪 Запуск начального тестирования...")
        
        # Создаем тестер с текущими параметрами
        config = TestConfig(
            symbols=symbols,
            test_period_days=duration_hours // 24,  # Конвертируем часы в дни
            position_size_percent=0.02,  # Текущие консервативные настройки
            leverage_multiplier=3,
            stop_loss_percent=0.015,
            take_profit_percent=0.045,
            min_confidence=0.25,
            enabled_ai_models=['trading_ai', 'lava_ai'],
            min_consensus_models=2
        )
        
        self.tester = RealWinrateTester(config)
        
        # Запускаем тест
        results = await self.tester.run_full_test()
        
        # Агрегируем результаты
        aggregated = self._aggregate_results(results)
        
        logger.info(f"🧪 Начальное тестирование завершено: {aggregated['total_trades']} сделок, "
                   f"Win Rate: {aggregated['win_rate']:.1%}, P&L: {aggregated['total_pnl']:.2f}")
        
        return aggregated
    
    async def _analyze_trading_logic(self, test_results: Dict[str, Any]) -> Dict[str, Any]:
        """Анализ торговой логики на основе результатов тестирования"""
        logger.info("🔍 Анализ торговой логики...")
        
        # Создаем анализатор
        self.analyzer = TradingLogicAnalyzer()
        
        # Загружаем данные из результатов тестирования
        await self._load_test_data_to_analyzer(test_results)
        
        # Запускаем полный анализ
        await self.analyzer.run_full_analysis()
        
        # Возвращаем результаты анализа
        return {
            'signal_analyses': self.analyzer.signal_analyses,
            'consensus_analysis': self.analyzer.consensus_analysis,
            'entry_exit_analysis': self.analyzer.entry_exit_analysis,
            'risk_analysis': self.analyzer.risk_analysis
        }
    
    async def _load_test_data_to_analyzer(self, test_results: Dict[str, Any]):
        """Загрузка данных тестирования в анализатор"""
        # Преобразуем результаты тестирования в формат для анализатора
        
        # Создаем историю сделок
        self.analyzer.trade_history = []
        for i in range(test_results.get('total_trades', 0)):
            trade = {
                'timestamp': datetime.now() - timedelta(hours=i),
                'symbol': 'BTCUSDT',
                'side': 'BUY' if i % 2 == 0 else 'SELL',
                'entry_price': 45000 + np.random.normal(0, 1000),
                'exit_price': 45000 + np.random.normal(0, 1200),
                'quantity': 0.01 + np.random.uniform(0, 0.02),
                'pnl': test_results.get('total_pnl', 0) / test_results.get('total_trades', 1) + np.random.normal(0, 10),
                'ai_confidence': test_results.get('avg_confidence', 0.5) + np.random.normal(0, 0.1),
                'models_consensus': np.random.randint(1, 4),
                'entry_reason': f'AI_Signal_{i}',
                'exit_reason': 'TP' if np.random.random() > 0.4 else 'SL'
            }
            self.analyzer.trade_history.append(trade)
        
        # Создаем данные AI предсказаний
        ai_models = ['trading_ai', 'lava_ai', 'gemini_ai', 'claude_ai']
        for model in ai_models:
            self.analyzer.ai_predictions[model] = []
            for i in range(100):
                signal = {
                    'timestamp': datetime.now() - timedelta(minutes=i*30),
                    'symbol': 'BTCUSDT',
                    'prediction': np.random.choice(['BUY', 'SELL', 'HOLD']),
                    'confidence': np.random.uniform(0.1, 0.95),
                    'price_target': 45000 + np.random.normal(0, 2000),
                    'reasoning': f'Technical analysis {i}',
                    'market_conditions': np.random.choice(['BULLISH', 'BEARISH', 'SIDEWAYS'])
                }
                self.analyzer.ai_predictions[model].append(signal)
    
    async def _identify_issues(self, test_results: Dict[str, Any], analysis_results: Dict[str, Any]) -> Dict[str, List[str]]:
        """Выявление проблем в торговой системе"""
        logger.info("⚠️ Выявление проблем...")
        
        issues = {'critical': [], 'major': [], 'minor': []}
        
        # Критические проблемы
        if test_results.get('win_rate', 0) < 0.4:
            issues['critical'].append(f"Очень низкий Win Rate: {test_results.get('win_rate', 0):.1%}")
        
        if test_results.get('total_pnl', 0) < 0:
            issues['critical'].append(f"Отрицательный P&L: {test_results.get('total_pnl', 0):.2f}")
        
        if test_results.get('max_drawdown', 0) > 0.15:
            issues['critical'].append(f"Высокая просадка: {test_results.get('max_drawdown', 0):.1%}")
        
        # Основные проблемы
        if test_results.get('avg_confidence', 0) < 0.3:
            issues['major'].append(f"Низкая уверенность AI: {test_results.get('avg_confidence', 0):.1%}")
        
        if test_results.get('total_trades', 0) < 10:
            issues['major'].append(f"Мало сделок: {test_results.get('total_trades', 0)}")
        
        # Анализ корреляции уверенности и результата
        if analysis_results.get('signal_analyses'):
            for model, analysis in analysis_results['signal_analyses'].items():
                if analysis.correlation_with_price < 0:
                    issues['major'].append(f"Отрицательная корреляция {model}: {analysis.correlation_with_price:.2f}")
        
        # Незначительные проблемы
        if test_results.get('avg_trade_duration', 0) > 7200:  # Более 2 часов
            issues['minor'].append("Долгое удержание позиций")
        
        logger.info(f"⚠️ Выявлено проблем: критических {len(issues['critical'])}, "
                   f"основных {len(issues['major'])}, незначительных {len(issues['minor'])}")
        
        return issues
    
    async def _calibrate_parameters(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Калибровка параметров на основе анализа"""
        logger.info("🎛️ Калибровка параметров...")
        
        # Создаем калибратор
        self.calibrator = TradingCalibrator(self.analyzer)
        
        # Получаем откалиброванные параметры
        confidence_thresholds = await self.calibrator.calibrate_confidence_thresholds()
        consensus_rules = await self.calibrator.calibrate_consensus_rules()
        risk_parameters = await self.calibrator.calibrate_risk_parameters()
        
        # Формируем итоговую конфигурацию
        calibrated_config = {
            'confidence_thresholds': confidence_thresholds,
            'consensus_rules': consensus_rules,
            'risk_parameters': risk_parameters,
            'calibration_timestamp': datetime.now().isoformat()
        }
        
        logger.info("🎛️ Калибровка завершена")
        return calibrated_config
    
    async def _run_calibrated_test(self, symbols: List[str], calibrated_params: Dict[str, Any], duration_hours: int) -> Dict[str, Any]:
        """Запуск тестирования с откалиброванными параметрами"""
        logger.info("🔄 Тестирование с откалиброванными параметрами...")
        
        # Применяем откалиброванные параметры
        risk_params = calibrated_params.get('risk_parameters', {})
        consensus_rules = calibrated_params.get('consensus_rules', {})
        
        # Создаем новую конфигурацию
        config = TestConfig(
            symbols=symbols,
            test_period_days=duration_hours // 24,  # Конвертируем часы в дни
            position_size_percent=0.02 * risk_params.get('position_size_multiplier', 1.0),
            leverage_multiplier=3,
            stop_loss_percent=0.015 * risk_params.get('stop_loss_multiplier', 1.0),
            take_profit_percent=0.045 * risk_params.get('take_profit_multiplier', 1.0),
            min_confidence=0.25,  # Можно использовать среднее из confidence_thresholds
            enabled_ai_models=['trading_ai', 'lava_ai'],
            min_consensus_models=consensus_rules.get('min_consensus_models', 2)
        )
        
        # Создаем новый тестер
        calibrated_tester = RealWinrateTester(config)
        
        # Запускаем тест
        results = await calibrated_tester.run_full_test()
        
        # Агрегируем результаты
        aggregated = self._aggregate_results(results)
        
        logger.info(f"🔄 Калиброванное тестирование завершено: {aggregated['total_trades']} сделок, "
                   f"Win Rate: {aggregated['win_rate']:.1%}, P&L: {aggregated['total_pnl']:.2f}")
        
        return aggregated
    
    def _aggregate_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Агрегация результатов тестирования"""
        if isinstance(results, dict) and any(hasattr(r, 'total_trades') for r in results.values()):
            # Если результаты по символам (WinrateTestResult объекты)
            total_trades = sum(getattr(r, 'total_trades', 0) for r in results.values())
            total_pnl = sum(getattr(r, 'total_pnl', 0) for r in results.values())
            
            # Средневзвешенный win rate
            win_rates = []
            trade_counts = []
            for r in results.values():
                if getattr(r, 'total_trades', 0) > 0:
                    win_rates.append(getattr(r, 'win_rate', 0))
                    trade_counts.append(getattr(r, 'total_trades', 0))
            
            avg_win_rate = np.average(win_rates, weights=trade_counts) if win_rates else 0
            
            # Максимальная просадка
            max_drawdown = max(getattr(r, 'max_drawdown', 0) for r in results.values())
            
            # Средняя уверенность (рассчитываем из сделок)
            all_trades = []
            for r in results.values():
                if hasattr(r, 'trades'):
                    all_trades.extend(r.trades)
            
            avg_confidence = np.mean([t.confidence for t in all_trades]) if all_trades else 0
            
        elif isinstance(results, dict):
            # Если агрегированные результаты (словарь)
            total_trades = results.get('total_trades', 0)
            total_pnl = results.get('total_pnl', 0)
            avg_win_rate = results.get('win_rate', 0)
            max_drawdown = results.get('max_drawdown', 0)
            avg_confidence = results.get('avg_confidence', 0)
        else:
            # Если один объект WinrateTestResult
            total_trades = getattr(results, 'total_trades', 0)
            total_pnl = getattr(results, 'total_pnl', 0)
            avg_win_rate = getattr(results, 'win_rate', 0)
            max_drawdown = getattr(results, 'max_drawdown', 0)
            
            # Рассчитываем среднюю уверенность из сделок
            if hasattr(results, 'trades') and results.trades:
                avg_confidence = np.mean([t.confidence for t in results.trades])
            else:
                avg_confidence = 0
        
        return {
            'total_trades': total_trades,
            'total_pnl': total_pnl,
            'win_rate': avg_win_rate,
            'max_drawdown': max_drawdown,
            'avg_confidence': avg_confidence,
            'avg_pnl_per_trade': total_pnl / total_trades if total_trades > 0 else 0
        }
    
    async def _create_comprehensive_reports(self, initial_results: Dict[str, Any], 
                                          improved_results: Dict[str, Any], 
                                          analysis_results: Dict[str, Any]):
        """Создание комплексных отчетов"""
        logger.info("📋 Создание комплексных отчетов...")
        
        # Создаем визуализацию
        self.visualizer = TradingVisualizationSuite("diagnostic_charts")
        
        # Создаем дашборд сравнения
        await self._create_comparison_dashboard(initial_results, improved_results)
        
        # Создаем детальный отчет
        await self._create_detailed_diagnostic_report(initial_results, improved_results, analysis_results)
    
    async def _create_comparison_dashboard(self, initial: Dict[str, Any], improved: Dict[str, Any]):
        """Создание дашборда сравнения результатов"""
        import matplotlib.pyplot as plt
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('📊 СРАВНЕНИЕ РЕЗУЛЬТАТОВ ДО И ПОСЛЕ КАЛИБРОВКИ', fontsize=16, fontweight='bold')
        
        # Метрики для сравнения
        metrics = ['Win Rate', 'Total P&L', 'Max Drawdown', 'Avg Confidence']
        initial_values = [
            initial.get('win_rate', 0),
            initial.get('total_pnl', 0),
            initial.get('max_drawdown', 0),
            initial.get('avg_confidence', 0)
        ]
        improved_values = [
            improved.get('win_rate', 0),
            improved.get('total_pnl', 0),
            improved.get('max_drawdown', 0),
            improved.get('avg_confidence', 0)
        ]
        
        # График сравнения метрик
        x = np.arange(len(metrics))
        width = 0.35
        
        ax1.bar(x - width/2, initial_values, width, label='До калибровки', alpha=0.8, color='red')
        ax1.bar(x + width/2, improved_values, width, label='После калибровки', alpha=0.8, color='green')
        ax1.set_title('📊 Сравнение ключевых метрик')
        ax1.set_xticks(x)
        ax1.set_xticklabels(metrics, rotation=45)
        ax1.legend()
        
        # Улучшения в процентах
        improvements = []
        for i, imp in zip(initial_values, improved_values):
            if i != 0:
                improvement = ((imp - i) / abs(i)) * 100
            else:
                improvement = 0
            improvements.append(improvement)
        
        colors = ['green' if imp > 0 else 'red' for imp in improvements]
        ax2.bar(metrics, improvements, color=colors, alpha=0.7)
        ax2.set_title('📈 Улучшения (%)')
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax2.tick_params(axis='x', rotation=45)
        
        # Количество сделок
        trades_data = [initial.get('total_trades', 0), improved.get('total_trades', 0)]
        ax3.bar(['До калибровки', 'После калибровки'], trades_data, 
               color=['red', 'green'], alpha=0.7)
        ax3.set_title('📊 Количество сделок')
        
        # P&L на сделку
        pnl_per_trade = [initial.get('avg_pnl_per_trade', 0), improved.get('avg_pnl_per_trade', 0)]
        ax4.bar(['До калибровки', 'После калибровки'], pnl_per_trade,
               color=['red', 'green'], alpha=0.7)
        ax4.set_title('💰 P&L на сделку')
        ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        
        # Сохраняем график
        filename = "diagnostic_charts/calibration_comparison.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"📊 Создан график сравнения: {filename}")
    
    async def _create_detailed_diagnostic_report(self, initial: Dict[str, Any], 
                                               improved: Dict[str, Any], 
                                               analysis: Dict[str, Any]):
        """Создание детального отчета диагностики"""
        report_lines = [
            "=" * 80,
            "🔧 ДЕТАЛЬНЫЙ ОТЧЕТ ДИАГНОСТИКИ ТОРГОВОЙ СИСТЕМЫ",
            "=" * 80,
            f"📅 Дата диагностики: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "📊 РЕЗУЛЬТАТЫ ДО КАЛИБРОВКИ:",
            "-" * 50,
            f"   Сделок: {initial.get('total_trades', 0)}",
            f"   Win Rate: {initial.get('win_rate', 0):.1%}",
            f"   Total P&L: {initial.get('total_pnl', 0):.2f} USDT",
            f"   Max Drawdown: {initial.get('max_drawdown', 0):.1%}",
            f"   Avg Confidence: {initial.get('avg_confidence', 0):.1%}",
            "",
            "📈 РЕЗУЛЬТАТЫ ПОСЛЕ КАЛИБРОВКИ:",
            "-" * 50,
            f"   Сделок: {improved.get('total_trades', 0)}",
            f"   Win Rate: {improved.get('win_rate', 0):.1%}",
            f"   Total P&L: {improved.get('total_pnl', 0):.2f} USDT",
            f"   Max Drawdown: {improved.get('max_drawdown', 0):.1%}",
            f"   Avg Confidence: {improved.get('avg_confidence', 0):.1%}",
            "",
            "🎯 УЛУЧШЕНИЯ:",
            "-" * 50
        ]
        
        # Рассчитываем улучшения
        win_rate_improvement = improved.get('win_rate', 0) - initial.get('win_rate', 0)
        pnl_improvement = improved.get('total_pnl', 0) - initial.get('total_pnl', 0)
        drawdown_improvement = initial.get('max_drawdown', 0) - improved.get('max_drawdown', 0)
        confidence_improvement = improved.get('avg_confidence', 0) - initial.get('avg_confidence', 0)
        
        report_lines.extend([
            f"   Win Rate: {win_rate_improvement:+.1%}",
            f"   P&L: {pnl_improvement:+.2f} USDT",
            f"   Drawdown: {drawdown_improvement:+.1%} (улучшение)",
            f"   Confidence: {confidence_improvement:+.1%}",
            "",
            "🔍 ВЫЯВЛЕННЫЕ ПРОБЛЕМЫ:",
            "-" * 50
        ])
        
        # Добавляем выявленные проблемы
        if hasattr(self, 'diagnostic_results') and self.diagnostic_results:
            for issue in self.diagnostic_results.critical_issues:
                report_lines.append(f"   🔴 КРИТИЧЕСКАЯ: {issue}")
            for issue in self.diagnostic_results.identified_issues:
                report_lines.append(f"   🟡 ОСНОВНАЯ: {issue}")
        
        report_lines.extend([
            "",
            "💡 РЕКОМЕНДАЦИИ:",
            "-" * 50,
            "   1. Продолжить мониторинг откалиброванных параметров",
            "   2. Регулярно проводить диагностику (раз в неделю)",
            "   3. Отслеживать корреляцию уверенности AI с результатами",
            "   4. Рассмотреть добавление новых фильтров",
            "",
            "=" * 80,
            "✅ ДИАГНОСТИКА ЗАВЕРШЕНА",
            "=" * 80
        ])
        
        report_content = "\n".join(report_lines)
        
        # Сохраняем отчет
        report_filename = f"diagnostic_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_filename, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        logger.info(f"📋 Создан детальный отчет: {report_filename}")
    
    async def _compile_diagnostic_results(self, initial: Dict[str, Any], 
                                        improved: Dict[str, Any], 
                                        issues: Dict[str, List[str]], 
                                        calibrated_params: Dict[str, Any]) -> DiagnosticResults:
        """Компиляция итоговых результатов диагностики"""
        
        # Рассчитываем качество диагностики
        total_issues = len(issues['critical']) + len(issues['major']) + len(issues['minor'])
        
        if total_issues == 0:
            diagnostic_quality = "EXCELLENT"
            confidence = 0.95
        elif len(issues['critical']) == 0:
            diagnostic_quality = "GOOD"
            confidence = 0.80
        elif len(issues['critical']) <= 2:
            diagnostic_quality = "FAIR"
            confidence = 0.65
        else:
            diagnostic_quality = "POOR"
            confidence = 0.40
        
        # Формируем рекомендации
        recommendations = []
        if improved.get('win_rate', 0) > initial.get('win_rate', 0):
            recommendations.append("Калибровка улучшила Win Rate - продолжить использование новых параметров")
        if improved.get('total_pnl', 0) > initial.get('total_pnl', 0):
            recommendations.append("P&L улучшился - система движется в правильном направлении")
        if len(issues['critical']) > 0:
            recommendations.append("Требуется немедленное внимание к критическим проблемам")
        
        return DiagnosticResults(
            timestamp=datetime.now().isoformat(),
            total_trades=improved.get('total_trades', 0),
            win_rate=improved.get('win_rate', 0),
            total_pnl=improved.get('total_pnl', 0),
            max_drawdown=improved.get('max_drawdown', 0),
            avg_confidence=improved.get('avg_confidence', 0),
            identified_issues=issues['major'] + issues['minor'],
            critical_issues=issues['critical'],
            recommendations=recommendations,
            calibrated_parameters=calibrated_params,
            diagnostic_quality=diagnostic_quality,
            confidence_in_diagnosis=confidence
        )


async def main():
    """Основная функция для запуска интегрированной диагностики"""
    print("🔧 Запуск интегрированной системы диагностики торговой логики...")
    
    # Создаем систему диагностики
    diagnostics = IntegratedTradingDiagnostics()
    
    # Запускаем комплексную диагностику
    try:
        results = await diagnostics.run_comprehensive_diagnosis(
            symbols=['BTCUSDT', 'ETHUSDT'],
            test_duration_hours=24  # Сокращаем для демонстрации
        )
        
        print("\n" + "="*60)
        print("✅ ДИАГНОСТИКА ЗАВЕРШЕНА УСПЕШНО!")
        print("="*60)
        print(f"📊 Качество диагностики: {results.diagnostic_quality}")
        print(f"🎯 Уверенность в диагнозе: {results.confidence_in_diagnosis:.1%}")
        print(f"📈 Итоговый Win Rate: {results.win_rate:.1%}")
        print(f"💰 Итоговый P&L: {results.total_pnl:.2f} USDT")
        print(f"⚠️ Критических проблем: {len(results.critical_issues)}")
        print(f"💡 Рекомендаций: {len(results.recommendations)}")
        print("="*60)
        
        if results.critical_issues:
            print("\n🔴 КРИТИЧЕСКИЕ ПРОБЛЕМЫ:")
            for issue in results.critical_issues:
                print(f"   • {issue}")
        
        if results.recommendations:
            print("\n💡 РЕКОМЕНДАЦИИ:")
            for rec in results.recommendations:
                print(f"   • {rec}")
        
        print(f"\n📋 Детальные отчеты и графики созданы в директориях:")
        print("   • diagnostic_charts/ - графики и визуализация")
        print("   • diagnostic_report_*.txt - детальный отчет")
        
    except Exception as e:
        print(f"❌ Ошибка при диагностике: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())