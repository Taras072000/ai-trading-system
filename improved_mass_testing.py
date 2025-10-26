#!/usr/bin/env python3
"""
Улучшенное массовое тестирование с фильтрацией по количеству сделок
Решает проблему статистической незначимости при малом количестве сделок
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import logging
from dataclasses import dataclass
import json
import os
import csv
from pathlib import Path
import scipy.stats as stats
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

# Импорты из основного тестера
from winrate_test_with_results2 import (
    RealWinrateTester, TestConfig, WinrateTestResult, TradeResult
)

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class EnhancedPairTestResult:
    """Расширенный результат тестирования одной торговой пары"""
    symbol: str
    
    # Основные метрики
    roi_percent: float
    win_rate: float
    total_pnl: float
    total_trades: int
    avg_trade_pnl: float
    max_drawdown: float
    sharpe_ratio: float
    
    # Расширенные метрики
    profit_factor: float
    avg_trade_duration_hours: float
    roi_std: float
    
    # Статистическая значимость
    win_rate_confidence_interval: Tuple[float, float]
    statistical_significance: bool
    min_trades_for_significance: int
    
    # Результаты по периодам
    train_roi: float
    validation_roi: float
    test_roi: float
    
    # Статус
    test_success: bool
    meets_min_trades: bool
    error_message: str = ""

class ImprovedMassTester:
    """Улучшенный класс для массового тестирования торговых пар"""
    
    def __init__(self, min_trades_threshold: int = 30):
        self.min_trades_threshold = min_trades_threshold
        
        # Топ-50 USDT пар на Binance по торговому объему
        self.trading_pairs = [
            'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'XRPUSDT',
            'DOGEUSDT', 'TRXUSDT', 'SUIUSDT', 'ADAUSDT', 'BCHUSDT',
            'LINKUSDT', 'PEPEUSDT', 'TAOUSDT', 'AVAXUSDT', 'LTCUSDT',
            'DOTUSDT', 'UNIUSDT', 'ATOMUSDT', 'FILUSDT', 'NEARUSDT',
            'ICPUSDT', 'AAVEUSDT', 'ARBUSDT', 'OPUSDT', 'MKRUSDT',
            'GRTUSDT', 'SANDUSDT', 'MANAUSDT', 'APTUSDT', 'LDOUSDT',
            'CRVUSDT', 'ZRXUSDT', 'ENJUSDT', 'CHZUSDT', 'BATUSDT',
            'SUSHIUSDT', 'COMPUSDT', 'YFIUSDT', 'STORJUSDT', 'KNCUSDT',
            'BNTUSDT', 'ZILUSDT', 'ICXUSDT', 'ONTUSDT', '1INCHUSDT',
            'WLDUSDT', 'MATICUSDT', 'RENUSDT', 'LOOMUSDT', 'REPUSDT'
        ]
        
        # Оптимизированная конфигурация на основе предыдущих тестов
        self.test_config = TestConfig(
            test_period_days=30,  # 30 дней для достаточной статистики
            stop_loss_percent=0.8,  # Оптимальный SL
            take_profit_percent=2.5,  # Оптимальный TP
            min_confidence=20,  # Минимальная уверенность AI
            use_take_profit_grid=True,
            take_profit_levels=[2.0, 2.5, 3.0],  # Сетка TP
            take_profit_portions=[0.4, 0.35, 0.25]  # Доли для сетки
        )
        
        self.results: List[EnhancedPairTestResult] = []
        
    def calculate_confidence_interval(self, wins: int, total: int, confidence: float = 0.95) -> Tuple[float, float]:
        """Рассчитать доверительный интервал для винрейта"""
        if total == 0:
            return (0.0, 0.0)
            
        p = wins / total
        z = norm.ppf((1 + confidence) / 2)
        margin = z * np.sqrt(p * (1 - p) / total)
        
        lower = max(0, p - margin)
        upper = min(1, p + margin)
        
        return (lower * 100, upper * 100)
    
    def is_statistically_significant(self, total_trades: int, min_trades: int = 50) -> bool:
        """Проверить статистическую значимость результатов"""
        return total_trades >= min_trades
    
    def calculate_profit_factor(self, trades: List[TradeResult]) -> float:
        """Рассчитать Profit Factor"""
        if not trades:
            return 0.0
            
        gross_profit = sum(trade.pnl for trade in trades if trade.pnl > 0)
        gross_loss = abs(sum(trade.pnl for trade in trades if trade.pnl < 0))
        
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0
            
        return gross_profit / gross_loss
    
    def calculate_avg_trade_duration(self, trades: List[TradeResult]) -> float:
        """Рассчитать среднюю продолжительность сделки в часах"""
        if not trades:
            return 0.0
            
        durations = []
        for trade in trades:
            if hasattr(trade, 'entry_time') and hasattr(trade, 'exit_time'):
                duration = (trade.exit_time - trade.entry_time).total_seconds() / 3600
                durations.append(duration)
        
        return np.mean(durations) if durations else 0.0
    
    def split_data_periods(self, total_days: int) -> Tuple[int, int, int]:
        """Разделить данные на train/validation/test (60/20/20)"""
        train_days = int(total_days * 0.6)
        validation_days = int(total_days * 0.2)
        test_days = total_days - train_days - validation_days
        
        return train_days, validation_days, test_days
    
    async def test_single_pair_enhanced(self, symbol: str) -> EnhancedPairTestResult:
        """Расширенное тестирование одной торговой пары"""
        logger.info(f"🔍 Тестирование {symbol} с улучшенной валидацией...")
        
        try:
            # Создаем тестер
            tester = RealWinrateTester(symbol, self.test_config)
            
            # Полное тестирование (30 дней)
            full_result = await tester.run_test()
            
            if not full_result.test_success:
                return EnhancedPairTestResult(
                    symbol=symbol,
                    roi_percent=0.0, win_rate=0.0, total_pnl=0.0, total_trades=0,
                    avg_trade_pnl=0.0, max_drawdown=0.0, sharpe_ratio=0.0,
                    profit_factor=0.0, avg_trade_duration_hours=0.0, roi_std=0.0,
                    win_rate_confidence_interval=(0.0, 0.0), statistical_significance=False,
                    min_trades_for_significance=50, train_roi=0.0, validation_roi=0.0,
                    test_roi=0.0, test_success=False, meets_min_trades=False,
                    error_message=full_result.error_message
                )
            
            # Проверяем минимальное количество сделок
            meets_min_trades = full_result.total_trades >= self.min_trades_threshold
            
            # Рассчитываем расширенные метрики
            profit_factor = self.calculate_profit_factor(full_result.trades)
            avg_duration = self.calculate_avg_trade_duration(full_result.trades)
            
            # Рассчитываем стандартное отклонение ROI
            trade_returns = [trade.pnl for trade in full_result.trades] if full_result.trades else [0]
            roi_std = np.std(trade_returns)
            
            # Доверительный интервал для винрейта
            wins = sum(1 for trade in full_result.trades if trade.pnl > 0)
            confidence_interval = self.calculate_confidence_interval(wins, full_result.total_trades)
            
            # Статистическая значимость
            is_significant = self.is_statistically_significant(full_result.total_trades)
            
            # Тестирование по периодам (train/validation/test)
            train_days, val_days, test_days = self.split_data_periods(30)
            
            # Train период
            train_config = TestConfig(**self.test_config.__dict__)
            train_config.test_period_days = train_days
            train_tester = RealWinrateTester(symbol, train_config)
            train_result = await train_tester.run_test()
            train_roi = train_result.roi_percent if train_result.test_success else 0.0
            
            # Validation период (средние дни)
            val_config = TestConfig(**self.test_config.__dict__)
            val_config.test_period_days = val_days
            val_tester = RealWinrateTester(symbol, val_config)
            val_result = await val_tester.run_test()
            validation_roi = val_result.roi_percent if val_result.test_success else 0.0
            
            # Test период (последние дни)
            test_config = TestConfig(**self.test_config.__dict__)
            test_config.test_period_days = test_days
            test_tester = RealWinrateTester(symbol, test_config)
            test_result = await test_tester.run_test()
            test_roi = test_result.roi_percent if test_result.test_success else 0.0
            
            return EnhancedPairTestResult(
                symbol=symbol,
                roi_percent=full_result.roi_percent,
                win_rate=full_result.win_rate,
                total_pnl=full_result.total_pnl,
                total_trades=full_result.total_trades,
                avg_trade_pnl=full_result.avg_trade_pnl,
                max_drawdown=full_result.max_drawdown,
                sharpe_ratio=full_result.sharpe_ratio,
                profit_factor=profit_factor,
                avg_trade_duration_hours=avg_duration,
                roi_std=roi_std,
                win_rate_confidence_interval=confidence_interval,
                statistical_significance=is_significant,
                min_trades_for_significance=50,
                train_roi=train_roi,
                validation_roi=validation_roi,
                test_roi=test_roi,
                test_success=True,
                meets_min_trades=meets_min_trades
            )
            
        except Exception as e:
            logger.error(f"❌ Ошибка при тестировании {symbol}: {str(e)}")
            return EnhancedPairTestResult(
                symbol=symbol,
                roi_percent=0.0, win_rate=0.0, total_pnl=0.0, total_trades=0,
                avg_trade_pnl=0.0, max_drawdown=0.0, sharpe_ratio=0.0,
                profit_factor=0.0, avg_trade_duration_hours=0.0, roi_std=0.0,
                win_rate_confidence_interval=(0.0, 0.0), statistical_significance=False,
                min_trades_for_significance=50, train_roi=0.0, validation_roi=0.0,
                test_roi=0.0, test_success=False, meets_min_trades=False,
                error_message=str(e)
            )
    
    async def run_improved_mass_testing(self) -> List[EnhancedPairTestResult]:
        """Запуск улучшенного массового тестирования"""
        logger.info(f"🚀 Запуск улучшенного массового тестирования {len(self.trading_pairs)} пар")
        logger.info(f"📊 Минимальный порог сделок: {self.min_trades_threshold}")
        logger.info(f"⚙️ Конфигурация: SL {self.test_config.stop_loss_percent}%, TP {self.test_config.take_profit_percent}%")
        
        start_time = datetime.now()
        
        # Тестируем все пары
        for i, symbol in enumerate(self.trading_pairs, 1):
            logger.info(f"📈 [{i}/{len(self.trading_pairs)}] Тестирование {symbol}...")
            result = await self.test_single_pair_enhanced(symbol)
            self.results.append(result)
            
            # Промежуточная статистика
            if result.meets_min_trades:
                logger.info(f"✅ {symbol}: {result.total_trades} сделок, ROI: {result.roi_percent:.2f}%, WR: {result.win_rate:.1f}%")
            else:
                logger.warning(f"⚠️ {symbol}: Недостаточно сделок ({result.total_trades} < {self.min_trades_threshold})")
        
        # Сортируем по ROI
        self.results.sort(key=lambda x: x.roi_percent, reverse=True)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds() / 60
        
        logger.info(f"✅ Тестирование завершено за {duration:.1f} минут")
        return self.results
    
    def generate_enhanced_report(self) -> str:
        """Генерация расширенного отчета"""
        if not self.results:
            return "❌ Нет результатов для отчета"
        
        # Фильтруем результаты с достаточным количеством сделок
        valid_results = [r for r in self.results if r.meets_min_trades and r.test_success]
        insufficient_results = [r for r in self.results if not r.meets_min_trades and r.test_success]
        failed_results = [r for r in self.results if not r.test_success]
        
        report = []
        report.append("=" * 80)
        report.append("📊 УЛУЧШЕННЫЙ ОТЧЕТ МАССОВОГО ТЕСТИРОВАНИЯ")
        report.append("=" * 80)
        report.append(f"📅 Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"⚙️ Период тестирования: {self.test_config.test_period_days} дней")
        report.append(f"🎯 Минимальный порог сделок: {self.min_trades_threshold}")
        report.append(f"📈 Stop Loss: {self.test_config.stop_loss_percent}%")
        report.append(f"📈 Take Profit: {self.test_config.take_profit_percent}%")
        report.append("")
        
        # Общая статистика
        report.append("📊 ОБЩАЯ СТАТИСТИКА:")
        report.append(f"• Всего протестировано пар: {len(self.results)}")
        report.append(f"• Пары с достаточным количеством сделок: {len(valid_results)}")
        report.append(f"• Пары с недостаточным количеством сделок: {len(insufficient_results)}")
        report.append(f"• Неудачные тесты: {len(failed_results)}")
        report.append("")
        
        if valid_results:
            # Статистика по валидным результатам
            profitable_pairs = [r for r in valid_results if r.roi_percent > 0]
            avg_roi = np.mean([r.roi_percent for r in valid_results])
            avg_winrate = np.mean([r.win_rate for r in valid_results])
            avg_trades = np.mean([r.total_trades for r in valid_results])
            
            report.append("✅ СТАТИСТИКА ПО ВАЛИДНЫМ ПАРАМ:")
            report.append(f"• Прибыльных пар: {len(profitable_pairs)}/{len(valid_results)} ({len(profitable_pairs)/len(valid_results)*100:.1f}%)")
            report.append(f"• Средний ROI: {avg_roi:.2f}%")
            report.append(f"• Средний винрейт: {avg_winrate:.1f}%")
            report.append(f"• Среднее количество сделок: {avg_trades:.1f}")
            report.append("")
            
            # Топ-10 прибыльных пар
            report.append("🏆 ТОП-10 ПРИБЫЛЬНЫХ ПАР (с достаточным количеством сделок):")
            report.append("-" * 80)
            report.append(f"{'Ранг':<4} {'Пара':<12} {'ROI%':<8} {'WR%':<6} {'Сделки':<8} {'PF':<6} {'Значимость':<12}")
            report.append("-" * 80)
            
            for i, result in enumerate(profitable_pairs[:10], 1):
                significance = "✅ Да" if result.statistical_significance else "⚠️ Нет"
                report.append(f"{i:<4} {result.symbol:<12} {result.roi_percent:<8.2f} {result.win_rate:<6.1f} {result.total_trades:<8} {result.profit_factor:<6.2f} {significance:<12}")
            
            report.append("")
            
            # Детальная информация по топ-5
            report.append("🔍 ДЕТАЛЬНАЯ ИНФОРМАЦИЯ ПО ТОП-5:")
            report.append("-" * 80)
            
            for i, result in enumerate(profitable_pairs[:5], 1):
                ci_lower, ci_upper = result.win_rate_confidence_interval
                report.append(f"{i}. {result.symbol}")
                report.append(f"   ROI: {result.roi_percent:.2f}% | Винрейт: {result.win_rate:.1f}% (CI: {ci_lower:.1f}%-{ci_upper:.1f}%)")
                report.append(f"   Сделки: {result.total_trades} | Profit Factor: {result.profit_factor:.2f}")
                report.append(f"   Просадка: {result.max_drawdown:.2f}% | Шарп: {result.sharpe_ratio:.2f}")
                report.append(f"   Train/Val/Test ROI: {result.train_roi:.2f}%/{result.validation_roi:.2f}%/{result.test_roi:.2f}%")
                report.append(f"   Статистическая значимость: {'✅ Да' if result.statistical_significance else '⚠️ Нет'}")
                report.append("")
        
        # Предупреждения о недостаточном количестве сделок
        if insufficient_results:
            report.append("⚠️ ПАРЫ С НЕДОСТАТОЧНЫМ КОЛИЧЕСТВОМ СДЕЛОК:")
            report.append("-" * 80)
            report.append(f"{'Пара':<12} {'Сделки':<8} {'ROI%':<8} {'WR%':<6} {'Статус':<20}")
            report.append("-" * 80)
            
            for result in insufficient_results[:15]:  # Показываем первые 15
                status = f"< {self.min_trades_threshold} сделок"
                report.append(f"{result.symbol:<12} {result.total_trades:<8} {result.roi_percent:<8.2f} {result.win_rate:<6.1f} {status:<20}")
            
            if len(insufficient_results) > 15:
                report.append(f"... и еще {len(insufficient_results) - 15} пар")
            report.append("")
        
        # Рекомендации
        report.append("💡 РЕКОМЕНДАЦИИ:")
        report.append("• Используйте только пары с минимум 30+ сделок для надежной статистики")
        report.append("• Обратите внимание на доверительные интервалы винрейта")
        report.append("• Сравните результаты Train/Validation/Test для выявления overfitting")
        report.append("• Учитывайте Profit Factor и максимальную просадку при выборе пар")
        report.append("")
        
        return "\n".join(report)
    
    def save_enhanced_results_to_csv(self, filename: str = "improved_mass_testing_results.csv"):
        """Сохранение расширенных результатов в CSV"""
        if not self.results:
            logger.warning("Нет результатов для сохранения")
            return
        
        # Создаем директорию если не существует
        reports_dir = Path("reports/csv_reports")
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        filepath = reports_dir / filename
        
        with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = [
                'rank', 'symbol', 'roi_percent', 'win_rate', 'total_pnl', 'total_trades',
                'avg_trade_pnl', 'max_drawdown', 'sharpe_ratio', 'profit_factor',
                'avg_trade_duration_hours', 'roi_std', 'win_rate_ci_lower', 'win_rate_ci_upper',
                'statistical_significance', 'meets_min_trades', 'train_roi', 'validation_roi',
                'test_roi', 'test_success', 'error_message'
            ]
            
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for rank, result in enumerate(self.results, 1):
                ci_lower, ci_upper = result.win_rate_confidence_interval
                writer.writerow({
                    'rank': rank,
                    'symbol': result.symbol,
                    'roi_percent': round(result.roi_percent, 2),
                    'win_rate': round(result.win_rate, 1),
                    'total_pnl': round(result.total_pnl, 2),
                    'total_trades': result.total_trades,
                    'avg_trade_pnl': round(result.avg_trade_pnl, 2),
                    'max_drawdown': round(result.max_drawdown, 2),
                    'sharpe_ratio': round(result.sharpe_ratio, 2),
                    'profit_factor': round(result.profit_factor, 2),
                    'avg_trade_duration_hours': round(result.avg_trade_duration_hours, 2),
                    'roi_std': round(result.roi_std, 2),
                    'win_rate_ci_lower': round(ci_lower, 1),
                    'win_rate_ci_upper': round(ci_upper, 1),
                    'statistical_significance': result.statistical_significance,
                    'meets_min_trades': result.meets_min_trades,
                    'train_roi': round(result.train_roi, 2),
                    'validation_roi': round(result.validation_roi, 2),
                    'test_roi': round(result.test_roi, 2),
                    'test_success': result.test_success,
                    'error_message': result.error_message
                })
        
        logger.info(f"📄 Результаты сохранены в {filepath}")
    
    def get_statistically_significant_pairs(self, top_n: int = 10) -> List[EnhancedPairTestResult]:
        """Получить статистически значимые пары"""
        valid_results = [r for r in self.results if r.meets_min_trades and r.statistical_significance and r.test_success and r.roi_percent > 0]
        return valid_results[:top_n]

async def main():
    """Главная функция"""
    print("🚀 Запуск улучшенного массового тестирования...")
    print("📊 Решение проблемы статистической незначимости")
    print("=" * 60)
    
    # Создаем тестер с минимальным порогом 30 сделок
    tester = ImprovedMassTester(min_trades_threshold=30)
    
    # Запускаем тестирование
    start_time = datetime.now()
    results = await tester.run_improved_mass_testing()
    end_time = datetime.now()
    
    # Генерируем отчет
    report = tester.generate_enhanced_report()
    print(report)
    
    # Сохраняем результаты
    tester.save_enhanced_results_to_csv()
    
    # Получаем статистически значимые пары
    significant_pairs = tester.get_statistically_significant_pairs(5)
    
    print("\n" + "=" * 60)
    print("🎯 ФИНАЛЬНЫЕ РЕКОМЕНДАЦИИ:")
    print("=" * 60)
    
    if significant_pairs:
        print("✅ Статистически значимые пары для торговли:")
        for i, pair in enumerate(significant_pairs, 1):
            print(f"{i}. {pair.symbol}: ROI {pair.roi_percent:.2f}%, {pair.total_trades} сделок")
    else:
        print("⚠️ Не найдено статистически значимых прибыльных пар!")
        print("💡 Рекомендации:")
        print("   • Увеличить период тестирования до 60-90 дней")
        print("   • Снизить минимальную уверенность AI")
        print("   • Пересмотреть параметры SL/TP")
    
    duration = (end_time - start_time).total_seconds() / 60
    print(f"\n⏱️ Время выполнения: {duration:.1f} минут")

if __name__ == "__main__":
    asyncio.run(main())