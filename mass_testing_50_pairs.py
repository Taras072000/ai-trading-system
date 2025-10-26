#!/usr/bin/env python3
"""
Массовое тестирование 50 торговых пар на 30-дневном периоде
Цель: найти топ-5 самых прибыльных пар и отсеять убыточные
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging
from dataclasses import dataclass
import json
import os
import csv
from pathlib import Path

# Импорты из основного тестера
from winrate_test_with_results2 import (
    RealWinrateTester, TestConfig, WinrateTestResult, TradeResult
)

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class PairTestResult:
    """Результат тестирования одной торговой пары"""
    symbol: str
    roi_percent: float
    win_rate: float
    total_pnl: float
    total_trades: int
    avg_trade_pnl: float
    max_drawdown: float
    sharpe_ratio: float
    test_success: bool
    error_message: str = ""

class MassTester:
    """Класс для массового тестирования торговых пар"""
    
    def __init__(self):
        # Топ-50 USDT пар на Binance по торговому объему
        self.trading_pairs = [
            'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'XRPUSDT',
            'DOGEUSDT', 'TRXUSDT', 'SUIUSDT', 'ADAUSDT', 'BCHUSDT',
            'LINKUSDT', 'PEPEUSDT', 'TAOUSDT', 'AVAXUSDT', 'LTCUSDT',
            'AAVEUSDT', 'WLDUSDT', 'DOTUSDT', 'MATICUSDT', 'UNIUSDT',
            'ATOMUSDT', 'FILUSDT', 'ICPUSDT', 'NEARUSDT', 'APTUSDT',
            'OPUSDT', 'ARBUSDT', 'MKRUSDT', 'GRTUSDT', 'SANDUSDT',
            'MANAUSDT', 'CRVUSDT', 'LDOUSDT', 'COMPUSDT', 'SUSHIUSDT',
            'YFIUSDT', '1INCHUSDT', 'ENJUSDT', 'CHZUSDT', 'BATUSDT',
            'ZRXUSDT', 'RENUSDT', 'KNCUSDT', 'STORJUSDT', 'BNTUSDT',
            'LOOMUSDT', 'REPUSDT', 'ZILUSDT', 'ICXUSDT', 'ONTUSDT'
        ]
        
        # Оптимизированная конфигурация для массового тестирования
        self.config = TestConfig(
            test_period_days=30,  # 30-дневный тест
            start_balance=100.0,
            symbols=[],  # Будет заполняться для каждой пары
            commission_rate=0.001,
            position_size_percent=0.10,
            min_position_value_usdt=5.0,
            leverage_multiplier=10.0,
            
            # Оптимизированные настройки Stop Loss/Take Profit
            stop_loss_percent=0.008,  # 0.8%
            take_profit_percent=0.025,  # 2.5%
            
            # Оптимизированная сетка тейк-профитов
            use_take_profit_grid=True,
            take_profit_levels=[0.020, 0.025, 0.030],  # [2.0%, 2.5%, 3.0%]
            take_profit_portions=[0.40, 0.35, 0.25],   # [40%, 35%, 25%]
            
            # Повышенная минимальная уверенность
            min_confidence=0.20,  # 20%
            
            # Настройки для быстрого тестирования
            min_consensus_models=2,
            max_hold_hours=3,
            max_trades_per_day=10,
            
            # Отключение строгих фильтров для массового тестирования
            use_strict_filters=False,
            require_volume_confirmation=False,
            use_time_filter=False,
            
            debug_mode=False  # Отключить детальное логирование для скорости
        )
        
        self.results: List[PairTestResult] = []
        
    async def test_single_pair(self, symbol: str) -> PairTestResult:
        """Тестирование одной торговой пары"""
        logger.info(f"🔄 Тестирование пары {symbol}...")
        
        try:
            # Создаем конфигурацию для конкретной пары
            pair_config = TestConfig(
                **{k: v for k, v in self.config.__dict__.items() if k != 'symbols'}
            )
            pair_config.symbols = [symbol]
            
            # Создаем тестер для этой пары
            tester = RealWinrateTester(pair_config)
            await tester.initialize()
            
            # Запускаем тест
            test_results = await tester.run_full_test()
            
            if symbol in test_results:
                result = test_results[symbol]
                
                # Рассчитываем ROI
                roi_percent = (result.total_pnl / self.config.start_balance) * 100
                
                return PairTestResult(
                    symbol=symbol,
                    roi_percent=roi_percent,
                    win_rate=result.win_rate,
                    total_pnl=result.total_pnl,
                    total_trades=result.total_trades,
                    avg_trade_pnl=result.avg_trade_pnl,
                    max_drawdown=result.max_drawdown,
                    sharpe_ratio=result.sharpe_ratio,
                    test_success=True
                )
            else:
                return PairTestResult(
                    symbol=symbol,
                    roi_percent=0.0,
                    win_rate=0.0,
                    total_pnl=0.0,
                    total_trades=0,
                    avg_trade_pnl=0.0,
                    max_drawdown=0.0,
                    sharpe_ratio=0.0,
                    test_success=False,
                    error_message="Нет результатов тестирования"
                )
                
        except Exception as e:
            logger.error(f"❌ Ошибка при тестировании {symbol}: {str(e)}")
            return PairTestResult(
                symbol=symbol,
                roi_percent=0.0,
                win_rate=0.0,
                total_pnl=0.0,
                total_trades=0,
                avg_trade_pnl=0.0,
                max_drawdown=0.0,
                sharpe_ratio=0.0,
                test_success=False,
                error_message=str(e)
            )
    
    async def run_mass_testing(self) -> List[PairTestResult]:
        """Запуск массового тестирования всех пар"""
        logger.info(f"🚀 Начинаем массовое тестирование {len(self.trading_pairs)} торговых пар на 30-дневном периоде")
        logger.info(f"⚙️ Настройки: Stop Loss {self.config.stop_loss_percent*100:.1f}%, Take Profit {self.config.take_profit_percent*100:.1f}%, Min Confidence {self.config.min_confidence*100:.0f}%")
        
        self.results = []
        
        for i, symbol in enumerate(self.trading_pairs, 1):
            logger.info(f"📊 Прогресс: {i}/{len(self.trading_pairs)} - Тестируем {symbol}")
            
            result = await self.test_single_pair(symbol)
            self.results.append(result)
            
            # Показываем промежуточный результат
            if result.test_success:
                logger.info(f"✅ {symbol}: ROI {result.roi_percent:.2f}%, Win Rate {result.win_rate:.1f}%, Trades {result.total_trades}")
            else:
                logger.warning(f"❌ {symbol}: Тест не удался - {result.error_message}")
        
        return self.results
    
    def generate_comprehensive_report(self) -> str:
        """Генерация подробного отчета по всем парам"""
        
        # Сортируем результаты по ROI (убывание)
        successful_results = [r for r in self.results if r.test_success]
        failed_results = [r for r in self.results if not r.test_success]
        
        successful_results.sort(key=lambda x: x.roi_percent, reverse=True)
        
        report = []
        report.append("=" * 80)
        report.append("📊 ОТЧЕТ ПО МАССОВОМУ ТЕСТИРОВАНИЮ 50 ТОРГОВЫХ ПАР (30 ДНЕЙ)")
        report.append("=" * 80)
        report.append("")
        
        # Общая статистика
        total_pairs = len(self.trading_pairs)
        successful_pairs = len(successful_results)
        failed_pairs = len(failed_results)
        
        report.append(f"📈 ОБЩАЯ СТАТИСТИКА:")
        report.append(f"   • Всего пар протестировано: {total_pairs}")
        report.append(f"   • Успешно протестировано: {successful_pairs}")
        report.append(f"   • Не удалось протестировать: {failed_pairs}")
        report.append("")
        
        if successful_results:
            # Статистика по успешным тестам
            profitable_pairs = [r for r in successful_results if r.roi_percent > 0]
            losing_pairs = [r for r in successful_results if r.roi_percent <= 0]
            
            avg_roi = np.mean([r.roi_percent for r in successful_results])
            avg_win_rate = np.mean([r.win_rate for r in successful_results])
            total_trades = sum([r.total_trades for r in successful_results])
            
            report.append(f"💰 АНАЛИЗ ПРИБЫЛЬНОСТИ:")
            report.append(f"   • Прибыльных пар: {len(profitable_pairs)} ({len(profitable_pairs)/successful_pairs*100:.1f}%)")
            report.append(f"   • Убыточных пар: {len(losing_pairs)} ({len(losing_pairs)/successful_pairs*100:.1f}%)")
            report.append(f"   • Средний ROI: {avg_roi:.2f}%")
            report.append(f"   • Средний винрейт: {avg_win_rate:.1f}%")
            report.append(f"   • Общее количество сделок: {total_trades}")
            report.append("")
            
            # ТОП-5 самых прибыльных пар
            report.append("🏆 ТОП-5 САМЫХ ПРИБЫЛЬНЫХ ПАР:")
            report.append("-" * 60)
            for i, result in enumerate(successful_results[:5], 1):
                report.append(f"{i}. {result.symbol:12} | ROI: {result.roi_percent:+7.2f}% | Win Rate: {result.win_rate:5.1f}% | Trades: {result.total_trades:3d} | P&L: ${result.total_pnl:+7.2f}")
            report.append("")
            
            # ТОП-5 самых убыточных пар
            if len(successful_results) > 5:
                report.append("💸 ТОП-5 САМЫХ УБЫТОЧНЫХ ПАР:")
                report.append("-" * 60)
                worst_results = successful_results[-5:]
                worst_results.reverse()  # От худшего к лучшему из худших
                for i, result in enumerate(worst_results, 1):
                    report.append(f"{i}. {result.symbol:12} | ROI: {result.roi_percent:+7.2f}% | Win Rate: {result.win_rate:5.1f}% | Trades: {result.total_trades:3d} | P&L: ${result.total_pnl:+7.2f}")
                report.append("")
            
            # Детальная таблица всех результатов
            report.append("📋 ПОЛНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ:")
            report.append("-" * 80)
            report.append(f"{'Ранг':<4} {'Пара':<12} {'ROI %':<8} {'Win Rate %':<10} {'Trades':<7} {'P&L $':<10} {'Avg Trade':<10}")
            report.append("-" * 80)
            
            for i, result in enumerate(successful_results, 1):
                report.append(f"{i:<4} {result.symbol:<12} {result.roi_percent:+7.2f} {result.win_rate:9.1f} {result.total_trades:6d} {result.total_pnl:+9.2f} {result.avg_trade_pnl:+9.2f}")
        
        # Неудачные тесты
        if failed_results:
            report.append("")
            report.append("❌ ПАРЫ С НЕУДАЧНЫМИ ТЕСТАМИ:")
            report.append("-" * 50)
            for result in failed_results:
                report.append(f"   • {result.symbol}: {result.error_message}")
        
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def save_results_to_csv(self, filename: str = "mass_testing_results.csv"):
        """Сохранение результатов в CSV файл"""
        
        csv_path = Path("reports") / "csv_reports" / filename
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = [
                'rank', 'symbol', 'roi_percent', 'win_rate', 'total_pnl', 
                'total_trades', 'avg_trade_pnl', 'max_drawdown', 'sharpe_ratio',
                'test_success', 'error_message'
            ]
            
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            # Сортируем по ROI
            successful_results = [r for r in self.results if r.test_success]
            successful_results.sort(key=lambda x: x.roi_percent, reverse=True)
            
            # Записываем успешные результаты
            for i, result in enumerate(successful_results, 1):
                writer.writerow({
                    'rank': i,
                    'symbol': result.symbol,
                    'roi_percent': round(result.roi_percent, 2),
                    'win_rate': round(result.win_rate, 1),
                    'total_pnl': round(result.total_pnl, 2),
                    'total_trades': result.total_trades,
                    'avg_trade_pnl': round(result.avg_trade_pnl, 2),
                    'max_drawdown': round(result.max_drawdown, 2),
                    'sharpe_ratio': round(result.sharpe_ratio, 2),
                    'test_success': True,
                    'error_message': ''
                })
            
            # Записываем неудачные результаты
            failed_results = [r for r in self.results if not r.test_success]
            for result in failed_results:
                writer.writerow({
                    'rank': 'N/A',
                    'symbol': result.symbol,
                    'roi_percent': 0.0,
                    'win_rate': 0.0,
                    'total_pnl': 0.0,
                    'total_trades': 0,
                    'avg_trade_pnl': 0.0,
                    'max_drawdown': 0.0,
                    'sharpe_ratio': 0.0,
                    'test_success': False,
                    'error_message': result.error_message
                })
        
        logger.info(f"💾 Результаты сохранены в файл: {csv_path}")
        return str(csv_path)
    
    def get_top_profitable_pairs(self, top_n: int = 5) -> List[PairTestResult]:
        """Получение топ-N самых прибыльных пар"""
        successful_results = [r for r in self.results if r.test_success]
        successful_results.sort(key=lambda x: x.roi_percent, reverse=True)
        return successful_results[:top_n]

async def main():
    """Главная функция для запуска массового тестирования"""
    
    print("🚀 Запуск системы массового тестирования 50 торговых пар")
    print("⏱️  Период тестирования: 30 дней")
    print("🎯 Цель: найти топ-5 самых прибыльных пар")
    print("")
    
    # Создаем тестер
    mass_tester = MassTester()
    
    # Запускаем массовое тестирование
    start_time = datetime.now()
    results = await mass_tester.run_mass_testing()
    end_time = datetime.now()
    
    # Генерируем отчет
    report = mass_tester.generate_comprehensive_report()
    print(report)
    
    # Сохраняем результаты в CSV
    csv_file = mass_tester.save_results_to_csv()
    
    # Показываем топ-5 прибыльных пар
    top_pairs = mass_tester.get_top_profitable_pairs(5)
    
    print("\n" + "=" * 60)
    print("🏆 ИТОГОВЫЕ РЕКОМЕНДАЦИИ - ТОП-5 ПРИБЫЛЬНЫХ ПАР:")
    print("=" * 60)
    
    if top_pairs:
        for i, pair in enumerate(top_pairs, 1):
            print(f"{i}. {pair.symbol} - ROI: {pair.roi_percent:+.2f}%, Win Rate: {pair.win_rate:.1f}%")
        
        print(f"\n✅ Рекомендуется использовать эти {len(top_pairs)} пар для торговли")
        print(f"📊 Средний ROI топ-5: {np.mean([p.roi_percent for p in top_pairs]):.2f}%")
        print(f"📈 Средний винрейт топ-5: {np.mean([p.win_rate for p in top_pairs]):.1f}%")
    else:
        print("❌ Не найдено прибыльных пар. Требуется дополнительная оптимизация стратегии.")
    
    # Время выполнения
    execution_time = end_time - start_time
    print(f"\n⏱️  Время выполнения: {execution_time}")
    print(f"💾 Детальные результаты сохранены в: {csv_file}")

if __name__ == "__main__":
    asyncio.run(main())