#!/usr/bin/env python3
"""
🚀 ТЕСТ КРИТИЧЕСКИХ ИСПРАВЛЕНИЙ
==============================

Быстрый тест для проверки эффективности критических исправлений:
- Уменьшенный размер позиций (2% вместо 10%)
- Умеренное плечо (3x вместо 10x)
- Расширенные стоп-лоссы (1.5% вместо 0.8%)
- Повышенная минимальная уверенность (30% вместо 20%)
- Новые параметры контроля риска

Использование:
    python test_critical_fixes.py
"""

import asyncio
import sys
import os
from datetime import datetime, timedelta
import logging

# Добавляем текущую директорию в путь для импорта
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from winrate_test_with_results2 import RealWinrateTester, TestConfig

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_critical_fixes():
    """
    🧪 ТЕСТ КРИТИЧЕСКИХ ИСПРАВЛЕНИЙ
    
    Запускает короткий тест с новыми консервативными параметрами
    """
    print("🔧 ТЕСТ КРИТИЧЕСКИХ ИСПРАВЛЕНИЙ ТОРГОВОЙ СИСТЕМЫ")
    print("=" * 60)
    print(f"🕐 Время запуска: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Конфигурация с критическими исправлениями
    config = TestConfig(
        test_period_days=3,  # Короткий тест на 3 дня
        start_balance=100.0,
        symbols=['BTCUSDT', 'ETHUSDT'],  # Только 2 топ пары для быстрого теста
        
        # КРИТИЧЕСКИЕ ИСПРАВЛЕНИЯ:
        position_size_percent=0.02,  # 2% вместо 10%
        leverage_multiplier=3.0,     # 3x вместо 10x
        stop_loss_percent=0.015,     # 1.5% вместо 0.8%
        take_profit_percent=0.045,   # 4.5% (соотношение 1:3)
        min_confidence=0.30,         # 30% вместо 20%
        
        # Новые параметры контроля риска:
        max_portfolio_drawdown=10.0,
        max_consecutive_losses=3,
        confidence_correlation_check=True,
        min_correlation_threshold=0.1,
        
        # Отладка включена
        debug_mode=True,
        
        # Консенсус
        min_consensus_models=1,  # Временно 1 для тестирования
        enabled_ai_models=['trading_ai']  # Только одна модель для быстрого теста
    )
    
    print("📊 НОВЫЕ ПАРАМЕТРЫ:")
    print(f"   💰 Размер позиции: {config.position_size_percent*100}% (было 10%)")
    print(f"   📈 Плечо: {config.leverage_multiplier}x (было 10x)")
    print(f"   🛑 Стоп-лосс: {config.stop_loss_percent*100}% (было 0.8%)")
    print(f"   🎯 Тейк-профит: {config.take_profit_percent*100}% (было 2.5%)")
    print(f"   🎲 Мин. уверенность: {config.min_confidence*100}% (было 20%)")
    print(f"   📉 Макс. просадка: {config.max_portfolio_drawdown}%")
    print()
    
    try:
        # Запуск тестирования
        logger.info("🚀 Запуск тестирования с критическими исправлениями...")
        
        tester = RealWinrateTester(config)
        results_dict = await tester.run_full_test()
        
        # Объединяем результаты всех символов
        all_trades = []
        total_pnl = 0.0
        for symbol, result in results_dict.items():
            all_trades.extend(result.trades)
            total_pnl += result.total_pnl
        
        # Создаем объединенный результат
        if all_trades:
            from types import SimpleNamespace
            results = SimpleNamespace()
            results.trades = all_trades
            results.total_pnl = total_pnl
        else:
            results = None
        
        if results:
            print("✅ РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ:")
            print("=" * 40)
            
            # Основные метрики
            total_trades = len(results.trades)
            profitable_trades = len([t for t in results.trades if t.pnl > 0])
            win_rate = (profitable_trades / total_trades * 100) if total_trades > 0 else 0
            total_pnl = sum(t.pnl for t in results.trades)
            
            print(f"📊 Общее количество сделок: {total_trades}")
            print(f"💰 Общий P&L: {total_pnl:.2f} USDT")
            print(f"🎯 Win Rate: {win_rate:.1f}%")
            print(f"📈 Средний P&L на сделку: {total_pnl/total_trades:.2f} USDT" if total_trades > 0 else "N/A")
            
            # Сравнение с предыдущими результатами
            print("\n🔄 СРАВНЕНИЕ С ПРЕДЫДУЩИМИ РЕЗУЛЬТАТАМИ:")
            print("=" * 50)
            print(f"   Win Rate: {win_rate:.1f}% (было 41.2%) - {'🟢 УЛУЧШЕНИЕ' if win_rate > 41.2 else '🔴 УХУДШЕНИЕ'}")
            
            # Анализ риска
            if total_trades > 0:
                max_loss = min(t.pnl for t in results.trades)
                max_profit = max(t.pnl for t in results.trades)
                print(f"   Макс. убыток: {max_loss:.2f} USDT")
                print(f"   Макс. прибыль: {max_profit:.2f} USDT")
                
                # Проверка просадки
                running_pnl = 0
                max_drawdown = 0
                peak = 0
                for trade in results.trades:
                    running_pnl += trade.pnl
                    if running_pnl > peak:
                        peak = running_pnl
                    drawdown = (peak - running_pnl) / config.start_balance * 100
                    if drawdown > max_drawdown:
                        max_drawdown = drawdown
                
                print(f"   Макс. просадка: {max_drawdown:.1f}% (было 34.1%) - {'🟢 УЛУЧШЕНИЕ' if max_drawdown < 34.1 else '🔴 УХУДШЕНИЕ'}")
            
            # Анализ уверенности
            if results.trades:
                avg_confidence = sum(t.confidence for t in results.trades) / len(results.trades)
                print(f"   Средняя уверенность: {avg_confidence*100:.1f}% (было 0.125%) - {'🟢 УЛУЧШЕНИЕ' if avg_confidence > 0.00125 else '🔴 УХУДШЕНИЕ'}")
            
            print("\n🎯 ВЫВОДЫ:")
            if win_rate > 50 and total_pnl > 0:
                print("✅ КРИТИЧЕСКИЕ ИСПРАВЛЕНИЯ РАБОТАЮТ!")
                print("   Система показывает улучшение по ключевым метрикам")
            elif win_rate > 41.2:
                print("🟡 ЧАСТИЧНОЕ УЛУЧШЕНИЕ")
                print("   Win rate улучшился, но требуется дополнительная настройка")
            else:
                print("🔴 ТРЕБУЕТСЯ ДОПОЛНИТЕЛЬНАЯ РАБОТА")
                print("   Необходимо проверить другие компоненты системы")
        
        else:
            print("❌ Тестирование не дало результатов")
            print("   Возможно, слишком строгие фильтры или проблемы с данными")
    
    except Exception as e:
        logger.error(f"❌ Ошибка при тестировании: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

async def main():
    """Основная функция"""
    exit_code = await test_critical_fixes()
    return exit_code

if __name__ == "__main__":
    # Запускаем тест
    exit_code = asyncio.run(main())
    sys.exit(exit_code)