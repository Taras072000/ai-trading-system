#!/usr/bin/env python3
"""
🎯 ТЕСТ СБАЛАНСИРОВАННЫХ НАСТРОЕК
================================

Тест с оптимизированными параметрами после анализа критических исправлений:
- min_confidence снижен до 25% (было 30%)
- Включены 2 лучшие AI модели: trading_ai + lava_ai
- Сохранены консервативные параметры риск-менеджмента
- Цель: увеличить количество сделок при сохранении контроля риска

Использование:
    python test_balanced_settings.py
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

async def test_balanced_settings():
    """
    🎯 ТЕСТ СБАЛАНСИРОВАННЫХ НАСТРОЕК
    
    Запускает тест с оптимизированными параметрами
    """
    print("🎯 ТЕСТ СБАЛАНСИРОВАННЫХ НАСТРОЕК ТОРГОВОЙ СИСТЕМЫ")
    print("=" * 65)
    print(f"🕐 Время запуска: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Конфигурация с балансированными настройками
    config = TestConfig(
        test_period_days=3,  # Короткий тест на 3 дня
        start_balance=100.0,
        symbols=['BTCUSDT', 'ETHUSDT'],  # Только 2 топ пары для быстрого теста
        
        # СБАЛАНСИРОВАННЫЕ ПАРАМЕТРЫ:
        position_size_percent=0.02,  # 2% (консервативно)
        leverage_multiplier=3.0,     # 3x (умеренно)
        stop_loss_percent=0.015,     # 1.5% (расширенный)
        take_profit_percent=0.045,   # 4.5% (соотношение 1:3)
        min_confidence=0.25,         # 25% (сбалансировано)
        
        # Контроль риска:
        max_portfolio_drawdown=10.0,
        max_consecutive_losses=3,
        confidence_correlation_check=True,
        min_correlation_threshold=0.1,
        
        # Консенсус (2 модели):
        min_consensus_models=2,
        enabled_ai_models=['trading_ai', 'lava_ai'],
        
        # Отладка
        debug_mode=True
    )
    
    print("📊 СБАЛАНСИРОВАННЫЕ ПАРАМЕТРЫ:")
    print(f"   💰 Размер позиции: {config.position_size_percent*100}%")
    print(f"   📈 Плечо: {config.leverage_multiplier}x")
    print(f"   🛑 Стоп-лосс: {config.stop_loss_percent*100}%")
    print(f"   🎯 Тейк-профит: {config.take_profit_percent*100}%")
    print(f"   🎲 Мин. уверенность: {config.min_confidence*100}% (снижено с 30%)")
    print(f"   🤖 AI модели: {', '.join(config.enabled_ai_models)}")
    print(f"   📉 Макс. просадка: {config.max_portfolio_drawdown}%")
    print()
    
    try:
        # Запуск тестирования
        logger.info("🚀 Запуск тестирования со сбалансированными настройками...")
        
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
            print("\n🔄 СРАВНЕНИЕ С ПРЕДЫДУЩИМИ ТЕСТАМИ:")
            print("=" * 50)
            print("КРИТИЧЕСКИЕ ИСПРАВЛЕНИЯ (30% confidence, 1 модель):")
            print(f"   Сделок: {total_trades} (было 5) - {'🟢 УЛУЧШЕНИЕ' if total_trades > 5 else '🔴 УХУДШЕНИЕ'}")
            print(f"   Win Rate: {win_rate:.1f}% (было 20%) - {'🟢 УЛУЧШЕНИЕ' if win_rate > 20 else '🔴 УХУДШЕНИЕ'}")
            
            print("\nОРИГИНАЛЬНЫЕ РЕЗУЛЬТАТЫ (агрессивные настройки):")
            print(f"   Win Rate: {win_rate:.1f}% (было 41.2%) - {'🟢 УЛУЧШЕНИЕ' if win_rate > 41.2 else '🔴 УХУДШЕНИЕ'}")
            
            # Анализ риска
            if total_trades > 0:
                max_loss = min(t.pnl for t in results.trades)
                max_profit = max(t.pnl for t in results.trades)
                print(f"\n📊 АНАЛИЗ РИСКА:")
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
                
                print(f"   Макс. просадка: {max_drawdown:.1f}%")
                
                # Анализ уверенности
                avg_confidence = sum(t.confidence for t in results.trades) / len(results.trades)
                print(f"   Средняя уверенность: {avg_confidence*100:.1f}%")
                
                # Анализ консенсуса
                consensus_strengths = [t.consensus_strength for t in results.trades if hasattr(t, 'consensus_strength')]
                if consensus_strengths:
                    avg_consensus = sum(consensus_strengths) / len(consensus_strengths)
                    print(f"   Средняя сила консенсуса: {avg_consensus:.1f} моделей")
            
            print("\n🎯 ВЫВОДЫ:")
            if win_rate > 50 and total_pnl > 0 and total_trades >= 10:
                print("✅ ОТЛИЧНЫЙ РЕЗУЛЬТАТ!")
                print("   Система сбалансирована и показывает хорошие результаты")
            elif win_rate > 40 and total_trades >= 8:
                print("🟡 ХОРОШИЙ ПРОГРЕСС")
                print("   Система улучшается, но требует дополнительной настройки")
            elif total_trades > 5:
                print("🟡 ЧАСТИЧНОЕ УЛУЧШЕНИЕ")
                print("   Количество сделок увеличилось, но качество требует работы")
            else:
                print("🔴 ТРЕБУЕТСЯ ДОПОЛНИТЕЛЬНАЯ РАБОТА")
                print("   Необходимо дальнейшее улучшение параметров")
        
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
    exit_code = await test_balanced_settings()
    return exit_code

if __name__ == "__main__":
    # Запускаем тест
    exit_code = asyncio.run(main())
    sys.exit(exit_code)