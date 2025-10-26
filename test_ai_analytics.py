#!/usr/bin/env python3
"""
Тестовый скрипт для проверки новой аналитики AI моделей
"""

import asyncio
from datetime import datetime, timedelta
from winrate_test_with_results2 import (
    TestConfig, RealWinrateTester, TradeResult, AIModelDecision, 
    ConsensusSignal, AIModelPerformance, WinrateTestResult
)

def create_test_data():
    """Создает тестовые данные для проверки аналитики"""
    
    # Создаем тестовые решения AI моделей
    model_decisions = [
        AIModelDecision(
            model_name="TradingAI",
            action="BUY",
            confidence=0.75,
            reasoning="Strong bullish signal",
            timestamp=datetime.now()
        ),
        AIModelDecision(
            model_name="LavaAI", 
            action="BUY",
            confidence=0.68,
            reasoning="Momentum indicator positive",
            timestamp=datetime.now()
        ),
        AIModelDecision(
            model_name="LGBMAI",
            action="SELL",
            confidence=0.55,
            reasoning="Weak signal",
            timestamp=datetime.now()
        ),
        AIModelDecision(
            model_name="MistralAI",
            action="BUY", 
            confidence=0.82,
            reasoning="Technical analysis positive",
            timestamp=datetime.now()
        )
    ]
    
    # Создаем консенсусный сигнал
    consensus_signal = ConsensusSignal(
        symbol="BTCUSDT",
        timestamp=datetime.now(),
        price=50000.0,
        final_action="BUY",
        consensus_strength=3,  # 3 из 4 моделей согласились
        participating_models=model_decisions,
        confidence_avg=0.71
    )
    
    # Создаем тестовые сделки
    trades = [
        TradeResult(
            symbol="BTCUSDT",
            entry_time=datetime.now() - timedelta(hours=2),
            entry_price=50000.0,
            exit_time=datetime.now(),
            exit_price=51000.0,
            direction="LONG",
            pnl=100.0,
            pnl_percent=2.0,
            confidence=0.71,
            ai_model="Consensus_3",
            consensus_strength=3,
            participating_models=model_decisions,
            consensus_signal=consensus_signal
        ),
        TradeResult(
            symbol="BTCUSDT",
            entry_time=datetime.now() - timedelta(hours=4),
            entry_price=49000.0,
            exit_time=datetime.now() - timedelta(hours=2),
            exit_price=48500.0,
            direction="LONG",
            pnl=-50.0,
            pnl_percent=-1.0,
            confidence=0.65,
            ai_model="Consensus_2",
            consensus_strength=2,
            participating_models=model_decisions[:2],
            consensus_signal=None
        )
    ]
    
    # Создаем производительность AI моделей
    ai_models_performance = {
        "TradingAI": AIModelPerformance(
            model_name="TradingAI",
            total_signals=10,
            signals_used_in_trades=5,
            winning_signals=3,
            losing_signals=2,
            signal_accuracy=60.0,
            avg_confidence=0.72,
            contribution_to_pnl=75.0,
            consensus_participation_rate=0.8
        ),
        "LavaAI": AIModelPerformance(
            model_name="LavaAI",
            total_signals=8,
            signals_used_in_trades=4,
            winning_signals=2,
            losing_signals=2,
            signal_accuracy=50.0,
            avg_confidence=0.65,
            contribution_to_pnl=25.0,
            consensus_participation_rate=0.7
        ),
        "LGBMAI": AIModelPerformance(
            model_name="LGBMAI",
            total_signals=12,
            signals_used_in_trades=6,
            winning_signals=4,
            losing_signals=2,
            signal_accuracy=66.7,
            avg_confidence=0.68,
            contribution_to_pnl=50.0,
            consensus_participation_rate=0.6
        ),
        "MistralAI": AIModelPerformance(
            model_name="MistralAI",
            total_signals=9,
            signals_used_in_trades=4,
            winning_signals=3,
            losing_signals=1,
            signal_accuracy=75.0,
            avg_confidence=0.78,
            contribution_to_pnl=80.0,
            consensus_participation_rate=0.9
        )
    }
    
    # Создаем статистику консенсуса
    consensus_stats = {
        'total_consensus_signals': 15,
        'avg_consensus_strength': 2.8,
        'trades_with_2_models': 5,
        'trades_with_3_models': 8,
        'trades_with_4_models': 2
    }
    
    # Создаем результат тестирования
    test_result = WinrateTestResult(
        symbol="BTCUSDT",
        total_trades=2,
        winning_trades=1,
        losing_trades=1,
        win_rate=50.0,
        total_pnl=50.0,
        total_pnl_percent=0.5,
        avg_trade_pnl=25.0,
        max_drawdown=-50.0,
        sharpe_ratio=0.5,
        trades=trades,
        ai_models_performance=ai_models_performance,
        consensus_stats=consensus_stats
    )
    
    return {"BTCUSDT": test_result}

def test_analytics():
    """Тестирует новую аналитику"""
    print("🧪 Тестирование новой аналитики AI моделей...")
    
    # Создаем конфигурацию
    config = TestConfig(
        test_period_days=7,
        symbols=["BTCUSDT"],
        min_consensus_models=2
    )
    
    # Создаем тестер
    tester = RealWinrateTester(config)
    
    # Создаем тестовые данные
    test_results = create_test_data()
    
    # Генерируем отчет
    report = tester.generate_report(test_results)
    
    print("\n" + "="*80)
    print("📊 ТЕСТОВЫЙ ОТЧЕТ С НОВОЙ АНАЛИТИКОЙ:")
    print("="*80)
    print(report)
    print("="*80)
    
    print("\n✅ Тест аналитики завершен успешно!")
    print("🔍 Проверьте, что отчет содержит:")
    print("   - Анализ производительности AI моделей")
    print("   - Рейтинг моделей по прибыли и точности")
    print("   - Анализ консенсуса")
    print("   - Рекомендации по настройке")

if __name__ == "__main__":
    test_analytics()