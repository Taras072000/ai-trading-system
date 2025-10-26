#!/usr/bin/env python3
"""
🕯️ Тестирование нового свечного визуализатора
"""

import sys
import os
from datetime import datetime, timedelta
from detailed_trade_visualizer import DetailedTradeVisualizer, TradeVisualizationData

def create_test_trades():
    """Создает тестовые сделки для проверки свечных графиков"""
    
    base_time = datetime.now() - timedelta(hours=6)
    
    trades = []
    
    # Сделка 1: Прибыльная LONG BTCUSDT
    trade1 = TradeVisualizationData(
        symbol="BTCUSDT",
        entry_time=base_time,
        exit_time=base_time + timedelta(hours=2),
        entry_price=42000.0,
        exit_price=42500.0,
        direction="LONG",
        pnl=50.0,
        pnl_percent=1.19,
        confidence=85.5,
        ai_model="TradingAI",
        position_size=1000.0,
        commission=2.5,
        exit_reason="take_profit_1",
        partial_exits=[],
        market_data=None,
        tp_levels=[42300.0, 42600.0, 42900.0],
        sl_level=41700.0
    )
    trades.append(trade1)
    
    # Сделка 2: Убыточная SHORT BTCUSDT
    trade2 = TradeVisualizationData(
        symbol="BTCUSDT",
        entry_time=base_time + timedelta(hours=3),
        exit_time=base_time + timedelta(hours=4, minutes=30),
        entry_price=42400.0,
        exit_price=42600.0,
        direction="SHORT",
        pnl=-20.0,
        pnl_percent=-0.47,
        confidence=72.3,
        ai_model="TradingAI",
        position_size=1000.0,
        commission=2.5,
        exit_reason="stop_loss",
        partial_exits=[],
        market_data=None,
        tp_levels=[42200.0, 42000.0, 41800.0],
        sl_level=42700.0
    )
    trades.append(trade2)
    
    # Сделка 3: Прибыльная LONG ETHUSDT
    trade3 = TradeVisualizationData(
        symbol="ETHUSDT",
        entry_time=base_time + timedelta(hours=1),
        exit_time=base_time + timedelta(hours=3, minutes=45),
        entry_price=2500.0,
        exit_price=2580.0,
        direction="LONG",
        pnl=80.0,
        pnl_percent=3.2,
        confidence=91.2,
        ai_model="TradingAI",
        position_size=1000.0,
        commission=2.5,
        exit_reason="take_profit_2",
        partial_exits=[],
        market_data=None,
        tp_levels=[2530.0, 2560.0, 2590.0],
        sl_level=2450.0
    )
    trades.append(trade3)
    
    return trades

def test_candlestick_visualizer():
    """Тестирует новый свечной визуализатор"""
    
    print("🕯️ Тестирование свечного визуализатора...")
    
    # Создаем тестовые данные
    trades = create_test_trades()
    
    # Группируем сделки по символам
    btc_trades = [t for t in trades if t.symbol == "BTCUSDT"]
    eth_trades = [t for t in trades if t.symbol == "ETHUSDT"]
    
    # Создаем визуализатор
    visualizer = DetailedTradeVisualizer("reports/detailed_charts/candlestick_test")
    
    print(f"📊 Создание свечных графиков для {len(btc_trades)} сделок BTCUSDT...")
    visualizer.create_individual_trade_charts("BTCUSDT", btc_trades)
    
    print(f"📊 Создание свечных графиков для {len(eth_trades)} сделок ETHUSDT...")
    visualizer.create_individual_trade_charts("ETHUSDT", eth_trades)
    
    print(f"📈 Создание общего свечного графика для BTCUSDT...")
    visualizer.create_pair_summary_chart("BTCUSDT", btc_trades)
    
    print(f"📈 Создание общего свечного графика для ETHUSDT...")
    visualizer.create_pair_summary_chart("ETHUSDT", eth_trades)
    
    # Создаем портфельный обзор
    results = {
        "total_trades": len(trades),
        "profitable_trades": len([t for t in trades if t.pnl > 0]),
        "total_pnl": sum(t.pnl for t in trades),
        "win_rate": len([t for t in trades if t.pnl > 0]) / len(trades) * 100,
        "symbols": ["BTCUSDT", "ETHUSDT"],
        "trades_by_symbol": {
            "BTCUSDT": btc_trades,
            "ETHUSDT": eth_trades
        }
    }
    
    print(f"📊 Создание портфельного обзора...")
    visualizer.create_portfolio_overview(results)
    
    print(f"✅ Тестирование завершено!")
    print(f"📁 Результаты сохранены в: {visualizer.session_dir}")
    
    return visualizer.session_dir

if __name__ == "__main__":
    try:
        output_dir = test_candlestick_visualizer()
        print(f"\n🎯 Свечные графики успешно созданы!")
        print(f"📂 Проверьте результаты в папке: {output_dir}")
        
    except Exception as e:
        print(f"❌ Ошибка тестирования: {e}")
        import traceback
        traceback.print_exc()