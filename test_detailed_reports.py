#!/usr/bin/env python3
"""
Тестовый скрипт для демонстрации новых функций детальной отчётности
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
from pathlib import Path

# Импорты из основного файла
from winrate_test_with_results2 import (
    TestConfig, TradeResult, WinrateTestResult, RealWinrateTester,
    AIModelDecision, ConsensusSignal, AIModelPerformance
)

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_test_trades() -> List[TradeResult]:
    """Создаёт тестовые сделки для демонстрации отчётности"""
    trades = []
    base_time = datetime.now() - timedelta(days=30)
    
    # Прибыльные сделки
    profitable_trades_data = [
        {"symbol": "BTCUSDT", "pnl": 150.0, "pnl_percent": 15.0, "confidence": 85.0, "ai_model": "TradingAI", "consensus": 3},
        {"symbol": "ETHUSDT", "pnl": 120.0, "pnl_percent": 12.0, "confidence": 78.0, "ai_model": "MistralAI", "consensus": 4},
        {"symbol": "ADAUSDT", "pnl": 80.0, "pnl_percent": 8.0, "confidence": 72.0, "ai_model": "LGBMAI", "consensus": 2},
        {"symbol": "BTCUSDT", "pnl": 95.0, "pnl_percent": 9.5, "confidence": 81.0, "ai_model": "LavaAI", "consensus": 3},
        {"symbol": "ETHUSDT", "pnl": 110.0, "pnl_percent": 11.0, "confidence": 76.0, "ai_model": "TradingAI", "consensus": 2},
        {"symbol": "DOGEUSDT", "pnl": 65.0, "pnl_percent": 6.5, "confidence": 69.0, "ai_model": "MistralAI", "consensus": 3},
        {"symbol": "XRPUSDT", "pnl": 75.0, "pnl_percent": 7.5, "confidence": 74.0, "ai_model": "LGBMAI", "consensus": 2},
        {"symbol": "BTCUSDT", "pnl": 200.0, "pnl_percent": 20.0, "confidence": 92.0, "ai_model": "MistralAI", "consensus": 4},
    ]
    
    # Убыточные сделки
    losing_trades_data = [
        {"symbol": "SOLUSDT", "pnl": -45.0, "pnl_percent": -4.5, "confidence": 65.0, "ai_model": "LavaAI", "consensus": 2},
        {"symbol": "ADAUSDT", "pnl": -30.0, "pnl_percent": -3.0, "confidence": 62.0, "ai_model": "TradingAI", "consensus": 2},
        {"symbol": "DOGEUSDT", "pnl": -25.0, "pnl_percent": -2.5, "confidence": 68.0, "ai_model": "LGBMAI", "consensus": 2},
        {"symbol": "XRPUSDT", "pnl": -35.0, "pnl_percent": -3.5, "confidence": 71.0, "ai_model": "LavaAI", "consensus": 3},
        {"symbol": "SOLUSDT", "pnl": -50.0, "pnl_percent": -5.0, "confidence": 66.0, "ai_model": "TradingAI", "consensus": 2},
    ]
    
    all_trades_data = profitable_trades_data + losing_trades_data
    
    for i, trade_data in enumerate(all_trades_data):
        entry_time = base_time + timedelta(hours=i * 6)
        exit_time = entry_time + timedelta(hours=np.random.randint(2, 12))
        
        # Создаём фиктивные AI решения
        participating_models = []
        for j in range(trade_data["consensus"]):
            model_names = ["TradingAI", "LavaAI", "LGBMAI", "MistralAI"]
            participating_models.append(AIModelDecision(
                model_name=model_names[j % len(model_names)],
                action="BUY" if trade_data["pnl"] > 0 else "SELL",
                confidence=trade_data["confidence"] + np.random.uniform(-5, 5),
                reasoning=f"Test reasoning for {trade_data['symbol']}",
                timestamp=entry_time
            ))
        
        entry_price = 50000.0 if "BTC" in trade_data["symbol"] else 2000.0
        exit_price = entry_price * (1 + trade_data["pnl_percent"] / 100)
        
        trade = TradeResult(
            symbol=trade_data["symbol"],
            entry_time=entry_time,
            entry_price=entry_price,
            exit_time=exit_time,
            exit_price=exit_price,
            direction="LONG" if trade_data["pnl"] > 0 else "SHORT",
            pnl=trade_data["pnl"],
            pnl_percent=trade_data["pnl_percent"],
            confidence=trade_data["confidence"],
            ai_model=trade_data["ai_model"],
            consensus_strength=trade_data["consensus"],
            participating_models=participating_models,
            position_size=1000.0,
            commission=1.0
        )
        trades.append(trade)
    
    return trades

def create_test_results() -> Dict[str, WinrateTestResult]:
    """Создаёт тестовые результаты для демонстрации"""
    trades = create_test_trades()
    results = {}
    
    # Группируем сделки по символам
    symbols_trades = {}
    for trade in trades:
        if trade.symbol not in symbols_trades:
            symbols_trades[trade.symbol] = []
        symbols_trades[trade.symbol].append(trade)
    
    # Создаём результаты для каждого символа
    for symbol, symbol_trades in symbols_trades.items():
        winning_trades = len([t for t in symbol_trades if t.pnl > 0])
        total_trades = len(symbol_trades)
        total_pnl = sum(t.pnl for t in symbol_trades)
        
        results[symbol] = WinrateTestResult(
            symbol=symbol,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=total_trades - winning_trades,
            win_rate=(winning_trades / total_trades * 100) if total_trades > 0 else 0,
            total_pnl=total_pnl,
            total_pnl_percent=total_pnl,  # Упрощённо
            avg_trade_pnl=total_pnl / total_trades if total_trades > 0 else 0,
            max_drawdown=0.0,  # Упрощённо
            sharpe_ratio=1.5,  # Фиктивное значение
            trades=symbol_trades
        )
    
    return results

async def main():
    """Основная функция для тестирования новых отчётов"""
    print("🧪 " + "=" * 70 + " 🧪")
    print("🚀 ТЕСТИРОВАНИЕ НОВЫХ ФУНКЦИЙ ДЕТАЛЬНОЙ ОТЧЁТНОСТИ")
    print("🧪 " + "=" * 70 + " 🧪")
    
    # Создаём конфигурацию
    config = TestConfig()
    
    # Создаём тестер
    tester = RealWinrateTester(config)
    
    # Создаём тестовые данные
    results = create_test_results()
    
    print(f"📊 Создано {sum(len(r.trades) for r in results.values())} тестовых сделок")
    print(f"📈 Символы: {', '.join(results.keys())}")
    print()
    
    # Генерируем отчёт с новыми функциями
    report = tester.generate_report(results)
    
    # Выводим отчёт
    print(report)
    
    print("\n" + "🎉 " * 20)
    print("✅ Тестирование завершено успешно!")
    print("📁 Проверьте папку reports/csv_reports/ для CSV файлов")

if __name__ == "__main__":
    asyncio.run(main())