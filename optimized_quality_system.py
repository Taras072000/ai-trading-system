#!/usr/bin/env python3
"""
Оптимизированная качественная торговая система
Более реалистичные параметры для генерации сделок
"""

import pandas as pd
import numpy as np
import asyncio
import json
import talib
from datetime import datetime
from data_collector import BinanceDataCollector

class OptimizedQualitySystem:
    def __init__(self):
        # Оптимизированные параметры
        self.params = {
            'min_confidence': 0.6,  # Снижено с 0.75
            'min_signal_strength': 5,  # Снижено с 7
            'min_volume_ratio': 1.2,  # Снижено с 1.5
            'max_volatility': 0.08,  # Увеличено с 0.05
            'max_trades_per_day': 5,  # Увеличено с 3
            'min_time_between_trades': 30,  # Снижено с 60
            'take_profit': 0.015,  # 1.5%
            'stop_loss': 0.008,  # 0.8%
            'min_target_progress': 0.6,  # 60%
            'desired_target_progress': 0.75  # 75%
        }
        
        self.balance = 10000
        self.position = None
        self.trades = []
        self.daily_trades = {}
        self.last_trade_time = None
        
    def calculate_quality_indicators(self, df):
        """Расчет индикаторов качества"""
        # RSI
        df['rsi'] = talib.RSI(df['close'].values, timeperiod=14)
        
        # MACD
        macd, macd_signal, macd_hist = talib.MACD(df['close'].values)
        df['macd'] = macd
        df['macd_signal'] = macd_signal
        df['macd_hist'] = macd_hist
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = talib.BBANDS(df['close'].values, timeperiod=20)
        df['bb_upper'] = bb_upper
        df['bb_middle'] = bb_middle
        df['bb_lower'] = bb_lower
        
        # SMA
        df['sma_10'] = talib.SMA(df['close'].values, timeperiod=10)
        df['sma_20'] = talib.SMA(df['close'].values, timeperiod=20)
        df['sma_50'] = talib.SMA(df['close'].values, timeperiod=50)
        
        # Volume ratio
        df['volume_sma'] = talib.SMA(df['volume'].values, timeperiod=20)
        df['volume_sma_ratio'] = df['volume'] / df['volume_sma']
        
        # Volatility
        df['volatility'] = df['close'].rolling(20).std() / df['close']
        
        # Breakout quality
        df['price_range'] = df['high'] - df['low']
        df['avg_range'] = df['price_range'].rolling(20).mean()
        df['breakout_quality'] = df['price_range'] / df['avg_range']
        
        return df
        
    def calculate_signal_confidence(self, row):
        """Расчет уверенности в сигнале"""
        confidence_factors = []
        
        # RSI confidence (более мягкие условия)
        if row['rsi'] < 35 or row['rsi'] > 65:
            confidence_factors.append(0.8)
        elif row['rsi'] < 45 or row['rsi'] > 55:
            confidence_factors.append(0.6)
        else:
            confidence_factors.append(0.4)
            
        # MACD confidence
        macd_diff = abs(row['macd'] - row['macd_signal'])
        if macd_diff > 30:
            confidence_factors.append(0.8)
        elif macd_diff > 10:
            confidence_factors.append(0.6)
        else:
            confidence_factors.append(0.4)
            
        # Bollinger Bands confidence
        bb_width = row['bb_upper'] - row['bb_lower']
        if bb_width > 0:
            bb_position = (row['close'] - row['bb_lower']) / bb_width
            if bb_position < 0.2 or bb_position > 0.8:
                confidence_factors.append(0.8)
            elif bb_position < 0.3 or bb_position > 0.7:
                confidence_factors.append(0.6)
            else:
                confidence_factors.append(0.4)
        else:
            confidence_factors.append(0.5)
            
        # Volume confidence
        if row['volume_sma_ratio'] > 1.8:
            confidence_factors.append(0.8)
        elif row['volume_sma_ratio'] > 1.2:
            confidence_factors.append(0.6)
        else:
            confidence_factors.append(0.3)
            
        return np.mean(confidence_factors)
        
    def calculate_signal_strength(self, row):
        """Расчет силы сигнала"""
        strength = 0
        
        # RSI strength (более мягкие условия)
        if row['rsi'] < 25 or row['rsi'] > 75:
            strength += 3
        elif row['rsi'] < 35 or row['rsi'] > 65:
            strength += 2
        elif row['rsi'] < 45 or row['rsi'] > 55:
            strength += 1
            
        # MACD strength
        macd_diff = abs(row['macd'] - row['macd_signal'])
        if macd_diff > 50:
            strength += 3
        elif macd_diff > 20:
            strength += 2
        elif macd_diff > 10:
            strength += 1
            
        # Bollinger Bands strength
        bb_width = row['bb_upper'] - row['bb_lower']
        if bb_width > 0:
            bb_position = (row['close'] - row['bb_lower']) / bb_width
            if bb_position < 0.15 or bb_position > 0.85:
                strength += 2
            elif bb_position < 0.25 or bb_position > 0.75:
                strength += 1
            
        # Trend alignment strength
        sma_alignment = (row['sma_10'] > row['sma_20'] > row['sma_50'])
        if sma_alignment or (row['sma_10'] < row['sma_20'] < row['sma_50']):
            strength += 1
            
        return min(strength, 10)
    
    def check_trade_limits(self, timestamp):
        """Проверка лимитов на торговлю"""
        current_date = timestamp.date()
        
        # Проверка дневного лимита
        if current_date in self.daily_trades:
            if self.daily_trades[current_date] >= self.params['max_trades_per_day']:
                return False
                
        # Проверка времени между сделками
        if self.last_trade_time:
            time_diff = (timestamp - self.last_trade_time).total_seconds() / 60
            if time_diff < self.params['min_time_between_trades']:
                return False
                
        return True
    
    def generate_quality_signal(self, df, i):
        """Генерация качественного сигнала"""
        if i < 50:  # Нужна история для расчетов
            return None
            
        row = df.iloc[i]
        timestamp = pd.to_datetime(row.name)
        
        # Проверка лимитов торговли
        if not self.check_trade_limits(timestamp):
            return None
            
        confidence = self.calculate_signal_confidence(row)
        strength = self.calculate_signal_strength(row)
        
        # Более мягкие условия
        if confidence < self.params['min_confidence']:
            return None
            
        if strength < self.params['min_signal_strength']:
            return None
            
        if row['volume_sma_ratio'] < self.params['min_volume_ratio']:
            return None
            
        if row['volatility'] > self.params['max_volatility']:
            return None
            
        # Определение направления сигнала
        signal_type = None
        
        # Long условия (более мягкие)
        long_conditions = [
            row['rsi'] < 40,  # Было 30
            row['macd'] > row['macd_signal'],
            row['close'] < row['bb_lower'] * 1.01,  # Было 1.002
            row['sma_10'] > row['sma_20'],
            row['volume_sma_ratio'] > self.params['min_volume_ratio'],
            row['breakout_quality'] > 0.002  # Было 0.003
        ]
        
        # Short условия (более мягкие)
        short_conditions = [
            row['rsi'] > 60,  # Было 70
            row['macd'] < row['macd_signal'],
            row['close'] > row['bb_upper'] * 0.99,  # Было 0.998
            row['sma_10'] < row['sma_20'],
            row['volume_sma_ratio'] > self.params['min_volume_ratio'],
            row['breakout_quality'] > 0.002  # Было 0.003
        ]
        
        # Требуем выполнения минимум 4 из 6 условий (было 5)
        if sum(long_conditions) >= 4:
            signal_type = 'long'
        elif sum(short_conditions) >= 4:
            signal_type = 'short'
            
        if signal_type:
            return {
                'timestamp': timestamp,
                'signal': signal_type,
                'price': row['close'],
                'confidence': confidence,
                'strength': strength,
                'volume_ratio': row['volume_sma_ratio'],
                'volatility': row['volatility'],
                'breakout_quality': row['breakout_quality']
            }
            
        return None
    
    def quality_backtest(self, df):
        """Бэктест с фокусом на качество"""
        # Расчет индикаторов
        df = self.calculate_quality_indicators(df)
        
        self.balance = 10000
        self.position = None
        
        for i in range(len(df)):
            row = df.iloc[i]
            current_price = row['close']
            timestamp = pd.to_datetime(row.name)
            
            # Проверка открытой позиции
            if self.position:
                pnl_pct = 0
                if self.position['side'] == 'long':
                    pnl_pct = (current_price - self.position['entry_price']) / self.position['entry_price']
                else:
                    pnl_pct = (self.position['entry_price'] - current_price) / self.position['entry_price']
                
                # Проверка условий закрытия
                should_close = False
                exit_reason = ""
                
                if pnl_pct >= self.params['take_profit']:
                    should_close = True
                    exit_reason = "take_profit"
                elif pnl_pct <= -self.params['stop_loss']:
                    should_close = True
                    exit_reason = "stop_loss"
                
                if should_close:
                    pnl = self.position['size'] * pnl_pct
                    self.balance += pnl
                    
                    # Расчет прогресса к цели
                    target_progress = min(pnl_pct / self.params['take_profit'], 1.0) if pnl_pct > 0 else 0
                    
                    self.trades.append({
                        'entry_time': self.position['timestamp'],
                        'exit_time': timestamp,
                        'side': self.position['side'],
                        'entry_price': self.position['entry_price'],
                        'exit_price': current_price,
                        'size': self.position['size'],
                        'pnl': pnl,
                        'pnl_pct': pnl_pct,
                        'exit_reason': exit_reason,
                        'target_progress': target_progress,
                        'confidence': self.position['confidence'],
                        'strength': self.position['strength']
                    })
                    
                    self.position = None
                    self.last_trade_time = timestamp
                    
                    # Обновление дневной статистики
                    current_date = timestamp.date()
                    if current_date not in self.daily_trades:
                        self.daily_trades[current_date] = 0
                    self.daily_trades[current_date] += 1
            
            # Поиск новых сигналов
            if not self.position:
                signal = self.generate_quality_signal(df, i)
                if signal:
                    position_size = self.balance * 0.95  # 95% баланса
                    
                    self.position = {
                        'timestamp': signal['timestamp'],
                        'side': signal['signal'],
                        'entry_price': signal['price'],
                        'size': position_size,
                        'confidence': signal['confidence'],
                        'strength': signal['strength']
                    }
        
        return self.balance
    
    def calculate_results(self, final_balance):
        """Расчет результатов"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if not self.trades:
            return {
                'timestamp': timestamp,
                'total_trades': 0,
                'profitable_trades': 0,
                'winrate': 0,
                'total_pnl': 0,
                'final_balance': final_balance,
                'tp_trades': 0,
                'sl_trades': 0,
                'tp_sl_ratio': 0,
                'min_target_progress': 0,
                'desired_target_progress': 0
            }
        
        profitable_trades = sum(1 for trade in self.trades if trade['pnl'] > 0)
        total_pnl = sum(trade['pnl'] for trade in self.trades)
        winrate = profitable_trades / len(self.trades) * 100
        
        # Статистика по TP/SL
        tp_trades = sum(1 for trade in self.trades if trade['exit_reason'] == 'take_profit')
        sl_trades = sum(1 for trade in self.trades if trade['exit_reason'] == 'stop_loss')
        tp_sl_ratio = tp_trades / sl_trades if sl_trades > 0 else float('inf')
        
        # Прогресс к целям
        target_progresses = [trade['target_progress'] for trade in self.trades]
        min_target_achieved = sum(1 for p in target_progresses if p >= self.params['min_target_progress'])
        desired_target_achieved = sum(1 for p in target_progresses if p >= self.params['desired_target_progress'])
        
        min_target_progress = min_target_achieved / len(self.trades) * 100
        desired_target_progress = desired_target_achieved / len(self.trades) * 100
        
        return {
            'timestamp': timestamp,
            'total_trades': len(self.trades),
            'profitable_trades': profitable_trades,
            'winrate': winrate,
            'total_pnl': total_pnl,
            'total_pnl_pct': (total_pnl / 10000) * 100,
            'final_balance': final_balance,
            'tp_trades': tp_trades,
            'sl_trades': sl_trades,
            'tp_sl_ratio': tp_sl_ratio,
            'min_target_progress': min_target_progress,
            'desired_target_progress': desired_target_progress,
            'avg_confidence': np.mean([trade['confidence'] for trade in self.trades]),
            'avg_strength': np.mean([trade['strength'] for trade in self.trades])
        }

async def run_optimized_test():
    """Запуск оптимизированного теста"""
    print("🎯 Запуск оптимизированной качественной системы...")
    
    # Загрузка данных
    async with BinanceDataCollector() as collector:
        df = await collector.get_historical_data('BTCUSDT', '1h', 500)
    
    if df is None or df.empty:
        print("❌ Не удалось загрузить данные")
        return
        
    print(f"📊 Загружено {len(df)} свечей")
    
    # Создание и запуск системы
    system = OptimizedQualitySystem()
    final_balance = system.quality_backtest(df)
    
    # Расчет результатов
    results = system.calculate_results(final_balance)
    
    # Вывод результатов
    print(f"\n🎯 === РЕЗУЛЬТАТЫ ОПТИМИЗИРОВАННОЙ КАЧЕСТВЕННОЙ СИСТЕМЫ ===")
    print(f"📈 Всего сделок: {results['total_trades']}")
    print(f"✅ Прибыльных: {results['profitable_trades']}")
    print(f"🎯 Винрейт: {results['winrate']:.1f}%")
    print(f"💰 Общий PnL: {results['total_pnl_pct']:.2f}%")
    print(f"📊 Финальный баланс: ${results['final_balance']:.2f}")
    print(f"🎯 Take Profit: {results['tp_trades']}")
    print(f"🛑 Stop Loss: {results['sl_trades']}")
    print(f"📊 TP/SL соотношение: {results['tp_sl_ratio']:.2f}")
    print(f"🎯 Прогресс к цели 60%: {results['min_target_progress']:.1f}%")
    print(f"🌟 Прогресс к цели 75%: {results['desired_target_progress']:.1f}%")
    print(f"📊 Средняя уверенность: {results['avg_confidence']:.2f}")
    print(f"💪 Средняя сила: {results['avg_strength']:.1f}")
    
    # Сохранение результатов
    filename = f"optimized_quality_results_{results['timestamp']}.json"
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"💾 Результаты сохранены в {filename}")

if __name__ == "__main__":
    asyncio.run(run_optimized_test())