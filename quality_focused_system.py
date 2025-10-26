#!/usr/bin/env python3
"""
Система с фокусом на качество сигналов
Цель: минимизировать количество стоп-лоссов через строгую фильтрацию
"""

import pandas as pd
import numpy as np
import asyncio
import json
import talib
from datetime import datetime
from data_collector import BinanceDataCollector

class QualityFocusedSystem:
    def __init__(self):
        
        # Очень строгие параметры для качественных сигналов
        self.params = {
            'min_confidence': 0.85,  # Очень высокая уверенность
            'min_signal_strength': 9,  # Только самые сильные сигналы
            'min_volume_ratio': 1.5,  # Высокий объем
            'min_trend_alignment': 0.8,  # Сильное выравнивание тренда
            'min_momentum_strength': 0.7,  # Сильный моментум
            'max_volatility': 0.015,  # Ограничение волатильности
            'min_support_resistance_distance': 0.005,  # Минимальное расстояние до уровней
            'stop_loss_pct': 0.2,  # Тайт стоп
            'take_profit_pct': 0.6,  # Увеличенный профит
            'max_trades_per_day': 3,  # Ограничение количества сделок
            'min_time_between_trades': 120,  # Минимум 2 часа между сделками
        }
        
        self.trades = []
        self.last_trade_time = None
        self.daily_trades = {}
        
    def calculate_quality_indicators(self, df):
        """Расчет индикаторов качества сигнала"""
        # Базовые индикаторы через talib
        df['rsi'] = talib.RSI(df['close'].values, timeperiod=14)
        df['macd'], df['macd_signal'], _ = talib.MACD(df['close'].values)
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(df['close'].values)
        df['sma_10'] = talib.SMA(df['close'].values, timeperiod=10)
        df['sma_20'] = talib.SMA(df['close'].values, timeperiod=20)
        df['sma_50'] = talib.SMA(df['close'].values, timeperiod=50)
        
        # Дополнительные индикаторы качества
        df['price_ma_distance'] = abs(df['close'] - df['sma_20']) / df['close']
        df['volume_sma_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        df['volatility'] = df['close'].pct_change().rolling(20).std()
        
        # Индикатор силы тренда
        df['trend_strength'] = abs(df['close'] - df['close'].shift(20)) / df['close'].shift(20)
        
        # Индикатор качества пробоя
        df['breakout_quality'] = 0.0
        for i in range(20, len(df)):
            high_20 = df['high'].iloc[i-20:i].max()
            low_20 = df['low'].iloc[i-20:i].min()
            current_price = df['close'].iloc[i]
            
            if current_price > high_20:
                df.loc[df.index[i], 'breakout_quality'] = (current_price - high_20) / high_20
            elif current_price < low_20:
                df.loc[df.index[i], 'breakout_quality'] = (low_20 - current_price) / low_20
                
        return df
    
    def calculate_signal_confidence(self, row):
        """Расчет уверенности в сигнале с множественными факторами"""
        confidence_factors = []
        
        # RSI фактор
        if 25 <= row['rsi'] <= 35 or 65 <= row['rsi'] <= 75:
            confidence_factors.append(0.8)
        elif row['rsi'] < 25 or row['rsi'] > 75:
            confidence_factors.append(0.9)
        else:
            confidence_factors.append(0.3)
            
        # MACD фактор
        if abs(row['macd']) > abs(row['macd_signal']) * 1.2:
            confidence_factors.append(0.8)
        else:
            confidence_factors.append(0.4)
            
        # Bollinger Bands фактор
        bb_width = row['bb_upper'] - row['bb_lower']
        if bb_width > 0:
            bb_position = (row['close'] - row['bb_lower']) / bb_width
            if bb_position < 0.2 or bb_position > 0.8:
                confidence_factors.append(0.9)
            else:
                confidence_factors.append(0.5)
        else:
            confidence_factors.append(0.5)
            
        # Volume фактор
        if row['volume_sma_ratio'] > self.params['min_volume_ratio']:
            confidence_factors.append(0.9)
        else:
            confidence_factors.append(0.4)
            
        # Volatility фактор (низкая волатильность = выше качество)
        if row['volatility'] < self.params['max_volatility']:
            confidence_factors.append(0.8)
        else:
            confidence_factors.append(0.3)
            
        # Trend strength фактор
        if row['trend_strength'] > 0.01:
            confidence_factors.append(0.8)
        else:
            confidence_factors.append(0.4)
            
        # Breakout quality фактор
        if row['breakout_quality'] > 0.005:
            confidence_factors.append(0.9)
        else:
            confidence_factors.append(0.5)
            
        return np.mean(confidence_factors)
    
    def calculate_signal_strength(self, row):
        """Расчет силы сигнала (1-10)"""
        strength = 0
        
        # RSI strength
        if row['rsi'] < 20 or row['rsi'] > 80:
            strength += 3
        elif row['rsi'] < 30 or row['rsi'] > 70:
            strength += 2
        elif row['rsi'] < 40 or row['rsi'] > 60:
            strength += 1
            
        # MACD strength
        if abs(row['macd'] - row['macd_signal']) > 50:
            strength += 2
        elif abs(row['macd'] - row['macd_signal']) > 20:
            strength += 1
            
        # Volume strength
        if row['volume_sma_ratio'] > 2.0:
            strength += 2
        elif row['volume_sma_ratio'] > 1.5:
            strength += 1
            
        # Bollinger strength
        bb_width = row['bb_upper'] - row['bb_lower']
        if bb_width > 0:
            bb_position = (row['close'] - row['bb_lower']) / bb_width
            if bb_position < 0.1 or bb_position > 0.9:
                strength += 2
            elif bb_position < 0.2 or bb_position > 0.8:
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
        """Генерация высококачественного сигнала"""
        if i < 50:  # Нужна история для расчетов
            return None
            
        row = df.iloc[i]
        timestamp = pd.to_datetime(row.name)
        
        # Проверка лимитов торговли
        if not self.check_trade_limits(timestamp):
            return None
            
        confidence = self.calculate_signal_confidence(row)
        strength = self.calculate_signal_strength(row)
        
        # Очень строгие условия
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
        
        # Long условия (очень строгие)
        long_conditions = [
            row['rsi'] < 30,
            row['macd'] > row['macd_signal'],
            row['close'] < row['bb_lower'] * 1.002,  # Близко к нижней полосе
            row['sma_10'] > row['sma_20'],
            row['volume_sma_ratio'] > self.params['min_volume_ratio'],
            row['breakout_quality'] > 0.003
        ]
        
        # Short условия (очень строгие)
        short_conditions = [
            row['rsi'] > 70,
            row['macd'] < row['macd_signal'],
            row['close'] > row['bb_upper'] * 0.998,  # Близко к верхней полосе
            row['sma_10'] < row['sma_20'],
            row['volume_sma_ratio'] > self.params['min_volume_ratio'],
            row['breakout_quality'] > 0.003
        ]
        
        # Требуем выполнения минимум 5 из 6 условий
        if sum(long_conditions) >= 5:
            signal_type = 'long'
        elif sum(short_conditions) >= 5:
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
        # Сначала рассчитываем индикаторы
        df = self.calculate_quality_indicators(df)
        
        balance = 10000
        position = None
        
        for i in range(len(df)):
            row = df.iloc[i]
            timestamp = pd.to_datetime(row.name)
            
            # Проверка выхода из позиции
            if position:
                pnl_pct = 0
                close_reason = None
                
                if position['side'] == 'long':
                    pnl_pct = (row['close'] - position['entry_price']) / position['entry_price'] * 100
                    if pnl_pct >= self.params['take_profit_pct']:
                        close_reason = 'TP'
                    elif pnl_pct <= -self.params['stop_loss_pct']:
                        close_reason = 'SL'
                else:  # short
                    pnl_pct = (position['entry_price'] - row['close']) / position['entry_price'] * 100
                    if pnl_pct >= self.params['take_profit_pct']:
                        close_reason = 'TP'
                    elif pnl_pct <= -self.params['stop_loss_pct']:
                        close_reason = 'SL'
                
                if close_reason:
                    balance *= (1 + pnl_pct / 100)
                    
                    trade = {
                        'entry_time': position['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
                        'exit_time': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                        'side': position['side'],
                        'entry_price': position['entry_price'],
                        'exit_price': row['close'],
                        'pnl_pct': pnl_pct,
                        'close_reason': close_reason,
                        'confidence': position['confidence'],
                        'strength': position['strength']
                    }
                    self.trades.append(trade)
                    
                    # Обновляем статистику
                    current_date = timestamp.date()
                    if current_date not in self.daily_trades:
                        self.daily_trades[current_date] = 0
                    self.daily_trades[current_date] += 1
                    self.last_trade_time = timestamp
                    
                    position = None
            
            # Поиск нового сигнала
            if not position:
                signal = self.generate_quality_signal(df, i)
                if signal:
                    position = {
                        'timestamp': signal['timestamp'],
                        'side': signal['signal'],
                        'entry_price': signal['price'],
                        'confidence': signal['confidence'],
                        'strength': signal['strength']
                    }
        
        return balance
    
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
        
        profitable_trades = len([t for t in self.trades if t['pnl_pct'] > 0])
        tp_trades = len([t for t in self.trades if t['close_reason'] == 'TP'])
        sl_trades = len([t for t in self.trades if t['close_reason'] == 'SL'])
        
        total_pnl = sum(t['pnl_pct'] for t in self.trades)
        avg_profit = np.mean([t['pnl_pct'] for t in self.trades if t['pnl_pct'] > 0]) if profitable_trades > 0 else 0
        avg_loss = np.mean([t['pnl_pct'] for t in self.trades if t['pnl_pct'] < 0]) if (len(self.trades) - profitable_trades) > 0 else 0
        avg_confidence = np.mean([t['confidence'] for t in self.trades])
        
        winrate = (profitable_trades / len(self.trades)) * 100
        
        return {
            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'total_trades': len(self.trades),
            'profitable_trades': profitable_trades,
            'winrate': winrate,
            'total_pnl': total_pnl,
            'avg_profit': avg_profit,
            'avg_loss': avg_loss,
            'final_balance': final_balance,
            'avg_confidence': avg_confidence,
            'tp_trades': tp_trades,
            'sl_trades': sl_trades,
            'tp_sl_ratio': tp_trades / sl_trades if sl_trades > 0 else float('inf'),
            'min_target_progress': (winrate / 60) * 100,
            'desired_target_progress': (winrate / 75) * 100,
            'trades': self.trades
        }

def run_quality_test():
    """Запуск теста качественной системы"""
    print("🎯 Запуск системы с фокусом на качество сигналов...")
    
    # Загрузка данных
    collector = BinanceDataCollector()
    
    async def get_data():
        async with collector:
            return await collector.get_historical_data('BTCUSDT', '1m', days=7)
    
    # Получение данных
    df = asyncio.run(get_data())
    if df.empty:
        print("❌ Не удалось получить данные")
        return
    
    print(f"📊 Загружено {len(df)} свечей")
    
    # Создание и запуск системы
    system = QualityFocusedSystem()
    final_balance = system.quality_backtest(df)
    
    # Расчет результатов
    results = system.calculate_results(final_balance)
    
    # Вывод результатов
    print(f"\n🎯 === РЕЗУЛЬТАТЫ КАЧЕСТВЕННОЙ СИСТЕМЫ ===")
    print(f"📈 Всего сделок: {results['total_trades']}")
    print(f"✅ Прибыльных: {results['profitable_trades']}")
    print(f"🎯 Винрейт: {results['winrate']:.1f}%")
    print(f"💰 Общий PnL: {results['total_pnl']:.2f}%")
    print(f"📊 Финальный баланс: ${results['final_balance']:.2f}")
    print(f"🎯 Take Profit: {results['tp_trades']}")
    print(f"🛑 Stop Loss: {results['sl_trades']}")
    print(f"📊 TP/SL соотношение: {results['tp_sl_ratio']:.2f}")
    print(f"🎯 Прогресс к цели 60%: {results['min_target_progress']:.1f}%")
    print(f"🌟 Прогресс к цели 75%: {results['desired_target_progress']:.1f}%")
    
    # Сохранение результатов
    filename = f"quality_focused_results_{results['timestamp']}.json"
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"💾 Результаты сохранены в {filename}")

if __name__ == "__main__":
    run_quality_test()