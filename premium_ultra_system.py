#!/usr/bin/env python3
"""
🏆 ПРЕМИУМ УЛЬТРА-СИСТЕМА ДЛЯ ДОСТИЖЕНИЯ 60%+ ВИНРЕЙТА
Основана на лучших результатах сбалансированной v2 системы
Цель: Минимальный винрейт 60%, желаемый 75%
"""

import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime
import asyncio
from data_collector import BinanceDataCollector
from utils.timezone_utils import get_utc_now
import warnings
warnings.filterwarnings('ignore')

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PremiumUltraSystem:
    def __init__(self):
        # 🎯 ПРЕМИУМ ПАРАМЕТРЫ ДЛЯ ВЫСОКОГО ВИНРЕЙТА
        self.min_volume_ratio = 1.20  # Повышен для лучшего качества
        self.min_confidence = 0.70    # Высокая уверенность
        self.min_signal_strength = 6  # Только сильные сигналы
        self.min_trend_alignment = 0.75  # Строгое выравнивание тренда
        
        # 📊 УЛУЧШЕННЫЕ ПАРАМЕТРЫ РИСК-МЕНЕДЖМЕНТА
        self.stop_loss_pct = 0.25     # Уменьшен SL
        self.take_profit_pct = 0.55   # Увеличен TP для лучшего R:R
        self.max_trades_per_day = 8   # Ограничение на качество
        
        # 🔍 ПРЕМИУМ ФИЛЬТРЫ
        self.min_rsi_divergence = 15  # Сильная дивергенция
        self.min_macd_strength = 0.0008  # Сильный MACD сигнал
        self.min_bb_squeeze = 0.015   # Значительное сжатие BB
        self.min_volume_spike = 1.8   # Сильный всплеск объема
        
        self.results = []
        self.balance = 10000
        self.trades_today = 0
        self.last_trade_date = None
        
    def calculate_premium_indicators(self, df):
        """Расчет премиум индикаторов с улучшенной точностью"""
        
        # 📈 БАЗОВЫЕ ИНДИКАТОРЫ
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()
        
        # 🎯 RSI с улучшенной логикой
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # 📊 MACD с сигнальной линией
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # 🔔 Bollinger Bands
        bb_period = 20
        df['bb_middle'] = df['close'].rolling(bb_period).mean()
        bb_std = df['close'].rolling(bb_period).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        # 📈 Stochastic
        low_14 = df['low'].rolling(14).min()
        high_14 = df['high'].rolling(14).max()
        df['stoch_k'] = 100 * ((df['close'] - low_14) / (high_14 - low_14))
        df['stoch_d'] = df['stoch_k'].rolling(3).mean()
        
        # 💹 Volume indicators
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # 🎯 ПРЕМИУМ ДОПОЛНИТЕЛЬНЫЕ ИНДИКАТОРЫ
        
        # Williams %R
        df['williams_r'] = -100 * ((high_14 - df['close']) / (high_14 - low_14))
        
        # CCI (Commodity Channel Index)
        tp = (df['high'] + df['low'] + df['close']) / 3
        df['cci'] = (tp - tp.rolling(20).mean()) / (0.015 * tp.rolling(20).std())
        
        # ATR для волатильности
        tr1 = df['high'] - df['low']
        tr2 = abs(df['high'] - df['close'].shift())
        tr3 = abs(df['low'] - df['close'].shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['atr'] = tr.rolling(14).mean()
        
        # Momentum
        df['momentum'] = df['close'] / df['close'].shift(10) - 1
        
        return df
    
    def generate_premium_signal(self, df, i):
        """Генерация премиум сигналов с ультра-строгими критериями"""
        
        if i < 60:  # Нужна история для расчетов
            return None, 0, 0
            
        current = df.iloc[i]
        prev = df.iloc[i-1]
        prev2 = df.iloc[i-2]
        
        # 🎯 ПРЕМИУМ УСЛОВИЯ ДЛЯ LONG
        long_conditions = []
        long_strength = 0
        
        # 1. RSI условия (улучшенные)
        if 25 <= current['rsi'] <= 45 and current['rsi'] > prev['rsi']:
            long_conditions.append("rsi_oversold_recovery")
            long_strength += 2
            
        # 2. MACD бычий сигнал
        if (current['macd'] > current['macd_signal'] and 
            prev['macd'] <= prev['macd_signal'] and
            current['macd_histogram'] > 0.0005):
            long_conditions.append("macd_bullish_cross")
            long_strength += 2
            
        # 3. Bollinger Bands отскок
        if (current['close'] <= current['bb_lower'] * 1.02 and 
            current['close'] > prev['close'] and
            current['bb_width'] > self.min_bb_squeeze):
            long_conditions.append("bb_bounce")
            long_strength += 2
            
        # 4. Stochastic условия
        if (current['stoch_k'] < 25 and current['stoch_k'] > prev['stoch_k'] and
            current['stoch_d'] < 25):
            long_conditions.append("stoch_oversold")
            long_strength += 1
            
        # 5. Volume подтверждение
        if current['volume_ratio'] >= self.min_volume_spike:
            long_conditions.append("volume_spike")
            long_strength += 2
            
        # 6. Williams %R
        if current['williams_r'] > -85 and prev['williams_r'] <= -85:
            long_conditions.append("williams_recovery")
            long_strength += 1
            
        # 7. CCI условие
        if current['cci'] < -150 and current['cci'] > prev['cci']:
            long_conditions.append("cci_oversold_recovery")
            long_strength += 1
            
        # 8. Momentum подтверждение
        if current['momentum'] > -0.02 and current['momentum'] > prev['momentum']:
            long_conditions.append("momentum_positive")
            long_strength += 1
            
        # 9. Тренд выравнивание (улучшенное)
        trend_score = 0
        if current['close'] > current['sma_20']:
            trend_score += 0.3
        if current['sma_20'] > current['sma_50']:
            trend_score += 0.3
        if current['ema_12'] > current['ema_26']:
            trend_score += 0.4
            
        if trend_score >= self.min_trend_alignment:
            long_conditions.append("trend_alignment")
            long_strength += 2
        
        # 🎯 ПРЕМИУМ УСЛОВИЯ ДЛЯ SHORT
        short_conditions = []
        short_strength = 0
        
        # 1. RSI условия
        if 55 <= current['rsi'] <= 75 and current['rsi'] < prev['rsi']:
            short_conditions.append("rsi_overbought_decline")
            short_strength += 2
            
        # 2. MACD медвежий сигнал
        if (current['macd'] < current['macd_signal'] and 
            prev['macd'] >= prev['macd_signal'] and
            current['macd_histogram'] < -0.0005):
            short_conditions.append("macd_bearish_cross")
            short_strength += 2
            
        # 3. Bollinger Bands отскок вниз
        if (current['close'] >= current['bb_upper'] * 0.98 and 
            current['close'] < prev['close'] and
            current['bb_width'] > self.min_bb_squeeze):
            short_conditions.append("bb_rejection")
            short_strength += 2
            
        # 4. Stochastic условия
        if (current['stoch_k'] > 75 and current['stoch_k'] < prev['stoch_k'] and
            current['stoch_d'] > 75):
            short_conditions.append("stoch_overbought")
            short_strength += 1
            
        # 5. Volume подтверждение
        if current['volume_ratio'] >= self.min_volume_spike:
            short_conditions.append("volume_spike")
            short_strength += 2
            
        # 6. Williams %R
        if current['williams_r'] < -15 and prev['williams_r'] >= -15:
            short_conditions.append("williams_decline")
            short_strength += 1
            
        # 7. CCI условие
        if current['cci'] > 150 and current['cci'] < prev['cci']:
            short_conditions.append("cci_overbought_decline")
            short_strength += 1
            
        # 8. Momentum подтверждение
        if current['momentum'] < 0.02 and current['momentum'] < prev['momentum']:
            short_conditions.append("momentum_negative")
            short_strength += 1
            
        # 9. Тренд выравнивание для SHORT
        bear_trend_score = 0
        if current['close'] < current['sma_20']:
            bear_trend_score += 0.3
        if current['sma_20'] < current['sma_50']:
            bear_trend_score += 0.3
        if current['ema_12'] < current['ema_26']:
            bear_trend_score += 0.4
            
        if bear_trend_score >= self.min_trend_alignment:
            short_conditions.append("bear_trend_alignment")
            short_strength += 2
        
        # 🏆 ПРЕМИУМ КРИТЕРИИ ОТБОРА
        # Требуется минимум 5 из 9 условий И сила >= 8
        if len(long_conditions) >= 5 and long_strength >= 8:
            confidence = min(0.95, 0.5 + (long_strength * 0.05) + (len(long_conditions) * 0.03))
            if confidence >= self.min_confidence:
                return 'long', confidence, long_strength
                
        if len(short_conditions) >= 5 and short_strength >= 8:
            confidence = min(0.95, 0.5 + (short_strength * 0.05) + (len(short_conditions) * 0.03))
            if confidence >= self.min_confidence:
                return 'short', confidence, short_strength
        
        return None, 0, 0
    
    def premium_backtest(self, df):
        """Премиум бэктест с улучшенным риск-менеджментом"""
        
        df = self.calculate_premium_indicators(df)
        trades = []
        current_position = None
        
        logger.info(f"🚀 Запуск премиум бэктеста на {len(df)} свечах...")
        
        for i in range(len(df)):
            current_date = pd.to_datetime(df.iloc[i]['timestamp']).date()
            
            # Сброс счетчика сделок в новый день
            if self.last_trade_date != current_date:
                self.trades_today = 0
                self.last_trade_date = current_date
            
            # Проверка лимита сделок в день
            if self.trades_today >= self.max_trades_per_day:
                continue
                
            current_price = df.iloc[i]['close']
            
            # Закрытие позиции
            if current_position:
                pnl_pct = 0
                close_reason = ""
                
                if current_position['side'] == 'long':
                    pnl_pct = (current_price - current_position['entry_price']) / current_position['entry_price'] * 100
                    
                    if pnl_pct >= self.take_profit_pct:
                        close_reason = "TP"
                    elif pnl_pct <= -self.stop_loss_pct:
                        close_reason = "SL"
                        
                else:  # short
                    pnl_pct = (current_position['entry_price'] - current_price) / current_position['entry_price'] * 100
                    
                    if pnl_pct >= self.take_profit_pct:
                        close_reason = "TP"
                    elif pnl_pct <= -self.stop_loss_pct:
                        close_reason = "SL"
                
                if close_reason:
                    self.balance *= (1 + pnl_pct / 100)
                    
                    trade_result = {
                        'entry_time': current_position['entry_time'],
                        'exit_time': df.iloc[i]['timestamp'],
                        'side': current_position['side'],
                        'entry_price': current_position['entry_price'],
                        'exit_price': current_price,
                        'pnl_pct': pnl_pct,
                        'close_reason': close_reason,
                        'confidence': current_position['confidence'],
                        'strength': current_position['strength']
                    }
                    
                    trades.append(trade_result)
                    current_position = None
                    continue
            
            # Поиск новых сигналов
            if not current_position:
                signal, confidence, strength = self.generate_premium_signal(df, i)
                
                if signal and confidence >= self.min_confidence and strength >= self.min_signal_strength:
                    current_position = {
                        'side': signal,
                        'entry_price': current_price,
                        'entry_time': df.iloc[i]['timestamp'],
                        'confidence': confidence,
                        'strength': strength
                    }
                    self.trades_today += 1
        
        return trades
    
    def run_premium_test(self):
        """Запуск премиум теста системы"""
        
        logger.info("🏆 ЗАПУСК ПРЕМИУМ УЛЬТРА-СИСТЕМЫ")
        logger.info("=" * 60)
        
        # Загрузка данных через data_collector
        collector = BinanceDataCollector()
        
        async def get_data():
            async with collector:
                return await collector.get_historical_data('BTCUSDT', '1m', days=7)
        
        try:
            df = asyncio.run(get_data())
            if df is None or len(df) == 0:
                logger.error("❌ Не удалось загрузить данные")
                return
            logger.info(f"📊 Загружено {len(df)} свечей")
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки данных: {e}")
            return
        
        # Запуск бэктеста
        trades = self.premium_backtest(df)
        
        if not trades:
            logger.warning("⚠️ Не найдено сделок с премиум критериями!")
            logger.info("💡 Рекомендация: Снизить строгость параметров")
            return
        
        # Анализ результатов
        total_trades = len(trades)
        profitable_trades = len([t for t in trades if t['pnl_pct'] > 0])
        winrate = (profitable_trades / total_trades) * 100 if total_trades > 0 else 0
        
        total_pnl = sum([t['pnl_pct'] for t in trades])
        avg_profit = np.mean([t['pnl_pct'] for t in trades if t['pnl_pct'] > 0]) if profitable_trades > 0 else 0
        avg_loss = np.mean([t['pnl_pct'] for t in trades if t['pnl_pct'] < 0]) if (total_trades - profitable_trades) > 0 else 0
        
        tp_trades = len([t for t in trades if t['close_reason'] == 'TP'])
        sl_trades = len([t for t in trades if t['close_reason'] == 'SL'])
        
        avg_confidence = np.mean([t['confidence'] for t in trades])
        
        # Вывод результатов
        logger.info("🏆 РЕЗУЛЬТАТЫ ПРЕМИУМ УЛЬТРА-СИСТЕМЫ")
        logger.info("=" * 60)
        logger.info(f"📊 Всего сделок: {total_trades}")
        logger.info(f"🏆 Прибыльных сделок: {profitable_trades}")
        logger.info(f"📈 Винрейт: {winrate:.1f}%")
        logger.info(f"💰 Средняя прибыль: {avg_profit:.2f}%")
        logger.info(f"📉 Средний убыток: {avg_loss:.2f}%")
        logger.info(f"💵 Общая доходность: {total_pnl:.2f}%")
        logger.info(f"🏦 Финальный баланс: ${self.balance:.2f}")
        logger.info(f"🎯 Средняя уверенность: {avg_confidence:.3f}")
        logger.info("")
        
        # Прогресс к цели
        min_target = 60
        desired_target = 75
        min_progress = (winrate / min_target) * 100
        desired_progress = (winrate / desired_target) * 100
        
        logger.info(f"📊 Прогресс к минимальной цели: {winrate:.1f}% из {min_target}% ({min_progress:.1f}%)")
        logger.info(f"🎯 Прогресс к желаемой цели: {winrate:.1f}% из {desired_target}% ({desired_progress:.1f}%)")
        logger.info("")
        
        # Детальная статистика
        logger.info("📊 Детальная статистика:")
        logger.info(f"✅ Take Profit сделок: {tp_trades} ({tp_trades/total_trades*100:.1f}%)")
        logger.info(f"❌ Stop Loss сделок: {sl_trades} ({sl_trades/total_trades*100:.1f}%)")
        logger.info(f"💰 Средняя прибыль TP: {avg_profit:.2f}%")
        logger.info(f"📉 Средний убыток SL: {avg_loss:.2f}%")
        
        if avg_loss != 0:
            rr_ratio = abs(avg_profit / avg_loss)
            logger.info(f"⚖️ Risk-Reward соотношение: 1:{rr_ratio:.2f}")
        
        high_conf_trades = [t for t in trades if t['confidence'] > 0.75]
        if high_conf_trades:
            high_conf_profitable = len([t for t in high_conf_trades if t['pnl_pct'] > 0])
            high_conf_winrate = (high_conf_profitable / len(high_conf_trades)) * 100
            logger.info(f"🎯 Винрейт высокой уверенности (>0.75): {high_conf_winrate:.1f}% ({high_conf_profitable}/{len(high_conf_trades)})")
        
        logger.info("=" * 60)
        
        # Сохранение результатов
        timestamp = get_utc_now().strftime("%Y%m%d_%H%M%S")
        filename = f"premium_ultra_results_{timestamp}.json"
        
        results = {
            'timestamp': timestamp,
            'total_trades': total_trades,
            'profitable_trades': profitable_trades,
            'winrate': winrate,
            'total_pnl': total_pnl,
            'avg_profit': avg_profit,
            'avg_loss': avg_loss,
            'final_balance': self.balance,
            'avg_confidence': avg_confidence,
            'tp_trades': tp_trades,
            'sl_trades': sl_trades,
            'min_target_progress': min_progress,
            'desired_target_progress': desired_progress,
            'trades': trades
        }
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"💾 Результаты сохранены в {filename}")

if __name__ == "__main__":
    system = PremiumUltraSystem()
    system.run_premium_test()