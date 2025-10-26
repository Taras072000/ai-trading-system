#!/usr/bin/env python3
"""
🔧 КАЛИБРОВКА LAVA AI МОДЕЛИ
Специальный скрипт для настройки параметров lava_ai модели
"""

import os
import sys
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any
import pandas as pd
import numpy as np

# Добавляем путь к модулям
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ai_modules.lava_ai import LavaAI
from data_collector import DataManager

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LavaAICalibrator:
    """Калибратор для настройки lava_ai модели"""
    
    def __init__(self):
        self.data_manager = DataManager()
        self.lava_ai = None
        self.test_symbols = ['TAOUSDT', 'CRVUSDT', 'ZRXUSDT', 'APTUSDT', 'SANDUSDT']
        
    async def initialize(self):
        """Инициализация модели"""
        logger.info("🚀 Инициализация Lava AI для калибровки...")
        self.lava_ai = LavaAI()
        logger.info("✅ Lava AI инициализирована")
    
    async def test_confidence_thresholds(self):
        """Тестирование различных порогов уверенности"""
        logger.info("🔍 Тестирование порогов уверенности для Lava AI...")
        
        thresholds = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
        results = {}
        
        # Получаем данные для тестирования
        test_data = {}
        for symbol in self.test_symbols:
            try:
                df = await self.data_manager.ensure_data_available(
                    symbol=symbol,
                    interval='1h',
                    days=7
                )
                if df is not None and len(df) > 50:
                    test_data[symbol] = df
                    logger.info(f"✅ Данные для {symbol}: {len(df)} свечей")
            except Exception as e:
                logger.error(f"❌ Ошибка загрузки данных для {symbol}: {e}")
        
        if not test_data:
            logger.error("❌ Нет данных для тестирования")
            return
        
        # Тестируем каждый порог
        for threshold in thresholds:
            logger.info(f"🎯 Тестирование порога уверенности: {threshold:.1%}")
            
            total_signals = 0
            valid_signals = 0
            confidences = []
            
            for symbol, df in test_data.items():
                try:
                    # Тестируем на последних 20 точках данных
                    for i in range(len(df) - 20, len(df), 2):
                        if i < 50:
                            continue
                            
                        current_data = df.iloc[:i+1].copy()
                        
                        # Получаем сигнал от lava_ai
                        try:
                            signal = self.lava_ai.get_signal(current_data, symbol)
                            
                            if signal and isinstance(signal, dict) and 'confidence' in signal:
                                total_signals += 1
                                confidence = signal['confidence']
                                confidences.append(confidence)
                                
                                if confidence >= threshold:
                                    valid_signals += 1
                                    
                        except Exception as e:
                            logger.debug(f"Ошибка получения сигнала: {e}")
                            continue
                            
                except Exception as e:
                    logger.error(f"❌ Ошибка тестирования {symbol}: {e}")
                    continue
            
            # Сохраняем результаты
            success_rate = (valid_signals / total_signals * 100) if total_signals > 0 else 0
            avg_confidence = np.mean(confidences) if confidences else 0
            
            results[threshold] = {
                'total_signals': total_signals,
                'valid_signals': valid_signals,
                'success_rate': success_rate,
                'avg_confidence': avg_confidence
            }
            
            logger.info(f"   📊 Порог {threshold:.1%}: {valid_signals}/{total_signals} сигналов ({success_rate:.1f}%)")
        
        return results
    
    async def analyze_signal_patterns(self):
        """Анализ паттернов сигналов lava_ai"""
        logger.info("📈 Анализ паттернов сигналов Lava AI...")
        
        signal_types = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
        confidence_ranges = {
            '0-10%': 0, '10-20%': 0, '20-30%': 0, '30-40%': 0, 
            '40-50%': 0, '50-60%': 0, '60-70%': 0, '70-80%': 0,
            '80-90%': 0, '90-100%': 0
        }
        
        all_confidences = []
        all_signals = []
        
        # Получаем данные
        for symbol in self.test_symbols:
            try:
                df = await self.data_manager.ensure_data_available(
                    symbol=symbol,
                    interval='1h',
                    days=7
                )
                
                if df is None or len(df) < 50:
                    continue
                
                logger.info(f"📊 Анализ паттернов для {symbol}...")
                
                # Анализируем каждые 3 часа
                for i in range(50, len(df), 3):
                    try:
                        current_data = df.iloc[:i+1].copy()
                        signal = self.lava_ai.get_signal(current_data, symbol)
                        
                        if signal and isinstance(signal, dict):
                            confidence = signal.get('confidence', 0)
                            direction = signal.get('direction', 'HOLD')
                            
                            all_confidences.append(confidence)
                            all_signals.append(signal)
                            
                            # Классификация по типу сигнала
                            if direction == 1 or direction == 'BUY':
                                signal_types['BUY'] += 1
                            elif direction == -1 or direction == 'SELL':
                                signal_types['SELL'] += 1
                            else:
                                signal_types['HOLD'] += 1
                            
                            # Классификация по уверенности
                            conf_percent = confidence * 100
                            if conf_percent < 10:
                                confidence_ranges['0-10%'] += 1
                            elif conf_percent < 20:
                                confidence_ranges['10-20%'] += 1
                            elif conf_percent < 30:
                                confidence_ranges['20-30%'] += 1
                            elif conf_percent < 40:
                                confidence_ranges['30-40%'] += 1
                            elif conf_percent < 50:
                                confidence_ranges['40-50%'] += 1
                            elif conf_percent < 60:
                                confidence_ranges['50-60%'] += 1
                            elif conf_percent < 70:
                                confidence_ranges['60-70%'] += 1
                            elif conf_percent < 80:
                                confidence_ranges['70-80%'] += 1
                            elif conf_percent < 90:
                                confidence_ranges['80-90%'] += 1
                            else:
                                confidence_ranges['90-100%'] += 1
                                
                    except Exception as e:
                        logger.debug(f"Ошибка анализа сигнала: {e}")
                        continue
                        
            except Exception as e:
                logger.error(f"❌ Ошибка анализа {symbol}: {e}")
                continue
        
        return {
            'signal_types': signal_types,
            'confidence_ranges': confidence_ranges,
            'all_confidences': all_confidences,
            'total_signals': len(all_signals),
            'avg_confidence': np.mean(all_confidences) if all_confidences else 0,
            'median_confidence': np.median(all_confidences) if all_confidences else 0,
            'std_confidence': np.std(all_confidences) if all_confidences else 0
        }
    
    def generate_calibration_report(self, threshold_results, pattern_analysis):
        """Генерация отчета по калибровке"""
        print("\n" + "="*80)
        print("🔧 КАЛИБРОВКА LAVA AI - ОТЧЕТ")
        print("="*80)
        
        print(f"\n📊 АНАЛИЗ ПАТТЕРНОВ СИГНАЛОВ:")
        print(f"   📈 Всего сигналов: {pattern_analysis['total_signals']}")
        print(f"   🎯 Средняя уверенность: {pattern_analysis['avg_confidence']:.1%}")
        print(f"   📊 Медианная уверенность: {pattern_analysis['median_confidence']:.1%}")
        print(f"   📏 Стандартное отклонение: {pattern_analysis['std_confidence']:.1%}")
        
        print(f"\n🎯 РАСПРЕДЕЛЕНИЕ ПО ТИПАМ СИГНАЛОВ:")
        total_signals = pattern_analysis['total_signals']
        for signal_type, count in pattern_analysis['signal_types'].items():
            percentage = (count / total_signals * 100) if total_signals > 0 else 0
            print(f"   {signal_type}: {count} ({percentage:.1f}%)")
        
        print(f"\n📊 РАСПРЕДЕЛЕНИЕ ПО УВЕРЕННОСТИ:")
        for range_name, count in pattern_analysis['confidence_ranges'].items():
            percentage = (count / total_signals * 100) if total_signals > 0 else 0
            print(f"   {range_name}: {count} ({percentage:.1f}%)")
        
        print(f"\n🎯 ТЕСТИРОВАНИЕ ПОРОГОВ УВЕРЕННОСТИ:")
        print("┌─────────┬─────────┬─────────┬─────────┬─────────────┐")
        print("│  ПОРОГ  │ СИГНАЛЫ │ ВАЛИДНЫЕ│ УСПЕХ % │ СР.УВЕРЕН.  │")
        print("├─────────┼─────────┼─────────┼─────────┼─────────────┤")
        
        best_threshold = None
        best_score = 0
        
        for threshold, result in threshold_results.items():
            success_rate = result['success_rate']
            avg_conf = result['avg_confidence']
            
            # Комбинированная оценка: баланс между количеством сигналов и качеством
            score = success_rate * (avg_conf * 100) / 100
            if score > best_score:
                best_score = score
                best_threshold = threshold
            
            print(f"│ {threshold:6.1%} │ {result['total_signals']:7} │ {result['valid_signals']:7} │ {success_rate:6.1f}% │ {avg_conf:10.1%} │")
        
        print("└─────────┴─────────┴─────────┴─────────┴─────────────┘")
        
        print(f"\n💡 РЕКОМЕНДАЦИИ ПО КАЛИБРОВКЕ:")
        
        if best_threshold:
            print(f"   🎯 Оптимальный порог уверенности: {best_threshold:.1%}")
            print(f"   📊 Ожидаемая производительность: {threshold_results[best_threshold]['success_rate']:.1f}%")
        
        # Анализ проблем
        low_confidence_signals = sum([
            pattern_analysis['confidence_ranges']['0-10%'],
            pattern_analysis['confidence_ranges']['10-20%']
        ])
        
        if low_confidence_signals > total_signals * 0.5:
            print(f"   ⚠️ ПРОБЛЕМА: {low_confidence_signals} сигналов с очень низкой уверенностью (<20%)")
            print(f"      - Рекомендуется пересмотреть алгоритм расчета уверенности")
        
        if pattern_analysis['avg_confidence'] < 0.3:
            print(f"   ⚠️ ПРОБЛЕМА: Средняя уверенность слишком низкая ({pattern_analysis['avg_confidence']:.1%})")
            print(f"      - Рекомендуется настроить параметры модели")
        
        # Рекомендации по настройке
        print(f"\n🔧 РЕКОМЕНДУЕМЫЕ НАСТРОЙКИ:")
        if best_threshold:
            print(f"   min_confidence = {best_threshold:.2f}")
        
        if pattern_analysis['signal_types']['HOLD'] > total_signals * 0.8:
            print(f"   ⚠️ Слишком много HOLD сигналов - увеличить чувствительность")
        elif pattern_analysis['signal_types']['HOLD'] < total_signals * 0.3:
            print(f"   ⚠️ Слишком мало HOLD сигналов - уменьшить чувствительность")

async def main():
    """Основная функция калибровки"""
    try:
        calibrator = LavaAICalibrator()
        await calibrator.initialize()
        
        # Тестируем пороги уверенности
        threshold_results = await calibrator.test_confidence_thresholds()
        
        # Анализируем паттерны сигналов
        pattern_analysis = await calibrator.analyze_signal_patterns()
        
        # Генерируем отчет
        calibrator.generate_calibration_report(threshold_results, pattern_analysis)
        
    except Exception as e:
        logger.error(f"❌ Критическая ошибка калибровки: {e}")
        sys.exit(1)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())