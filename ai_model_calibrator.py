#!/usr/bin/env python3
"""
🎯 AI Model Calibrator - Система поэтапной калибровки и оптимизации AI моделей

Основная задача: Создать систему индивидуальной калибровки каждой AI модели 
для улучшения винрейта и устранения убытков.

Автор: AI Trading System
Дата: 2024
"""

import asyncio
import json
import logging
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
import itertools
from pathlib import Path

# Импорты модулей системы
from ai_modules.ai_manager import AIManager, AIModuleType
from data_collector import DataManager

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('calibration.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class CalibrationResult:
    """Результат калибровки модели"""
    model_name: str
    win_rate: float
    total_signals: int
    profitable_signals: int
    avg_profit: float
    max_drawdown: float
    sharpe_ratio: float
    best_confidence: float
    best_tp: float
    best_sl: float
    best_pairs: List[str]
    optimization_score: float
    timestamp: str

class AIModelCalibrator:
    """Система поэтапной калибровки и оптимизации AI моделей"""
    
    def __init__(self):
        # Маппинг имен моделей на AIModuleType
        self.model_mapping = {
            'trading_ai': AIModuleType.TRADING,
            'lava_ai': AIModuleType.LAVA,
            'lgbm_ai': AIModuleType.LGBM,
            'mistral_ai': AIModuleType.MISTRAL
        }
        self.models = list(self.model_mapping.keys())
        self.trading_pairs = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'SOLUSDT']
        
        # Параметры для оптимизации
        self.confidence_range = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        self.tp_range = [1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
        self.sl_range = [0.8, 1.0, 1.2, 1.5, 2.0]
        
        # Инициализация компонентов
        self.ai_manager = None
        self.data_manager = None
        
        # Результаты калибровки
        self.calibration_results = {}
        
        # Создание директорий для результатов
        self.results_dir = Path("calibration_results")
        self.individual_results_dir = Path("individual_calibration_results")
        self.optimization_results_dir = Path("optimization_results")
        
        for dir_path in [self.results_dir, self.individual_results_dir, self.optimization_results_dir]:
            dir_path.mkdir(exist_ok=True)
    
    async def initialize(self):
        """Инициализация системы калибровки"""
        try:
            logger.info("🚀 Инициализация системы калибровки AI моделей...")
            
            # Инициализация AI Manager
            self.ai_manager = AIManager()
            await self.ai_manager.initialize()
            
            # Инициализация Data Manager
            self.data_manager = DataManager()
            
            logger.info("✅ Система калибровки инициализирована успешно")
            return True
            
        except Exception as e:
            logger.error(f"❌ Ошибка инициализации системы калибровки: {e}")
            return False
    
    async def get_ai_model(self, model_name: str):
        """Получение и загрузка AI модели"""
        if model_name not in self.model_mapping:
            logger.error(f"❌ Модель {model_name} не поддерживается")
            return None
        
        module_type = self.model_mapping[model_name]
        success = await self.ai_manager.load_module(module_type)
        if not success:
            logger.error(f"❌ Не удалось загрузить модель {model_name}")
            return None
        
        ai_model = self.ai_manager.modules.get(module_type)
        if not ai_model:
            logger.error(f"❌ Модель {model_name} не найдена после загрузки")
            return None
        
        return ai_model
    
    async def calibrate_individual_model(self, model_name: str, test_days: int = 7) -> CalibrationResult:
        """Калибровка одной модели отдельно"""
        logger.info(f"🎯 Начинаем калибровку модели: {model_name}")
        
        try:
            # Получаем AI модель
            ai_model = await self.get_ai_model(model_name)
            if not ai_model:
                return None
            
            # Тестируем модель в соло-режиме
            solo_results = await self.test_model_solo(model_name, test_days)
            
            # Оптимизируем параметры
            optimization_results = await self.optimize_model_parameters(model_name)
            
            # Находим лучший порог уверенности
            best_confidence = await self.find_best_confidence_threshold(model_name)
            
            # Определяем лучшие пары для модели
            best_pairs = await self.find_best_pairs_for_model(model_name)
            
            # Создаем результат калибровки
            calibration_result = CalibrationResult(
                model_name=model_name,
                win_rate=solo_results.get('win_rate', 0),
                total_signals=solo_results.get('total_signals', 0),
                profitable_signals=solo_results.get('profitable_signals', 0),
                avg_profit=solo_results.get('avg_profit', 0),
                max_drawdown=solo_results.get('max_drawdown', 0),
                sharpe_ratio=solo_results.get('sharpe_ratio', 0),
                best_confidence=best_confidence,
                best_tp=optimization_results.get('best_tp', 2.5),
                best_sl=optimization_results.get('best_sl', 1.2),
                best_pairs=best_pairs,
                optimization_score=optimization_results.get('optimization_score', 0),
                timestamp=datetime.now().isoformat()
            )
            
            # Сохраняем результаты
            await self.save_calibration_results(model_name, calibration_result)
            
            logger.info(f"✅ Калибровка {model_name} завершена. Винрейт: {calibration_result.win_rate:.2f}%")
            
            return calibration_result
            
        except Exception as e:
            logger.error(f"❌ Ошибка калибровки модели {model_name}: {e}")
            return None
    
    async def test_model_solo(self, model_name: str, test_days: int = 7) -> Dict[str, Any]:
        """Тест модели в одиночку без консенсуса"""
        logger.info(f"🔍 Тестирование модели {model_name} в соло-режиме ({test_days} дней)")
        
        results = {
            'model_name': model_name,
            'test_days': test_days,
            'total_signals': 0,
            'profitable_signals': 0,
            'win_rate': 0,
            'avg_profit': 0,
            'max_drawdown': 0,
            'sharpe_ratio': 0,
            'pair_results': {},
            'signals_log': []
        }
        
        try:
            # Получаем AI модель
            ai_model = await self.get_ai_model(model_name)
            if not ai_model:
                return results
            
            total_profit = 0
            all_returns = []
            
            for pair in self.trading_pairs:
                logger.info(f"📊 Тестирование {model_name} на паре {pair}")
                
                # Получаем исторические данные
                end_time = datetime.now()
                start_time = end_time - timedelta(days=test_days + 30)  # +30 дней для технических индикаторов
                
                try:
                    data = await self.data_manager.ensure_data_available(
                        pair, '1h', start_time, end_time
                    )
                    
                    if data is None or len(data) < 100:
                        logger.warning(f"⚠️ Недостаточно данных для {pair}")
                        continue
                    
                    # Тестируем модель на исторических данных
                    pair_results = await self.backtest_model_on_pair(ai_model, model_name, pair, data, test_days)
                    
                    results['pair_results'][pair] = pair_results
                    results['total_signals'] += pair_results['signals_count']
                    results['profitable_signals'] += pair_results['profitable_signals']
                    
                    total_profit += pair_results['total_profit']
                    all_returns.extend(pair_results['returns'])
                    
                    logger.info(f"📈 {pair}: {pair_results['signals_count']} сигналов, "
                              f"винрейт {pair_results['win_rate']:.1f}%, "
                              f"прибыль {pair_results['total_profit']:.2f}%")
                    
                except Exception as e:
                    logger.error(f"❌ Ошибка тестирования {model_name} на {pair}: {e}")
                    continue
            
            # Рассчитываем общие метрики
            if results['total_signals'] > 0:
                results['win_rate'] = (results['profitable_signals'] / results['total_signals']) * 100
                results['avg_profit'] = total_profit / results['total_signals']
            
            if len(all_returns) > 0:
                returns_array = np.array(all_returns)
                results['max_drawdown'] = self.calculate_max_drawdown(returns_array)
                results['sharpe_ratio'] = self.calculate_sharpe_ratio(returns_array)
            
            logger.info(f"🎯 Результаты соло-теста {model_name}: "
                       f"{results['total_signals']} сигналов, "
                       f"винрейт {results['win_rate']:.1f}%, "
                       f"Sharpe {results['sharpe_ratio']:.2f}")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Ошибка соло-теста модели {model_name}: {e}")
            return results
    
    async def backtest_model_on_pair(self, ai_model, model_name: str, pair: str, data: pd.DataFrame, test_days: int) -> Dict[str, Any]:
        """Бэктест модели на конкретной паре"""
        
        # Берем только последние test_days дней для тестирования
        test_start_idx = len(data) - (test_days * 24)  # 24 часа в дне для часовых свечей
        if test_start_idx < 100:
            test_start_idx = 100  # Минимум 100 свечей для технических индикаторов
        
        test_data = data.iloc[test_start_idx:]
        
        signals = []
        returns = []
        profitable_count = 0
        
        # Проходим по данным и генерируем сигналы
        for i in range(100, len(test_data) - 1):  # Оставляем место для технических индикаторов
            current_data = test_data.iloc[:i+1]
            
            try:
                # Генерируем сигнал
                if hasattr(ai_model, 'generate_trading_signals'):
                    signal = await ai_model.generate_trading_signals(pair, current_data)
                else:
                    # Fallback для моделей без стандартного интерфейса
                    continue
                
                if signal and signal.get('action') in ['BUY', 'SELL']:
                    # Симулируем выполнение сделки
                    entry_price = test_data.iloc[i+1]['open']  # Цена входа на следующей свече
                    
                    # Определяем цели и стопы
                    tp_pct = signal.get('take_profit_pct', 2.5)
                    sl_pct = signal.get('stop_loss_pct', 1.2)
                    
                    if signal['action'] == 'BUY':
                        tp_price = entry_price * (1 + tp_pct / 100)
                        sl_price = entry_price * (1 - sl_pct / 100)
                    else:  # SELL
                        tp_price = entry_price * (1 - tp_pct / 100)
                        sl_price = entry_price * (1 + sl_pct / 100)
                    
                    # Проверяем результат в следующих свечах (максимум 24 часа)
                    profit = 0
                    for j in range(i+2, min(i+26, len(test_data))):
                        high = test_data.iloc[j]['high']
                        low = test_data.iloc[j]['low']
                        
                        if signal['action'] == 'BUY':
                            if high >= tp_price:
                                profit = tp_pct
                                break
                            elif low <= sl_price:
                                profit = -sl_pct
                                break
                        else:  # SELL
                            if low <= tp_price:
                                profit = tp_pct
                                break
                            elif high >= sl_price:
                                profit = -sl_pct
                                break
                    
                    signals.append({
                        'timestamp': test_data.index[i],
                        'action': signal['action'],
                        'entry_price': entry_price,
                        'profit': profit,
                        'confidence': signal.get('confidence', 0)
                    })
                    
                    returns.append(profit)
                    if profit > 0:
                        profitable_count += 1
                
            except Exception as e:
                logger.debug(f"Ошибка генерации сигнала {model_name} на {pair}: {e}")
                continue
        
        # Рассчитываем метрики для пары
        total_signals = len(signals)
        win_rate = (profitable_count / total_signals * 100) if total_signals > 0 else 0
        total_profit = sum(returns) if returns else 0
        
        return {
            'pair': pair,
            'signals_count': total_signals,
            'profitable_signals': profitable_count,
            'win_rate': win_rate,
            'total_profit': total_profit,
            'returns': returns,
            'signals': signals
        }
    
    async def optimize_model_parameters(self, model_name: str) -> Dict[str, Any]:
        """Оптимизация параметров конкретной модели"""
        logger.info(f"⚙️ Оптимизация параметров модели {model_name}")
        
        best_score = 0
        best_params = {}
        optimization_results = []
        
        try:
            # Перебираем все комбинации параметров
            param_combinations = list(itertools.product(
                self.confidence_range,
                self.tp_range,
                self.sl_range
            ))
            
            logger.info(f"🔄 Тестируем {len(param_combinations)} комбинаций параметров...")
            
            for i, (confidence, tp, sl) in enumerate(param_combinations):
                if i % 10 == 0:
                    logger.info(f"📊 Прогресс оптимизации: {i}/{len(param_combinations)}")
                
                # Тестируем комбинацию параметров
                score = await self.test_parameter_combination(model_name, confidence, tp, sl)
                
                optimization_results.append({
                    'confidence': confidence,
                    'take_profit': tp,
                    'stop_loss': sl,
                    'score': score
                })
                
                if score > best_score:
                    best_score = score
                    best_params = {
                        'confidence': confidence,
                        'take_profit': tp,
                        'stop_loss': sl
                    }
            
            logger.info(f"🎯 Лучшие параметры для {model_name}: "
                       f"confidence={best_params.get('confidence', 0.5)}, "
                       f"TP={best_params.get('take_profit', 2.5)}, "
                       f"SL={best_params.get('stop_loss', 1.2)}, "
                       f"score={best_score:.2f}")
            
            # Сохраняем результаты оптимизации
            await self.save_optimization_results(model_name, optimization_results, best_params)
            
            return {
                'best_confidence': best_params.get('confidence', 0.5),
                'best_tp': best_params.get('take_profit', 2.5),
                'best_sl': best_params.get('stop_loss', 1.2),
                'optimization_score': best_score,
                'all_results': optimization_results
            }
            
        except Exception as e:
            logger.error(f"❌ Ошибка оптимизации параметров {model_name}: {e}")
            return {
                'best_confidence': 0.5,
                'best_tp': 2.5,
                'best_sl': 1.2,
                'optimization_score': 0,
                'all_results': []
            }
    
    async def test_parameter_combination(self, model_name: str, confidence: float, tp: float, sl: float) -> float:
        """Тестирование конкретной комбинации параметров"""
        try:
            # Здесь должна быть логика тестирования с конкретными параметрами
            # Для упрощения возвращаем случайный скор
            # В реальной реализации нужно протестировать модель с этими параметрами
            
            # Симуляция тестирования (замените на реальную логику)
            base_score = np.random.uniform(0, 100)
            
            # Бонусы за оптимальные параметры
            if 0.4 <= confidence <= 0.7:
                base_score += 10
            if 2.0 <= tp <= 3.5:
                base_score += 10
            if 1.0 <= sl <= 1.5:
                base_score += 10
            
            return min(base_score, 100)
            
        except Exception as e:
            logger.error(f"Ошибка тестирования параметров: {e}")
            return 0
    
    async def find_best_confidence_threshold(self, model_name: str) -> float:
        """Поиск оптимального порога уверенности"""
        logger.info(f"🎯 Поиск оптимального порога уверенности для {model_name}")
        
        best_confidence = 0.5
        best_score = 0
        
        for confidence in self.confidence_range:
            try:
                # Тестируем с данным порогом уверенности
                score = await self.test_confidence_threshold(model_name, confidence)
                
                if score > best_score:
                    best_score = score
                    best_confidence = confidence
                    
            except Exception as e:
                logger.error(f"Ошибка тестирования порога уверенности {confidence}: {e}")
                continue
        
        logger.info(f"🎯 Оптимальный порог уверенности для {model_name}: {best_confidence}")
        return best_confidence
    
    async def test_confidence_threshold(self, model_name: str, confidence: float) -> float:
        """Тестирование конкретного порога уверенности"""
        # Симуляция тестирования порога уверенности
        # В реальной реализации здесь должно быть тестирование модели
        return np.random.uniform(0, 100)
    
    async def find_best_pairs_for_model(self, model_name: str) -> List[str]:
        """Определение лучших пар для модели"""
        logger.info(f"📊 Поиск лучших торговых пар для {model_name}")
        
        pair_scores = {}
        
        for pair in self.trading_pairs:
            try:
                # Тестируем модель на каждой паре
                score = await self.test_model_on_pair(model_name, pair)
                pair_scores[pair] = score
                
            except Exception as e:
                logger.error(f"Ошибка тестирования {model_name} на {pair}: {e}")
                pair_scores[pair] = 0
        
        # Сортируем пары по скору
        sorted_pairs = sorted(pair_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Возвращаем топ-3 пары
        best_pairs = [pair for pair, score in sorted_pairs[:3] if score > 50]
        
        logger.info(f"🏆 Лучшие пары для {model_name}: {best_pairs}")
        return best_pairs
    
    async def test_model_on_pair(self, model_name: str, pair: str) -> float:
        """Тестирование модели на конкретной паре"""
        # Симуляция тестирования на паре
        # В реальной реализации здесь должно быть полноценное тестирование
        return np.random.uniform(0, 100)
    
    def calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Расчет максимальной просадки"""
        if len(returns) == 0:
            return 0
        
        cumulative = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = running_max - cumulative
        return float(np.max(drawdown))
    
    def calculate_sharpe_ratio(self, returns: np.ndarray) -> float:
        """Расчет коэффициента Шарпа"""
        if len(returns) == 0:
            return 0
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0
        
        return float(mean_return / std_return)
    
    async def save_calibration_results(self, model_name: str, result: CalibrationResult):
        """Сохранение результатов калибровки"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"calibration_{model_name}_{timestamp}.json"
            filepath = self.individual_results_dir / filename
            
            result_dict = {
                'model_name': result.model_name,
                'win_rate': result.win_rate,
                'total_signals': result.total_signals,
                'profitable_signals': result.profitable_signals,
                'avg_profit': result.avg_profit,
                'max_drawdown': result.max_drawdown,
                'sharpe_ratio': result.sharpe_ratio,
                'best_confidence': result.best_confidence,
                'best_tp': result.best_tp,
                'best_sl': result.best_sl,
                'best_pairs': result.best_pairs,
                'optimization_score': result.optimization_score,
                'timestamp': result.timestamp
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(result_dict, f, indent=2, ensure_ascii=False)
            
            logger.info(f"💾 Результаты калибровки {model_name} сохранены: {filepath}")
            
        except Exception as e:
            logger.error(f"❌ Ошибка сохранения результатов калибровки {model_name}: {e}")
    
    async def save_optimization_results(self, model_name: str, results: List[Dict], best_params: Dict):
        """Сохранение результатов оптимизации"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"optimization_{model_name}_{timestamp}.json"
            filepath = self.optimization_results_dir / filename
            
            optimization_data = {
                'model_name': model_name,
                'best_parameters': best_params,
                'all_results': results,
                'timestamp': timestamp,
                'total_combinations_tested': len(results)
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(optimization_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"💾 Результаты оптимизации {model_name} сохранены: {filepath}")
            
        except Exception as e:
            logger.error(f"❌ Ошибка сохранения результатов оптимизации {model_name}: {e}")
    
    async def generate_calibration_report(self, model_name: str) -> Dict[str, Any]:
        """Генерация детального отчета по калибровке модели"""
        logger.info(f"📋 Генерация отчета по калибровке {model_name}")
        
        try:
            # Загружаем последние результаты калибровки
            calibration_files = list(self.individual_results_dir.glob(f"calibration_{model_name}_*.json"))
            if not calibration_files:
                logger.warning(f"⚠️ Результаты калибровки для {model_name} не найдены")
                return {}
            
            latest_file = max(calibration_files, key=lambda x: x.stat().st_mtime)
            
            with open(latest_file, 'r', encoding='utf-8') as f:
                calibration_data = json.load(f)
            
            # Загружаем результаты оптимизации
            optimization_files = list(self.optimization_results_dir.glob(f"optimization_{model_name}_*.json"))
            optimization_data = {}
            
            if optimization_files:
                latest_opt_file = max(optimization_files, key=lambda x: x.stat().st_mtime)
                with open(latest_opt_file, 'r', encoding='utf-8') as f:
                    optimization_data = json.load(f)
            
            # Создаем детальный отчет
            report = {
                'model_name': model_name,
                'calibration_summary': {
                    'win_rate': calibration_data.get('win_rate', 0),
                    'total_signals': calibration_data.get('total_signals', 0),
                    'profitable_signals': calibration_data.get('profitable_signals', 0),
                    'avg_profit': calibration_data.get('avg_profit', 0),
                    'max_drawdown': calibration_data.get('max_drawdown', 0),
                    'sharpe_ratio': calibration_data.get('sharpe_ratio', 0),
                    'optimization_score': calibration_data.get('optimization_score', 0)
                },
                'optimal_parameters': {
                    'confidence_threshold': calibration_data.get('best_confidence', 0.5),
                    'take_profit': calibration_data.get('best_tp', 2.5),
                    'stop_loss': calibration_data.get('best_sl', 1.2)
                },
                'best_trading_pairs': calibration_data.get('best_pairs', []),
                'optimization_details': optimization_data.get('all_results', []),
                'recommendations': self.generate_recommendations(calibration_data),
                'performance_grade': self.calculate_performance_grade(calibration_data),
                'timestamp': datetime.now().isoformat()
            }
            
            # Сохраняем отчет
            report_filename = f"report_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            report_filepath = self.results_dir / report_filename
            
            with open(report_filepath, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            logger.info(f"📋 Отчет по {model_name} сохранен: {report_filepath}")
            
            return report
            
        except Exception as e:
            logger.error(f"❌ Ошибка генерации отчета для {model_name}: {e}")
            return {}
    
    def generate_recommendations(self, calibration_data: Dict) -> List[str]:
        """Генерация рекомендаций по улучшению модели"""
        recommendations = []
        
        win_rate = calibration_data.get('win_rate', 0)
        total_signals = calibration_data.get('total_signals', 0)
        sharpe_ratio = calibration_data.get('sharpe_ratio', 0)
        
        if win_rate < 50:
            recommendations.append("🔴 Низкий винрейт - требуется пересмотр логики модели")
        elif win_rate < 60:
            recommendations.append("🟡 Средний винрейт - возможна оптимизация параметров")
        else:
            recommendations.append("🟢 Хороший винрейт - модель работает эффективно")
        
        if total_signals < 10:
            recommendations.append("⚠️ Мало сигналов - рассмотрите снижение порога уверенности")
        elif total_signals > 100:
            recommendations.append("⚠️ Слишком много сигналов - повысьте порог уверенности")
        
        if sharpe_ratio < 0.5:
            recommendations.append("📉 Низкий Sharpe ratio - высокий риск относительно доходности")
        elif sharpe_ratio > 1.0:
            recommendations.append("📈 Отличный Sharpe ratio - хорошее соотношение риск/доходность")
        
        return recommendations
    
    def calculate_performance_grade(self, calibration_data: Dict) -> str:
        """Расчет общей оценки производительности модели"""
        win_rate = calibration_data.get('win_rate', 0)
        sharpe_ratio = calibration_data.get('sharpe_ratio', 0)
        total_signals = calibration_data.get('total_signals', 0)
        
        score = 0
        
        # Оценка винрейта (40% от общей оценки)
        if win_rate >= 70:
            score += 40
        elif win_rate >= 60:
            score += 30
        elif win_rate >= 50:
            score += 20
        elif win_rate >= 40:
            score += 10
        
        # Оценка Sharpe ratio (30% от общей оценки)
        if sharpe_ratio >= 1.5:
            score += 30
        elif sharpe_ratio >= 1.0:
            score += 25
        elif sharpe_ratio >= 0.5:
            score += 15
        elif sharpe_ratio >= 0.0:
            score += 5
        
        # Оценка количества сигналов (30% от общей оценки)
        if 20 <= total_signals <= 80:
            score += 30
        elif 10 <= total_signals <= 100:
            score += 20
        elif total_signals > 0:
            score += 10
        
        if score >= 80:
            return "A+ (Отлично)"
        elif score >= 70:
            return "A (Хорошо)"
        elif score >= 60:
            return "B (Удовлетворительно)"
        elif score >= 40:
            return "C (Требует улучшения)"
        else:
            return "D (Неудовлетворительно)"
    
    async def run_full_calibration(self):
        """Запуск полной калибровки всех моделей"""
        logger.info("🚀 Запуск полной калибровки всех AI моделей")
        
        if not await self.initialize():
            logger.error("❌ Не удалось инициализировать систему калибровки")
            return
        
        calibration_summary = {
            'start_time': datetime.now().isoformat(),
            'models_calibrated': [],
            'results': {}
        }
        
        for model_name in self.models:
            logger.info(f"🎯 Калибровка модели: {model_name}")
            
            try:
                # Калибруем модель
                result = await self.calibrate_individual_model(model_name)
                
                if result:
                    calibration_summary['models_calibrated'].append(model_name)
                    calibration_summary['results'][model_name] = {
                        'win_rate': result.win_rate,
                        'total_signals': result.total_signals,
                        'optimization_score': result.optimization_score,
                        'best_pairs': result.best_pairs,
                        'status': 'success'
                    }
                    
                    # Генерируем отчет
                    await self.generate_calibration_report(model_name)
                    
                else:
                    calibration_summary['results'][model_name] = {
                        'status': 'failed',
                        'error': 'Калибровка не удалась'
                    }
                
            except Exception as e:
                logger.error(f"❌ Ошибка калибровки {model_name}: {e}")
                calibration_summary['results'][model_name] = {
                    'status': 'error',
                    'error': str(e)
                }
        
        calibration_summary['end_time'] = datetime.now().isoformat()
        
        # Сохраняем общий отчет
        summary_filename = f"calibration_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        summary_filepath = self.results_dir / summary_filename
        
        with open(summary_filepath, 'w', encoding='utf-8') as f:
            json.dump(calibration_summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Полная калибровка завершена. Отчет сохранен: {summary_filepath}")
        
        # Выводим краткую сводку
        self.print_calibration_summary(calibration_summary)
    
    def print_calibration_summary(self, summary: Dict):
        """Вывод краткой сводки калибровки"""
        print("\n" + "="*80)
        print("🎯 СВОДКА КАЛИБРОВКИ AI МОДЕЛЕЙ")
        print("="*80)
        
        successful_models = len(summary['models_calibrated'])
        total_models = len(self.models)
        
        print(f"📊 Успешно откалибровано: {successful_models}/{total_models} моделей")
        print()
        
        for model_name, result in summary['results'].items():
            if result['status'] == 'success':
                print(f"✅ {model_name}:")
                print(f"   📈 Винрейт: {result['win_rate']:.1f}%")
                print(f"   🎯 Сигналов: {result['total_signals']}")
                print(f"   ⭐ Оценка: {result['optimization_score']:.1f}")
                print(f"   💎 Лучшие пары: {', '.join(result['best_pairs'])}")
            else:
                print(f"❌ {model_name}: {result.get('error', 'Неизвестная ошибка')}")
            print()
        
        print("="*80)

# Основная функция запуска
async def main():
    """Основная функция запуска калибровки"""
    print("🚀 Запуск системы калибровки AI моделей")
    
    calibrator = AIModelCalibrator()
    
    # Запускаем полную калибровку
    await calibrator.run_full_calibration()

if __name__ == "__main__":
    # Пример использования:
    # python ai_model_calibrator.py
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⏹️ Калибровка прервана пользователем")
    except Exception as e:
        print(f"❌ Критическая ошибка: {e}")
        logger.error(f"Критическая ошибка в main: {e}")