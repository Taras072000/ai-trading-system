#!/usr/bin/env python3
"""
Система автоматической оптимизации параметров для AI моделей
"""

import asyncio
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
import itertools
import json
from dataclasses import dataclass, asdict

# Импорты AI модулей
from ai_modules.lava_ai import LavaAI
from ai_modules.lgbm_ai import LGBMAI
from ai_modules.mistral_ai import MistralAI

# Импорты системных модулей
from data_collector import DataManager
from utils.timezone_utils import get_utc_now

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class OptimizationConfig:
    """Конфигурация для оптимизации параметров"""
    model_name: str
    symbol: str = "BTCUSDT"
    test_days: int = 7
    confidence_range: List[float] = None
    tp_range: List[float] = None
    sl_range: List[float] = None
    max_combinations: int = 50
    
    def __post_init__(self):
        if self.confidence_range is None:
            self.confidence_range = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        if self.tp_range is None:
            self.tp_range = [1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
        if self.sl_range is None:
            self.sl_range = [0.8, 1.0, 1.2, 1.5, 2.0]

@dataclass
class OptimizationResult:
    """Результат оптимизации"""
    model_name: str
    best_params: Dict[str, float]
    best_score: float
    win_rate: float
    total_trades: int
    profit_factor: float
    max_drawdown: float
    sharpe_ratio: float
    all_results: List[Dict]

class ParameterOptimizer:
    """Система автоматической оптимизации параметров AI моделей"""
    
    def __init__(self):
        self.data_manager = DataManager()
        self.models = {}
        self.optimization_results = {}
        
    async def initialize(self) -> bool:
        """Инициализация оптимизатора"""
        try:
            logger.info("🔄 Инициализация Parameter Optimizer...")
            
            # Инициализация моделей
            await self._initialize_models()
            
            logger.info("✅ Parameter Optimizer инициализирован")
            return True
            
        except Exception as e:
            logger.error(f"❌ Ошибка инициализации: {e}")
            return False
    
    async def _initialize_models(self):
        """Инициализация AI моделей"""
        try:
            # Инициализация LavaAI
            self.models['lava_ai'] = LavaAI()
            await self.models['lava_ai'].initialize()
            logger.info("✅ lava_ai инициализирована")
            
            # Инициализация LGBMAI
            self.models['lgbm_ai'] = LGBMAI()
            await self.models['lgbm_ai'].initialize()
            logger.info("✅ lgbm_ai инициализирована")
            
            # Инициализация MistralAI
            self.models['mistral_ai'] = MistralAI()
            await self.models['mistral_ai'].initialize()
            logger.info("✅ mistral_ai инициализирована")
            
        except Exception as e:
            logger.error(f"❌ Ошибка инициализации моделей: {e}")
            raise
    
    async def optimize_model(self, config: OptimizationConfig) -> OptimizationResult:
        """Оптимизация параметров для конкретной модели"""
        logger.info(f"🎯 Начинаем оптимизацию {config.model_name}...")
        
        if config.model_name not in self.models:
            raise ValueError(f"Модель {config.model_name} не найдена")
        
        # Получение исторических данных
        data = await self._get_historical_data(config.symbol, config.test_days)
        if data is None or len(data) < 100:
            raise ValueError(f"Недостаточно данных для {config.symbol}")
        
        # Генерация комбинаций параметров
        param_combinations = self._generate_param_combinations(config)
        logger.info(f"📊 Тестируем {len(param_combinations)} комбинаций параметров")
        
        # Тестирование всех комбинаций
        results = []
        for i, params in enumerate(param_combinations):
            if i % 10 == 0:
                logger.info(f"⏳ Прогресс: {i}/{len(param_combinations)} ({i/len(param_combinations)*100:.1f}%)")
            
            result = await self._test_parameter_combination(
                config.model_name, data, params
            )
            results.append(result)
        
        # Поиск лучших параметров
        best_result = self._find_best_parameters(results)
        
        # Создание результата оптимизации
        optimization_result = OptimizationResult(
            model_name=config.model_name,
            best_params=best_result['params'],
            best_score=best_result['score'],
            win_rate=best_result['win_rate'],
            total_trades=best_result['total_trades'],
            profit_factor=best_result['profit_factor'],
            max_drawdown=best_result['max_drawdown'],
            sharpe_ratio=best_result['sharpe_ratio'],
            all_results=results
        )
        
        # Сохранение результатов
        self.optimization_results[config.model_name] = optimization_result
        await self._save_optimization_results(optimization_result)
        
        logger.info(f"✅ Оптимизация {config.model_name} завершена!")
        logger.info(f"🏆 Лучшие параметры: {best_result['params']}")
        logger.info(f"📈 Винрейт: {best_result['win_rate']:.2%}")
        logger.info(f"💰 Profit Factor: {best_result['profit_factor']:.2f}")
        
        return optimization_result
    
    async def _get_historical_data(self, symbol: str, days: int) -> Optional[pd.DataFrame]:
        """Получение исторических данных"""
        try:
            logger.info(f"📊 Загружаем данные для {symbol} за {days} дней...")
            
            # Получение данных через DataManager
            data = await self.data_manager.ensure_data_available(
                symbol=symbol,
                interval='1h',
                days=days
            )
            
            if data is not None and len(data) > 0:
                # Убеждаемся что индекс - datetime
                if 'timestamp' in data.columns:
                    data['timestamp'] = pd.to_datetime(data['timestamp'])
                    data.set_index('timestamp', inplace=True)
                elif not isinstance(data.index, pd.DatetimeIndex):
                    logger.warning("⚠️ Индекс данных не является datetime")
                
                logger.info(f"✅ Загружено {len(data)} свечей для {symbol}")
                return data
            else:
                logger.error(f"❌ Не удалось загрузить данные для {symbol}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки данных: {e}")
            return None
    
    def _generate_param_combinations(self, config: OptimizationConfig) -> List[Dict]:
        """Генерация комбинаций параметров для тестирования"""
        combinations = []
        
        # Создаем все возможные комбинации
        for confidence in config.confidence_range:
            for tp in config.tp_range:
                for sl in config.sl_range:
                    combinations.append({
                        'confidence_threshold': confidence,
                        'take_profit': tp,
                        'stop_loss': sl
                    })
        
        # Ограничиваем количество комбинаций
        if len(combinations) > config.max_combinations:
            # Выбираем равномерно распределенные комбинации
            step = len(combinations) // config.max_combinations
            combinations = combinations[::step][:config.max_combinations]
        
        return combinations
    
    async def _test_parameter_combination(self, model_name: str, data: pd.DataFrame, params: Dict) -> Dict:
        """Тестирование одной комбинации параметров"""
        try:
            model = self.models[model_name]
            
            # Симуляция торговли с данными параметрами
            trades = []
            balance = 1000.0  # Начальный баланс
            position = None
            
            for i in range(100, len(data)):  # Начинаем с 100 свечи для индикаторов
                current_data = data.iloc[:i+1]
                current_price = data.iloc[i]['close']
                
                # Получение сигнала от модели
                signal = await self._get_model_signal(model_name, model, current_data)
                
                if signal is None:
                    continue
                
                # Проверка уверенности
                confidence = self._extract_confidence(signal)
                if confidence < params['confidence_threshold']:
                    continue
                
                # Логика торговли
                action = self._extract_action(signal)
                
                if position is None and action in ['BUY', 'SELL']:
                    # Открытие позиции
                    position = {
                        'action': action,
                        'entry_price': current_price,
                        'entry_time': data.index[i],
                        'tp_price': current_price * (1 + params['take_profit']/100) if action == 'BUY' 
                                   else current_price * (1 - params['take_profit']/100),
                        'sl_price': current_price * (1 - params['stop_loss']/100) if action == 'BUY' 
                                   else current_price * (1 + params['stop_loss']/100)
                    }
                
                elif position is not None:
                    # Проверка закрытия позиции
                    should_close = False
                    exit_reason = ""
                    
                    if position['action'] == 'BUY':
                        if current_price >= position['tp_price']:
                            should_close = True
                            exit_reason = "TP"
                        elif current_price <= position['sl_price']:
                            should_close = True
                            exit_reason = "SL"
                    else:  # SELL
                        if current_price <= position['tp_price']:
                            should_close = True
                            exit_reason = "TP"
                        elif current_price >= position['sl_price']:
                            should_close = True
                            exit_reason = "SL"
                    
                    if should_close:
                        # Закрытие позиции
                        if position['action'] == 'BUY':
                            pnl = (current_price - position['entry_price']) / position['entry_price'] * 100
                        else:
                            pnl = (position['entry_price'] - current_price) / position['entry_price'] * 100
                        
                        trades.append({
                            'entry_time': position['entry_time'],
                            'exit_time': data.index[i],
                            'action': position['action'],
                            'entry_price': position['entry_price'],
                            'exit_price': current_price,
                            'pnl': pnl,
                            'exit_reason': exit_reason
                        })
                        
                        balance *= (1 + pnl/100)
                        position = None
            
            # Расчет метрик
            metrics = self._calculate_metrics(trades, balance)
            metrics['params'] = params
            
            return metrics
            
        except Exception as e:
            logger.error(f"❌ Ошибка тестирования параметров: {e}")
            return {
                'params': params,
                'score': 0,
                'win_rate': 0,
                'total_trades': 0,
                'profit_factor': 0,
                'max_drawdown': 100,
                'sharpe_ratio': 0
            }
    
    async def _get_model_signal(self, model_name: str, model: Any, data: pd.DataFrame) -> Optional[Dict]:
        """Получение сигнала от модели"""
        try:
            if model_name == 'lava_ai':
                if hasattr(model, 'generate_trading_signals'):
                    return await model.generate_trading_signals(data)
                elif hasattr(model, 'analyze_market_data'):
                    return await model.analyze_market_data(data)
            
            elif model_name == 'lgbm_ai':
                if hasattr(model, 'get_signal'):
                    return await model.get_signal(data)
                elif hasattr(model, 'predict'):
                    return await model.predict(data)
            
            elif model_name == 'mistral_ai':
                if hasattr(model, 'get_signal'):
                    return await model.get_signal(data)
                elif hasattr(model, 'analyze'):
                    return await model.analyze(data)
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Ошибка получения сигнала от {model_name}: {e}")
            return None
    
    def _extract_confidence(self, signal: Any) -> float:
        """Извлечение уверенности из сигнала"""
        if isinstance(signal, dict):
            return signal.get('confidence', 0.0)
        return 0.0
    
    def _extract_action(self, signal: Any) -> str:
        """Извлечение действия из сигнала"""
        if isinstance(signal, dict):
            if 'signal' in signal:
                return signal['signal']
            elif 'action' in signal:
                return signal['action']
            elif 'direction' in signal:
                direction = signal['direction']
                if direction > 0.5:
                    return 'BUY'
                elif direction < -0.5:
                    return 'SELL'
        return 'HOLD'
    
    def _calculate_metrics(self, trades: List[Dict], final_balance: float) -> Dict:
        """Расчет торговых метрик"""
        if not trades:
            return {
                'score': 0,
                'win_rate': 0,
                'total_trades': 0,
                'profit_factor': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0
            }
        
        # Базовые метрики
        total_trades = len(trades)
        winning_trades = [t for t in trades if t['pnl'] > 0]
        losing_trades = [t for t in trades if t['pnl'] < 0]
        
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        
        # Profit Factor
        gross_profit = sum(t['pnl'] for t in winning_trades)
        gross_loss = abs(sum(t['pnl'] for t in losing_trades))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Максимальная просадка
        balance_curve = [1000.0]
        for trade in trades:
            balance_curve.append(balance_curve[-1] * (1 + trade['pnl']/100))
        
        peak = balance_curve[0]
        max_drawdown = 0
        for balance in balance_curve:
            if balance > peak:
                peak = balance
            drawdown = (peak - balance) / peak * 100
            max_drawdown = max(max_drawdown, drawdown)
        
        # Sharpe Ratio (упрощенный)
        returns = [t['pnl'] for t in trades]
        if len(returns) > 1:
            sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
        else:
            sharpe_ratio = 0
        
        # Общий скор (комбинированная метрика)
        score = (win_rate * 0.3 + 
                min(profit_factor/2, 1) * 0.3 + 
                max(0, (100-max_drawdown)/100) * 0.2 + 
                max(0, sharpe_ratio/10) * 0.2)
        
        return {
            'score': score,
            'win_rate': win_rate,
            'total_trades': total_trades,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio
        }
    
    def _find_best_parameters(self, results: List[Dict]) -> Dict:
        """Поиск лучших параметров по скору"""
        if not results:
            raise ValueError("Нет результатов для анализа")
        
        # Сортировка по скору
        sorted_results = sorted(results, key=lambda x: x['score'], reverse=True)
        return sorted_results[0]
    
    async def _save_optimization_results(self, result: OptimizationResult):
        """Сохранение результатов оптимизации"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"optimization_{result.model_name}_{timestamp}.json"
            
            # Конвертация в JSON-совместимый формат
            result_dict = asdict(result)
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(result_dict, f, ensure_ascii=False, indent=2, default=str)
            
            logger.info(f"💾 Результаты сохранены в {filename}")
            
        except Exception as e:
            logger.error(f"❌ Ошибка сохранения результатов: {e}")
    
    async def optimize_all_models(self) -> Dict[str, OptimizationResult]:
        """Оптимизация всех доступных моделей"""
        logger.info("🚀 Начинаем оптимизацию всех моделей...")
        
        results = {}
        
        for model_name in self.models.keys():
            try:
                logger.info(f"\n{'='*50}")
                logger.info(f"🎯 Оптимизация {model_name}")
                logger.info(f"{'='*50}")
                
                config = OptimizationConfig(model_name=model_name)
                result = await self.optimize_model(config)
                results[model_name] = result
                
            except Exception as e:
                logger.error(f"❌ Ошибка оптимизации {model_name}: {e}")
        
        logger.info("\n🏁 Оптимизация всех моделей завершена!")
        return results
    
    def print_summary(self):
        """Вывод сводки по всем оптимизированным моделям"""
        if not self.optimization_results:
            logger.info("📊 Нет результатов оптимизации")
            return
        
        logger.info("\n" + "="*80)
        logger.info("📊 СВОДКА РЕЗУЛЬТАТОВ ОПТИМИЗАЦИИ")
        logger.info("="*80)
        
        for model_name, result in self.optimization_results.items():
            logger.info(f"\n🤖 {model_name.upper()}:")
            logger.info(f"   📈 Винрейт: {result.win_rate:.2%}")
            logger.info(f"   💰 Profit Factor: {result.profit_factor:.2f}")
            logger.info(f"   📉 Max Drawdown: {result.max_drawdown:.2f}%")
            logger.info(f"   📊 Sharpe Ratio: {result.sharpe_ratio:.2f}")
            logger.info(f"   🎯 Лучшие параметры: {result.best_params}")

async def main():
    """Главная функция"""
    optimizer = ParameterOptimizer()
    
    # Инициализация
    if not await optimizer.initialize():
        logger.error("❌ Не удалось инициализировать оптимизатор")
        return
    
    # Оптимизация всех моделей
    results = await optimizer.optimize_all_models()
    
    # Вывод сводки
    optimizer.print_summary()

if __name__ == "__main__":
    asyncio.run(main())