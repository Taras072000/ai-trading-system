#!/usr/bin/env python3
"""
🎯 Individual Model Tester - Система индивидуального тестирования каждой AI модели

Цель: Протестировать каждую модель отдельно с детальным анализом для выявления проблем
"""

import asyncio
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import json
import os

# AI модули
from ai_modules.ai_manager import AIManager
from ai_modules.lava_ai import LavaAI
from ai_modules.lgbm_ai import LGBMAI
from ai_modules.mistral_ai import MistralAI
from ai_modules.trading_ai import TradingAI
from ai_modules.reinforcement_learning_engine import ReinforcementLearningEngine

# Системные модули
from data_collector import DataManager
from utils.timezone_utils import get_utc_now

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class IndividualModelTester:
    """Класс для индивидуального тестирования AI моделей"""
    
    def __init__(self):
        self.ai_manager = AIManager()
        self.data_manager = DataManager()
        self.models = {}
        self.test_symbols = ["BTCUSDT", "ETHUSDT"]
        
    async def initialize(self) -> bool:
        """Инициализация всех компонентов"""
        try:
            logger.info("🚀 Инициализация Individual Model Tester...")
            
            # Инициализация AI Manager
            await self.ai_manager.initialize()
            
            # Инициализация отдельных моделей
            await self._initialize_individual_models()
            
            logger.info("✅ Individual Model Tester инициализирован")
            return True
            
        except Exception as e:
            logger.error(f"❌ Ошибка инициализации: {e}")
            return False
    
    async def _initialize_individual_models(self):
        """Инициализация отдельных моделей"""
        
        # LavaAI
        try:
            self.models['lava_ai'] = LavaAI()
            await self.models['lava_ai'].initialize()
            logger.info("✅ lava_ai инициализирована")
        except Exception as e:
            logger.error(f"❌ Ошибка инициализации lava_ai: {e}")
        
        # LGBMAI
        try:
            self.models['lgbm_ai'] = LGBMAI()
            await self.models['lgbm_ai'].initialize()
            logger.info("✅ lgbm_ai инициализирована")
        except Exception as e:
            logger.error(f"❌ Ошибка инициализации lgbm_ai: {e}")
        
        # MistralAI
        try:
            self.models['mistral_ai'] = MistralAI()
            await self.models['mistral_ai'].initialize()
            logger.info("✅ mistral_ai инициализирована")
        except Exception as e:
            logger.error(f"❌ Ошибка инициализации mistral_ai: {e}")
    
    async def test_model_individually(self, model_name: str, symbol: str = "BTCUSDT", days: int = 7) -> Dict:
        """Тестирование одной модели индивидуально"""
        
        logger.info(f"\n🔍 === ИНДИВИДУАЛЬНОЕ ТЕСТИРОВАНИЕ {model_name.upper()} ===")
        logger.info(f"📊 Символ: {symbol}")
        logger.info(f"📅 Период: {days} дней")
        
        if model_name not in self.models:
            logger.error(f"❌ Модель {model_name} не найдена")
            return {"error": f"Модель {model_name} не найдена"}
        
        model = self.models[model_name]
        
        try:
            # Загрузка данных
            logger.info(f"📈 Загружаем данные для {symbol}...")
            data = await self._get_historical_data(symbol, days)
            
            if data is None or len(data) < 10:
                logger.warning(f"⚠️ Недостаточно данных для {symbol}")
                return {"error": f"Недостаточно данных для {symbol}"}
            
            logger.info(f"✅ Загружено {len(data)} свечей")
            
            # Тестирование различных методов модели
            results = {
                "model_name": model_name,
                "symbol": symbol,
                "data_points": len(data),
                "test_timestamp": get_utc_now().isoformat(),
                "methods_tested": {},
                "signals_generated": [],
                "errors": []
            }
            
            # Тестируем разные методы в зависимости от модели
            await self._test_model_methods(model, model_name, data, results)
            
            # Анализ результатов
            self._analyze_model_performance(results)
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Ошибка тестирования {model_name}: {e}")
            return {"error": str(e)}
    
    async def _test_model_methods(self, model: Any, model_name: str, data: pd.DataFrame, results: Dict):
        """Тестирование различных методов модели"""
        
        # Список методов для тестирования в зависимости от модели
        methods_to_test = {
            'lava_ai': ['generate_trading_signals', 'analyze_market_data'],
            'lgbm_ai': ['predict', 'get_prediction'],
            'mistral_ai': ['get_trading_signal', 'analyze_market']
        }
        
        model_methods = methods_to_test.get(model_name, [])
        
        for method_name in model_methods:
            logger.info(f"🔧 Тестируем метод: {method_name}")
            
            try:
                if hasattr(model, method_name):
                    method = getattr(model, method_name)
                    
                    # Пробуем вызвать метод с разными параметрами
                    signal = await self._call_model_method(method, data, model_name)
                    
                    if signal:
                        results["methods_tested"][method_name] = "success"
                        results["signals_generated"].append({
                            "method": method_name,
                            "signal": signal,
                            "timestamp": get_utc_now().isoformat()
                        })
                        logger.info(f"✅ {method_name}: {signal}")
                    else:
                        results["methods_tested"][method_name] = "no_signal"
                        logger.warning(f"⚠️ {method_name}: Нет сигнала")
                        
                else:
                    results["methods_tested"][method_name] = "method_not_found"
                    logger.warning(f"⚠️ Метод {method_name} не найден в {model_name}")
                    
            except Exception as e:
                results["methods_tested"][method_name] = f"error: {str(e)}"
                results["errors"].append(f"{method_name}: {str(e)}")
                logger.error(f"❌ Ошибка в {method_name}: {e}")
    
    async def _call_model_method(self, method: Any, data: pd.DataFrame, model_name: str) -> Any:
        """Вызов метода модели с правильными параметрами"""
        
        try:
            # Для разных моделей используем разные параметры
            if model_name == 'lava_ai':
                if method.__name__ == 'generate_trading_signals':
                    return await method(data)
                elif method.__name__ == 'analyze_market_data':
                    return await method(data)
                    
            elif model_name == 'lgbm_ai':
                if method.__name__ in ['predict', 'get_prediction']:
                    return await method(data)
                    
            elif model_name == 'mistral_ai':
                if method.__name__ in ['get_trading_signal', 'analyze_market']:
                    return await method(data)
            
            # Если не подошло ни одно условие, пробуем просто вызвать
            return await method(data)
            
        except Exception as e:
            logger.error(f"Ошибка вызова метода {method.__name__}: {e}")
            return None
    
    def _analyze_model_performance(self, results: Dict):
        """Анализ производительности модели"""
        
        total_methods = len(results["methods_tested"])
        successful_methods = len([m for m in results["methods_tested"].values() if m == "success"])
        signals_count = len(results["signals_generated"])
        errors_count = len(results["errors"])
        
        results["performance_analysis"] = {
            "total_methods_tested": total_methods,
            "successful_methods": successful_methods,
            "success_rate": (successful_methods / total_methods * 100) if total_methods > 0 else 0,
            "signals_generated": signals_count,
            "errors_count": errors_count,
            "status": "working" if successful_methods > 0 else "not_working"
        }
        
        # Рекомендации
        recommendations = []
        if successful_methods == 0:
            recommendations.append("❌ Модель не работает - все методы завершились ошибкой")
        elif successful_methods < total_methods:
            recommendations.append("⚠️ Модель работает частично - некоторые методы не работают")
        else:
            recommendations.append("✅ Модель работает корректно")
            
        if signals_count == 0:
            recommendations.append("⚠️ Модель не генерирует сигналы")
        elif signals_count > 0:
            recommendations.append(f"📊 Модель сгенерировала {signals_count} сигналов")
            
        results["recommendations"] = recommendations
    
    async def _get_historical_data(self, symbol: str, days: int) -> Optional[pd.DataFrame]:
        """Получение исторических данных"""
        try:
            # ensure_data_available уже возвращает DataFrame
            data = await self.data_manager.ensure_data_available(symbol, "1h", days)
            
            if data is not None and len(data) > 0:
                # Устанавливаем timestamp как индекс если это еще не сделано
                if 'timestamp' in data.columns and data.index.name != 'timestamp':
                    data = data.set_index('timestamp')
                
                # Берем последние данные за указанный период
                if len(data) > 0:
                    return data.tail(days * 24)  # Примерно days * 24 часа
                
            return None
            
        except Exception as e:
            logger.error(f"Ошибка загрузки данных для {symbol}: {e}")
            return None
    
    async def test_all_models_individually(self) -> Dict:
        """Тестирование всех моделей индивидуально"""
        
        logger.info("\n🎯 === ЗАПУСК ИНДИВИДУАЛЬНОГО ТЕСТИРОВАНИЯ ВСЕХ МОДЕЛЕЙ ===")
        
        all_results = {}
        
        for model_name in self.models.keys():
            logger.info(f"\n{'='*60}")
            logger.info(f"🔍 ТЕСТИРОВАНИЕ МОДЕЛИ: {model_name.upper()}")
            logger.info(f"{'='*60}")
            
            model_results = {}
            
            # Тестируем на разных символах
            for symbol in self.test_symbols:
                result = await self.test_model_individually(model_name, symbol, days=3)
                model_results[symbol] = result
                
                # Короткая пауза между тестами
                await asyncio.sleep(1)
            
            all_results[model_name] = model_results
            
            # Анализ общих результатов модели
            self._analyze_overall_model_performance(model_name, model_results)
        
        # Сохранение результатов
        await self._save_test_results(all_results)
        
        return all_results
    
    def _analyze_overall_model_performance(self, model_name: str, model_results: Dict):
        """Анализ общей производительности модели"""
        
        logger.info(f"\n📊 === АНАЛИЗ МОДЕЛИ {model_name.upper()} ===")
        
        working_symbols = 0
        total_signals = 0
        total_errors = 0
        
        for symbol, result in model_results.items():
            if "performance_analysis" in result:
                analysis = result["performance_analysis"]
                if analysis["status"] == "working":
                    working_symbols += 1
                total_signals += analysis["signals_generated"]
                total_errors += analysis["errors_count"]
        
        total_symbols = len(model_results)
        success_rate = (working_symbols / total_symbols * 100) if total_symbols > 0 else 0
        
        logger.info(f"✅ Работающие пары: {working_symbols}/{total_symbols} ({success_rate:.1f}%)")
        logger.info(f"📊 Всего сигналов: {total_signals}")
        logger.info(f"❌ Всего ошибок: {total_errors}")
        
        if success_rate == 0:
            logger.error(f"❌ МОДЕЛЬ {model_name.upper()} НЕ РАБОТАЕТ!")
        elif success_rate < 50:
            logger.warning(f"⚠️ МОДЕЛЬ {model_name.upper()} РАБОТАЕТ ПЛОХО")
        else:
            logger.info(f"✅ МОДЕЛЬ {model_name.upper()} РАБОТАЕТ ХОРОШО")
    
    async def _save_test_results(self, results: Dict):
        """Сохранение результатов тестирования"""
        
        # Создаем папку для результатов
        os.makedirs("individual_test_results", exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"individual_test_results/individual_test_{timestamp}.json"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"💾 Результаты сохранены: {filename}")
            
        except Exception as e:
            logger.error(f"❌ Ошибка сохранения результатов: {e}")

async def main():
    """Основная функция"""
    
    tester = IndividualModelTester()
    
    if not await tester.initialize():
        logger.error("❌ Ошибка инициализации")
        return
    
    # Запуск индивидуального тестирования всех моделей
    results = await tester.test_all_models_individually()
    
    logger.info("\n🎉 === ТЕСТИРОВАНИЕ ЗАВЕРШЕНО ===")

if __name__ == "__main__":
    asyncio.run(main())