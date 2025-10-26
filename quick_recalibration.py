#!/usr/bin/env python3
"""
🚀 БЫСТРАЯ ПОВТОРНАЯ КАЛИБРОВКА AI МОДЕЛЕЙ
Скрипт для диагностики и повторной калибровки с улучшенными параметрами
"""

import asyncio
import logging
import json
import os
from datetime import datetime
from ai_model_calibrator import AIModelCalibrator
from data_collector import DataManager

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('quick_recalibration.log'),
        logging.StreamHandler()
    ]
)

class QuickRecalibrator:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.calibrator = None
        self.data_manager = None
        
        # Улучшенные параметры для повторной калибровки
        self.improved_params = {
            'trading_ai': {
                'confidence_range': [0.1, 0.2, 0.3, 0.4],
                'tp_range': [2.0, 2.5, 3.0, 3.5],
                'sl_range': [1.0, 1.2, 1.5, 1.8],
                'test_days': 14
            },
            'lava_ai': {
                'confidence_range': [0.2, 0.3, 0.4, 0.5],
                'tp_range': [1.5, 2.0, 2.5, 3.0],
                'sl_range': [1.0, 1.2, 1.5, 2.0],
                'test_days': 14
            },
            'lgbm_ai': {
                'confidence_range': [0.2, 0.3, 0.4, 0.5],
                'tp_range': [2.0, 2.5, 3.0, 3.5],
                'sl_range': [1.0, 1.2, 1.5, 1.8],
                'test_days': 14
            },
            'mistral_ai': {
                'confidence_range': [0.2, 0.3, 0.4, 0.5],
                'tp_range': [2.5, 3.0, 3.5, 4.0],
                'sl_range': [1.2, 1.5, 1.8, 2.0],
                'test_days': 14
            }
        }
    
    async def run_diagnostics(self):
        """Запуск диагностики системы"""
        self.logger.info("🔍 Запуск диагностики системы...")
        
        try:
            # Проверка DataManager
            self.logger.info("📊 Проверка DataManager...")
            self.data_manager = DataManager()
            
            # Проверка доступности данных
            test_pairs = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'ADAUSDT']
            available_data = {}
            
            for pair in test_pairs:
                try:
                    # Попытка получить данные за последние 7 дней
                    data = await self.data_manager.get_historical_data(pair, '1h', 7)
                    if data is not None and len(data) > 0:
                        available_data[pair] = len(data)
                        self.logger.info(f"✅ {pair}: {len(data)} записей")
                    else:
                        self.logger.warning(f"⚠️ {pair}: Нет данных")
                        available_data[pair] = 0
                except Exception as e:
                    self.logger.error(f"❌ {pair}: Ошибка - {e}")
                    available_data[pair] = 0
            
            # Сохранение результатов диагностики
            diagnostic_report = {
                'timestamp': datetime.now().isoformat(),
                'data_availability': available_data,
                'total_pairs_with_data': sum(1 for count in available_data.values() if count > 0),
                'recommendations': self._generate_recommendations(available_data)
            }
            
            with open('diagnostic_report.json', 'w', encoding='utf-8') as f:
                json.dump(diagnostic_report, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"📋 Диагностика завершена. Пар с данными: {diagnostic_report['total_pairs_with_data']}/5")
            return diagnostic_report
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка диагностики: {e}")
            return None
    
    def _generate_recommendations(self, data_availability):
        """Генерация рекомендаций на основе диагностики"""
        recommendations = []
        
        pairs_with_data = [pair for pair, count in data_availability.items() if count > 0]
        pairs_without_data = [pair for pair, count in data_availability.items() if count == 0]
        
        if len(pairs_with_data) == 0:
            recommendations.append("КРИТИЧНО: Нет доступных данных для торговых пар")
            recommendations.append("Проверить подключение к Binance API")
            recommendations.append("Запустить сбор данных: python data_collector.py")
        elif len(pairs_with_data) < 3:
            recommendations.append("ВНИМАНИЕ: Мало доступных данных")
            recommendations.append(f"Доступны пары: {', '.join(pairs_with_data)}")
            recommendations.append("Рекомендуется собрать больше данных")
        else:
            recommendations.append("ХОРОШО: Достаточно данных для калибровки")
            recommendations.append(f"Доступны пары: {', '.join(pairs_with_data)}")
        
        if pairs_without_data:
            recommendations.append(f"Отсутствуют данные для: {', '.join(pairs_without_data)}")
        
        return recommendations
    
    async def run_improved_calibration(self, models_to_calibrate=None):
        """Запуск улучшенной калибровки"""
        self.logger.info("🚀 Запуск улучшенной калибровки...")
        
        try:
            self.calibrator = AIModelCalibrator()
            
            if models_to_calibrate is None:
                models_to_calibrate = ['trading_ai', 'lava_ai', 'lgbm_ai', 'mistral_ai']
            
            results = {}
            
            for model_name in models_to_calibrate:
                self.logger.info(f"🔧 Калибровка {model_name} с улучшенными параметрами...")
                
                try:
                    # Получение улучшенных параметров для модели
                    params = self.improved_params.get(model_name, {})
                    
                    # Запуск калибровки с новыми параметрами
                    result = await self.calibrator.calibrate_individual_model(
                        model_name, 
                        **params
                    )
                    
                    results[model_name] = result
                    self.logger.info(f"✅ {model_name} откалиброван")
                    
                except Exception as e:
                    self.logger.error(f"❌ Ошибка калибровки {model_name}: {e}")
                    results[model_name] = {'error': str(e)}
            
            # Сохранение результатов
            calibration_report = {
                'timestamp': datetime.now().isoformat(),
                'improved_calibration': True,
                'models_calibrated': len([r for r in results.values() if 'error' not in r]),
                'total_models': len(models_to_calibrate),
                'results': results
            }
            
            filename = f"improved_calibration_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(calibration_report, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"📊 Улучшенная калибровка завершена. Результаты сохранены в {filename}")
            return calibration_report
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка улучшенной калибровки: {e}")
            return None
    
    async def run_single_model_test(self, model_name, test_days=7):
        """Тест одной модели с детальной диагностикой"""
        self.logger.info(f"🧪 Детальный тест модели {model_name}...")
        
        try:
            if self.calibrator is None:
                self.calibrator = AIModelCalibrator()
            
            # Получение AI модели
            ai_model = await self.calibrator.get_ai_model(model_name)
            if ai_model is None:
                self.logger.error(f"❌ Модель {model_name} не найдена")
                return None
            
            # Тест генерации сигналов
            test_pairs = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
            signals_generated = 0
            
            for pair in test_pairs:
                try:
                    # Попытка получить сигнал от модели
                    signal = await ai_model.get_trading_signal(pair)
                    if signal and signal.get('action') != 'HOLD':
                        signals_generated += 1
                        self.logger.info(f"✅ {model_name} сгенерировал сигнал для {pair}: {signal.get('action')}")
                    else:
                        self.logger.info(f"⚪ {model_name} не сгенерировал сигнал для {pair}")
                except Exception as e:
                    self.logger.warning(f"⚠️ Ошибка получения сигнала {model_name} для {pair}: {e}")
            
            test_result = {
                'model_name': model_name,
                'signals_generated': signals_generated,
                'pairs_tested': len(test_pairs),
                'signal_rate': signals_generated / len(test_pairs) if test_pairs else 0,
                'status': 'WORKING' if signals_generated > 0 else 'NOT_GENERATING_SIGNALS'
            }
            
            self.logger.info(f"📊 Тест {model_name}: {signals_generated}/{len(test_pairs)} сигналов")
            return test_result
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка теста модели {model_name}: {e}")
            return {'model_name': model_name, 'error': str(e)}
    
    async def full_diagnostic_and_recalibration(self):
        """Полная диагностика и повторная калибровка"""
        self.logger.info("🎯 Запуск полной диагностики и повторной калибровки...")
        
        # Этап 1: Диагностика
        diagnostic_result = await self.run_diagnostics()
        if diagnostic_result is None:
            self.logger.error("❌ Диагностика не удалась. Прерывание процесса.")
            return
        
        # Этап 2: Тест моделей
        self.logger.info("🧪 Тестирование моделей...")
        model_tests = {}
        for model in ['trading_ai', 'lava_ai', 'lgbm_ai', 'mistral_ai']:
            test_result = await self.run_single_model_test(model)
            model_tests[model] = test_result
        
        # Этап 3: Определение моделей для повторной калибровки
        models_to_recalibrate = [
            model for model, test in model_tests.items() 
            if test and test.get('status') == 'NOT_GENERATING_SIGNALS'
        ]
        
        if models_to_recalibrate:
            self.logger.info(f"🔧 Модели для повторной калибровки: {', '.join(models_to_recalibrate)}")
            calibration_result = await self.run_improved_calibration(models_to_recalibrate)
        else:
            self.logger.info("✅ Все модели работают корректно. Повторная калибровка не требуется.")
            calibration_result = None
        
        # Создание итогового отчета
        final_report = {
            'timestamp': datetime.now().isoformat(),
            'diagnostic_result': diagnostic_result,
            'model_tests': model_tests,
            'calibration_result': calibration_result,
            'summary': {
                'data_pairs_available': diagnostic_result.get('total_pairs_with_data', 0) if diagnostic_result else 0,
                'models_working': len([t for t in model_tests.values() if t and t.get('status') == 'WORKING']),
                'models_recalibrated': len(models_to_recalibrate) if models_to_recalibrate else 0
            }
        }
        
        filename = f"full_diagnostic_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(final_report, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"📋 Полный отчет сохранен в {filename}")
        self.logger.info("🎉 Диагностика и повторная калибровка завершены!")

async def main():
    """Главная функция"""
    print("🚀 БЫСТРАЯ ПОВТОРНАЯ КАЛИБРОВКА AI МОДЕЛЕЙ")
    print("=" * 50)
    
    recalibrator = QuickRecalibrator()
    
    # Запуск полной диагностики и повторной калибровки
    await recalibrator.full_diagnostic_and_recalibration()

if __name__ == "__main__":
    asyncio.run(main())