#!/usr/bin/env python3
"""
🚀 Скрипт запуска калибровки AI моделей
Поэтапная калибровка каждой модели для устранения убыточности
"""

import asyncio
import logging
import sys
from datetime import datetime
from ai_model_calibrator import AIModelCalibrator

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'calibration_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

async def run_individual_model_calibration(model_name: str):
    """Запуск калибровки одной модели"""
    logger.info(f"🎯 Запуск калибровки модели: {model_name}")
    
    calibrator = AIModelCalibrator()
    
    if not await calibrator.initialize():
        logger.error("❌ Не удалось инициализировать систему")
        return None
    
    try:
        result = await calibrator.calibrate_individual_model(model_name)
        
        # Вывод результатов
        best_result = result.best_result
        logger.info(f"\n{'='*60}")
        logger.info(f"✅ РЕЗУЛЬТАТЫ КАЛИБРОВКИ: {model_name}")
        logger.info(f"{'='*60}")
        logger.info(f"📊 Винрейт: {best_result.win_rate:.1f}%")
        logger.info(f"💰 Общий P&L: {best_result.total_pnl:.2f}%")
        logger.info(f"📈 Средняя прибыль на сделку: {best_result.avg_profit_per_trade:.2f}%")
        logger.info(f"📉 Максимальная просадка: {best_result.max_drawdown:.2f}%")
        logger.info(f"📊 Sharpe Ratio: {best_result.sharpe_ratio:.2f}")
        logger.info(f"🔄 Общее количество сделок: {best_result.total_trades}")
        logger.info(f"⏱️ Среднее время удержания: {best_result.avg_holding_time:.1f} часов")
        logger.info(f"📈 Улучшение: {result.improvement_percentage:.1f}%")
        
        # Лучшие параметры
        best_config = result.best_config
        logger.info(f"\n🔧 ОПТИМАЛЬНЫЕ ПАРАМЕТРЫ:")
        logger.info(f"   Порог уверенности: {best_config.confidence_threshold}")
        logger.info(f"   Take Profit: {best_config.take_profit}%")
        logger.info(f"   Stop Loss: {best_config.stop_loss}%")
        logger.info(f"   Мин. объем: {best_config.min_volume_threshold}")
        logger.info(f"   Мин. волатильность: {best_config.min_volatility_threshold}")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Ошибка калибровки {model_name}: {e}")
        return None

async def run_full_calibration():
    """Запуск полной калибровки всех моделей"""
    logger.info("🚀 ЗАПУСК ПОЛНОЙ КАЛИБРОВКИ ВСЕХ AI МОДЕЛЕЙ")
    logger.info("="*80)
    
    calibrator = AIModelCalibrator()
    results = await calibrator.run_full_calibration()
    
    if results:
        logger.info("\n🎉 КАЛИБРОВКА ЗАВЕРШЕНА УСПЕШНО!")
        logger.info("📋 Проверьте папку reports/calibration для детальных отчетов")
    else:
        logger.error("❌ Калибровка завершилась с ошибками")

async def main():
    """Главная функция"""
    if len(sys.argv) > 1:
        model_name = sys.argv[1]
        valid_models = ['trading_ai', 'lava_ai', 'lgbm_ai', 'mistral_ai', 'reinforcement_learning_engine']
        
        if model_name in valid_models:
            await run_individual_model_calibration(model_name)
        elif model_name == 'all':
            await run_full_calibration()
        else:
            logger.error(f"❌ Неизвестная модель: {model_name}")
            logger.info(f"✅ Доступные модели: {', '.join(valid_models)}")
            logger.info("✅ Или используйте 'all' для калибровки всех моделей")
    else:
        logger.info("🎯 Выберите режим калибровки:")
        logger.info("1. Калибровка всех моделей")
        logger.info("2. Калибровка отдельной модели")
        
        choice = input("Введите номер (1 или 2): ").strip()
        
        if choice == '1':
            await run_full_calibration()
        elif choice == '2':
            logger.info("Доступные модели:")
            models = ['trading_ai', 'lava_ai', 'lgbm_ai', 'mistral_ai', 'reinforcement_learning_engine']
            for i, model in enumerate(models, 1):
                logger.info(f"{i}. {model}")
            
            try:
                model_choice = int(input("Выберите номер модели: ").strip()) - 1
                if 0 <= model_choice < len(models):
                    await run_individual_model_calibration(models[model_choice])
                else:
                    logger.error("❌ Неверный номер модели")
            except ValueError:
                logger.error("❌ Введите корректный номер")
        else:
            logger.error("❌ Неверный выбор")

if __name__ == "__main__":
    asyncio.run(main())