#!/usr/bin/env python3
"""
Mass Training Script - Массовое обучение AI моделей
Обучает модели для всех указанных монет с 3-летними данными

Использование:
    python mass_training_script.py
"""

import asyncio
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
import json

# Добавляем текущую директорию в путь для импортов
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from trading_ai_trainer import TradingAITrainer
from data_collector import DataManager

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('mass_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MassTrainingManager:
    """Менеджер для массового обучения моделей"""
    
    def __init__(self):
        # Список монет для обучения
        self.symbols = ['BNBUSDT', 'ADAUSDT', 'SOLUSDT', 'AVAXUSDT', 'DOTUSDT']
        
        # Параметры обучения
        self.training_config = {
            'days': 1095,  # 3 года данных (365 * 3)
            'target_winrate': 0.75,  # Целевой винрейт 75%
            'lookback_period': 50,
            'prediction_horizon': 1,
            'min_confidence': 0.6,
            'risk_reward_ratio': 2.0,
            'stop_loss_pct': 0.02,
            'take_profit_pct': 0.04
        }
        
        # Результаты обучения
        self.training_results = {}
        
        # Создаем директорию для результатов
        self.results_dir = Path("training_results")
        self.results_dir.mkdir(exist_ok=True)
    
    async def train_single_symbol(self, symbol: str) -> Dict[str, Any]:
        """Обучение модели для одной монеты"""
        logger.info(f"🚀 Начинаем обучение модели для {symbol}")
        
        try:
            # Создаем тренер для символа
            trainer = TradingAITrainer(symbol)
            
            # Обновляем конфигурацию тренера
            trainer.config.update(self.training_config)
            
            logger.info(f"📊 Загружаем {self.training_config['days']} дней данных для {symbol}")
            
            # Загружаем данные за 3 года
            data = await trainer.load_market_data(self.training_config['days'])
            
            if data is None or len(data) < 1000:
                logger.error(f"❌ Недостаточно данных для {symbol}: {len(data) if data is not None else 0} записей")
                return {
                    'symbol': symbol,
                    'status': 'failed',
                    'error': 'Insufficient data',
                    'data_points': len(data) if data is not None else 0
                }
            
            logger.info(f"✅ Загружено {len(data)} записей для {symbol}")
            
            # Подготавливаем признаки с полным набором индикаторов
            logger.info(f"🔧 Подготавливаем признаки для {symbol}")
            features_df = trainer.prepare_features(data)
            
            if features_df is None or len(features_df) == 0:
                logger.error(f"❌ Ошибка подготовки признаков для {symbol}")
                return {
                    'symbol': symbol,
                    'status': 'failed',
                    'error': 'Feature preparation failed'
                }
            
            logger.info(f"✅ Подготовлено {len(features_df)} признаков для {symbol}")
            
            # Создаем метки
            logger.info(f"🏷️ Создаем метки для {symbol}")
            labels, features_df = trainer.create_labels(data, features_df)
            
            if labels is None or len(labels) == 0:
                logger.error(f"❌ Ошибка создания меток для {symbol}")
                return {
                    'symbol': symbol,
                    'status': 'failed',
                    'error': 'Label creation failed'
                }
            
            # Обучаем модели
            logger.info(f"🤖 Обучаем модели для {symbol}")
            training_results = trainer.train_models(features_df, labels)
            
            # Оцениваем модели
            logger.info(f"📈 Оцениваем модели для {symbol}")
            evaluation = trainer.evaluate_model(training_results)
            
            # Сохраняем лучшую модель
            logger.info(f"💾 Сохраняем модель для {symbol}")
            trainer.save_model()
            
            # Формируем результат
            result = {
                'symbol': symbol,
                'status': 'success',
                'data_points': len(data),
                'features_count': len(features_df.columns),
                'labels_count': len(labels),
                'best_model': evaluation.get('best_model', 'unknown'),
                'best_score': evaluation.get('best_score', 0),
                'training_time': datetime.now().isoformat(),
                'model_path': f"models/trading_ai/{symbol}_trading_model.joblib",
                'evaluation': evaluation
            }
            
            logger.info(f"✅ Обучение {symbol} завершено успешно!")
            logger.info(f"📊 Лучшая модель: {result['best_model']}")
            logger.info(f"🎯 Точность: {result['best_score']:.4f}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Ошибка обучения {symbol}: {str(e)}", exc_info=True)
            return {
                'symbol': symbol,
                'status': 'failed',
                'error': str(e),
                'training_time': datetime.now().isoformat()
            }
    
    async def train_all_symbols(self) -> Dict[str, Any]:
        """Обучение моделей для всех символов"""
        logger.info("🚀 Начинаем массовое обучение AI моделей")
        logger.info(f"📋 Символы для обучения: {', '.join(self.symbols)}")
        logger.info(f"📅 Период данных: {self.training_config['days']} дней (3 года)")
        
        start_time = datetime.now()
        
        # Обучаем модели последовательно (чтобы не перегружать систему)
        for symbol in self.symbols:
            result = await self.train_single_symbol(symbol)
            self.training_results[symbol] = result
            
            # Сохраняем промежуточные результаты
            await self.save_intermediate_results()
        
        end_time = datetime.now()
        training_duration = end_time - start_time
        
        # Формируем итоговый отчет
        summary = self.generate_summary(training_duration)
        
        # Сохраняем финальные результаты
        await self.save_final_results(summary)
        
        logger.info("🎉 Массовое обучение завершено!")
        self.print_summary(summary)
        
        return {
            'summary': summary,
            'results': self.training_results
        }
    
    def generate_summary(self, training_duration) -> Dict[str, Any]:
        """Генерация сводки результатов"""
        successful = [r for r in self.training_results.values() if r['status'] == 'success']
        failed = [r for r in self.training_results.values() if r['status'] == 'failed']
        
        summary = {
            'total_symbols': len(self.symbols),
            'successful_trainings': len(successful),
            'failed_trainings': len(failed),
            'success_rate': len(successful) / len(self.symbols) * 100,
            'training_duration': str(training_duration),
            'training_config': self.training_config,
            'timestamp': datetime.now().isoformat()
        }
        
        if successful:
            summary['best_models'] = {
                r['symbol']: {
                    'model': r['best_model'],
                    'score': r['best_score'],
                    'data_points': r['data_points']
                }
                for r in successful
            }
            
            summary['average_score'] = sum(r['best_score'] for r in successful) / len(successful)
            summary['total_data_points'] = sum(r['data_points'] for r in successful)
        
        if failed:
            summary['failed_symbols'] = {
                r['symbol']: r['error']
                for r in failed
            }
        
        return summary
    
    def print_summary(self, summary: Dict[str, Any]):
        """Вывод сводки в консоль"""
        print("\n" + "="*60)
        print("📊 СВОДКА РЕЗУЛЬТАТОВ МАССОВОГО ОБУЧЕНИЯ")
        print("="*60)
        print(f"📋 Всего символов: {summary['total_symbols']}")
        print(f"✅ Успешно обучено: {summary['successful_trainings']}")
        print(f"❌ Неудачных: {summary['failed_trainings']}")
        print(f"📈 Процент успеха: {summary['success_rate']:.1f}%")
        print(f"⏱️ Время обучения: {summary['training_duration']}")
        
        if 'average_score' in summary:
            print(f"🎯 Средняя точность: {summary['average_score']:.4f}")
            print(f"📊 Всего точек данных: {summary['total_data_points']:,}")
        
        if 'best_models' in summary:
            print("\n🏆 ЛУЧШИЕ МОДЕЛИ:")
            for symbol, info in summary['best_models'].items():
                print(f"  {symbol}: {info['model']} (точность: {info['score']:.4f})")
        
        if 'failed_symbols' in summary:
            print("\n❌ НЕУДАЧНЫЕ ОБУЧЕНИЯ:")
            for symbol, error in summary['failed_symbols'].items():
                print(f"  {symbol}: {error}")
        
        print("="*60)
    
    async def save_intermediate_results(self):
        """Сохранение промежуточных результатов"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.results_dir / f"intermediate_results_{timestamp}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.training_results, f, indent=2, ensure_ascii=False)
    
    async def save_final_results(self, summary: Dict[str, Any]):
        """Сохранение финальных результатов"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Сохраняем полные результаты
        results_filename = self.results_dir / f"mass_training_results_{timestamp}.json"
        with open(results_filename, 'w', encoding='utf-8') as f:
            json.dump({
                'summary': summary,
                'detailed_results': self.training_results
            }, f, indent=2, ensure_ascii=False)
        
        # Сохраняем краткую сводку
        summary_filename = self.results_dir / f"training_summary_{timestamp}.txt"
        with open(summary_filename, 'w', encoding='utf-8') as f:
            f.write("СВОДКА РЕЗУЛЬТАТОВ МАССОВОГО ОБУЧЕНИЯ AI МОДЕЛЕЙ\n")
            f.write("="*60 + "\n\n")
            f.write(f"Дата и время: {summary['timestamp']}\n")
            f.write(f"Всего символов: {summary['total_symbols']}\n")
            f.write(f"Успешно обучено: {summary['successful_trainings']}\n")
            f.write(f"Неудачных: {summary['failed_trainings']}\n")
            f.write(f"Процент успеха: {summary['success_rate']:.1f}%\n")
            f.write(f"Время обучения: {summary['training_duration']}\n")
            
            if 'average_score' in summary:
                f.write(f"Средняя точность: {summary['average_score']:.4f}\n")
                f.write(f"Всего точек данных: {summary['total_data_points']:,}\n")
            
            f.write("\nПараметры обучения:\n")
            for key, value in summary['training_config'].items():
                f.write(f"  {key}: {value}\n")
            
            if 'best_models' in summary:
                f.write("\nЛучшие модели:\n")
                for symbol, info in summary['best_models'].items():
                    f.write(f"  {symbol}: {info['model']} (точность: {info['score']:.4f}, данных: {info['data_points']:,})\n")
            
            if 'failed_symbols' in summary:
                f.write("\nНеудачные обучения:\n")
                for symbol, error in summary['failed_symbols'].items():
                    f.write(f"  {symbol}: {error}\n")
        
        logger.info(f"📄 Результаты сохранены в: {results_filename}")
        logger.info(f"📄 Сводка сохранена в: {summary_filename}")

async def main():
    """Главная функция"""
    print("🚀 Запуск массового обучения AI моделей")
    print("📋 Монеты: BNBUSDT, ADAUSDT, SOLUSDT, AVAXUSDT, DOTUSDT")
    print("📅 Период: 3 года данных")
    print("🎯 Цель: винрейт 75%+")
    print("\n" + "="*60)
    
    # Создаем менеджер обучения
    manager = MassTrainingManager()
    
    try:
        # Запускаем массовое обучение
        results = await manager.train_all_symbols()
        
        print("\n🎉 Массовое обучение завершено успешно!")
        return results
        
    except KeyboardInterrupt:
        logger.info("⏹️ Обучение прервано пользователем")
        print("\n⏹️ Обучение прервано пользователем")
        return None
        
    except Exception as e:
        logger.error(f"❌ Критическая ошибка: {str(e)}", exc_info=True)
        print(f"\n❌ Критическая ошибка: {str(e)}")
        return None

if __name__ == "__main__":
    # Запускаем массовое обучение
    results = asyncio.run(main())