#!/usr/bin/env python3
"""
Тестирование улучшенной модели с новой стратегией разметки и балансировкой классов
"""

import asyncio
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any
import warnings
warnings.filterwarnings('ignore')

from trading_ai_trainer import TradingAITrainer
from enhanced_labeling_strategy import EnhancedLabelingStrategy, OriginalLabelingStrategy
from data_collector import DataManager

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_improved_model():
    """Тестирование улучшенной модели"""
    
    logger.info("=== Тестирование улучшенной модели ===")
    
    # Инициализация компонентов
    data_manager = DataManager()
    enhanced_strategy = EnhancedLabelingStrategy()
    
    # Создание тренера с улучшенной стратегией
    trainer = TradingAITrainer(symbol='BTCUSDT')
    
    # Заменяем стратегию разметки на улучшенную
    trainer.labeling_strategy = enhanced_strategy
    trainer.config['use_improved_labeling'] = True  # Включаем улучшенную стратегию
    
    try:
        # Загрузка данных
        logger.info("Загрузка данных...")
        end_time = datetime.now()
        start_time = end_time - timedelta(days=30)  # Последние 30 дней
        
        data = await data_manager.ensure_data_available('BTCUSDT', '1h', 30)
        
        if data is None or len(data) < 100:
            logger.error("Недостаточно данных для тестирования")
            return
        
        logger.info(f"Загружено {len(data)} записей данных")
        
        # Подготовка признаков
        logger.info("Подготовка признаков...")
        features_df = trainer.prepare_features(data)
        logger.info(f"Подготовлено {len(features_df)} признаков")
        
        # Создание меток с улучшенной стратегией
        logger.info("Создание меток с улучшенной стратегией...")
        labels, filtered_features_df = trainer.create_labels(data, features_df)
        
        if labels is None or len(labels) == 0:
            logger.error("Не удалось создать метки")
            return
        
        logger.info(f"Создано {len(labels)} меток")
        
        # Анализ распределения классов
        unique, counts = np.unique(labels, return_counts=True)
        logger.info("Распределение классов после улучшенной стратегии:")
        for class_label, count in zip(unique, counts):
            percentage = (count / len(labels)) * 100
            class_name = ['SELL', 'HOLD', 'BUY'][int(class_label)]
            logger.info(f"  {class_name}: {count} ({percentage:.1f}%)")
        
        # Обучение модели
        logger.info("Обучение модели...")
        results = trainer.train_models(filtered_features_df, labels)
        
        if results:
            logger.info("Результаты обучения:")
            for model_name, metrics in results.items():
                logger.info(f"\n{model_name}:")
                logger.info(f"  Точность: {metrics.get('accuracy', 'N/A'):.3f}")
                logger.info(f"  F1-score: {metrics.get('f1_score', 'N/A'):.3f}")
                
                # Матрица ошибок
                if 'confusion_matrix' in metrics:
                    logger.info(f"\nМатрица ошибок:\n{metrics['confusion_matrix']}")
                
                # Детальный отчет
                if 'classification_report' in metrics:
                    logger.info(f"\nДетальный отчет по классам:\n{metrics['classification_report']}")
        
        logger.info("\n=== Тестирование завершено ===")
        
    except Exception as e:
        logger.error(f"Ошибка при тестировании: {e}")
        import traceback
        traceback.print_exc()

async def compare_strategies():
    """Сравнение оригинальной и улучшенной стратегий"""
    
    logger.info("\n=== Сравнение стратегий разметки ===")
    
    # Инициализация компонентов
    data_manager = DataManager()
    enhanced_strategy = EnhancedLabelingStrategy()
    original_strategy = OriginalLabelingStrategy()
    
    try:
        # Загрузка данных
        end_time = datetime.now()
        start_time = end_time - timedelta(days=30)
        data = await data_manager.ensure_data_available('BTCUSDT', '1h', 30)
        
        if data is None or len(data) < 100:
            logger.error("Недостаточно данных для сравнения")
            return
        
        # Тест 1: Оригинальная стратегия
        logger.info("\n--- Тест оригинальной стратегии ---")
        trainer_original = TradingAITrainer(symbol='BTCUSDT')
        
        # Заменяем стратегию разметки на оригинальную
        trainer_original.labeling_strategy = original_strategy
        trainer_original.config['use_improved_labeling'] = False  # Отключаем улучшенную стратегию
        
        features_df = trainer_original.prepare_features(data)
        labels_original, filtered_features_original = trainer_original.create_labels(data, features_df)
        
        if labels_original is not None:
            unique_orig, counts_orig = np.unique(labels_original, return_counts=True)
            logger.info("Распределение классов (оригинальная стратегия):")
            for class_label, count in zip(unique_orig, counts_orig):
                percentage = (count / len(labels_original)) * 100
                class_name = ['SELL', 'HOLD', 'BUY'][int(class_label)]
                logger.info(f"  {class_name}: {count} ({percentage:.1f}%)")
        
        # Тест 2: Улучшенная стратегия
        logger.info("\n--- Тест улучшенной стратегии ---")
        trainer_enhanced = TradingAITrainer(symbol='BTCUSDT')
        
        # Заменяем стратегию разметки на улучшенную
        trainer_enhanced.labeling_strategy = enhanced_strategy
        trainer_enhanced.config['use_improved_labeling'] = True  # Включаем улучшенную стратегию
        
        features_df = trainer_enhanced.prepare_features(data)
        labels_enhanced, filtered_features_enhanced = trainer_enhanced.create_labels(data, features_df)
        
        if labels_enhanced is not None:
            unique_enh, counts_enh = np.unique(labels_enhanced, return_counts=True)
            logger.info("Распределение классов (улучшенная стратегия):")
            for class_label, count in zip(unique_enh, counts_enh):
                percentage = (count / len(labels_enhanced)) * 100
                class_name = ['SELL', 'HOLD', 'BUY'][int(class_label)]
                logger.info(f"  {class_name}: {count} ({percentage:.1f}%)")
        
        # Сравнение результатов
        logger.info("\n--- Сравнение результатов ---")
        if labels_original is not None and labels_enhanced is not None:
            logger.info(f"Количество меток (оригинальная): {len(labels_original)}")
            logger.info(f"Количество меток (улучшенная): {len(labels_enhanced)}")
            
            # Вычисляем энтропию для оценки сбалансированности
            def calculate_entropy(labels):
                unique, counts = np.unique(labels, return_counts=True)
                probabilities = counts / len(labels)
                entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
                return entropy
            
            entropy_orig = calculate_entropy(labels_original)
            entropy_enh = calculate_entropy(labels_enhanced)
            
            logger.info(f"Энтропия (оригинальная): {entropy_orig:.3f}")
            logger.info(f"Энтропия (улучшенная): {entropy_enh:.3f}")
            logger.info("Более высокая энтропия указывает на лучшую сбалансированность классов")
        
    except Exception as e:
        logger.error(f"Ошибка при сравнении стратегий: {e}")
        import traceback
        traceback.print_exc()

async def main():
    """Главная функция"""
    logger.info("Запуск тестирования улучшенной модели...")
    
    # Тестирование улучшенной модели
    await test_improved_model()
    
    # Сравнение стратегий
    await compare_strategies()
    
    logger.info("Тестирование завершено!")

if __name__ == "__main__":
    asyncio.run(main())