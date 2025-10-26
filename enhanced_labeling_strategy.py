#!/usr/bin/env python3
"""
Улучшенная стратегия разметки данных для торгового ИИ
"""

import pandas as pd
import numpy as np
import logging
from typing import Tuple, Optional
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class LabelingStrategy(ABC):
    """Базовый класс для стратегий разметки"""
    
    @abstractmethod
    def create_labels(self, features_df: pd.DataFrame, **kwargs) -> Tuple[np.ndarray, pd.DataFrame]:
        """Создание меток для данных"""
        pass

class EnhancedLabelingStrategy(LabelingStrategy):
    """
    Улучшенная стратегия разметки с адаптивными порогами и балансировкой классов
    """
    
    def __init__(self, 
                 base_take_profit_pct: float = 2.0,
                 base_stop_loss_pct: float = 1.5,
                 volatility_adjustment: bool = True,
                 balance_classes: bool = True,
                 min_class_ratio: float = 0.15):
        """
        Инициализация улучшенной стратегии разметки
        
        Args:
            base_take_profit_pct: Базовый процент тейк-профита
            base_stop_loss_pct: Базовый процент стоп-лосса
            volatility_adjustment: Использовать адаптацию к волатильности
            balance_classes: Балансировать классы
            min_class_ratio: Минимальное соотношение для каждого класса
        """
        self.base_take_profit_pct = base_take_profit_pct
        self.base_stop_loss_pct = base_stop_loss_pct
        self.volatility_adjustment = volatility_adjustment
        self.balance_classes = balance_classes
        self.min_class_ratio = min_class_ratio
        
    def calculate_adaptive_thresholds(self, features_df: pd.DataFrame) -> Tuple[float, float]:
        """
        Вычисление адаптивных порогов на основе волатильности
        
        Args:
            features_df: DataFrame с признаками
            
        Returns:
            Tuple[float, float]: (take_profit_pct, stop_loss_pct)
        """
        if not self.volatility_adjustment or 'current_price' not in features_df.columns:
            return self.base_take_profit_pct, self.base_stop_loss_pct
        
        # Вычисляем волатильность за последние 24 периода
        close_prices = features_df['current_price'].dropna()
        if len(close_prices) < 24:
            return self.base_take_profit_pct, self.base_stop_loss_pct
        
        # Скользящая волатильность (стандартное отклонение доходностей)
        returns = close_prices.pct_change().dropna()
        volatility = returns.rolling(window=24, min_periods=12).std()
        
        # Средняя волатильность за период
        avg_volatility = volatility.mean()
        
        if pd.isna(avg_volatility) or avg_volatility == 0:
            return self.base_take_profit_pct, self.base_stop_loss_pct
        
        # Адаптация порогов к волатильности
        # При высокой волатильности увеличиваем пороги
        volatility_multiplier = max(0.5, min(2.0, avg_volatility * 100))  # Нормализация
        
        adaptive_take_profit = self.base_take_profit_pct * volatility_multiplier
        adaptive_stop_loss = self.base_stop_loss_pct * volatility_multiplier
        
        # Ограничиваем пороги разумными значениями
        adaptive_take_profit = max(1.0, min(5.0, adaptive_take_profit))
        adaptive_stop_loss = max(0.5, min(3.0, adaptive_stop_loss))
        
        logger.info(f"Адаптивные пороги: TP={adaptive_take_profit:.2f}%, SL={adaptive_stop_loss:.2f}% (волатильность: {avg_volatility:.4f})")
        
        return adaptive_take_profit, adaptive_stop_loss
    
    def create_enhanced_labels(self, features_df: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Создание улучшенных меток с адаптивными порогами и балансировкой
        
        Args:
            features_df: DataFrame с признаками
            
        Returns:
            Tuple[np.ndarray, pd.DataFrame]: (labels, filtered_features_df)
        """
        if 'current_price' not in features_df.columns:
            logger.error("Отсутствует колонка 'current_price' в данных")
            return np.array([]), features_df
        
        # Используем current_price вместо close
        close_prices = features_df['current_price']
        
        # Вычисляем адаптивные пороги
        take_profit_pct, stop_loss_pct = self.calculate_adaptive_thresholds(features_df)
        
        # Создаем базовые метки
        labels = []
        close_prices = features_df['current_price'].values
        
        for i in range(len(close_prices) - 20):  # Оставляем 20 периодов для анализа будущего
            current_price = close_prices[i]
            future_prices = close_prices[i+1:i+21]  # Следующие 20 периодов
            
            if len(future_prices) == 0:
                continue
            
            # Вычисляем максимальную и минимальную доходность
            max_return = (np.max(future_prices) - current_price) / current_price * 100
            min_return = (np.min(future_prices) - current_price) / current_price * 100
            
            # Определяем метку на основе адаптивных порогов
            if max_return >= take_profit_pct and abs(min_return) < stop_loss_pct:
                label = 2  # BUY
            elif abs(min_return) >= stop_loss_pct and max_return < take_profit_pct:
                label = 0  # SELL
            else:
                label = 1  # HOLD
            
            labels.append(label)
        
        labels = np.array(labels)
        
        # Обрезаем features_df до размера меток
        filtered_features_df = features_df.iloc[:len(labels)].copy()
        
        if len(labels) == 0:
            logger.error("Не удалось создать метки")
            return labels, filtered_features_df
        
        # Анализ первоначального распределения
        unique, counts = np.unique(labels, return_counts=True)
        initial_distribution = dict(zip(unique, counts))
        
        logger.info("Первоначальное распределение классов:")
        for class_label, count in initial_distribution.items():
            percentage = (count / len(labels)) * 100
            class_name = ['SELL', 'HOLD', 'BUY'][int(class_label)]
            logger.info(f"  {class_name}: {count} ({percentage:.1f}%)")
        
        # Балансировка классов, если включена
        if self.balance_classes:
            labels, filtered_features_df = self._balance_classes(labels, filtered_features_df)
        
        return labels, filtered_features_df
    
    def _balance_classes(self, labels: np.ndarray, features_df: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Балансировка классов путем undersampling доминирующего класса
        
        Args:
            labels: Массив меток
            features_df: DataFrame с признаками
            
        Returns:
            Tuple[np.ndarray, pd.DataFrame]: Сбалансированные данные
        """
        unique, counts = np.unique(labels, return_counts=True)
        
        if len(unique) < 2:
            logger.warning("Недостаточно классов для балансировки")
            return labels, features_df
        
        # Находим минимальный размер класса
        min_count = np.min(counts)
        target_count = max(min_count, int(len(labels) * self.min_class_ratio))
        
        logger.info(f"Целевой размер каждого класса: {target_count}")
        
        # Собираем индексы для каждого класса
        balanced_indices = []
        
        for class_label in unique:
            class_indices = np.where(labels == class_label)[0]
            
            if len(class_indices) > target_count:
                # Случайная выборка для уменьшения класса
                np.random.seed(42)  # Для воспроизводимости
                selected_indices = np.random.choice(class_indices, target_count, replace=False)
            else:
                # Используем все доступные индексы
                selected_indices = class_indices
            
            balanced_indices.extend(selected_indices)
        
        # Сортируем индексы для сохранения временного порядка
        balanced_indices = sorted(balanced_indices)
        
        # Применяем балансировку
        balanced_labels = labels[balanced_indices]
        balanced_features = features_df.iloc[balanced_indices].copy()
        
        # Логируем результат балансировки
        unique_balanced, counts_balanced = np.unique(balanced_labels, return_counts=True)
        
        logger.info("Распределение после балансировки:")
        for class_label, count in zip(unique_balanced, counts_balanced):
            percentage = (count / len(balanced_labels)) * 100
            class_name = ['SELL', 'HOLD', 'BUY'][int(class_label)]
            logger.info(f"  {class_name}: {count} ({percentage:.1f}%)")
        
        # Проверяем качество балансировки
        balance_ratio = np.min(counts_balanced) / np.max(counts_balanced)
        logger.info(f"Коэффициент баланса после обработки: {balance_ratio:.3f}")
        
        return balanced_labels, balanced_features
    
    def create_labels(self, features_df: pd.DataFrame, **kwargs) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Основной метод создания меток (реализация абстрактного метода)
        
        Args:
            features_df: DataFrame с признаками
            **kwargs: Дополнительные параметры
            
        Returns:
            Tuple[np.ndarray, pd.DataFrame]: (labels, filtered_features_df)
        """
        return self.create_enhanced_labels(features_df)

class OriginalLabelingStrategy(LabelingStrategy):
    """
    Оригинальная стратегия разметки для сравнения
    """
    
    def __init__(self, take_profit_pct: float = 2.0, stop_loss_pct: float = 1.5):
        self.take_profit_pct = take_profit_pct
        self.stop_loss_pct = stop_loss_pct
    
    def create_labels(self, features_df: pd.DataFrame, **kwargs) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Создание меток по оригинальной стратегии
        
        Args:
            features_df: DataFrame с признаками
            **kwargs: Дополнительные параметры
            
        Returns:
            Tuple[np.ndarray, pd.DataFrame]: (labels, filtered_features_df)
        """
        if 'current_price' not in features_df.columns:
            logger.error("Отсутствует колонка 'current_price' в данных")
            return np.array([]), features_df
        
        labels = []
        close_prices = features_df['current_price'].values
        
        for i in range(len(close_prices) - 20):
            current_price = close_prices[i]
            future_prices = close_prices[i+1:i+21]
            
            if len(future_prices) == 0:
                continue
            
            max_return = (np.max(future_prices) - current_price) / current_price * 100
            min_return = (np.min(future_prices) - current_price) / current_price * 100
            
            if max_return >= self.take_profit_pct and abs(min_return) < self.stop_loss_pct:
                label = 2  # BUY
            elif abs(min_return) >= self.stop_loss_pct and max_return < self.take_profit_pct:
                label = 0  # SELL
            else:
                label = 1  # HOLD
            
            labels.append(label)
        
        labels = np.array(labels)
        filtered_features_df = features_df.iloc[:len(labels)].copy()
        
        return labels, filtered_features_df