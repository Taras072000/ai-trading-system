#!/usr/bin/env python3
"""
Улучшенная стратегия разметки данных для торговой модели
Цель: создать более сбалансированные классы BUY/SELL/HOLD
"""

import numpy as np
import pandas as pd
from typing import Tuple, List
import logging

logger = logging.getLogger(__name__)

class ImprovedLabelingStrategy:
    """Улучшенная стратегия разметки с адаптивными порогами"""
    
    def __init__(self, config: dict = None):
        self.config = config or {
            'prediction_horizon': 1,
            'target_class_balance': 0.3,  # Целевая доля BUY/SELL классов (30% каждый)
            'min_return_threshold': 0.002,  # Минимальный порог доходности (0.2%)
            'volatility_multiplier': 1.5,  # Множитель волатильности для адаптивных порогов
            'use_percentile_thresholds': True,  # Использовать процентильные пороги
            'buy_percentile': 75,  # 75-й процентиль для BUY
            'sell_percentile': 25,  # 25-й процентиль для SELL
            'momentum_weight': 0.3,  # Вес моментума в решении
            'volume_weight': 0.2,   # Вес объема в решении
        }
    
    def calculate_adaptive_thresholds(self, returns: np.ndarray) -> Tuple[float, float]:
        """Вычисляет адаптивные пороги на основе волатильности данных"""
        volatility = np.std(returns)
        mean_return = np.mean(returns)
        
        # Базовые пороги на основе волатильности
        base_threshold = max(self.config['min_return_threshold'], 
                           volatility * self.config['volatility_multiplier'])
        
        buy_threshold = mean_return + base_threshold
        sell_threshold = mean_return - base_threshold
        
        logger.info(f"Адаптивные пороги: BUY={buy_threshold:.6f}, SELL={sell_threshold:.6f}")
        logger.info(f"Волатильность: {volatility:.6f}, Средняя доходность: {mean_return:.6f}")
        
        return buy_threshold, sell_threshold
    
    def calculate_percentile_thresholds(self, returns: np.ndarray) -> Tuple[float, float]:
        """Вычисляет пороги на основе процентилей"""
        buy_threshold = np.percentile(returns, self.config['buy_percentile'])
        sell_threshold = np.percentile(returns, self.config['sell_percentile'])
        
        logger.info(f"Процентильные пороги: BUY={buy_threshold:.6f} ({self.config['buy_percentile']}%), "
                   f"SELL={sell_threshold:.6f} ({self.config['sell_percentile']}%)")
        
        return buy_threshold, sell_threshold
    
    def calculate_momentum_score(self, data: pd.DataFrame, idx: int, window: int = 5) -> float:
        """Вычисляет оценку моментума для улучшения разметки"""
        if idx < window:
            return 0.0
        
        recent_prices = data.iloc[idx-window:idx+1]['close'].values
        if len(recent_prices) < 2:
            return 0.0
        
        # Простой моментум как наклон линейной регрессии
        x = np.arange(len(recent_prices))
        slope = np.polyfit(x, recent_prices, 1)[0]
        
        # Нормализуем относительно текущей цены
        current_price = recent_prices[-1]
        momentum_score = slope / current_price if current_price > 0 else 0.0
        
        return momentum_score
    
    def calculate_volume_score(self, data: pd.DataFrame, idx: int, window: int = 10) -> float:
        """Вычисляет оценку объема для улучшения разметки"""
        if idx < window:
            return 0.0
        
        recent_volumes = data.iloc[idx-window:idx+1]['volume'].values
        if len(recent_volumes) < 2:
            return 0.0
        
        current_volume = recent_volumes[-1]
        avg_volume = np.mean(recent_volumes[:-1])
        
        # Относительный объем
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
        
        # Нормализуем и ограничиваем
        volume_score = min(2.0, max(0.5, volume_ratio)) - 1.0  # Диапазон [-0.5, 1.0]
        
        return volume_score
    
    def create_enhanced_labels(self, data: pd.DataFrame, features_df: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame]:
        """Создает улучшенные метки с учетом множественных факторов"""
        logger.info("Создание улучшенных меток...")
        
        labels = []
        returns = []
        momentum_scores = []
        volume_scores = []
        
        # Первый проход: собираем все доходности
        for idx in features_df['index']:
            if idx + self.config['prediction_horizon'] >= len(data):
                continue
                
            current_price = data.iloc[idx]['close']
            future_price = data.iloc[idx + self.config['prediction_horizon']]['close']
            
            future_return = (future_price - current_price) / current_price
            returns.append(future_return)
        
        returns_array = np.array(returns)
        
        # Вычисляем пороги
        if self.config['use_percentile_thresholds']:
            buy_threshold, sell_threshold = self.calculate_percentile_thresholds(returns_array)
        else:
            buy_threshold, sell_threshold = self.calculate_adaptive_thresholds(returns_array)
        
        # Второй проход: создаем метки с учетом дополнительных факторов
        valid_indices = []
        for i, idx in enumerate(features_df['index']):
            if idx + self.config['prediction_horizon'] >= len(data):
                continue
            
            if i >= len(returns):
                break
                
            future_return = returns[i]
            
            # Вычисляем дополнительные факторы
            momentum_score = self.calculate_momentum_score(data, idx)
            volume_score = self.calculate_volume_score(data, idx)
            
            momentum_scores.append(momentum_score)
            volume_scores.append(volume_score)
            
            # Базовая метка на основе доходности
            if future_return > buy_threshold:
                base_label = 1  # BUY
            elif future_return < sell_threshold:
                base_label = 2  # SELL
            else:
                base_label = 0  # HOLD
            
            # Корректируем метку с учетом моментума и объема
            final_label = self.adjust_label_with_factors(
                base_label, future_return, momentum_score, volume_score,
                buy_threshold, sell_threshold
            )
            
            labels.append(final_label)
            valid_indices.append(i)
        
        # Обрезаем features_df до размера labels
        features_df = features_df.iloc[valid_indices].copy()
        
        # Статистика
        labels_array = np.array(labels)
        self.log_labeling_statistics(labels_array, returns_array, momentum_scores, volume_scores,
                                   buy_threshold, sell_threshold)
        
        return labels_array, features_df
    
    def adjust_label_with_factors(self, base_label: int, future_return: float, 
                                momentum_score: float, volume_score: float,
                                buy_threshold: float, sell_threshold: float) -> int:
        """Корректирует метку с учетом дополнительных факторов"""
        
        # Если базовая метка HOLD, проверяем возможность изменения
        if base_label == 0:
            # Сильный положительный моментум + высокий объем могут превратить HOLD в BUY
            if (momentum_score > 0.001 and volume_score > 0.2 and 
                future_return > buy_threshold * 0.7):
                return 1
            
            # Сильный отрицательный моментум + высокий объем могут превратить HOLD в SELL
            if (momentum_score < -0.001 and volume_score > 0.2 and 
                future_return < sell_threshold * 0.7):
                return 2
        
        # Если базовая метка BUY, но слабый моментум - возможно HOLD
        elif base_label == 1:
            if momentum_score < -0.0005 and volume_score < 0.1:
                return 0
        
        # Если базовая метка SELL, но положительный моментум - возможно HOLD
        elif base_label == 2:
            if momentum_score > 0.0005 and volume_score < 0.1:
                return 0
        
        return base_label
    
    def log_labeling_statistics(self, labels: np.ndarray, returns: np.ndarray,
                              momentum_scores: List[float], volume_scores: List[float],
                              buy_threshold: float, sell_threshold: float):
        """Логирует детальную статистику разметки"""
        
        hold_count = np.sum(labels == 0)
        buy_count = np.sum(labels == 1)
        sell_count = np.sum(labels == 2)
        total = len(labels)
        
        logger.info(f"=== СТАТИСТИКА УЛУЧШЕННОЙ РАЗМЕТКИ ===")
        logger.info(f"Всего меток: {total}")
        logger.info(f"HOLD: {hold_count} ({hold_count/total*100:.1f}%)")
        logger.info(f"BUY:  {buy_count} ({buy_count/total*100:.1f}%)")
        logger.info(f"SELL: {sell_count} ({sell_count/total*100:.1f}%)")
        
        logger.info(f"\nПороги:")
        logger.info(f"BUY threshold:  {buy_threshold:.6f}")
        logger.info(f"SELL threshold: {sell_threshold:.6f}")
        
        logger.info(f"\nСтатистика доходности:")
        logger.info(f"Средняя: {np.mean(returns):.6f}")
        logger.info(f"Стд. откл.: {np.std(returns):.6f}")
        logger.info(f"Мин: {np.min(returns):.6f}")
        logger.info(f"Макс: {np.max(returns):.6f}")
        
        if momentum_scores:
            logger.info(f"\nСтатистика моментума:")
            logger.info(f"Средний: {np.mean(momentum_scores):.6f}")
            logger.info(f"Стд. откл.: {np.std(momentum_scores):.6f}")
        
        if volume_scores:
            logger.info(f"\nСтатистика объема:")
            logger.info(f"Средний: {np.mean(volume_scores):.6f}")
            logger.info(f"Стд. откл.: {np.std(volume_scores):.6f}")
        
        # Анализ доходности по классам
        if buy_count > 0:
            buy_returns = returns[labels == 1]
            logger.info(f"\nДоходность BUY класса:")
            logger.info(f"Средняя: {np.mean(buy_returns):.6f}")
            logger.info(f"Медиана: {np.median(buy_returns):.6f}")
        
        if sell_count > 0:
            sell_returns = returns[labels == 2]
            logger.info(f"\nДоходность SELL класса:")
            logger.info(f"Средняя: {np.mean(sell_returns):.6f}")
            logger.info(f"Медиана: {np.median(sell_returns):.6f}")

def test_labeling_strategy():
    """Тестирование стратегии разметки"""
    # Создаем тестовые данные
    np.random.seed(42)
    n_samples = 1000
    
    # Симулируем ценовые данные с трендом и волатильностью
    prices = [100.0]
    volumes = []
    
    for i in range(n_samples):
        # Добавляем тренд и случайность
        trend = 0.0001 * np.sin(i * 0.01)  # Слабый синусоидальный тренд
        noise = np.random.normal(0, 0.01)  # Случайный шум
        
        new_price = prices[-1] * (1 + trend + noise)
        prices.append(new_price)
        
        # Объем коррелирует с волатильностью
        volume = np.random.lognormal(10, 0.5)
        volumes.append(volume)
    
    # Создаем DataFrame
    data = pd.DataFrame({
        'close': prices[1:],  # Убираем первую цену
        'volume': volumes
    })
    
    features_df = pd.DataFrame({
        'index': range(len(data))
    })
    
    # Тестируем стратегию
    strategy = ImprovedLabelingStrategy()
    labels, features_df_filtered = strategy.create_enhanced_labels(data, features_df)
    
    print("Тестирование завершено успешно!")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_labeling_strategy()