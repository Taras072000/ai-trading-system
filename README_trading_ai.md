# Trading AI - Полное руководство по обучению и использованию

## Обзор системы

Система Trading AI предназначена для автоматического анализа рыночных данных и генерации торговых сигналов с целевым винрейтом 75%+. Система включает в себя:

1. **Enhanced Technical Indicators** - Расширенные технические индикаторы
2. **Data Preparation Pipeline** - Пайплайн подготовки данных и feature engineering
3. **Trading AI Trainer** - Система обучения моделей машинного обучения
4. **Advanced Backtester** - Продвинутая система бэктестинга
5. **Data Collector** - Модуль сбора данных с Binance API

## Установка зависимостей

```bash
pip install -r requirements_training.txt
```

## Быстрый старт

### 1. Сбор данных

```python
from data_collector import BinanceDataCollector, DataManager

# Инициализация коллектора данных
collector = BinanceDataCollector()
data_manager = DataManager()

# Загрузка исторических данных
symbol = "BTCUSDT"
interval = "1h"
days = 365

data = await collector.get_historical_data(symbol, interval, days)
data_manager.save_data(data, f"{symbol}_{interval}_{days}d.csv")
```

### 2. Подготовка данных

```python
from data_preparation import DataPipeline

# Настройки пайплайна
config = {
    'scaling_method': 'standard',
    'feature_selection': True,
    'selection_method': 'mutual_info',
    'n_features': 50,
    'label_config': {
        'prediction_horizon': 5,
        'threshold_buy': 0.02,
        'threshold_sell': -0.02,
        'method': 'classification'
    }
}

# Подготовка данных
pipeline = DataPipeline(config)
X, y = pipeline.prepare_data(data)
```

### 3. Обучение модели

```python
from trading_ai_trainer import TradingAITrainer

# Инициализация тренера
trainer = TradingAITrainer(
    target_win_rate=0.75,
    models=['xgboost', 'lightgbm', 'random_forest']
)

# Обучение
best_model, metrics = await trainer.train_models(X, y)
```

### 4. Бэктестинг

```python
from advanced_backtester import AdvancedBacktester

# Генерация сигналов
signals = trainer.predict_signals(X_test)

# Запуск бэктестинга
backtester = AdvancedBacktester(initial_balance=10000)
results = backtester.run_backtest(test_data, signals, "BTCUSDT")

# Генерация отчета
report = backtester.generate_report(results)
print(report)
```

## Детальное описание модулей

### Enhanced Technical Indicators

Модуль содержит более 50 технических индикаторов, разделенных на категории:

#### Трендовые индикаторы
- SMA, EMA (различные периоды)
- MACD и его вариации
- Parabolic SAR
- ADX (Average Directional Index)
- Aroon Oscillator

#### Осцилляторы
- RSI (различные периоды)
- Stochastic Oscillator
- Williams %R
- CCI (Commodity Channel Index)
- ROC (Rate of Change)

#### Индикаторы волатильности
- Bollinger Bands
- ATR (Average True Range)
- Keltner Channels
- Donchian Channels

#### Объемные индикаторы
- OBV (On-Balance Volume)
- Chaikin Money Flow
- Volume Price Trend
- Accumulation/Distribution Line

```python
from enhanced_indicators import EnhancedTechnicalIndicators

indicators = EnhancedTechnicalIndicators()

# Расчет всех индикаторов
all_indicators = indicators.calculate_all_indicators(data)

# Анализ силы сигнала
signal_strength = indicators.get_signal_strength(data)
```

### Data Preparation Pipeline

Полный пайплайн подготовки данных включает:

#### 1. Очистка данных
- Удаление дубликатов
- Обработка пропущенных значений
- Удаление выбросов методом IQR
- Валидация OHLC данных

#### 2. Feature Engineering
- **Ценовые признаки**: соотношения цен, типичная цена, логарифмические доходности
- **Технические индикаторы**: все индикаторы из enhanced_indicators
- **Объемные признаки**: объемные индикаторы и их производные
- **Временные признаки**: циклические признаки времени
- **Статистические признаки**: скользящие статистики, z-score, квантили
- **Паттерн-признаки**: свечные паттерны, ценовые паттерны
- **Лаговые признаки**: значения с задержкой
- **Скользящие признаки**: экстремумы, диапазоны

#### 3. Отбор признаков
- Mutual Information
- F-regression
- Удаление признаков с нулевой дисперсией

#### 4. Масштабирование
- StandardScaler
- MinMaxScaler
- RobustScaler

```python
# Пример настройки feature engineering
feature_config = {
    'price_features': True,
    'technical_indicators': True,
    'volume_features': True,
    'time_features': True,
    'statistical_features': True,
    'pattern_features': True,
    'lag_features': True,
    'rolling_features': True
}

pipeline = DataPipeline({
    'feature_config': feature_config,
    'scaling_method': 'robust',
    'n_features': 100
})
```

### Trading AI Trainer

Система обучения поддерживает несколько алгоритмов машинного обучения:

#### Поддерживаемые модели
- **Random Forest**: Устойчив к переобучению, хорошо работает с категориальными признаками
- **XGBoost**: Высокая производительность, встроенная регуляризация
- **LightGBM**: Быстрое обучение, эффективное использование памяти
- **Gradient Boosting**: Классический градиентный бустинг

#### Оптимизация гиперпараметров
Используется Optuna для автоматической оптимизации:

```python
# Пример настройки оптимизации
trainer = TradingAITrainer(
    target_win_rate=0.75,
    models=['xgboost', 'lightgbm'],
    optimization_trials=100,
    cv_folds=5
)
```

#### Метрики оценки
- **Win Rate**: Процент прибыльных сделок
- **Precision/Recall**: Для каждого класса (BUY/SELL/HOLD)
- **F1-Score**: Гармоническое среднее precision и recall
- **ROC-AUC**: Площадь под ROC-кривой
- **Profit Factor**: Отношение прибыли к убыткам

### Advanced Backtester

Продвинутая система бэктестинга с детальной аналитикой:

#### Основные возможности
- Реалистичное моделирование торговли
- Учет комиссий и проскальзывания
- Стоп-лоссы и тейк-профиты
- Управление рисками
- Детальная аналитика результатов

#### Метрики анализа
- **Торговые метрики**: винрейт, количество сделок, средняя длительность
- **Финансовые метрики**: Sharpe ratio, максимальная просадка, profit factor
- **Риск-метрики**: VaR, максимальные последовательные убытки

```python
# Настройка бэктестера
backtester = AdvancedBacktester(
    initial_balance=10000,
    commission=0.001,  # 0.1% комиссия
    max_position_size=0.95,  # 95% от баланса
    stop_loss_pct=0.02,  # 2% стоп-лосс
    take_profit_pct=0.04   # 4% тейк-профит
)
```

## Полный пример обучения

```python
import asyncio
from data_collector import BinanceDataCollector, DataManager
from data_preparation import DataPipeline
from trading_ai_trainer import TradingAITrainer
from advanced_backtester import AdvancedBacktester

async def full_training_pipeline():
    # 1. Сбор данных
    collector = BinanceDataCollector()
    data = await collector.get_historical_data("BTCUSDT", "1h", 365)
    
    # 2. Подготовка данных
    config = {
        'scaling_method': 'standard',
        'feature_selection': True,
        'n_features': 50,
        'label_config': {
            'prediction_horizon': 5,
            'threshold_buy': 0.02,
            'threshold_sell': -0.02
        }
    }
    
    pipeline = DataPipeline(config)
    X, y = pipeline.prepare_data(data)
    
    # 3. Разделение на train/test
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # 4. Обучение
    trainer = TradingAITrainer(target_win_rate=0.75)
    best_model, metrics = await trainer.train_models(X_train, y_train)
    
    print(f"Лучшая модель: {best_model}")
    print(f"Метрики: {metrics}")
    
    # 5. Генерация сигналов для тестовых данных
    test_data = data[split_idx:]
    signals = trainer.predict_signals(X_test)
    
    # 6. Бэктестинг
    backtester = AdvancedBacktester()
    results = backtester.run_backtest(test_data, signals)
    
    # 7. Анализ результатов
    report = backtester.generate_report(results)
    print(report)
    
    # 8. Сохранение модели
    trainer.save_model("best_trading_model.pkl")
    
    return results

# Запуск
if __name__ == "__main__":
    results = asyncio.run(full_training_pipeline())
```

## Оптимизация для достижения винрейта 75%+

### 1. Настройка порогов
```python
# Более консервативные пороги для повышения точности
label_config = {
    'threshold_buy': 0.025,   # 2.5% вместо 2%
    'threshold_sell': -0.025, # -2.5% вместо -2%
    'prediction_horizon': 3   # Более короткий горизонт
}
```

### 2. Увеличение количества признаков
```python
# Используем больше признаков для лучшего качества
pipeline_config = {
    'n_features': 100,  # Вместо 50
    'selection_method': 'mutual_info'
}
```

### 3. Ансамбль моделей
```python
# Используем несколько моделей для повышения стабильности
trainer = TradingAITrainer(
    models=['xgboost', 'lightgbm', 'random_forest'],
    use_ensemble=True,
    ensemble_method='voting'
)
```

### 4. Дополнительная фильтрация сигналов
```python
# Фильтруем сигналы по уверенности модели
def filter_signals(signals, confidence_threshold=0.8):
    filtered_signals = signals.copy()
    low_confidence = signals['confidence'] < confidence_threshold
    filtered_signals.loc[low_confidence, 'action'] = 'HOLD'
    return filtered_signals
```

## Мониторинг и улучшение

### 1. Отслеживание метрик
- Ведите журнал всех экспериментов
- Сравнивайте результаты разных конфигураций
- Анализируйте ошибки модели

### 2. Регулярное переобучение
- Переобучайте модель на новых данных
- Используйте скользящее окно для обучения
- Адаптируйтесь к изменениям рынка

### 3. A/B тестирование
- Тестируйте новые стратегии на исторических данных
- Сравнивайте производительность разных подходов
- Внедряйте улучшения постепенно

## Заключение

Данная система предоставляет полный набор инструментов для создания высокоэффективной торговой AI системы. Ключевые факторы успеха:

1. **Качественные данные**: Тщательная очистка и подготовка данных
2. **Богатый набор признаков**: Использование множества технических индикаторов
3. **Правильная настройка**: Оптимизация гиперпараметров и порогов
4. **Тщательное тестирование**: Детальный бэктестинг и анализ результатов
5. **Постоянное улучшение**: Мониторинг и адаптация к изменениям рынка

При правильном использовании система способна достичь целевого винрейта 75%+ и обеспечить стабильную прибыльность торговых операций.