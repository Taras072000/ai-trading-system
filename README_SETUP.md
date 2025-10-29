# 🐍 Инструкция по настройке виртуального окружения

## ✅ Что уже сделано:

1. **Создано виртуальное окружение** в папке `venv/`
2. **Установлены все необходимые зависимости**:
   - mplfinance (для графиков свечей)
   - pandas, numpy (обработка данных)
   - matplotlib, seaborn, plotly (визуализация)
   - scikit-learn, lightgbm, xgboost (машинное обучение)
   - TA-Lib (технический анализ)
   - python-binance (API Binance)
   - llama-cpp-python (AI модели)
   - и другие...

## 🚀 Как запускать скрипты:

### Вариант 1: Ручная активация
```bash
source venv/bin/activate
python winrate_test_with_results2.py
```

### Вариант 2: Использование готового скрипта
```bash
./activate_env.sh
```

## 📦 Установленные пакеты:

Все зависимости перечислены в файле `requirements.txt`. Если нужно установить дополнительные пакеты:

```bash
source venv/bin/activate
pip install название_пакета
```

## 🔧 Решенные проблемы:

- ✅ ModuleNotFoundError: No module named 'mplfinance'
- ✅ ModuleNotFoundError: No module named 'talib'
- ✅ ModuleNotFoundError: No module named 'binance'
- ✅ Настроено виртуальное окружение Python

## 💡 Полезные команды:

```bash
# Активировать окружение
source venv/bin/activate

# Посмотреть установленные пакеты
pip list

# Деактивировать окружение
deactivate

# Запустить тест винрейта
python winrate_test_with_results2.py
```

## 🎯 Результат:

Скрипт `winrate_test_with_results2.py` теперь успешно запускается и работает с AI моделями для анализа торговых сигналов!