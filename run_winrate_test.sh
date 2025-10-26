#!/bin/bash

echo "🚀 Запуск тестирования винрейта через venv311..."
echo "=================================================="

# Проверяем существование виртуального окружения
if [ ! -d "venv311" ]; then
    echo "❌ Ошибка: Виртуальное окружение venv311 не найдено!"
    echo "Создайте его командой: python3.11 -m venv venv311"
    exit 1
fi

# Проверяем существование скрипта
if [ ! -f "winrate_test_with_results2.py" ]; then
    echo "❌ Ошибка: Файл winrate_test_with_results2.py не найден!"
    exit 1
fi

echo "✅ Активация виртуального окружения venv311..."
source venv311/bin/activate

echo "🔍 Проверка установленных пакетов..."
python -c "import mplfinance; print('✅ mplfinance:', mplfinance.__version__)" 2>/dev/null || echo "⚠️ mplfinance не найден"

echo "🎯 Запуск тестирования винрейта..."
echo "=================================================="

# Запуск скрипта с обработкой ошибок
python winrate_test_with_results2.py

exit_code=$?

echo "=================================================="
if [ $exit_code -eq 0 ]; then
    echo "✅ Тестирование завершено успешно!"
else
    echo "❌ Тестирование завершено с ошибкой (код: $exit_code)"
fi

deactivate
echo "🔚 Виртуальное окружение деактивировано"