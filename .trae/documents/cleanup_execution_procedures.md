# Процедуры выполнения очистки проекта

## 1. Подготовительные процедуры

### 1.1 Создание рабочей среды

```bash
#!/bin/bash
# Скрипт подготовки к очистке проекта

# Создание директории для резервных копий
mkdir -p ./cleanup_backups
mkdir -p ./cleanup_logs

# Создание лог-файла
LOG_FILE="./cleanup_logs/cleanup_$(date +%Y%m%d_%H%M%S).log"
echo "Начало процедуры очистки: $(date)" > $LOG_FILE

# Функция логирования
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a $LOG_FILE
}

log "Подготовка к очистке проекта AI торговой системы"
```

### 1.2 Анализ зависимостей

```bash
#!/bin/bash
# Скрипт анализа зависимостей

analyze_dependencies() {
    local file_to_check=$1
    log "Анализ зависимостей для: $file_to_check"
    
    # Поиск импортов файла
    echo "=== ИМПОРТЫ ===" >> "./cleanup_logs/dependencies_$file_to_check.txt"
    grep -n "^import\|^from.*import" "$file_to_check" >> "./cleanup_logs/dependencies_$file_to_check.txt" 2>/dev/null
    
    # Поиск использования файла в других модулях
    echo "=== ИСПОЛЬЗОВАНИЕ ===" >> "./cleanup_logs/dependencies_$file_to_check.txt"
    filename=$(basename "$file_to_check" .py)
    find . -name "*.py" -exec grep -l "$filename" {} \; >> "./cleanup_logs/dependencies_$file_to_check.txt" 2>/dev/null
    
    log "Анализ зависимостей завершен для: $file_to_check"
}

# Анализ всех файлов для удаления
FILES_TO_ANALYZE=(
    "test_fix.py"
    "test_fix_simple.py"
    "test_bug_fix.py"
    "debug_ai_models.py"
    "mock_ai_trading_system.py"
    "simple_calibrator.py"
)

for file in "${FILES_TO_ANALYZE[@]}"; do
    if [ -f "$file" ]; then
        analyze_dependencies "$file"
    else
        log "ПРЕДУПРЕЖДЕНИЕ: Файл $file не найден"
    fi
done
```

### 1.3 Создание резервных копий

```bash
#!/bin/bash
# Скрипт создания резервных копий

create_backup() {
    local backup_name="backup_$(date +%Y%m%d_%H%M%S)"
    local backup_path="./cleanup_backups/$backup_name.tar.gz"
    
    log "Создание резервной копии: $backup_path"
    
    # Исключаем временные файлы и кэш
    tar --exclude='*.pyc' \
        --exclude='__pycache__' \
        --exclude='.git' \
        --exclude='cleanup_backups' \
        --exclude='cleanup_logs' \
        -czf "$backup_path" .
    
    if [ $? -eq 0 ]; then
        log "Резервная копия создана успешно: $backup_path"
        echo "$backup_path" > "./cleanup_backups/latest_backup.txt"
    else
        log "ОШИБКА: Не удалось создать резервную копию"
        exit 1
    fi
}

create_backup
```

## 2. Процедуры удаления по этапам

### 2.1 Этап 1: Удаление тестовых файлов

```bash
#!/bin/bash
# Этап 1: Удаление тестовых файлов

STAGE="stage1_test_files"
log "=== ЭТАП 1: УДАЛЕНИЕ ТЕСТОВЫХ ФАЙЛОВ ==="

# Список тестовых файлов для удаления
TEST_FILES=(
    "test_fix.py"
    "test_fix_simple.py"
    "test_bug_fix.py"
    "test_lava_debug.py"
    "test_lava_fixed.py"
    "test_lava_improvements.py"
    "test_lava_winrate.py"
    "test_mistral_local.py"
    "test_orchestrator_debug.py"
)

# Создание резервной копии перед этапом
create_backup

# Удаление файлов
for file in "${TEST_FILES[@]}"; do
    if [ -f "$file" ]; then
        log "Удаление тестового файла: $file"
        rm -f "$file"
        if [ $? -eq 0 ]; then
            log "✅ Успешно удален: $file"
        else
            log "❌ Ошибка удаления: $file"
        fi
    else
        log "⚠️ Файл не найден: $file"
    fi
done

# Проверка системы после удаления
log "Проверка системы после удаления тестовых файлов..."
python -c "import sys; sys.path.append('.'); import main; print('✅ Основной модуль загружается корректно')" 2>&1 | tee -a $LOG_FILE

if [ $? -eq 0 ]; then
    log "✅ Этап 1 завершен успешно"
else
    log "❌ Этап 1 завершен с ошибками"
fi
```

### 2.2 Этап 2: Удаление отладочных файлов

```bash
#!/bin/bash
# Этап 2: Удаление отладочных файлов

STAGE="stage2_debug_files"
log "=== ЭТАП 2: УДАЛЕНИЕ ОТЛАДОЧНЫХ ФАЙЛОВ ==="

# Список отладочных файлов для удаления
DEBUG_FILES=(
    "debug_ai_models.py"
    "debug_consensus.py"
    "debug_consensus_logic.py"
    "debug_get_ai_signals.py"
    "debug_signals.py"
    "debug_signals_issue.py"
    "simple_signal_debug.py"
)

# Создание резервной копии перед этапом
create_backup

# Удаление файлов
for file in "${DEBUG_FILES[@]}"; do
    if [ -f "$file" ]; then
        log "Удаление отладочного файла: $file"
        rm -f "$file"
        if [ $? -eq 0 ]; then
            log "✅ Успешно удален: $file"
        else
            log "❌ Ошибка удаления: $file"
        fi
    else
        log "⚠️ Файл не найден: $file"
    fi
done

# Проверка AI модулей
log "Проверка AI модулей после удаления отладочных файлов..."
python -c "
try:
    from ai_modules.ai_manager import AIManager
    print('✅ AI Manager загружается корректно')
except Exception as e:
    print(f'❌ Ошибка загрузки AI Manager: {e}')
" 2>&1 | tee -a $LOG_FILE

if [ $? -eq 0 ]; then
    log "✅ Этап 2 завершен успешно"
else
    log "❌ Этап 2 завершен с ошибками"
fi
```

### 2.3 Этап 3: Удаление mock и demo файлов

```bash
#!/bin/bash
# Этап 3: Удаление mock и demo файлов

STAGE="stage3_mock_demo_files"
log "=== ЭТАП 3: УДАЛЕНИЕ MOCK И DEMO ФАЙЛОВ ==="

# Список mock и demo файлов для удаления
MOCK_DEMO_FILES=(
    "mock_ai_trading_system.py"
    "demo_system.py"
    "simple_calibrator.py"
)

# Создание резервной копии перед этапом
create_backup

# Удаление файлов
for file in "${MOCK_DEMO_FILES[@]}"; do
    if [ -f "$file" ]; then
        log "Удаление mock/demo файла: $file"
        rm -f "$file"
        if [ $? -eq 0 ]; then
            log "✅ Успешно удален: $file"
        else
            log "❌ Ошибка удаления: $file"
        fi
    else
        log "⚠️ Файл не найден: $file"
    fi
done

# Проверка основной системы
log "Проверка основной системы после удаления mock/demo файлов..."
python -c "
try:
    import config
    print('✅ Конфигурация загружается корректно')
    from winrate_test_with_results2 import RealWinrateTester
    print('✅ Основной тестер загружается корректно')
except Exception as e:
    print(f'❌ Ошибка загрузки: {e}')
" 2>&1 | tee -a $LOG_FILE

if [ $? -eq 0 ]; then
    log "✅ Этап 3 завершен успешно"
else
    log "❌ Этап 3 завершен с ошибками"
fi
```

### 2.4 Этап 4: Удаление устаревших конфигураций

```bash
#!/bin/bash
# Этап 4: Удаление устаревших конфигураций

STAGE="stage4_old_configs"
log "=== ЭТАП 4: УДАЛЕНИЕ УСТАРЕВШИХ КОНФИГУРАЦИЙ ==="

# Создание резервной копии перед этапом
create_backup

# Удаление конфигурационных файлов с датами
log "Удаление конфигурационных файлов с датами..."
find . -name "*_20251024_*.json" -type f | while read file; do
    log "Удаление конфигурации: $file"
    rm -f "$file"
    if [ $? -eq 0 ]; then
        log "✅ Успешно удален: $file"
    else
        log "❌ Ошибка удаления: $file"
    fi
done

# Удаление диагностических отчетов с датами
log "Удаление диагностических отчетов с датами..."
find . -name "*_20251024_*.txt" -type f | while read file; do
    log "Удаление отчета: $file"
    rm -f "$file"
    if [ $? -eq 0 ]; then
        log "✅ Успешно удален: $file"
    else
        log "❌ Ошибка удаления: $file"
    fi
done

# Проверка конфигурации
log "Проверка конфигурации после удаления устаревших файлов..."
python -c "
try:
    import config
    cfg = config.Config()
    print('✅ Конфигурация инициализируется корректно')
except Exception as e:
    print(f'❌ Ошибка конфигурации: {e}')
" 2>&1 | tee -a $LOG_FILE

if [ $? -eq 0 ]; then
    log "✅ Этап 4 завершен успешно"
else
    log "❌ Этап 4 завершен с ошибками"
fi
```

## 3. Процедуры тестирования

### 3.1 Быстрое тестирование системы

```bash
#!/bin/bash
# Быстрое тестирование системы

quick_system_test() {
    log "=== БЫСТРОЕ ТЕСТИРОВАНИЕ СИСТЕМЫ ==="
    
    # Тест 1: Импорт основных модулей
    log "Тест 1: Импорт основных модулей"
    python -c "
import sys
sys.path.append('.')

modules_to_test = [
    'main',
    'config',
    'winrate_test_with_results2',
    'advanced_strategy_tester',
    'sequential_strategy_tester'
]

for module in modules_to_test:
    try:
        __import__(module)
        print(f'✅ {module} - OK')
    except Exception as e:
        print(f'❌ {module} - ОШИБКА: {e}')
" 2>&1 | tee -a $LOG_FILE
    
    # Тест 2: AI модули
    log "Тест 2: AI модули"
    python -c "
try:
    from ai_modules.ai_manager import AIManager
    print('✅ AI Manager - OK')
    from ai_modules.trading_ai import TradingAI
    print('✅ Trading AI - OK')
    from ai_modules.lava_ai import LavaAI
    print('✅ Lava AI - OK')
except Exception as e:
    print(f'❌ AI модули - ОШИБКА: {e}')
" 2>&1 | tee -a $LOG_FILE
    
    # Тест 3: Конфигурация
    log "Тест 3: Конфигурация"
    python -c "
try:
    import config
    cfg = config.Config()
    print('✅ Конфигурация - OK')
except Exception as e:
    print(f'❌ Конфигурация - ОШИБКА: {e}')
" 2>&1 | tee -a $LOG_FILE
}

quick_system_test
```

### 3.2 Полное тестирование системы

```bash
#!/bin/bash
# Полное тестирование системы

full_system_test() {
    log "=== ПОЛНОЕ ТЕСТИРОВАНИЕ СИСТЕМЫ ==="
    
    # Тест 1: Запуск основного тестера
    log "Тест 1: Запуск основного тестера"
    timeout 60 python winrate_test_with_results2.py --test-days 1 --quick-mode 2>&1 | tee -a $LOG_FILE
    
    if [ $? -eq 0 ]; then
        log "✅ Основной тестер работает корректно"
    else
        log "❌ Проблемы с основным тестером"
    fi
    
    # Тест 2: Проверка AI модулей
    log "Тест 2: Проверка AI модулей"
    python ai_models_diagnostics.py 2>&1 | tee -a $LOG_FILE
    
    # Тест 3: Проверка последовательного тестирования
    log "Тест 3: Проверка последовательного тестирования"
    timeout 120 python run_sequential_testing.py --quick-mode 2>&1 | tee -a $LOG_FILE
    
    if [ $? -eq 0 ]; then
        log "✅ Последовательное тестирование работает корректно"
    else
        log "❌ Проблемы с последовательным тестированием"
    fi
}

full_system_test
```

## 4. Процедуры отката

### 4.1 Автоматический откат

```bash
#!/bin/bash
# Процедура автоматического отката

rollback_to_backup() {
    local backup_file=$1
    
    if [ -z "$backup_file" ]; then
        # Использовать последнюю резервную копию
        if [ -f "./cleanup_backups/latest_backup.txt" ]; then
            backup_file=$(cat "./cleanup_backups/latest_backup.txt")
        else
            log "❌ Не найден файл последней резервной копии"
            return 1
        fi
    fi
    
    if [ ! -f "$backup_file" ]; then
        log "❌ Файл резервной копии не найден: $backup_file"
        return 1
    fi
    
    log "🔄 Начало отката к резервной копии: $backup_file"
    
    # Создание резервной копии текущего состояния
    current_backup="./cleanup_backups/before_rollback_$(date +%Y%m%d_%H%M%S).tar.gz"
    tar -czf "$current_backup" . --exclude='cleanup_backups' --exclude='cleanup_logs'
    log "📦 Создана резервная копия текущего состояния: $current_backup"
    
    # Очистка текущего состояния (кроме служебных папок)
    find . -maxdepth 1 -type f -not -name ".*" -delete
    find . -maxdepth 1 -type d -not -name "." -not -name "cleanup_backups" -not -name "cleanup_logs" -not -name ".git" -exec rm -rf {} +
    
    # Восстановление из резервной копии
    tar -xzf "$backup_file"
    
    if [ $? -eq 0 ]; then
        log "✅ Откат выполнен успешно"
        return 0
    else
        log "❌ Ошибка при откате"
        return 1
    fi
}

# Функция для экстренного отката
emergency_rollback() {
    log "🚨 ЭКСТРЕННЫЙ ОТКАТ"
    rollback_to_backup
    
    if [ $? -eq 0 ]; then
        log "✅ Экстренный откат выполнен успешно"
        quick_system_test
    else
        log "❌ Экстренный откат не удался"
    fi
}
```

## 5. Процедуры валидации

### 5.1 Проверка целостности проекта

```bash
#!/bin/bash
# Проверка целостности проекта

validate_project_integrity() {
    log "=== ПРОВЕРКА ЦЕЛОСТНОСТИ ПРОЕКТА ==="
    
    # Проверка основных файлов
    CRITICAL_FILES=(
        "main.py"
        "config.py"
        "winrate_test_with_results2.py"
        "advanced_strategy_tester.py"
        "sequential_strategy_tester.py"
    )
    
    log "Проверка критических файлов..."
    for file in "${CRITICAL_FILES[@]}"; do
        if [ -f "$file" ]; then
            log "✅ $file - присутствует"
        else
            log "❌ $file - ОТСУТСТВУЕТ"
        fi
    done
    
    # Проверка директорий AI модулей
    AI_DIRS=(
        "ai_modules"
        "config"
        "database"
        "utils"
    )
    
    log "Проверка критических директорий..."
    for dir in "${AI_DIRS[@]}"; do
        if [ -d "$dir" ]; then
            log "✅ $dir/ - присутствует"
        else
            log "❌ $dir/ - ОТСУТСТВУЕТ"
        fi
    done
    
    # Проверка синтаксиса Python файлов
    log "Проверка синтаксиса Python файлов..."
    find . -name "*.py" -not -path "./cleanup_*" | while read file; do
        python -m py_compile "$file" 2>/dev/null
        if [ $? -eq 0 ]; then
            log "✅ $file - синтаксис корректен"
        else
            log "❌ $file - ОШИБКА СИНТАКСИСА"
        fi
    done
}

validate_project_integrity
```

### 5.2 Проверка производительности

```bash
#!/bin/bash
# Проверка производительности

measure_performance() {
    log "=== ИЗМЕРЕНИЕ ПРОИЗВОДИТЕЛЬНОСТИ ==="
    
    # Измерение времени запуска
    log "Измерение времени запуска основного модуля..."
    start_time=$(date +%s.%N)
    python -c "import main; print('Модуль загружен')" >/dev/null 2>&1
    end_time=$(date +%s.%N)
    startup_time=$(echo "$end_time - $start_time" | bc)
    log "⏱️ Время запуска: ${startup_time}s"
    
    # Измерение использования памяти
    log "Измерение использования памяти..."
    python -c "
import psutil
import os
import main

process = psutil.Process(os.getpid())
memory_mb = process.memory_info().rss / 1024 / 1024
print(f'💾 Использование памяти: {memory_mb:.1f} MB')
" 2>&1 | tee -a $LOG_FILE
    
    # Подсчет файлов проекта
    log "Подсчет файлов проекта..."
    total_files=$(find . -type f -name "*.py" | wc -l)
    total_size=$(du -sh . | cut -f1)
    log "📁 Общее количество Python файлов: $total_files"
    log "📦 Общий размер проекта: $total_size"
}

measure_performance
```

## 6. Финальные процедуры

### 6.1 Обновление документации

```bash
#!/bin/bash
# Обновление документации

update_documentation() {
    log "=== ОБНОВЛЕНИЕ ДОКУМЕНТАЦИИ ==="
    
    # Обновление .gitignore
    log "Обновление .gitignore..."
    if ! grep -q "cleanup_backups/" .gitignore; then
        echo "cleanup_backups/" >> .gitignore
        log "✅ Добавлена папка cleanup_backups в .gitignore"
    fi
    
    if ! grep -q "cleanup_logs/" .gitignore; then
        echo "cleanup_logs/" >> .gitignore
        log "✅ Добавлена папка cleanup_logs в .gitignore"
    fi
    
    # Создание отчета об очистке
    CLEANUP_REPORT="./cleanup_logs/cleanup_final_report_$(date +%Y%m%d_%H%M%S).md"
    cat > "$CLEANUP_REPORT" << EOF
# Отчет об очистке проекта

## Дата выполнения
$(date)

## Удаленные файлы
$(cat $LOG_FILE | grep "✅ Успешно удален" | wc -l) файлов удалено успешно

## Ошибки
$(cat $LOG_FILE | grep "❌" | wc -l) ошибок обнаружено

## Детальный лог
См. файл: $LOG_FILE

## Статистика проекта после очистки
- Общее количество Python файлов: $(find . -type f -name "*.py" | wc -l)
- Общий размер проекта: $(du -sh . | cut -f1)

## Резервные копии
Созданы резервные копии в папке: ./cleanup_backups/
EOF
    
    log "✅ Создан финальный отчет: $CLEANUP_REPORT"
}

update_documentation
```

### 6.2 Финальная валидация

```bash
#!/bin/bash
# Финальная валидация

final_validation() {
    log "=== ФИНАЛЬНАЯ ВАЛИДАЦИЯ ==="
    
    # Запуск всех проверок
    validate_project_integrity
    measure_performance
    quick_system_test
    
    # Создание сводного отчета
    log "=== СВОДНЫЙ ОТЧЕТ ==="
    log "Процедура очистки завершена: $(date)"
    
    # Подсчет результатов
    total_errors=$(cat $LOG_FILE | grep "❌" | wc -l)
    total_warnings=$(cat $LOG_FILE | grep "⚠️" | wc -l)
    total_success=$(cat $LOG_FILE | grep "✅" | wc -l)
    
    log "📊 СТАТИСТИКА:"
    log "   Успешных операций: $total_success"
    log "   Предупреждений: $total_warnings"
    log "   Ошибок: $total_errors"
    
    if [ $total_errors -eq 0 ]; then
        log "🎉 ОЧИСТКА ЗАВЕРШЕНА УСПЕШНО!"
        return 0
    else
        log "⚠️ ОЧИСТКА ЗАВЕРШЕНА С ОШИБКАМИ"
        return 1
    fi
}

final_validation
```

## 7. Главный скрипт выполнения

```bash
#!/bin/bash
# Главный скрипт выполнения очистки проекта

set -e  # Остановка при ошибке

# Импорт всех функций
source ./cleanup_procedures.sh

main() {
    log "🚀 НАЧАЛО ПРОЦЕДУРЫ ОЧИСТКИ ПРОЕКТА"
    
    # Этап 1: Подготовка
    log "📋 Этап 1: Подготовка"
    create_backup
    
    # Этап 2: Анализ
    log "🔍 Этап 2: Анализ зависимостей"
    # analyze_dependencies (выполняется вручную)
    
    # Этап 3: Удаление файлов
    log "🗑️ Этап 3: Удаление файлов"
    
    # Подэтап 3.1: Тестовые файлы
    log "📝 Подэтап 3.1: Удаление тестовых файлов"
    # stage1_test_files
    
    # Подэтап 3.2: Отладочные файлы
    log "🐛 Подэтап 3.2: Удаление отладочных файлов"
    # stage2_debug_files
    
    # Подэтап 3.3: Mock и demo файлы
    log "🎭 Подэтап 3.3: Удаление mock и demo файлов"
    # stage3_mock_demo_files
    
    # Подэтап 3.4: Устаревшие конфигурации
    log "⚙️ Подэтап 3.4: Удаление устаревших конфигураций"
    # stage4_old_configs
    
    # Этап 4: Тестирование
    log "🧪 Этап 4: Тестирование системы"
    quick_system_test
    full_system_test
    
    # Этап 5: Финализация
    log "📝 Этап 5: Финализация"
    update_documentation
    final_validation
    
    log "🎉 ПРОЦЕДУРА ОЧИСТКИ ЗАВЕРШЕНА!"
}

# Обработка ошибок
trap 'log "❌ КРИТИЧЕСКАЯ ОШИБКА! Выполняется экстренный откат..."; emergency_rollback' ERR

# Запуск главной функции
main "$@"
```

---

**Примечание**: Все скрипты должны выполняться из корневой директории проекта с соответствующими правами доступа.