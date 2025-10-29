#!/bin/bash
# restore_system.sh - Скрипт восстановления торговой системы из бэкапа

set -e  # Остановка при ошибке

BACKUP_ARCHIVE="$1"  # Путь к архиву бэкапа
RESTORE_DIR="/Users/mac/Documents/Peper Binance v4 Clean"

# Функция логирования
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

if [[ -z "$BACKUP_ARCHIVE" ]]; then
    echo "❌ Использование: $0 <путь_к_архиву_бэкапа>"
    echo "📝 Пример: $0 /Users/mac/Documents/Backups/Peper_Binance_v4_20241225_143022.tar.gz"
    exit 1
fi

if [[ ! -f "$BACKUP_ARCHIVE" ]]; then
    log "❌ ОШИБКА: Файл архива не найден: $BACKUP_ARCHIVE"
    exit 1
fi

log "🔄 Восстановление системы из: $BACKUP_ARCHIVE"

# Проверка контрольной суммы
ARCHIVE_DIR=$(dirname "$BACKUP_ARCHIVE")
ARCHIVE_NAME=$(basename "$BACKUP_ARCHIVE" .tar.gz)
if [[ -f "$ARCHIVE_DIR/${ARCHIVE_NAME}.sha256" ]]; then
    log "🔐 Проверка контрольной суммы..."
    cd "$ARCHIVE_DIR"
    if ! shasum -a 256 -c "${ARCHIVE_NAME}.sha256"; then
        log "❌ ОШИБКА: Контрольная сумма не совпадает!"
        exit 1
    fi
    log "✅ Контрольная сумма корректна"
fi

# Остановка текущей системы
log "⏹️  Остановка текущей торговой системы..."
pkill -f "python.*main.py" 2>/dev/null || true
pkill -f "python.*multi_ai" 2>/dev/null || true
sleep 2

# Создание резервной копии текущего состояния (если директория существует)
if [[ -d "$RESTORE_DIR" ]]; then
    CURRENT_BACKUP="/Users/mac/Documents/Peper Binance v4 Clean_CURRENT_$(date +%Y%m%d_%H%M%S)"
    log "💾 Создание резервной копии текущего состояния: $CURRENT_BACKUP"
    mv "$RESTORE_DIR" "$CURRENT_BACKUP"
fi

# Извлечение архива
log "📦 Извлечение архива..."
cd "/Users/mac/Documents"
tar -xzf "$BACKUP_ARCHIVE"

# Переименование в рабочую директорию
EXTRACTED_DIR=$(basename "$BACKUP_ARCHIVE" .tar.gz)
mv "$EXTRACTED_DIR" "Peper Binance v4 Clean"

log "✅ Система восстановлена в: $RESTORE_DIR"

# Проверка восстановленной системы
log "🔍 Проверка восстановленной системы..."

# Проверка критических файлов
CRITICAL_FILES=(
    "main.py"
    "config.py"
    "config_params.py"
    "ai_modules/ai_manager.py"
    "ai_modules/trading_ai.py"
    "models/BTCUSDT_trading_model.joblib"
    "config/config_manager.py"
)

for file in "${CRITICAL_FILES[@]}"; do
    if [[ ! -f "$RESTORE_DIR/$file" ]]; then
        log "❌ ОШИБКА: Отсутствует критический файл: $file"
        exit 1
    fi
    log "✅ $file"
done

# Проверка Python зависимостей
cd "$RESTORE_DIR"
if [[ -f "requirements.txt" ]]; then
    log "🐍 Проверка зависимостей Python..."
    python3 -m pip check 2>/dev/null || log "⚠️  Некоторые зависимости могут требовать переустановки"
fi

# Установка прав выполнения для скриптов
chmod +x "$RESTORE_DIR"/*.sh 2>/dev/null || true

log "🎉 Восстановление прошло успешно!"
echo ""
echo "✅ СИСТЕМА ВОССТАНОВЛЕНА!"
echo "📂 Директория: $RESTORE_DIR"
echo "🔄 Система готова к работе"