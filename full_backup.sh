#!/bin/bash
# full_backup.sh - Главный скрипт для создания полного бэкапа торговой системы Peper Binance v4

set -e  # Остановка при ошибке

# Конфигурация
SOURCE_DIR="/Users/mac/Documents/Peper Binance v4 Clean"
BACKUP_BASE_DIR="/Users/mac/Documents/Backups"
BACKUP_DATE=$(date +"%Y%m%d_%H%M%S")
BACKUP_DIR="$BACKUP_BASE_DIR/Peper_Binance_v4_$BACKUP_DATE"
LOG_FILE="$BACKUP_BASE_DIR/backup_log_$BACKUP_DATE.txt"

# Функция логирования
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Проверка существования исходной директории
if [[ ! -d "$SOURCE_DIR" ]]; then
    log "ОШИБКА: Исходная директория не найдена: $SOURCE_DIR"
    exit 1
fi

# Создание директории для бэкапов
mkdir -p "$BACKUP_BASE_DIR"

log "🚀 Начало создания бэкапа торговой системы Peper Binance v4"
log "📂 Источник: $SOURCE_DIR"
log "📁 Назначение: $BACKUP_DIR"

# Остановка системы
log "⏹️  Остановка торговой системы..."
pkill -f "python.*main.py" 2>/dev/null || true
pkill -f "python.*multi_ai" 2>/dev/null || true
sleep 2

# Создание директории бэкапа
mkdir -p "$BACKUP_DIR"

# Копирование критических компонентов
log "🔥 Копирование критических компонентов..."
cp -r "$SOURCE_DIR/ai_modules" "$BACKUP_DIR/" && log "✅ ai_modules скопированы"
cp -r "$SOURCE_DIR/models" "$BACKUP_DIR/" && log "✅ models скопированы"
cp -r "$SOURCE_DIR/config" "$BACKUP_DIR/" && log "✅ config скопированы"
cp "$SOURCE_DIR/main.py" "$BACKUP_DIR/" && log "✅ main.py скопирован"
cp "$SOURCE_DIR/config.py" "$BACKUP_DIR/" && log "✅ config.py скопирован"
cp "$SOURCE_DIR/config_params.py" "$BACKUP_DIR/" && log "✅ config_params.py скопирован"

# Копирование важных компонентов
log "📋 Копирование важных компонентов..."
cp -r "$SOURCE_DIR/database" "$BACKUP_DIR/" 2>/dev/null && log "✅ database скопирована" || log "⚠️  database не найдена"
cp -r "$SOURCE_DIR/analytics" "$BACKUP_DIR/" 2>/dev/null && log "✅ analytics скопирована" || log "⚠️  analytics не найдена"
cp -r "$SOURCE_DIR/utils" "$BACKUP_DIR/" 2>/dev/null && log "✅ utils скопированы" || log "⚠️  utils не найдены"
cp -r "$SOURCE_DIR/.trae" "$BACKUP_DIR/" 2>/dev/null && log "✅ .trae скопирована" || log "⚠️  .trae не найдена"
cp "$SOURCE_DIR/requirements.txt" "$BACKUP_DIR/" 2>/dev/null && log "✅ requirements.txt скопирован" || log "⚠️  requirements.txt не найден"

# Копирование дополнительных файлов
log "📄 Копирование дополнительных файлов..."
cp "$SOURCE_DIR"/*.py "$BACKUP_DIR/" 2>/dev/null || true
cp "$SOURCE_DIR"/*.md "$BACKUP_DIR/" 2>/dev/null || true
cp "$SOURCE_DIR"/*.txt "$BACKUP_DIR/" 2>/dev/null || true
cp "$SOURCE_DIR"/*.json "$BACKUP_DIR/" 2>/dev/null || true

# Создание манифеста
log "📝 Создание манифеста бэкапа..."
cat > "$BACKUP_DIR/backup_manifest.txt" << EOF
🎯 Бэкап торговой системы Peper Binance v4
📅 Дата создания: $BACKUP_DATE
📂 Источник: $SOURCE_DIR
🔢 Версия системы: v4

🔥 Критические компоненты:
- ai_modules/ ($(ls -1 "$BACKUP_DIR/ai_modules" 2>/dev/null | wc -l) файлов)
- models/ ($(ls -1 "$BACKUP_DIR/models" 2>/dev/null | wc -l) файлов)
- config/ ($(ls -1 "$BACKUP_DIR/config" 2>/dev/null | wc -l) файлов)
- main.py, config.py, config_params.py

📊 Размер бэкапа: $(du -sh "$BACKUP_DIR" | cut -f1)
📋 Лог бэкапа: $LOG_FILE
EOF

# Создание архива
log "🗜️  Создание сжатого архива..."
cd "$BACKUP_BASE_DIR"
tar -czf "Peper_Binance_v4_$BACKUP_DATE.tar.gz" "Peper_Binance_v4_$BACKUP_DATE/"
log "✅ Архив создан: Peper_Binance_v4_$BACKUP_DATE.tar.gz"

# Создание контрольной суммы
log "🔐 Создание контрольной суммы..."
shasum -a 256 "Peper_Binance_v4_$BACKUP_DATE.tar.gz" > "Peper_Binance_v4_$BACKUP_DATE.sha256"
log "✅ Контрольная сумма создана"

# Валидация бэкапа
log "🔍 Валидация бэкапа..."
ARCHIVE_SIZE=$(ls -lh "Peper_Binance_v4_$BACKUP_DATE.tar.gz" | awk '{print $5}')
log "📊 Размер архива: $ARCHIVE_SIZE"

# Проверка критических файлов в бэкапе
CRITICAL_FILES=(
    "main.py"
    "config.py"
    "ai_modules/trading_ai.py"
    "models/BTCUSDT_trading_model.joblib"
)

for file in "${CRITICAL_FILES[@]}"; do
    if [[ -f "$BACKUP_DIR/$file" ]]; then
        log "✅ $file присутствует в бэкапе"
    else
        log "❌ ОШИБКА: $file отсутствует в бэкапе"
        exit 1
    fi
done

log "🎉 Бэкап успешно создан и проверен"
log "📁 Архив: $BACKUP_BASE_DIR/Peper_Binance_v4_$BACKUP_DATE.tar.gz"
log "📊 Размер: $ARCHIVE_SIZE"
log "🔐 Контрольная сумма: $BACKUP_BASE_DIR/Peper_Binance_v4_$BACKUP_DATE.sha256"

echo ""
echo "🎉 БЭКАП ЗАВЕРШЕН УСПЕШНО!"
echo "📁 Архив: Peper_Binance_v4_$BACKUP_DATE.tar.gz"
echo "📊 Размер: $ARCHIVE_SIZE"
echo "📋 Лог: $LOG_FILE"
echo ""
echo "🔄 Теперь можно безопасно приступать к реализации плана улучшений!"