#!/bin/bash

# Директория скрипта → корень проекта (родитель scripts/) → данные в project/data
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOCAL_PATH="$PROJECT_ROOT/data"

# Конфигурация
REMOTE_USER="root"
REMOTE_HOST="87.228.88.218"
REMOTE_PATH="/root/BinanceParser/data_BTCUSDC"

# Проверка наличия локальной директории
mkdir -p "$LOCAL_PATH"

# Загрузка данных
echo "📥 Начинаю скачивание data_BTCUSDC с $REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH"
echo "   → сохраняю в $LOCAL_PATH/data_BTCUSDC/"

rsync -avz "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}/" "${LOCAL_PATH}/data_BTCUSDC/"

if [ $? -eq 0 ]; then
    echo "✅ Успешно загружено в $LOCAL_PATH/data_BTCUSDC"
else
    echo "❌ Ошибка при скачивании!"
    exit 1
fi