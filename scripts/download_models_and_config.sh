#!/bin/bash

# Директория скрипта → корень проекта (родитель scripts/)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MODELS_DIR="$PROJECT_ROOT/models"

# Конфигурация
REMOTE_USER="root"
REMOTE_HOST="87.228.88.219"

REMOTE_MODEL_MID="/root/AlgoTradingSystem_c1/models/model_mid.json"
REMOTE_MODEL_SPREAD="/root/AlgoTradingSystem_c1/models/model_spread.json"
REMOTE_OPT_RESULTS="/root/AlgoTradingSystem_c1/opt_results.json"

# Локальные пути
LOCAL_MODEL_MID="$MODELS_DIR/model_mid.json"
LOCAL_MODEL_SPREAD="$MODELS_DIR/model_spread.json"
LOCAL_OPT_RESULTS="$PROJECT_ROOT/opt_results.json"

# Подготовка локальных директорий
mkdir -p "$MODELS_DIR"

echo "📥 Начинаю скачивание файлов с $REMOTE_USER@$REMOTE_HOST"

rsync -avz "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_MODEL_MID}" "$LOCAL_MODEL_MID"
if [ $? -ne 0 ]; then
    echo "❌ Ошибка при скачивании model_mid.json"
    exit 1
fi

rsync -avz "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_MODEL_SPREAD}" "$LOCAL_MODEL_SPREAD"
if [ $? -ne 0 ]; then
    echo "❌ Ошибка при скачивании model_spread.json"
    exit 1
fi

rsync -avz "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_OPT_RESULTS}" "$LOCAL_OPT_RESULTS"
if [ $? -ne 0 ]; then
    echo "❌ Ошибка при скачивании opt_results.json"
    exit 1
fi

echo "✅ Файлы успешно загружены:"
echo "   - $LOCAL_MODEL_MID"
echo "   - $LOCAL_MODEL_SPREAD"
echo "   - $LOCAL_OPT_RESULTS"
