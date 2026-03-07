#!/usr/bin/env bash
set -euo pipefail

# ============================================================
#  Скрипт установки зависимостей PP_ML_PLATFORM
# ============================================================

PADDLE_VERSION="3.2.0"
PADDLE_CPU_INDEX="https://www.paddlepaddle.org.cn/packages/stable/cpu/"
PADDLE_CU118_INDEX="https://www.paddlepaddle.org.cn/packages/stable/cu118/"
PADDLE_CU126_INDEX="https://www.paddlepaddle.org.cn/packages/stable/cu126/"
PADDLEOCR_URL="paddleocr[all]"

TORCH_VERSION="2.9.0"
TORCH_CPU_INDEX="https://download.pytorch.org/whl/cpu"
TORCH_CU118_INDEX="https://download.pytorch.org/whl/cu118"
TORCH_CU126_INDEX="https://download.pytorch.org/whl/cu126"

# ---- Определение режима GPU --------------------------------

detect_gpu_mode() {
    if ! command -v nvidia-smi &>/dev/null; then
        echo "cpu"
        return
    fi

    local driver_version
    driver_version=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
    if [[ -z "$driver_version" ]]; then
        echo "cpu"
        return
    fi

    local major="${driver_version%%.*}"
    if (( major >= 550 )); then
        echo "cu126"
    elif (( major >= 450 )); then
        echo "cu118"
    else
        echo "cpu"
    fi
}

# ---- Основной блок ------------------------------------------

MODE="${1:-auto}"   # cpu | cu118 | cu126 | auto

if [[ "$MODE" == "auto" ]]; then
    MODE=$(detect_gpu_mode)
    echo "Автоопределение: mode=$MODE"
fi

echo ""
echo "=== Установка PyTorch ==="

case "$MODE" in
    cpu)
        echo "Установка torch + torchaudio (CPU)..."
        python -m pip install "torch==${TORCH_VERSION}" "torchaudio" --index-url "$TORCH_CPU_INDEX"
        ;;
    cu118)
        echo "Установка torch + torchaudio (CUDA 11.8)..."
        python -m pip install "torch==${TORCH_VERSION}" "torchaudio" --index-url "$TORCH_CU118_INDEX"
        ;;
    cu126)
        echo "Установка torch + torchaudio (CUDA 12.6)..."
        python -m pip install "torch==${TORCH_VERSION}" "torchaudio" --index-url "$TORCH_CU126_INDEX"
        ;;
esac

echo ""
echo "=== Установка PaddlePaddle ==="

case "$MODE" in
    cpu)
        echo "Установка paddlepaddle (CPU)..."
        python -m pip install "paddlepaddle==${PADDLE_VERSION}" -i "$PADDLE_CPU_INDEX"
        ;;
    cu118)
        echo "Установка paddlepaddle-gpu (CUDA 11.8, driver ≥450.80.02)..."
        python -m pip install "paddlepaddle-gpu==${PADDLE_VERSION}" -i "$PADDLE_CU118_INDEX"
        ;;
    cu126)
        echo "Установка paddlepaddle-gpu (CUDA 12.6, driver ≥550.54.14)..."
        python -m pip install "paddlepaddle-gpu==${PADDLE_VERSION}" -i "$PADDLE_CU126_INDEX"
        ;;
    *)
        echo "Неизвестный режим: $MODE"
        echo "Использование: $0 [cpu|cu118|cu126|auto]"
        exit 1
        ;;
esac

echo ""
echo "=== Установка PaddleOCR ==="
python -m pip install "$PADDLEOCR_URL"

echo "=== Установка основных зависимостей ==="
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo ""
echo "=== Готово ==="