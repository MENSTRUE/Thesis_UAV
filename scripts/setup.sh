#!/usr/bin/env bash
# Setup Linux/macOS. Jalankan dari root proyek.
#   ./scripts/setup.sh            -> default: GPU nVidia (onnxruntime-gpu CUDA)
#   ./scripts/setup.sh -CPU       -> ONNX Runtime CPU saja
#   ./scripts/setup.sh -DML       -> ONNX Runtime DirectML
#   ./scripts/setup.sh -Pip       -> pakai pip, bukan uv
set -euo pipefail
cd "$(dirname "$0")/.."
export UV_LINK_MODE=copy  # hindari warning hardlink antar filesystem

CPU=0
DML=0
PIP=0
for arg in "$@"; do
    case "$arg" in
        -CPU) CPU=1 ;;
        -DML) DML=1 ;;
        -Pip) PIP=1 ;;
        *) echo "Argumen tak dikenal: $arg" >&2; exit 1 ;;
    esac
done

if [ ! -f .python-version ]; then
    echo ".python-version tidak ditemukan" >&2
    exit 1
fi

if [ "$PIP" -eq 0 ]; then
    if ! command -v uv >/dev/null 2>&1; then
        echo "[uv] belum ada, menginstall..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.local/bin:$PATH"
    fi
    UV="uv"
else
    UV=""
    if ! command -v python3 >/dev/null 2>&1; then
        echo "python3 belum terpasang" >&2
        exit 1
    fi
fi

if [ "$PIP" -eq 1 ]; then
    python3 -m venv .venv
    VENV=".venv/bin"
else
    uv venv --python "$(cat .python-version)" .venv
    VENV=".venv/bin"
fi

echo "[deps] Base (HAR + Tello)..."
if [ "$PIP" -eq 1 ]; then
    "$VENV/python" -m pip install --upgrade pip
    "$VENV/python" -m pip install -r requirements/requirements-har.txt
else
    uv pip install -p .venv -r requirements/requirements-har.txt
fi

if [ "$DML" -eq 1 ]; then
    echo "[deps] DirectML..."
    VARIANT="dml"
elif [ "$CPU" -eq 1 ]; then
    echo "[deps] CPU..."
    VARIANT="cpu"
else
    echo "[deps] GPU (nvidia)..."
    VARIANT="gpu"
fi
if [ "$PIP" -eq 1 ]; then
    "$VENV/python" -m pip install -r requirements/requirements-full.txt -r "requirements/requirements-$VARIANT.txt"
else
    uv pip install -p .venv -r requirements/requirements-full.txt -r "requirements/requirements-$VARIANT.txt"
fi

echo ""
echo "Selesai. Jalankan:"
echo "  $VENV/python main.py"