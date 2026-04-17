#!/bin/bash
# One-time setup: create venv, install dependencies, download models.
#
# Usage:
#   bash setup.sh              # full setup
#   bash setup.sh --deps-only  # skip model downloads
#
# Prerequisites:
#   - Python 3.11+ with python3-venv
#   - NVIDIA GPU with CUDA 12.x
#   - Ollama 0.20+ (for inference)
#   - ~20 GB disk for HuggingFace model cache
#   - ~30 GB disk for Ollama models

set -euo pipefail
cd "$(dirname "$0")"

DEPS_ONLY=false
[ "${1:-}" = "--deps-only" ] && DEPS_ONLY=true

echo "=== Forge Setup ==="
echo ""

# Step 0: Check system dependencies
echo "--- Checking system dependencies ---"
MISSING=""

python3 --version >/dev/null 2>&1 || MISSING="$MISSING python3"
python3 -c "import venv" 2>/dev/null || MISSING="$MISSING python3.13-venv"
[ -f /usr/include/python3.13/Python.h ] || MISSING="$MISSING python3.13-dev"
gcc --version >/dev/null 2>&1 || MISSING="$MISSING gcc"
cmake --version >/dev/null 2>&1 || MISSING="$MISSING cmake"
[ -f /usr/include/curl/curl.h ] || MISSING="$MISSING libcurl4-openssl-dev"
[ -f /usr/include/openssl/ssl.h ] || MISSING="$MISSING libssl-dev"
nvidia-smi >/dev/null 2>&1 || MISSING="$MISSING nvidia-driver"

if [ -n "$MISSING" ]; then
    echo "ERROR: Missing system packages:$MISSING"
    echo ""
    echo "Install with:"
    echo "  sudo apt install -y$MISSING"
    echo ""
    echo "See docs/setup.pitfalls.md for details."
    exit 1
fi
echo "  All system dependencies OK"
echo ""

# Step 1: Python venv
if [ ! -d ".venv" ]; then
    echo "--- Creating Python venv ---"
    python3 -m venv .venv
    .venv/bin/pip install --upgrade pip
else
    echo "--- Python venv exists ---"
fi

# Step 2: Install dependencies (staged to avoid version conflicts)
echo "--- Installing PyTorch with CUDA ---"
.venv/bin/pip install torch==2.11.0+cu128 torchvision==0.26.0+cu128 triton==3.6.0 \
    --extra-index-url https://download.pytorch.org/whl/cu128

echo "--- Installing HuggingFace stack ---"
.venv/bin/pip install transformers==5.5.0 trl==0.24.0 datasets==4.3.0 \
    peft==0.19.1 accelerate==1.13.0 bitsandbytes==0.49.2 gguf

echo "--- Installing Unsloth (--no-deps to avoid torch version conflict) ---"
.venv/bin/pip install --no-deps unsloth_zoo==2026.4.8
.venv/bin/pip install --no-deps "unsloth @ git+https://github.com/unslothai/unsloth.git@d20b3067"

# Verify
echo ""
echo "--- Verifying installation ---"
.venv/bin/python3 -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available'
print(f'  PyTorch {torch.__version__}, CUDA: {torch.cuda.get_device_name(0)}')
import unsloth
print(f'  Unsloth {unsloth.__version__}')
import trl
print(f'  TRL {trl.__version__}')
print('  All dependencies OK')
"

if [ "$DEPS_ONLY" = true ]; then
    echo ""
    echo "=== Setup complete (deps only) ==="
    exit 0
fi

# Step 3: Download HuggingFace models (for training)
echo ""
echo "--- Downloading HuggingFace models for training ---"
echo "  These are cached at ~/.cache/huggingface/ (~10 GB each)"
echo ""

.venv/bin/python3 -c "
from huggingface_hub import snapshot_download

print('  Downloading unsloth/Qwen3-14B (resolves to bnb-4bit variant)...')
snapshot_download('unsloth/Qwen3-14B-unsloth-bnb-4bit', local_files_only=False)
print('  Done.')

print('  Downloading unsloth/gemma-4-31b-it (resolves to bnb-4bit variant)...')
snapshot_download('unsloth/gemma-4-31b-it-unsloth-bnb-4bit', local_files_only=False)
print('  Done.')

# Note: train.py uses short names (unsloth/Qwen3-14B, unsloth/gemma-4-31b-it).
# Unsloth internally resolves these to the -unsloth-bnb-4bit variants cached above.
"

# Step 4: Pull Ollama models (for inference)
echo ""
echo "--- Pulling Ollama models for inference ---"

if command -v ollama &>/dev/null; then
    OLLAMA_VER=$(ollama --version 2>&1 | grep -oP '[\d.]+' | head -1)
    echo "  Ollama version: $OLLAMA_VER"

    echo "  Pulling qwen3:14b..."
    ollama pull qwen3:14b

    echo "  Pulling gemma4:31b..."
    ollama pull gemma4:31b
else
    echo "  WARNING: Ollama not installed. Install from https://ollama.com/download"
    echo "  Then run: ollama pull qwen3:14b && ollama pull gemma4:31b"
fi

echo ""
echo "=== Setup complete ==="
echo ""
echo "Next steps:"
echo "  source .venv/bin/activate"
echo "  python3 extract_pipeline.py --skip-judge --skip-instruct  # extract data"
echo "  python3 train.py --model qwen3-14b                       # train"
