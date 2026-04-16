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

# Step 1: Python venv
if [ ! -d ".venv" ]; then
    echo "--- Creating Python venv ---"
    python3 -m venv .venv
    .venv/bin/pip install --upgrade pip
else
    echo "--- Python venv exists ---"
fi

# Step 2: Install dependencies
echo "--- Installing dependencies ---"
.venv/bin/pip install -r requirements.txt

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

print('  Downloading unsloth/Qwen3-14B-unsloth-bnb-4bit...')
snapshot_download('unsloth/Qwen3-14B-unsloth-bnb-4bit', local_files_only=False)
print('  Done.')

print('  Downloading unsloth/gemma-4-31b-it-unsloth-bnb-4bit...')
snapshot_download('unsloth/gemma-4-31b-it-unsloth-bnb-4bit', local_files_only=False)
print('  Done.')
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
