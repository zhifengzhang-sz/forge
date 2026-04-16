#!/bin/bash
# Phase 4: Merge LoRA, convert to GGUF, quantise.
#
# Usage:
#     bash export.sh qwen3-14b
#     bash export.sh gemma4-31b
#
# Prerequisites:
#     - LoRA adapter at ./{model}-typescript-lora/
#     - llama.cpp cloned at ./llama.cpp/
#     - For gemma4-31b: close Ollama first (merge needs ~62 GB RAM)

set -euo pipefail

MODEL_KEY="${1:?Usage: bash export.sh <qwen3-14b|gemma4-31b>}"

case "$MODEL_KEY" in
    qwen3-14b)
        OUTTYPE="bf16"
        QUANT="Q8_0"
        ;;
    gemma4-31b)
        OUTTYPE="f16"
        QUANT="Q6_K"
        ;;
    *)
        echo "Unknown model: $MODEL_KEY. Use qwen3-14b or gemma4-31b."
        exit 1
        ;;
esac

LORA_DIR="./${MODEL_KEY}-typescript-lora"
MERGED_DIR="./${MODEL_KEY}-typescript-merged"
GGUF_DIR="./gguf"
GGUF_F16="${GGUF_DIR}/${MODEL_KEY}-typescript-${OUTTYPE}.gguf"
GGUF_QUANT="${GGUF_DIR}/${MODEL_KEY}-typescript-$(echo $QUANT | tr '[:upper:]' '[:lower:]').gguf"

echo "=== Export: ${MODEL_KEY} ==="
echo "  LoRA adapter:  ${LORA_DIR}"
echo "  Output type:   ${OUTTYPE}"
echo "  Quantisation:  ${QUANT}"
echo ""

# Check prerequisites
if [ ! -d "$LORA_DIR" ]; then
    echo "ERROR: LoRA adapter not found at ${LORA_DIR}. Run train.py first."
    exit 1
fi

# Check RAM for 31B merge
if [ "$MODEL_KEY" = "gemma4-31b" ]; then
    FREE_GB=$(free -g | awk '/^Mem:/ {print $7}')
    if [ "$FREE_GB" -lt 70 ]; then
        echo "WARNING: Only ${FREE_GB} GB RAM available. The 31B merge needs ~62 GB."
        echo "Close Ollama and other memory-intensive processes:"
        echo "  sudo systemctl stop ollama"
        read -p "Continue anyway? [y/N] " confirm
        [ "$confirm" = "y" ] || exit 1
    fi
fi

# Step 1: Merge LoRA adapter
echo "--- Step 1: Merge LoRA adapter ---"
python3 -c "
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name='${LORA_DIR}',
    load_in_4bit=False,
)
model.save_pretrained_merged(
    '${MERGED_DIR}',
    tokenizer,
    save_method='merged_16bit',
)
print('Merge complete: ${MERGED_DIR}')
"

# Step 2: Convert to GGUF
echo "--- Step 2: Convert to GGUF ---"
if [ ! -d "./llama.cpp" ]; then
    echo "Cloning llama.cpp..."
    git clone --depth=1 https://github.com/ggml-org/llama.cpp
    cd llama.cpp && pip install -r requirements.txt && cd ..
fi

mkdir -p "$GGUF_DIR"
python3 llama.cpp/convert-hf-to-gguf.py \
    "$MERGED_DIR" \
    --outfile "$GGUF_F16" \
    --outtype "$OUTTYPE"

echo "GGUF created: ${GGUF_F16}"

# Step 3: Quantise
echo "--- Step 3: Quantise to ${QUANT} ---"
if [ ! -f "./llama.cpp/build/bin/llama-quantize" ]; then
    echo "Building llama.cpp..."
    cd llama.cpp && cmake -B build -DGGML_CUDA=ON && cmake --build build --config Release -j$(nproc) && cd ..
fi

./llama.cpp/build/bin/llama-quantize "$GGUF_F16" "$GGUF_QUANT" "$QUANT"
echo "Quantised: ${GGUF_QUANT}"

# Step 4: Cleanup intermediate files
echo "--- Step 4: Cleanup ---"
rm -rf "$MERGED_DIR"
rm -f "$GGUF_F16"
echo "Removed: ${MERGED_DIR}, ${GGUF_F16}"

echo ""
echo "=== Export complete ==="
echo "  Final GGUF: ${GGUF_QUANT}"
echo "  Next: ollama create ts-forge -f Modelfile"
