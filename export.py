#!/usr/bin/env python3
"""Phase 4: Export fine-tuned model to GGUF for Ollama.

Uses Unsloth's built-in save_pretrained_gguf() which handles
merge + convert + quantize in one call. No llama.cpp needed.

Usage:
    python3 export.py --model qwen3-14b
    python3 export.py --model qwen3-14b --quant q8_0
    python3 export.py --model gemma4-31b --quant q6_k
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("export")

ROOT = Path(__file__).parent

MODELS = {
    "qwen3-14b": {"default_quant": "q8_0"},
    "gemma4-31b": {"default_quant": "q6_k"},
}


def export(model_key: str, quant: str):
    lora_dir = ROOT / f"{model_key}-typescript-lora"
    gguf_dir = ROOT / "gguf"

    if not lora_dir.exists():
        log.error("LoRA adapter not found at %s. Run train.py first.", lora_dir)
        sys.exit(1)

    gguf_dir.mkdir(exist_ok=True)

    log.info("=== Export: %s (quantization: %s) ===", model_key, quant)

    from unsloth import FastLanguageModel

    # Load the LoRA adapter (Unsloth merges automatically during GGUF export)
    log.info("Loading LoRA adapter from %s...", lora_dir)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(lora_dir),
        load_in_4bit=True,
    )

    # Export to GGUF — Unsloth handles merge + convert + quantize
    output_dir = str(gguf_dir / f"{model_key}-typescript")
    log.info("Exporting to GGUF (%s) at %s...", quant, output_dir)
    model.save_pretrained_gguf(
        output_dir,
        tokenizer,
        quantization_method=quant,
    )

    log.info("=== Export complete ===")
    log.info("GGUF files at: %s", output_dir)
    log.info("Next: ollama create ts-forge -f <Modelfile>")
    log.info("Unsloth auto-generates a Modelfile with the correct chat template.")


def main():
    parser = argparse.ArgumentParser(description="Export fine-tuned model to GGUF")
    parser.add_argument("--model", required=True, choices=list(MODELS.keys()))
    parser.add_argument("--quant", help="Quantization method (default: model-specific)")
    args = parser.parse_args()

    quant = args.quant or MODELS[args.model]["default_quant"]
    export(args.model, quant)


if __name__ == "__main__":
    main()
