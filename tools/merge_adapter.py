"""Recover merged 16-bit safetensors from a saved LoRA adapter.

Needed because Unsloth's save_pretrained_merged silently skips the
disk write when HF_HUB_OFFLINE=1 is set and the base-model lookup in
determine_base_model_source returns None (unsloth_zoo/saving_utils.py
line 1891). This script loads the saved adapter on top of the
unquantized base (cached locally) and emits merged 16-bit safetensors
for the manual GGUF pipeline.

Usage:
  python v2/merge_adapter.py <adapter_dir> <out_dir>

Example:
  python v2/merge_adapter.py v2/stack/fp/gguf/adapter v2/stack/fp/gguf

Run without HF_HUB_OFFLINE so the cache-aware HF loader can resolve
the non-quantized base. Network is not actually used — the 28 GB
snapshot is already local.
"""
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "unsloth/qwen3-14b"  # unquantized 16-bit base (28 GB, cached locally)


def main() -> None:
    if len(sys.argv) != 3:
        sys.stderr.write(f"Usage: {sys.argv[0]} <adapter_dir> <out_dir>\n")
        sys.exit(2)
    adapter = sys.argv[1]
    out = sys.argv[2]

    print(f"Loading unquantized base {BASE} (bf16)...")
    base = AutoModelForCausalLM.from_pretrained(
        BASE, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
    )
    tok = AutoTokenizer.from_pretrained(BASE)

    print(f"Attaching LoRA adapter from {adapter}...")
    model = PeftModel.from_pretrained(base, adapter)

    print("Merging LoRA into base weights (merge_and_unload)...")
    merged = model.merge_and_unload()

    out_path = Path(out)
    out_path.mkdir(parents=True, exist_ok=True)
    print(f"Saving merged safetensors to {out_path}/ (max_shard_size=5GB)...")
    merged.save_pretrained(
        str(out_path),
        safe_serialization=True,
        max_shard_size="5GB",
    )
    tok.save_pretrained(str(out_path))
    print("Done.")


if __name__ == "__main__":
    main()
