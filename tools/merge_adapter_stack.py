"""Merge a stack of LoRA adapters into a single 16-bit safetensors export.

Given two (or more) LoRA adapter directories, this loads them onto the
unquantized base, activates all simultaneously, and merge_and_unloads
the combined weights:

    W_merged = W_base + sum(ΔW_adapter_i for each adapter active)

The result is a single set of safetensors suitable for the standard
manual GGUF pipeline (convert_hf_to_gguf + llama-quantize Q4_K_M).

Usage:
    python tools/merge_adapter_stack.py \\
      --adapter v3.0-rslora/gguf/adapter \\
      --adapter v3.2/gguf/adapter_fp_recovery \\
      --out v3.2/gguf

The order of --adapter flags is the activation order. Mathematically
the final merged weights are commutative (sum is sum), but the merge
operation may be applied sequentially.
"""
import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "unsloth/qwen3-14b"  # unquantized base, cached locally


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", action="append", required=True,
                    help="path to a saved LoRA adapter dir; pass multiple times for stacking")
    ap.add_argument("--out", required=True, help="output directory for merged safetensors")
    args = ap.parse_args()

    if len(args.adapter) < 1:
        raise SystemExit("Need at least one --adapter")

    print(f"Loading unquantized base {BASE} (bf16)...")
    merged = AutoModelForCausalLM.from_pretrained(
        BASE, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
    )
    tok = AutoTokenizer.from_pretrained(BASE)

    # PEFT 0.19's PeftModel.set_adapter does not accept a list of names
    # (PeftMixedModel does, but not for our base PeftModel here). Instead
    # apply adapters one at a time, merging each into the base before
    # loading the next. Mathematically equivalent to summing all deltas:
    #     W_final = W_0 + ΔW_a + ΔW_b
    # since each merge step is W_i+1 = W_i + ΔW_i.
    for i, path in enumerate(args.adapter):
        name = f"adapter_{i}"
        print(f"[{i+1}/{len(args.adapter)}] Loading adapter ({path}) as '{name}'...")
        peft_model = PeftModel.from_pretrained(merged, path, adapter_name=name)
        print(f"[{i+1}/{len(args.adapter)}] merge_and_unload (W_{i+1} = W_{i} + ΔW)...")
        merged = peft_model.merge_and_unload()
        del peft_model

    out_path = Path(args.out)
    out_path.mkdir(parents=True, exist_ok=True)
    print(f"Saving merged safetensors to {out_path} (max_shard_size=5GB)...")
    merged.save_pretrained(
        str(out_path),
        safe_serialization=True,
        max_shard_size="5GB",
    )
    tok.save_pretrained(str(out_path))
    print("Done.")


if __name__ == "__main__":
    main()
