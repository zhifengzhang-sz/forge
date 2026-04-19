"""Minimum-viable v0 trainer. Uses the project's existing venv (Unsloth installed).

    .venv/bin/python v0/train_v0.py --data v0/data/xstate_curated.jsonl --out v0/gguf/curated
    .venv/bin/python v0/train_v0.py --data v0/data/xstate_extracted.jsonl --out v0/gguf/extracted

Forgetting-protective hyperparams (vs main pipeline): r=16, lr=1e-4, 2 epochs.

DEVIATION FROM solution.md: uses qwen3-14b instead of Qwen3-Coder-30B-A3B because
the Coder MoE 4-bit import path has known issues per Unsloth docs and qwen3-14b
is already cached locally (~11 GB). v0 tests whether the *technique* works; the
model-choice question is deferred to v1.
"""

import argparse
from pathlib import Path

from unsloth import FastLanguageModel  # must import unsloth first for patching
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer

MODEL = "unsloth/qwen3-14b-unsloth-bnb-4bit"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--quant", default="q4_k_m")
    args = ap.parse_args()

    model, tok = FastLanguageModel.from_pretrained(
        MODEL, max_seq_length=4096, load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=16, lora_alpha=16, lora_dropout=0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth",
    )

    ds = load_dataset("json", data_files=args.data, split="train")
    ds = ds.map(lambda x: {"text": tok.apply_chat_template(x["messages"], tokenize=False)})

    SFTTrainer(
        model=model, tokenizer=tok, train_dataset=ds,
        args=SFTConfig(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            num_train_epochs=2,
            learning_rate=1e-4,
            bf16=True, logging_steps=5,
            output_dir=str(Path(args.out) / "trainer"),
            save_strategy="no",
        ),
    ).train()

    Path(args.out).mkdir(parents=True, exist_ok=True)
    model.save_pretrained_gguf(args.out, tok, quantization_method=args.quant)
    print(f"Saved GGUF to {args.out}")


if __name__ == "__main__":
    main()
