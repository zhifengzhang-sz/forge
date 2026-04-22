"""v1 trainer. Fork of v0.7/train.py with v1 defaults.

    .venv/bin/python v1/train.py

Deltas vs v0.7:
  - Default rank locked to 64 (v0.7 winner; r=128 deferred per review).
  - Data path defaults to v1/data/synth.verified.jsonl (4885 records
    including 600 pre-expanded anchor records, 30 unique × 20 reps).
  - Anchor mixing is DISABLED by default — merge.py already included
    the 600 anchor records, so passing --anchor-ratio here would
    double-mix. If you somehow feed primary-only data, pass
    --anchor-ratio 0.12 explicitly.
"""

import argparse
import json
import random
from pathlib import Path

from unsloth import FastLanguageModel  # must import unsloth first for patching
from datasets import Dataset
from trl import SFTConfig, SFTTrainer

MODEL = "unsloth/qwen3-14b-unsloth-bnb-4bit"
REPO_ROOT = Path(__file__).resolve().parent.parent
ANCHORS_SRC = REPO_ROOT / "v1/seeds/anchors.jsonl"


def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def mix_anchors(primary: list[dict], anchors: list[dict], ratio: float, seed: int) -> list[dict]:
    if ratio <= 0:
        return primary
    n_target = round(len(primary) * ratio / (1 - ratio))
    reps = max(1, -(-n_target // len(anchors)))
    pool = (anchors * reps)[:n_target]
    mixed = primary + pool
    random.Random(seed).shuffle(mixed)
    return mixed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="v1/data/synth.verified.jsonl")
    ap.add_argument("--out", default="v1/gguf")
    ap.add_argument("--quant", default="q4_k_m")
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument(
        "--anchor-ratio", type=float, default=0.0,
        help="Default 0 because merge.py already mixed in 600 anchor records. "
             "Pass >0 only if feeding primary-only data.",
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    primary = load_jsonl(Path(args.data))
    if args.anchor_ratio > 0:
        anchors = load_jsonl(ANCHORS_SRC)
        records = mix_anchors(primary, anchors, args.anchor_ratio, args.seed)
        print(f"[train] primary={len(primary)}  anchors_mixed={len(records)-len(primary)}  total={len(records)}  rank={args.rank}")
    else:
        records = primary
        print(f"[train] primary={len(primary)} (anchors already included)  rank={args.rank}")

    model, tok = FastLanguageModel.from_pretrained(
        MODEL, max_seq_length=4096, load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.rank, lora_alpha=args.rank, lora_dropout=0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth",
    )

    # Strip "domain" field so only "messages" is passed to the chat template
    clean = [{"messages": r["messages"]} for r in records]
    ds = Dataset.from_list(clean)
    ds = ds.map(lambda x: {"text": tok.apply_chat_template(x["messages"], tokenize=False)})

    SFTTrainer(
        model=model, tokenizer=tok, train_dataset=ds,
        args=SFTConfig(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            num_train_epochs=args.epochs,
            learning_rate=1e-4,
            bf16=True, logging_steps=5,
            output_dir=str(Path(args.out) / "trainer"),
            save_strategy="no",
            seed=args.seed,
        ),
    ).train()

    Path(args.out).mkdir(parents=True, exist_ok=True)
    model.save_pretrained_gguf(args.out, tok, quantization_method=args.quant)
    print(f"Saved GGUF to {args.out}")


if __name__ == "__main__":
    main()
