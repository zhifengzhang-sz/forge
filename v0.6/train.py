"""v0.6 trainer. Adapted from v0/train_v0.py.

    .venv/bin/python v0.6/train.py --data v0.6/data/synth.verified.jsonl --out v0.6/gguf

Deltas vs v0: r=32 (was 16), 3 epochs (was 2), +10% capability anchors
mixed in from v0/data/xstate_curated.jsonl indices 35-42 to preserve
tool-calling and general reasoning.
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
ANCHORS_SRC = REPO_ROOT / "v0/data/xstate_curated.jsonl"
ANCHOR_INDICES = range(35, 43)  # 8 non-XState capability anchors


def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_anchors() -> list[dict]:
    all_curated = load_jsonl(ANCHORS_SRC)
    return [all_curated[i] for i in ANCHOR_INDICES]


def mix_anchors(xstate: list[dict], anchors: list[dict], ratio: float, seed: int) -> list[dict]:
    n_target = round(len(xstate) * ratio / (1 - ratio))
    reps = max(1, -(-n_target // len(anchors)))  # ceil div
    pool = (anchors * reps)[:n_target]
    mixed = xstate + pool
    random.Random(seed).shuffle(mixed)
    return mixed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="v0.6/data/synth.verified.jsonl")
    ap.add_argument("--out", default="v0.6/gguf")
    ap.add_argument("--quant", default="q4_k_m")
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--anchor-ratio", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    xstate = load_jsonl(Path(args.data))
    anchors = load_anchors()
    records = mix_anchors(xstate, anchors, args.anchor_ratio, args.seed)
    print(f"[train] xstate={len(xstate)}  anchors_mixed_in={len(records)-len(xstate)}  total={len(records)}")

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

    ds = Dataset.from_list(records)
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
