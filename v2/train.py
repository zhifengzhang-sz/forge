"""v2.0 trainer. Fresh-from-base joint multi-task — v0.7's proven recipe.

    .venv/bin/python v2/train.py

Delta vs v1.2: reverts warm-start (MODEL back to raw base) and points at
v2/data/synth.verified.jsonl (2604 atomic-drill records — see v2/merge.py
and docs/v2.plan.md). Everything else identical to v1/v1.1/v1.2:
SFTTrainer, bnb-4bit, 7 target modules, r=64, epochs=3, lr=1e-4, seed=42,
anchor-ratio=0 (merge already mixed in 240 anchor records at 8 reps each).
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
    ap.add_argument("--data", default="v2/data/synth.verified.jsonl")
    ap.add_argument("--out", default="v2/gguf")
    ap.add_argument("--quant", default="q4_k_m")
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument(
        "--anchor-ratio", type=float, default=0.0,
        help="Default 0 because merge.py already mixed in the 240 anchor "
             "records (30 x 8 reps). Pass >0 only if feeding primary-only data.",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--rslora", action="store_true",
                    help="Use rsLoRA (alpha/sqrt(r)) scaling per Kalajdzievski 2023 "
                         "(arxiv 2312.03732). Default is vanilla (alpha/r). "
                         "rsLoRA often produces the gains people incorrectly attribute "
                         "to higher rank with vanilla scaling.")
    ap.add_argument("--lora-alpha", type=int, default=None,
                    help="Override lora_alpha. Default = rank (gives effective scaler "
                         "1.0 for vanilla, sqrt(r) for rsLoRA). For ablation studies "
                         "decoupling alpha magnitude from scaling type.")
    args = ap.parse_args()

    primary = load_jsonl(Path(args.data))
    if args.anchor_ratio > 0:
        anchors = load_jsonl(ANCHORS_SRC)
        records = mix_anchors(primary, anchors, args.anchor_ratio, args.seed)
        print(f"[train] primary={len(primary)}  anchors_mixed={len(records)-len(primary)}  total={len(records)}  rank={args.rank}")
    else:
        records = primary
        print(f"[train] primary={len(primary)} (anchors already included)  rank={args.rank}")

    print(f"[train] fresh-from-base MODEL={MODEL}")
    model, tok = FastLanguageModel.from_pretrained(
        MODEL, max_seq_length=4096, load_in_4bit=True,
    )
    lora_alpha = args.lora_alpha if args.lora_alpha is not None else args.rank
    scaling_type = "rsLoRA (alpha/sqrt(r))" if args.rslora else "vanilla (alpha/r)"
    if args.rslora:
        eff_scaler = lora_alpha / (args.rank ** 0.5)
    else:
        eff_scaler = lora_alpha / args.rank
    print(f"[train] LoRA scaling: {scaling_type}, alpha={lora_alpha}, rank={args.rank}, "
          f"effective scaler ≈ {eff_scaler:.2f}")
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.rank, lora_alpha=lora_alpha, lora_dropout=0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth",
        use_rslora=args.rslora,
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

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save the LoRA adapter first (cheap, recoverable if merge step fails).
    adapter_dir = out_dir / "adapter"
    model.save_pretrained(str(adapter_dir))
    tok.save_pretrained(str(adapter_dir))
    print(f"Saved LoRA adapter to {adapter_dir}")

    # Save merged 16-bit safetensors. Unsloth's save_pretrained_gguf has
    # been broken across v1.1/v1.2/v2.0 (HUNYUAN_VL AttributeError in the
    # auto-downloaded converter script). Emit merged safetensors here and
    # run the manual GGUF pipeline separately — see docs/v2.plan.md
    # §"GGUF conversion".
    model.save_pretrained_merged(args.out, tok, save_method="merged_16bit")
    print(f"Saved merged 16-bit safetensors to {args.out}")
    print("Next: manual GGUF conversion per docs/v2.plan.md §GGUF conversion")


if __name__ == "__main__":
    main()
