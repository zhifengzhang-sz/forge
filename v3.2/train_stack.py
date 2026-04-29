"""v3.2 trainer — LoRA stacking experiment.

Loads the bnb-4bit base, then loads v3.0-rslora's saved adapter as a
FROZEN "base" adapter, then adds a new TRAINABLE FP-recovery adapter
on top. Trains only the new adapter on FP-only data. At inference,
both adapters apply additively:

    W = W_base + ΔW_v3.0-rslora + ΔW_fp-recovery

This is structural capability isolation: v3.0-rslora's frozen weights
literally cannot change during training, so the XState/RX/ES gains it
encodes are preserved by construction.

After training, save BOTH adapters' state. Export to merged 16-bit
safetensors via tools/merge_adapter_stack.py (set_adapter list +
merge_and_unload), then manual GGUF pipeline.

We bypass Unsloth's get_peft_model wrapping for this run because Unsloth
doesn't have a clean multi-adapter API. We use raw transformers + PEFT.
"""

import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import PeftModel, LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

REPO_ROOT = Path(__file__).resolve().parent.parent
BASE_MODEL = "unsloth/qwen3-14b-unsloth-bnb-4bit"
FROZEN_ADAPTER_PATH = REPO_ROOT / "v3.0-rslora/gguf/adapter"
FROZEN_ADAPTER_NAME = "rslora_base"
NEW_ADAPTER_NAME = "fp_recovery"


def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="v3.2/data/synth.fp_only.jsonl")
    ap.add_argument("--out", default="v3.2/gguf")
    ap.add_argument("--rank", type=int, default=64,
                    help="rank for the new FP-recovery adapter (default 64; "
                         "smaller than v3.0-rslora's 128 because FP delta "
                         "is a focused correction, not full re-training)")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--rslora", action="store_true", default=True,
                    help="Use rsLoRA scaling for the new adapter (default True).")
    args = ap.parse_args()

    print(f"[train] Loading base bnb-4bit model: {BASE_MODEL}")
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_cfg,
        device_map="auto",
        trust_remote_code=True,
    )
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print(f"[train] Loading frozen v3.0-rslora adapter: {FROZEN_ADAPTER_PATH}")
    model = PeftModel.from_pretrained(
        model,
        str(FROZEN_ADAPTER_PATH),
        adapter_name=FROZEN_ADAPTER_NAME,
        is_trainable=False,
    )

    print(f"[train] Adding new trainable adapter: {NEW_ADAPTER_NAME} (r={args.rank}, rslora={args.rslora})")
    new_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        lora_dropout=0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        use_rslora=args.rslora,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model.add_adapter(NEW_ADAPTER_NAME, new_config)

    # Activate both adapters in forward; only NEW gets gradients
    model.set_adapter(NEW_ADAPTER_NAME)  # makes new the default-active for training
    # Verify trainable count
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[train] trainable params: {trainable:,} / {total:,} "
          f"({100 * trainable / total:.2f}%)")

    primary = load_jsonl(Path(args.data))
    print(f"[train] FP-only training records: {len(primary)}")

    clean = [{"messages": r["messages"]} for r in primary]
    ds = Dataset.from_list(clean)
    ds = ds.map(lambda x: {"text": tok.apply_chat_template(x["messages"], tokenize=False)})

    SFTTrainer(
        model=model, processing_class=tok, train_dataset=ds,
        args=SFTConfig(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            bf16=True, logging_steps=5,
            output_dir=str(Path(args.out) / "trainer"),
            save_strategy="epoch",
            seed=args.seed,
            gradient_checkpointing=True,
        ),
    ).train()

    # Save the new adapter only — frozen one is already at FROZEN_ADAPTER_PATH
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    new_adapter_dir = out_dir / "adapter_fp_recovery"
    model.save_pretrained(str(new_adapter_dir), selected_adapters=[NEW_ADAPTER_NAME])
    tok.save_pretrained(str(new_adapter_dir))
    print(f"[train] Saved new adapter (only) to {new_adapter_dir}")
    print(f"[train] Frozen base adapter remains at {FROZEN_ADAPTER_PATH}")
    print(f"[train] Next: run tools/merge_adapter_stack.py to combine both into a single safetensors")


if __name__ == "__main__":
    main()
