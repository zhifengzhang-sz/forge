#!/usr/bin/env python3
"""Phase 3: QLoRA fine-tuning with Unsloth.

Usage:
    python3 train.py --model qwen3-14b
    python3 train.py --model gemma4-31b
    python3 train.py --model qwen3-14b --dry-run   # verify setup without training
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
log = logging.getLogger("train")

ROOT = Path(__file__).parent
DATASET_PATH = ROOT / "dataset" / "typescript_training.jsonl"

MODELS = {
    "qwen3-14b": {
        "hf_name": "Qwen/Qwen3-14B-Instruct",
        "batch_size": 4,
        "grad_accum": 4,
    },
    "gemma4-31b": {
        "hf_name": "google/gemma-4-31b-it",
        "batch_size": 2,
        "grad_accum": 8,
    },
}


def check_prerequisites():
    if not DATASET_PATH.exists():
        log.error("Training data not found at %s. Run extract_pipeline.py first.", DATASET_PATH)
        sys.exit(1)

    line_count = sum(1 for _ in open(DATASET_PATH))
    log.info("Training data: %s (%d examples)", DATASET_PATH, line_count)

    try:
        import torch
        if not torch.cuda.is_available():
            log.error("CUDA not available. GPU required for training.")
            sys.exit(1)
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1024**3
        log.info("GPU: %s (%.1f GB)", gpu_name, gpu_mem)
    except ImportError:
        log.error("PyTorch not installed. Run: pip install torch")
        sys.exit(1)

    try:
        import unsloth
        log.info("Unsloth version: %s", unsloth.__version__)
    except ImportError:
        log.error("Unsloth not installed. Run: pip install unsloth")
        sys.exit(1)


def train(model_key: str, dry_run: bool = False):
    config = MODELS[model_key]
    log.info("=== Training %s (%s) ===", model_key, config["hf_name"])

    check_prerequisites()

    from unsloth import FastLanguageModel
    from trl import SFTTrainer
    from transformers import TrainingArguments
    from datasets import load_dataset

    # Load model in 4-bit
    log.info("Loading model %s in 4-bit...", config["hf_name"])
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config["hf_name"],
        max_seq_length=8192,
        load_in_4bit=True,
        dtype=None,  # auto-detect bf16
    )

    # Apply LoRA
    log.info("Applying LoRA (r=32, alpha=32)...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )

    # Load dataset
    log.info("Loading dataset from %s...", DATASET_PATH)
    dataset = load_dataset("json", data_files=str(DATASET_PATH))

    # Split for validation (5% held out for early stopping)
    split = dataset["train"].train_test_split(test_size=0.05, seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]
    log.info("Train: %d examples, Eval: %d examples", len(train_dataset), len(eval_dataset))

    if dry_run:
        log.info("Dry run — model loaded, dataset split. Exiting before training.")
        return

    # Output directory
    output_dir = ROOT / "checkpoints" / model_key
    lora_dir = ROOT / f"{model_key}-typescript-lora"

    # Training
    log.info("Starting training (batch=%d, grad_accum=%d, effective_batch=%d)...",
             config["batch_size"], config["grad_accum"],
             config["batch_size"] * config["grad_accum"])

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        max_seq_length=8192,
        dataset_num_proc=4,
        args=TrainingArguments(
            per_device_train_batch_size=config["batch_size"],
            gradient_accumulation_steps=config["grad_accum"],
            warmup_ratio=0.03,
            num_train_epochs=3,
            learning_rate=2e-4,
            bf16=True,
            logging_steps=10,
            output_dir=str(output_dir),
            save_strategy="steps",
            save_steps=50,
            eval_strategy="steps",
            eval_steps=50,
            load_best_model_at_end=True,
            lr_scheduler_type="cosine",
            optim="adamw_8bit",
            seed=42,
        ),
    )

    trainer.train()

    # Save LoRA adapter
    log.info("Saving LoRA adapter to %s...", lora_dir)
    model.save_pretrained(str(lora_dir))
    tokenizer.save_pretrained(str(lora_dir))

    log.info("Training complete. Adapter saved to %s", lora_dir)
    log.info("Next: run export.sh %s", model_key)


def main():
    parser = argparse.ArgumentParser(description="QLoRA fine-tuning")
    parser.add_argument("--model", required=True, choices=list(MODELS.keys()),
                        help="Model to train")
    parser.add_argument("--dry-run", action="store_true",
                        help="Load model and dataset, verify setup, don't train")
    args = parser.parse_args()

    train(args.model, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
