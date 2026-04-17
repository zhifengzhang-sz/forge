# Lessons Learned

Everything that went wrong, surprised us, or required course correction during the first build of the forge pipeline. Written as a reference for future iterations and for anyone building a similar local fine-tuning pipeline.

## Model Selection

### Gemma 4 17B does not exist
The original design assumed a Gemma 4 17B model. It doesn't exist. The Gemma 4 family is: E2B (2.3B), E4B (4.5B), 26B-A4B MoE (3.8B active), and 31B Dense. There is no mid-range dense model. Always verify model availability on the vendor's model card before committing to a design.

### MoE models can't be QLoRA-tuned on consumer GPUs
Gemma 4 26B-A4B MoE looked ideal (77% LiveCodeBench, fast inference). But MoE models require 16-bit LoRA (not 4-bit QLoRA) because expert routing interacts poorly with quantization. This pushes VRAM above 40 GB, exceeding the RTX 5090's 32 GB. We discovered this during the CEO review via web search, not from the Unsloth docs. The lesson: always verify the training VRAM requirement for your specific model architecture, not just inference.

### VRAM estimates in design docs are wrong
The design estimated Qwen3-14B training at ~9 GB. Reality: batch_size=4 with r=32 LoRA hit OOM at 30.4 GB on a 31.3 GB card. The estimates didn't account for activation memory, optimizer states on GPU, and Triton kernel workspace. Practical fix: batch_size=2 with grad_accum=8. Design doc VRAM numbers should be treated as lower bounds, not actuals.

### HuggingFace model names are confusing
- `Qwen/Qwen3-14B-Instruct` — requires auth, may not exist (Qwen3 has no separate Instruct variant)
- `unsloth/Qwen3-14B` — no auth required, pre-quantized for training
- `qwen3:14b` — Ollama name, pre-quantized GGUF for inference only

Three different ecosystems (vendor HF, Unsloth HF, Ollama) with three different naming conventions for the same model. The training pipeline uses Unsloth names. The inference pipeline uses Ollama names. They are not interchangeable.

## Environment Setup

### Python packaging is a minefield
The install order matters. Installing unsloth before torch, or torch without the CUDA index URL, or unsloth without unsloth_zoo, or unsloth_zoo without torchvision — all produce different cryptic errors. The only reliable path is the exact sequence in `requirements.txt` with pinned versions.

### System packages not obvious
Training requires `python3.13-dev` (for Triton's CUDA compilation) and `python3.13-venv` (for creating the virtual environment). Neither is installed by default on Ubuntu. Both produce non-obvious errors that don't mention the missing package by name.

### Triton compiles CUDA code at runtime
On first training run, Triton compiles a C module (`cuda_utils.c`) using gcc against Python headers. If `python3.13-dev` is missing, you get a gcc error deep in a stack trace that doesn't mention Python headers. This is not documented in Unsloth or Triton's installation guides.

### Ollama version conflicts
Having both snap and apt versions of Ollama installed causes version mismatches. The systemd service runs one version while the CLI runs another. Gemma 4 models require 0.20+. The install script doesn't handle the conflict — you must manually remove the snap version first.

## Data Pipeline

### Focus terms must match definitions, not just usage
The original focus terms (`setup(`, `Observable<`) match call sites but miss function/type definitions (`export function setup<`, `export type Observable`). The extraction pipeline parses definitions. This mismatch caused XState to yield only 29 units initially instead of 189+.

### Balance formula: never use min
The original `2 × min(domain_sizes)` formula let one small domain cap everything. With XState at 29 units, the cap was 58 — crushing FP's 2,741 units down to 58. Switched to `max(2 × median, 100)` with a 500 max.

### Held-out eval set is needed before training, not before extraction
The original design blocked extraction if the eval set didn't exist. This is wrong — you need to see extraction results before you can write meaningful eval examples. The eval set prevents training data contamination, so it's needed before training, not before extraction.

### Training data costs money — check it in
The instruction-generation step (Claude API or template-based) costs money and time. The resulting JSONL is small (~2 MB). Check it into git. Raw extracted units (before instruction generation) are deterministic from repos + config — don't check those in.

### Template-based instructions are free but limited
Without an API key, we used template-based instructions (`lib/common/template_instruct.py`). These produce less varied instructions than LLM-generated ones. The model may memorize template phrasings instead of generalizing. This is a known quality tradeoff documented in `docs/data.provenance.md`.

## Training

### Unsloth SFTTrainer is not vanilla TRL
Unsloth patches TRL's SFTTrainer heavily. The `formatting_func` parameter works differently. Passing a function that returns `[text]` causes errors. The correct approach: pre-map the dataset to a `text` column using `dataset.map()` and don't pass `formatting_func` at all.

### Dropout 0 enables Unsloth fast patching
Unsloth warns: "Dropout = 0 is supported for fast patching. You are using dropout = 0.05." Setting dropout to 0 enables memory-saving optimizations. On a tight VRAM budget (31 GB for 14B model), this matters.

### Actual training results for Qwen3-14B
- Batch size 2, grad accum 8, effective batch 16, LoRA r=32, dropout 0
- 1,048 train examples, 56 eval examples, 3 epochs, 198 steps
- Training time: 12 minutes on RTX 5090
- Training loss: 0.4346, eval loss: 0.3817 (healthy, no overfitting)
- LoRA adapter size: 501 MB
- Peak VRAM: ~30.4 GB out of 31.3 GB (tight but fit)

### PYTORCH_CUDA_ALLOC_CONF helps with fragmentation
Setting `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` reduces CUDA memory fragmentation. This doesn't increase total VRAM but makes better use of what's available. Worth adding to the training command by default.

### llama.cpp's requirements.txt destroys the training venv
The export script clones llama.cpp and runs `pip install -r requirements.txt` inside it. This installs a CPU-only torch, downgrades transformers, and breaks every training dependency. The fix: only install the `gguf` package (already in our requirements.txt) and never run llama.cpp's requirements.txt.

### unsloth_zoo declares torch<2.11.0 but works with 2.11.0
The `unsloth_zoo==2026.4.8` package declares `torch<2.11.0` but runs fine with 2.11.0+cu128. Installing everything via a single `pip install -r requirements.txt` fails because pip enforces the constraint. The fix: install torch first, then install unsloth_zoo with `--no-deps`. This is fragile and should be tested on every version bump.

### warmup_ratio is deprecated
Transformers 5.5+ deprecates `warmup_ratio` in favor of `warmup_steps`. Not a blocker (it still works with a warning) but should be updated.

## Design Process

### CEO review caught model selection errors
The /plan-ceo-review caught: Gemma 4 17B doesn't exist, 26B-A4B MoE can't be QLoRA-tuned on 32 GB, Qwen3 as an alternative. Without the review, we would have hit these at training time instead of design time.

### Eng review caught export script bugs
The /plan-eng-review caught: wrong llama.cpp script name (underscores vs hyphens), missing bf16 for Qwen3, no disk space budget, no held-out exclusion in dedup. All would have been runtime failures.

### Implementation audit found real bugs
The "find every deviation from the design" audit found: broken 5xx retry logic, missing held-out fingerprint exclusion, fragile type classification, stale model names. Running this audit after each implementation phase is worth the time.

### Docs must stay in sync
Every code change should update the design doc. We hit multiple cases where the doc said one thing and the code did another (model names, LoRA rank, balance formula, file structure). The deviation audit is the enforcement mechanism.

## What We'd Do Differently

1. **Verify model availability first.** Before writing any code, confirm that every model referenced in the design actually exists and can be downloaded without auth.

2. **Test the full setup on a clean machine.** The requirements.txt and setup.sh should be tested from scratch, not built incrementally by fixing errors.

3. **Start with a tiny training run.** 10 examples, 1 epoch, verify the full pipeline (extract → train → export → serve → query) works end-to-end before scaling up to 1,000+ examples.

4. **Pin everything from day one.** requirements.txt with pinned versions, source repo commit hashes, model download checksums. Reproducibility is not optional.

5. **Budget VRAM with 30% headroom.** If the model + LoRA + batch fits in 25 GB on paper, assume 32 GB in practice. Activation memory, Triton workspace, and PyTorch allocator overhead are real.
