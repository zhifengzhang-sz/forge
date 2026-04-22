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
The old export script (export.sh, now removed) cloned llama.cpp and ran `pip install -r requirements.txt` inside it. This installs a CPU-only torch, downgrades transformers, and breaks every training dependency. The fix: only install the `gguf` package (already in our requirements.txt) and never run llama.cpp's requirements.txt.

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

### Use Unsloth's built-in GGUF export, not llama.cpp
Our original export.sh cloned llama.cpp, ran its requirements.txt (which destroyed the training venv by downgrading torch to CPU-only), then ran separate merge/convert/quantize steps. Unsloth has `save_pretrained_gguf()` that does all three in one call, handles the chat template correctly, and auto-generates the Ollama Modelfile. The entire export.sh was unnecessary. The lesson: before building custom glue, check if the framework already provides the functionality.

### Unsloth's GGUF export still uses llama.cpp under the hood
Even `save_pretrained_gguf()` clones and builds llama.cpp internally. It just automates the process. This means:
- `cmake` must be installed on the system (not in requirements.txt)
- First export run takes extra time to clone + build llama.cpp (~5 min)
- The build is cached at `./llama.cpp-unsloth/` for subsequent runs
- It prompts for confirmation to install missing system packages, which fails in non-interactive mode (background processes). Pipe empty input to auto-accept: `echo "" | python3 export.py`

## First End-to-End Success

The full pipeline works: extract → train → export → Ollama → query.

- **Model:** Qwen3-14B fine-tuned on 1,104 TypeScript examples
- **Training:** 12 min, 3 epochs, train_loss=0.43, eval_loss=0.38
- **Export:** Unsloth save_pretrained_gguf → Q8_0 (15 GB GGUF)
- **Ollama:** `ts-forge:latest`, 15 GB, runs on RTX 5090
- **First query result:** Generated Effect-TS style email validation with typed errors, pipe composition, multiple utility variants (Either, Option, Promise). Used `Effect.gen`, `yield*`, `Effect.fail` patterns from training data.

The model clearly learned domain patterns from the training data. Quality assessment pending formal evaluation (Phase 7).

## What We'd Do Differently

1. **Find a known-working reference pipeline first.** Before building custom glue, search for someone who has done exactly this (Unsloth → GGUF → Ollama) on similar hardware and follow their approach. We wasted hours on an export.sh that Unsloth already handles natively.

2. **Verify model availability first.** Before writing any code, confirm that every model referenced in the design actually exists and can be downloaded without auth.

3. **Test the full pipeline end-to-end with 10 examples before scaling.** Extract 10 units, train 1 epoch, export, import to Ollama, query. Verify the entire chain works before scaling to 1,000+. We scaled training before verifying export worked.

4. **List ALL system packages upfront.** python3.13-venv, python3.13-dev, gcc, cmake. Every one of these caused a separate failure. setup.sh should check them all before starting.

5. **Test the full setup on a clean machine.** The requirements.txt and setup.sh should be tested from scratch, not built incrementally by fixing errors.

6. **Pin everything from day one.** requirements.txt with pinned versions, source repo commit hashes, model download checksums. Reproducibility is not optional. But beware of overly strict pins (unsloth_zoo's torch<2.11.0 constraint) that require `--no-deps` workarounds.

7. **Budget VRAM with 30% headroom.** If the model + LoRA + batch fits in 25 GB on paper, assume 32 GB in practice. Activation memory, Triton workspace, and PyTorch allocator overhead are real.

8. **Never run third-party requirements.txt in your venv.** llama.cpp's requirements.txt downgraded torch from CUDA to CPU-only and broke everything. Always isolate or only install the specific package you need (e.g. `pip install gguf`).

9. **Non-interactive mode breaks interactive prompts.** Background processes can't answer `input()` prompts. Unsloth's export prompts to install cmake. Pipe empty input or pre-install dependencies.

## Training Methodology — v1 findings

### Every version trained fresh from raw base. This was a mistake.

All four forge versions (v0, v0.6, v0.7, v1) load the same raw model in `train.py`:

```python
MODEL = "unsloth/qwen3-14b-unsloth-bnb-4bit"
model = FastLanguageModel.get_peft_model(model, r=..., ...)  # fresh zero-initialized LoRA
```

`get_peft_model` initializes a new LoRA with all-zero adapter weights. Each version's LoRA has never seen any prior version's weights. Every training run re-learns the whole surface from scratch.

This decision was never made — it was inherited from the v0 starter script and never questioned across four iterations. The stated justifications are real:
- Clean ablations (r=32 vs r=64, or v0.6 vs v0.7 vs v1 comparison) are unambiguous when all share a common base.
- Full reproducibility: any version can be rebuilt from committed data alone; no prior artifact dependency.
- No compounding of subtle biases from prior runs.

**But the cost is real and v1 exposed it.** v0.7-r64 hit RX=4.80 and FP=4.40 on 240 and 320 training pairs respectively. v1 had *more* data for both domains (345 new RX + 240 v0.7 reused = 585 total; 874 new FP + 320 v0.7 reused = 1194 total) and scored *lower*: RX=4.53 and FP=3.80. v1 burned previously-proven capability and had to re-learn it badly. The `fp-04` eval prompt regressed from clean `pipe(fetchUser(id), TE.chainW(fetchOrders))` in v0.7 to a non-compiling `class NetworkDown extends FetchError` hybrid in v1 — a strict regression despite having all of v0.7's FP training data available.

**The fix is sequential LoRA / continue-from-prior training.** Unsloth's `save_pretrained_gguf` merges the LoRA into the base weights and emits merged safetensors (see `v0.7/gguf/r64/model-0000N-of-0006.safetensors`). For v1.1 and beyond, load that merged artifact as the base model instead of raw `unsloth/qwen3-14b-unsloth-bnb-4bit`, then apply a fresh LoRA on top. This preserves prior capabilities in the frozen merged weights while letting new LoRA training target only the delta (ES patterns, anchor expansion, etc.).

Concretely for v1.1:

```python
# OLD (v1 and earlier): fresh from raw base
MODEL = "unsloth/qwen3-14b-unsloth-bnb-4bit"

# NEW (v1.1): warm-start from v0.7-r64's proven capabilities
MODEL = "v0.7/gguf/r64"  # merged safetensors, already contains v0.7's LoRA
model, tok = FastLanguageModel.from_pretrained(MODEL, ...)
model = FastLanguageModel.get_peft_model(model, r=64, ...)  # fresh LoRA on top
```

Tradeoffs of warm-starting:
- **Loses clean ablation**: you can't directly compare v1.1-on-v0.7 vs v0.6 because the bases differ. You can still compare v1.1-on-v0.7 vs v0.7-original to measure the v1 delta.
- **Compounds bias**: if v0.7 had a subtle quirk, v1.1 inherits it. Trade this off against not burning 4.40→3.80 FP every iteration.
- **Harder to reproduce**: now v1.1 requires v0.7's committed safetensors; the dependency chain is longer.
- **Reduces training cost**: less to re-learn → can train fewer epochs or focus the new LoRA on just the delta domain (ES).

For a project where the whole point is "each version should be at least as good as the last on trained domains", the clean-ablation benefit isn't worth the capability loss. **v1.1 should start from v0.7-r64**, and every future run should default to warm-starting from the last shipped version unless there's a specific experimental reason not to.

### n=5 per domain is too small when thresholds claim tight precision

v1's predeclared thresholds (XState≥4.5, FP≥4.0, Reactive≥4.60, ES≥3.8) are tighter than the instrument measuring them. At n=5, a single response scored 2 drops the domain mean by 0.40 — enough to flip "pass" to "regression" on a single prompt. FP's regression in v1 is driven almost entirely by one prompt (fp-04) where v1 produced compile-broken code. If that single prompt had scored 4 instead of 2, FP would be 4.20, only borderline vs 4.0.

The regression signal is real, but its magnitude is amplified by small-n volatility. Eval expansion to n=10 was deferred in the v1 plan ("don't change the yardstick and the experiment in the same release") — that was correct for v1, but **v1.1 should expand eval before predicting threshold behavior, not after.**

### Warm-start would have prevented the v1 halt

If v1 had warm-started from v0.7-r64, the LoRA initialization would already encode the RX=4.80 and FP=4.40 capabilities. Training on v1's data would *add* ES + XState-depth + new-anchors on top, not reset + re-learn + get-worse-at-FP. Projected outcome: ES up (3.10 → ~4.50), XState modest gain, FP/RX held at v0.7 levels. No regression. No halt.

This isn't hindsight bias — it was the obvious right choice and we missed it for four versions in a row. v1's most valuable output is this realization, not the ES score.

### Lessons added to "What We'd Do Differently"

10. **Default to warm-start, not fresh-from-base.** Each training version should continue from the prior shipped version's merged weights, not re-initialize from raw base. Fresh-from-base is a research-mode choice for ablations; production iteration is sequential. Cost of ignoring this: v1 traded v0.7's proven FP=4.40 for v1's FP=3.80, despite having all of v0.7's FP data plus 874 new FP pairs. Three runs at ~75 min each were effectively wasted on relearning what the prior run already knew.

11. **Predeclared thresholds must respect the eval's resolution.** At n=5 per domain, the noise floor is roughly ±0.40 on a single response. Predeclaring `≥4.60` or `≥4.50` thresholds is claiming precision the eval can't deliver. Either expand the eval (n=10 minimum) or soften the thresholds to ranges ("4.3-4.7") so single-response volatility doesn't flip binary ship/no-ship decisions.

12. **"Don't regress" needs a per-domain absolute floor, not a flat delta.** A flat "0.3 drop from v0.7 halts" is fine for domains with room (FP at 4.40) but harsh for near-ceiling domains (RX at 4.80 has only 0.20 of headroom before 5.0). Better rule: halt if a domain falls below both (a) v0.7 minus 0.3 AND (b) an absolute floor that matches the domain's trained maturity (e.g. RX≥4.50 as an absolute floor independent of prior run).
