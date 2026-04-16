# Forge — TypeScript-Specialised Local LLM Pipeline

Fine-tune open-weight models to write idiomatic TypeScript in four domains: functional programming (fp-ts/Effect), reactive streams (RxJS), state machines (XState v5), and event sourcing. Serve locally via Ollama, integrate with Claude Code.

See [docs/design.md](docs/design.md) for the full design document.

## Hardware Requirements

| Component | Minimum |
|---|---|
| GPU | NVIDIA RTX 5090 (32 GB GDDR7) |
| System RAM | 128 GB |
| Disk | 300 GB free |
| CUDA | 12.x |
| Python | 3.11+ |

## Models

Two candidate base models are trained and evaluated. The winner is selected after evaluation.

| Model | Training VRAM | Inference VRAM | Quantisation |
|---|---|---|---|
| Qwen3-14B | ~9 GB (4-bit QLoRA) | ~14 GB (Q8_0) | Q8_0 recommended |
| Gemma 4 31B | ~20 GB (4-bit QLoRA) | ~24 GB (Q6_K) | Q6_K recommended |

## Quick Start

### 1. Install Ollama

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Verify: `ollama --version` (requires 0.20+).

### 2. Pull base models

```bash
ollama pull qwen3:14b
ollama pull gemma4:31b
```

### 3. Install Python dependencies

```bash
python -m venv .venv
source .venv/bin/activate

pip install unsloth
pip install trl transformers datasets peft accelerate bitsandbytes
pip install anthropic   # for instruction generation (Phase 2)
```

### 4. Create the eval held-out set

Write 50 test examples per domain **before** running data extraction. These go in `eval/held_out/` and are excluded from training data by SHA-256 fingerprint.

```bash
mkdir -p eval/held_out
# Create fp.json, reactive.json, xstate.json, eventsourcing.json
# Each file: array of {instruction, expected_output} pairs
```

### 5. Extract and curate training data (Phase 1-2)

```bash
# Full pipeline (clone, extract, score, dedup, balance, generate instructions)
python3 extract_pipeline.py

# Estimate API cost without calling Claude
python3 extract_pipeline.py --dry-run

# Extract and curate only (skip instruction generation)
python3 extract_pipeline.py --skip-instruct

# Include git history diffs (requires full clone, slower)
python3 extract_pipeline.py --full-history
```

The pipeline runs in order:
1. **Clone** — 5 source repos to `repos/` (skips if already cloned)
2. **Walk** — find `.ts` files matching domain focus terms
3. **Extract** — parse exported functions, types, and optionally git diffs
4. **Score** — quality scoring based on TS patterns and domain signals (threshold: 0.3)
5. **Dedup** — SHA-256 fingerprint exact deduplication
6. **Balance** — cap each domain at 2x the smallest, max 500 per domain
7. **Instruct** — generate instructions via Claude API (requires `ANTHROPIC_API_KEY`)

Output:
- `dataset/typescript_training.jsonl` — training data (~1,600-2,000 examples)
- `dataset/metadata.jsonl` — sidecar with domain, source, unit type per example
- `dataset/rejected.jsonl` — failed instruction generations for review

The script prints an estimated API cost before starting and asks for confirmation.

### 6. Fine-tune (Phase 3)

Train each model sequentially:

```bash
# Train Qwen3-14B (~18 minutes)
python train.py --model qwen3-14b

# Train Gemma 4 31B (~31 minutes)
python train.py --model gemma4-31b
```

Training uses QLoRA with rank 32, saving checkpoints every 50 steps with early stopping on validation loss.

### 7. Export to GGUF (Phase 4)

Close Ollama before exporting the 31B model (the merge step loads ~62 GB into RAM):

```bash
sudo systemctl stop ollama

# Export Qwen3-14B (uses bf16 precision)
bash export.sh qwen3-14b

# Export Gemma 4 31B (uses f16 precision)
bash export.sh gemma4-31b

sudo systemctl start ollama
```

Each export run:
1. Merges the LoRA adapter into the base weights
2. Converts to GGUF via llama.cpp
3. Quantises (Q8_0 for Qwen3, Q6_K for Gemma 4)
4. Cleans up intermediate files

### 8. Import into Ollama (Phase 5)

```bash
# Create Ollama model from the evaluation winner
ollama create ts-forge -f Modelfile
```

Edit `Modelfile` to point to the winning model's GGUF:

```
# Qwen3-14B winner:
FROM ./gguf/qwen3-14b-typescript-q8_0.gguf

# Or Gemma 4 31B winner:
# FROM ./gguf/gemma4-31b-typescript-q6_k.gguf
```

Verify:

```bash
ollama run ts-forge "Write an fp-ts pipe that validates an email and returns Either<ValidationError, Email>"
```

### 9. Evaluate (Phase 7)

```bash
python eval/run_tests.py --model ts-forge --suite eval/tests/*.json --output eval/results/$(date +%Y%m%d).json
```

Three evaluation axes:
- **Syntactic correctness** — `tsc --strict --noEmit` (target: >90% pass)
- **Domain pattern fidelity** — signal term matching (target: >0.75 per domain)
- **Semantic correctness** — manual review against held-out set (200 examples)

### 10. Integrate with Claude Code (Phase 6)

```bash
export ANTHROPIC_BASE_URL="http://localhost:11434"
export ANTHROPIC_API_KEY="ollama"
claude --model ts-forge
```

Or add to your project's `.claude/settings.json`:

```json
{
  "model": "ts-forge",
  "env": {
    "ANTHROPIC_BASE_URL": "http://localhost:11434",
    "ANTHROPIC_API_KEY": "ollama"
  }
}
```

## Project Structure

```
forge/
├── README.md
├── extract/                 # Phase 1-2: modules
│   ├── clone.py             # Clone source repositories
│   ├── walk.py              # Walk .ts files, filter by focus terms
│   ├── extract.py           # Parse semantic units
│   ├── score.py             # Quality scoring
│   ├── dedup.py             # SHA-256 deduplication
│   ├── balance.py           # Domain balancing
│   └── instruct.py          # Instruction generation (Claude API)
├── extract_pipeline.py      # Phase 1-2: orchestrator
├── train.py                 # Phase 3: QLoRA fine-tuning
├── export.sh                # Phase 4: merge, convert, quantise, cleanup
├── Modelfile                # Phase 5: Ollama model definition
├── docs/
│   └── design.md            # Full design document (v2.1)
├── repos/                   # Cloned source repositories (gitignored)
├── dataset/
│   ├── typescript_training.jsonl
│   ├── metadata.jsonl
│   └── rejected.jsonl
├── checkpoints/             # Training checkpoints (gitignored)
├── gguf/                    # GGUF model files (gitignored)
└── eval/
    ├── run_tests.py
    ├── held_out/            # Created BEFORE Phase 1
    ├── tests/
    └── results/
```

## Retraining

Retrain when a target library releases a major version, evaluation fidelity drops below 0.70, or your codebase conventions diverge from the training data. See [docs/design.md](docs/design.md) Section 12.

To add your own codebase as a training source:

```python
# In extract/clone.py, add to the REPOS list:
{"url": "file:///path/to/your/project", "domain": "internal", "name": "my-project"},
```

And add matching focus terms in `extract/walk.py`:

```python
FOCUS_TERMS["internal"] = ["YourPattern", "yourFunction"]
```

## License

Training data is extracted from Apache 2.0 / MIT licensed repositories. Model weights inherit the base model license (Apache 2.0 for Qwen3, Gemma Terms of Use for Gemma 4).
