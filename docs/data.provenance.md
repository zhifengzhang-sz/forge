# Data Provenance

How training data is produced, verified, and reproduced.

## Data Flow

```
Source Repos (git, pinned)
    ↓
Phase 1: Extract (.ts files → semantic units)
    ↓  deterministic, no API cost
Phase 2a: Regex Score (surface quality signals)
    ↓  deterministic, no API cost
Phase 2b: Dedup (SHA-256 fingerprint)
    ↓  deterministic, no API cost
Phase 2c: LLM Judge (semantic quality, Claude Haiku)
    ↓  ~$2-3, optional, results saved to dataset/judge_results.jsonl
Phase 2d: Balance (cap per domain)
    ↓  deterministic, no API cost
Phase 2e: Instruct (generate instructions)
    ↓  Two options:
    ↓  A) Template-based (free, deterministic) — lib/common/template_instruct.py
    ↓  B) LLM-generated (Claude Sonnet, ~$3-5) — lib/common/instruct.py
    ↓
Training Data (checked into git)
```

Phases 1, 2a, 2b, 2d are deterministic given the same repos and config.
Phases 2c and 2e call the Claude API and cost money. Their outputs are
checked into git so they don't need to be regenerated.

## Source Repositories

All training data originates from these open-source repositories:

| Repo | Version | Domain | License |
|---|---|---|---|
| gcanti/fp-ts | HEAD at clone time | typescript.fp | MIT |
| Effect-TS/effect | HEAD at clone time | typescript.fp | MIT |
| ReactiveX/rxjs | HEAD at clone time | typescript.reactive | Apache 2.0 |
| statelyai/xstate | HEAD at clone time | typescript.xstate | MIT |
| oskardudycz/EventSourcing.NodeJS | HEAD at clone time | typescript.eventsourcing | MIT |

### Pinning repo versions

Default clone uses `--depth=1` (latest HEAD). For exact reproducibility,
pin to a specific commit after cloning:

```bash
cd repos/fp-ts && git log --oneline -1  # record the commit hash
```

Store hashes in `dataset/source_versions.json` after each pipeline run:

```json
{
  "fp-ts": "abc1234",
  "effect": "def5678",
  "rxjs": "ghi9012",
  "xstate": "jkl3456",
  "EventSourcing.NodeJS": "mno7890"
}
```

To reproduce from pinned versions:

```bash
cd repos/fp-ts && git checkout abc1234
```

## Extraction (Phase 1)

**Tool:** `lib/typescript/extract.py`

**What it extracts:**
- Exported function declarations (`export function`, `export const`)
- Type aliases and interfaces (`export type`, `export interface`)
- Inline machine definitions (`const x = createMachine(`, `setup(`) from test and example files
- Git diffs (with `--full-history` flag only)

**What it skips:**
- Files in `node_modules/`, `dist/`, `build/`, `.git/`
- Generated type declarations (`*.d.ts`)
- Test files (`*.spec.ts`, `*.test.ts`) except for XState (where test files contain the best machine configuration examples)
- Effect's `internal/` directory (implementation plumbing, not public API)

**Context preserved:** Each unit includes the file's import block so the model
learns which imports are needed for each pattern.

**File filtering:** Only files containing at least one domain focus term are
processed. Focus terms include both call-site patterns (`setup(`) and
definition-site patterns (`setup<`, `StateMachine`) to capture declarations,
not just usage. Full term lists are in `app/typescript/*/config.py`.

## Regex Quality Scoring (Phase 2a)

**Tool:** `lib/typescript/score.py`

**Formula:**

```
score = ts_signals + domain_signals + diff_bonus - penalties
```

Each component:

| Component | How it works | Cap |
|---|---|---|
| ts_signals | +0.08 per TypeScript pattern match (readonly, generics, export, as const, type annotations) | 0.40 |
| domain_signals | +0.08 per domain focus term found in the code | 0.40 |
| diff_bonus | +0.10 if the unit is a git diff | 0.10 |
| penalties | -0.05 for console.*, TODO, FIXME; -0.10 for `any` type | varies |

**Threshold:** Units scoring below 0.30 are discarded.

**Minimum length:** Units shorter than 80 characters are scored 0.00.

Regex scoring is a fast, free first pass. It catches surface noise but cannot
evaluate semantic quality (whether the code demonstrates a useful pattern).

## Deduplication (Phase 2b)

**Tool:** `lib/common/dedup.py`

**Method:** SHA-256 hash of stripped code (whitespace and comments removed),
truncated to 16 hex characters.

**Held-out exclusion:** If `eval/held_out/*.json` files exist, their code
examples are fingerprinted and excluded from training data to prevent
eval contamination.

**Near-duplicates:** Intentionally not removed. Slight variations of the same
pattern (e.g. `Option` chain vs `Either` chain) are valuable training signal.

## LLM Quality Judge (Phase 2c)

**Tool:** `lib/common/judge.py`

**Model:** Claude Haiku (`claude-haiku-4-5-20251001`)

**Cost:** ~$2-3 for ~5,000 units

**Rubric:** Each unit is evaluated on 5 criteria:

1. **Pattern clarity** — does it demonstrate a clear, learnable pattern?
2. **Idiomaticity** — is this how an expert would write it?
3. **Self-containedness** — can the model learn from this without external context?
4. **Domain signal** — does it teach something specific to the target domain?
5. **Training value** — would including this improve model output?

**Scoring:** 1-5 scale. Units scoring below 3 are discarded.

| Score | Meaning |
|---|---|
| 5 | Excellent. Clear pattern, idiomatic, self-contained, high domain signal. |
| 4 | Good. Solid example with minor issues. Worth including. |
| 3 | Borderline. Has some value but also noise. Include only if dataset is small. |
| 2 | Poor. Mostly boilerplate or glue code. Skip. |
| 1 | Bad. Wrong patterns or no domain relevance. Skip. |

**Output:** `dataset/judge_results.jsonl` with score, reason, source, and domain
for every evaluated unit. This file is checked into git.

**Reviewing judge results:** Use the data reviewer agent in Claude Code:

```
Read agents/data.reviewer.md for instructions, then review the training
data quality. Read dataset/judge_results.jsonl, sample examples from
each domain, and tell me if the data looks good for fine-tuning.
```

## Domain Balancing (Phase 2d)

**Tool:** `lib/common/balance.py`

**Formula:**

```
cap = min(max(2 * median(domain_sizes), 100), 500)
```

Uses median (not min) so one small domain doesn't crush larger ones.
Floor of 100 ensures minimum representation. Max of 500 prevents any
domain from dominating.

Domains smaller than the cap contribute all their units.
Units are selected in descending quality score order.

## Instruction Generation (Phase 2e)

Two methods available. Both produce the same output format.

### Method A: Template-based (free, deterministic)

**Tool:** `lib/common/template_instruct.py`

**Cost:** Free

**Method:** For each code unit, a training instruction is constructed from:
1. The **domain** (fp, reactive, xstate, eventsourcing)
2. The **unit type** (function or type)
3. The **primary identifier** extracted from the code (e.g. `ask`, `repeatWhen`, `SetupReturn`)
4. **Domain types** referenced in the code (e.g. Either, Observable, StateMachine)

These are combined using per-domain templates:

| Domain | Example instruction |
|---|---|
| typescript.fp | "Implement the fp-ts function `ask` that works with Reader, Task using pipe/flow composition" |
| typescript.reactive | "Implement the RxJS operator `repeatWhen` that transforms Observable streams with proper subscription lifecycle" |
| typescript.xstate | "Define the XState v5 types for `SetupReturn` including MachineContext, EventObject, and ActorLogic generics" |
| typescript.eventsourcing | "Implement `getEventStore` for an event-sourced system with aggregate state evolution and event stream operations" |

Template selection is deterministic: the same unit always produces the same instruction
(selected by `hash(fingerprint) % len(templates)`).

**Limitations:** Templates produce less varied instructions than LLM generation.
The model may learn to associate specific phrasings with specific patterns
rather than generalizing. For higher quality, use Method B.

**Run standalone:**

```bash
python3 -m lib.common.template_instruct
```

### Method B: LLM-generated (requires API key)

**Tool:** `lib/common/instruct.py`

**Model:** Claude Sonnet (`claude-sonnet-4-20250514`)

**Cost:** ~$3-5 for ~1,000 units

**Method:** Each code unit is sent to Claude with the prompt: "Given this code
and its domain, generate one natural instruction that would produce this code."

**Advantages over templates:** More varied phrasing, context-aware descriptions,
catches nuances that templates miss. Produces better training signal for
instruction-following.

**Validation rules:**
- Must be phrased as a task (implement, create, write)
- Rejected if it contains "explain", "describe", "what is", "what are", "how does", "why"
- Must mention a domain-specific term (e.g. "fp-ts", "observable", "xstate")
- Rejected instructions are logged to `dataset/rejected.jsonl`

### Output format (both methods)

`dataset/typescript_training.jsonl` — messages-only JSONL compatible with Unsloth/TRL SFTTrainer:

```json
{
  "messages": [
    {"role": "user", "content": "Implement the RxJS operator `repeatWhen`..."},
    {"role": "assistant", "content": "import { Observable } from '...'\n\nexport function repeatWhen..."}
  ],
  "id": "abc1234f-0042"
}
```

Metadata sidecar at `dataset/metadata.jsonl`:

```json
{"id": "abc1234f-0042", "domain": "typescript.reactive", "source": "rxjs:src/internal/operators/repeatWhen.ts", "unit_type": "function", "quality_score": 0.64}
```

## Reproducing the Dataset

### From checked-in training data (free, instant)

The training data in `dataset/typescript_training.jsonl` is ready to use.
No pipeline run needed. Proceed directly to `train.py`.

### From scratch with templates (free, ~10 seconds)

```bash
# Extract, score, dedup, balance, template instructions
python3 extract_pipeline.py --skip-judge --skip-instruct
python3 -m lib.common.template_instruct
```

### From scratch with LLM instructions (requires API key, ~$5-8)

```bash
export ANTHROPIC_API_KEY="your-key"
python3 extract_pipeline.py
```

### Partial re-runs

```bash
# Re-extract only (free, ~2 seconds)
python3 extract_pipeline.py --skip-judge --skip-instruct

# Re-extract + re-judge (~$2-3)
python3 extract_pipeline.py --skip-instruct

# Re-run everything (~$5-8)
python3 extract_pipeline.py

# Single topic only
python3 extract_pipeline.py --topics typescript.xstate --skip-judge --skip-instruct

# Dry run (estimate cost, no API calls)
python3 extract_pipeline.py --dry-run
```

### What changes between runs

| Change | Re-run needed? | Cost |
|---|---|---|
| Source repo updated | Full pipeline | ~$5-8 |
| Focus terms changed | Full pipeline | ~$5-8 |
| Scoring weights changed | From judge step | ~$2-3 |
| Judge rubric changed | From judge step | ~$2-3 |
| Balance formula changed | From balance step | Free |
| New topic added | Full pipeline for new topic | ~$2 per topic |

## Files Tracked in Git

| File | What | Cost to produce | Why track |
|---|---|---|---|
| `dataset/typescript_training.jsonl` | Training data (messages) | ~$3-5 | Costs money, final artifact |
| `dataset/metadata.jsonl` | Source/domain per example | ~$3-5 (same run) | Traceability |
| `dataset/judge_results.jsonl` | Quality scores + reasoning | ~$2-3 | Audit trail, calibration |
| `dataset/rejected.jsonl` | Failed instructions | ~$3-5 (same run) | Debug instruction generation |

## Files NOT Tracked

| File | What | Why not track |
|---|---|---|
| `dataset/extracted_units.jsonl` | Raw extraction output | Deterministic from repos + config |
| `repos/` | Cloned source repos | Large, re-cloneable |
| `checkpoints/` | Training checkpoints | Large (28-62 GB) |
| `gguf/` | Quantized models | Large (14-24 GB) |
