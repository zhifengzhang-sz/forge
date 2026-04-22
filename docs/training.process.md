# Training Process — v2 Architecture

This document is the handoff for a **fresh Claude Code session** to
execute the better training process v1 surfaced the need for. All
facts needed are captured here or referenced via committed files. For
prior context see `v1/decision.md`, `docs/lessons.learned.md`
(especially the "Training Methodology — v1 findings" section), and
`TODOS.md`.

## Why this doc exists

v1 hit a regression gate halt. FP dropped from v0.7's 4.40 to 3.80,
Reactive from 4.80 to 4.53. Root cause: every forge version (v0,
v0.6, v0.7, v1) trained a **fresh zero-initialized LoRA on top of
raw Qwen3-14B**. Each run re-learned domains that prior runs already
nailed, and v1 re-learned FP *worse* than v0.7 despite having all of
v0.7's FP data plus 874 new FP pairs.

Four iterations of training went by without anyone questioning the
`MODEL = "unsloth/qwen3-14b-unsloth-bnb-4bit"` line. The cost:
~75 min of GPU per version spent re-learning + a shipped regression
in v1.

This document proposes a better training process and lays out the
execution order.

## Principles — the six changes

### 1. Warm-start, not fresh

Each training version continues from the last shipped version's
merged safetensors, not from raw base.

```python
# OLD (v0-v1): fresh zero-init LoRA on raw base
MODEL = "unsloth/qwen3-14b-unsloth-bnb-4bit"

# NEW (v1.1+): warm-start from prior winner's merged weights
MODEL = "v0.7/gguf/r64"  # or latest shipped version
```

Unsloth's `save_pretrained_gguf` already emits merged safetensors to
`{version}/gguf/{arm}/model-0000N-of-0006.safetensors`. No new
infrastructure needed — just change the MODEL path in `train.py`.

**Benefit:** prior capabilities baked into the frozen merged base.
New LoRA learns only the delta (e.g., ES patterns). No more burning
RX=4.80 to re-learn it badly.

**Cost:** loses clean ablation ("v1.1 vs v0.6" is no longer
meaningful because bases differ). Still supports "v1.1 vs v0.7" as
the one meaningful comparison.

### 2. Delta training — only train what's new or broken

With warm-start, new training data can be minimal. If v0.7 trained on
1053 records and v1.1 warm-starts from v0.7, v1.1's training data
only needs to cover the *delta*: new ES domain, new anchor surface,
corrections for any prior-version issues. Maybe 1500–2000 records
instead of 4885.

**Benefit:** training time drops from 73 min (v1) to ~20–30 min. Over
10 versions, this saves 60+ GPU-hours and ~10× more iteration
opportunities.

**Cost:** none material. Prior version's data lives in the base
weights, not the new dataset.

### 3. Mid-training regression canary

Currently `train.py` runs `SFTTrainer.train()` to completion, then
evals. No signal during training. If the model is drifting away from
FP capability at step 400 of 1800, you only find out at the end.

Add a `TrainerCallback` that runs a 4-prompt canary (one per trained
domain) every N steps. If any domain drops >0.5 below its
prior-version baseline mid-training, halt.

```python
class RegressionCanary(TrainerCallback):
    FLOORS = {"xstate": 3.8, "fp": 3.8, "reactive": 4.3, "es": 3.5}
    def on_step_end(self, args, state, control, **kw):
        if state.global_step % 500 == 0:
            scores = quick_canary(model, prompts=CANARY_4)
            if any(scores[d] < self.FLOORS[d] for d in scores):
                control.should_training_stop = True
```

**Benefit:** prevents burning 73 min to discover a regression at the
end. Caps the cost of a bad data mix.

**Cost:** ~30s overhead per canary × 4 canaries = 2 min added to a
typical run. Negligible.

### 4. Per-domain absolute floor, not flat delta

v1's regression gate used `drop ≥ 0.3 from prior` uniformly across
domains. But RX at 4.80 has 0.20 of headroom before 5.0; a 0.3 drop
means it fell into never-observed territory. FP at 4.40 has 0.60 of
headroom; a 0.3 drop is significant but within normal variance.

Better gate: **each domain has both a delta threshold AND an
absolute floor**. Halt if either fires.

```python
REGRESSION_GATE = {
    "xstate":    {"delta": -0.3, "floor": 3.8},
    "fp":        {"delta": -0.3, "floor": 4.0},
    "reactive":  {"delta": -0.2, "floor": 4.5},  # tight; near-ceiling domain
    "es":        {"delta": -0.3, "floor": 3.5},
}
```

**Benefit:** informative halt conditions. "You regressed 0.3 vs
prior" vs "you dropped into never-observed territory" are different
problems and deserve different responses.

**Cost:** configuration only. Absorbed into `combine_v1.py`'s
regression gate logic.

### 5. Eval resolution — n=10 minimum

At n=5 per domain, a single response scored 2 drops the domain mean
by 0.40 — enough to flip ship/no-ship on one prompt. v1's FP
regression was driven almost entirely by one prompt (fp-04) where
the model wrote non-compiling code. If fp-04 had scored 4, FP would
be 4.20, borderline vs 4.0 but not a halt.

**Expand eval to n=10 per domain.** n=15 for FP specifically (the
domain with the most noise). Total prompt count: 24 → ~55.

Backfill v0.7-r64 through the expanded set in the same session to
preserve apples-to-apples comparability.

**Benefit:** noise floor drops from ±0.40 (n=5 single-response swing)
to ±0.20 (n=10). Tight thresholds like `FP ≥ 4.0` become meaningful
instead of precision-claim-tighter-than-instrument.

**Cost:** prompt authoring (~30 min per domain × 4 domains = 2h
one-time). Grading cost grows linearly; ~6 min per run instead of 3.

### 6. Data provenance on every record

Every synthesis record should carry its source metadata:

```json
{
  "messages": [...],
  "domain": "fp",
  "source": {
    "wave": "B2",
    "subagent_id": "a1b1e3abad4f99ed3",
    "theme": "Either chain",
    "pattern_ref": "fp-patterns-08",
    "phrasing": "build-from-scratch",
    "timestamp": "2026-04-21T12:34:56Z"
  }
}
```

**Benefit:** when a domain regresses, you can immediately query
"which batches contributed to the bottom-quartile responses in the
new model?" and identify the bad training signal. For v1's FP
regression, this would have pointed at B2 or B22 (the 68–70%
survival batches) as likely culprits.

**Cost:** metadata plumbing in each synthesis subagent's output
format, the merge script's schema, and the verifier's pass-through.
~2h of code work, then free forever.

## Phase execution order

### Phase 1 — v1.1: validate warm-start

**Goal:** prove that warm-starting from v0.7-r64 eliminates v1's FP
and RX regressions, using the same 4885-record dataset.

**Changes:**
1. Fork `v1/train.py` → `v1.1/train.py`.
2. Change `MODEL = "unsloth/qwen3-14b-unsloth-bnb-4bit"` → `MODEL =
   "v0.7/gguf/r64"`.
3. Update output path to `v1.1/gguf`.
4. Everything else identical: r=64, epochs=3, lr=1e-4, seed=42,
   anchor-ratio=0 (data already mixed).

**Execution:**
```bash
cd /home/zzhang/dev/ai/models/forge
.venv/bin/python v1.1/train.py
# ~20-75 min depending on whether we also cut epochs
# then ollama create ts-forge-v1.1 -f v1.1/gguf_gguf/Modelfile
.venv/bin/python v0/run_eval.py --model ts-forge-v1.1 --arm v1.1
.venv/bin/python v0/tool_call_smoke.py --model ts-forge-v1.1 --arm v1.1
# two-grader + combine as in Phase G
```

**Predicted outcome** (warm-start hypothesis):
- XState: ~4.20 (holds — v1 was +0.10)
- FP: ~4.40 (back to v0.7 baseline — this is the key test)
- Reactive: ~4.80 (back to v0.7 baseline)
- ES: ~4.50 (holds — v1's gain carries forward through data)
- Overall: ~4.50, premise exceeded, Phase I ships.

**Failure modes:**
- FP stays at 3.80 → warm-start alone is insufficient; data mix IS
  the problem (move to Phase 3 data-mix fix).
- FP lands between 3.80 and 4.40 → partial fix; both warm-start and
  data-mix issues contribute.
- RX doesn't recover → unexpected; investigate RX-specific training
  (may be a capacity issue for r=64 across 4 domains; Phase 4 r=128
  becomes relevant).

**Cost:** ~2h wall (75 min training + 15 min GGUF + 2 min eval + 30
min grading + reporting). Single-line code change.

**Decision gate:** run this first, regardless of other plans. The
answer to "is warm-start the right architecture" gates everything
else.

### Phase 2 — v1.2: delta-only training

**Only if Phase 1 works.** Rebuild the training dataset to contain
only what's *new* vs v0.7:
- ES: all 1364 verified ES pairs (new domain)
- XState: only the 702 *new* pairs (not v0.6's 440, which are in
  v0.7's base)
- FP: skip entirely (v0.7's 320 pairs in the base already hit 4.40)
- RX: skip entirely (same logic)
- Anchors: only the 18 NEW anchors × 20 reps = 360, not all 30 × 20

Total v1.2 dataset: ~2426 records instead of 4885. Training time
drops to ~30–35 min.

**Predicted outcome:** identical to Phase 1's v1.1, but at half the
training cost. Establishes delta-only training as the new norm.

### Phase 3 — v1.3: tighten verifier gates + expand eval

**Changes:**
1. `v0.7/verify.py`: tighten FP positive gate. Current gate:
   "any one of 20 tokens". New gate: "at least one STRUCTURAL pattern
   match":
   - `Effect\.gen\(function\s*\*`
   - `pipe\([^)]*,\s*(E|O|TE)\.`
   - `Layer\.(succeed|effect)\(`
   - `Context\.(GenericTag|Tag)<`
2. Similar structural tightening for ES and RX gates.
3. Eval expansion: author 5 new prompts per domain (total +20 ES/FP/
   RX/XState prompts) + backfill v0.7-r64 through them.

**Cost:** ~4h one-time (2h prompt authoring, 1h verifier tightening
with test cases, 1h rerun v0.7-r64 through new eval).

**Benefit:** future runs catch structurally weak synthesis BEFORE it
contaminates training. Eval noise drops from ±0.40 to ±0.20 on
domain means.

### Phase 4 — v1.4: mid-training regression canary

**Infrastructure work.** Add `RegressionCanary` callback to
`SFTTrainer`. Pseudocode in Principle 3 above. Canary probe is a
4-prompt suite (1 per trained domain) that runs every 500 steps via
direct `model.generate()` (no ollama roundtrip — still in the
training process).

**Benefit:** caps the cost of a bad data mix at ~N × canary_frequency
× per-step time, not full training time. If data mix is toxic, we
find out at step 500 instead of step 1800.

**Cost:** ~1 day of engineering (callback logic, quick_canary
implementation, floor calibration). Absorbed into `v1.4/train.py`
and all subsequent.

### Phase 5 — v2: data provenance + audit trail

Update the synthesis subagent prompts to include source metadata.
Update `v1/merge.py` to preserve metadata through dedup + anchor
expansion. Update `verify.py` to pass it through. Update grading
output format to include the source metadata of each response's
training context.

**Benefit:** debuggable training. "FP regressed → pull bottom-3
FP responses → trace their training data sources → find B22
over-represented → investigate B22's specific subagent prompt."

**Cost:** ~2h plumbing work. Zero recurring cost.

### Phase 6 — v2.0: first full "good process" run

Once Phases 1–5 are in place: a clean-room v2 that uses all six
principles from day one. Characteristics:
- Warm-starts from v1.N's winner (whichever ships after Phase 1)
- Trains delta only (whichever domain is being expanded — maybe
  SolidJS, Svelte, or a depth expansion on XState animations)
- Tightened verifier gates applied to new synthesis
- n=10+ eval infrastructure
- Mid-training regression canary active
- All records carry provenance

v2.0 should cost ~100 min wall-time end-to-end (down from v1's 18–20
h), produce a ship-quality model, and have a debuggable audit trail.

## Open questions / tradeoffs

### Warm-start vs LoRA stacking

An alternative to warm-start-from-merged is **stacking LoRAs**: keep
v0.7's LoRA as a separate adapter, load it alongside a new LoRA, and
train only the new one. Preserves the ability to disable prior
training for ablations. More complex, not yet needed.

**Recommendation:** go with warm-start-from-merged for now. Revisit
stacking if we ever need to A/B-ablate "what did v0.7 teach us"
vs "what did v1.1 teach us."

### r=64 vs r=128 at cumulative scale

Warm-start from v0.7-r64 means the merged base already contains a
rank-64 LoRA's worth of learned patterns. Adding a new r=64 LoRA on
top gives cumulative capacity 128-ish. Plausible that r=32 for the
new LoRA is enough, which would cut training time further. **r=128
A/B becomes less interesting** once warm-start is active, because
the effective capacity is higher anyway.

**Recommendation:** default new LoRA rank to r=32 in warm-start
runs, fall back to r=64 if new-domain learning stalls.

### What happens when v0.7's merged base has a bug?

Compounding risk: if v0.7 has a subtle quirk (e.g., a rare
hallucination pattern), every warm-started descendant inherits it.
There's no clean way to remove a learned pattern once it's in the
merged base.

**Recommendation:** (a) keep periodic clean-base runs as a sanity
check every 3–4 versions, (b) maintain `ts-forge-v0.7-r64:latest`
as the rollback tag forever. If v1.N drifts, we can always re-warm
from a known-good ancestor.

### When to retire fresh-from-base entirely?

Fresh-from-base still has value for:
- New-model ports (e.g., moving to Qwen 4.0 when it drops).
- Ablation studies ("what does our data alone teach?").
- Debugging suspected warm-start compounding.

**Recommendation:** keep `v0.7/train.py`-style fresh-from-base
scripts as reference. Don't delete. Just don't default to them for
iteration.

## What's already built vs what's new

**Already committed:**
- `v0.7/gguf/r64/model-000{1-6}-of-6.safetensors` (warm-start
  source)
- `v1/data/synth.verified.jsonl` (4885-record dataset for v1.1
  reuse test)
- `v1/merge.py` (dataset assembly)
- `v0.7/verify.py` (per-domain verifier — needs tightening per
  Phase 3)
- `v0/grading/combine_v1.py` (regression gate logic — needs
  per-domain floor extension per Phase 4)
- Two-grader + ad-hoc 3rd grader protocol (established by v1)
- `v0/eval/v0.json` (24-prompt suite — needs expansion per Phase 3)

**New work for v1.1:**
- `v1.1/train.py` (fork of v1/train.py with MODEL path changed)
- `v1.1/` directory scaffold (gguf/, logs/, decision.md)
- Rerun the full Phase F–H pipeline against ts-forge-v1.1

## Acceptance criteria for this plan

Before executing Phase 1, a fresh session should confirm:

1. `v0.7/gguf/r64/` still contains all 6 safetensors shards + config
   + tokenizer (warm-start base is intact).
2. `v1/data/synth.verified.jsonl` exists with 4885 records (the
   v1.1 training input).
3. `v1/train.py` runs cleanly today as a baseline (no env rot).
4. The regression gate in `combine_v1.py` still fires on v1's data
   (so we can verify v1.1 doesn't hit it).
5. nvidia-pl.service is active + 500W power cap verified
   (pre-flight from v1 plan).

## Starting point for the next session

```bash
cd /home/zzhang/dev/ai/models/forge
git pull  # if working from a fresh clone
cat docs/training.process.md  # read this file
cat v1/decision.md  # context on why we're doing this

# Phase 1 kickoff:
mkdir -p v1.1
cp v1/train.py v1.1/train.py
# edit v1.1/train.py: change MODEL path + output dir
# then run the pipeline

.venv/bin/python v1.1/train.py
# ollama create ts-forge-v1.1 -f v1.1/gguf_gguf/Modelfile
.venv/bin/python v0/run_eval.py --model ts-forge-v1.1 --arm v1.1
.venv/bin/python v0/tool_call_smoke.py --model ts-forge-v1.1 --arm v1.1
# dispatch grader A + grader B subagents (see v1 Phase G pattern)
.venv/bin/python v0/grading/combine_v1.1.py  # or fork combine_v1.py
```

The session-state trail: commits → `v1/decision.md` (v1 outcome) →
`docs/lessons.learned.md` (v1's structural lesson) →
`docs/training.process.md` (this file, the plan) → execution.

## What this plan is NOT

- Not a scope change. The four-domain TypeScript target remains.
- Not a model-family change. Still Qwen3-14B, still LoRA, still
  Ollama.
- Not a quality loosening. The regression gate tightens (absolute
  floors added), the verifier tightens, the eval expands.
- Not a ship-speed tradeoff. Phase 1 is a 2h validation, not a
  multi-day rework.
- Not a rejection of v1. v1 produced one legitimate win (ES +1.40),
  one legitimate failure (FP regression), and one structural
  insight (warm-start). All three feed forward into v1.1 and v2.

## Summary

The fix is one line (`MODEL = "v0.7/gguf/r64"`). The architecture
change takes six principles and five phases to do well. Phase 1
is the cheapest possible test of the single most important
principle, and should happen first.

v1's most valuable output is this plan, not the shipped model.
