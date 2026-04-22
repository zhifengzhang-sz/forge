# TODOS

Captured deferred work from plan reviews. Each entry: what, why, pros/cons,
context, dependencies.

---

## v1.1 (top priority): warm-start from v0.7-r64

**What:** Fork `v1/train.py` → `v1.1/train.py`. Change the base model
load from `unsloth/qwen3-14b-unsloth-bnb-4bit` to `v0.7/gguf/r64/`
(the merged safetensors from v0.7's winner run). Apply a fresh
`r=64` LoRA on top of that warm-started base. Train on v1's same
4885-record dataset (or a rebalanced variant — see v1.2 below).
Re-run Phase F/G/H.

**Why:** v1 scored FP=3.80 and RX=4.53, regressing from v0.7's
FP=4.40 and RX=4.80. Root cause per `docs/lessons.learned.md`: v1
trained a fresh LoRA from raw Qwen3-14B, so it had to re-learn
domains v0.7 already nailed — and re-learned FP worse because the
v1 data mix has subtler FP quality issues. Warm-starting from
v0.7-r64 preserves v0.7's FP=4.40 and RX=4.80 capabilities in the
frozen base weights; the new LoRA only needs to add ES + XState
depth + anchor expansion on top.

**Pros:**
- No relearning cost — prior capabilities are baked in.
- FP/RX regression should disappear (or at worst stay flat).
- ES gain (3.10 → 4.50 in v1) should hold.
- Training may be faster (less to learn) or allow fewer epochs.
- Unlocks the "each version ≥ prior" guarantee the project has
  implicitly been aiming for.

**Cons:**
- Loses clean ablation: can't compare v1.1 directly vs v0.6 because
  the base differs. Can still measure v1.1 delta vs v0.7.
- Compounds any subtle v0.7 quirks forward.
- Dependency on `v0.7/gguf/r64/` being preserved (it is, committed
  to the repo, ~28GB in safetensors).

**Context:** See `docs/lessons.learned.md` "Training Methodology — v1
findings" for the full analysis. This was the single most important
lesson from v1.

**Depends on:** Nothing new. `v0.7/gguf/r64/model-*.safetensors` is
already on disk. Just change `MODEL` in train.py to that path.

**Estimated cost:** ~75 min training + 15 min GGUF + 2 min eval + 10
min grading. Total ~100 min for a complete v1.1 run.

---

## v1.2 (after v1.1 validates warm-start): rebalanced data mix + tighter FP gate

**What:** If v1.1 confirms warm-start is the right architecture, v1.2
investigates whether the v1 data mix itself has issues independent of
the warm-start question. Specifically:
- Drop FP volume to match v0.7's proportion (~30% of total, ~300
  records) — remove the weaker v1 FP batches (B2 at 68% survival and
  B22 at 70% were flagged).
- Tighten FP verifier positive gate — require actual `Effect.gen`
  usage, not just "contains `pipe(`".
- Drop anchor ratio from 12% back to 8% (30 unique × 12 reps = 360,
  not 600).
- Train on warm-start base from v1.1.

**Why:** v1's FP verifier gate is looser than XState's — 20 positive
tokens satisfies the gate even if the code is structurally weak (e.g.
`fp-04`'s `class extends interface`). Some v1 FP batches (B2, B22)
let through records that teach the model subtler mistakes. A v1.2
with tighter FP gate + trimmed FP set should separate "data mix was
bad" from "fresh-start was bad."

**Depends on:** v1.1 complete, FP still below baseline.

---

## v1.3: r=128 A/B ablation on 4885 dataset

**What:** Run v1.1 or v1.2's winning recipe at r=128 as a single-
variable ablation against r=64. Warm-started from v0.7-r64.

**Why:** The v1 plan deferred r=128 because "validation loss didn't
stagnate at r=64" in v0.7. v1 has now shown FP drift at r=64 on 5×
data — some of that drift might be capacity starvation (4 domains,
1194 FP records, r=64 rank may not have enough capacity to preserve
all FP patterns). r=128 adds ~11% training time (~83 min vs 75 min)
for a clean capacity answer.

**Depends on:** v1.1 shows warm-start fixes the regression. If v1.1
still regresses, capacity isn't the answer and skip this.

---

## Production-in-the-loop validation (v1.5)

**What:** Dogfood `ts-forge-v1:latest` against real Claude Code tasks with a
rubric comparing outputs to Opus on the same tasks. Target 2 weeks of daily
driver use on the target surfaces (XState, FP, RX, ES).

**Why:** v0.7/decision.md recommendation #4 has stayed open through v1. Eval
scores are proxies for "is this good at writing idiomatic TypeScript"; real
usage tests "is this good enough that the user would actually switch from
Claude Code + Opus." The bigger unlock after v0.7 was always production
validation, not eval-score chasing.

**Pros:** Closes the real gap question. Generates a 2-week log of pain points
that shape v2 priorities. Forces the "ship" definition gap to resolve.

**Cons:** Rubric design needs real work (what counts as "Opus-equivalent"?).
2-week dogfood burden on the user. Selection bias (the user knows they're
being tested).

**Context:** v0.7 r=64 hit 4.10 domain avg vs Opus 4.95. v1 targets 4.5+.
Both are eval-scored. No one has asked "on a real bug fix, which one did I
prefer?" yet.

**Depends on:** v1 complete with `ts-forge-v1:latest` shipped to ollama.

---

## r=128 A/B at 5000 scale (v1.1)

**What:** Run r=128 as a single-variable ablation once v1's scale+ES signal
is in. Same data, same recipe, only rank changes. Grade with same
two-grader-plus-tiebreak protocol.

**Why:** 2026 LoRA literature says r=128 adds ~11% training slowdown with no
consistent accuracy gain over r=64 unless validation loss stagnates. v0.7
validation didn't stagnate at r=64 (FP lifted 0.90 over r=32). At 5000 pairs
the question becomes: does capacity help at higher data volume? v1 answer
would be definitive.

**Pros:** Closes the "should future runs go to r=128" question with a single
training arm. Cheap (30min training + 5min grading).

**Cons:** v0.7 already showed no value in r=128-class questions at smaller
scale. Risk of spending subagent budget on a null result.

**Context:** v1 locked r=64 to isolate the scale+ES variable. v1.1 holds
everything constant from v1 except rank.

**Depends on:** v1 complete, v1 ship decision made, v1 training recipe
frozen.

---

## Eval expansion to n=10 per domain (v1.1)

**What:** Add 5 prompts per domain (XState/FP/RX/ES) and 5 capability anchor
prompts. Backfill `ts-forge-v0.7-r64` through the new set to preserve
apples-to-apples with v0-v0.7. Keep old prompts in place; new prompts are
additive.

**Why:** Current n=5 gives a binomial CI of roughly ±1 point on domain avg.
v1's predeclared thresholds (XState≥4.5, ES≥3.8) claim precision tighter
than the instrument. Expansion to n=10 roughly halves the CI and lets
future runs make real precision claims.

**Pros:** Statistical rigor. Removes the "n=5 caveat" from all future
decisions. Apples-to-apples preserved via v0.7-r64 backfill.

**Cons:** Prompt authoring cost (~30min). Grading cost grows linearly but
only by ~3min per run.

**Context:** v1 deliberately kept n=5 to isolate scale+ES as the variable.
v1.1 is the obvious place to expand because it's the next controlled run
anyway.

**Depends on:** v1 complete. Prompt authoring can start any time before
v1.1.
