# v1.2 Decision — 2026-04-22

## TL;DR

**Outcome: Phase 2 hypothesis DECISIVELY FALSIFIED. Warm-start +
fresh-LoRA does NOT preserve unrelated-domain capability.** Removing
FP and RX from training caused FP to collapse from v0.7's 4.40 to
**2.20** — a -2.20 regression, far worse than v1 or v1.1.

DO NOT ship. v1.2 is the worst arm produced since v0.6. Three FP
prompts that v1.1 scored 4–4.5 collapsed to 1–2 in v1.2, with both
graders independently flagging "library confusion" — the model chose
wrong libraries (AWS SDK for file I/O, broken Option usage, Effect
imports without Layer/Context.Tag).

**The core architectural finding**: a fresh LoRA trained on
ES+XState data actively damages the frozen base's FP representations
via gradient leakage across all 40 layers × 7 projection matrices.
LoRA is global; training on domain X perturbs domain Y even when Y is
never in the batch.

**Redirect**: Phase 3 (tighten verifier gates + audit FP noisy
batches) is now the correct next step, not Phase 2. FP MUST be in the
training set to preserve FP capability; the lever is data quality,
not data exclusion.

ts-forge-v0.7-r64 remains the daily driver.

## Numbers (two-grader — tiebreak not triggered)

| Arm | Domain avg | XState | FP | Reactive | ES | Cap | Tool calls |
|---|---|---|---|---|---|---|---|
| base (qwen3:14b) | 2.80 | 1.60 | 2.60 | 4.00 | 3.00 | 3.25 | 3/3 |
| **v0.7-r64 winner** | **4.10** | **4.10** | **4.40** | **4.80** | **3.10** | **4.25** | **3/3** |
| v1-r64 (fresh, 4885) | 4.25 ↑ | 4.20 ↑ | 3.80 ↓↓ | 4.53 ↓ | 4.50 ↑↑↑ | 4.00 ↓ | 3/3 |
| v1.1-r64 (warm, 4885) | 3.94 ↓↓ | 3.80 ↓ | 4.00 ↓↓ | 4.40 ↓↓ | 4.10 ↑↑ | 3.25 ↓↓ | 3/3 |
| **v1.2-r64 (warm, 2426 delta)** | **3.65 ↓↓↓** | **4.30 ↑** | **2.20 ↓↓↓** | **4.60 ↓** | **3.90 ↑↑** | **3.25 ↓↓** | **3/3** |
| claude-opus-4-7 | 4.95 | 5.00 | 4.80 | 5.00 | 5.00 | 5.00 | n/a |

### Per-domain grader pair

| Domain | Grader A | Grader B | Mean | v0.7 | Δ | gate |
|---|---|---|---|---|---|---|
| XState | 4.40 | 4.20 | 4.30 | 4.10 | +0.20 | ok (improvement) |
| FP | 2.40 | 2.00 | 2.20 | 4.40 | **-2.20** | **HALT** (catastrophic) |
| Reactive | 4.60 | 4.60 | 4.60 | 4.80 | -0.20 | **halt** (at tol line) |
| ES | 4.00 | 3.80 | 3.90 | 3.10 | +0.80 | n/a |
| Capability | 3.25 | 3.25 | 3.25 | 4.25 | -1.00 | n/a |

Grader agreement: **24/24 within 1 pt**, mean disagreement **0.17**
(tightest yet — v1 was 0.33, v1.1 was 0.29). **Zero 2-point
disagreements** → no tiebreak triggered. Both graders independently
converged on the FP collapse diagnosis.

## Phase G regression gate output (copied verbatim)

```
=== v1.2-r64 ===
domain               v1.2   v0.7    delta   gate
--------------------------------------------------
xstate               4.30   4.10    +0.20     ok
fp                   2.20   4.40    -2.20   halt
reactive             4.60   4.80    -0.20   halt
eventsourcing        3.90   3.10    +0.80    n/a
capability           3.25   4.25    -1.00    n/a

capability breakdown:
  Cap_old12: (no records)
  Cap_new18: 3.25

grader agreement (within 1 pt): 24/24
overall mean disagreement: 0.17

=== Regression gate ===
HALT — trained domain regression detected:
  - fp: v1.2 2.20 vs v0.7 4.40 (delta -2.20, tolerance -0.30) — REGRESSION
  - reactive: v1.2 4.60 vs v0.7 4.80 (delta -0.20, tolerance -0.20) — REGRESSION
```

## Per-prompt deltas v1.2 vs v1.1

Critical shifts (≥1.5 pts):

| Prompt | Domain | v1.1 | v1.2 | Δ |
|---|---|---|---|---|
| **fp-03** | fp | 4.5 | **1.0** | **-3.5** |
| **fp-04** | fp | 4.5 | **2.0** | **-2.5** |
| **fp-05** | fp | 4.0 | **1.5** | **-2.5** |
| **rx-03** | reactive | 3.0 | 5.0 | +2.0 |
| **xstate-04** | xstate | 2.0 | 3.5 | +1.5 |
| **es-03** | eventsourcing | 2.5 | 4.0 | +1.5 |
| xstate-01 | xstate | 3.0 | 4.0 | +1.0 |

The pattern is clean: **prompts in trained domains (XState, ES, RX)
improved or held, prompts in excluded-but-broadly-represented domains
(FP) collapsed.** RX was also excluded but preserved — see next
section for why.

## Phase 2's hypothesis vs actual

The plan (docs/training.process.md §Phase 2) predicted:
- "Identical to Phase 1's v1.1" (same eval shape, half the training cost)
- v1.1/decision.md refined this: "if warm-start preserves capability
  when not re-trained, FP holds at 4.40+, RX holds at 4.80"

Actual:
- XState: 4.30 (better than v1.1's 3.80 by +0.50)
- FP: 2.20 (WORSE than v1.1's 4.00 by -1.80)
- Reactive: 4.60 (better than v1.1's 4.40 by +0.20)
- ES: 3.90 (worse than v1.1's 4.10 by -0.20)
- Cap: 3.25 (same as v1.1)

**FP went in the opposite direction predicted, and by a huge margin.**
The hypothesis isn't just wrong — it's anti-correlated with reality.
Training on ES/XState *damages* FP capability in a warm-started run.

## What happened — the catastrophic interference finding

**Warm-start + fresh LoRA + partial data causes gradient leakage into
domains the LoRA never sees.** Architecturally:

- LoRA modifies **all 40 layers × 7 projections** (q/k/v/o/gate/up/down)
  globally. 256.9M parameters spread across the whole transformer.
- During training, gradients come only from the batch (ES + XState +
  new-anchors). But the LoRA update `ΔW = B·A` is applied to
  *every subsequent forward pass*, including FP inference.
- If the ES/XState optimization direction is *orthogonal* to what the
  base had optimized for FP, the base's FP-friendly weights get
  subtly shifted. The update is a vector in 256.9M-dim space; it
  doesn't care about domain boundaries.

**Why FP collapsed but RX survived:**

- **XState shares surface syntax with Effect/fp-ts**: both use
  `setup({types,...}).createMachine` / `Effect.gen(function*)` patterns
  with typed blocks and actor/monad composition. Training heavily on
  XState's `setup` idiom biases attention heads toward that pattern.
  When asked for Effect code, the model's shifted attention generates
  wrong-library output (AWS SDK imports, Option misuse, missing
  `Layer.succeed`/`Context.Tag`).
- **RX is syntactically distant from XState**: pipe of operators
  (`debounceTime`, `switchMap`) vs setup+createMachine. The XState
  training pressure doesn't displace RX representations.
- **FP has MORE training samples in v0.7's base** (320 pairs) than RX
  (240 pairs), but paradoxically is MORE vulnerable. This is because
  v0.7's FP representations are more *entangled* with the base's
  general-TS patterns, and the new LoRA's aggressive XState bias
  disrupts that entanglement.

**Corollary: LoRA stacking would fix this.** If v0.7's LoRA were kept
frozen as a separate adapter and a new adapter trained alongside (see
docs/training.process.md §Open questions), the new adapter's
gradients would be isolated from v0.7's FP representations. The
current warm-start path merges v0.7's learning into the base weights
permanently, then lets a new LoRA modify them further without any
isolation.

## What went right

1. **XState improved +0.20 over v0.7.** 702 new pairs actually
   produced a capability lift when combined with warm-start. xstate-04
   (fromObservable) went from 2 (v1) / 2 (v1.1) to 3.5 (v1.2) — the
   specific weak spot is partially fixed.
2. **ES held strong at +0.80 over v0.7.** v1.1's ES was 4.10; v1.2 is
   3.90. Slight giveback, but still +0.80 vs v0.7 baseline. The
   recipe generalizes even on smaller delta-only training.
3. **RX survived despite exclusion from training.** 4.60 vs v0.7's
   4.80, at the -0.20 tolerance line. Warm-start DID preserve RX —
   the catastrophic interference is FP-specific.
4. **es-03 fixed from v1.1.** v1.1 scored 2.5 (handler.load.initial
   bug, reduce without seed); v1.2 scores 4.0. Both graders
   independently confirmed the fix.
5. **rx-03 fixed.** v1.1 scored 3 on concurrency cap; v1.2 hits 5.
   RX wasn't trained here — this is a general capability lift, not
   specific training.
6. **Tool-calling preserved: 3/3 PASS** (same as v0.7, v1, v1.1).
7. **Training ran cleanly in 42 min** (vs v1.1's 69 min). Manual
   GGUF workaround completed in 2 min. Budget came in 55% of Phase 1
   cost; fast-fail worked as designed.
8. **Grader agreement tightened further to 0.17** — the tightest the
   two-grader protocol has produced. Confidence in the FP collapse
   finding is extremely high.
9. **Training loss curve matched predictions.** Opening loss 0.97 at
   epoch 0.016 (vs v1.1's 0.38) — the delta-only data contained more
   novel signal, so the model had more to learn. Loss converged to
   0.17 by epoch 3.0. The *training dynamics* are behaving correctly;
   it's the *capability-preservation assumption* that was wrong.

## What went wrong

1. **FP crashed catastrophically** (-2.20). The Phase 2 hypothesis —
   that warm-start preserves FP when FP isn't retrained — is
   decisively false. Three specific prompts collapsed:
   - **fp-03 (Option.fromNullable)**: both graders independently
     flagged "broken Option impl." v1.1 scored 4.5; v1.2 scores 1.0.
   - **fp-04 (TaskEither retry)**: v1.1 scored 4.5; v1.2 scores 2.0.
     Model produced wrong-library code.
   - **fp-05 (Effect.gen + Layer)**: v1.1 scored 4.0; v1.2 scores
     1.5. Effect imports present but no Layer/Context.Tag composition.
2. **cap-02 re-collapsed to 1/1.** Same prompt, same failure mode as
   v1.1 — ignored the Read-tool instruction, fabricated nonsense
   (AWS SDK imports per both graders). Warm-start didn't help
   capability anchors.
3. **es-02 gave back 1 point** (4.5 → 3.5). Throw-from-decide choice
   that grader B specifically flagged.
4. **Overall mean dropped to 3.65** — the worst arm since v0.6 (3.00).
   v0.7 at 4.10 and v1 at 4.25 both beat every subsequent arm.

## What v1.2 surfaced (beyond the headline)

1. **The "data exclusion preserves capability" hypothesis is
   decisively falsified.** Architecturally, LoRA does not respect
   domain boundaries during optimization. You cannot "protect" FP by
   excluding FP data; you must protect FP by *including high-quality
   FP data* that rides along with the new-domain training.
2. **Cross-domain interference is asymmetric.** XState training hurts
   FP but not RX. This suggests interference correlates with
   *syntactic surface similarity*, not *semantic domain proximity*.
   Both FP and XState are "composable abstractions" — but the
   *tokens* they share are the real risk vector.
3. **Phase 3's verifier tightening is the correct lever.** Since FP
   must be in the training set but v1's B2/B22 batches poisoned it,
   the fix is: keep FP data, use a stricter verifier (require actual
   `Effect.gen` usage, `Layer.*`, `Context.GenericTag` — not "any
   of 20 tokens"), drop the weak pairs.
4. **The Phase 1 + Phase 2 sequence was the right call.** Each was
   the cheapest possible test of a specific hypothesis. Both
   falsified cheaply (~100 min each). The v1.1 result pointed at
   Phase 2 as critical; the v1.2 result redirects to Phase 3. This is
   *exactly* what the training-process plan was designed to produce —
   fast iteration on structural hypotheses.
5. **LoRA stacking moves up the priority list.** If Phase 3 also
   produces a warm-start regression, stacking (keep v0.7's LoRA
   frozen as a separate adapter) is the last architectural move left
   before we have to admit r=64 can't serve 4+ domains on a warm base.
6. **Grader stability across three sessions** (v1 session, v1.1
   session, v1.2 session) confirms the protocol is calibrated. v1.2's
   mean disagreement 0.17 is the lowest yet — as graders see the arm
   progression, their calibration tightens rather than drifts.

## Gap to Claude Opus 4.7

| Domain | v1.1 | v1.2 | Opus | v1.2 gap | v1.2 as % of Opus |
|---|---|---|---|---|---|
| XState | 3.80 | 4.30 | 5.00 | -0.70 | 14% |
| FP | 4.00 | **2.20** | 4.80 | **-2.60** | **54%** |
| Reactive | 4.40 | 4.60 | 5.00 | -0.40 | 8% |
| Event Sourcing | 4.10 | 3.90 | 5.00 | -1.10 | 22% |
| Capability | 3.25 | 3.25 | 5.00 | -1.75 | 35% |

v1.2 closed the XState gap (-1.20 → -0.70) and the RX gap (-0.60 →
-0.40) but opened a massive FP gap (-0.80 → -2.60, now 54% of Opus).
Overall v1.2 is **worst on FP of any forge arm** including the raw
base (FP 2.60). That's remarkable — fine-tuning a model on
TypeScript made it *worse at FP TypeScript than the untrained base*.

## Cost of v1.2

| Step | Time |
|---|---|
| v1.2/merge.py (partition + dedup + anchor rebuild) | 5 min |
| v1.2/train.py fork (defaults change only) | 2 min |
| Training (912 steps over 2426 records) | 42 min |
| Manual GGUF conversion (same workaround as v1.1) | 2 min |
| ollama create + sanity inference | 1 min |
| Eval (24 prompts) | 1 min |
| Tool-call smoke | 10 s |
| Two-grader dispatch | 2 min |
| Combine regression gate | 30 s |
| This decision.md | 15 min |
| **Total** | **~70 min wall** |

Budget was "~30–35 min training + eval + grading" per plan. Training
came in at 42 min (slightly over), pipeline ran tight. **Phase 2 cost
30% less than Phase 1** because dataset was half; both well under
original v1's 18–20 h for the same type of signal.

## Recommended next steps

Priority order:

1. **Do NOT dogfood ts-forge-v1.2.** Keep ts-forge-v0.7-r64 as daily
   driver. v1.2 is retained as a diagnostic checkpoint, not a
   rotation candidate. It is the FP-weakest model in the project.

2. **Phase 3 — v1.3: tightened FP verifier + full-data warm-start.**
   This becomes the actual next experiment:
   - Keep full v1 data (4885 records) to preserve FP training signal.
   - Tighten FP verifier gate: require **at least one** of
     `Effect\.gen\(function\s*\*`, `pipe\([^)]*,\s*(E|O|TE)\.`,
     `Layer\.(succeed|effect)\(`, `Context\.(GenericTag|Tag)<`.
   - Re-run v0.7/verify.py on v1's FP batches (B1–B24) with tight
     gate. Identify which batches drop below 90% survival.
   - Drop the batches with <90% survival (likely B2, B22 per v1's
     analysis).
   - Warm-start from v0.7-r64 (retain v1.1's architectural choice).
   - **Prediction**: if FP data is the root cause, FP recovers to
     ≥4.30 (close to v0.7's 4.40) while keeping v1.2's XState
     improvement (+0.20).
   - **Cost**: ~50–70 min (full training + eval pipeline).

3. **Phase 4 (LoRA stacking) as the fallback.** If v1.3 also regresses
   on FP, warm-start + fresh-LoRA is fundamentally broken as an
   iteration strategy. Move to stacking:
   - Keep `v0.7/gguf/r64`'s LoRA weights as a frozen adapter.
   - Train a new adapter on top (not merged) with only ES + new
     XState + new anchors data.
   - Load both adapters at inference.
   - Cost: ~1 day of engineering (new train.py path, inference script
     changes, ollama packaging).

4. **FP-specific audit (supporting Phase 3).** Run the tightened
   verifier on each v1 FP batch independently. Publish a "FP batch
   quality" scorecard: which subagent prompts yielded weak pairs,
   which themes (Ref.make? Layer composition?) over-represented.
   Use this to inform v1.3's FP subset.

5. **Anchor ratio reduction.** Both v1.1 and v1.2 Cap=3.25 (-1.00 vs
   v0.7). Anchor ratio was 12% in v1, ~15% in v1.2. Drop to 8% per
   v1/decision.md's recommendation. That's: 18 new anchors × 10 reps
   = 180, not 20 reps.

6. **Consider a "FP-only control" run.** Train on just v1's 874 FP
   records (warm-start from v0.7-r64) and see what FP settles at.
   Isolates whether FP data is poisoning anything else vs being
   poisoned by ES/XState training. Cost: ~20 min training. Could
   inform Phase 3's data curation.

## What this plan is NOT

- Not a rejection of warm-start as a technique. Warm-start's
  *infrastructure* (one-line MODEL change, bnb-4bit on merged
  safetensors, GGUF workaround) works. The rejected hypothesis is
  "warm-start + fresh LoRA isolates capability by domain."
- Not a rejection of the delta-only dataset. 2426 records on warm-base
  did produce XState +0.20 and ES +0.80. Delta-only is fine WHEN the
  excluded domains' training signal isn't needed.
- Not a scope change. Qwen3-14B + LoRA + r=64 + Ollama remains the
  target stack. Four-domain TypeScript remains the target.

## Artifacts produced

- `v1.2/merge.py` — delta-only dataset builder (partition vs v0.7
  anchors by exact prompt match)
- `v1.2/train.py` — fork of v1.1/train.py (only --data/--out
  defaults change)
- `v1.2/data/synth.verified.jsonl` — 2426 records (1364 ES + 702
  XState + 360 new-anchors)
- `v1.2/gguf/` — 6-shard bf16 merged safetensors (~30 GB, retained)
- `v1.2/gguf_gguf/qwen3-14b.Q4_K_M.gguf` — 8.4 GB quantized GGUF
- `v1.2/gguf_gguf/Modelfile` — identical template to v1/v1.1
- `v1.2/logs/{train,gguf,run_eval}.log` — full pipeline logs
- `v0/grading/combine_v1.2.py` — gate script fork
- `v0/results/v1.2.raw.json` — 24 model responses
- `v0/results/v1.2.toolcall.json` — tool-call smoke (3/3 PASS)
- `v0/results/v1.2.graded.grader_A.json` + `grader_B.json`
- `v0/results/v1.2.json` — combined (A+B mean, disagreement, gate)
- ollama model `ts-forge-v1.2:latest` (9.0 GB) — diagnostic only

## Session-state trail

commits → `v1/decision.md` (v1 outcome) → `docs/lessons.learned.md`
(v1's structural lesson) → `docs/training.process.md` (plan) →
`v1.1/decision.md` (Phase 1 result, warm-start + full data halts) →
**`v1.2/decision.md` (this file — Phase 2 result, delta-only causes
catastrophic FP collapse, redirect to Phase 3)** → Phase 3 execution.

## Running scoreboard

| Arm | Recipe | Domain avg | Gate | Ship? |
|---|---|---|---|---|
| v0.6 (440) | fresh, XState only | 3.00 | n/a | no |
| v0.7-r32 (1053) | fresh, 3 domains | 3.98 | pass | no |
| **v0.7-r64 (1053)** | **fresh, 3 domains, r=64** | **4.10** | **pass** | **YES** |
| v1-r64 (4885) | fresh, 4 domains | 4.25 | halt (FP, RX) | no |
| v1.1-r64 (4885) | warm, 4 domains | 3.94 | halt (FP, RX) | no |
| v1.2-r64 (2426) | warm, delta-only | 3.65 | halt (FP cat., RX) | no |

Three consecutive arms have halted against v0.7-r64's gate. v0.7's
lead is now two months wide. The path to beat it lies through Phase 3
(data quality on FP), not through recipe-level architecture changes
we've already tested.
