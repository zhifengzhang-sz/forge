# v1.1 Decision — 2026-04-22

## TL;DR

**Outcome: warm-start from v0.7-r64 did NOT eliminate v1's trained-domain
regressions. Phase 1 hypothesis is partially falsified.** Same data,
same recipe, same r=64, only MODEL path changed. Gate halts on FP and
Reactive — the exact two domains that halted v1. DO NOT ship
`ts-forge-v1.1:latest` as daily driver.

The gate halted in the *same pattern* as v1, but at *different
magnitudes*. FP improved (v1 3.80 → v1.1 4.00, still below 4.10
tolerance); Reactive unchanged at the halt line; XState worsened; ES
gave back most of v1's gain; Capability worsened substantially. Net
overall mean 3.94 vs v1's 4.25 — **warm-start performed *worse* than
fresh-from-base on this eval.**

ts-forge-v0.7-r64 remains the daily driver.

## Numbers (two-grader — tiebreak not triggered)

| Arm | Domain avg | XState | FP | Reactive | ES | Cap | Tool calls |
|---|---|---|---|---|---|---|---|
| base (qwen3:14b) | 2.80 | 1.60 | 2.60 | 4.00 | 3.00 | 3.25 | 3/3 |
| **v0.7-r64 winner** | **4.10** | **4.10** | **4.40** | **4.80** | **3.10** | **4.25** | **3/3** |
| v1-r64 (fresh, 4885) | 4.25 ↑ | 4.20 ↑ | 3.80 ↓↓ | 4.53 ↓ | 4.50 ↑↑↑ | 4.00 ↓ | 3/3 |
| **v1.1-r64 (warm, 4885)** | **3.94 ↓↓** | **3.80 ↓** | **4.00 ↓↓** | **4.40 ↓↓** | **4.10 ↑↑** | **3.25 ↓↓** | **3/3** |
| claude-opus-4-7 | 4.95 | 5.00 | 4.80 | 5.00 | 5.00 | 5.00 | n/a |

### Per-domain grader pair

| Domain | Grader A | Grader B | Mean | v0.7 | Δ | gate |
|---|---|---|---|---|---|---|
| XState | 3.80 | 3.80 | 3.80 | 4.10 | -0.30 | ok (at tolerance) |
| FP | 4.00 | 4.00 | 4.00 | 4.40 | **-0.40** | **halt** |
| Reactive | 4.20 | 4.60 | 4.40 | 4.80 | **-0.40** | **halt** (tol -0.20) |
| ES | 4.40 | 3.80 | 4.10 | 3.10 | +1.00 | n/a |
| Capability | 3.25 | 3.25 | 3.25 | 4.25 | -1.00 | n/a |

Grader agreement: **24/24 within 1 pt**, mean disagreement **0.29**
(tighter than v1's 0.33). Zero 2-point disagreements → three-grader
tiebreak was not triggered. ES had the largest domain-level grader
spread (A 4.40 vs B 3.80 = 0.60) but within-prompt disagreement was
all ≤1pt.

## Decision matrix outcome

Applying the predeclared v1-style matrix (since Phase 1 kept the gate
identical):

| Condition | Conclusion | Hit? |
|---|---|---|
| XState ≥ 4.5 AND FP ≥ 4.0 AND Reactive ≥ 4.60 AND ES ≥ 3.8 | Ship | NO |
| All trained ≥ 3.5, below ship thresholds | Solid, don't ship | **YES** |
| Any trained drops ≥ 0.3 below v0.7 | Automated halt | **YES (FP -0.40, RX -0.40)** |
| ES stays below 3.5 | ES rework needed | NO (ES 4.10) |

Same two branches fire as v1: "Solid" AND "halt." Plus a
new-for-v1.1 pattern: **Phase 1's own failure-mode map fires on "FP
lands between 3.80 and 4.40 → partial fix; both warm-start and
data-mix issues contribute."**

## Phase G regression gate output (copied verbatim)

```
=== v1.1-r64 ===
domain               v1.1   v0.7    delta   gate
--------------------------------------------------
xstate               3.80   4.10    -0.30     ok
fp                   4.00   4.40    -0.40   halt
reactive             4.40   4.80    -0.40   halt
eventsourcing        4.10   3.10    +1.00    n/a
capability           3.25   4.25    -1.00    n/a

capability breakdown:
  Cap_old12: (no records)
  Cap_new18: 3.25

grader agreement (within 1 pt): 24/24
overall mean disagreement: 0.29

=== Regression gate ===
HALT — trained domain regression detected:
  - fp: v1.1 4.00 vs v0.7 4.40 (delta -0.40, tolerance -0.30) — REGRESSION
  - reactive: v1.1 4.40 vs v0.7 4.80 (delta -0.40, tolerance -0.20) — REGRESSION
```

## Phase 1's hypothesis vs actual

The plan (docs/training.process.md §Phase 1) predicted:

| Domain | Predicted | Actual | Miss |
|---|---|---|---|
| XState | ~4.20 (holds) | 3.80 | **-0.40** |
| FP | ~4.40 (back to v0.7) | 4.00 | **-0.40** |
| Reactive | ~4.80 (back to v0.7) | 4.40 | **-0.40** |
| ES | ~4.50 (holds v1 gain) | 4.10 | **-0.40** |
| Overall | ~4.50 | 3.94 | **-0.56** |

Every domain missed the prediction by ~-0.40. Not noise — a **systematic
bias** in the hypothesis. Warm-starting + training on the same full
data does *not* preserve the base's proven capability; the new LoRA
actively retrains over it.

## What happened — the structural finding

**Warm-start does not isolate new learning.** The Phase 1 plan framed
warm-start as "frozen merged base + new LoRA learns only delta." In
practice:

- LoRA modifies all targeted projections (q/k/v/o/gate/up/down) across
  all 40 layers. It is a **global** additive weight modification, not
  a local patch.
- The new LoRA's only training signal is match-the-data. If the
  training set contains FP pairs (320 pairs in v1's mix), the LoRA
  **retrains FP** whether v0.7 already nailed FP or not.
- Because the training FP pairs include noisy records (B2 at 68%
  survival, B22 at 70%), the retrained FP is weaker than v0.7's
  baseline — exactly the v1 failure mode, reproduced through the
  warm-start lens.
- The same logic applies to RX, XState, and Capability.

This is the **premise error** in Phase 1: "warm-start baked-in
capability" was conceptually wrong. The base's weights *are* frozen in
memory during training, but the LoRA's deltas add onto those weights,
and the LoRA's gradient is driven entirely by whatever data you feed
it. Re-feeding FP data = re-learning FP = susceptible to the same
noise v1 hit.

**Corollary:** Phase 2 (delta-only training) isn't an optimization.
It's the actual fix. If v0.7's base knows FP at 4.40, you preserve
that by **not training on FP at all** — let the frozen base handle FP
inference and only LoRA-patch the new domain (ES).

## What went right

1. **Warm-start mechanics work end-to-end.** Unsloth loaded v0.7's
   merged safetensors, bnb-4bit-quantized them at load, attached a
   fresh r=64 LoRA, trained 1833 steps, produced a valid 9GB GGUF.
   One-line change (`MODEL = str(REPO_ROOT / "v0.7/gguf/r64")`) was
   sufficient from an infrastructure standpoint.
2. **Training loss profile confirms warm-start signature.** Opened at
   ~0.38 at epoch 0.7 (fresh v1 opened ~1.5), converged to ~0.18 by
   epoch 3.0. Exact "start near convergence, small delta" curve the
   plan predicted — the *training dynamics* were correct; it was the
   *capability preservation inference* that was wrong.
3. **FP improved vs v1.** v1 4.20/3.80/3.40 → v1.1 4.00/4.00. v1's
   B2/B22 noise is still in the data but the frozen good-FP base
   lifted the floor from 3.80 to 4.00. Partial win.
4. **Tool-calling preserved: 3/3 PASS** (list_files, read_file ×2).
   Consistent with v1 and v0.7; tool-use is robust across recipes.
5. **Grader agreement tightened.** 0.29 mean disagreement vs v1's
   0.33, zero 2-point splits. Two-grader protocol is holding.
6. **GGUF conversion workaround.** Unsloth's `save_pretrained_gguf`
   failed on lazy-download of `convert_hf_to_gguf_original_gguf_*.py`,
   but the pre-built `~/.unsloth/llama.cpp/` has the tools we need.
   Manual `convert_hf_to_gguf.py --outtype f16` → `llama-quantize Q4_K_M`
   → Modelfile copy worked in ~2 minutes (see v1.1/logs/gguf.log).

## What went wrong

1. **FP stayed below gate.** 4.00 vs 4.10 tolerance. The failure mode
   the plan anticipated — "lands between 3.80 and 4.40" — fired.
   Contributing signal: fp-02 in v1.1 dropped from v1's 5 to 2
   (Effect.tryPromise misuse; wrapped an Effect-returning fn in a
   Promise thunk wrapper). Both graders flagged independently.
2. **Reactive regressed identically to v1.** 4.40 vs 4.60 tolerance,
   -0.40 delta. rx-03 specifically dropped from v1's 5 to 3 (both
   graders). Warm-start gave zero headroom here.
3. **XState went the wrong way.** v1 was +0.10 vs v0.7; v1.1 is -0.30.
   Specifically xstate-01 dropped from v1's 4 to 3 (both graders).
   xstate-04's `fromObservable` confusion (sendBack+setTimeout pattern)
   persisted from v1 — the new LoRA didn't clean it up.
4. **Capability regressed -1.00 vs v0.7.** Worst single-domain drop in
   v1.1. cap-02 collapsed to 1/1 (ignored the Read-tool instruction,
   produced fabricated Deno code with nonexistent imports). Anchors
   were 12% of training mix; still too high, and warm-start didn't
   help because the LoRA re-learned anchor responses from the same
   noisy training set.
5. **ES gave back 0.40 of v1's +1.40 gain.** 4.50 → 4.10. The 1364
   verified ES records are the same; the difference is that the fresh
   base had zero ES prior, while v0.7's warm base had 3.10 baseline —
   different starting conditions, and the new LoRA on warm base
   struggled to cement the patterns as hard.

## What v1.1 surfaced (beyond the headline)

1. **Warm-start is not a capability-preservation mechanism.** It is a
   training-init mechanism. The two concepts got conflated in the
   Phase 1 plan. Documented correction: `docs/training.process.md`
   Phase 1 should be retitled "warm-start + delta data" and require
   Phase 2's data restriction to test the *actual* preservation
   hypothesis.
2. **LoRA is global, not local.** r=64 on all q/k/v/o/gate/up/down
   across 40 layers = 256.9M trainable params spread across every
   attention and MLP in the network. Training on domain X's data
   perturbs representations for domains Y and Z too. The
   Phase 1 mental model — "LoRA learns the ES delta, everything else
   is frozen" — is architecturally incorrect.
3. **The "re-learn FP worse than v0.7" pattern is reproducible.** v1
   hit it; v1.1 hit it less severely but still hit it. The common
   factor is the 320-record FP set in `v1/data/synth.verified.jsonl`.
   Direct consequence: until that FP subset is either tightened (Phase
   3 verifier gate) or dropped (Phase 2 delta-only), FP regression
   will repeat on every Qwen3-14B r=64 run that includes it.
4. **Grader drift between sessions is real but small.** v1 was graded
   in one session (three-grader); v1.1 in a fresh session (two
   independent subagent graders). Inter-session drift appears to be
   ~±0.2 per domain mean. Too small to explain v1.1's -0.31 overall
   decline, but enough that single-digit deltas between arms should
   be treated as noise unless >0.3.
5. **Three-grader not triggered here.** v1 needed a third grader
   because FP had A=4.20/B=3.40 (0.80 domain spread and 0.40 mean
   disagreement). v1.1 has tighter grader coherence — ES A=4.40/B=3.80
   is the biggest domain spread (0.60), but per-prompt disagreements
   never exceeded 1pt. Two-grader sufficed.
6. **The warm-start training curve's shape is diagnostic.** Loss
   starting at ~0.38 and flat-declining to ~0.18 is *signature of
   successful warm-start at the init level* — the base really was
   already good. The model's output quality regression therefore can't
   be blamed on a failed warm-start; it came from LoRA-level
   overfit-to-noisy-data.

## Gap to Claude Opus 4.7

| Domain | v1 | v1.1 | Opus | v1.1 gap | v1.1 as % of Opus |
|---|---|---|---|---|---|
| XState | 4.20 | 3.80 | 5.00 | -1.20 | 24% |
| FP | 3.80 | 4.00 | 4.80 | -0.80 | 17% |
| Reactive | 4.53 | 4.40 | 5.00 | -0.60 | 12% |
| Event Sourcing | 4.50 | 4.10 | 5.00 | -0.90 | 18% |
| Capability | 4.00 | 3.25 | 5.00 | -1.75 | 35% |

v1.1 closed the FP gap slightly vs v1 (-1.00 → -0.80) but opened gaps
on XState (-0.80 → -1.20), ES (-0.50 → -0.90), and Capability (-1.00 →
-1.75). Overall further from Opus than v1 was.

## Cost of v1.1

| Step | Time |
|---|---|
| Fork v1/train.py → v1.1/train.py (one-line MODEL change) | 5 min |
| Pre-flight acceptance checks (five criteria from plan) | 5 min |
| Training (step 1/1833 → 1833/1833) | 69 min |
| GGUF conversion (manual workaround — f16 convert + Q4_K_M quantize) | 2 min |
| ollama create ts-forge-v1.1 + smoke test | 2 min |
| Eval (24 prompts via run_eval.py) | 1 min |
| Tool-call smoke (3 prompts) | 10 s |
| Two-grader dispatch (parallel subagents) | 3 min |
| Fork + run combine_v1.1.py | 2 min |
| This decision.md | 15 min |
| **Total** | **~100 min wall** |

The plan's Phase 1 estimate was "~2 h wall." Actual ~1h 40min, within
budget. **Phase 1 was the cheapest meaningful test and it delivered
actionable signal within the estimated window.** Even though the
result is a negative, the signal is high-quality.

## Recommended next steps

Priority order:

1. **Do NOT dogfood ts-forge-v1.1 as daily driver.** Keep
   `ts-forge-v0.7-r64:latest` as the recommended local model. v1.1 is
   retained as a diagnostic checkpoint, not a rotation candidate.

2. **Phase 2 — v1.2: delta-only training.** This is now the critical
   test, not an optimization. Build a training set that contains
   *only* what v0.7 didn't train on:
   - **Include:** 1364 verified ES pairs + 702 NEW XState pairs (v0.6
     didn't have these) + 18 NEW anchors × 8–10 reps = ~144 anchor
     records.
   - **Exclude:** all 320 FP pairs, all 585 RX pairs, all 12 OLD
     anchors.
   - **Target size:** ~2210 records (vs 4885 in v1/v1.1).
   - **Prediction:** if warm-start preserves capability when not
     re-trained, FP and RX hold at v0.7's 4.40/4.80. ES holds at
     ~4.00+. If *this* still regresses, warm-start is truly broken
     (not just data-mix-sensitive).
   - **Cost:** ~30–35 min training (half of v1.1's 69 min).

3. **If Phase 2 also regresses on FP/RX**, the warm-start mechanism
   itself is suspect. Fall back to **LoRA stacking** (docs/training.
   process.md §Open questions): keep v0.7's LoRA frozen, train a new
   adapter alongside. More complex but clean capability isolation.

4. **Investigate xstate-04's fromObservable pattern.** It scored 2 in
   both v1 and v1.1. The training set clearly contains examples where
   `fromObservable` is used as a sendBack+setTimeout shim. Audit which
   synthesis batch produced those and drop or rewrite them. Grep
   v1/data/synth.verified.jsonl for `fromObservable` uses.

5. **Phase 3 (verifier tightening) becomes less urgent.** If Phase 2
   preserves FP via data-exclusion, we don't need to ship a tighter FP
   verifier gate immediately — it's still valuable, but the
   regression root cause was data inclusion, not verifier slack.

6. **Three-grader not triggered; two-grader is calibrated.** For
   future arms, continue default two-grader with ad-hoc grader C on
   ≥0.3 per-prompt disagreement. v1.1 had none such.

7. **Preserve artifacts.** `v1.1/gguf_gguf/qwen3-14b.Q4_K_M.gguf` +
   `v1.1/gguf/*.safetensors` are retained. If Phase 2's result
   suggests data-mix rebalancing can recover specific v1.1 domains,
   we can re-train from the same warm-start base with adjusted data.

## What v1.1 was worth

Phase 1 cost ~100 minutes and the returned signal is:

1. **Warm-start, at the infra level, works.** The training dynamics
   are correct (loss curve, memory budget, GGUF export, inference
   quality).
2. **Warm-start, at the capability-preservation level, does NOT work
   when the training data still contains the preserved domains.**
   This kills the Phase 1 premise as written and redirects effort to
   Phase 2 (delta-only data) as the actual preservation test.
3. **The FP regression in v1 was not primarily a fresh-base problem.**
   It re-appeared under warm-start with less magnitude but same sign.
   This confirms the FP data itself contains the issue — specific
   batches need auditing.
4. **ts-forge-v0.7-r64 remains the best daily driver.** Two arms have
   now attempted to beat it and two arms have halted. The v0.7 recipe
   + r=64 + 1053 records is still the best known point. The burden of
   proof rests with Phase 2 to beat it.

v1.1 is a legitimate negative result. Negative results at 100 minutes
each are cheap signal compared to 18–20 hours for v1. This is the kind
of result Phase 1 was built to produce: fail fast, inform what to try
next, protect v0.7-r64 from being displaced by an unverified arm.

## What this plan is NOT

- Not a rejection of warm-start. Warm-start remains useful for Phase 2
  (delta-only) where its one real benefit — not having to re-learn
  existing patterns — actually applies.
- Not a rejection of the v1 data. The 4885 records are still good for
  ES training. They just shouldn't be the entire training set for a
  warm-started run where FP/RX preservation matters.
- Not a scope or architecture change. Qwen3-14B + LoRA + r=64 + Ollama
  remains the target stack.
- Not a ship gate relaxation. Gate tolerances for Phase 2 should stay
  at v1's levels (xstate/fp -0.3, rx -0.2).

## Artifacts produced

- `v1.1/train.py` — one-line fork of v1/train.py, MODEL changed to
  absolute `v0.7/gguf/r64` path
- `v1.1/gguf/` — 6-shard bf16 merged safetensors (~30 GB, retained)
- `v1.1/gguf_gguf/qwen3-14b.Q4_K_M.gguf` — 8.4 GB quantized GGUF
- `v1.1/gguf_gguf/Modelfile` — identical template to v1
- `v1.1/logs/train.log` — full trainer output (loss curve, step
  timing)
- `v1.1/logs/gguf.log` — manual conversion workaround output
- `v1.1/logs/run_eval.log` + `toolcall.log` — eval + smoke logs
- `v0/grading/combine_v1.1.py` — combine script fork
- `v0/results/v1.1.raw.json` — 24 model responses
- `v0/results/v1.1.toolcall.json` — tool-call smoke results (3/3 PASS)
- `v0/results/v1.1.graded.grader_A.json` + `grader_B.json` — per-prompt
  scores + rationales, independent two-grader protocol
- `v0/results/v1.1.json` — combined (A+B mean, disagreement, gate)
- `ollama` model `ts-forge-v1.1:latest` (9.0 GB)

## Session-state trail

commits → `v1/decision.md` (v1 outcome) → `docs/lessons.learned.md`
(v1's structural lesson) → `docs/training.process.md` (plan) →
**`v1.1/decision.md` (this file — Phase 1 result, hypothesis
partially falsified, redirect to Phase 2)** → Phase 2 execution.
