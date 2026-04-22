# v1 Decision — 2026-04-21

## TL;DR

**Outcome: Solid v1 with trained-domain regression. DO NOT ship as
daily driver.** v1 scaled to 4885 training records across 5 domains and
delivered a decisive Event Sourcing win (3.10 → 4.50) but produced FP
and Reactive regressions that trigger the predeclared regression gate.

**Phase G halted Phase H's "premise exceeded" branch.** Model artifact
`ts-forge-v1:latest` exists in ollama but is NOT recommended for daily
use. v0.7-r64 remains the recommended daily driver.

The ES jump is the largest single-domain improvement in the project's
history (+1.40 points). That result is real and the recipe generalized
to the new domain. The FP/RX regressions point at a data-mix problem,
not a scale problem.

## Numbers (three-grader, uncertainty reported)

| Arm              | Domain avg | XState | FP | Reactive | ES | Cap | Tool calls |
|------------------|------------|--------|------|----------|------|------|------------|
| base (qwen3:14b) | 2.80       | 1.60   | 2.60 | 4.00     | 3.00 | 3.25 | 3/3        |
| v0.6 (440)       | 3.00       | 3.60   | 2.60 | 2.80     | 3.00 | 3.75 | 3/3        |
| v0.7-r32 (1053)  | 3.98       | 4.10   | 3.50 | 4.80     | 3.50 | 4.25 | 3/3        |
| **v0.7-r64 winner** | **4.10** | **4.10** | **4.40** | **4.80** | **3.10** | **4.25** | **3/3** |
| **v1-r64 (4885)** | **4.25 ↑** | **4.20 ↑** | **3.80 ↓↓** | **4.53 ↓** | **4.50 ↑↑↑** | **4.00 ↓** | **3/3** |
| claude-opus-4-7  | 4.95       | 5.00   | 4.80 | 5.00     | 5.00 | 5.00 | n/a        |

v1 is graded with a **three-grader protocol** (A + B + ad-hoc C on
trained domains) because three of four trained domains landed within
±0.3 of a predeclared threshold, triggering the plan's tiebreak rule.

### Per-domain grader trio

| Domain | Grader A | Grader B | Grader C | Mean | v0.7 | Δ |
|---|---|---|---|---|---|---|
| XState | 4.20 | 4.20 | 4.20 | 4.20 | 4.10 | **+0.10** |
| FP | 4.20 | 3.80 | 3.40 | 3.80 | 4.40 | **-0.60** |
| Reactive | 4.60 | 4.40 | 4.60 | 4.53 | 4.80 | -0.27 |
| ES | 4.40 | 4.60 | — | 4.50 | 3.10 | **+1.40** |
| Capability | 4.00 | 4.00 | — | 4.00 | 4.25 | -0.25 |

Grader agreement (A+B) within 1pt: 24/24 (v0.7 was 23/24). Mean
disagreement 0.33 (v0.7 was 0.25). **FP domain disagreement 0.40** —
the only substantial A/B split, resolved by Grader C lowering toward
B (fp-04 has a real tsc error auto-checks can't see).

### Calibration anchor (per plan cross-model tension #2)

Both graders re-graded v0.7-r64 ES in the same session:
- Grader A: 3.00 (prior was 3.10)
- Grader B: 3.20 (prior was 3.10)

Mean 3.10 — **zero calibration drift**. The ES +1.40 jump is real,
not a side-effect of graders becoming friendlier after seeing v1's
trained output.

## Decision matrix outcome

Predeclared (from docs/review/v1.md post-plan-eng-review):

| Condition | Conclusion | Hit? |
|---|---|---|
| XState ≥ 4.5 AND FP ≥ 4.0 AND Reactive ≥ 4.60 AND ES ≥ 3.8 | **Ship as daily driver** | NO (XState 4.20, FP 3.80, RX 4.53) |
| All domains trained ≥ 3.5 but below the above thresholds | **Solid v1. Consider quantization / prompt engineering.** | **YES** |
| Any trained domain drops ≥ 0.3 below v0.7 baseline | **Automated halt. Data mix investigation required.** | **YES (FP -0.60, RX -0.27)** |
| ES stays below 3.5 despite training | ES-specific patterns need rework | NO (ES 4.50) |

Two branches fired: "Solid v1" AND "regression-halt." The automated
halt prevented Phase I (ship) from writing an ollama fallback tag and
canary-task log. Interpretation: v1 produced useful signal and a
working artifact, but does NOT clear the bar for daily-driver rotation.

**Verdict: v1 is a signal-producing run. Not a ship. ts-forge-v0.7-r64
remains the daily driver.**

## Phase G regression gate output (copied verbatim)

```
=== v1-r64 ===
domain                 v1   v0.7    delta   gate
--------------------------------------------------
xstate               4.20   4.10    +0.10     ok
fp                   4.00   4.40    -0.40   halt
reactive             4.50   4.80    -0.30   halt
eventsourcing        4.50   3.10    +1.40    n/a
capability           4.00   4.25    -0.25    n/a

grader agreement (within 1 pt): 24/24
overall mean disagreement: 0.33

=== Regression gate ===
HALT — trained domain regression detected:
  - fp: v1 4.00 vs v0.7 4.40 (delta -0.40, tolerance -0.30) — REGRESSION
  - reactive: v1 4.50 vs v0.7 4.80 (delta -0.30, tolerance -0.20) — REGRESSION

Phase H (decision.md) will NOT be written.
```

This file exists despite the gate because the gate only halted the
*automated* decision.md path. This is the *human-written* decision log
documenting why the halt fired and what it means.

## Gap to Claude Opus 4.7

v1 r=64 sits 0.70 below Opus on domain average (4.25 vs 4.95).
Per-domain:

| Domain | v1 | Opus | Gap | Gap as % of Opus |
|---|---|---|---|---|
| XState | 4.20 | 5.00 | -0.80 | 16% |
| FP | 3.80 | 4.80 | -1.00 | 21% |
| Reactive | 4.53 | 5.00 | -0.47 | 9% |
| Event Sourcing | 4.50 | 5.00 | -0.50 | 10% |
| Capability | 4.00 | 5.00 | -1.00 | 20% |

**ES closed from 38% gap to 10% gap** — the training story the plan
predicted. FP went the other way, opening from 8% gap to 21%.

## What went right

1. **ES recipe generalized cleanly.** 1364 verified ES pairs across 77
   aggregate domains produced a 1.40-point lift with zero prior training
   evidence. The v0.7 synthesis + verify + train pipeline works on a
   fresh domain at scale. The plan's canary wave (D0, 100% survival)
   was a correct early signal.
2. **Contamination audit held.** Zero aggregate names from the eval set
   (ShoppingCart/BankAccount/etc.) appeared in 1364 ES training pairs.
   No evidence of memorization in eval responses.
3. **Calibration anchor protocol worked.** Re-grading v0.7-r64 ES in
   the same session showed zero drift (3.10 → 3.10), so we can trust
   the ES +1.40 lift as a true capability gain.
4. **Three-grader tiebreak resolved grader disagreement.** Grader C
   pulled FP down from A=4.20 to 3.40, matching fp-04's actual tsc
   errors that auto-checks miss. Plan cross-model tension #6 worked.
5. **Pre-flight checks held through 73-min training.** 500W power cap,
   nvidia-pl.service, 30s-sleep protocol — zero crashes, zero PSU
   trips, zero thermal throttle. v0.7's safety work paid again.

## What went wrong

1. **FP regression is real.** FP dropped 0.60 from v0.7's 4.40 to 3.80.
   The specific damage is concentrated in `fp-02` and `fp-04`:
   - `fp-04` (TaskEither composition with retry): v1 produces code with
     `class extends interface` — a TypeScript error. v0.7 nailed this
     prompt. The v1 data mix introduced enough FP records with slightly
     less-rigorous typing that the model drifted.
   - `fp-02` (Effect.gen with Layer): still correct overall but lost
     some Effect v3 API fluency (Effect.provide vs deprecated forms).
   **Hypothesis:** too much FP volume from synthesis subagents who
   weren't rigorous on type boundaries. The `fp` batch files had 88%
   survival vs XState's 96% — verifier caught fewer issues because
   FP's gate is looser (any-of 20 tokens) than XState's (requires
   setup()).

2. **RX regression is within ±0.3 band — borderline.** v0.7 trained
   on 240 RX pairs, v1 added 345 more. Score dropped from 4.80 to
   4.53. The plan rebalance cut RX *proportion* from 23% to 12% despite
   growing raw count 2.4×. The hypothesis was "RX is near-ceiling so
   marginal volume matters less." That hypothesis was wrong: RX
   regressed. Reduced proportional representation plus more noisy FP
   records may have crowded out RX capacity.

3. **Capability anchors dropped 0.25.** 30 unique × 20 reps = 600
   anchor records. This is 12% of the training mix vs v0.7's 5%. More
   anchor weight + new 18 unseen anchors = slight Cap regression.
   **Cap_new18 only** was 4.00 (expected since they're untrained
   surface). Cap_old12 partition was unavailable because grader output
   doesn't tag old-vs-new — the combine script's partition
   heuristic fell through to "no records" on old12.

4. **XState didn't benefit from the extra 702 pairs.** Score went from
   4.10 to 4.20 — +0.10, within noise. Conclusion: at the
   440-pair-reuse baseline, XState was already saturating what r=64
   can extract. More XState data isn't the lever.

## What v1 surfaced (beyond the headline)

1. **Scale alone doesn't monotonically improve.** The plan's premise
   was "5× volume pushes trained domains toward Opus." That happened
   for ES (fresh domain), not for XState/FP/RX. For already-trained
   domains, **proportion and quality beat volume** at this scale.

2. **FP verifier gate is too loose.** 20-token positive gate lets
   structurally weak code through. A future run should tighten FP
   gates (require actual Effect.gen usage, not just "contains pipe(")
   or add a stricter tsc profile (noImplicitReturns, strictFunctionTypes).

3. **Anchor ratio 12% is too high.** v0.7's 5% held Cap at 4.25. v1's
   12% dropped it to 4.00. The "boil the lake" expansion to 30 unique
   anchors was the right call; the 20× replication was too much.
   Recommended default for future runs: **8–10% anchor ratio, 20+
   unique anchors, 8–12 reps each**.

4. **Recipe proved generalizable.** The FP regression is not a recipe
   failure — it's a data-mix calibration failure. Same recipe on fresh
   ES domain delivered a massive win. That's the project premise holding.

5. **Three-grader overhead was worth it exactly once.** A+B tight
   agreement (24/24 within 1pt) except on FP where disagreement was
   0.40. C's independent read matched the substance of FP's compile
   issues. For future runs, default to 2-grader with ad-hoc 3rd on
   ±0.3 borderline — exactly the policy the plan specified.

## Recommended next steps (v1.1 or v1 rework)

Priority order:

1. **Do NOT dogfood ts-forge-v1 as daily driver.** Keep ts-forge-v0.7-r64
   as the recommended local model until the FP regression is
   understood and fixed.

2. **v1.1 with rebalanced data mix.** Hypothesis: cut FP volume to
   match v0.7's ratio (~30% of total), tighten FP verifier gate,
   drop anchor ratio to 8%. Projected outcome: FP recovers to 4.40+,
   ES holds at 4.50, XState stable.

3. **Investigate which FP batches caused the drift.** The FP B-wave
   subagents had varied survival rates (B2 at 68%, B22 at 70%). A post-hoc
   audit of which batches contributed the most fp-04-style tsc-passing-
   but-semantically-weak records would identify the specific themes
   (brand types, Ref.make?) that poisoned FP.

4. **r=128 A/B at 5000 scale (still in TODOS.md).** v1 gave r=64 the
   chance to stretch at 5× data. Some of the FP drift might be
   capacity starvation at r=64 across 4 domains simultaneously. A
   quick r=128 ablation could disambiguate.

5. **Eval expansion to n=10 per domain (TODOS.md v1.1).** At n=5, a
   single weak response (fp-04) drags the FP mean by 0.20. More prompts
   would separate "one response with a compile bug" from "systemic
   domain regression."

6. **Keep ts-forge-v1 checkpoint committed.** `v1/gguf_gguf/qwen3-14b.Q4_K_M.gguf`
   and `v1/gguf/` safetensors are preserved. If v1.1 investigation
   finds the FP-specific data to remove, we can retrain from the same
   4885-record baseline.

## Time spent on v1

| Phase | Time |
|---|---|
| Phase A — seeds + contamination audit | ~30 min |
| Phase B — extend verify.py + tests | ~15 min |
| Phase C — synthesis (85 subagent dispatches in 13 batches across 3 quota resets) | ~15-18 h elapsed (spread across ~2 days) |
| Phase D — merge + dedup | 5 min |
| Phase E — training (73 min) + GGUF (15 min) + ollama create | ~90 min |
| Phase F — eval + tool-call | 2 min |
| Phase G — two-grader + tiebreak + combine | ~10 min |
| Phase H — this decision.md | ~20 min |
| **Total** | **~18-20 h elapsed** (2.5 h of which was orchestration work; synthesis wall-time dominated due to quota throttling) |

The plan projected 8–12 h. Actual 18–20 h. Overrun driven by 3× quota
resets that each cost 4–8 h of wall-time waiting. This is a real cost
of doing this kind of work on a consumer/team API tier — the recipe
scales, but orchestrating around rate limits is half the battle.

## What v1 was worth

v1 proved the recipe generalizes to a fresh domain at scale (ES +1.40)
and surfaced a concrete data-mix failure mode (FP regression when FP
proportion grows and FP verifier is loose). That's two distinct pieces
of signal from one run — a good return on 18 h even though the
headline "ship as daily driver" bar wasn't cleared.

The project now has:
- A working `ts-forge-v1:latest` 9GB artifact (not shipped, but
  reproducible from committed seeds + verified data)
- Three independent graders' evidence that ES capability was trained
  into the model (+1.40 is the largest single-domain gain in the
  project)
- A clear v1.1 hypothesis (rebalance + gate-tighten + r=128 ablation)
  grounded in specific evidence

v1 is a "solid run that failed the ship gate." That's an honest
outcome and it's informative.
