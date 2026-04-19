# TODOS

Captured deferred work from plan reviews. Each entry: what, why, pros/cons,
context, dependencies.

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
