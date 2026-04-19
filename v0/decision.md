# v0 Decision — 2026-04-17

## TL;DR

**Outcome: (C) with a directional signal.** Neither fine-tuned arm clears the
predeclared High threshold (≥ 3.5) on the target domain. Curated *regressed*
versus the base model on XState. Extracted *slightly* improved over base.

**Catastrophic forgetting (B) ruled out** — both arms preserve 3/3 tool calls.

## Numbers

| Domain         | Base | Curated     | Extracted    |
| -------------- | ---- | ----------- | ------------ |
| **xstate** (target) | **1.60** | **1.20** ↓ -0.40 | **1.80** ↑ +0.20 |
| fp             | 2.60 | 2.40 (-0.20) | 2.60 (=) |
| reactive       | 4.00 | 4.00 (=)     | 4.00 (=)     |
| eventsourcing  | 3.00 | 3.00 (=)     | 3.00 (=)     |
| capability     | 3.25 | 3.25 (=)     | 3.25 (=)     |
| **Domain avg** | **2.80** | **2.65** ↓ | **2.85** ↑ |
| **Tool calls** | 3/3  | 3/3          | 3/3          |

Predeclared thresholds (from `v0/readme.md`):
- High: domain prompt avg ≥ 3.5 AND capability avg ≥ 3.5
- Low: any of the above < 3.5

All three arms are **Low** by the predeclared rule. Decision-matrix row:

| Base | Curated | Extracted | Action |
|------|---------|-----------|--------|
| Low  | Low     | Low       | **Reconsider premise.** |

## Why curated regressed

The 43 hand-curated examples (29 v5 patterns + 6 v4→v5 fixes + 8 capability
anchors) caused the model to **invent APIs that don't exist**:

- `xstate-01 curated`: emitted `setup({...}).withConfig({}).states({...})` — neither `.withConfig()` nor `.states()` are methods on `setup()`'s return.
- `xstate-04 curated`: still uses `services:` (v4 keyword) and `(context, event)` v4 callback signature, despite 6 explicit v4→v5 fix examples in the training data.
- `xstate-05 curated` (the v4→v5 conversion test): still outputs `assign({ count: (ctx) => ctx.count + 1 })` with v4 callback signature. The 6 conversion examples did not generalize.

The pattern: fine-tuning on 43 short examples pushed the model toward more *confident* output (longer `<think>` blocks, more decisive final answers) without correcting the v4/v5 confusion. It learned shape (often mis-remembered shape) over substance.

## Why extracted edged out

The 46 extracted examples were full files from `xstate/examples/` — each example
is a complete, real, working v5 machine. The model saw whole machine
definitions in context, not synthetic snippets. The slight improvement on
xstate (+0.20) suggests *exposure to authentic full-file v5 code* is more useful
than carefully designed micro-examples at this scale.

But +0.20 is well within noise of human grading. Don't over-read it.

## What this means for the project

Per the four-row matrix in the v0 plan:

> **(C) Fine-tuned doesn't beat base.** Either 50 examples wasn't enough (try
> 150 hand-curated before concluding), or the extraction-based approach was
> always going to be wrong for your actual use case. **Reconsider the premise.**

Concretely, three things to update beliefs on:

1. **Hand-curating ~50 examples does not work for v5 API correction.** The
   model cannot learn a syntactic distinction (v4 `cond:` vs v5 `guard:`) from
   ~6 contrastive examples. This invalidates one premise of the v0 plan.

2. **Extracted-usage data shows directional promise.** The +0.20 lift came from
   *less curated, more authentic* data. This contradicts the assumption that
   curation quality dominates at small scale.

3. **The base model's reasoning mode (`<think>`) is part of the problem.** Both
   arms produced 5–10x longer outputs than base on XState prompts, with most
   tokens spent in confused `<think>` blocks. The reasoning-mode amplification
   may interact badly with small fine-tunes on a different task surface.

## What was *not* tested by v0

- **A larger dataset (500–5000 examples).** v0 only tested ~50 per arm.
- **Qwen3-Coder-30B-A3B** (the review's recommended base model). v0 used
  qwen3-14b for caching reasons; the choice is documented in `train_v0.py`.
- **A pure code-only training set** (no capability anchors). The 19% anchor
  ratio may have suppressed XState-specific signal too aggressively.
- **Disabling reasoning mode** during inference. Base qwen3 outputs `<think>`
  blocks by default; both arms preserved this.

## Recommended next steps

In priority order:

1. **Don't scale curation as-is.** Per (1) above, more 50-example batches won't
   move the needle on syntactic API correction. If we want fine-tuning to fix
   v4/v5, we need either much more data, or a different signal (e.g., token-level
   DPO over v4 vs v5 outputs).

2. **Test with Qwen3-Coder-30B-A3B as the base.** A code-specialized base may
   not have the v4 leakage problem in the first place. If `qwen3-coder:30b`
   already scores ≥ 4 on these XState prompts (it scored 2.20 in our prior
   baseline — better than qwen3-14b's 1.60 but still not "good"), then the
   right move is "switch base, skip fine-tuning" — which the original review
   anticipated.

3. **Disable `<think>` for v5-style prompts.** Set `enable_thinking=False` or
   strip `<think>` content at inference time. The reasoning-mode confabulation
   is contributing to the failure mode independently of fine-tuning.

4. **Retire the extraction pipeline (`extract_pipeline.py`) for now.** Until
   we have positive evidence that extraction-based fine-tuning works, the
   pipeline infrastructure is solving a problem we can't yet show is the right
   one. Keep the code in `lib/` and `app/` (it's small, well-organised, and
   the dedup/balance/instruct modules are reusable). Just don't run it.

5. **Consider RAG / system-prompt approaches.** If the goal is "correct v5 code
   from a model that might have v4 leakage", retrieving the v5 docs into a
   long system prompt at inference time may be cheaper and more reliable than
   fine-tuning.

## Time spent on v0

| Phase | Time |
|---|---|
| 0 — eval set | 30 min |
| 1 — baseline (qwen3-coder + qwen3-14b) | 10 min runtime + 15 min grading |
| 2a — hand-curate 43 examples | 1 hr |
| 2b — extract 46 examples | 5 min |
| 3 — train both arms | 2 min runtime + 7 min GGUF conversion |
| 4 — eval both arms + tool calls | 15 min runtime + 30 min grading |
| 5 — this decision | 20 min |
| **Total** | **~3 hours** |

## What v0 was worth

It cost ~3 hours and ~$0 in API spend. In return:

- Ruled out outcome (B) catastrophic forgetting — the failure mode v0 was most designed to catch.
- Showed that the simplest hand-curated approach *regresses* on the target domain at this scale — a result we would not have predicted ahead of time.
- Established that the extraction pipeline's underlying premise (*usage* sources beat *internals* sources) is at least directionally correct, even at tiny scale.
- Identified the `<think>` interaction as a confounding variable.

The most important consequence: **we now know fine-tuning is not the cheapest
path to "Claude Code with idiomatic XState v5".** RAG over the v5 docs, or
switching to Qwen3-Coder, are both plausibly stronger options that v0 didn't
preclude — but v0 made it clear the existing pipeline wasn't the answer.
