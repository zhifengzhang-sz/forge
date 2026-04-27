# Cross-Version Conclusions — v0 through v3

**Date span**: 2026-04-15 (v0 starter) → 2026-04-25 (v3 final)
**Substrate**: Qwen3-14B (bnb-4bit) on RTX 5090 (32 GB VRAM)
**Toolchain**: Unsloth + transformers + trl + peft + bitsandbytes, llama.cpp for GGUF, ollama for inference
**Eval**: 24-prompt suite at `tools/eval/suite.json` (5 xstate / 5 fp / 5 rx / 5 es / 4 cap), blind two-grader Claude-subagent protocol

This document is the keystone synthesis across the entire investigation
arc. It captures the lessons that survived multiple iterations and the
mistakes that took multiple iterations to identify. Per-version detail
lives in each version's `decision.md`; the v3 phase specifically is
documented in `docs/findings.v3.conclusions.md`.

---

## The arc in one paragraph

We started in v0 trying to fine-tune Qwen3-14B on a small TypeScript code
corpus. v0.6 and v0.7 established the atomic-drill synthesis discipline
that worked, with v0.7-r64 shipping as the daily driver at FP=4.40 /
RX=4.80 / XState=4.10 (historical scores, in their original grader
session). v1 added Event Sourcing and discovered FP regression; v1.1 and
v1.2 attempted to fix it via warm-start from v0.7's merged weights and
falsified that hypothesis catastrophically (v1.2 FP collapsed to 2.20).
v2.0 reverted to fresh-from-base joint multi-task at r=64 and "halted"
against the historical v0.7 floor — but v3 then revealed this halt was a
measurement artifact: re-evaluating v0.7-r64 in the same grader session
as v2.0 showed v0.7's true contemporaneous FP was ~3.90, not 4.40, and
v2.0's joint training actually beat v0.7 across all four trained
domains. v3 then explored capacity scaling (vanilla r=128, rsLoRA r=128,
data-augmentation, LoRA stacking) and found a capacity wall: at fixed
parameter budget, single-LoRA can max some domains but not all. The
final daily driver `ts-forge-v3:latest` (=v3.0 vanilla) beats Gemma 4
31B by +0.37 under the user's FP+Reactive-style priority lens.

---

## Versions in chronological order

| Version | Date | Recipe | Key result | Decision doc |
|---|---|---|---|---|
| v0 | 2026-04-15 | r=32, ad-hoc seeds | Directional baseline; tooling shakedown | `v0/decision.md` |
| v0.6 | 2026-04-17 | r=32, atomic XState drill | XState 1.6 → 4.0 (partial win) | `v0.6/decision.md` |
| v0.7 | 2026-04-19 | r=64, atomic 3-domain drill | XState=4.10 / FP=4.40 / RX=4.80 (shipped) | `v0.7/decision.md` |
| v1 | 2026-04-21 | r=64, fresh-base + 4885 records (v0.7 + ES + new XState/FP/RX) | FP regressed 4.40 → 3.80 | `v1/decision.md` |
| v1.1 | 2026-04-22 | r=64, warm-start from v0.7 + same data | FP partial recovery 3.80 → 4.00, still halted | `v1.1/decision.md` |
| v1.2 | 2026-04-22 | r=64, warm-start + delta-only data | FP CATASTROPHIC 4.40 → 2.20 (under base) | `v1.2/decision.md` |
| v2.0 | 2026-04-22 | r=64, fresh-base + 2604 atomic records | "HALT" vs ossified v0.7 floor | `v2/decision.md` |
| v3.0 | 2026-04-24 | r=128 vanilla, same v2.0 corpus | Best both-domain (FP+Reactive-style) avg | — |
| v3.0-rslora | 2026-04-24 | r=128 rsLoRA, same data | Best XState (4.40), FP collapsed (3.40) | — |
| v3.1 | 2026-04-24 | r=128 rsLoRA, +280 FP records | FP recovered (4.20), other domains -0.5 each | — |
| v3.2 | 2026-04-25 | LoRA stacking (frozen v3.0-rslora + new fp_recovery) | FP REGRESSED — stacking failed | — |
| **v3 daily driver** | 2026-04-25 | = v3.0 vanilla, alias `ts-forge-v3:latest` | **Beats Gemma 4 31B by +0.37 on FP+Reactive-style** | `docs/findings.v3.conclusions.md` |

---

## Methodology evolution (the most-load-bearing lesson)

The single most important thing we learned over this arc isn't about
LoRA or training data — it's about **measurement methodology**. Five
mistakes, each surfaced by being burned by them once, then corrected.

### Mistake 1: Floor gates against historical scores

For v0 through v2.0, every arm gated against "v0.7-r64's historical
floor" — XState ≥4.10 minus tolerance, FP ≥4.40 minus tolerance, etc.
These numbers came from a single grader session on v0.7's ship day.
They were never re-validated.

**What we found in v3**: re-evaluating v0.7-r64 with current graders
showed FP=3.90, not 4.40 (cross-session drift of −0.50). At least three
of the v1, v1.1, v1.2, v2.0 "halts" were measurement artifacts, not
real regressions. We had been comparing new arms to numbers that
contained 0.5–0.9 of non-reproducible session drift.

**Correction**: contemporaneous baseline. Every grader session that
produces ship-gate data must include the reference arm AND the
strongest off-the-shelf alternative on the same suite, with the same
graders, in the same parallel batch. Compute deltas, not absolute
thresholds. `tools/gate/combine_delta.py` enforces this.

### Mistake 2: Never measuring the untrained base

For v0 through v2.0, we never measured raw Qwen3-14B on our 24-prompt
suite. The base was the implicit reference but never the *measured*
reference.

**What we found in v3**: base Qwen3-14B scores XState=1.30 (essentially
zero), FP=2.80, Reactive=4.40 (already strong from pretraining), ES=3.30,
Cap=3.75. Absolute training scores conflate prior knowledge with training
delta.

**Correction**: always measure raw base on the same suite, in the same
session, before any fine-tune. Train delta = arm − base, not arm − v0.7.
For Qwen3-14B specifically, our largest training contribution was XState
(+3.10 from v3.0-rslora) and FP (+1.70 from v3.0 vanilla). Reactive
training added only +0.10–0.50 — we'd been polishing capability the base
already had.

### Mistake 3: Never measuring the off-the-shelf alternative

For v0 through v2.0, we compared new arms to v0.7 historical, which is
fine-tune-vs-fine-tune. We never asked: how does our fine-tune compare
to a strong off-the-shelf model the user could run instead?

**What we found in v3**: Gemma 4 31B (no fine-tune, off-the-shelf) scores
4.12 average on our suite. Our v3.0 vanilla scores 4.27 average — a
margin of only +0.15 against an off-the-shelf reference twice the size.
On individual domains, Gemma beats us on ES; we beat Gemma on XState
significantly.

**Correction**: include the off-the-shelf reference in every grader
session. Sets the bar that fine-tuning must clear to be worth the
effort. If we'd done this from v0, we might have invested differently
(e.g., less polish on Reactive where the base is already strong).

### Mistake 4: n=5 per domain is too small for meaningful gates

24 prompts / 5 domains = 5 prompts per domain (4 for capability). At
that count, a single prompt's flip from 5 to 3 shifts the domain mean
by 0.4. Gate tolerances of ±0.3 are sub-noise.

**Status**: not yet corrected. Open question 5 in the v3 conclusions doc
proposes expanding the suite to 60–80 prompts. Pending.

### Mistake 5: Cross-session grader drift was unaccounted-for

LLM-as-judge grader instances dispatched in different sessions — even
the same Claude model — produce systematically different scores. Our
measured drift was 0.5–0.9 per domain, larger than the gate tolerances
we'd been using.

**Correction**: same-session calibration is mandatory. Two graders
dispatched in the same parallel batch typically agree within 0.17–0.29
mean disagreement (acceptable). Graders dispatched in different
sessions (even on the same outputs) can drift 0.5+. Don't trust
historical numbers.

The literature posture (see `docs/findings.v3.conclusions.md` §"Literature
posture") notes this is partially documented but our quantitative split
between intra- and inter-session variance is sharper than published
benchmarks report — a methodology data point worth preserving.

---

## Architectural lessons (LoRA-specific, transferable to next base model)

### Warm-start does NOT preserve capability — falsified across v1.1 and v1.2

Hypothesis (going into v1.1): load v0.7's merged weights as the base,
attach a fresh r=64 LoRA, train new domain (ES) on top. Prior
capabilities (FP=4.40) should ride through unchanged.

What happened:
- v1.1 (warm-start + full data including FP): FP regressed to 4.00
- v1.2 (warm-start + delta-only data, no FP): FP CATASTROPHICALLY
  collapsed to 2.20 — *below* the untrained base of 2.80

Mechanism: a LoRA modifies all 40 layers × 7 projection matrices = 256.9M
trainable parameters. Training on any data perturbs ALL weights. FP-
serving weights drift even when FP isn't in the batches. This is the
"LoRA is global, not local" finding that defined the entire warm-start
arc.

Documented in `v1.1/decision.md` and `v1.2/decision.md`. Don't repeat
this experiment. Stacking (v3.2, also failed for different reasons)
was the architectural alternative we tried.

### Atomic pattern drilling > compositional scenarios — established at v0.7, confirmed by v1's failure

v0.7's FP=4.40 came from 8 atomic-drill batches × 40 variations. Each
batch drilled ONE API surface (Effect.gen, Context.GenericTag, Schedule,
Option chain, TaskEither, Brand, acquireRelease, Effect.tap).

v1 added 24 more FP batches as "compositional scenarios" — "build an
HTTP handler with Effect", "hexagonal architecture with services",
"event-sourcing command handler". Spread 874 records across ~20 topics
at ~5–10 examples per pattern. Insufficient per-pattern density. FP
dropped to 3.80.

For new domain teaching: pick ONE API surface per batch, drill 30–40
minimal variations. Save compositional scenarios for second-phase
polish on top of solid atomic foundation.

### At fixed rank, capacity is fungible across domains — established by v3.1 vs v3.0-rslora

v3.0-rslora (r=128 rsLoRA, 2604 records): XState=4.40 / FP=3.40 / RX=4.90
/ ES=4.10. Strong on most, weak on FP.

v3.1 (r=128 rsLoRA, 2604 + 280 new FP records): XState=4.20 / FP=4.20 /
RX=4.40 / ES=4.10. FP recovered, but XState and RX dropped by 0.20–0.50.

Adding 280 records to FP didn't expand capacity — it redistributed it.
At fixed r=128 / 514M trainable parameters, there's a zero-sum-ish
trade-off across domains.

This is well-established in the literature (LoRI, Ortho-LoRA, MTL-LoRA,
Caruana 1997). We rediscovered it; the literature posture notes this
isn't novel.

### LoRA stacking math is additive but inference can be dominated — v3.2 failure

We trained a new r=64 fp_recovery adapter on FP-only data, on top of
the frozen v3.0-rslora r=128 adapter. Mathematically W = W₀ + ΔW_a +
ΔW_b. Should improve FP without disturbing XState/RX/ES.

Empirically: FP REGRESSED (3.40 → 2.80). XState dropped slightly too.

Hypothesized mechanism: the frozen v3.0-rslora has ~11.3× effective
LoRA scaling (rsLoRA at r=128 with alpha=128). The new adapter has
~8× effective scaling (rsLoRA at r=64). At inference, the frozen
adapter's larger magnitude in the FP-relevant directions dominates,
overpowering the new adapter's corrective deltas on held-out prompts.

The literature posture (LoRA Soups arxiv 2410.13025, Crowded-in-B-Space
arxiv 2604.16826) confirms naïve LoRA sums are not safely additive.
TIES/DARE-style sign-and-magnitude pre-processing is needed for clean
multi-LoRA composition. We didn't apply such pre-processing.

### Vanilla LoRA scaling can outperform rsLoRA on the underrepresented domain — Finding 1, partially counter-consensus

This is the one finding that's potentially publishable. See
`docs/findings.v3.conclusions.md` §"Literature posture" and
`docs/paper.rslora-imbalanced.md` for full ablation plan.

Empirical core:
- Same data (2604 records, FP at 12% share), same r=128, same recipe
- Vanilla LoRA (`alpha=r`, effective scaler ≈1×) → FP=4.50 (+1.70 over base)
- rsLoRA (`alpha/sqrt(r)`, effective scaler ≈11.3×) → FP=3.40 (+0.40 over base)

For dense-share domains, rsLoRA helped (XState 3.70 → 4.40, +0.70). For
the sparse-share FP, rsLoRA HURT relative to vanilla. Stronger effective
scaling amplifies dominant-domain bias.

Kalajdzievski 2023 (arxiv 2312.03732, the rsLoRA paper) claims monotone
gains across r and never tests imbalanced multi-task. Our regime is
under-documented in his framing.

---

## Data lessons (transferable)

### Synthesis discipline matters more than rank or scaling

Three data-discipline rules survived the arc:

1. **Atomic pattern drilling**: one API surface per batch, 30–40 minimal
   variations. House rules in `v0.7/seeds/patterns.fp.md`.
2. **Pattern coverage gaps WILL surface under stronger LoRA influence.**
   v0.7's FP corpus covered 8 of 15 specified patterns; the 7 missing
   patterns (Layer composition, Tagged errors+catchTag, etc.) didn't
   harm vanilla scaling but caused rsLoRA's FP collapse on prompts
   exercising those gaps. v3.1 closed 7/7 with 280 new records.
3. **Anchor ratio: 8–12% of corpus**, never above 40%. The v2-stack-fp
   arm with 43% anchors performed worse on FP than joint training with
   9% anchors despite identical FP records.

### Imbalanced corpus composition has stronger effects under rsLoRA than vanilla

Our 4-domain corpus: ES 52%, XState 17%, FP 12%, RX 9%, anchors 9%. ES
dominates by record count. Vanilla LoRA's mild influence let FP records
be absorbed despite the imbalance. rsLoRA's strong influence amplified
the imbalance — ES patterns crowded out FP. This is the data-side
counterpart to Finding 1's recipe-side observation.

### Verifier discipline: tighten upstream, not downstream

v1 attempted to fix FP by tightening the verifier gate post-hoc. v1.2
showed this is a band-aid. The real fix is upstream synthesis discipline:
pick atomic patterns, drill them, then let a simple verifier catch
obvious misses. `v0.7/verify.py` is loose-ish on FP idioms, and that
was fine for v0.7 because the synthesis was disciplined.

### Joint multi-task training beats single-task at the same rank

Same FP records (320) trained:
- FP-only at r=64 (v2-stack-fp): FP score 3.90
- Joint 4-domain at r=64 (v2.0-r64): FP score 4.50

The 4-domain arm had 4× more total optimizer steps and lower anchor
ratio (9% vs 43%) — confounds, not pure data-composition causation. But
the literature on multi-task / cocktail effect (Caruana 1997, MUPPET,
Flan-T5, Cocktail Effect 2024) is rich and consistently confirms this
direction. Don't train single-domain when the goal is single-domain
quality on a joint-shared base.

---

## Recipe lessons (the actual training recipe to recommend)

For a future arm on Qwen3-14B (or porting to Qwen3.5+):

1. **Fresh from base**, not warm-start from a prior fine-tune. Warm-start
   was falsified across v1.1/v1.2.
2. **r=128 with vanilla scaling** OR **r=128 with rsLoRA** depending on
   priority weighting:
   - **Priority on sparse-share domains** (e.g., FP at 12%): vanilla
     scaling. Empirical evidence: v3.0 vanilla beat v3.0-rslora on FP by
     +1.10.
   - **Priority on dense-share domains** (e.g., XState at 17%, ES at
     52%): rsLoRA. Empirical evidence: v3.0-rslora beat v3.0 vanilla on
     XState by +0.70.
   - The trade-off is real and binding at fixed rank with imbalanced data.
3. **Joint multi-task corpus**, not single-domain. Cocktail effect.
4. **Atomic pattern drilling per domain**, 30–40 records per pattern, all
   patterns from the spec covered (per Finding 4 in the v3 conclusions —
   training-data gaps surface under stronger LoRA influence).
5. **Anchor ratio 8–12%**.
6. **Always measure: raw base + off-the-shelf reference + new arm in the
   same grader session**. Compute deltas, not absolute thresholds.

---

## Investment thesis going forward

Public TypeScript fine-tuning at the 14B substrate has reached
diminishing returns. v3.0 vanilla beats Gemma 4 31B (4.27 vs 4.12, or
+0.37 under FP+Reactive priority lens), but the absolute gap is small
and each recipe iteration produces trade-offs rather than uniform gains.

The unique value going forward isn't "make a better TypeScript model" —
it's "fine-tune on proprietary knowledge that no off-the-shelf model
can have." Specifically: codebases like `qi-v2-qicore` (the user's
`Result<T>` / `QiError` library) where Gemma 4 31B and Opus 4.7 have
zero training data and our pipeline is the only viable path on consumer
hardware.

The substrate work (v3) is the foundation; proprietary library training
is the actual product. The methodology, tools, and recipe lessons
captured here are reusable for any proprietary codebase.

---

## Open questions across the entire arc

These are the questions that remain unresolved at the close of v3.
None block the daily-driver ship; each is an investment direction.

1. **Eval suite expansion**: 24 prompts → 60–80. n=5/domain is too small
   for tight gates. Pre-requisite for any future-arm gate to be
   reliable.

2. **rsLoRA crossover characterization**: Finding 1 is the publishable
   seed. Required ablations: rank sweep (r=64/128/256), alpha decoupling
   from scaling type, minority-domain corpus-share sweep (5%/10%/20%/30%).
   Documented in `docs/paper.rslora-imbalanced.md`.

3. **LoRA stacking with strength balancing**: v3.2 stacking failed
   because adapter scalings were imbalanced. TIES/DARE-style merging
   could fix this — pre-process adapters with sign-election and
   magnitude-pruning before sum. Untested.

4. **Style preference training**: all current records are
   competence-when-instructed (prompt names paradigm). The model
   doesn't learn preference-when-free. Synthesizing ~500 ambiguous
   prompts with FP/RX-default responses could shift the model's
   defaults. Marquardt 2025 (arxiv 2507.04889) and LIMA suggest
   100–1000 records is sufficient for style shift.

5. **Substrate port to Qwen3.5**: dense small variants released
   2026-03-02. Native multimodal isn't useful for code, but the
   improved base may close the gap to Gemma 4 31B without further
   recipe work. Migrating substrate is a multi-day revalidation.

6. **Proprietary library training (qi-v2-qicore)**: build a data-
   extraction pipeline from a real TypeScript codebase, synthesize
   instruction/response pairs that exercise its API, train a
   per-codebase LoRA. This is the project's actual product and the
   highest-leverage future direction.

---

## Reusable artifacts (the durable assets)

**Tools** (`tools/`):
- `tools/eval/run.py` — eval runner, model-agnostic, with `--temperature` and `--session` flags
- `tools/eval/tool_call_smoke.py` — tool-call canary
- `tools/eval/suite.json` — 24-prompt suite
- `tools/gate/combine_delta.py` — contemporaneous delta gate
- `tools/gate/grader_prompt.md` — grader template
- `tools/merge_adapter.py` — single-adapter merge recovery
- `tools/merge_adapter_stack.py` — multi-adapter sequential merge

**Synthesis seeds**:
- `v0.7/seeds/patterns.fp.md` — 15 FP pattern specs (8 from v0.7, 7
  filled in v3.1)
- `v0.7/seeds/patterns.rx.md`, `patterns.xstate.md` — domain pattern specs
- `v0.7/seeds/phrasings.md` — instruction-template diversity
- `v0.7/verify.py` — multi-domain TypeScript code verifier (compile + idiom + length)

**Data**:
- `v3.1/data/synth.verified.jsonl` (2882 records, 4 domains, 15/15 FP
  patterns covered)

**Models** (in ollama):
- `ts-forge-v3:latest` ← daily driver (= v3.0 vanilla)
- `ts-forge-v3.0-rslora:latest` ← XState specialist
- `ts-forge-v0.7-r64:latest` ← prior daily driver (retained)

**Documents**:
- `docs/conclusions.cross-version.md` — this document
- `docs/findings.v3.conclusions.md` — v3 phase specifics, literature posture
- `docs/findings.multi-domain.md` — earlier multi-task finding with literature
- `docs/lessons.learned.md` — ad-hoc lessons accumulated during v0–v1
- `docs/paper.rslora-imbalanced.md` — paper outline + ablation plan for Finding 1
- Per-version `decision.md` files (v0 through v2)

---

## Summary in three sentences

**Methodology:** We learned (the hard way, across 5 versions) that
contemporaneous baseline measurement is non-negotiable; historical
floor gates are unreliable due to 0.5–0.9 cross-session grader drift.
**Architecture:** Warm-start LoRA does not preserve capability;
multi-task joint training at fixed rank has a zero-sum capacity wall;
single-LoRA can be optimized for FP+Reactive-style or for dense-share
domains, but not both simultaneously without rank scaling. **Recipe:**
For underrepresented domains under data imbalance, vanilla LoRA scaling
outperformed rsLoRA at the same rank — a partial-counter-consensus
finding (Kalajdzievski 2023 doesn't test imbalanced multi-task) that's
the seed of `docs/paper.rslora-imbalanced.md`.

The v3 daily driver `ts-forge-v3:latest` is the practical artifact;
this document is the reusable knowledge.
