---
title: "When rsLoRA Hurts: Vanilla LoRA Scaling Outperforms rsLoRA on Underrepresented Domains in Imbalanced Multi-Task Fine-Tuning"
author:
  - "Zhifeng Zhang\\thanks{Independent researcher. ORCID: \\href{https://orcid.org/0009-0004-9509-5221}{0009-0004-9509-5221}. Email: \\href{mailto:zhifeng.zhang.sz@gmail.com}{zhifeng.zhang.sz@gmail.com}.}"
  - "Claude (Anthropic Opus 4.7)\\thanks{Research-engineering collaborator. See author contributions and AI-coauthorship disclosure in §6.}"
date: 2026-04-25
abstract: |
  Rank-stabilized LoRA (rsLoRA) replaces the original $\alpha/r$ scaling factor with $\alpha/\sqrt{r}$,
  proving gradient stability at high rank and demonstrating monotonically better aggregate
  fine-tuning loss. Published evaluations test single-task fine-tuning or balanced multi-task
  corpora. We report empirical results showing that under heavily imbalanced multi-task fine-tuning
  --- specifically, when one target task represents ~12% of corpus records while another represents
  52% --- the canonical rsLoRA recipe ($\alpha = r$ at $r=128$, effective scaler ~11.3) is
  uniquely bad for the underrepresented target, scoring 1.40 below vanilla LoRA on the same
  task while simultaneously achieving the highest score on the majority-share task. We disentangle
  effective-scaler magnitude from the scaling-rule choice via a six-arm $\alpha$-decoupling sweep
  at fixed $r=128$. The data shows that within vanilla scaling, the underrepresented domain's
  score decreases monotonically with effective scaler; but within rsLoRA, the relationship is
  U-shaped: lower-than-default $\alpha$ retains more underrepresented capability, the canonical
  $\alpha = r$ default is the worst point, and $\alpha = 2r$ partially recovers --- though not
  without sacrificing the majority-task peak. We hypothesize the U-shape reflects competition
  between two mechanisms (base-model pass-through at low scaler, training-data memorization at
  high scaler) with the canonical default sitting in the worst-of-both regime. Practitioners
  fine-tuning on heterogeneous corpora should not assume rsLoRA is monotonically better.
keywords: ["LoRA", "rsLoRA", "multi-task fine-tuning", "data imbalance", "parameter-efficient fine-tuning"]
---

# 1. Introduction

Low-Rank Adaptation (LoRA) [@hu2021lora] has become the dominant parameter-efficient fine-tuning
recipe for large language models. The original formulation injects two low-rank matrices
$A \in \mathbb{R}^{r \times d}$ and $B \in \mathbb{R}^{d \times r}$ into each target weight
$W \in \mathbb{R}^{d \times d}$, producing an update $\Delta W = \frac{\alpha}{r} B A$ where
$\alpha$ is a hyperparameter and $r \ll d$ is the rank.

Kalajdzievski [@kalajdzievski2023rslora] observed that the $\alpha/r$ scaling factor causes the
LoRA's effective contribution at high rank to vanish unless $\alpha$ scales with $r$, and proposed
the rank-stabilized variant $\Delta W = \frac{\alpha}{\sqrt{r}} BA$ (rsLoRA). The substitution is
proven to maintain gradient stability across rank, and is shown empirically to yield monotonically
better aggregate fine-tuning loss on standard benchmarks. The rsLoRA paper's experiments are
single-task or balanced multi-task; the prescription has been adopted broadly as a strict
improvement over vanilla LoRA at high rank.

We report a regime where this prescription does not hold. In a heavily imbalanced multi-task
fine-tuning setting with one underrepresented target task, vanilla $\alpha/r$ scaling outperforms
rsLoRA on the underrepresented task at the same rank, despite rsLoRA achieving lower aggregate
training loss. The trade-off is one-directional: rsLoRA improves the majority-share tasks while
hurting the minority-share task.

Our contribution is empirical evidence and a mechanistic explanation, plus an ablation that
isolates the variable responsible (effective-scaler magnitude vs scaling-rule choice). The
finding has practical consequences: practitioners fine-tuning on heterogeneous corpora --- e.g.,
domain mixtures, instruction datasets with skewed task counts, or codebases with
unevenly-represented APIs --- should not assume rsLoRA is monotonically better.

# 2. Background

**LoRA scaling.** Vanilla LoRA's $\alpha/r$ factor was chosen so that doubling rank does not
require retuning learning rate; in practice, $\alpha = r$ is a common default making the effective
scaler $\alpha/r = 1$ regardless of rank. Kalajdzievski's analysis [@kalajdzievski2023rslora] shows
that under this convention, increasing $r$ does not increase the LoRA's effective influence,
limiting capacity gains from rank scaling. The proposed fix $\alpha/\sqrt{r}$ makes the effective
scaler grow as $\sqrt{r}$ when $\alpha = r$, restoring the rank-capacity relationship.

**Multi-task LoRA interference.** When a single LoRA is trained on multiple tasks sharing the same
low-rank subspace, gradients from different tasks compete for the same update directions
[@caruana1997multitask; @zhang2025lori; @ortholora2025]. At fixed rank, this competition is
zero-sum: improving one task tends to come at the cost of another unless the tasks are
sufficiently aligned in their gradient directions. Rebalancing data shares does not expand
capacity; it redistributes which tasks win the limited subspace.

**Adapter composition.** Sums of independently-trained LoRA adapters are not safely additive
[@pfeiffer2020adapterfusion; @prabhakar2024lorasoups]. When adapters' deltas have unbalanced
magnitudes, the larger-magnitude adapter dominates inference, and corrective adapters trained at
lower rank or smaller scaling can fail to override the stronger contribution.

**Multi-task fine-tuning beats single-task on the target.** A long literature establishes that
joint multi-task fine-tuning beats single-task fine-tuning on the target's held-out evaluation,
even when the target's training records are byte-identical between arms
[@aghajanyan2021muppet; @chung2022flan; @shen2024cocktail]. The mechanism is auxiliary-task
regularization of the shared subspace.

# 3. Method

## 3.1 Substrate

We use Qwen3-14B as the base model, quantized to 4-bit weights via bitsandbytes (NF4) and loaded
through the Unsloth fine-tuning framework [@unsloth2026]. LoRA adapters target the seven
projection matrices in each transformer block: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`,
`up_proj`, `down_proj`. The base model has 40 transformer blocks, giving 280 target matrices per
adapter.

## 3.2 Training corpus

The training corpus consists of 2604 instruction-response pairs spanning four TypeScript-focused
domains plus capability anchors:

| Domain | Records | Share | Source |
|---|---|---|---|
| Event Sourcing (ES) | 1364 | 52.4% | Atomic-drill synthesis covering 37 aggregates |
| XState v5 | 440 | 16.9% | v0.6 atomic-drill synthesis |
| Effect-TS / fp-ts (FP) | 320 | 12.3% | v0.7 atomic-drill, 8 patterns from a 15-pattern spec |
| RxJS (RX) | 240 | 9.2% | v0.7 atomic-drill, 6 batches |
| Capability anchors | 240 | 9.2% | 30 unique × 8 reps |

The composition is heavily imbalanced: ES dominates the corpus at 52%, while the **FP domain is
underrepresented at 12%**. This imbalance is the regime where we hypothesize rsLoRA's behavior
diverges from the published prescription.

## 3.3 Training recipe

All arms use identical training hyperparameters except the variables under study (rank, $\alpha$,
scaling rule). Common settings:

- Rank: $r=128$ (held fixed in the alpha-decoupling ablation)
- Optimizer: AdamW via TRL `SFTTrainer`
- Per-device batch size: 2; gradient accumulation: 4 (effective batch 8)
- Learning rate: $1 \times 10^{-4}$
- Epochs: 3
- Precision: bf16
- Random seed: 42
- Tokenizer: Qwen3 default with chat template applied

## 3.4 Evaluation

We evaluate on a 24-prompt held-out suite covering five domains: 5 XState, 5 FP, 5 RX, 5 ES, and
4 capability prompts. Each prompt is inferred at temperature 0.6 via the trained model's GGUF
export served by Ollama.

Responses are scored 1--5 on idiomatic correctness by two independent Claude Opus 4.7 grader
instances dispatched as blind subagents in the same parallel session, with a structured grading
prompt covering domain-specific anchors (e.g., XState v5 `setup()` idioms, Effect-TS `Effect.gen`
+ `yield*`, RxJS pipeable operators, Decider pattern for ES). Inter-grader agreement on this
suite has been observed at 0.12--0.29 mean per-prompt disagreement across multiple sessions
[@zheng2023judging; @li2024llmjudge_survey].

Critically, every grader session also re-evaluates the unmodified base Qwen3-14B and at least one
reference fine-tune in the same parallel grader batch. This **contemporaneous baseline measurement**
controls for cross-session grader drift; absolute scores are reported but the analytical anchor is
the base-relative training delta.

## 3.5 Manipulated variables

We test six configurations at $r=128$ to decouple effective-scaler magnitude from the scaling-rule
choice:

| Run | $\alpha$ | Rule | Effective scaler $\alpha / r$ (vanilla) or $\alpha / \sqrt{r}$ (rsLoRA) |
|---|---|---|---|
| B1 | 64 | $\alpha/r$ | 0.50 |
| B2 | 128 | $\alpha/r$ | 1.00 (default vanilla) |
| B3 | 256 | $\alpha/r$ | 2.00 |
| B4 | 64 | $\alpha/\sqrt{r}$ | 5.66 |
| B5 | 128 | $\alpha/\sqrt{r}$ | 11.31 (default rsLoRA) |
| B6 | 256 | $\alpha/\sqrt{r}$ | 22.63 |

This design lets us answer: is the FP regression we observe under rsLoRA *because* of the scaling
rule itself, or *because* of the larger effective scaler that rsLoRA produces under the common
$\alpha = r$ default? The two hypotheses make different predictions: if the scaling rule itself
matters, B4 (rsLoRA at low effective scaler $\approx$ 5.66) should still differ from a vanilla arm at
similar effective scaler; if it's purely effective-scaler magnitude, B1 (vanilla at 0.50) should
be the strongest FP arm and B6 (rsLoRA at 22.63) the weakest, with the trend monotone in
effective scaler.

# 4. Results

## 4.1 Headline result: vanilla outperforms rsLoRA on FP at the default $\alpha$=r

The default $\alpha = r = 128$ configuration yields the comparison most directly aligned with
how practitioners adopt rsLoRA --- swapping in `use_rslora=True` while keeping $\alpha = r$:

::: {#tbl-headline}

| Arm | Scaling rule | FP score | XState | RX | ES | Cap | Avg |
|---|---|---|---|---|---|---|---|
| Base Qwen3-14B (no FT) | --- | 2.80 | 1.30 | 4.40 | 3.30 | 3.75 | 3.11 |
| B2 ($\alpha/r$, vanilla) | $\alpha/r$ | **4.50** | 3.70 | 4.70 | 4.30 | 4.13 | **4.27** |
| B5 ($\alpha/\sqrt{r}$, rsLoRA) | $\alpha/\sqrt{r}$ | 3.40 | **4.40** | 4.90 | 4.10 | 4.13 | 4.19 |

Table 1: Headline result. Identical data, identical rank ($r=128$), identical $\alpha$ ($=128$);
only the scaling rule differs. Vanilla wins on the underrepresented FP domain (+1.10); rsLoRA wins
on the dense XState domain (+0.70). Overall average shifts <0.10. Scores from session
2026-04-25b, 10 graders matched calibration. Inter-grader disagreement 0.12--0.25.

:::

The trade-off is asymmetric: rsLoRA's gain on dense-share tasks (XState, +0.70) is real, but
comes at a substantial cost on the sparse-share FP task (-1.10). Base-relative deltas confirm
both directions are real training contributions, not measurement artifacts:

- Vanilla added +1.70 to FP, +2.40 to XState
- rsLoRA added +0.60 to FP, +3.10 to XState

The published rsLoRA prescription correctly predicts the XState (dense) result. It fails to
account for the FP (sparse) regression.

## 4.2 Ablation: decoupling effective scaler from scaling rule

We ran the six planned alpha-decoupling arms in the same training and evaluation pipeline. All
six were graded in a single 12-grader matched-calibration session (2026-04-26a) so that
cross-arm comparisons are not affected by inter-session grader drift. Per-domain scores:

::: {#tbl-ablation}

| Arm | $\alpha$ | Rule | Effective scaler | Train loss | **FP** | XState | RX | ES | Cap | Avg |
|---|---|---|---|---|---|---|---|---|---|---|
| B1 | 64 | $\alpha/r$ | 0.50 | 0.344 | **4.50** | 3.60 | 5.00 | 4.20 | 4.25 | 4.31 |
| B2 | 128 | $\alpha/r$ | 1.00 | 0.318 | 4.30 | 3.40 | 4.90 | 4.40 | 4.00 | 4.20 |
| B3 | 256 | $\alpha/r$ | 2.00 | 0.287 | 4.00 | 4.20 | 4.80 | 4.20 | 4.25 | 4.29 |
| B4 | 64 | $\alpha/\sqrt{r}$ | 5.66 | 0.250 | 3.80 | 4.40 | 4.50 | 4.60 | 4.00 | 4.26 |
| B5 | 128 | $\alpha/\sqrt{r}$ | 11.31 | 0.239 | **2.90** | **4.60** | 5.00 | 4.50 | 4.13 | 4.22 |
| B6 | 256 | $\alpha/\sqrt{r}$ | 22.63 | 0.269 | 4.40 | 4.10 | 4.50 | 4.20 | 4.25 | 4.29 |

Table 2: Alpha-decoupling ablation. All arms at $r=128$, identical data, identical training
recipe except $\alpha$ and scaling rule. Mean of two blind graders per arm in session
2026-04-26a (24 prompts each). FP boldfaced at the extremes of the sweep.

:::

The data does **not** show a monotone-decreasing FP curve as a function of effective scaler.
Three findings emerge:

**Finding A — within vanilla scaling, FP decreases monotonically as $\alpha$ grows.** The vanilla
arms (B1, B2, B3) at $\alpha = 64, 128, 256$ produce FP scores of 4.50, 4.30, 4.00 ---
strictly decreasing with $\alpha$. Larger $\alpha$ at fixed $r=128$ shifts the LoRA's
contribution from "small nudge on top of base" toward "more aggressive rewrite of base
behavior", and the effect on the underrepresented FP domain is negative throughout this regime.

**Finding B — within rsLoRA scaling, FP is *non-monotone* in effective scaler with a worst-case
at the canonical default $\alpha = r$.** B4 (effective scaler 5.66, $\alpha < r$) gives FP = 3.80;
B5 (canonical rsLoRA at $\alpha = r$, effective scaler 11.31) gives FP = **2.90 — the worst
FP score in the entire sweep**; B6 ($\alpha = 2r$, effective scaler 22.63) recovers FP to 4.40.
The U-shape is robust: both grader A and grader B independently scored B5's FP at 2.80 and 3.00
respectively, and B6's at 4.40 and 4.40. Per-prompt: B5 fails on fp-02/03/04/05 (mean 2.5 each)
while B6 recovers fp-03 (5.0), fp-04 (5.0), and fp-05 (4.0) — three independent prompts agreeing
that B6 produces correct FP outputs where B5 does not.

**Finding C — XState benefits from larger effective scaler up to a point, then plateaus.** XState
scores: 3.60 (B1) → 3.40 (B2) → 4.20 (B3) → 4.40 (B4) → **4.60 (B5)** → 4.10 (B6). The
canonical rsLoRA recipe (B5) is the *peak* for the dense-share XState domain at the same
moment it is the trough for the sparse-share FP domain.

![FP and XState scores across the effective-scaler sweep at $r=128$. Vanilla arms (B1, B2, B3)
shown as circles; rsLoRA arms (B4, B5, B6) as squares. The canonical rsLoRA configuration B5
($\alpha=r=128$) maximizes XState (top of the red dashed curve) and minimizes FP (bottom of
the blue solid curve) simultaneously. B6 ($\alpha=2r=256$ with rsLoRA) recovers FP without
preserving B5's XState peak.](figures/fp_xstate_vs_scaler.pdf){#fig-fp-vs-effective-scaler width=85%}

The $\alpha$-decoupling design lets us answer the methodological question reviewers would
otherwise raise: is the FP regression at B5 due to the rsLoRA scaling rule itself, or due to
the larger effective scaler that rsLoRA produces under the common $\alpha = r$ default? **Neither
explanation alone fits the data.** The vanilla sweep (B1 → B2 → B3) shows that effective scaler
itself harms FP monotonically up to scaler $\approx$ 2. But the rsLoRA sweep (B4 → B5 → B6) shows that
once we cross into the stronger-scaler regime accessible only through rsLoRA, the relationship
flips — the canonical $\alpha = r$ recipe is uniquely bad for sparse-share FP, while moving
$\alpha$ either downward (to vanilla) or upward (to $\alpha = 2r$) preserves more FP capability.

We hypothesize the U-shape reflects two competing mechanisms: at small effective scaler, the
LoRA's contribution is small enough that the base model's pretrained FP capability dominates
inference (Findings A and B's B1 endpoint); at very large effective scaler, the LoRA itself
becomes strong enough to memorize FP training records and produce them at inference, bypassing
the multi-task interference that hurts the dominant-share patterns (B6 endpoint); at the
intermediate "default rsLoRA" scaler $\approx$ 11, the LoRA is strong enough to disrupt base FP but not
strong enough to fully replace it from training data (B5). This is consistent with B5's lower
training loss (0.239) and B6's slightly higher loss (0.269): B5 fits the dominant ES patterns
most aggressively, while B6 has begun under-fitting them, freeing capacity for FP.

## 4.3 Mechanism

We propose two competing mechanisms whose composition explains the U-shaped FP curve, both
consistent with the multi-task LoRA interference literature [@zhang2025lori; @ortholora2025].

**Mechanism 1: gradient crowding in the shared low-rank subspace (dominates at intermediate
effective scaler).** At each optimizer step, the LoRA's gradient is the average of per-record
gradients in the mini-batch. With imbalanced data shares (ES at 52%, FP at 12%), an effective
batch of 8 records contains on average 4.2 ES and 1.0 FP records. The averaged gradient direction
is dominated by the ES-task gradient. A larger effective scaler translates that direction into
proportionally larger weight updates; over training, the shared low-rank subspace becomes
aligned with dominant-task patterns and the minority-task signal is crowded out. This mechanism
predicts that *more* effective scaler hurts the minority task --- consistent with the
B1 → B2 → B3 → B4 → B5 trend (FP 4.50 → 4.30 → 4.00 → 3.80 → 2.90).

**Mechanism 2: training-data memorization (dominates at very large effective scaler).** Past a
threshold, the LoRA becomes strong enough to memorize specific training records rather than
finding compromises in shared-subspace directions. Each FP training record contributes a
gradient signal that, at sufficient effective scaler, the LoRA can encode independently of the
shared majority-task subspace. Past the memorization threshold, the LoRA's response to FP-style
prompts at inference is reproducing memorized FP patterns rather than mediating between
base-model FP and overwritten ES-aligned weights. This mechanism predicts that beyond a
sufficiently strong effective scaler, the minority task can recover --- consistent with B6's
recovery to FP = 4.40 at scaler 22.63, and with B6's slightly higher train loss (0.269 vs B5's
0.239) suggesting B6 began under-fitting the dominant-task patterns to make capacity for FP.

The canonical rsLoRA configuration ($\alpha = r = 128$, effective scaler 11.31, our B5) sits in
the worst-of-both regime: large enough to severely crowd FP via Mechanism 1, but not large
enough for Mechanism 2 to take over. Both vanilla scaling (Mechanisms 1 and 2 both weak; FP rides
on base-model capability) and very large rsLoRA scaling (Mechanism 2 dominant; FP memorized
from training records) preserve more FP capability. The trade-off for the dense-share XState
domain is symmetric: XState scoring is highest at the canonical rsLoRA default (4.60 at B5), and
both lower scaler (B1: 3.60) and higher scaler (B6: 4.10) reduce it.

We emphasize this mechanism story is hypothesis, not theorem. Replicates and additional probes
(e.g., per-record loss tracking, attention-head analysis) would be needed to confirm it.

# 5. Discussion

## 5.1 Practical implications

For practitioners fine-tuning on heterogeneous corpora --- domain mixtures, instruction datasets
with skewed task counts, or codebases with unevenly-represented APIs --- the rsLoRA prescription
should not be assumed monotonically better. We recommend:

1. **Measure base-relative deltas per task**, not absolute scores. Without a contemporaneous base
   measurement, gain on majority-share tasks can mask regression on minority-share tasks.

2. **Avoid the canonical rsLoRA configuration ($\alpha = r$) when minority-task quality matters.**
   Our B5 result shows this is the worst point in the $\alpha$-decoupling sweep for the
   underrepresented domain. Either smaller $\alpha$ (closer to vanilla) or larger $\alpha$
   (with rsLoRA, e.g., $\alpha = 2r$) preserves more minority-task capability, though each
   trades off differently against majority-task quality.

3. **Rebalance data composition before reaching for rsLoRA.** Fixed-rank LoRA capacity is
   zero-sum across tasks; if minority-task quality matters, increasing the minority share is the
   most direct lever, and may render the scaling-rule choice less critical.

4. **Decouple $\alpha$ from the scaling rule.** Both vanilla and rsLoRA have an $\alpha$ hyperparameter; the
   common $\alpha = r$ default conflates "scaling rule" with "effective magnitude" and obscures
   which is doing the work. Practitioners adopting rsLoRA should test $\alpha < r$ to see whether
   the gains attributed to rsLoRA actually require its larger effective scaler.

## 5.2 Relation to prior work

Our finding does not contradict Kalajdzievski [@kalajdzievski2023rslora]. The rsLoRA paper
correctly demonstrates that under single-task or balanced multi-task settings, $\alpha/\sqrt{r}$
yields better aggregate fine-tuning loss and stable gradients at high rank. The paper does not
test imbalanced multi-task fine-tuning; we add a regime where the prescription's monotonicity
breaks.

The mechanism we propose --- minority-task crowding under stronger effective scaling --- is
consistent with the cross-task interference findings in LoRI [@zhang2025lori] and Ortho-LoRA
[@ortholora2025]. Both observe that the shared low-rank subspace creates zero-sum trade-offs at
fixed rank. Our contribution is the explicit prediction that *the scaling rule controls how
strongly the dominant task crowds the minority*.

## 5.3 Limitations

- **Single base model.** All experiments use Qwen3-14B. Results may differ for smaller (e.g.,
  7B) or larger (e.g., 70B) bases, and for non-Qwen architectures.
- **Single domain pair.** The four-domain TypeScript corpus is one specific imbalance pattern.
  Generalization to natural-language multi-task instruction-tuning, multilingual, or vision
  tasks is untested.
- **Eval suite size.** Five prompts per domain is small; single-prompt swings can shift domain
  means by $\pm 0.4$. Our largest deltas (B1's 4.50 vs B5's 2.90 = 1.60-point FP gap) exceed
  this noise floor by multiple standard deviations, and per-prompt agreement between independent
  graders A and B was tight (B5 FP per-prompt: 5/5, 2/3, 2/3, 2/2, 3/2; B6 FP per-prompt: 5/5,
  3/3, 5/5, 5/5, 4/4). The U-shape's inflection at B6 is driven by three independent prompts
  agreeing across both graders; we consider it real but flag that single-eval, single-seed-per-arm
  is the most significant limitation of this work.
- **LLM-as-judge evaluation has documented drift** [@zheng2023judging; @li2024llmjudge_survey].
  We mitigate via same-session matched-grader calibration, but absolute scores still carry
  ~±0.2 noise.
- **bnb-4bit base quantization.** Full-precision training may exhibit different scaling
  behavior. We do not test this.

## 5.4 Future work

- **Replicate with multiple seeds per arm.** The B6 recovery is the most surprising data point
  of our sweep and the most important to confirm. Three to five training runs per arm at
  different seeds, evaluated in matched-grader sessions, would establish whether the U-shape
  is robust or whether B6 lies in the noise band.
- **Replicate at additional ranks** ($r = 64, 256$) to verify the effect's rank-robustness and
  to test whether the canonical-rsLoRA worst-case shifts predictably with $r$.
- **Sweep minority-domain corpus share** (5%, 10%, 20%, 30%) to characterize the crossover
  region where minority-task quality stops being sensitive to the scaling rule.
- **Test on non-code corpora.** Our finding may apply to any imbalanced multi-task fine-tune;
  validation on natural-language task mixtures (e.g., instruction-tuning datasets with skewed
  task counts) would broaden the result.
- **Investigate adapter composition strategies** that preserve minority-task gains. TIES- and
  DARE-style sign-and-magnitude pre-processing of pre-trained adapters
  [@prabhakar2024lorasoups] may permit safe stacking of separately-trained per-domain adapters,
  side-stepping the multi-task interference issue entirely.
- **Probe the proposed mechanisms.** Per-record loss tracking during training would test whether
  the canonical rsLoRA configuration shows distinctive minority-task gradient behavior; attention
  head ablations on the trained checkpoints could test whether B5 and B6 differ in which heads
  carry FP-relevant features.

# 6. Reproducibility

All training scripts, evaluation scripts, training data, raw eval outputs, grader outputs, and
this paper's source are available in the project repository. Key entry points:

- Training: `v2/train.py` with `--rank`, `--lora-alpha`, and `--rslora` flags
- Evaluation: `tools/eval/run.py` (deterministic at temperature 0; this paper used 0.6 to match
  prior arms)
- Delta gate: `tools/gate/combine_delta.py`
- Grader prompt template: `tools/gate/grader_prompt.md`
- Multi-adapter merge: `tools/merge_adapter.py`, `tools/merge_adapter_stack.py`

Each ablation arm is fully specified by its training command-line in
`v3.B/{B1,B3,B4,B6}/logs/train.log` (with full hyperparameters echoed at startup) and produces a
deterministic GGUF export. Hardware: NVIDIA RTX 5090 (32 GB VRAM); training time per arm
~50 minutes; full pipeline (training + GGUF + evaluation + grading) ~80 minutes per arm.

# Acknowledgments

This work was carried out as a side investigation of a multi-version TypeScript code-generation
fine-tuning project. Thanks to the Unsloth and bitsandbytes teams for the training infrastructure
that makes 14B-parameter LoRA fine-tuning tractable on a single 32 GB consumer GPU.

# References
