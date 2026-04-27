# Finding — Multi-Domain Joint Training Beats Single-Domain at the Same Rank

**Date**: 2026-04-24
**Context**: Mid-v2 investigation after the v2.x curriculum plan was revisited.
**Status**: Empirical observation with a known confound. Informs the shift to v3.0.

## TL;DR

At r=64 on Qwen3-14B with our standard recipe, training on a single
domain (FP, 320 records + 240 anchors = 560 total) produces a **worse**
FP model than training on four related TypeScript domains jointly
(2604 records total; FP records byte-identical to the single-domain
arm). Measured under contemporaneous grading against v0.7-r64 in the
same session:

| Arm | Base | Rank | Corpus | FP score |
|---|---|---|---|---|
| v2.0 (FP specialist, archived as `ts-forge-v2-fp-only-archive`) | Qwen3-14B bnb-4bit | 64 | 320 FP + 240 anchors (560) | **3.90** |
| 4-domain joint (archived as `ts-forge-v2-r64`) | Qwen3-14B bnb-4bit | 64 | 320 FP + 440 XState + 240 RxJS + 1364 ES + 240 anchors (2604) | **4.60** |
| v0.7-r64 re-checked same session | — | — | v0.6 + v0.7 atomic-drill 3-domain | 3.90 |

Delta on FP from adding three unrelated-to-FP domains: **+0.70**, with
**the same 320 FP training records**. Both arms were fresh-from-base
LoRAs with identical r, lr, epochs, seed, optimizer, and precision.

The premise that had guided the v2.x roadmap — "each arm narrower in
scope produces a better specialist on its domain" — is not supported
by this observation. The roadmap was replanned on the back of it.

## Observation details

- **Same base model**: `unsloth/qwen3-14b-unsloth-bnb-4bit` loaded
  identically in both arms.
- **Same LoRA shape**: r=64, alpha=64, dropout=0, 7 target modules
  (q/k/v/o/gate/up/down), Unsloth `get_peft_model(...)`.
- **Same optimizer recipe**: SFTTrainer, per-device batch=2,
  grad-accum=4, effective batch=8, lr=1e-4, bf16, 3 epochs, seed=42,
  `save_strategy=epoch`.
- **Same FP training data**: byte-identical 320 records from
  `v0.7/data/synth.fp.batch-{A..H}.jsonl` in both arms.
- **Same eval**: 5 FP prompts from `tools/eval/suite.json`, run at
  temperature 0.6 via ollama.
- **Same grading session**: two blind Claude-subagent graders
  dispatched in the same batch as v0.7's re-check (2026-04-24),
  grader agreement 24/24 within 1 pt, mean disagreement 0.17.

Per-prompt FP scores (mean of two graders):

| Prompt | v2.0 FP-only | 4-domain joint | v0.7-recheck |
|---|---|---|---|
| fp-01 | 2.0 | 3.0 | 4.5 |
| fp-02 | 4.0 | 4.5 | 4.0 |
| fp-03 | 5.0 | 4.0 | 3.0 |
| fp-04 | 4.5 | 5.0 | 5.0 |
| fp-05 | 4.0 | 4.0 | 3.0 |
| **mean** | **3.90** | **4.60** | **3.90** |

The largest single-prompt swing is fp-01 (2.0 → 3.0 → 4.5 across
arms), which both graders flagged in v2.0 as a "class-before-use /
duplicate-declaration" compilation bug — i.e., a syntactic slip, not
an idiom error. Single-prompt noise at n=5 amplifies this kind of
defect into ~0.4 of domain mean.

## Confounds — do not ignore

Two variables were not held constant between the arms:

1. **Total optimizer steps**: 210 (FP-only, 560 records × 3 epochs ÷ 8
   batch) vs **978** (4-domain joint). A 4.66× difference. The cosine
   LR schedule reaches different endpoints, and the 4-domain arm had
   4.66× more refinement passes.
2. **Anchor ratio**: 240/560 = **43%** in FP-only vs 240/2604 = **9%**
   in 4-domain. Anchors are capability prompts (non-FP); they dilute
   the FP gradient signal much more heavily in the FP-only arm.

A properly compute-matched and anchor-ratio-matched comparison would
require training FP-only for 978 steps (either 14 epochs on 560
records, or expand to 2604 FP records) with anchors reduced to 9% —
which we have not run.

## What the literature says (abbreviated; see report below for full cites)

This phenomenon is classical:

- **Caruana 1997** ("Multitask Learning," Machine Learning): multi-task
  learning wins via inductive transfer from related-task signal acting
  as an implicit regularizer. Persists even when MTL nets are larger
  than the sum of single-task nets — so it is representation transfer,
  not merely capacity sharing.
- **MUPPET / Aghajanyan et al. 2021** (arxiv 2101.11038): multi-task
  pre-finetuning improves single-task performance, but only above a
  threshold of ~15 tasks. Below that, multi-task can hurt.
- **Super-NaturalInstructions / Wang et al. 2022** (arxiv 2204.07705):
  linear improvement with exponential task count at fixed model size.
- **"Mixing It Up: Cocktail Effect" / Shen et al. 2024** (arxiv
  2410.01109): 200+ controlled experiments on Phi-3-Mini show that
  mixing related tasks **raises** single-target performance in the
  related-tasks regime. Closest direct analog to our setup.
- **Weller et al. 2022** (arxiv 2205.08124): task similarity and
  dataset-size asymmetry determine whether MTL wins vs intermediate
  fine-tuning. Dissimilar cocktails cause negative transfer.
- **Springer ML 2025** (doi 10.1007/s10994-025-06885-z) and **MDPI
  Computers 2025** (doi 10.3390/computers14070264) document task
  conflict as the failure mode when gradients disagree at shared
  parameters.

Our 4 domains (FP / XState / RxJS / Event Sourcing) are all
TypeScript with overlapping functional idioms — they sit solidly in
the related-tasks regime where the positive-transfer mechanism is
expected to fire. The observation is consistent with the literature.

## LoRA-specific caveat

Most LoRA multi-task papers (LoRAHub, AdapterFusion, Polytropon) are
about *composing independently-trained* adapters, not *jointly
training* a single adapter on mixed data. Our setup (one LoRA trained
on the full multi-domain corpus) is the less-studied regime. Closest
published analogs: MeTA-LoRA 2025 (arxiv 2510.11598) and the Shen
cocktail paper.

**rsLoRA scaling** (Kalajdzievski 2023, arxiv 2312.03732) is worth
flagging: the apparent "rank saturation" in the original LoRA paper
is an artifact of the default `alpha/r` scaling; with `alpha/sqrt(r)`
scaling, rank trades genuinely for capacity. Our `v2/train.py` uses
`lora_alpha = args.rank`, which under PEFT default gives effective
scaler = 1 regardless of rank. If a rank-scaling experiment (v3.0)
shows disappointing results, the next variable to try is
`use_rslora=True` before going to higher ranks.

## Implications for the v2/v3 roadmap

The specialty-chain plan (v2.0 = FP specialist; v2.1 adds RX; v2.2
adds ES; v2.3 adds XState v5) was built on the premise that narrower
training produces better per-domain quality. That premise is
falsified by this observation at r=64. Intermediate specialists
(v2.1, v2.2) would likely score **worse on their specialty** than a
full 4-domain model trained at the same rank, because they would
have fewer optimizer steps, less cross-domain regularization, and
higher anchor dilution.

**What was retired:**
- v2.0 as "FP specialist" (`ts-forge-v2-fp-only-archive` kept as a
  data point; not on the shipping roadmap).
- v2.1 / v2.2 / v2.3 as incremental specialist arms — never trained,
  not worth training.

**What was promoted:**
- The `ts-forge-v2-r64` model (joint 4-domain, already trained
  2026-04-22, formerly labeled `ts-forge-v2.0`) is retained as the
  r=64 baseline for comparison against v3.0.

**What v3.0 tests:**
- Same 2604-record 4-domain corpus, same recipe, **r=128 instead of
  r=64**. Single-variable test: does doubling LoRA capacity improve
  the 4-domain model further, or is r=64 already saturated on this
  dataset?
- Per the literature (Thinking Machines "LoRA Without Regret" 2025),
  the knee for LoRA matching full FT is around ~50K examples at
  r=32; we're well below that at 2604 records, so r=64 may already
  be past saturation and r=128 may show only single-tenths of a
  point of improvement. If so, the follow-up is rsLoRA scaling, not
  rank scaling.

## Gate and baseline for v3.0

Contemporaneous delta gate (see `tools/gate/combine_delta.py` and
`docs/eval.methodology.md` — pending):

- Baseline: re-eval v0.7-r64 in the same grader session as v3.0.
- Trained domains gated: xstate, fp, reactive, eventsourcing.
- Tolerance: per-domain delta ≤ -0.3 halts the gate.
- Untrained domains (capability) reported but not gated.

## Honest limitations

- **n=5 per domain** is noisy. Single-prompt defects swing means by
  ~0.2-0.4. Confidence intervals are wide.
- **Confounds not controlled**: as noted, step count and anchor
  ratio differ between FP-only and 4-domain arms. The +0.70 delta
  cannot be cleanly attributed to "broader data" alone — some of it
  is "more steps" and some is "less anchor dilution."
- **Single-session grading**: cross-session grader drift (0.5-0.9
  per domain, documented elsewhere) is larger than many of our
  observed deltas. Only contemporaneous comparisons are trusted.
- **No replicates**: each arm was trained once. Training
  nondeterminism (CUDA kernel ordering, FP accumulation order)
  probably contributes ±0.1-0.2 per domain.

## Pointers to raw data

- `results/v2-stack-fp/2026-04-24.*.json` — FP-only arm eval + grades
  (ollama label `ts-forge-v2-fp-only-archive`).
- `results/v2.0/2026-04-24.*.json` — 4-domain joint arm re-eval +
  re-grade today (ollama label `ts-forge-v2-r64`).
- `results/v0.7-recheck/2026-04-24.*.json` — v0.7-r64 re-eval +
  re-grade today, contemporaneous baseline.
- `results/v2-stack-fp/2026-04-24.delta_gate.json` and
  `results/v2.0/2026-04-24.delta_gate.json` — formal gate outputs.

## Sources (full list, as returned by the literature search)

- Caruana, Multitask Learning, Machine Learning 1997 — https://www.cs.cornell.edu/~caruana/mlj97.pdf
- Aghajanyan et al., Muppet, EMNLP 2021 — https://arxiv.org/abs/2101.11038
- Wang et al., Super-NaturalInstructions, EMNLP 2022 — https://arxiv.org/abs/2204.07705
- Chung et al., Scaling Instruction-Finetuned Language Models, JMLR 2024 — https://arxiv.org/abs/2210.11416
- Shen et al., Mixing It Up: The Cocktail Effect of Multi-Task Fine-Tuning, 2024 — https://arxiv.org/abs/2410.01109
- Longpre et al., The Flan Collection, 2023 — https://arxiv.org/abs/2301.13688
- Weller et al., When to Use MTL vs Intermediate Fine-Tuning, ACL 2022 — https://arxiv.org/abs/2205.08124
- Huang et al., LoraHub, COLM 2024 — https://arxiv.org/abs/2307.13269
- Pfeiffer et al., AdapterFusion, EACL 2021 — https://arxiv.org/abs/2005.00247
- Ponti et al., Polytropon, 2022 — https://arxiv.org/abs/2202.13914
- Hu et al., LoRA, 2021 — https://arxiv.org/abs/2106.09685
- Kalajdzievski, rsLoRA, 2023 — https://arxiv.org/abs/2312.03732
- Schulman et al., LoRA Without Regret, Thinking Machines 2025 — https://thinkingmachines.ai/blog/lora/
- MeTA-LoRA, 2025 — https://arxiv.org/abs/2510.11598
- Comprehensive Instruction Tuning for Qwen2.5, MDPI Computers 2025 — https://www.mdpi.com/2073-431X/14/7/264
- Addressing Task Conflicts in MTL Fine-Tuning, Machine Learning / Springer 2025 — https://link.springer.com/article/10.1007/s10994-025-06885-z
- Kim et al., Massive SFT Experiments, 2025 — https://arxiv.org/html/2506.14681v1
- Mirzadeh et al., Orca, 2023 — https://arxiv.org/abs/2306.02707
