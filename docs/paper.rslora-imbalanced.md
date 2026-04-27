# Paper outline — When rsLoRA Hurts: Vanilla LoRA Scaling Outperforms rsLoRA on Underrepresented Domains in Imbalanced Multi-Task Fine-Tuning

**Status**: outline + ablation plan. Not yet written. Existing empirical evidence
in `results/` (v3 phase). Required ablations not yet run.

**Authors**:
- **Zhifeng Zhang** ([ORCID 0009-0004-9509-5221](https://orcid.org/0009-0004-9509-5221)) — investigation lead, data synthesis design, training-recipe iteration, decisions about scope and direction across v0 → v3
- **Claude (Anthropic Opus 4.7)** — research-engineering collaborator: implementation of training/eval/grading pipeline, iteration on ablation design, literature search, drafting

(Author rendering can be adjusted — happy to use full name + affiliation, GitHub
handle, or a different ordering if preferred. AI-coauthor convention is evolving;
some venues require human-only authors with AI use disclosed in
acknowledgments. See "AI-coauthorship disclosure" note below.)
**Venue**: self-hosted preprint in this GitHub repo, optionally also arxiv if
formal DOI is wanted, optionally also workshop submission (e.g., NeurIPS PEFT
workshop, ICLR Tiny Papers, ML Reproducibility Challenge) if external review
is desired.
**Estimated length**: 6–8 pages (workshop-paper format), 2 figures, 3 tables.

---

## Title (alternatives, pick one)

1. **When rsLoRA Hurts: Vanilla LoRA Scaling Outperforms rsLoRA on Underrepresented Domains in Imbalanced Multi-Task Fine-Tuning**
2. **A Regime Where rsLoRA Underperforms Vanilla LoRA: Imbalanced Multi-Task Data Composition**
3. **Effective Scaling vs Data Imbalance: An Empirical Limit of rsLoRA**

(1) is most descriptive. (3) is most academic-flavored.

## Core claim

For LoRA fine-tuning on multi-task corpora with heavily imbalanced data
shares, the rank-stabilized (`alpha/sqrt(r)`) scaling proposed by
Kalajdzievski (2023) **monotonically harms the underrepresented task**
relative to vanilla (`alpha/r`) scaling at the same rank, despite
reducing aggregate training loss. The mechanism is amplification of
gradient bias toward the dominant-share task, crowding the
underrepresented task out of the shared low-rank subspace.

## Abstract (draft, ~200 words)

Rank-stabilized LoRA (rsLoRA), introduced by Kalajdzievski (2023),
replaces the original `alpha/r` scaling factor with `alpha/sqrt(r)`,
proving gradient stability at high rank and demonstrating monotonically
better aggregate loss across rank settings. Published evaluations test
single-task fine-tuning or balanced multi-task corpora. We report
empirical results showing that **under heavily imbalanced multi-task
fine-tuning** — specifically, when one target task represents only
~12% of corpus records while another represents 52% — vanilla `alpha/r`
scaling at r=128 outperforms rsLoRA on the underrepresented task by
+1.10 in held-out evaluation, while rsLoRA wins on the majority-share
tasks. We attribute this to the larger effective scaler under rsLoRA
amplifying gradient bias toward the dominant-share task in the shared
low-rank subspace, consistent with the cross-task interference
mechanism documented in LoRI (Zhang et al. 2025) and Ortho-LoRA
(2025). Our ablations vary rank (r=64/128/256), alpha scaling
independently of `alpha/r` vs `alpha/sqrt(r)`, and minority-domain
corpus share (5%–30%), establishing the crossover region where
vanilla scaling becomes the dominant choice. We recommend that
practitioners measure base-relative deltas on each task and consider
vanilla scaling for sparse-share targets in heterogeneous corpora.

## Sections

### 1. Introduction (~1 page)

- The promise of rsLoRA (Kalajdzievski 2023): gradient stability + monotone aggregate loss gains
- The under-tested regime: imbalanced multi-task fine-tuning
- Real-world motivation: code domains, instruction-tuning corpora, fine-tuning on private codebases all have natural imbalance
- Our contribution: empirical demonstration that rsLoRA underperforms vanilla LoRA on the underrepresented domain in this regime, plus ablations characterizing the crossover

### 2. Background (~1 page)

- LoRA basics (Hu et al. 2021): low-rank adapters, scaling factor `alpha/r`
- rsLoRA (Kalajdzievski 2023): theoretical motivation, `alpha/sqrt(r)` derivation, balanced-task results
- Multi-task LoRA interference (Caruana 1997, Zhang et al. 2025 LoRI, Ortho-LoRA 2025)
- Adapter composition pitfalls (Pfeiffer et al. AdapterFusion, Yadav et al. TIES, LoRA Soups)

### 3. Method (~1 page)

- **Substrate**: Qwen3-14B (bnb-4bit quantized via Unsloth/bitsandbytes), 7 target modules (q/k/v/o/gate/up/down) × 40 layers
- **Training**: SFTTrainer, batch=2, grad_accum=4, lr=1e-4, 3 epochs, bf16, seed=42
- **Hardware**: RTX 5090 32 GB
- **Eval**: 24-prompt held-out suite covering 5 TypeScript domains
- **Grading**: blind two-grader Claude-subagent protocol with contemporaneous baseline measurement (key methodological contribution)
- **Corpus**: 2604 records across 4 imbalanced TypeScript domains (XState 17%, FP 12%, RX 9%, ES 52%, capability 9%)
- **Manipulated variables**: rank, alpha, scaling type (vanilla / rsLoRA), corpus composition

### 4. Primary result (already collected from v3 phase, ~1 page)

| Arm | r | alpha | Scaling | FP score | XState | RX | ES | Cap | Avg |
|---|---|---|---|---|---|---|---|---|---|
| Base Qwen3-14B (no FT) | — | — | — | 2.80 | 1.30 | 4.40 | 3.30 | 3.75 | 3.11 |
| v3.0 vanilla | 128 | 128 | `alpha/r` | **4.50** | 3.70 | 4.70 | 4.30 | 4.13 | 4.27 |
| v3.0-rslora | 128 | 128 | `alpha/sqrt(r)` | 3.40 | **4.40** | 4.90 | 4.10 | 4.13 | 4.19 |

**Key observation**: at identical rank, identical data, identical training
records, identical hyperparameters except scaling type, vanilla's FP score
is +1.10 higher than rsLoRA's. rsLoRA's XState is +0.70 higher than
vanilla's. The trade-off is binding.

Base-relative training deltas:
- Vanilla added +1.70 to FP, +2.40 to XState
- rsLoRA added +0.60 to FP, +3.10 to XState
- For the dense-share XState (17%), rsLoRA's stronger influence helped (+0.70 over vanilla)
- For the sparse-share FP (12%), rsLoRA's stronger influence hurt (-1.10 from vanilla)

### 5. Ablations (the experiment plan to close reviewer gaps)

The published rsLoRA paper claims monotonic gains across rank. Our
counter-claim is that this depends on data balance. Required ablations
to defend the claim:

#### Ablation A: Rank sweep (4 arms, ~3 hours)

Hold data fixed at the v3 corpus (2604 records, FP at 12%). Vary rank.

| Run | r | alpha | Scaling | Status |
|---|---|---|---|---|
| A1 | 64 | 64 | `alpha/r` | NEW — ~50 min |
| A2 | 64 | 64 | `alpha/sqrt(r)` | NEW — ~50 min |
| A3 | 128 | 128 | `alpha/r` (=v3.0) | DONE |
| A4 | 128 | 128 | `alpha/sqrt(r)` (=v3.0-rslora) | DONE |
| A5 | 256 | 256 | `alpha/r` | NEW — ~70 min |
| A6 | 256 | 256 | `alpha/sqrt(r)` | NEW — ~70 min |

Predicted: vanilla beats rsLoRA on FP at all three ranks; gap widens with
rank. (At r=64, vanilla effective scaler 1× vs rsLoRA 8×; at r=256,
vanilla 1× vs rsLoRA 16×.)

#### Ablation B: Alpha decoupling (6 arms, ~5 hours)

Hold rank at r=128, hold data fixed. Vary alpha to decouple the
"effective-scaler magnitude" question from the rsLoRA-vs-vanilla
identification.

| Run | r | alpha | Scaling | Effective scaler | Status |
|---|---|---|---|---|---|
| B1 | 128 | 64 | `alpha/r` | 0.5× | NEW |
| B2 | 128 | 128 | `alpha/r` (=A3=v3.0) | 1.0× | DONE |
| B3 | 128 | 256 | `alpha/r` | 2.0× | NEW |
| B4 | 128 | 64 | `alpha/sqrt(r)` | 5.7× | NEW |
| B5 | 128 | 128 | `alpha/sqrt(r)` (=A4=v3.0-rslora) | 11.3× | DONE |
| B6 | 128 | 256 | `alpha/sqrt(r)` | 22.6× | NEW |

This isolates "effective-scaler magnitude" as the actual independent
variable. If vanilla's advantage on FP comes purely from low effective
scaler, then B1 should outperform B2 on FP, and B4 (rsLoRA at low alpha)
should match or beat B2 (vanilla at default). If the advantage is
specific to vanilla scaling, then B4 should NOT match B2.

This is the methodologically critical ablation — reviewers will demand it.

#### Ablation C: Corpus-share sweep (8 arms, ~7 hours)

Hold rank at r=128, hold alpha=128. Vary FP's corpus share by
sub-sampling or replicating FP records.

| Run | FP share | FP records | Total records | Vanilla | rsLoRA |
|---|---|---|---|---|---|
| C1 | 5% | 130 | 2414 | NEW | NEW |
| C2 | 12% | 320 | 2604 (=v3 corpus) | DONE (=A3) | DONE (=A4) |
| C3 | 20% | 600 (320 ×1.875) | 3000 | NEW | NEW |
| C4 | 30% | 1100 (320 ×3.4) | 3700 | NEW | NEW |

Predicted: at low FP share (5%, 12%), vanilla wins FP. At high FP share
(20%+), rsLoRA matches or beats vanilla on FP. Crossover region locates
the data-imbalance threshold where vanilla's advantage emerges.

#### Total experiment cost

Approximately 12 new training runs at ~50–70 min each = 12–14 hours of
compute. Plus pipeline (merge + GGUF + eval + graders) ~30 min per arm
= 6 hours. Total: **~20 hours of compute** to produce the full ablation
matrix. Two days of background runs.

Plus same-session grader evaluation across all 18 arms (12 new + 6 already
done) = 36 grader dispatches. ~3–4 hours of subagent time.

### 6. Mechanism / discussion

The mechanism we hypothesize, in plain language: a LoRA's gradient at
each step is the average of per-record gradients in the batch. With
imbalanced batches (e.g., 8/8 batches average 4 ES records, 1 FP record),
the gradient direction is dominated by the majority share. A larger
effective scaler (rsLoRA at r=128 = 11.3×) translates that dominant
gradient direction into larger weight updates per step. The shared
low-rank subspace gets occupied by the dominant-task pattern; minority-
task patterns get crowded out.

Literature support:
- Zhang et al. 2025 LoRI (arxiv 2504.07448): "low-rank constraint limits the
  optimization landscape's capacity to accommodate diverse task requirements"
- Ortho-LoRA 2025 (arxiv 2601.09684): task-specific gradients collide in shared
  low-rank subspace; conflicts are zero-sum at fixed r
- Caruana 1997: shared parameters create both transfer and interference; bounded
  capacity makes trade-offs intrinsic

What's novel: the explicit prediction that *effective scaling magnitude*
controls how strongly the dominant task crowds the minority. rsLoRA's
correctness for gradient stability is preserved; the unintended
consequence under data imbalance is the new finding.

### 7. Limitations

- Single base model (Qwen3-14B). Results may differ for smaller (7B) or
  larger (70B) bases. Future work: replicate at 7B and 70B class.
- Single domain pair (TypeScript). Results may differ for natural-language
  multi-task instruction-tuning. Future work: replicate on Tülu/UltraFeedback.
- LLM-as-judge evaluation has documented drift; we mitigate with same-session
  matched-grader calibration but absolute scores still have ±0.2 noise.
- Eval suite is small (n=5/domain). Future work: expand to 60–80 prompts
  for tighter confidence intervals.
- We use bnb-4bit quantization. Full-precision training may exhibit
  different scaling behavior.

### 8. Related work

- Hu et al. 2021 — LoRA original (arxiv 2106.09685)
- Kalajdzievski 2023 — rsLoRA (arxiv 2312.03732) — primary reference
- Pfeiffer et al. 2020 — AdapterFusion (arxiv 2005.00247)
- Caruana 1997 — Multitask Learning
- Aghajanyan et al. 2021 — MUPPET (arxiv 2101.11038)
- Zhang et al. 2025 — LoRI (arxiv 2504.07448)
- Ortho-LoRA 2025 — Task conflict in shared subspace (arxiv 2601.09684)
- Yang et al. 2024 — MTL-LoRA (arxiv 2410.09437)
- Yadav et al. — TIES merging
- Prabhakar et al. 2024 — LoRA Soups (arxiv 2410.13025)
- Shen et al. 2024 — Cocktail Effect (arxiv 2410.01109)

### 9. Reproducibility

- Training scripts: `v2/train.py` (with `--rslora` flag and configurable `--rank`)
- Eval pipeline: `tools/eval/run.py`, `tools/gate/combine_delta.py`
- Hyperparameters fixed: lr=1e-4, batch=2, grad_accum=4, 3 epochs, seed=42
- All training data derived from `v0.7/seeds/patterns.fp.md` and `v0.7/seeds/phrasings.md` plus subagent synthesis (subagent prompts in `tools/gate/grader_prompt.md` for grading)
- Verified data: `v3.1/data/synth.verified.jsonl` (2882 records covering all 15 FP patterns + 4-domain mix)
- Hardware: RTX 5090 (32 GB VRAM); should reproduce on any 32 GB card with comparable performance

---

## Production checklist

If pursuing the paper:

- [ ] Run Ablation A (rank sweep, 4 new arms, ~3 hrs)
- [ ] Run Ablation B (alpha decoupling, 4 new arms, ~3 hrs) — most critical
- [ ] Run Ablation C (corpus-share sweep, 8 new arms, ~7 hrs)
- [ ] Same-session grader evaluation across all 18 arms
- [ ] Compute base-relative deltas with confidence intervals (bootstrap on per-prompt scores)
- [ ] Generate two figures: (a) FP score vs rank for vanilla and rsLoRA; (b) FP score vs corpus share for vanilla and rsLoRA
- [ ] Draft paper (~6 pages)
- [ ] Internal review (1 round)
- [ ] Publish: commit `paper.pdf` to this repo's `docs/paper/` directory; tag a v1.0 release; optionally arxiv submission for DOI

Estimated wall time end-to-end if done in earnest: **3-5 days** of mixed
compute + writing. Could be condensed if dedicated focus.

---

## Where to publish

In order of effort:

1. **This GitHub repo (lowest effort)**: commit `docs/paper/paper.pdf`, link
   from README. Zero gatekeeping. The user explicitly OK with this option.

2. **arXiv preprint (low effort, formal DOI)**: arxiv accepts unrefereed
   preprints in cs.LG. Provides a citable DOI. About 1 day to format
   to arxiv-acceptable LaTeX.

3. **Workshop submission (medium effort, peer review)**: NeurIPS / ICLR /
   ACL all run PEFT-related workshops. Acceptance is competitive but
   feedback is valuable. About 1 week of formatting and response cycle.

4. **Conference / journal (high effort)**: empirical paper of this scope
   is more workshop-y than full-conference; pursue only if a fuller
   experimental scope (e.g., multiple base models) is added.

The user-stated preference is option (1) — self-host in this repo. That
keeps the work in one place and doesn't require external gatekeeping.
Option (2) takes one extra day if a citable DOI matters.

---

## Author contributions (CRediT-style)

For transparency in any version of this paper that ships externally:

- **Conceptualization, Funding acquisition, Resources, Project administration**: Zhifeng Zhang
- **Investigation, Methodology**: Zhifeng Zhang (research direction, hypothesis, scope decisions); Claude Opus 4.7 (implementation choices, ablation design proposals)
- **Data curation, Software**: Claude Opus 4.7 (data synthesis subagent orchestration, training/eval/grading pipeline implementation, tooling); Zhifeng Zhang (synthesis discipline rules, verifier design, oversight)
- **Formal analysis, Visualization**: jointly
- **Writing — original draft**: Claude Opus 4.7 (initial draft of cross-version conclusions, paper outline, literature posture)
- **Writing — review & editing**: Zhifeng Zhang (corrections, framing pushback, methodology questioning that surfaced load-bearing findings such as the cross-session grader-drift mistake)
- **Validation, Decision-making**: Zhifeng Zhang (ship/halt/pivot calls; reframing of priority lens; final recipe selection)

## AI-coauthorship disclosure note

Different venues handle AI authorship differently:

- **Self-published GitHub preprint (the stated plan)**: no formal restriction. Listing Claude Opus 4.7 as co-author is fine. Use this paper's CRediT-style contributions statement above for full transparency.
- **arXiv preprint**: arXiv allows AI as co-author when disclosed; their guidance is to list specific model+version (e.g., "Claude Opus 4.7 (Anthropic, 2026)"). Acceptable per current arXiv policy.
- **Workshop / conference / journal**: many venues now explicitly disallow AI co-authorship and require AI use to be disclosed in acknowledgments instead. Examples: ACL, ICLR, Nature, Science. If submitting to such a venue, change the author line to "Zhifeng Zhang" alone and add an Acknowledgments section: "AI assistance: this work used Claude Opus 4.7 (Anthropic) for [specific tasks]; the author had final responsibility for all claims, methodology, and results."

The format here defaults to the GitHub-preprint convention. Easy to switch if
the publication venue changes.

## Status

Outline only. The two existing arms (v3.0 vanilla and v3.0-rslora) are
the seed evidence. The 12 new ablation arms are the work that closes
methodological gaps. None of the new arms have been run.

This document exists so future-you (or any collaborator) can pick up
the experimental thread without rediscovering the design.
