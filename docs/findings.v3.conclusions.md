# v3 Investigation — Conclusions

**Date**: 2026-04-25
**Substrate**: `unsloth/qwen3-14b-unsloth-bnb-4bit` (Qwen3-14B, bnb-4bit)
**Hardware**: RTX 5090 (32 GB VRAM)
**Toolchain**: Unsloth 2026.4.6, transformers 5.5.0, trl current, peft 0.19.1, bitsandbytes-CUDA 12.8
**Eval suite**: `tools/eval/suite.json` — 24 prompts (5 xstate, 5 fp, 5 reactive, 5 eventsourcing, 4 capability)
**Daily driver after v3 conclusions**: `ts-forge-v3:latest` (alias of `ts-forge-v3.0` — vanilla LoRA r=128). Selected after reframing the priority lens as **FP + Reactive-style** (where Reactive-style groups xstate/rx/es as one paradigm of typed event-driven streams). Under that lens, v3.0 vanilla is the highest-scoring arm; see §"FP + Reactive-style consolidated lens" below.

This document is the durable artifact of the v3 investigation. It captures
methodology lessons that transfer across base models and recipes, the
empirical scores from contemporaneously-graded evals, and the open questions
left for future arms.

---

## Lead methodology lesson

**Every grader session that produces ship-gate data must include the
untrained base model AND the strongest off-the-shelf alternative on the
same suite, with the same graders, in the same parallel batch.**

Without that, "regression vs prior arm" is meaningless — we can't tell
if a trained model is below the baseline, below the untrained base, or
just measured under stricter graders than last time.

This rule is non-negotiable for any future arm. If we move to a new
base model (e.g., Qwen3.5), the FIRST measurement on that model is the
RAW base on the suite, BEFORE any fine-tune. That establishes the
baseline against which every fine-tuned arm gets evaluated.

The corollary failure: from v0 through v2, we gated against the
"v0.7-r64 historical floor" without ever measuring v0.7-r64 in the
same grader session as the new arm. Five arms (v1, v1.1, v1.2, v2.0,
v2-stack-fp) "halted" against numbers that contained 0.5–0.9 of
non-reproducible session drift. At least three of those halt
decisions were measurement artifacts, not real regressions.

---

## Reference scores (session 2026-04-25b, 10-grader matched calibration)

All 5 arms evaluated and graded in the same session, same prompts,
same temperature 0.6, same blind-protocol grader pool dispatched in
one parallel batch.

| Domain | Base Qwen3-14B¹ | Gemma 4 31B | v0.7-r64 (prior driver) | v3.0 (vanilla r=128) | **v3 (= v3.0-rslora)** | v3.1 (FP-augmented) | v3.2 (stacked) |
|---|---|---|---|---|---|---|---|
| XState | 1.30 | 3.50 | 3.80 | 3.70 | **4.40** | 4.20 | 4.00 |
| FP | 2.80 | 3.70 | 4.10 | 4.50 | 3.40 | **4.20** | 2.80 |
| Reactive | 4.40 | 4.90 | 4.80 | 4.70 | **4.90** | 4.40 | **5.00** |
| Event Sourcing | 3.30 | **4.50** | 2.90 | 4.30 | 4.10 | 4.10 | 4.30 |
| Capability | 3.75 | 4.00 | 4.00 | 4.13 | 4.13 | 4.00 | 4.13 |
| **Average** | **3.11** | **4.12** | **3.92** | **4.27** | **4.19** | **4.18** | **4.05** |

¹ Base Qwen3-14B was measured in session 2026-04-24f (the day before).
Other measurements are from the matched session 2026-04-25b. Cross-session
drift on this suite has been observed at 0.1–0.2 per domain even on the
same model — treat the base-Qwen3 column as ±0.2 indicative, not exact.

Grader agreement across all 5 arms in 2026-04-25b: 24/24 within 1 pt,
mean disagreement 0.12–0.29 per arm. Tightest agreement we've achieved.

---

## Per-arm summary

### v3.0 (vanilla LoRA, r=128, 2604 records)
- Joint 4-domain training on v0.6 XState (440) + v0.7 FP (320) + v0.7 RX (240) + v1 ES (1364) + 240 anchors.
- Vanilla LoRA scaling: `alpha = r = 128`, effective LoRA scaler ≈ 1.
- Train_loss: 0.318
- Beats every per-domain target except XState (3.70 vs Gemma 3.50 = +0.20 only).
- **Best avg of all v3 arms (4.27).** But the avg edge over Gemma is +0.15 — not "sizeable."

### v3.0-rslora (rsLoRA, r=128, same 2604 records)
- **Same data and rank as v3.0.** Only difference: `use_rslora=True` → effective LoRA scaler ≈ 11.3.
- Train_loss: 0.239 (25% tighter than vanilla).
- **Strongest XState in the entire project: 4.40** (Gemma 3.50 by +0.90).
- FP regressed to 3.40 (vs v3.0 vanilla's 4.50). The stronger LoRA influence amplified ES-dominant data signal at FP's expense.
- Retained as a domain specialist (best for XState-heavy work) but NOT the daily driver, because FP regression is too severe under the FP-priority lens.

### v3.0 vanilla (chosen daily driver)
- **Aliased as `ts-forge-v3:latest`** in ollama after the reframing to FP + Reactive-style as the priority domains.
- Highest FP score across all v3.x arms: 4.50 (training added +1.70 over base 2.80 — the largest single-domain training contribution we've measured).
- Highest both-domain (FP, Reactive-style) average: 4.37.
- Counter-intuitive but empirical finding: vanilla LoRA scaling (effective scaler 1×) transferred FP **better** than rsLoRA scaling (≈ 11.3×) at our 12%-FP corpus composition. Stronger LoRA influence amplified the ES-dominance bias; milder influence absorbed FP cleanly.

### v3.1 (rsLoRA, r=128, 2882 records)
- v3 corpus + 280 new FP records covering 7 missing patterns from `v0.7/seeds/patterns.fp.md`.
- Closes FP gap: FP 3.40 → 4.20 (+0.80).
- But XState dropped 4.40 → 4.20, RX dropped 4.90 → 4.40, ES held.
- Demonstrates the **capacity wall**: at fixed r=128, more FP records redistribute capacity rather than expanding it.

### v3.2 (LoRA stacking — frozen v3 adapter + new r=64 FP-recovery adapter)
- Trained the new adapter on FP-only data (598 FP records + 60 anchors) on top of the frozen v3 adapter.
- **Failed.** FP regressed to 2.80 — worse than the frozen v3 base alone (3.40).
- Likely cause: the new r=64 adapter's smaller deltas were overwhelmed by v3's r=128 + rsLoRA effective scaling at inference. The "fix" couldn't override the existing FP behavior on held-out prompts.
- Independent capability isolation didn't deliver the expected combined-strength result.
- Useful negative result: confirms that single-LoRA stacking with imbalanced effective scaling between adapters produces an inference-time domination effect that defeats the additive-deltas premise.

---

## FP + Reactive-style consolidated lens

After running the matched 5-way grader sessions, the user reframed
priorities: the actual workloads are **FP + Reactive-style**, where
"Reactive-style" includes xstate v5, RxJS, and Event Sourcing because
all three share the typed-event, immutable, declarative paradigm vs
imperative TypeScript. Under this 2-domain lens (FP weighted equally
with the average of the three reactive-style domains):

| Arm | FP | Reactive-style avg | Both-domain avg |
|---|---|---|---|
| Base Qwen3-14B | 2.80 | 3.00 | 2.90 |
| Gemma 4 31B | 3.70 | 4.30 | 4.00 |
| v0.7-r64 | 4.10 | 3.83 | 3.97 |
| **v3.0 vanilla** ★ daily driver | **4.50** | **4.23** | **4.37** |
| v3.0-rslora | 3.40 | 4.47 | 3.94 |
| v3.1 | 4.20 | 4.23 | 4.22 |
| v3.2 stacked | 2.80 | 4.43 | 3.62 |

Under this lens, v3.0 vanilla beats Gemma 4 31B by **+0.37** on the
both-domain average. That IS a sizeable margin — the project's
original goal achieved, but only when measured against the workload
the user actually cares about, not the equally-weighted 5-domain
gate we'd been using.

## Recipe lessons (transferable to next base model)

### Vanilla LoRA scaling at r=128: small effective scaler, but real training transfer
- `alpha = r` with `alpha/r` (vanilla) gives effective LoRA scaler = 1 regardless of rank.
- This makes the LoRA's per-layer numerical contribution to forward pass small relative to base weights.
- **But that does NOT mean training transfer is small.** v3.0 vanilla added +1.70 FP over base — the largest training delta we've measured.
- Counter-intuitive finding: at imbalanced data composition (FP at 12% of corpus), milder LoRA influence transferred FP **better** than rsLoRA. The stronger rsLoRA influence amplifies the dominant-domain (ES at 52%) bias, crowding out sparse-domain learning.
- **Don't equate "small effective scaler" with "weak training effect"** — they're orthogonal.

### rsLoRA scaling lets rank actually translate to capacity
- `alpha/sqrt(r)` (rsLoRA, Kalajdzievski 2023) gives effective scaler ≈ 11.3 at r=128 with `alpha=r`.
- Train_loss drops 25% (0.32 → 0.24). Real per-domain gains (XState +0.50–0.90, ES +0.20).
- **But**: stronger influence amplifies BOTH well-trained and under-trained patterns. Sparse-data domains (FP at 320 records, 12% of corpus) collapse because the LoRA fragmentary memorization dominates over base-model fallback.

### "Synthesis fills training gaps" is partial fix
- v3.1 added 280 records to fill FP's 7 missing patterns. FP recovered (+0.80).
- But other domains regressed: capacity at fixed r=128 got redistributed.
- **The wall: at fixed parameter budget, single-LoRA can't max all domains.** Each gain comes from another domain's loss.

### LoRA stacking (frozen + new) doesn't trivially compose
- v3.2 tested the architecturally-cleanest fix: freeze the strong adapter, layer a new corrective adapter for the weak domain.
- Stacking math is correct (sum of deltas).
- BUT: at inference, smaller adapter's contribution is overwhelmed by stronger frozen adapter's effective scaling. Training-time loss reflects combined model; inference behavior on held-out prompts reverts to dominant-adapter patterns.
- Need careful adapter-strength balancing if stacking is to work. Or: use vanilla scaling on the frozen base + rsLoRA on the new adapter to invert dominance.

### Warm-start does not preserve capability (already established v1.1/v1.2)
- v1.1 warm-started from v0.7's merged weights with full data → FP regressed.
- v1.2 warm-started + delta-only data (no FP) → FP catastrophically collapsed to 2.20.
- Mechanism: a LoRA is a global weight delta; new training perturbs all domains regardless of what's in the batch.

### Atomic pattern drilling > compositional scenarios (already established v1)
- v0.7's FP=4.40 came from 8 atomic-drill batches × 40 variations.
- v1's FP=3.80 came from compositional scenarios (e.g., "build an HTTP handler with Effect").
- For new domain teaching, drill ONE API surface per batch. Compositional records add complexity without per-pattern training density.

---

## Eval methodology lessons

### Cross-session grader drift is real, large, and unaccounted-for in raw scores
- Same model, same prompts, two grader sessions on different days: 0.5–0.9 per-domain swing.
- v0.7-r64 measured at ship time (FP=4.40) vs re-checked 2026-04-24 (FP=3.90). Same model.
- **Implication**: never gate against historical numbers. Always re-eval the reference baseline in the same session as the new arm.

### Temperature 0.6 sampling adds further per-eval noise
- Same model + same prompts + same graders, two eval runs at temp=0.6 → ±0.2 domain swing.
- Single-prompt variance amplifies to ±0.4 in domain mean at n=5.
- **Mitigation**: `tools/eval/run.py --temperature 0` (now the default) makes outputs deterministic. Still need to re-eval baseline in same session for grader drift.

### Two blind graders are sufficient when calibration is held
- 24/24 within 1 pt and mean disagreement 0.12–0.29 across the v3.x sessions.
- Three-grader tiebreak hasn't been needed.

### Eval suite is too small (n=5/domain) for tight gates
- Single-prompt swings at n=5 amplify to ±0.4 domain mean. Gate tolerances of ±0.3 are sub-noise.
- **Future**: expand `tools/eval/suite.json` from 24 to 60–80 prompts. Add 6–10 prompts per domain. Larger n shrinks single-prompt weight.

---

## Reference benchmarks (the "where do we sit" anchors)

- **Base Qwen3-14B (no fine-tune)**: 3.11 avg. Strong only on Reactive (4.40, RxJS is well-represented in pretraining). XState 1.30 — essentially zero capability without training.
- **Gemma 4 31B (no fine-tune)**: 4.12 avg. Strong on Reactive (4.90), ES (4.50). Weak on XState (3.50). Strong off-the-shelf reference but NOT frontier.
- **Claude Opus 4.7 (historical, frontier)**: 4.95 avg per `v1.1/decision.md`. The actual frontier ceiling. Not deployable locally.
- **v3 daily driver**: 4.19 avg (slightly above Gemma 4 31B in 9 GB vs 19 GB). XState 4.40 (genuinely best across all measured options including Gemma).

---

## What we did and did NOT achieve (honest)

The original framing was "make a small (9 GB) model better than Gemma 4 31B
by sizeable margin on every domain." Under the **5-domain equally-weighted
lens**, this was NOT achieved — best v3 arm sat +0.07 to +0.15 over Gemma
on average, with XState +0.90 but other domains tying or losing. The
capacity wall (v3.0 ≈ v3.0-rslora ≈ v3.1 in average score) is real:
single-LoRA at fixed parameter budget can't both fix sparse-domain gaps
AND preserve dense-domain gains.

Under the **FP + Reactive-style 2-domain lens** (the user's actual
workload priority), v3.0 vanilla DID achieve the goal: 4.37 both-domain
avg vs Gemma's 4.00 = +0.37, sizeable.

This reframing changed which recipe was the right ship target:
- v3.0-rslora was the strongest XState specialist (4.40 vs Gemma 3.50)
  but had FP collapse (3.40).
- v3.0 vanilla had stronger FP (4.50) and broadly comparable Reactive-style
  (4.23 vs v3.0-rslora's 4.47) — better balanced for the priority lens.

Lesson: **the right recipe depends on the priority weighting**. We had
been gating against an equally-weighted 5-domain average; the user's
actual workload is FP-heavy with Reactive-style as the second priority.
Once the priority weighting was made explicit, the model selection
became unambiguous.

---

## What this means for the project's value proposition

For PUBLIC TypeScript code, fine-tuning a 14B model marginally beats Gemma 4 31B
but doesn't dominate. **The genuine moat is proprietary knowledge** that no
off-the-shelf model can have:

- Codebases: `qi-v2-qicore` (the user's `Result<T>` / `QiError` library) and any other private codebase.
- Internal frameworks, project-specific conventions, idioms.
- Legal/compliance-sensitive code, IP-bound code that can't be in public training data.

For these, Gemma 4 31B has zero training. Our 14B + LoRA + Unsloth pipeline
is the only viable path on consumer hardware. The substrate work (v3) is
the foundation; the proprietary-library training is the actual product.

The v3 substrate at v3-r128-rsLoRA is "good enough" to layer proprietary
knowledge on top via future stacked adapters or warm-start variants.

---

## Porting checklist for next base model (Qwen3.5 or beyond)

When migrating to a newer base (e.g., Qwen3.5 dense small variants released
2026-03-02), the methodology is portable. Order of operations:

1. **Establish base reference.** Eval the raw new-base model on the suite,
   in a grader session, BEFORE any fine-tune. This is the baseline.
2. **Establish off-the-shelf reference.** Eval Gemma 4 31B (or whatever the
   strongest off-the-shelf code model is at the time) in the same grader
   session. Sets the bar to clear.
3. **Run vanilla LoRA r=64 baseline.** Mirrors v0.7 substrate; sanity-check
   that the recipe still works on the new toolchain.
4. **Run rsLoRA r=128.** This is v3 substrate. Should outperform vanilla.
5. **All gates contemporaneous.** Eval new arm + same-session base + Gemma in
   one batch. Compute deltas, not absolute thresholds.
6. **Don't expand the corpus until the substrate is validated.**

Re-usable tools that survive the port:
- `tools/eval/run.py` — eval runner, model-agnostic
- `tools/eval/tool_call_smoke.py` — tool-call canary, model-agnostic
- `tools/eval/suite.json` — 24-prompt suite
- `tools/gate/combine_delta.py` — contemporaneous delta gate
- `tools/gate/grader_prompt.md` — grader template
- `tools/merge_adapter.py` — single-adapter merge recovery
- `tools/merge_adapter_stack.py` — multi-adapter sequential merge
- `v2/train.py` — training script (with `--rslora` flag); just change `MODEL =`

---

## Literature posture (deep search 2026-04-25)

How each finding sits relative to published research:

| # | Finding | Posture |
|---|---|---|
| 1 | Vanilla LoRA scaling beat rsLoRA on the 12%-share sparse domain at r=128 | **Novel / partially counter-consensus.** Kalajdzievski 2023 (arxiv 2312.03732) — the rsLoRA paper — claims monotone aggregate-loss gains across r, but only tests single-task or balanced multi-task. Imbalanced multi-task (this regime) is not evaluated. LoRI (arxiv 2504.07448) and Ortho-LoRA (arxiv 2601.09684) document the interference mechanism that supports our observation. |
| 2 | Stacked frozen+new adapter — stronger adapter dominates inference, new adapter's corrective deltas overwhelmed | **Partially documented.** The general principle (naïve LoRA sums fail) is established by AdapterFusion (arxiv 2005.00247), LoRA Soups (arxiv 2410.13025), Crowded-in-B-Space (arxiv 2604.16826), TIES/DARE. The specific scaling-mismatch failure mode (rsLoRA-trained frozen adapter dominating smaller-rank rsLoRA new adapter) is fresh evidence. |
| 3 | At fixed rank, capacity is fungible across domains | **Well-established. We rediscovered it.** LoRI (arxiv 2504.07448), Ortho-LoRA (arxiv 2601.09684), MTL-LoRA (arxiv 2410.09437), Caruana 1997. |
| 4 | Cross-session LLM-judge drift 0.5–0.9 vs intra-session 0.17–0.29 | **Partially documented.** Drift is known qualitatively; this quantitative split is sharper than published numbers (MT-Bench arxiv 2306.05685, Position-Bias arxiv 2406.07791, LLM-as-Judge Survey arxiv 2411.15594, Scoring Bias arxiv 2506.22316). |
| 5 | Base-relative attribution required for training-effect measurement | **Methodological orthodoxy.** Kumar 2022 (arxiv 2202.10054), MUPPET (arxiv 2101.11038), Amuro & Char (arxiv 2408.06663). |
| 6 | Multi-task joint > single-task on target domain at same rank | **Well-established cocktail effect.** Caruana 1997, MUPPET, Flan-T5 (arxiv 2210.11416), Cocktail Effect (arxiv 2410.01109). |

The publishable extension, if ever pursued, is Finding 1. The required
ablation per literature reviewers: (a) replicate at r=64 and r=256 to show
rank-robust; (b) decouple `alpha` magnitude from the rsLoRA-vs-vanilla
choice (vary alpha independently); (c) sweep minority-domain corpus share
(5%, 10%, 20%, 30%) to find the crossover. Each ablation is ~50 min of
training; 6–8 arms = a publishable empirical paper on "when rsLoRA hurts."

## Open questions

These remain unresolved at the end of v3. None are blocking the daily-driver
ship, but each could be the focus of future work.

1. **Why did v3.2 stacking actually regress FP?** The math says
   `W = W₀ + ΔW_v3 + ΔW_fp_recovery` should be additive and at minimum
   neutral on FP relative to frozen v3. Yet held-out FP went 3.40 → 2.80.
   Hypothesis: stronger r=128 rsLoRA dominates smaller r=64 rsLoRA at
   inference; the fp_recovery adapter's deltas can't override on prompts
   not seen in training. Test: train fp_recovery at r=128 + rsLoRA (same
   as frozen) and re-evaluate. If parity, the dominance hypothesis is
   confirmed.

2. **Does style-preference training shift defaults?** All current
   training records explicitly name the target paradigm in the prompt
   (e.g., "Use Effect.gen + yield* to..."). The model learns
   competence-when-instructed, not preference-when-free. Untested:
   synthesize ~500 ambiguous prompts that admit FP/RX/imperative all,
   with idiomatic FP responses. Train as a separate batch. Measure
   whether v3 + ambiguous-prompt-batch defaults to FP vs imperative.
   Cited literature: Marquardt 2025 (arxiv 2507.04889), LIMA, Ghosh 2024.

3. **What's the right rsLoRA alpha for this dataset composition?**
   `lora_alpha = r = 128` gives effective scaling 11.3. Possibly too
   strong for our ES-dominant 52%-of-corpus mix. Test with
   `lora_alpha = sqrt(r) * 4 ≈ 45` (effective scaling ≈ 4) to moderate
   the redistribution effect. Single experiment. ~50 min.

4. **Does Qwen3.5 close the gap to Gemma 4 31B for free?** Gemma's edge
   is mostly from broader pretraining data. Qwen3.5 dense small variants
   were released 2026-03-02 and may already match or exceed Gemma at
   smaller parameter counts. Migrating substrate to Qwen3.5 might
   render the substrate-vs-Gemma gap question moot.

5. **Eval suite expansion (n=5 → n=15+ per domain).** Reduces
   single-prompt variance to acceptable levels for tighter gates.
   Independent of any training arm.

6. **Proprietary-library training (qi-v2-qicore as concrete case).**
   Build the data-extraction pipeline from a real codebase. Synthesize
   (instruction, code) pairs that exercise its API surface. Train a
   per-codebase LoRA stacked on top of v3 substrate. Validate that the
   resulting model writes correct code USING that codebase. This is
   the real product the framework enables.

---

## Artifacts produced (v3 phase)

**Models in ollama**:
- `ts-forge-v3:latest` ← daily driver (alias of v3.0-rslora)
- `ts-forge-v3.0:latest` ← r=128 vanilla
- `ts-forge-v3.0-rslora:latest` ← same model as v3 (kept for traceability)
- `ts-forge-v3.1:latest` ← r=128 rsLoRA, FP-augmented corpus
- `ts-forge-v3.2:latest` ← stacked LoRA (failed experiment, retained for reference)
- `ts-forge-v0.7-r64:latest` ← previous daily driver (retained for comparison)

**Trained adapters and merged safetensors**:
- `v3.0/`, `v3.0-rslora/`, `v3.1/`, `v3.2/` — full training+gguf artifacts per arm
- `v2/data/synth.verified.jsonl` (2604) — base v3 corpus
- `v3.1/data/synth.verified.jsonl` (2882) — corpus with 7 missing FP patterns filled
- `v3.1/data/synth.fp.batch-{I..O}.verified.jsonl` (278) — 7 new FP pattern batches

**Tools**:
- `tools/eval/{run.py, suite.json, tool_call_smoke.py}` — eval pipeline
- `tools/gate/{combine_delta.py, grader_prompt.md}` — delta gate + grader template
- `tools/merge_adapter.py` — single-adapter merge recovery
- `tools/merge_adapter_stack.py` — multi-adapter sequential merge

**Results** (under `results/<arm>/<session>.{raw,graded.grader_A,graded.grader_B,delta_gate}.json`):
- 2026-04-22 (initial v3.0)
- 2026-04-24a, 2026-04-24e, 2026-04-24f (cross-arm session calibrations)
- 2026-04-25a, 2026-04-25b (final 4-way and 5-way comparisons)

**Findings docs**:
- `docs/findings.multi-domain.md` — earlier finding (multi-task joint > single-task) with literature
- `docs/findings.v3.conclusions.md` — this document

---

## Session-state trail

- Substrate iteration: v0 → v0.6 → v0.7 (shipped as the prior daily driver) → v1, v1.1, v1.2 (warm-start halts) → v2.0 (joint 4-domain at r=64) → v3.0, v3, v3.1, v3.2 (the v3 phase, captured here).
- Methodology corrections discovered along the way: contemporaneous delta gate, temperature 0 default eval, base-model measurement requirement, off-the-shelf reference inclusion.

The next phase, when it begins, should not be more substrate iteration on
public TypeScript — that has reached diminishing returns. It should be
either (a) eval suite expansion, (b) Qwen3.5 substrate port, or
(c) proprietary-library training as the actual product. Each is an
independent investment and can wait until decision fatigue clears.
