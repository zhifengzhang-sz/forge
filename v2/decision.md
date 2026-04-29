# v2.0 Decision — 2026-04-22

## TL;DR

**Outcome: v2.0 HALTS. The atomic-drill + fresh-base + 4-domain joint
hypothesis is partially falsified.** Same data discipline as v0.7
(atomic pattern drilling, fresh LoRA from raw bnb-4bit base), now
extended to four domains. XState and FP both regressed past the
v0.7-r64 gate. Event Sourcing landed strongly (+1.20 vs v0.7's zero
prior). Reactive held at 4.70. Capability barely moved.

The 4-domain joint-training recipe at r=64 is **capacity-limited**:
the 2604-record training set includes v0.7's 440 XState + 320 FP +
240 RX atomic-drill records unchanged, but the added 1364 ES records
and the broader 4-domain shuffle displaced XState and FP
representations. Single-variable change (adding ES) caused
single-direction regression on the two domains that share the most
surface tokens with ES (XState via setup/typed-block idioms, FP via
pipe composition and Effect.gen).

DO NOT ship `ts-forge-v2.0:latest` as daily driver.
`ts-forge-v0.7-r64:latest` remains the daily driver.

**Next arm: v2.0b — r=128 variant** (single lever change: rank only).
Plan's fallback path is explicit about this: "if RX < 4.50 OR FP
<4.00, test r=128 as v2.0b variant." Both conditions plus XState's
larger drop now implicate capacity.

## Numbers (two-grader — tiebreak not triggered)

| Arm | Domain avg | XState | FP | Reactive | ES | Cap | Tool calls |
|---|---|---|---|---|---|---|---|
| base (qwen3:14b) | 2.80 | 1.60 | 2.60 | 4.00 | 3.00 | 3.25 | 3/3 |
| **v0.7-r64 winner** | **4.10** | **4.10** | **4.40** | **4.80** | **3.10** | **4.25** | **3/3** |
| v1-r64 (fresh, 4885) | 4.25 ↑ | 4.20 ↑ | 3.80 ↓↓ | 4.53 ↓ | 4.50 ↑↑↑ | 4.00 ↓ | 3/3 |
| v1.1-r64 (warm, 4885) | 3.94 ↓↓ | 3.80 ↓ | 4.00 ↓↓ | 4.40 ↓↓ | 4.10 ↑↑ | 3.25 ↓↓ | 3/3 |
| v1.2-r64 (warm, 2426 delta) | 3.65 ↓↓↓ | 4.30 ↑ | 2.20 ↓↓↓ | 4.60 ↓ | 3.90 ↑↑ | 3.25 ↓↓ | 3/3 |
| **v2.0-r64 (fresh, 2604 atomic)** | **4.04 ↓** | **3.40 ↓↓** | **3.80 ↓↓** | **4.70 ↓** | **4.30 ↑↑↑** | **4.00 ↓** | **3/3** |
| claude-opus-4-7 | 4.95 | 5.00 | 4.80 | 5.00 | 5.00 | 5.00 | n/a |

### Per-domain grader pair

| Domain | Grader A | Grader B | Mean | v0.7 | Δ | gate |
|---|---|---|---|---|---|---|
| XState | 3.40 | 3.40 | 3.40 | 4.10 | **-0.70** | **HALT** |
| FP | 3.80 | 3.80 | 3.80 | 4.40 | **-0.60** | **HALT** |
| Reactive | 4.60 | 4.80 | 4.70 | 4.80 | -0.10 | ok |
| ES | 4.20 | 4.40 | 4.30 | 3.10 | +1.20 | n/a |
| Capability | 4.00 | 4.00 | 4.00 | 4.25 | -0.25 | n/a |

Grader agreement: **24/24 within 1 pt**, mean disagreement **0.17**.
Zero 2-point disagreements → three-grader tiebreak was not triggered.
Identical tightness to v1.2 (0.17). Graders converged independently
on the same issues (xstate-01 v4-`cond:` bleed, xstate-04 fabricated
`Subscribable`, cap-02 file-read hallucination).

## Per-prompt scores

| id | domain | A | B | mean | notes |
|---|---|---|---|---|---|
| xstate-01 | xstate | 3 | 3 | 3.0 | v4 `cond:` inside v5 setup; guard references broken |
| xstate-02 | xstate | 2 | 2 | 2.0 | guard omitted entirely despite prompt |
| xstate-03 | xstate | 5 | 5 | 5.0 | textbook v5 invoke+fromPromise |
| xstate-04 | xstate | 2 | 2 | 2.0 | fabricated `Subscribable` import from xstate |
| xstate-05 | xstate | 5 | 5 | 5.0 | clean v4→v5 conversion |
| fp-01 | fp | 3 | 3 | 3.0 | chains raw fns over Either; never uses `chain`; type-incorrect |
| fp-02 | fp | 3 | 3 | 3.0 | fetchUser defined but never called; no fallback on UserNotFound |
| fp-03 | fp | 4 | 4 | 4.0 | correct Effect.tryPromise + catchTag |
| fp-04 | fp | 5 | 5 | 5.0 | idiomatic pipe + TE.chainW |
| fp-05 | fp | 4 | 4 | 4.0 | solid Context.GenericTag + Layer |
| rx-01 | reactive | 5 | 5 | 5.0 | pipe + mergeMap clean |
| rx-02 | reactive | 5 | 5 | 5.0 | switchMap + debounceTime idiomatic |
| rx-03 | reactive | 5 | 5 | 5.0 | combineLatest clean |
| rx-04 | reactive | 4 | 4 | 4.0 | functional but lacks contrasting Subject demo |
| rx-05 | reactive | 4 | 5 | 4.5 | takeUntil pattern correct |
| es-01 | es | 5 | 5 | 5.0 | best ES answer; clean Decider |
| es-02 | es | 4 | 4 | 4.0 | solid evolve/decide |
| es-03 | es | 4 | 5 | 4.5 | optimistic concurrency expressed well |
| es-04 | es | 4 | 4 | 4.0 | discriminated unions correct |
| es-05 | es | 4 | 4 | 4.0 | concurrent write handling ok |
| cap-01 | capability | 4 | 5 | 4.5 | correct fib; closing fence typo (cosmetic) |
| cap-02 | capability | 2 | 1 | 1.5 | **hallucinated package.json content without tool use** |
| cap-03 | capability | 5 | 5 | 5.0 | clean refactor |
| cap-04 | capability | 5 | 5 | 5.0 | top_by_score correct |

## Phase G regression gate output (copied verbatim)

```
=== v2.0-r64 ===
domain               v2.0   v0.7    delta   gate
--------------------------------------------------
xstate               3.40   4.10    -0.70   halt
fp                   3.80   4.40    -0.60   halt
reactive             4.70   4.80    -0.10     ok
eventsourcing        4.30   3.10    +1.20    n/a
capability           4.00   4.25    -0.25    n/a

capability breakdown:
  Cap_old12: (no records)
  Cap_new18: 4.00

grader agreement (within 1 pt): 24/24
overall mean disagreement: 0.17

=== Regression gate ===
HALT — trained domain regression detected:
  - xstate: v2.0 3.40 vs v0.7 4.10 (delta -0.70, tolerance -0.30) — REGRESSION
  - fp: v2.0 3.80 vs v0.7 4.40 (delta -0.60, tolerance -0.30) — REGRESSION
```

## v2.0's hypothesis vs actual

The plan (docs/v2.plan.md §Predicted outcomes) predicted:

| Domain | Predicted | Actual | Miss |
|---|---|---|---|
| XState | 4.10 ± 0.2 | 3.40 | **-0.70** |
| FP | 4.30 – 4.50 | 3.80 | **-0.50 to -0.70** |
| Reactive | 4.60 – 4.80 | 4.70 | hit |
| Event Sourcing | 4.30 – 4.60 | 4.30 | hit (bottom of range) |
| Capability | 4.10 – 4.30 | 4.00 | -0.10 |
| Overall | 4.15 – 4.45 | 4.04 | -0.11 (below floor) |

RX and ES hit the predicted ranges. XState and FP both missed below
the ranges by ~0.5-0.7. Not noise — the miss is structural and one-
directional, exactly on the two domains the plan flagged as
syntactically overlapping with ES (§Principles 2, echoing
`lessons.learned.md` §"Cross-domain interference correlates with
syntactic surface").

## Structural finding — atomic data alone doesn't fix capacity

v1.1/v1.2 established that warm-start does not isolate capabilities
because LoRA is global. v2.0 establishes the complementary finding:
**atomic-drill data quality is necessary but not sufficient when
r=64 has to serve 4 domains.** v0.7's recipe worked on 3 domains
with 1000 records because r=64 had enough capacity per domain.
Adding ES (1364 records, largest single-domain contribution) at the
same r=64 pushed XState and FP below their v0.7 levels despite
feeding identical atomic-drill records for those domains.

Two hypotheses for v2.0b to discriminate:

1. **Capacity starvation** — r=64 can't encode 4 domains worth of
   structure; more rank solves it. Test: v2.0b at r=128.
2. **ES-induced interference** — ES's Decider pattern
   (discriminated-union events, generator-less style) has specific
   syntactic collisions with Effect's `Effect.gen`/`yield*` pattern
   that no rank increase fixes. Test: would appear as XState+FP
   regression persisting even at r=128.

The v0.7-r64 baseline at 3 domains was 4.10 avg; v2.0-r64 at 4
domains is 4.04 avg. Ratio argument: dividing r=64 across 4 domains
vs 3 should preferentially hurt the domains with the most structural
variety (XState's typed-block setup patterns, FP's Effect generator
control flow). That's exactly what we see.

## What went right

1. **Event Sourcing scored 4.30** — 1364 atomic-drill records
   produced clean Decider answers across all 5 ES prompts. es-01
   scored 5/5 from both graders; es-02 through es-05 scored 4-4.5.
   This is the largest positive delta of any arm (+1.20 vs v0.7).
   The atomic-drill synthesis discipline (37 aggregates × 40
   variations) translated directly to eval strength.
2. **Reactive held at 4.70** (v0.7 4.80, delta -0.10). The four
   rx-01 through rx-04 prompts all scored 4-5. No library-confusion
   errors — RxJS's surface is distant enough from Effect/XState that
   it coexisted cleanly in the 4-domain mix, consistent with the
   v1.1/v1.2 finding that RxJS shares few surface tokens with the
   other domains.
3. **Tool-calling preserved: 3/3 PASS** (list_files, read_file ×2).
   Consistent with every prior arm.
4. **Grader agreement tightest-ever: 0.17 mean disagreement**, zero
   2-point splits. Two-grader protocol is robust; both graders
   independently converged on the same error diagnoses (v4 cond bleed,
   Subscribable fabrication, cap-02 hallucination).
5. **Training dynamics clean.** 978 steps in 42:23 wall (2.60 s/it),
   train_loss converged from 1.64 → 0.34 over 3 epochs. No OOM, no
   NaN. Same hardware and recipe as v1.1/v1.2.

## What went wrong

1. **XState regressed -0.70** despite using the identical 440 v0.6
   XState records that v0.7 trained on. xstate-01 produced v4-style
   `cond:` guard syntax inside a v5 `setup({...})` block (both
   graders flagged); xstate-04 fabricated a `Subscribable` type
   import from `xstate` that doesn't exist. Neither failure mode
   appeared in v0.7's outputs. The only training difference vs v0.7
   is the addition of ES (1364 records) and the shift to joint 4-way
   shuffle.
2. **FP regressed -0.60** despite using the identical 320 v0.7 FP
   atomic-drill records. fp-01 and fp-02 scored 3/3 both graders
   where v0.7 had 4/5. fp-02 defined but never called `fetchUser`
   and omitted the required Option fallback — a composition failure
   the atomic-drill records should have taught. ES's generator-style
   control flow likely displaced the attention capacity for
   Effect.gen's specifically-ordered `yield*` patterns.
3. **Capability dropped -0.25**. cap-02 (file-read task) hallucinated
   package.json content instead of emitting a tool-call — both
   graders scored 1-2. Anchor count (30 × 8 = 240, 9.2% ratio) was
   within the recommended 8-10% range per v1/decision.md, but the
   18 new anchors' specific task-following pattern didn't register
   strongly enough against the 4-domain training mix.
4. **Three training attempts before success.** Attempt 1 crashed at
   tokenizer load (`[Errno 99] Cannot assign requested address`, HF
   HEAD retry exhaustion); root cause: missing `HF_HUB_OFFLINE=1`
   despite the model being in the HF cache from v1/v1.1/v1.2.
   Attempt 2 trained 42 min successfully but silently lost the LoRA
   because Unsloth's `save_pretrained_gguf` doesn't write shards
   when the base is bnb-4bit (v1.1/v1.2 loaded already-merged
   safetensors and got a copy-from-local-model side-effect that
   masked the bug). Attempt 3 trained 42 min, saved the LoRA
   explicitly, then `save_pretrained_merged` silently skipped the
   merged save for a different reason — `determine_base_model_source`
   returned None under `HF_HUB_OFFLINE=1`. Recovery: manual
   `PeftModel.from_pretrained(...).merge_and_unload()` from the saved
   adapter, then resume the manual GGUF pipeline. Total wasted wall:
   ~100 minutes of training compute. Plan has been updated with
   these fixes (see §"Plan corrections committed" below).

## Plan corrections committed

`docs/v2.plan.md` was updated during this arm to close the loops
that caused attempts 1-2 and very nearly caused 3:

- §"v2.0 training config" now requires replacing v1's broken
  `save_pretrained_gguf` call with explicit `save_pretrained`
  (adapter) + `save_pretrained_merged` (merged 16-bit). Also
  mandates `save_strategy="epoch"` so an adapter checkpoint lands
  even on save-block crash.
- §"v2.0 training config" now specifies the `HF_HUB_OFFLINE=1
  TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments
  :True` launch env vars.
- §"GGUF conversion" now states that `v2/gguf/` must already
  contain merged 16-bit safetensors before manual conversion, and
  explains why bnb-4bit runs don't get the v1.1/v1.2
  copy-from-local-model side-effect.
- New §"Recovery from partial save failure" with concrete recovery
  paths for both adapter-only and no-save cases.
- §"Acceptance criteria" extended with two new items: HF cache
  presence check and the offline env-var launch requirement.
- §"What's already built vs what's new" flags the template debt in
  `v1/train.py` (save_pretrained_gguf + save_strategy="no") as traps
  forks must fix.

These changes should make v2.0b and any future arm avoid the same
three-attempt cascade.

## Decision matrix outcome

Applying the plan's §Failure modes table directly:

| Observation | Interpretation | Next move |
|---|---|---|
| FP < 4.00 (got 3.80) | "Unexpected — something in the simpler recipe broke. Diff v2.0 data against v0.7+v1.ES. Likely a dedup mishap or a stale batch." | Data diff done (§Data verification below); nothing broken. Promote to v2.0b path. |
| RX held at 4.70 > 4.50 | not triggered | — |
| ES held at 4.30 > 4.00 | not triggered | — |
| Cap held at 4.00 > 4.00 | at edge; not triggered | — |
| XState drop (not in table) | novel — table assumed XState would hold | Add as v2.0b signal: two syntactically-overlapping domains (XState, FP) both regressed, pointing to capacity/interference. |

FP specifically was < 4.00 — the plan said "diff v2.0 data against
v0.7+v1.ES, likely dedup mishap or stale batch." That diff is easy:
v2/data was built from v0.7/data/synth.fp.batch-{A..H}.jsonl
(320 records, unchanged) + v0.7/data/synth.rx.batch-*.jsonl (240,
unchanged) + v1/data/synth.es.batch-D*.verified.jsonl (1364,
unchanged) + v0.6/data/synth.verified.jsonl (440 XState, unchanged)
+ 30 anchors × 8 reps. Hash dedup ran (post-dedup identical to pre
at 2364 pre-anchor records). No stale batch. The FP drop is NOT a
data mishap — the training data is byte-identical to what produced
v0.7's FP=4.40. The cause has to be in the training recipe, which
leaves capacity (r=64 across 4 domains) or interference (ES pattern
bleed into FP inference) as the remaining explanations.

## Recommended next steps

Priority order:

1. **Do NOT dogfood ts-forge-v2.0 as daily driver.** Keep
   `ts-forge-v0.7-r64:latest` as the recommended local model. v2.0
   is retained as a diagnostic checkpoint — it shows that 4-domain
   atomic-drill alone doesn't clear the gate at r=64.

2. **v2.0b — r=128 variant.** Same 2604-record atomic-drill dataset,
   only `--rank 128`. This is the plan's explicit fallback and is
   now the critical single-variable test. **Prediction (bounded
   from this arm's evidence):**
   - If v2.0b clears the gate on XState AND FP → capacity was the
     root cause; ship v2.0b and move to v2.1 (add XState extension).
   - If v2.0b clears FP but not XState → domain-specific
     interference; investigate XState synthesis vs ES syntax overlap.
   - If v2.0b regresses similarly → 4-domain joint at this dataset
     composition is fundamentally harder than 3-domain; need data
     rebalancing (ES weight reduction) or architecture change (LoRA
     stacking).

   Cost: ~60-90 min training (v1 r=32→r=64 jump was ~2× wall; r=64
   →r=128 should be ~1.5-1.8×) + 3 min GGUF + 1 min eval + 3 min
   graders = ~70-95 min end-to-end.

3. **Investigate xstate-01 and xstate-04 training data.** Both
   failures are novel vs v0.7 — xstate-01's `cond:` syntax mismatch
   and xstate-04's fabricated `Subscribable` type didn't appear in
   v0.7 outputs. Grep v0.6/data/synth.verified.jsonl for `cond:` and
   `Subscribable` to see whether v0.6 records contain those strings
   (baseline leakage would make the symptom appear when attention
   weights shift, even without adding new records). If they're
   absent from v0.6 training data, the regression is pure
   interference from the non-XState records in the v2 mix.

4. **Investigate cap-02's tool-call omission.** The model DID call
   tools for tc-01/02/03 (tool-call smoke 3/3 PASS). cap-02 failed
   because the prompt style in v0/eval/v0.json differs from the
   explicit function-definition format used in tool_call_smoke.py.
   v0's anchor set should include at least one file-read-via-tool
   example matching v0.json's cap-02 phrasing. 18 new anchors
   (v1 additions) evidently didn't cover this shape.

5. **Data contribution analysis (optional before v2.0b).** Run
   ablation on v2's 2604 records to quantify per-domain loss
   contribution. Not required to unblock v2.0b, but would inform
   v2.1-v2.3 rebalancing decisions.

6. **Two-grader protocol is well-calibrated.** 0.17 mean
   disagreement, zero 2-point splits — same tightness as v1.2. No
   need to add third grader by default for v2.0b.

7. **Preserve artifacts.** `v2/gguf/adapter/adapter_model.safetensors`
   (991 MB LoRA) and `v2/gguf/model-*.safetensors` (29 GB merged
   16-bit) and `v2/gguf_gguf/qwen3-14b.Q4_K_M.gguf` (8.6 GB GGUF)
   are retained. If v2.0b's result suggests r=64 is actually
   sufficient but a specific data rebalance is needed, we can
   reuse the v2/gguf/adapter as a diagnostic point.

## Gap to Claude Opus 4.7

| Domain | v1.2 | v2.0 | Opus | v2.0 gap | v2.0 as % of Opus |
|---|---|---|---|---|---|
| XState | 4.30 | 3.40 | 5.00 | -1.60 | 32% |
| FP | 2.20 | 3.80 | 4.80 | -1.00 | 21% |
| Reactive | 4.60 | 4.70 | 5.00 | -0.30 | 6% |
| Event Sourcing | 3.90 | 4.30 | 5.00 | -0.70 | 14% |
| Capability | 3.25 | 4.00 | 5.00 | -1.00 | 20% |

v2.0 closed v1.2's FP catastrophe (2.20 → 3.80, +1.60) but opened
XState gap (4.30 → 3.40) and RX is now within 0.30 of Opus. ES is
the strongest domain vs Opus (-0.70) — the atomic-drill recipe
produces near-frontier quality when the domain has enough training
density.

## Cost of v2.0

| Step | Time |
|---|---|
| Fork v1.2/merge.py → v2/merge.py (drop v1 FP/RX/XState; add v0.6 XState, v0.7 FP/RX; anchor reps 20→8) | 5 min |
| Fork v1/train.py → v2/train.py (+ fix save block; see §"What went wrong" #4) | 5 min |
| Merge to build v2/data/synth.verified.jsonl (2604 records) | 2 sec |
| Training attempts 1+2+3 (2 failures + success) | ~110 min |
| Manual merge recovery (v2/merge_adapter.py — load base, attach LoRA, merge_and_unload, save safetensors) | 6 min |
| Manual GGUF pipeline (convert_hf_to_gguf F16 + llama-quantize Q4_K_M + Modelfile copy) | 2 min |
| ollama create ts-forge-v2.0 | 20 sec |
| Eval (24 prompts via run_eval.py) | 1 min |
| Tool-call smoke (3 prompts) | 15 sec |
| Two-grader dispatch (parallel subagents) | ~1.5 min |
| Run combine_v2.0.py | 1 sec |
| This decision.md + plan corrections | 25 min |
| **Total** | **~2.8 h wall** |

Plan estimate was "~45 min training + ~60 min end-to-end = ~1.75 h."
Actual ~2.8 h, with the overrun driven entirely by the three-attempt
training cascade (+~65 min) and the manual recovery flow (+~6 min).
The corrected plan should bring future arms back to the 1.5-1.75 h
budget.

## What v2.0 is NOT

- **Not a successful 4-domain scale-up.** XState and FP both
  regressed below the v0.7 baseline. The recipe that worked on 3
  domains does not straightforwardly extend to 4 at r=64.
- **Not a data-quality problem.** v0.6 XState, v0.7 FP, v0.7 RX, v1
  ES all passed their verifiers and all produced the expected scores
  in prior arms where they shipped alone. The training set contains
  the same records that produced v0.7's 4.10 average across 3
  domains. The failure is in the training recipe (capacity /
  interference), not the upstream synthesis.
- **Not a warm-start test.** v2.0 explicitly reverted to fresh-from-
  base after v1.1/v1.2 falsified the warm-start hypothesis.
- **Not ruled out.** v2.0b at r=128 is the direct next test. If
  r=128 clears the gate, v2.0's hypothesis is rescuable via capacity
  scaling, and the full plan (v2.1-v2.3 adding XState/RX/FP breadth)
  remains on track.

## Artifacts produced

- `v2/merge.py` — fork of v1.2/merge.py; drops v1 FP/RX/XState, adds
  v0.6/v0.7 atomic sources, ANCHOR_REPS=8
- `v2/train.py` — fork of v1/train.py with fresh-from-base MODEL +
  explicit save block (adapter + merged_16bit) replacing the broken
  `save_pretrained_gguf`
- `v2/merge_adapter.py` — recovery script that loads unquantized
  base + saved adapter, merges via PEFT, and writes merged
  safetensors to v2/gguf/
- `v2/data/synth.verified.jsonl` — 2604 records (440 XState + 320
  FP + 240 RX + 1364 ES + 240 anchors)
- `v2/gguf/adapter/` — 991 MB LoRA adapter (preserved for recovery)
- `v2/gguf/model-*.safetensors` — 6-shard merged 16-bit safetensors
  (~29 GB, retained)
- `v2/gguf_gguf/qwen3-14b.Q4_K_M.gguf` — 8.6 GB quantized GGUF
- `v2/gguf_gguf/Modelfile` — same template as v1/v1.1/v1.2
- `v2/logs/train.log` — attempt-3 trainer output (attempts 1-2
  preserved as `.attempt1`/`.attempt2`)
- `v2/logs/merge.log` — adapter merge recovery output
- `v2/logs/gguf_convert.log` — F16 GGUF conversion output
- `v2/logs/quantize.log` — Q4_K_M quantize output
- `v2/logs/run_eval.log` + `toolcall.log` — eval + smoke logs
- `v0/grading/combine_v2.0.py` — combine script fork
- `v0/results/v2.0.raw.json` — 24 model responses
- `v0/results/v2.0.toolcall.json` — tool-call smoke results (3/3 PASS)
- `v0/results/v2.0.graded.grader_A.json` + `grader_B.json` —
  per-prompt scores + rationales, independent two-grader protocol
- `v0/results/v2.0.json` — combined (A+B mean, disagreement, gate)
- `ollama` model `ts-forge-v2.0:latest` (9.0 GB)
- `docs/v2.plan.md` — patched with save-block requirements,
  offline env vars, recovery paths, HF cache pre-flight

## Session-state trail

commits → `v1/decision.md` (v1 halt, warm-start proposed) →
`v1.1/decision.md` (warm-start full data halts) →
`v1.2/decision.md` (warm-start delta-only catastrophically halts;
FP root cause identified) → `docs/lessons.learned.md` updated →
`docs/v2.plan.md` v2.0 proposal → **`v2/decision.md` (this file —
v2.0 halts on XState+FP, redirect to v2.0b r=128)** → v2.0b
execution.
