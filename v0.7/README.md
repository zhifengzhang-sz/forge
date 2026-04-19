# v0.7 — Multi-Domain Fine-Tuning

Multi-domain SFT run that extends v0.6's XState-only recipe to fp-ts/Effect-TS
and RxJS, targeting the v0.6 XState-leakage regression (fp-02, fp-05, rx-02
failures where the model produced `setup({...})` inside non-XState answers).

Authoritative plan: [`../docs/review/v0.7.md`](../docs/review/v0.7.md).
v0.6 precedent: [`../v0.6/README.md`](../v0.6/README.md).

## Layout

```
v0.7/
├── README.md                # this file
├── seeds/                   # synthesis prompt inputs
│   ├── patterns.xstate.md   # 30 XState v5 patterns (copy of v0.6)
│   ├── patterns.fp.md       # 15 fp-ts / Effect-TS v3 patterns (new)
│   ├── patterns.rx.md       # 10 RxJS v7+ patterns (new)
│   ├── phrasings.md         # 10 library-agnostic phrasing templates
│   ├── reference_examples.xstate.jsonl   # verbatim from v0.6
│   └── anchors.jsonl        # 12 capability anchors (8 v0 + 4 new)
├── data/                    # synthesized + verified JSONL (populated later)
└── gguf/                    # exported GGUF outputs (populated later)
```

## Locked-in decisions (from plan-eng-review 2026-04-19)

- **Domain ratio 32/36/29** — XState/FP/RX. Reuse 440 v0.6 XState pairs verbatim (no new XState synth). Synthesize ~500 FP + ~400 RX new.
- **Both r=32 AND r=64 trained sequentially.** r=32 first → GGUF → eval → grade, then r=64 same path. Pick winner by combined (XState + FP + Reactive) domain average.
- **5% capability anchor mix-in** (dropped from 10% per issue #4) with **12 unique anchors × ~6 reps = ~70 records**. 8 anchors reused from `v0/data/xstate_curated.jsonl` indices 35–42; 4 new (TypeScript generics, async/await, Python dataclass, SQL review).
- **Hard-isolated two-grader evaluation.** Grader A runs, writes `v0/results/v0.7.graded.grader_A.json`. Grader B is dispatched with explicit instruction to NOT read any `v0.7.graded.*` file. Combination step is a separate non-subagent script.
- **30s sleep** between `save_pretrained_gguf` return and `ollama create` to let GPU power settle — mitigates the v0.6 mid-eval crash pattern.
- **Per-domain verifier gates** — XState / FP / RX each get independent tsc probe dirs and distinct MUST-MATCH / MUST-NOT-MATCH idiom regexes. `setup\(` is in MUST-NOT-MATCH for FP and RX as a shape guard.
- **Borderline band ±0.3** around any decision threshold triggers a rerun under the alternate rank (r=32 ↔ r=64).

## What v0.7 must NOT do

- Do not modify `v0/eval/v0.json` (apples-to-apples comparator with v0 and v0.6).
- Do not put eval-prompt text into training data (same contamination rule as v0.6).
- Do not train without confirming `nvidia-pl.service` is active and power cap is 500 W.
- Do not grade with a subagent that also dispatched synthesis.

See `docs/review/v0.7.md` for the full plan and decision matrix.
