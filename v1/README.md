# v1 — Scale to 5000 pairs, add Event Sourcing

Execution artifact directory for v1. Plan lives in
`docs/review/v1.md`. v0.7 decision and recipe are the baseline.

## Layout

- `seeds/` — input patterns + phrasings + reference examples for
  synthesis subagents.
- `data/` — wave-by-wave JSONL outputs from Phase C + merged
  `synth.verified.jsonl` from Phase D.
- `logs/` — per-wave survival reports, training logs, canary outputs.
- `progress.md` — resumability ledger (each completed wave is a row;
  resuming picks up from the last row).
- `verify.py` — thin wrapper over `v0.7/verify.py` with v1-specific
  default paths. ES domain is baked into v0.7/verify.py in-place.
- `train.py` — fork of v0.7/train.py with v1 defaults (r=64, 5000
  records, anchor ratio 12%).
- `decision.md` — written at Phase H (not yet present).

## Seed inventory (Phase A complete)

- `patterns.xstate.md` — 30 XState v5 patterns (from v0.7).
- `patterns.fp.md` — 15 Effect / fp-ts patterns (from v0.7).
- `patterns.rx.md` — 10 RxJS v7 patterns (from v0.7).
- `patterns.es.md` — **15 ES patterns** (new; Decider, projections,
  snapshots, sagas, upcasting, CQRS, etc.).
- `reference_examples.xstate.jsonl` — 46 extracted XState examples
  (from v0.7).
- `reference_examples.es.jsonl` — **26 extracted ES examples** from
  `repos/EventSourcing.NodeJS` (hotelManagement, snapshots,
  foundations, closingTheBooks, from_crud_to_eventsourcing). Filtered
  to examples that pass the ES verifier's idiom+length gates.
- `anchors.jsonl` — **30 unique capability anchors** (12 original from
  v0.7 + 18 new covering TS utility types, bash, git, Python logging,
  Docker health, Zod wrappers, etc.).
- `phrasings.md` — 10 prompt-phrasing templates (from v0.7).

## Gates

Phase B extended `v0.7/verify.py` in-place with ES domain (DRY: v1
verifier is a thin wrapper, not a fork). Regression-proof: `python3
v0.7/test_verify.py` runs 55 non-skipped tests, all pass. ES domain
tests cover Decider-style, Oskar `when`-style, Effect-wrapped ES,
xstate-leakage rejection, setup()-leakage rejection, and comment
stripping parity.

ES idiom gate: 18 positive tokens (see `patterns.es.md` and
`v0.7/verify.py::_ES_MUST_MATCH`). Negative guards: `from "xstate"`
import and `setup(` shape. Effect.gen is NOT blocked — Effect + ES is
a valid idiom.

## Contamination audit

`seeds/contamination_audit.py` diffs the extracted ES reference
examples against the ES eval prompts (`v0/eval/v0.json` →
`eventsourcing` domain, es-01..es-05). No seed shares ≥ 50% of any
eval prompt's distinctive proper nouns. Pre-flight check before Phase
C synthesis.

## Next: Phase C Wave D0 (ES canary)

Single subagent, 40 ES pairs, using `patterns.es.md` + the 26
reference examples. Output: `data/synth.es.batch-D0.jsonl`. Run
`v0.7/verify.py --domain es` on the output. Human spot-check 10 pairs
before launching Wave D1–D34.
