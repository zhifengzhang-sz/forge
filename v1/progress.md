# v1 Execution Progress Ledger

Cross-session resumable record. Each wave appends a row here when the
verified batch lands. Resuming from this ledger: read to find the last
completed wave, continue from the next one.

## Waves

| Wave | Domain | Status | Input | Verified | Raw / Survived | Landed |
|---|---|---|---|---|---|---|
| D0 | ES canary | ✓ DONE | `v1/data/synth.es.batch-D0.jsonl` | `v1/data/synth.es.batch-D0.verified.jsonl` | 40 / 40 | 2026-04-19 |

## Pending

- Wave A (XState depth, 19 subagents, 760 pairs target)
- Wave B (FP expand, 22 subagents, 880 pairs target)
- Wave C (RX expand, 9 subagents, 360 pairs target)
- Wave D1-D34 (ES fresh, 34 subagents, 1360 pairs target) — gated on
  D0 spot-check go/no-go
- Anchor expansion (18 new anchor prompts × 20 reps → 600 anchor
  records; author once, replicate in dataset build)

## Verifier config notes

- ES probe dir includes `effect: ^3` (added after Wave D0 exposed the
  missing dep — fix committed in `v0.7/verify.py`).
- XState / FP / RX probe dirs unchanged from v0.7.
