# v1 Execution Progress Ledger

Cross-session resumable record. Each wave appends a row here when the
verified batch lands. Resuming from this ledger: read to find the last
completed wave, continue from the next one.

## Waves

| Wave | Domain | Status | Raw / Survived | % | Landed |
|---|---|---|---|---|---|
| D0 | ES canary | ✓ | 40 / 40 | 100% | 2026-04-19 |
| D1 | ES | ✓ | 40 / 35 | 88% | 2026-04-19 |
| D2 | ES | ✓ | 40 / 38 | 95% | 2026-04-19 |
| D3 | ES | ✓ | 40 / 29 | 73% | 2026-04-19 |
| D4 | ES | ✓ | 40 / 32 | 80% | 2026-04-19 |
| D5 | ES | ✓ | 40 / 34 | 85% | 2026-04-19 |
| D6 | ES | ✓ | 40 / 40 | 100% | 2026-04-19 |
| D7 | ES | ✓ | 40 / 40 | 100% | 2026-04-19 |
| D8 | ES | ✓ | 40 / 33 | 82% | 2026-04-19 |
| D9 | ES | ✓ | 40 / 36 | 90% | 2026-04-19 |
| D10 | ES | ✓ | 40 / 38 | 95% | 2026-04-19 |
| D11 | ES | ✓ | 40 / 40 | 100% | 2026-04-19 |
| D12 | ES | ✓ | 40 / 39 | 97% | 2026-04-19 |
| D13 | ES | ✓ | 40 / 38 | 95% | 2026-04-19 |
| D14 | ES | ✓ | 40 / 40 | 100% | 2026-04-19 |
| D15 | ES | ✓ | 40 / 35 | 88% | 2026-04-19 |
| D16 | ES | ✓ | 42 / 38 | 90% | 2026-04-19 |
| D17 | ES | ✓ | 41 / 35 | 85% | 2026-04-19 |
| D18 | ES | ✓ | 40 / 34 | 85% | 2026-04-19 |
| D19 | ES | ✓ | 45 / 40 | 89% | 2026-04-19 |
| D20 | ES | ✓ | 40 / 40 | 100% | 2026-04-20 |
| D21 | ES | ✓ | 40 / 33 | 83% | 2026-04-20 |
| D22 | ES | ✓ | 40 / 40 | 100% | 2026-04-20 |
| D23 | ES | ✓ | 40 / 35 | 88% | 2026-04-20 |
| D24 | ES | ✓ | 40 / 31 | 78% | 2026-04-20 |
| D25 | ES | ✓ | 40 / 37 | 93% | 2026-04-20 |
| D26 | ES | ✓ | 40 / 34 | 85% | 2026-04-20 |
| D27 | ES | ✓ | 44 / 36 | 82% | 2026-04-20 |

**Cumulative ES:** 1132 raw → 1020 verified (90.1% overall). D20-D27 batch 88.3%. Target: 1400. Remaining: 380 (~11 more subagents).

Note: several batches bounced on "out of extra usage" mid-run but wrote their files BEFORE exiting — verified output was usable despite the error message.

## Rejection lessons (for prompt tightening on D8+)

1. **prose inside `typescript` fence** — D3 especially. "Explain-then-
   implement" phrasing caused prose to land inside the fence, which
   tsc then sees as TS1434. Fix: prompt now says "prose goes BEFORE
   the fence; only compile-clean TS inside."
2. **missing_es_positive** — D1/D2/D3 have records that define `when`
   as an arrow variable and pass it to `reduce(when, {})` — regex
   only matches `when(` call sites. Fix: prompt requires at least one
   direct call-form idiom token in the code body.
3. **length:short** — D4 had 4 records under 150 chars (one at 147).
   Fix: prompt requires ≥200 char code body for margin.
4. **tsc DU `never` narrowing** — D5 records exhausted the DU via `if`
   early and then the fallthrough was typed `never`. Fix: prompt adds
   a test "if the fallthrough branch is reachable, don't assign to
   `never` there."
5. **tsc top-level `for await`** — D5. Fix: prompt says wrap in
   `async function main() { ... }` or inside an exported async.

## Pending

- Wave D8-D14 (ES fresh, 7 subagents, ~280 pairs with ~90% survival → ~250 verified)
- Wave D15-D21
- Wave D22-D28
- Wave D29-D34 + possibly D35-D37 if cumulative ES falls short of 1400
- Wave A (XState depth, 19 subagents, 760 pairs target)
- Wave B (FP expand, 22 subagents, 880 pairs target)
- Wave C (RX expand, 9 subagents, 360 pairs target)
- Anchor expansion (600 anchor records = 30 unique × 20 reps, mixed
  in at train time)

## Verifier config notes

- ES probe dir includes `effect: ^3` (added after Wave D0 exposed the
  missing dep — fix committed in `v0.7/verify.py`).
- XState / FP / RX probe dirs unchanged from v0.7.
