# Wave D0 — ES Canary Survival Report

Date: 2026-04-19
Wave: D0 (ES canary, single subagent, 40 pairs)
Input: `v1/data/synth.es.batch-D0.jsonl` (84 KB, 40 records)
Output: `v1/data/synth.es.batch-D0.verified.jsonl`

## Headline

**40/40 verifier survival (100%).**

| Gate | Result |
|---|---|
| Length (150-6000 chars) | 40/40 pass |
| Idiom MUST-MATCH (18 positive tokens) | 40/40 pass |
| Idiom MUST-NOT-MATCH (xstate import, setup leak) | 40/40 pass |
| tsc --strict --noEmit | 40/40 pass |
| tsc wall time | 0.79s (batched) |

## Coverage per wave target

| Target | Planned | Actual |
|---|---|---|
| Decider / evolve+decide | 6 | ~7 |
| Projection / read model | 5 | ~5 |
| Snapshot + version upgrade | 3 | ~3 |
| Optimistic concurrency (expectedRevision) | 3 | ~3 |
| CommandHandler | 4 | ~4 |
| Subscription from checkpoint | 3 | ~3 |
| aggregateStream + when reducer | 4 | ~4 |
| Upcasting | 2 | ~2 |
| Read model rebuild | 2 | ~2 |
| CQRS boundary | 2 | ~2 |
| Process manager / saga | 2 | ~2 |
| Integration event / outbox | 2 | ~2 |
| Misc (DUs, guards) | 2 | ~1 |
| **Total** | **40** | **40** |

| Phrasing template | Planned | Actual (qualitative) |
|---|---|---|
| Build-from-scratch | ~12 | close to target |
| How-to question | ~8 | close to target |
| Explain-then-implement | ~6 | close to target |
| Constraint-driven | ~6 | close to target |
| Capability add-on | ~4 | close to target |
| Refactor-for-idiom | ~2 | close to target |
| Targeted fix | ~2 | close to target |

## Verifier calibration finding

First run failed 5/40 on `TS2307: Cannot find module 'effect'`. The ES
idiom gate explicitly permits Effect-wrapped event store access, but
the probe dir's `package.json` didn't declare `effect` as a dependency.
Fix: `effect: ^3` added to `_package_json_for("es")` in
`v0.7/verify.py`. After reinstall, 40/40 passed.

This is a real design consistency issue, not a subagent quality
issue — the 5 Effect-wrapped records were idiomatically correct ES.

## Eval contamination cross-check

Subagent reports no `ShoppingCart`, `BankAccount`, `CartSummary`,
`CartCreated`, `ItemAdded`, `ItemRemoved`, `Deposit`, or `Withdraw`
appearances. Spot-check of first 10 records confirms: aggregate names
in use include Invoice, Reservation, Subscription, OrderFulfilment,
GuestStay, CashRegister, LoyaltyAccount — all outside the eval surface.

## Subagent self-reported uncertainties

1. A few Decider records use `"id" in state ? state.id : ""` to narrow
   unions — verbose but compiles under `strict`.
2. Upcast + project records (30, 31) rely on a single `assertRaw`
   helper to narrow `unknown` inputs — stylistically unusual but
   correct.
3. Record 21 (generic CommandHandler factory) has an inline mid-file
   `import { Effect } from "effect"` — legal TS, unusual placement.

All three compile clean. None are bugs. Flagged for human spot-check.

## Go / no-go recommendation

**GO.** 100% verifier survival on a fresh, unseen domain with 40 pairs
produced in ~21 min wall time is a strong signal. The recipe
generalizes to ES. Launch Wave D1-D34 (34 subagents, ~1360 more ES
pairs) in parallel waves of 7.

Risks for the bigger waves:
- Duplicate-aggregate-name leakage across subagents (each sees the
  same seeds, may converge on Invoice / Reservation). Mitigation:
  pass each subagent a unique numeric offset + theme hint.
- Effect probe install per new tmp dir — tolerable one-time cost.
