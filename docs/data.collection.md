# Data Collection Guide

Lessons from running the extraction pipeline on TypeScript source repositories.

## First Run Results (2026-04-16)

Shallow clone, no git diffs, quality threshold 0.30.

| Stage | typescript.fp | typescript.reactive | typescript.xstate | typescript.eventsourcing | Total |
|---|---|---|---|---|---|
| Files matched | 944 | 462 | 131 | 348 | 1,885 |
| Raw units extracted | 10,066 | 507 | 496 | 1,288 | 12,357 |
| After quality scoring | 4,394 | 275 | 198 | 182 | 5,049 |
| After dedup | 4,360 | 274 | 189 | 118 | 4,941 |
| After balance (cap=500) | 500 | 274 | 189 | 118 | 1,081 |

Source repos: fp-ts, Effect, RxJS, XState, EventSourcing.NodeJS.

## Focus Terms: Match Definitions, Not Just Call Sites

The pipeline extracts exported declarations (function definitions, type aliases, interfaces). Focus terms designed from a "calling code" perspective will miss most of them.

**Bad:** `setup(` only matches function calls like `const machine = setup({...})`.
**Good:** `setup(`, `setup<`, `StateMachine`, `MachineContext` matches both calls and `export function setup<TContext extends MachineContext>`.

This one change took XState from 29 qualifying units to 189.

When writing focus terms for a new topic:
- Include call-site patterns: `fn(`, `fn<`
- Include type/class names: `TypeName`, `ClassName`
- Include common re-exports and composition patterns: `fn,` (as in `import { fn, other }`)
- Run `--skip-instruct` first to verify yield before spending API credits

## Balance Formula

The balance step caps each domain to prevent one large repo from dominating. The formula is:

```
cap = min(max(2 * median(domain_sizes), 100), 500)
```

Using median instead of min is important. The original formula `2 * min(domain_sizes)` let XState's 29 units cap everything at 58. One small domain crushed the entire dataset.

Domains smaller than the cap contribute all their units. This is fine; the model still learns from them, they're just represented proportionally less.

## Quality Scoring Pitfalls

**Regex patterns must be valid.** `r":\s*\w+(<"` has an unclosed capture group. Python 3.13+ rejects this at runtime. Always test scoring patterns before running the full pipeline.

**Short unit filtering happens twice.** The extractor skips units under 50 chars. The scorer skips units under 80 chars (`min_unit_length`). The second filter is authoritative; the first just avoids scoring trivially small re-exports.

**`any` penalty is weighted higher (0.10 vs 0.05).** Using `any` in TypeScript is a stronger quality signal than a stray `console.log` or `TODO`. The penalty reflects this.

## Held-Out Eval Set

The pipeline warns but does not block if the held-out set is missing. This allows you to run extraction and inspect results before writing eval examples. The held-out set must exist before training to prevent data contamination.

The held-out set contains hand-written test cases representing target model behavior. It is not derived from the extracted data. SHA-256 fingerprint matching during the dedup step excludes any training unit that happens to match a held-out example.

## Increasing Yield

If the dataset is too small:

1. **Add `--full-history`** — clones full git history and extracts TypeScript diffs. Adds 10-30% more units per domain, but cloning takes longer and uses more disk.
2. **Add more source repos** — create additional `RepoConfig` entries in the topic config.
3. **Lower quality threshold** — change `QUALITY_THRESHOLD` in `lib/typescript/score.py` from 0.30 to 0.25. This adds marginal units that scored just below the cutoff. Inspect samples before deciding.
4. **Broaden focus terms** — add more type names, function names, and patterns from the target library's API surface.

## Cost Estimation

Instruction generation (Phase 2) calls the Claude API once per unit. At ~1,000 units with average context of ~500 tokens:

- Estimated input: ~500K tokens
- Estimated output: ~50K tokens
- Approximate cost: $2-5 depending on model choice

The pipeline prints the estimate and asks for confirmation before making API calls. Use `--dry-run` to see the estimate without running anything.
