# Canonical grader-subagent prompt template

Use this verbatim when dispatching grader subagents, substituting the
four `{PLACEHOLDER}` fields. Keeps per-arm grading consistent and
blind across sessions.

---

You are **Grader {LETTER}** for the {ARM_LABEL} eval of the forge fine-tuning
project. This is a blind two-grader protocol — do NOT read any other grader's
output, other arm decision docs, or prior grader results. Base your judgment
only on the prompt, the model's raw response, and your own expertise.

**Working directory**: /home/zzhang/dev/ai/models/forge

**Inputs**:
- `tools/eval/suite.json` — 24 prompts (5 xstate + 5 fp + 5 rx + 5 es + 4 cap).
- `{RAW_PATH}` — the model's 24 responses (keyed by prompt id).

**Context:** {CONTEXT_NOTE}

**Scoring scale** (1-5):
- 5 = idiomatic and correct
- 4 = good
- 3 = acceptable
- 2 = bad
- 1 = unusable

Grade each of the 24 prompts with an integer 1-5 score and a 1-3 sentence
rationale citing specific features (API usage, correctness, idiom, missing
patterns, wrong library, fabrication). `must_have` / `must_not_have` in
suite.json are hints, not determinative.

**Domain anchors**:
- **xstate**: XState v5 idioms. `setup({types, actors, guards, actions})` correct.
  v4 patterns (`services:`, `cond:`, positional `(ctx, event) =>`, bare
  `createMachine`) score low. `fromPromise`/`fromObservable` expected with
  input mappers.
- **fp** (Effect-TS or fp-ts): `Effect.gen(function*()…)` with `yield*`,
  `Effect.tryPromise`, `Effect.catchTag`, `Context.GenericTag`, `Layer`, pipe.
  Red flags: `throw` inside `Effect.gen`, `as never`, wrong libraries, fabricated
  imports.
- **reactive** (RxJS): pipeable operators (`pipe`, `mergeMap`, `switchMap`,
  `combineLatest`, `takeUntil`). Red flags: deprecated static methods, nested
  subscribe, incorrect pipe.
- **eventsourcing**: Decider pattern (`decide(cmd,state)->event[]`,
  `evolve(state,event)->state`), immutable state, discriminated-union events,
  exhaustive checks.
- **capability** (cap-*): did the model do the task cleanly. Fabrication
  (invented imports, hallucinated file contents, skipping tool calls) scores low.

**Output format** — write EXACTLY this JSON to `{OUT_PATH}`:
```json
{
  "model": "{MODEL_ID}",
  "arm": "{ARM_LABEL}",
  "grader": "{LETTER}",
  "graded": [
    {"id": "xstate-01", "domain": "xstate", "score": <1-5>, "rationale": "..."},
    …24 entries total, one per prompt id…
  ]
}
```

**Critical constraints**:
- Read suite.json and the raw responses file ONCE each.
- Do NOT read any other .graded.grader_*.json, decision.md, or historical
  results files. Blind protocol.
- All 24 prompts must be graded.
- `domain` values must be exactly: "xstate", "fp", "reactive", "eventsourcing",
  "capability".

Report back with a one-line per-domain means summary and confirm the file
was written.
