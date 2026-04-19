# v0 Grading Scripts

These scripts apply per-prompt 1-5 grades (with rationales) to the raw eval
outputs in `../results/`. Each script is the **complete record** of how a given
arm was graded — re-running it regenerates the corresponding `.json` file from
the matching `.raw.json` file.

The grader was Claude Opus 4.7 (1M context). Grades are not blinded — see
`../decision.md` § "What v0 was worth" for the methodology caveats.

## Scripts

| Script | Input | Output | Notes |
|---|---|---|---|
| `grade_qwen3coder30b.py` | `../results/base.qwen3coder30b.raw.json` | `../results/base.qwen3coder30b.json` | First baseline run, before model swap to qwen3:14b. Archived. |
| `grade_qwen3_14b_base.py` | `../results/base.raw.json` | `../results/base.json` | Authoritative baseline for v0 (matches the model that was fine-tuned). |
| `grade_phase4_arms.py` | `../results/curated.raw.json`, `../results/extracted.raw.json` | `../results/curated.json`, `../results/extracted.json` | Both fine-tuned arms. Also prints the comparison table. |
| `grade_claude_opus.py` | `../results/claude_opus.raw.json` | `../results/claude_opus.json` | Frontier baseline (Claude Opus 4.7 via fresh Claude Code subagent — no v0 conversation context). Used to verify Claude is viable as a teacher for v0.6 synthesis. |

## How to verify someone else's grading

1. Read each `<arm>.raw.json` to see the model output.
2. Read the corresponding `grade_*.py` to see the score + rationale per prompt.
3. Disagree? Edit the GRADES dict in the script and re-run — the `<arm>.json`
   will regenerate with your scores. Then compare against the existing one to
   surface the disagreements.

## Calibration

Grading scale (predeclared in `../readme.md`):

- **1** = unusable
- **2** = bad (significant errors, wrong language/approach)
- **3** = acceptable (works but has flaws or mediocre)
- **4** = good (mostly idiomatic, minor issues)
- **5** = idiomatic + correct

For XState specifically, the v5 idiom checklist driving the score:

- Uses `setup({ types, actors, guards, actions }).createMachine(...)` (not
  raw `createMachine` for non-trivial machines)
- `types: { context, events }` block for type inference
- `guard:` keyword (not v4's `cond:`)
- Destructured `({ context })` / `({ event })` action signatures (not v4's
  positional `(ctx, event)`)
- `event.output` for fromPromise results (not v4's `event.data`)
- `createActor(machine).start()` (not v4's `interpret(machine).start()`)
- Actors registered in `setup({ actors: { ... } })` rather than `services:`
