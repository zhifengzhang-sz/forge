# Data Collection Guide

Lessons from running the extraction pipeline on TypeScript source repositories.

**Prerequisites:** Run `bash setup.sh` first to install all dependencies and download models. Activate the venv with `source .venv/bin/activate` before running pipeline commands.

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

## LLM Quality Judge

After regex scoring and dedup, an LLM judge (Claude Haiku) evaluates each unit semantically. This catches problems the regex scorer misses: boilerplate that happens to use domain terms, config objects with generics, test utilities that aren't real patterns.

The judge scores 1-5. Units below 3 are dropped. Results are saved to `dataset/judge_results.jsonl` with scores and one-line reasoning.

Skip with `--skip-judge` for fast iteration during development. Run without skipping before instruction generation to ensure data quality.

To review judge results interactively, run the data reviewer agent in Claude Code:

```
Read agents/data.reviewer.md for instructions, then review the training
data quality. Read dataset/judge_results.jsonl, sample examples from
each domain, and tell me if the data looks good for fine-tuning.
```

## Cost Estimation

Two API cost stages in the pipeline:

**LLM judge** (Claude Haiku) — evaluates each unit after regex scoring:
- ~5,000 units at ~500 tokens context each
- Estimated cost: $2-3

**Instruction generation** (Claude Sonnet) — generates training prompts:
- ~1,000 units after judging and balancing
- Estimated input: ~500K tokens, output: ~50K tokens
- Approximate cost: $2-5 depending on model choice

The pipeline prints the estimate and asks for confirmation before making API calls. Use `--dry-run` to see the estimate without running anything.

## Training Data Format

The training JSONL uses HuggingFace's messages format:

```json
{
  "messages": [
    {"role": "user", "content": "Implement the RxJS operator `repeatWhen`..."},
    {"role": "assistant", "content": "import { Observable } from...\n\nexport function repeatWhen..."}
  ],
  "id": "abc1234f-0042"
}
```

**Unsloth SFTTrainer requires a `formatting_func`** that converts these messages into the model's chat template. The function must return a list of strings (not a single string). This applies the model-specific tokens (e.g. `<|im_start|>user`, `<|im_end|>` for Qwen3):

```python
def formatting_func(example):
    text = tokenizer.apply_chat_template(
        example["messages"], tokenize=False, add_generation_prompt=False
    )
    return [text]
```

**HuggingFace model names for training** use Unsloth's pre-quantized 4-bit variants (`unsloth/Qwen3-14B`, `unsloth/gemma-4-31b-it`), not the original vendor names (`Qwen/Qwen3-14B-Instruct`). The vendor models may require authentication; Unsloth's copies don't.
