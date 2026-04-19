# v0.6 — Closing the XState v5 Gap

See `docs/review/v0.6.md` for the authoritative plan. This README is a map.

## Purpose

v0 tested LoRA fine-tuning at 50-example scale (below the published ~1k–10k
floor) and saw no real gap-closing on XState. v0.6 is the real test: produce
~1500 synthesised (instruction, completion) pairs, each verified with
`tsc --strict --noEmit` against `xstate@5`, then fine-tune qwen3:14b at proper
scale and rank. Goal is XState 1.60 → 4.0+ on the unchanged `v0/eval/v0.json`.

The teacher is Claude Opus 4.7 (5.00/5.00 on v0 XState), driven through
Claude Code subagents — no direct Anthropic API spend.

## Directory layout

```
v0.6/
├── README.md                         (this file)
├── seeds/
│   ├── patterns.md                   30 XState v5 patterns w/ API surface
│   ├── phrasings.md                  10 instruction phrasing templates
│   └── reference_examples.jsonl      46 real v5 machines (copied from v0)
├── data/                             (empty) synth.raw / synth.verified land here
└── gguf/                             (empty) quantised model + Modelfile land here
```

## Locked-in decisions

| Decision          | Value                              | Source                   |
| ----------------- | ---------------------------------- | ------------------------ |
| LoRA rank `r`     | **32** (v0 was 16)                 | `docs/review/v0.6.md` §C |
| Epochs            | **3** (v0 was 2)                   | `docs/review/v0.6.md` §C |
| Capability anchors| **+10% mix-in** from v0 curated    | `docs/review/v0.6.md` §C |
| Candidate target  | **~1500** (from ~5000, sized down) | subagent bandwidth       |
| Synthesis path    | **subagent-driven, no API**        | user preference          |
| Eval set          | unchanged `v0/eval/v0.json`        | contamination boundary   |

## What lives where (later tasks)

- `train.py`, `verify.py`, `synth.py` will be added alongside this README.
- `data/synth.raw.jsonl`, `data/synth.verified.jsonl` produced by Phases A+B.
- `gguf/qwen3-14b.Q4_K_M.gguf` + `Modelfile` produced by Phase C.
- Grades re-use `v0/grading/` patterns (blinded, fresh subagents per arm).

## Guardrails

- Do not copy prompts from `v0/eval/v0.json` into any seed or training file.
- Do not touch `v0/` — additive-only work under `v0.6/`.
- Keep the tool-call canary green post-train (`v0/tool_call_smoke.py`).
