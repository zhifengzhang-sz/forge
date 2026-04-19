# v0 — Fine-tuning Spike

One-day exercise. Answers one question: on the user's actual TypeScript tasks,
does fine-tuning meaningfully beat base `qwen3-coder:30b`?

The main pipeline (`extract_pipeline.py`, `lib/`, `app/`, `train.py`,
`export.py`) is **paused** pending the outcome here. Nothing in the parent
directory is touched by v0.

## Layout

```
v0/
├── eval/v0.json              # 15-20 domain prompts + 3-5 capability regression
├── data/
│   ├── xstate_curated.jsonl  # 50-75 hand-written examples
│   └── xstate_extracted.jsonl # ~50 mechanically-extracted usage examples
├── train_v0.py               # ~50-line trainer, runs in unsloth/unsloth Docker
├── Modelfile.v0.curated      # Ollama config for curated adapter
├── Modelfile.v0.extracted    # Ollama config for extracted adapter
├── results/
│   ├── baseline.json         # Phase 1 grades on base qwen3-coder:30b
│   ├── curated.json          # Phase 4 grades on curated adapter
│   └── extracted.json        # Phase 4 grades on extracted adapter
├── gguf/                     # GGUF outputs from training
└── decision.md               # Phase 5 outcome
```

## Phases

0. **Write eval set** (30-45 min) — prompts represent real TypeScript work,
   each has `must_have` / `must_not_have` checks. Capability regression prompts
   (Python, tool calls, reasoning) catch catastrophic forgetting.
1. **Baseline base model** (30-60 min) — grade 1-5 manually. Decision gate:
   - ≥4.0: stop, base is good enough
   - 2.5-4.0: proceed with fine-tuning
   - ≤2.5: different strategy needed
2. **Two training datasets** in parallel:
   - **2a Curated**: 50-75 hand-written XState pairs (~40 patterns, ~10 v4→v5
     fixes, ~10 capability anchors)
   - **2b Extracted**: ~50 examples from XState `examples/` + consumer repos,
     one mechanically-applied instruction template (style matches 2a)
3. **Train** (5-10 min each) — `unsloth/Qwen3-Coder-30B-A3B-Instruct-unsloth-bnb-4bit`,
   r=16, lr=1e-4, 2 epochs. Forgetting-protective hyperparams.
4. **Re-evaluate** (30 min) — same eval, both adapters, blinded grading.
5. **Decide** — 4-row matrix:

| Base | Curated | Extracted | Action |
|---|---|---|---|
| Low | High | High | Retarget pipeline to usage sources. Existing infra mostly salvageable. |
| Low | High | Low | Retire pipeline. Hand-curation is the way. |
| Low | Low | Low | Reconsider premise (RAG? prompting? different base?). |
| High | — | — | Ship base model. |

## Grading discipline (decided up-front, not post-hoc)

- **Threshold for "High"**: domain prompt average ≥ 3.5 AND capability prompts
  average ≥ 3.5 (no regression).
- **Threshold for "Low"**: domain prompt average < 3.5 OR capability average < 3.5.
- **Blinding**: when grading Phase 4, shuffle outputs and hide which arm produced
  them. Grade all 60 outputs (3 arms × 20 prompts) before unblinding.
