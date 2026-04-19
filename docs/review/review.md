I'll do thorough research before making judgments. Your hunch is right — several design choices in this project look like they predate best practices that have emerged for Blackwell/RTX 5090. Let me verify this with current sources.I've done a solid pass through the project and cross-checked the assumptions against current sources. Your hunch is right — a lot of the pain documented in `setup.pitfalls.md` and `lessons.learned.md` was avoidable, and a few core design choices are worth revisiting. Below is what I found, ordered roughly by impact.

## 1. The biggest miss: you didn't need to build the environment from scratch

Unsloth has shipped an official Docker image that supports Blackwell (RTX 50-series and B200) since late 2025. NVIDIA's own developer blog walks through exactly this workflow: `docker run --gpus all unsloth/unsloth` and you're training. The image has PyTorch+CUDA 12.8, Triton, bitsandbytes, xformers, and Unsloth itself pre-compiled against sm_120. You also skip the `python3.13-dev` headaches, the `unsloth_zoo` torch-version-conflict dance, and the manual staged pip install entirely.

Your `setup.sh` is a lovingly-debugged solution to a problem Unsloth already solved. A single-container workflow would eliminate at least half of `setup.pitfalls.md`. If you want reproducibility across machines, the Docker image is also more reproducible than pinned pip versions, because the kernels are compiled into the image rather than compiled at install-time on whatever Ubuntu you happen to have.

The one caveat is that the NVIDIA blog notes xformers sometimes still needs a `TORCH_CUDA_ARCH_LIST="12.0"` rebuild from source, so you may hit one compile step — but that's it.

## 2. The model choice is questionable

You're fine-tuning `Qwen3-14B`, which is a general model, when `Qwen3-Coder-30B-A3B-Instruct` exists and is the community consensus pick for coding on a 32 GB RTX 5090. It's an MoE with 3.3B active out of 30.5B parameters, natively 256K context, and specifically pretrained on code.

Your `lessons.learned.md` records "MoE can't be QLoRA-tuned on consumer GPUs" — that was accurate for Gemma 4 26B-A4B in April 2026 (bitsandbytes can't yet quantize its 3D fused expert tensors), but **Qwen3-30B-A3B is a different story**. Unsloth's own docs say it fine-tunes in ~17.5 GB VRAM with their custom MoE kernels, with one caveat: they disable router-layer fine-tuning by default and you need to download the full 16-bit model first because the 4-bit BnB MoE import path has issues.

Starting from a code-specialized base model means you spend your LoRA rank on domain specialization rather than teaching the model what TypeScript is. It would also likely cut your required dataset size and probably beat your Gemma 4 31B candidate on every coding metric.

If you don't want the MoE complexity, **Qwen3-Coder-30B dense** (if available in 4-bit) or staying with Qwen3-14B is fine — but at least pick a code-specialized base. The two-model evaluation between `Qwen3-14B` and `Gemma 4 31B` is not actually comparing the best available options for your stated goal.

## 3. The Claude Code integration probably won't work the way you designed it

This is the finding that concerned me most. Your stated goal is "drop-in local backend for Claude Code." Three things will likely break this:

**(a) Claude Code needs tool-calling to do anything useful** (file reads, edits, bash). A recent GitHub thread on exactly this integration reports that under local Ollama, fine-tuned models often behave as plain LLMs with no tool access, while cloud proxy models work fine. The base Qwen3-14B supports tool calls; whether yours still does after fine-tuning depends on whether your training data preserves that capability — and yours doesn't. You have 1,100 examples of `(instruction → TypeScript code)`, none of which are tool-call traces. This is exactly the regime where Kalajdzievski (2024, "Scaling Laws for Forgetting When Fine-Tuning LLMs") found LoRA does NOT prevent catastrophic forgetting — there's a roughly linear relationship between fine-tuning performance and amount forgotten.

**(b) Claude Code maps Haiku/Sonnet/Opus to specific model names.** Just setting `ANTHROPIC_BASE_URL` and `ANTHROPIC_API_KEY` isn't enough; you also need `ANTHROPIC_DEFAULT_HAIKU_MODEL`, `ANTHROPIC_DEFAULT_SONNET_MODEL`, and `ANTHROPIC_DEFAULT_OPUS_MODEL` pointing at your local model, or it fails silently on internal role calls.

**(c) Newer Ollama has `ollama launch claude --model ts-forge`** as a canonical one-liner that handles this automatically.

I'd test the integration with the **base** Qwen3-Coder-30B via `ollama launch claude` first, before training anything. If the base model works well enough in Claude Code for your TypeScript workflow, you may find fine-tuning adds little beyond the risk of breaking tool calls.

## 4. The training data has a conceptual problem

Your pipeline extracts *implementations* from `fp-ts`, `rxjs`, `xstate`, and `EventSourcing.NodeJS` — that is, the internals of these libraries. But when a developer asks Claude Code "write fp-ts code to validate a form", they want library **usage**, not a new implementation of `pipe` or `Option.map`. You're training the model to recreate library internals, which is:

- not what users ask for
- already memorized by most code-pretrained models
- likely to produce weird output (the model will reinvent `pipe` when you ask it to *use* `pipe`)

Better sources would be downstream projects that *consume* these libraries — sample apps, tutorial repos, blog post code, the libraries' own examples/ directories (which you're skipping via `__tests__` and `examples` exclusions in some configs). XState's `examples/` directory in particular is goldmine territory that I don't see in your focus.

Separately: template-based instruction generation with deterministic template selection (hashing the fingerprint to pick 1 of 3 templates) means **each domain has only ~3 instruction phrasings, repeated hundreds of times**. The model will memorize those phrasings as keys. You'll get a model that performs great when the prompt matches the template verbatim and poorly when it doesn't. The LLM-generated variant (`instruct.py`) is strictly better and the $3-5 is a trivial cost.

Dataset size itself (~1,100 examples across 4 domains, so ~275 each) is at the low end for complex domain adaptation. The range that actually shows up in practice for this kind of task is 1,000-5,000 *per domain*, not total. For 4 domains that implies ~4,000-20,000 total. The quality > quantity rule holds, but there's a floor — your `--full-history` flag to mine git diffs is a smart move here and I'd use it by default.

## 5. Catastrophic forgetting is a real risk at your current settings

r=32, `lora_alpha=32`, 3 epochs, narrow domain, no general-coding mix-in data. This is aggressive. Published results (Kalajdzievski 2024, plus the DigitalOcean/Let's Data Science guides) suggest you should:

- drop `r` to 8 or 16 (r=32 has enough capacity to really overwrite things)
- mix in 10-20% general code examples (OSS Python, other TS, not just your 4 domains) specifically to anchor generic capability
- monitor not just eval loss on held-out in-domain examples but also a separate *out-of-domain* regression check (can it still write a for-loop? follow a general instruction? make a tool call?)
- stop early — 3 full epochs on a small dataset is where forgetting really kicks in

## 6. The pipeline is over-engineered for the actual scale

You built a language-abstraction layer (the `LanguageModule` protocol), a topic-config system, LLM-as-judge, template AND API-based instruction generation, a data reviewer agent, fingerprint-based held-out exclusion, balance formulas with floors and caps. This is production-grade infrastructure for ~1,000 examples.

The hidden cost is that every iteration on the *actual thing that matters* (model quality) goes through this big machine. I'd strongly suggest a "v0" path: pick one model, pick one domain (say, xstate), use `--skip-judge --skip-instruct`, write 100 instruction-completion pairs by hand in a JSONL, train, evaluate. That's a half-day loop. If xstate-only works, scale up. If it doesn't, the elaborate pipeline wasn't going to save you.

Your `lessons.learned.md` already identified this pattern ("Test the full pipeline end-to-end with 10 examples before scaling") — the project itself doesn't enforce it.

## 7. Smaller things worth fixing

The brace-matching TypeScript parser in `lib/typescript/extract.py` is fragile — it'll break on strings containing `{`, template literals, and a dozen other edge cases. `tree-sitter-typescript` is a 2-line dependency that gives you a proper AST. Given that extraction yield affects the whole pipeline, this is cheap to upgrade and removes a class of silent errors.

Scoring signals are weighted uniformly (0.08 per signal, capped at 0.4), which means after 5 domain terms you've saturated and more signal gives zero additional score. That's probably not what you want.

`checkpoints/` stores every 50 steps with `load_best_model_at_end=True` — at 198 steps for a 14B model with r=32 adapters, that's fine in size but the save cadence is tight. 100 steps is plenty.

`Modelfile` hardcodes `num_ctx 32768`. For a 14B at Q8_0 with 18GB KV headroom you could run 64K or higher; for the 31B at Q6_K with only 8GB headroom you may want to drop it.

## What I'd do differently

Concretely, if I were restarting this project with what we know now:

1. `docker run --gpus all unsloth/unsloth` — skip the setup saga entirely.
2. Start from `unsloth/Qwen3-Coder-30B-A3B-Instruct-unsloth-bnb-4bit` (code-specialized, MoE, fits).
3. Before training anything, verify the **base model** works end-to-end in Claude Code via `ollama launch claude --model qwen3-coder:30b`. Test whether it actually solves your TypeScript tasks unmodified. This may be all you need.
4. If fine-tuning is still justified, shift data collection from library internals to library usage (example directories, downstream consumers).
5. Use `r=16`, `lora_alpha=16`, 2 epochs, and mix ~15% general code in with domain data.
6. Use LLM-generated instructions, not templates — the $5 is nothing compared to the risk of the model memorizing template shapes.
7. Keep the held-out exclusion via fingerprint and the SHA-256 dedup — those are both right.
8. Evaluate on both in-domain *and* out-of-domain prompts, including a small tool-calling smoke test.

The docs themselves (`design.md`, `data.provenance.md`, `lessons.learned.md`) are genuinely well-written and catch a lot of real gotchas. The issue is mostly that the project committed to a particular approach (build everything from scratch, dual-model eval, elaborate pipeline) before verifying that the simpler paths (Docker, base Qwen3-Coder, direct Claude Code integration) wouldn't get you 80% of the way there for 10% of the effort.

Want me to sketch a concrete "v0" plan — the minimal path to verify whether fine-tuning adds value over the base Qwen3-Coder-30B, before you rebuild the pipeline?