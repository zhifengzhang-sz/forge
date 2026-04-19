# Setup Pitfalls

Problems encountered setting up the training environment on Ubuntu with RTX 5090. Documented so you don't hit them again.

## System Packages Required

Before `setup.sh`, install these system packages:

```bash
sudo apt install -y python3.13-venv python3.13-dev gcc cmake libcurl4-openssl-dev libssl-dev
```

- **python3.13-venv** — Python's `venv` module is not included by default on Ubuntu
- **python3.13-dev** — Python C headers needed by Triton to compile CUDA runtime modules
- **gcc** — C compiler needed by Triton (usually pre-installed on Ubuntu)
- **cmake** — needed by Unsloth's GGUF export to build llama.cpp
- **libcurl4-openssl-dev** — needed by llama.cpp's cmake build (Unsloth installs internally)
- **libssl-dev** — needed by llama.cpp's cmake build (Unsloth installs internally)

## Unsloth Dependencies

Unsloth's main package doesn't pull all its dependencies. The install order matters:

1. PyTorch with CUDA first (specific index URL required)
2. `unsloth` from git
3. `unsloth_zoo` — required but not auto-installed by unsloth
4. `torchvision` — required by unsloth's vision_utils import, even for text-only training

All of these are handled by `requirements.txt`. Don't install unsloth via `pip install unsloth` alone.

## HuggingFace Model Names

Vendor model names (`Qwen/Qwen3-14B-Instruct`, `google/gemma-4-31b-it`) require HuggingFace authentication even for open-source models.

Unsloth hosts pre-quantized copies that don't require auth:

| What you want | Wrong name (needs auth) | Correct name (no auth) |
|---|---|---|
| Qwen3 14B for training | `Qwen/Qwen3-14B-Instruct` | `unsloth/Qwen3-14B` |
| Gemma 4 31B for training | `google/gemma-4-31b-it` | `unsloth/gemma-4-31b-it` |

These are 4-bit NF4 quantized for training. The full-precision weights download automatically when Unsloth loads them.

## Qwen3 Has No Separate "Instruct" Model

Unlike Gemma 4 (which has a separate `-it` instruct variant), Qwen3 uses a single model with built-in thinking/non-thinking modes. There is no `Qwen3-14B-Instruct` — it's just `Qwen3-14B`.

## SFTTrainer Dataset Format

Unsloth's patched SFTTrainer does not accept a `formatting_func` the same way as vanilla TRL. The correct approach:

**Wrong:** Pass `formatting_func` to SFTTrainer constructor.

**Right:** Pre-map the dataset to a `text` column using `tokenizer.apply_chat_template()`, then pass the mapped dataset without `formatting_func`:

```python
def format_chat(examples):
    texts = []
    for messages in examples["messages"]:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        texts.append(text)
    return {"text": texts}

dataset = dataset.map(format_chat, batched=True, remove_columns=["messages", "id"])
```

## Triton Compilation Error

Triton compiles CUDA utility modules at runtime using gcc. If `python3.13-dev` is missing, the compilation fails with:

```
subprocess.CalledProcessError: Command '['/usr/bin/gcc', ... '-I/usr/include/python3.13']' returned non-zero exit status 1.
```

Fix: `sudo apt install -y python3.13-dev`

## Flash Attention 2

Unsloth reports "Flash Attention 2 installation seems to be broken. Using Xformers instead." This is expected on RTX 5090 (Blackwell) with current FA2 builds. Xformers provides equivalent performance. No action needed.

## llama.cpp Requirements Conflict

**Never run `pip install -r requirements.txt` inside the llama.cpp directory.** It installs CPU-only torch, downgrades transformers, and destroys the training venv. The only dependency needed from llama.cpp is the `gguf` Python package, which is already in our `requirements.txt`.

`export.py` uses Unsloth's `save_pretrained_gguf()` which handles llama.cpp internally. The old `export.sh` was removed because it cloned llama.cpp manually and its `pip install -r requirements.txt` destroyed the training venv.

## Unsloth Version Constraint Conflict

`unsloth_zoo==2026.4.8` declares `torch<2.11.0` but works fine with `torch==2.11.0+cu128`. A single `pip install -r requirements.txt` fails because pip enforces this constraint. The workaround is staged installation:

1. Install torch first: `pip install torch==2.11.0+cu128 ...`
2. Install HuggingFace stack: `pip install transformers trl datasets ...`
3. Install unsloth with `--no-deps`: `pip install --no-deps unsloth_zoo unsloth`

`setup.sh` handles this automatically.

## Unsloth GGUF Export Needs cmake and Prompts for Install

`save_pretrained_gguf()` still uses llama.cpp internally. On first run it clones, builds, and caches llama.cpp. This requires `cmake` on the system. If cmake is missing, Unsloth prompts to install it via `sudo apt-get install cmake -y`. This prompt fails in non-interactive contexts (background processes, CI) with `EOFError: EOF when reading a line`.

Fixes:
- Pre-install cmake: `sudo apt install -y cmake`
- For non-interactive runs, pipe empty input: `echo "" | python3 export.py --model qwen3-14b`
- After the first successful export, llama.cpp is cached and no further prompts occur

## RTX 5090 Power Cap (PSU Transient Protection)

The 5090 has a 575 W TDP but documented transient spikes to 700–800 W for brief moments during workload transitions (training finish → ollama model load, GGUF conversion → inference warm-up). A PSU at the low end of the 5090's recommended envelope (e.g. 850–1000 W with a high-core CPU) will trip overcurrent protection on those spikes — no kernel panic, no thermal event, the machine just cuts power and reboots.

This happened during the v0.6 training run: clean hardware-level cut at 2026-04-18 23:22:54 EDT, journal ends mid-sentence on a routine NetworkManager warning. Training and eval outputs survived because they had flushed to disk seconds before the cut.

**Mandatory mitigation before any heavy training run:**

```bash
sudo nvidia-smi -pm 1      # persistence mode (survives driver reloads)
sudo nvidia-smi -pl 500    # cap sustained draw at 500 W
```

Persistent across reboots via the checked-in systemd unit:

```bash
sudo cp ops/systemd/nvidia-pl.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now nvidia-pl.service
```

Verify with `systemctl status nvidia-pl.service` — should show `active (exited)`.

If the machine still crashes at 500 W, the PSU itself needs upgrading. Recommended envelope for 5090 + high-core CPU is **1200 W** (not the 850 W some vendor configs ship with).

Independent of the power cap, don't let training → ollama model load overlap: serialise the two (wait for `save_pretrained_gguf()` to return before launching `ollama create`) so the GPU has a moment to settle between sustained loads.

## Ollama Version

Gemma 4 models require Ollama 0.20+. The `curl install.sh` script may install an older version if a snap version is present. Check with `ollama --version`. If you have both snap and apt versions, remove snap first:

```bash
sudo snap remove ollama --purge
curl -fsSL https://ollama.com/install.sh | sh
```
