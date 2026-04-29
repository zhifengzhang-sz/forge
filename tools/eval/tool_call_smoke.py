"""Tool-call regression test.

Calls Ollama's OpenAI-compatible /v1/chat/completions with a tool definition.
A model that supports tool calls should emit tool_calls in the response.
A fine-tuned model that has lost tool calling will emit prose instead.

This is the catastrophic-forgetting canary — the failure mode that breaks
Claude Code integration.

Usage:
    python tools/eval/tool_call_smoke.py --model ts-forge-v2.0 --arm v2.0
    python tools/eval/tool_call_smoke.py --model ts-forge-v0.7-r64 --arm v0.7-recheck

Output: results/<arm>/<YYYY-MM-DD>.toolcall.json (+ legacy copy at
        v0/results/<arm>.toolcall.json).
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import requests

TOOLS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TOOLS_DIR.parent.parent
DEFAULT_OUT_ROOT = REPO_ROOT / "results"
LEGACY_OUT_ROOT = REPO_ROOT / "v0" / "results"
URL = "http://localhost:11434/v1/chat/completions"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file from disk.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute file path"}
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files in a directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "directory": {"type": "string", "description": "Absolute directory path"}
                },
                "required": ["directory"],
            },
        },
    },
]

PROMPTS = [
    {
        "id": "tc-01",
        "prompt": "What files are in /etc? Use the list_files tool to find out.",
        "expects_tool": "list_files",
    },
    {
        "id": "tc-02",
        "prompt": "Please read /etc/hostname and tell me the result.",
        "expects_tool": "read_file",
    },
    {
        "id": "tc-03",
        "prompt": "I need to see what's in the file at /tmp/notes.txt — please use your tools to do this.",
        "expects_tool": "read_file",
    },
]


def call(model: str, prompt: str) -> dict:
    r = requests.post(
        URL,
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "tools": TOOLS,
            "stream": False,
        },
        timeout=120,
    )
    r.raise_for_status()
    return r.json()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--out-dir", type=Path, default=None,
                    help=f"defaults to {DEFAULT_OUT_ROOT}/<arm>/")
    ap.add_argument("--no-legacy-symlink", action="store_true")
    args = ap.parse_args()

    out_dir = args.out_dir or (DEFAULT_OUT_ROOT / args.arm)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d")
    out_path = out_dir / f"{timestamp}.toolcall.json"

    results = {
        "model": args.model,
        "arm": args.arm,
        "session_date": timestamp,
        "calls": [],
    }
    for p in PROMPTS:
        print(f"[{p['id']}] expects {p['expects_tool']!r}... ", end="", flush=True)
        try:
            resp = call(args.model, p["prompt"])
            msg = resp["choices"][0]["message"]
            tool_calls = msg.get("tool_calls") or []
            called = [tc["function"]["name"] for tc in tool_calls]
            success = p["expects_tool"] in called
            results["calls"].append({
                "id": p["id"],
                "prompt": p["prompt"],
                "expected_tool": p["expects_tool"],
                "tools_called": called,
                "content": msg.get("content", ""),
                "success": success,
            })
            print(f"called={called}  {'PASS' if success else 'MISS'}")
        except Exception as e:
            print(f"ERROR: {e}")
            results["calls"].append({"id": p["id"], "error": str(e)})

    out_path.write_text(json.dumps(results, indent=2))
    pass_rate = sum(1 for c in results["calls"] if c.get("success")) / len(PROMPTS)
    print(f"\nTool-call pass rate: {pass_rate:.0%} ({sum(1 for c in results['calls'] if c.get('success'))}/{len(PROMPTS)})")
    print(f"Wrote {out_path}")

    if not args.no_legacy_symlink:
        LEGACY_OUT_ROOT.mkdir(parents=True, exist_ok=True)
        legacy_path = LEGACY_OUT_ROOT / f"{args.arm}.toolcall.json"
        legacy_path.write_text(json.dumps(results, indent=2))
        print(f"Wrote legacy copy: {legacy_path}")


if __name__ == "__main__":
    main()
