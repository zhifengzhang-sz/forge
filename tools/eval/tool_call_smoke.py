"""Tool-call regression test for v0.

Calls Ollama's OpenAI-compatible /v1/chat/completions with a tool definition.
A model that supports tool calls should emit tool_calls in the response.
A fine-tuned model that has lost tool calling will emit prose instead.

This is the v0-(B) canary — the failure mode that breaks Claude Code integration.

Usage:
    python v0/tool_call_smoke.py --model qwen3-coder:30b --arm base
    python v0/tool_call_smoke.py --model ts-forge-v0-curated --arm curated
    python v0/tool_call_smoke.py --model ts-forge-v0-extracted --arm extracted
"""

import argparse
import json
from pathlib import Path

import requests

V0 = Path(__file__).parent
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
    args = ap.parse_args()

    out_path = V0 / "results" / f"{args.arm}.toolcall.json"
    out_path.parent.mkdir(exist_ok=True)

    results = {"model": args.model, "arm": args.arm, "calls": []}
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


if __name__ == "__main__":
    main()
