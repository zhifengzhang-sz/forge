"""Run every prompt in the eval suite through an Ollama model.

Usage:
    python tools/eval/run.py --model ts-forge-v2.0 --arm v2.0
    python tools/eval/run.py --model ts-forge-v0.7-r64 --arm v0.7-recheck
    python tools/eval/run.py --model ts-forge-v2.0 --arm v2.0 --temperature 0.6  # legacy mode

Defaults:
  --suite:        tools/eval/suite.json
  --out-dir:      results/<arm>/
  --temperature:  0 (deterministic, reproducible across sessions)

Output: results/<arm>/<YYYY-MM-DD>.raw.json (and a legacy symlink at
        v0/results/<arm>.raw.json for backward-compat with archived combine_*.py scripts).

Temperature 0 is the new default. Prior sessions (pre-2026-04-24) ran at
temp=0.6 (Modelfile default) which introduced ±0.2 per-domain sampling
noise between eval runs. Use --temperature 0.6 only to reproduce a
historical run exactly.
"""

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

import requests

TOOLS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TOOLS_DIR.parent.parent
DEFAULT_SUITE = TOOLS_DIR / "suite.json"
DEFAULT_OUT_ROOT = REPO_ROOT / "results"
LEGACY_OUT_ROOT = REPO_ROOT / "v0" / "results"
OLLAMA_URL = "http://localhost:11434/api/generate"


def call_ollama(model: str, prompt: str, temperature: float, timeout: int = 180) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "seed": 42,
        },
    }
    if temperature == 0:
        payload["options"]["top_p"] = 1.0
        payload["options"]["top_k"] = 1
    r = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()["response"]


def auto_check(response: str, must_have: list[str], must_not_have: list[str]) -> dict:
    return {
        "must_have_hits": [s for s in must_have if s in response],
        "must_have_misses": [s for s in must_have if s not in response],
        "must_not_have_violations": [s for s in must_not_have if s in response],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    ap.add_argument("--out-dir", type=Path, default=None,
                    help=f"defaults to {DEFAULT_OUT_ROOT}/<arm>/")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="0 (default) for deterministic eval; 0.6 to reproduce pre-2026-04-24 runs")
    ap.add_argument("--session", default=None,
                    help="session label for the output filename (default: today's YYYY-MM-DD). "
                         "Use a suffix like 2026-04-24-evening when running multiple sessions per day.")
    ap.add_argument("--no-legacy-symlink", action="store_true",
                    help="skip writing the legacy v0/results/<arm>.raw.json copy")
    args = ap.parse_args()

    suite = json.loads(args.suite.read_text())
    out_dir = args.out_dir or (DEFAULT_OUT_ROOT / args.arm)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = args.session or datetime.now().strftime("%Y-%m-%d")
    out_path = out_dir / f"{timestamp}.raw.json"

    results = {
        "model": args.model,
        "arm": args.arm,
        "suite": str(args.suite.relative_to(REPO_ROOT) if args.suite.is_absolute() else args.suite),
        "temperature": args.temperature,
        "session_date": timestamp,
        "responses": [],
    }
    for i, p in enumerate(suite["prompts"], 1):
        print(f"[{i}/{len(suite['prompts'])}] {p['id']} ... ", end="", flush=True)
        t0 = time.time()
        try:
            response = call_ollama(args.model, p["prompt"], args.temperature)
            elapsed = time.time() - t0
            checks = auto_check(response, p["must_have"], p.get("must_not_have", []))
            results["responses"].append({
                "id": p["id"],
                "domain": p["domain"],
                "prompt": p["prompt"],
                "response": response,
                "elapsed_s": round(elapsed, 1),
                "auto_checks": checks,
            })
            print(f"{elapsed:.1f}s, hits={len(checks['must_have_hits'])}/{len(p['must_have'])}, "
                  f"violations={len(checks['must_not_have_violations'])}")
        except Exception as e:
            print(f"FAIL: {e}")
            results["responses"].append({"id": p["id"], "error": str(e)})

    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out_path}")

    # Backward-compat: also write the legacy v0/results/<arm>.raw.json
    # so archived combine_*.py scripts keep working.
    if not args.no_legacy_symlink:
        LEGACY_OUT_ROOT.mkdir(parents=True, exist_ok=True)
        legacy_path = LEGACY_OUT_ROOT / f"{args.arm}.raw.json"
        legacy_path.write_text(json.dumps(results, indent=2))
        print(f"Wrote legacy copy: {legacy_path}")


if __name__ == "__main__":
    main()
