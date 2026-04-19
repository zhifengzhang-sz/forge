"""Run every prompt in v0/eval/v0.json through an Ollama model.

Usage:
    python v0/run_eval.py --model qwen3-coder:30b --arm base
    python v0/run_eval.py --model ts-forge-v0-curated --arm curated
    python v0/run_eval.py --model ts-forge-v0-extracted --arm extracted

Writes raw outputs (no grades) to v0/results/<arm>.raw.json. Grade with grade.py.
"""

import argparse
import json
import time
from pathlib import Path

import requests

V0 = Path(__file__).parent
EVAL = V0 / "eval" / "v0.json"
OLLAMA_URL = "http://localhost:11434/api/generate"


def call_ollama(model: str, prompt: str, timeout: int = 180) -> str:
    r = requests.post(
        OLLAMA_URL,
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=timeout,
    )
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
    args = ap.parse_args()

    suite = json.loads(EVAL.read_text())
    out_path = V0 / "results" / f"{args.arm}.raw.json"
    out_path.parent.mkdir(exist_ok=True)

    results = {"model": args.model, "arm": args.arm, "responses": []}
    for i, p in enumerate(suite["prompts"], 1):
        print(f"[{i}/{len(suite['prompts'])}] {p['id']} ... ", end="", flush=True)
        t0 = time.time()
        try:
            response = call_ollama(args.model, p["prompt"])
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
    print(f"\nWrote {out_path}. Next: python v0/grade.py {out_path.name}")


if __name__ == "__main__":
    main()
