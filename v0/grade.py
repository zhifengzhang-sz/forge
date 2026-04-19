"""Interactive blinded grader for v0 evals.

Loads one or more <arm>.raw.json files, shuffles all responses across arms,
hides arm labels, prompts grader for 1-5 score per response, then unblinds
and writes per-arm graded results.

Usage:
    # Single arm (Phase 1 baseline — no blinding needed but consistent):
    python v0/grade.py base

    # Three-way blinded grading (Phase 4):
    python v0/grade.py base curated extracted
"""

import json
import random
import sys
from pathlib import Path

V0 = Path(__file__).parent
RESULTS = V0 / "results"


def load_raw(arm: str) -> dict:
    path = RESULTS / f"{arm}.raw.json"
    if not path.exists():
        sys.exit(f"Missing {path}. Run run_eval.py --arm {arm} first.")
    return json.loads(path.read_text())


def grade_one(prompt: str, response: str, auto_checks: dict, idx: int, total: int) -> int:
    print("\n" + "=" * 80)
    print(f"[{idx}/{total}]")
    print(f"PROMPT: {prompt}\n")
    print(f"RESPONSE:\n{response}\n")
    print(f"Auto-checks: hits={len(auto_checks.get('must_have_hits', []))}/"
          f"{len(auto_checks.get('must_have_hits', [])) + len(auto_checks.get('must_have_misses', []))}, "
          f"misses={auto_checks.get('must_have_misses', [])}, "
          f"violations={auto_checks.get('must_not_have_violations', [])}")
    while True:
        s = input("Grade (1=unusable, 2=bad, 3=acceptable, 4=good, 5=idiomatic) [1-5]: ").strip()
        if s in {"1", "2", "3", "4", "5"}:
            return int(s)


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    arms = sys.argv[1:]
    raws = {arm: load_raw(arm) for arm in arms}

    items = []
    for arm, raw in raws.items():
        for r in raw["responses"]:
            if "error" in r:
                continue
            items.append({"arm": arm, **r})

    random.seed(42)
    random.shuffle(items)

    grades: dict[tuple[str, str], int] = {}
    for i, item in enumerate(items, 1):
        score = grade_one(item["prompt"], item["response"], item.get("auto_checks", {}), i, len(items))
        grades[(item["arm"], item["id"])] = score

    for arm, raw in raws.items():
        graded = {**raw, "graded": []}
        for r in raw["responses"]:
            if "error" in r:
                graded["graded"].append({**r, "score": None})
            else:
                graded["graded"].append({**r, "score": grades.get((arm, r["id"]))})
        out = RESULTS / f"{arm}.json"
        out.write_text(json.dumps(graded, indent=2))
        domain_scores = [g["score"] for g in graded["graded"]
                         if g.get("score") and g["domain"] != "capability"]
        cap_scores = [g["score"] for g in graded["graded"]
                      if g.get("score") and g["domain"] == "capability"]
        print(f"\n{arm}: domain avg = {sum(domain_scores)/len(domain_scores):.2f} "
              f"(n={len(domain_scores)}), "
              f"capability avg = {sum(cap_scores)/len(cap_scores):.2f} "
              f"(n={len(cap_scores)})")
        print(f"  → {out}")


if __name__ == "__main__":
    main()
