"""Combine two grader outputs for v2.0 FP-only canary.

Fork of combine_v2.0.py for the first stacked-curriculum arm. The
training set is FP-only (320 v0.7 atomic records + 240 anchors),
fresh-from-base single LoRA at r=64. Non-FP domain scores will be at
base-model level (~1.6 XState, ~4.0 RX per the baseline in
lessons.learned.md) and are reported but not gated.

Strict gate:
  - fp MUST hit >= 4.40 (v0.7-r64 baseline) — this is the canary
    that validates (a) stacking infrastructure works end-to-end and
    (b) FP data + recipe reproduce v0.7's FP quality in isolation.

Reads:
  v0/results/v2.stack.fp.graded.grader_A.json
  v0/results/v2.stack.fp.graded.grader_B.json

Writes:
  v0/results/v2.stack.fp.json
"""
import json
import sys
from pathlib import Path
from statistics import mean

V0 = Path("/home/zzhang/dev/ai/models/forge/v0/results")

V07_BASELINE = {
    "xstate": 4.10,
    "fp": 4.40,
    "reactive": 4.80,
    "eventsourcing": 3.10,
    "capability": 4.25,
}

# Only FP is gated for this canary arm — XState/RX/ES aren't trained.
STRICT_FLOOR = {"fp": 4.40}

V07_ANCHOR_HINTS = [
    "fib", "top_by_score", "find_py_files", "rest_graphql", "refactor_200",
    "read_file_tool", "list_files_tool", "monorepo", "memoize", "promise_all",
    "dataclass", "sql_review",
]


def combine() -> dict:
    a_path = V0 / "v2.stack.fp.graded.grader_A.json"
    b_path = V0 / "v2.stack.fp.graded.grader_B.json"
    if not a_path.exists() or not b_path.exists():
        sys.stderr.write(
            f"ERROR: grader outputs missing.\n  {a_path}\n  {b_path}\n"
        )
        sys.exit(2)

    a = json.loads(a_path.read_text())
    b = json.loads(b_path.read_text())
    a_by_id = {g["id"]: g for g in a["graded"]}
    b_by_id = {g["id"]: g for g in b["graded"]}

    combined = []
    for pid in sorted(a_by_id):
        if pid not in b_by_id:
            sys.stderr.write(f"WARN: prompt {pid} missing from grader B; skipping\n")
            continue
        ag = a_by_id[pid]
        bg = b_by_id[pid]
        sa, sb = ag["score"], bg["score"]
        combined.append({
            "id": pid,
            "domain": ag["domain"],
            "grader_A_score": sa,
            "grader_A_rationale": ag.get("rationale", ""),
            "grader_B_score": sb,
            "grader_B_rationale": bg.get("rationale", ""),
            "mean": (sa + sb) / 2,
            "disagreement": abs(sa - sb),
        })

    out = {
        "model": a.get("model", "ts-forge-v2-stack-fp"),
        "arm": "v2.stack.fp-r64",
        "graders": ["A", "B"],
        "graded": combined,
    }
    (V0 / "v2.stack.fp.json").write_text(json.dumps(out, indent=2))
    return out


def domain_means(combined: dict) -> dict[str, float]:
    by_domain: dict[str, list[float]] = {}
    for g in combined["graded"]:
        by_domain.setdefault(g["domain"], []).append(g["mean"])
    return {d: mean(scores) for d, scores in by_domain.items()}


def capability_breakdown(combined: dict) -> dict[str, float]:
    old, new = [], []
    for g in combined["graded"]:
        if g["domain"] != "capability":
            continue
        pid = g["id"]
        is_old = any(h in pid for h in V07_ANCHOR_HINTS)
        (old if is_old else new).append(g["mean"])
    return {
        "Cap_old12": mean(old) if old else float("nan"),
        "Cap_new18": mean(new) if new else float("nan"),
    }


def strict_gate(means: dict[str, float]) -> list[str]:
    failures = []
    for dom, floor in STRICT_FLOOR.items():
        score = means.get(dom)
        if score is None:
            failures.append(f"{dom}: missing in graded output")
            continue
        if score < floor:
            failures.append(
                f"{dom}: v2.stack.fp {score:.2f} < strict floor {floor:.2f} — CANARY FAILED"
            )
    return failures


def report(combined: dict) -> int:
    means = domain_means(combined)
    print("\n=== v2.stack.fp-r64 (FP-only canary) ===")
    print(f"{'domain':<18} {'v2stk':>6} {'v0.7':>6} {'delta':>8} {'strict':>8}")
    print("-" * 52)
    for dom in ["xstate", "fp", "reactive", "eventsourcing", "capability"]:
        v = means.get(dom)
        base = V07_BASELINE.get(dom, 0.0)
        if v is None:
            print(f"{dom:<18} {'—':>6} {base:>6.2f}")
            continue
        delta = v - base
        floor = STRICT_FLOOR.get(dom)
        if floor is not None:
            gate = "PASS" if v >= floor else "FAIL"
        else:
            gate = "untrained"
        print(f"{dom:<18} {v:>6.2f} {base:>6.2f} {delta:>+8.2f} {gate:>8}")

    cap_breakdown = capability_breakdown(combined)
    print(f"\ncapability breakdown:")
    for k, v in cap_breakdown.items():
        if v != v:
            print(f"  {k}: (no records)")
        else:
            print(f"  {k}: {v:.2f}")

    rows = combined["graded"]
    agree_1pt = sum(1 for g in rows if g["disagreement"] <= 1)
    mean_disagree = mean(g["disagreement"] for g in rows)
    print(f"\ngrader agreement (within 1 pt): {agree_1pt}/{len(rows)}")
    print(f"overall mean disagreement: {mean_disagree:.2f}")

    failures = strict_gate(means)
    print(f"\n=== Canary gate ===")
    if failures:
        print("CANARY FAIL — FP did not reach v0.7 baseline:")
        for msg in failures:
            print(f"  - {msg}")
        print("\nStop before v2.1 and investigate.")
        return 1
    print("CANARY PASS — FP reproduces v0.7 baseline.")
    print("Infrastructure + data + recipe are sound.")
    print("Authorize v2.1 (stack RX adapter on top).")
    return 0


def main() -> None:
    combined = combine()
    exit_code = report(combined)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
