"""Combine two grader outputs for v2.0 into final results with regression gate.

Fork of combine_v1.2.py for the v2.0 fresh-from-base 4-domain arm. Baselines
and gate tolerances unchanged — v2 tests whether v0.7's proven recipe
(atomic-drill data + fresh-base + joint multi-task) holds when adding
Event Sourcing as a fourth domain. Gate stays at v1/v1.1/v1.2 levels so
arm-over-arm comparison is clean.

Reads:
  v0/results/v2.0.graded.grader_A.json
  v0/results/v2.0.graded.grader_B.json

Writes:
  v0/results/v2.0.json  -- combined with grader_A, grader_B, mean, disagreement

Regression gate (per docs/v2.plan.md §"v2.0 eval + grade"):
  - Compute per-domain v2.0_mean - v0.7_baseline.
  - If any trained domain (XState/FP/RX) drops by >= tolerance, EXIT NON-ZERO.
  - RX tolerance 0.20 (v0.7 - 0.2) — tight per cross-model tension #3.
"""
import json
import sys
from pathlib import Path
from statistics import mean

V0 = Path("/home/zzhang/dev/ai/models/forge/v0/results")

# v0.7-r64 baselines (from v0.7/decision.md; same as v1/v1.1/v1.2 combine)
V07_BASELINE = {
    "xstate": 4.10,
    "fp": 4.40,
    "reactive": 4.80,
    "eventsourcing": 3.10,  # informational; not gated (no v0.7 ES training)
    "capability": 4.25,
}

# Regression gate tolerances — identical to v1/v1.1/v1.2.
GATE_TOLERANCE = {
    "xstate": 0.30,
    "fp": 0.30,
    "reactive": 0.20,
}

# Original v0.7 anchor prompt identifiers (12 unique). Used to partition
# capability into Cap_old12 vs Cap_new18 — same hints as v1/v1.1/v1.2.
V07_ANCHOR_HINTS = [
    "fib", "top_by_score", "find_py_files", "rest_graphql", "refactor_200",
    "read_file_tool", "list_files_tool", "monorepo", "memoize", "promise_all",
    "dataclass", "sql_review",
]


def combine() -> dict:
    a_path = V0 / "v2.0.graded.grader_A.json"
    b_path = V0 / "v2.0.graded.grader_B.json"
    if not a_path.exists() or not b_path.exists():
        sys.stderr.write(
            f"ERROR: grader outputs missing.\n  {a_path}\n  {b_path}\n"
            "Run two-grader dispatch first.\n"
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
        "model": a.get("model", "ts-forge-v2.0"),
        "arm": "v2.0-r64",
        "graders": ["A", "B"],
        "graded": combined,
    }
    (V0 / "v2.0.json").write_text(json.dumps(out, indent=2))
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


def regression_gate(means: dict[str, float]) -> list[str]:
    failures = []
    for dom, tol in GATE_TOLERANCE.items():
        score = means.get(dom)
        baseline = V07_BASELINE[dom]
        if score is None:
            failures.append(f"{dom}: missing in v2.0 graded output")
            continue
        delta = score - baseline
        if delta <= -tol:
            failures.append(
                f"{dom}: v2.0 {score:.2f} vs v0.7 {baseline:.2f} "
                f"(delta {delta:+.2f}, tolerance -{tol:.2f}) — REGRESSION"
            )
    return failures


def report(combined: dict) -> int:
    means = domain_means(combined)
    print("\n=== v2.0-r64 ===")
    print(f"{'domain':<18} {'v2.0':>6} {'v0.7':>6} {'delta':>8} {'gate':>6}")
    print("-" * 50)
    for dom in ["xstate", "fp", "reactive", "eventsourcing", "capability"]:
        v = means.get(dom)
        base = V07_BASELINE.get(dom, 0.0)
        if v is None:
            print(f"{dom:<18} {'—':>6} {base:>6.2f}")
            continue
        tol = GATE_TOLERANCE.get(dom)
        delta = v - base
        gate = ("halt" if tol is not None and delta <= -tol
                else "ok" if tol is not None else "n/a")
        print(f"{dom:<18} {v:>6.2f} {base:>6.2f} {delta:>+8.2f} {gate:>6}")

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

    failures = regression_gate(means)
    print(f"\n=== Regression gate ===")
    if failures:
        print("HALT — trained domain regression detected:")
        for msg in failures:
            print(f"  - {msg}")
        print("\nv2/decision.md will document the halt.")
        return 1
    print("PASS — no trained-domain regression. Ship authorized.")
    return 0


def main() -> None:
    combined = combine()
    exit_code = report(combined)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
