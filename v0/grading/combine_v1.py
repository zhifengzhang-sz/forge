"""Combine two grader outputs for v1 into final results with regression gate.

Reads:
  v0/results/v1.graded.grader_A.json
  v0/results/v1.graded.grader_B.json
Also reads v0.7-r64 re-graded under the v1 session (per plan cross-model
tension #2 — calibration anchor) from the same grader output files — the
graders are expected to include both v1 scores AND v0.7-r64 ES re-grades.

Writes:
  v0/results/v1.json  -- combined with grader_A, grader_B, mean, disagreement

Regression gate (per plan Phase G):
  - Compute per-domain v1_mean - v0.7_baseline.
  - If any trained domain (XState/FP/RX) drops by >= 0.3, EXIT NON-ZERO.
  - RX threshold tightened to >= 4.60 (v0.7 - 0.2) per cross-model tension #3.

Capability breakdown (per plan cross-model tension #5):
  - Cap_old12 (original v0.7 anchors) vs Cap_new18 (new anchors).
  - Reported separately so we can isolate anchor-mix-vs-volume confounds.
"""
import json
import sys
from pathlib import Path
from statistics import mean

V0 = Path("/home/zzhang/dev/ai/models/forge/v0/results")

# v0.7-r64 baselines (from v0.7/decision.md)
V07_BASELINE = {
    "xstate": 4.10,
    "fp": 4.40,
    "reactive": 4.80,
    "eventsourcing": 3.10,  # informational; not gated
    "capability": 4.25,
}

# Regression gate tolerances (per plan)
# XState / FP: 0.3 drop from baseline triggers halt.
# RX: 0.2 drop triggers halt (tightened per cross-model tension #3).
GATE_TOLERANCE = {
    "xstate": 0.30,
    "fp": 0.30,
    "reactive": 0.20,
}

# Original v0.7 anchor prompt identifiers (the 12 unique from v0.7). These
# are used to partition Cap into Cap_old12 vs Cap_new18 for the confound
# check. Prompt IDs will depend on grader output schema — best-effort match
# by substring or by a tag field the graders may include.
V07_ANCHOR_HINTS = [
    "fib", "top_by_score", "find_py_files", "rest_graphql", "refactor_200",
    "read_file_tool", "list_files_tool", "monorepo", "memoize", "promise_all",
    "dataclass", "sql_review",
]


def combine_v1() -> dict:
    a_path = V0 / "v1.graded.grader_A.json"
    b_path = V0 / "v1.graded.grader_B.json"
    if not a_path.exists() or not b_path.exists():
        sys.stderr.write(
            f"ERROR: grader outputs missing.\n  {a_path}\n  {b_path}\n"
            "Run Phase G graders first.\n"
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
        "model": a.get("model", "ts-forge-v1"),
        "arm": "v1-r64",
        "graders": ["A", "B"],
        "graded": combined,
    }
    (V0 / "v1.json").write_text(json.dumps(out, indent=2))
    return out


def domain_means(combined: dict) -> dict[str, float]:
    by_domain: dict[str, list[float]] = {}
    for g in combined["graded"]:
        by_domain.setdefault(g["domain"], []).append(g["mean"])
    return {d: mean(scores) for d, scores in by_domain.items()}


def capability_breakdown(combined: dict) -> dict[str, float]:
    """Split capability domain into old-12 vs new-18 means."""
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


def regression_gate(v1_means: dict[str, float]) -> list[str]:
    """Return list of failure messages. Empty list = clear."""
    failures = []
    for dom, tol in GATE_TOLERANCE.items():
        v1_score = v1_means.get(dom)
        baseline = V07_BASELINE[dom]
        if v1_score is None:
            failures.append(f"{dom}: missing in v1 graded output")
            continue
        delta = v1_score - baseline
        if delta <= -tol:
            failures.append(
                f"{dom}: v1 {v1_score:.2f} vs v0.7 {baseline:.2f} "
                f"(delta {delta:+.2f}, tolerance -{tol:.2f}) — REGRESSION"
            )
    return failures


def report(combined: dict) -> int:
    v1_means = domain_means(combined)
    print("\n=== v1-r64 ===")
    print(f"{'domain':<18} {'v1':>6} {'v0.7':>6} {'delta':>8} {'gate':>6}")
    print("-" * 50)
    for dom in ["xstate", "fp", "reactive", "eventsourcing", "capability"]:
        v1 = v1_means.get(dom)
        base = V07_BASELINE.get(dom, 0.0)
        if v1 is None:
            print(f"{dom:<18} {'—':>6} {base:>6.2f}")
            continue
        tol = GATE_TOLERANCE.get(dom)
        delta = v1 - base
        gate = ("halt" if tol is not None and delta <= -tol
                else "ok" if tol is not None else "n/a")
        print(f"{dom:<18} {v1:>6.2f} {base:>6.2f} {delta:>+8.2f} {gate:>6}")

    # Cap breakdown
    cap_breakdown = capability_breakdown(combined)
    print(f"\ncapability breakdown:")
    for k, v in cap_breakdown.items():
        if v != v:  # NaN check
            print(f"  {k}: (no records)")
        else:
            print(f"  {k}: {v:.2f}")

    # Grader agreement
    rows = combined["graded"]
    agree_1pt = sum(1 for g in rows if g["disagreement"] <= 1)
    mean_disagree = mean(g["disagreement"] for g in rows)
    print(f"\ngrader agreement (within 1 pt): {agree_1pt}/{len(rows)}")
    print(f"overall mean disagreement: {mean_disagree:.2f}")

    # Regression gate
    failures = regression_gate(v1_means)
    print(f"\n=== Regression gate ===")
    if failures:
        print("HALT — trained domain regression detected:")
        for msg in failures:
            print(f"  - {msg}")
        print("\nPhase H (decision.md) will NOT be written.")
        return 1
    print("PASS — no trained-domain regression. Phase H authorized.")
    return 0


def main() -> None:
    combined = combine_v1()
    exit_code = report(combined)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
