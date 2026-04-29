"""Delta-based gate: compare arm scores to a contemporaneous baseline.

Replaces the per-arm combine_v*.py scripts in v0/grading/ which compared
new-arm scores against ossified historical floors. Those floors were shown
(2026-04-24) to drift 0.5-0.9 between grader sessions on the same model
outputs; comparison became noise-driven. Instead, this script takes both
the new arm's grades and the same-session baseline model's grades and
computes per-domain deltas that cancel session-level grader drift.

Expected inputs (both from the SAME session; dispatch graders on both
models' responses in one batch so grader calibration is held constant):

  results/<arm>/<session_date>.graded.grader_A.json
  results/<arm>/<session_date>.graded.grader_B.json
  results/<baseline>/<session_date>.graded.grader_A.json
  results/<baseline>/<session_date>.graded.grader_B.json

Usage:
  python tools/gate/combine_delta.py \\
    --arm v2-stack-fp --baseline v0.7-recheck \\
    --session 2026-04-24 \\
    --trained-domains fp

  python tools/gate/combine_delta.py \\
    --arm v2.0 --baseline v0.7-r64 \\
    --session 2026-04-22 \\
    --trained-domains xstate fp reactive eventsourcing

Default gate tolerance: -0.3 on any trained domain halts.
"""
import argparse
import json
import sys
from pathlib import Path
from statistics import mean

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS = REPO_ROOT / "results"
LEGACY_V0_RESULTS = REPO_ROOT / "v0" / "results"

V07_ANCHOR_HINTS = [
    "fib", "top_by_score", "find_py_files", "rest_graphql", "refactor_200",
    "read_file_tool", "list_files_tool", "monorepo", "memoize", "promise_all",
    "dataclass", "sql_review",
]


def load_graded_pair(arm: str, session: str) -> tuple[dict, dict]:
    """Load grader A+B outputs, falling back to legacy v0/results/ layout."""
    # New layout first
    primary = RESULTS / arm
    a_path = primary / f"{session}.graded.grader_A.json"
    b_path = primary / f"{session}.graded.grader_B.json"
    if a_path.exists() and b_path.exists():
        return json.loads(a_path.read_text()), json.loads(b_path.read_text())

    # Legacy layout (v0/results/<arm>.graded.grader_{A,B}.json)
    legacy_a = LEGACY_V0_RESULTS / f"{arm}.graded.grader_A.json"
    legacy_b = LEGACY_V0_RESULTS / f"{arm}.graded.grader_B.json"
    if legacy_a.exists() and legacy_b.exists():
        return json.loads(legacy_a.read_text()), json.loads(legacy_b.read_text())

    sys.stderr.write(
        f"ERROR: no graded pair found for arm='{arm}' session='{session}'.\n"
        f"  tried: {a_path}\n"
        f"  tried: {b_path}\n"
        f"  tried: {legacy_a}\n"
        f"  tried: {legacy_b}\n"
    )
    sys.exit(2)


def combine_scores(a: dict, b: dict) -> list[dict]:
    a_by_id = {g["id"]: g for g in a["graded"]}
    b_by_id = {g["id"]: g for g in b["graded"]}
    rows = []
    for pid in sorted(a_by_id):
        if pid not in b_by_id:
            sys.stderr.write(f"WARN: {pid} missing from grader B; skipping\n")
            continue
        ag = a_by_id[pid]
        bg = b_by_id[pid]
        sa, sb = ag["score"], bg["score"]
        rows.append({
            "id": pid,
            "domain": ag["domain"],
            "grader_A_score": sa,
            "grader_B_score": sb,
            "mean": (sa + sb) / 2,
            "disagreement": abs(sa - sb),
        })
    return rows


def domain_means(rows: list[dict]) -> dict[str, float]:
    by_domain: dict[str, list[float]] = {}
    for g in rows:
        by_domain.setdefault(g["domain"], []).append(g["mean"])
    return {d: mean(scores) for d, scores in by_domain.items()}


def capability_breakdown(rows: list[dict]) -> dict[str, float]:
    old, new = [], []
    for g in rows:
        if g["domain"] != "capability":
            continue
        pid = g["id"]
        is_old = any(h in pid for h in V07_ANCHOR_HINTS)
        (old if is_old else new).append(g["mean"])
    return {
        "Cap_old12": mean(old) if old else float("nan"),
        "Cap_new18": mean(new) if new else float("nan"),
    }


def grader_health(rows: list[dict]) -> tuple[int, float]:
    agree = sum(1 for g in rows if g["disagreement"] <= 1)
    return agree, mean(g["disagreement"] for g in rows)


def gate(arm_means: dict[str, float],
         baseline_means: dict[str, float],
         trained_domains: list[str],
         tolerance: float) -> list[str]:
    failures = []
    for dom in trained_domains:
        a = arm_means.get(dom)
        b = baseline_means.get(dom)
        if a is None or b is None:
            failures.append(f"{dom}: missing in graded output (arm={a}, baseline={b})")
            continue
        delta = a - b
        if delta <= -tolerance:
            failures.append(
                f"{dom}: arm {a:.2f} vs baseline {b:.2f} "
                f"(delta {delta:+.2f}, tolerance -{tolerance:.2f}) — REGRESSION"
            )
    return failures


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, help="arm label, e.g. v2-stack-fp")
    ap.add_argument("--baseline", required=True,
                    help="baseline arm label (same session), e.g. v0.7-recheck")
    ap.add_argument("--session", required=True,
                    help="session date YYYY-MM-DD (must match both arms' graded files)")
    ap.add_argument("--trained-domains", nargs="+", required=True,
                    choices=["xstate", "fp", "reactive", "eventsourcing"],
                    help="domains to gate (only trained domains should be gated)")
    ap.add_argument("--tolerance", type=float, default=0.3,
                    help="halt if any trained domain delta <= -tolerance (default 0.3)")
    args = ap.parse_args()

    arm_a, arm_b = load_graded_pair(args.arm, args.session)
    base_a, base_b = load_graded_pair(args.baseline, args.session)

    arm_rows = combine_scores(arm_a, arm_b)
    base_rows = combine_scores(base_a, base_b)
    arm_means = domain_means(arm_rows)
    base_means = domain_means(base_rows)

    out = {
        "arm": args.arm,
        "baseline": args.baseline,
        "session": args.session,
        "trained_domains": args.trained_domains,
        "tolerance": args.tolerance,
        "arm_scores": arm_rows,
        "baseline_scores": base_rows,
        "arm_domain_means": arm_means,
        "baseline_domain_means": base_means,
    }
    out_path = RESULTS / args.arm / f"{args.session}.delta_gate.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))

    print(f"\n=== {args.arm} vs contemporaneous {args.baseline} (session {args.session}) ===")
    print(f"{'domain':<15} {'arm':>6} {'base':>6} {'delta':>8} {'gated':>8}")
    print("-" * 50)
    for dom in ["xstate", "fp", "reactive", "eventsourcing", "capability"]:
        a = arm_means.get(dom)
        b = base_means.get(dom)
        if a is None or b is None:
            continue
        delta = a - b
        is_gated = dom in args.trained_domains
        tag = "trained" if is_gated else "untrained"
        print(f"{dom:<15} {a:>6.2f} {b:>6.2f} {delta:>+8.2f} {tag:>8}")

    arm_agree, arm_mdis = grader_health(arm_rows)
    base_agree, base_mdis = grader_health(base_rows)
    print(f"\ngrader health:")
    print(f"  {args.arm} graders: agree={arm_agree}/{len(arm_rows)}, mean_disagreement={arm_mdis:.2f}")
    print(f"  {args.baseline} graders: agree={base_agree}/{len(base_rows)}, mean_disagreement={base_mdis:.2f}")

    cap = capability_breakdown(arm_rows)
    print(f"\n{args.arm} capability breakdown:")
    for k, v in cap.items():
        print(f"  {k}: {v:.2f}" if v == v else f"  {k}: (no records)")

    failures = gate(arm_means, base_means, args.trained_domains, args.tolerance)
    print(f"\n=== Delta gate (tolerance -{args.tolerance:.2f} on trained domains) ===")
    if failures:
        print("HALT — contemporaneous regression detected:")
        for msg in failures:
            print(f"  - {msg}")
        sys.exit(1)
    print("PASS — no contemporaneous regression on any trained domain.")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
