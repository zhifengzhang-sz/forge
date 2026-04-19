"""Combine two grader outputs per arm into final v0.7 results with uncertainty bands.

Reads:
  v0/results/v0.7-{rank}.graded.grader_A.json
  v0/results/v0.7-{rank}.graded.grader_B.json

Writes:
  v0/results/v0.7-{rank}.json  -- combined with grader_A_score, grader_B_score, mean, disagreement

Also prints per-domain means and the comparison vs base/v0.6/claude_opus.
"""

import json
from pathlib import Path
from statistics import mean

V0 = Path("/home/zzhang/dev/ai/models/forge/v0/results")


def combine_arm(rank: int) -> dict:
    a = json.loads((V0 / f"v0.7-r{rank}.graded.grader_A.json").read_text())
    b = json.loads((V0 / f"v0.7-r{rank}.graded.grader_B.json").read_text())
    a_by_id = {g["id"]: g for g in a["graded"]}
    b_by_id = {g["id"]: g for g in b["graded"]}
    combined = []
    for pid in sorted(a_by_id):
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
        "model": a.get("model", f"ts-forge-v0.7-r{rank}"),
        "arm": f"v0.7-r{rank}",
        "graders": ["A", "B"],
        "graded": combined,
    }
    (V0 / f"v0.7-r{rank}.json").write_text(json.dumps(out, indent=2))
    return out


def domain_mean(combined: dict, domain: str) -> tuple[float, float, float]:
    rows = [g for g in combined["graded"] if g["domain"] == domain]
    if not rows:
        return 0.0, 0.0, 0.0
    return (
        mean(g["grader_A_score"] for g in rows),
        mean(g["grader_B_score"] for g in rows),
        mean(g["mean"] for g in rows),
    )


def summary(combined: dict) -> None:
    arm = combined["arm"]
    print(f"\n=== {arm} ===")
    print(f"{'domain':<15} {'A':>6} {'B':>6} {'mean':>6} {'disagree':>10}")
    print("-" * 50)
    rows = combined["graded"]
    for d in ["xstate", "fp", "reactive", "eventsourcing", "capability"]:
        dr = [g for g in rows if g["domain"] == d]
        if not dr:
            continue
        a = mean(g["grader_A_score"] for g in dr)
        b = mean(g["grader_B_score"] for g in dr)
        m = mean(g["mean"] for g in dr)
        disagree = mean(g["disagreement"] for g in dr)
        print(f"{d:<15} {a:>6.2f} {b:>6.2f} {m:>6.2f} {disagree:>10.2f}")
    non_cap = [g for g in rows if g["domain"] != "capability"]
    domain_avg = mean(g["mean"] for g in non_cap)
    overall_disagreement = mean(g["disagreement"] for g in rows)
    print(f"{'domain_avg':<15} {'':>6} {'':>6} {domain_avg:>6.2f}")
    print(f"grader agreement (within 1 pt): "
          f"{sum(1 for g in rows if g['disagreement'] <= 1)}/{len(rows)}")
    print(f"overall mean disagreement: {overall_disagreement:.2f}")


def main() -> None:
    r32 = combine_arm(32)
    r64 = combine_arm(64)
    summary(r32)
    summary(r64)

    # Cross-arm comparison
    print("\n=== Comparison vs prior arms ===")
    print(f"{'arm':<24}{'domain_avg':>12}{'cap_avg':>10}{'xstate':>10}{'fp':>10}{'rx':>10}{'es':>10}")
    for path, label in [
        ("base.json", "base (qwen3:14b)"),
        ("curated.json", "v0 curated"),
        ("extracted.json", "v0 extracted"),
        ("v0.6.json", "v0.6"),
        ("v0.7-r32.json", "v0.7 r=32"),
        ("v0.7-r64.json", "v0.7 r=64 (winner)"),
        ("claude_opus.json", "claude-opus-4-7"),
    ]:
        try:
            data = json.loads((V0 / path).read_text())["graded"]
        except FileNotFoundError:
            continue
        def avg(domain: str | None) -> float:
            scores = []
            for g in data:
                if g.get("domain") == domain or (domain is None and g.get("domain") != "capability"):
                    s = g.get("mean", g.get("score"))
                    if s is not None:
                        scores.append(s)
            return mean(scores) if scores else 0.0
        d_avg = avg(None)
        c_avg = avg("capability")
        xs = avg("xstate")
        fp = avg("fp")
        rx = avg("reactive")
        es = avg("eventsourcing")
        print(f"{label:<24}{d_avg:>12.2f}{c_avg:>10.2f}{xs:>10.2f}{fp:>10.2f}{rx:>10.2f}{es:>10.2f}")


if __name__ == "__main__":
    main()
