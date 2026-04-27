#!/usr/bin/env python3
"""Generate Figure 1: FP score vs effective LoRA scaler, vanilla and rsLoRA arms.

Reads grader outputs from results/B1/...B6/, computes per-arm domain means,
and plots FP and XState as a function of effective scaler with rule-distinct
markers.
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
SESSION = "2026-04-26a"

ARMS = {
    "B1": dict(alpha=64,  rule="vanilla", scaler=0.50),
    "B2": dict(alpha=128, rule="vanilla", scaler=1.00),
    "B3": dict(alpha=256, rule="vanilla", scaler=2.00),
    "B4": dict(alpha=64,  rule="rsLoRA",  scaler=5.66),
    "B5": dict(alpha=128, rule="rsLoRA",  scaler=11.31),
    "B6": dict(alpha=256, rule="rsLoRA",  scaler=22.63),
}


def load_means(arm):
    a = json.loads((REPO / f"results/{arm}/{SESSION}.graded.grader_A.json").read_text())
    b = json.loads((REPO / f"results/{arm}/{SESSION}.graded.grader_B.json").read_text())
    by_domain = {}
    for data in (a, b):
        for g in data["graded"]:
            by_domain.setdefault(g["domain"], []).append(g["score"])
    return {d: sum(s) / len(s) for d, s in by_domain.items()}


def main():
    rows = []
    for arm, cfg in ARMS.items():
        m = load_means(arm)
        rows.append(dict(arm=arm, **cfg, **m))

    fig, ax = plt.subplots(figsize=(7, 4.6))

    for rule, marker in [("vanilla", "o"), ("rsLoRA", "s")]:
        sub = [r for r in rows if r["rule"] == rule]
        sub.sort(key=lambda r: r["scaler"])
        x = [r["scaler"] for r in sub]
        fp = [r["fp"] for r in sub]
        xs = [r["xstate"] for r in sub]
        ax.plot(x, fp, marker + "-", color="tab:blue", label=f"FP, {rule}", linewidth=2, markersize=9)
        ax.plot(x, xs, marker + "--", color="tab:red", label=f"XState, {rule}", linewidth=2, markersize=9)

    # annotate arm labels at each point
    for r in rows:
        ax.annotate(r["arm"], (r["scaler"], r["fp"]), textcoords="offset points",
                    xytext=(6, -3), fontsize=8, color="tab:blue")

    ax.set_xscale("log")
    ax.set_xlabel(r"Effective LoRA scaler  ($\alpha/r$ or $\alpha/\sqrt{r}$)", fontsize=11)
    ax.set_ylabel("Held-out domain mean (1–5)", fontsize=11)
    ax.set_title("FP vs XState across the alpha-scaling sweep, $r{=}128$", fontsize=12)
    ax.set_ylim(2.5, 5.2)
    ax.grid(True, alpha=0.3)
    ax.axhline(5.0, color="gray", linewidth=0.5, linestyle=":")
    ax.legend(loc="lower right", framealpha=0.9, fontsize=9)

    out = REPO / "paper/figures/fp_xstate_vs_scaler.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    print(f"Wrote {out}")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=150)
    print(f"Wrote {out.with_suffix('.png')}")


if __name__ == "__main__":
    main()
