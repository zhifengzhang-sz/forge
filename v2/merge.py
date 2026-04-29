#!/usr/bin/env python3
"""v2.0 merge: atomic-drill reuse + v1 ES + reduced anchors.

Fork of v1.2/merge.py with sources swapped per docs/v2.plan.md
"Build script":
  Include: v0.6 XState (440) + v0.7 FP A-H (320, 8-batch atomic drill)
         + v0.7 RX A-F (240) + v1 ES D* (1364, 37-aggregate atomic drill)
         + 30 anchors x 8 reps (240)
  Exclude: v1 FP (874, compositional — caused v1 FP regression)
           v1 RX (345, mixed — small dip)
           v1 XState (702, mixed — marginal +0.10 gain)

Target: ~2604 records in v2/data/synth.verified.jsonl.

Anchor reps dropped from v1's 20 -> 8 per v1/decision.md recommendation
(8-10% anchor ratio). 240 / 2604 = 9.2%.
"""
from __future__ import annotations
import json
import random
from collections import Counter
from pathlib import Path

ROOT = Path(".")
V06_XSTATE = ROOT / "v0.6/data/synth.verified.jsonl"
V07_FP_BATCHES = sorted((ROOT / "v0.7/data").glob("synth.fp.batch-*.jsonl"))
V07_RX_BATCHES = sorted((ROOT / "v0.7/data").glob("synth.rx.batch-*.jsonl"))
V1_ES_BATCHES = sorted((ROOT / "v1/data").glob("synth.es.batch-*.verified.jsonl"))
ANCHORS = ROOT / "v1/seeds/anchors.jsonl"
OUT = ROOT / "v2/data/synth.verified.jsonl"

ANCHOR_REPS = 8  # 30 unique x 8 = 240 records (plan: 8% anchor ratio)
SEED = 42


def read_jsonl(path: Path) -> list[dict]:
    recs = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"  skip malformed line in {path}")
    return recs


def stamp_domain(recs: list[dict], domain: str) -> list[dict]:
    for r in recs:
        if "domain" not in r:
            r["domain"] = domain
    return recs


def dedup(recs: list[dict]) -> list[dict]:
    seen: set[str] = set()
    out: list[dict] = []
    for r in recs:
        msgs = r.get("messages", [])
        if len(msgs) < 2:
            continue
        key = (msgs[0].get("content", "") + "|" + msgs[1].get("content", "")).strip()
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def build_anchors() -> list[dict]:
    base = read_jsonl(ANCHORS)
    assert len(base) == 30, f"expected 30 anchors, got {len(base)}"
    out: list[dict] = []
    for rec in base:
        rec = dict(rec)
        rec["domain"] = rec.get("domain", "capability")
        for _ in range(ANCHOR_REPS):
            out.append(rec.copy())
    return out


def main() -> None:
    parts: list[tuple[str, list[dict]]] = []

    parts.append(("v0.6 XState reuse", stamp_domain(read_jsonl(V06_XSTATE), "xstate")))

    v07_fp: list[dict] = []
    for p in V07_FP_BATCHES:
        v07_fp.extend(read_jsonl(p))
    parts.append(("v0.7 FP reuse (A-H)", stamp_domain(v07_fp, "fp")))

    v07_rx: list[dict] = []
    for p in V07_RX_BATCHES:
        v07_rx.extend(read_jsonl(p))
    parts.append(("v0.7 RX reuse (A-F)", stamp_domain(v07_rx, "reactive")))

    for p in V1_ES_BATCHES:
        parts.append((f"v1 ES {p.name}", stamp_domain(read_jsonl(p), "eventsourcing")))

    combined: list[dict] = []
    for name, recs in parts:
        combined.extend(recs)
        print(f"  {name}: {len(recs)} records")

    print(f"\nTotal pre-dedup: {len(combined)}")
    combined = dedup(combined)
    print(f"Total post-dedup: {len(combined)}")

    anchors = build_anchors()
    combined.extend(anchors)
    print(f"Anchors appended: {len(anchors)} (30 unique x {ANCHOR_REPS} reps)")

    dc = Counter(r.get("domain", "unknown") for r in combined)
    print(f"\nPer-domain breakdown:")
    for dom, n in sorted(dc.items(), key=lambda kv: -kv[1]):
        print(f"  {dom:<14s} {n}")

    rng = random.Random(SEED)
    rng.shuffle(combined)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as fh:
        for rec in combined:
            fh.write(json.dumps(rec, ensure_ascii=False))
            fh.write("\n")

    print(f"\nWrote {OUT} ({len(combined)} records)")


if __name__ == "__main__":
    main()
