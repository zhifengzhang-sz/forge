#!/usr/bin/env python3
"""v3.2 stacking corpus: FP-only training data for the new (stacked) adapter.

The frozen base adapter (v3.0-rslora) already encodes XState/RX/ES strength.
This new adapter is trained ONLY on FP records — its job is to add an FP
correction layer on top of the frozen base, without re-touching other domains.

Sources:
  v0.7/data/synth.fp.batch-{A..H}.jsonl              (320 v0.7 atomic FP)
  v3.1/data/synth.fp.batch-{I..O}.verified.jsonl     (278 new pattern fills)
  v1/seeds/anchors.jsonl × 2 reps                    (60 anchor records)

Total target: ~658 records. Anchors are minimized to avoid teaching
non-FP capability through the new adapter. The anchors that remain prevent
the LoRA from drifting on capability prompts during FP training.

Output: v3.2/data/synth.fp_only.jsonl
"""
from __future__ import annotations
import json
import random
from collections import Counter
from pathlib import Path

ROOT = Path(".")
V07_FP_BATCHES = sorted((ROOT / "v0.7/data").glob("synth.fp.batch-*.jsonl"))
V31_FP_NEW = sorted((ROOT / "v3.1/data").glob("synth.fp.batch-*.verified.jsonl"))
ANCHORS = ROOT / "v1/seeds/anchors.jsonl"
OUT = ROOT / "v3.2/data/synth.fp_only.jsonl"

ANCHOR_REPS = 2  # 30 × 2 = 60 records (~9% of FP-only corpus)
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
    out: list[dict] = []
    for rec in base:
        rec = dict(rec)
        rec["domain"] = rec.get("domain", "capability")
        for _ in range(ANCHOR_REPS):
            out.append(rec.copy())
    return out


def main() -> None:
    parts: list[tuple[str, list[dict]]] = []

    v07_fp: list[dict] = []
    for p in V07_FP_BATCHES:
        v07_fp.extend(read_jsonl(p))
    parts.append(("v0.7 FP A-H (8 patterns)", stamp_domain(v07_fp, "fp")))

    v31_fp: list[dict] = []
    for p in V31_FP_NEW:
        v31_fp.extend(read_jsonl(p))
    parts.append(("v3.1 FP I-O (7 missing patterns)", stamp_domain(v31_fp, "fp")))

    combined: list[dict] = []
    for name, recs in parts:
        combined.extend(recs)
        print(f"  {name}: {len(recs)} records")

    print(f"\nFP pre-dedup: {len(combined)}")
    combined = dedup(combined)
    print(f"FP post-dedup: {len(combined)}")

    anchors = build_anchors()
    combined.extend(anchors)
    print(f"Anchors appended: {len(anchors)} (30 unique x {ANCHOR_REPS} reps)")

    dc = Counter(r.get("domain", "unknown") for r in combined)
    print("\nPer-domain breakdown:")
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
