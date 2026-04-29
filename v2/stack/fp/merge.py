#!/usr/bin/env python3
"""v2.0 (stacked curriculum) merge — FP-only corpus for the canary arm.

Sources:
  v0.7/data/synth.fp.batch-{A..H}.jsonl  (320 FP atomic records)
  v1/seeds/anchors.jsonl                  (30 anchors x 8 reps = 240)

Output:
  v2/stack/fp/data/synth.verified.jsonl   (560 records)

Canary purpose: fresh-from-base single-LoRA training on FP-only data
must reach FP >= 4.40 (v0.7's baseline). This validates:
  (a) the stacking pipeline infrastructure is correct when there are
      no prior frozen adapters to layer on, and
  (b) the FP data + recipe reproduce v0.7's FP quality when trained
      in isolation (no cross-domain gradient competition).

If this arm fails, the v2.x stacked curriculum cannot proceed on
FP's foundation. Debug infrastructure or data before adding RX.
"""
from __future__ import annotations
import json
import random
from collections import Counter
from pathlib import Path

ROOT = Path(".")
V07_FP_BATCHES = sorted((ROOT / "v0.7/data").glob("synth.fp.batch-*.jsonl"))
ANCHORS = ROOT / "v1/seeds/anchors.jsonl"
OUT = ROOT / "v2/stack/fp/data/synth.verified.jsonl"

ANCHOR_REPS = 8  # 30 x 8 = 240 (match v2.0-r64 ratio: 240 anchors / 560 = 43%)
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
    v07_fp: list[dict] = []
    for p in V07_FP_BATCHES:
        v07_fp.extend(read_jsonl(p))
    v07_fp = stamp_domain(v07_fp, "fp")
    print(f"  v0.7 FP (A-H): {len(v07_fp)} records")

    combined = dedup(v07_fp)
    print(f"  post-dedup: {len(combined)}")

    anchors = build_anchors()
    combined.extend(anchors)
    print(f"  + anchors: {len(anchors)} (30 x {ANCHOR_REPS} reps)")

    dc = Counter(r.get("domain", "unknown") for r in combined)
    print(f"\nPer-domain:")
    for dom, n in sorted(dc.items(), key=lambda kv: -kv[1]):
        print(f"  {dom:<12s} {n}")

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
