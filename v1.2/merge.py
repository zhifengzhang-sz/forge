#!/usr/bin/env python3
"""Phase 2 merge: delta-only training set for v1.2.

Plan (docs/training.process.md §Phase 2):
  Include: v1 ES batches + v1 *new* XState batches + 18 new anchors × 20 reps
  Exclude: v0.6 XState, v0.7 FP, v0.7 RX, v1 FP, v1 RX, 12 old anchors

v1/merge.py loaded v0.6 XState via `v0.6/data/synth.verified.jsonl` and
v0.7 FP/RX via `v0.7/data/synth.{fp,rx}.batch-*.jsonl`; we simply don't
load any of those here. v1's XState batches under v1/data/ are entirely
the NEW 702-pair set (v0.6's 440 live in v0.6/data/, not v1/data/).

Anchor partition: v0.7/seeds/anchors.jsonl defines the 12 old prompts.
Anything in v1/seeds/anchors.jsonl whose user prompt does not exactly
match one of those 12 is a new anchor.

Produces:
  v1.2/data/synth.verified.jsonl  (target ~2426 records)
"""
from __future__ import annotations
import json
import random
from collections import Counter
from pathlib import Path

ROOT = Path(".")
V1_DATA = ROOT / "v1/data"
V07_ANCHORS = ROOT / "v0.7/seeds/anchors.jsonl"
V1_ANCHORS = ROOT / "v1/seeds/anchors.jsonl"
OUT = ROOT / "v1.2/data/synth.verified.jsonl"

ANCHOR_REPS = 20  # 18 new × 20 = 360 records (plan §Phase 2)
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


def build_new_anchors() -> list[dict]:
    v07_prompts = {
        a["messages"][0]["content"].strip()
        for a in read_jsonl(V07_ANCHORS)
    }
    all_v1 = read_jsonl(V1_ANCHORS)
    new = [
        a for a in all_v1
        if a["messages"][0]["content"].strip() not in v07_prompts
    ]
    assert len(new) == 18, f"expected 18 new anchors, got {len(new)}"

    out: list[dict] = []
    for rec in new:
        rec = dict(rec)
        rec["domain"] = rec.get("domain", "capability")
        for _ in range(ANCHOR_REPS):
            out.append(rec.copy())
    return out


def main() -> None:
    parts: list[tuple[str, list[dict]]] = []

    v1_es = sorted(V1_DATA.glob("synth.es.batch-*.verified.jsonl"))
    for p in v1_es:
        parts.append((f"v1 ES {p.name}", stamp_domain(read_jsonl(p), "es")))

    v1_xstate = sorted(V1_DATA.glob("synth.xstate.batch-*.verified.jsonl"))
    for p in v1_xstate:
        parts.append((f"v1 XState {p.name}", stamp_domain(read_jsonl(p), "xstate")))

    combined: list[dict] = []
    for name, recs in parts:
        combined.extend(recs)
        print(f"  {name}: {len(recs)} records")

    print(f"\nTotal pre-dedup: {len(combined)}")
    combined = dedup(combined)
    print(f"Total post-dedup: {len(combined)}")

    anchors = build_new_anchors()
    combined.extend(anchors)
    print(f"New anchors appended: {len(anchors)} (18 unique × {ANCHOR_REPS} reps)")

    dc = Counter(r.get("domain", "unknown") for r in combined)
    print(f"\nPer-domain breakdown:")
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
