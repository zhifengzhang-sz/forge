#!/usr/bin/env python3
"""Phase D merge: concatenate all verified v1 batches + reused v0.6/v0.7
pairs + 30-unique anchor expansion (20 reps each = 600 anchor records).

Produces:
  v1/data/synth.raw.jsonl         (pre-merge concat, no anchor expansion)
  v1/data/synth.verified.jsonl    (final with anchor expansion + domain tags)

Does NOT re-run the tsc verifier; all inputs already passed verify.py at
batch time. This script trusts those gates and focuses on:
  - stamping `domain` on v0.6/v0.7 records that lack one
  - de-duplicating (by exact message match)
  - anchor expansion (30 unique anchors × 20 reps = 600 records)
  - shuffling with a fixed seed for deterministic ordering
  - final domain-count report
"""
from __future__ import annotations
import json
import random
from pathlib import Path

ROOT = Path(".")
V1_DATA = ROOT / "v1/data"
OUT_RAW = V1_DATA / "synth.raw.jsonl"
OUT_VERIFIED = V1_DATA / "synth.verified.jsonl"
ANCHORS = ROOT / "v1/seeds/anchors.jsonl"
ANCHOR_REPS = 20  # 30 unique × 20 = 600 records

# v0.6 XState: 440 pairs, reused as-is (already domain-tagged? check)
V06_XSTATE = ROOT / "v0.6/data/synth.verified.jsonl"
# v0.7 FP: 8 batch files
V07_FP_BATCHES = sorted((ROOT / "v0.7/data").glob("synth.fp.batch-*.jsonl"))
# v0.7 RX: 6 batch files
V07_RX_BATCHES = sorted((ROOT / "v0.7/data").glob("synth.rx.batch-*.jsonl"))
# v0.7 combined
V07_COMBINED = ROOT / "v0.7/data/synth.verified.jsonl"

# v1 batches
V1_ES = sorted(V1_DATA.glob("synth.es.batch-*.verified.jsonl"))
V1_XSTATE = sorted(V1_DATA.glob("synth.xstate.batch-*.verified.jsonl"))
V1_FP = sorted(V1_DATA.glob("synth.fp.batch-*.verified.jsonl"))
V1_RX = sorted(V1_DATA.glob("synth.rx.batch-*.verified.jsonl"))

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
    """Remove exact duplicates by message content hash."""
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
    """Load 30 unique anchors and replicate each ANCHOR_REPS times."""
    base = read_jsonl(ANCHORS)
    out = []
    for rec in base:
        rec = dict(rec)
        rec["domain"] = rec.get("domain", "capability")
        for _ in range(ANCHOR_REPS):
            out.append(rec.copy())
    return out


def main() -> None:
    parts: list[tuple[str, list[dict]]] = []

    # v0.6 XState reuse (domain: xstate)
    v06 = read_jsonl(V06_XSTATE)
    v06 = stamp_domain(v06, "xstate")
    parts.append(("v0.6 XState reuse", v06))

    # v0.7 FP + RX (already have domain field in most; stamp missing)
    v07_fp = []
    for p in V07_FP_BATCHES:
        v07_fp.extend(read_jsonl(p))
    v07_fp = stamp_domain(v07_fp, "fp")
    parts.append(("v0.7 FP reuse", v07_fp))

    v07_rx = []
    for p in V07_RX_BATCHES:
        v07_rx.extend(read_jsonl(p))
    v07_rx = stamp_domain(v07_rx, "rx")
    parts.append(("v0.7 RX reuse", v07_rx))

    # v1 new batches
    for p in V1_ES:
        recs = stamp_domain(read_jsonl(p), "es")
        parts.append((f"v1 ES {p.name}", recs))
    for p in V1_XSTATE:
        recs = stamp_domain(read_jsonl(p), "xstate")
        parts.append((f"v1 XState {p.name}", recs))
    for p in V1_FP:
        recs = stamp_domain(read_jsonl(p), "fp")
        parts.append((f"v1 FP {p.name}", recs))
    for p in V1_RX:
        recs = stamp_domain(read_jsonl(p), "rx")
        parts.append((f"v1 RX {p.name}", recs))

    # Concatenate NON-anchor sources (dedup applies here)
    combined: list[dict] = []
    for name, recs in parts:
        combined.extend(recs)
        print(f"  {name}: {len(recs)} records")

    print(f"\nTotal pre-dedup: {len(combined)}")
    combined = dedup(combined)
    print(f"Total post-dedup: {len(combined)}")

    # Anchor expansion — append AFTER dedup so the 600 replicas stay
    # intact (intentional over-representation of capability anchors).
    anchors = build_anchors()
    combined.extend(anchors)
    print(f"Anchors appended: {len(anchors)} (30 unique × {ANCHOR_REPS} reps)")

    # Domain breakdown
    from collections import Counter
    dc = Counter(r.get("domain", "unknown") for r in combined)
    print(f"\nPer-domain breakdown:")
    for dom, n in sorted(dc.items(), key=lambda kv: -kv[1]):
        print(f"  {dom:<12s} {n}")

    # Shuffle deterministically
    rng = random.Random(SEED)
    rng.shuffle(combined)

    # Write raw (same content — raw and verified are identical here since
    # all inputs already passed verify; we keep both names for parity with
    # v0.6/v0.7 pipeline convention)
    with OUT_RAW.open("w", encoding="utf-8") as fh:
        for rec in combined:
            fh.write(json.dumps(rec, ensure_ascii=False))
            fh.write("\n")
    with OUT_VERIFIED.open("w", encoding="utf-8") as fh:
        for rec in combined:
            fh.write(json.dumps(rec, ensure_ascii=False))
            fh.write("\n")

    print(f"\nWrote {OUT_RAW}")
    print(f"Wrote {OUT_VERIFIED}")


if __name__ == "__main__":
    main()
