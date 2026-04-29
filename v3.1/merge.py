#!/usr/bin/env python3
"""v3.1 merge: v3.0 corpus + 7 new FP atomic-drill batches filling the
missing patterns from `v0.7/seeds/patterns.fp.md` §2/4/6/8/10/12/14.

Sources (relative to repo root):
  v0.6/data/synth.verified.jsonl                    — 440 XState v0.6
  v0.7/data/synth.fp.batch-{A..H}.jsonl              — 320 v0.7 FP (8 covered patterns)
  v0.7/data/synth.rx.batch-*.jsonl                   — 240 v0.7 RX
  v1/data/synth.es.batch-D*.verified.jsonl           — 1364 v1 ES
  v3.1/data/synth.fp.batch-{I..O}.verified.jsonl     — 278 NEW FP (7 missing patterns)
  v1/seeds/anchors.jsonl × 8 reps                    — 240 anchors

Target: ~2882 records in v3.1/data/synth.verified.jsonl.

The 7 new FP batches close the data gaps documented in v3.0-rslora's
post-mortem (fp-04 collapse on TE.catchAll, fp-02/05 on Layer
composition). With Pattern 4 + Pattern 2 explicitly drilled, rsLoRA
scaling no longer overfits sparse FP signal.
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
V31_FP_NEW = sorted((ROOT / "v3.1/data").glob("synth.fp.batch-*.verified.jsonl"))
ANCHORS = ROOT / "v1/seeds/anchors.jsonl"
OUT = ROOT / "v3.1/data/synth.verified.jsonl"

ANCHOR_REPS = 8
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
    parts.append(("v0.7 FP (A-H, 8 covered patterns)", stamp_domain(v07_fp, "fp")))

    v31_fp: list[dict] = []
    for p in V31_FP_NEW:
        v31_fp.extend(read_jsonl(p))
    parts.append(("v3.1 FP (I-O, 7 missing patterns)", stamp_domain(v31_fp, "fp")))

    v07_rx: list[dict] = []
    for p in V07_RX_BATCHES:
        v07_rx.extend(read_jsonl(p))
    parts.append(("v0.7 RX (A-F)", stamp_domain(v07_rx, "reactive")))

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
