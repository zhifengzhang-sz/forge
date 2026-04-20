#!/usr/bin/env python3
"""Contamination audit: diff ES seed examples against ES eval prompts.

Fails with non-zero exit if any eval prompt's distinctive proper nouns
appear verbatim in any seed example's code. Reports overlap rates.

Distinctive proper nouns are extracted from each eval prompt by picking
identifiers that are:
  - CamelCase (Class/Type names) or
  - appearing in quoted strings (event type literals)
and removing generic domain words (Decider, CommandHandler, EventStore,
readStream, etc.) that are legitimately shared with the ES pattern surface.

Requires: reference_examples.es.jsonl in the same dir.
"""
from __future__ import annotations
import json
import re
import sys
from pathlib import Path

SEEDS_PATH = Path("v1/seeds/reference_examples.es.jsonl")
EVAL_PATH = Path("v0/eval/v0.json")

# Generic ES vocabulary that is intentionally shared — NOT contamination.
GENERIC = {
    "Decider", "CommandHandler", "EventStore", "Event", "Command", "State",
    "readStream", "appendToStream", "evolve", "decide", "aggregate",
    "Snapshot", "Projector", "project", "loadAggregate", "expectedVersion",
    "Aggregate", "View", "Version", "Promise", "Date", "Array", "Map",
    "Set", "Error", "Readonly", "Record", "Omit", "Pick", "Partial",
    "Required", "ReturnType", "Parameters",
}

CAMEL_RE = re.compile(r"\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b")
QUOTED_RE = re.compile(r"'([^']+)'|\"([^\"]+)\"")


def extract_distinctive(text: str) -> set[str]:
    toks: set[str] = set()
    for m in CAMEL_RE.findall(text):
        if m not in GENERIC:
            toks.add(m)
    for lit in QUOTED_RE.findall(text):
        s = lit[0] or lit[1]
        if s and s[0].isupper() and s not in GENERIC:
            toks.add(s)
    return toks


def load_eval_prompts() -> list[dict]:
    data = json.loads(EVAL_PATH.read_text())
    return [p for p in data["prompts"] if p["domain"] == "eventsourcing"]


def load_seed_codes() -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    with SEEDS_PATH.open() as fh:
        for line in fh:
            rec = json.loads(line)
            user_content = rec["messages"][0]["content"]
            asst_content = rec["messages"][1]["content"]
            out.append((user_content, asst_content))
    return out


def main() -> int:
    prompts = load_eval_prompts()
    seeds = load_seed_codes()

    print(f"eval ES prompts: {len(prompts)}")
    print(f"seed ES examples: {len(seeds)}")
    print()

    failures: list[tuple[str, str, set[str]]] = []
    per_prompt_stats: list[tuple[str, set[str], int, list[str]]] = []

    for p in prompts:
        prompt_text = p["prompt"] + " " + " ".join(p.get("must_have", []))
        distinctive = extract_distinctive(prompt_text)
        if not distinctive:
            per_prompt_stats.append((p["id"], set(), 0, []))
            continue
        hits_by_seed: list[tuple[str, set[str]]] = []
        for label, asst in seeds:
            overlap = {t for t in distinctive if t in asst}
            if overlap:
                hits_by_seed.append((label, overlap))
        overlap_rate = len(hits_by_seed) / len(seeds) * 100
        per_prompt_stats.append((p["id"], distinctive, len(hits_by_seed), [s[0] for s in hits_by_seed[:3]]))
        for label, overlap in hits_by_seed:
            # Flag any seed where >= 50% of the distinctive tokens appear
            if len(overlap) / len(distinctive) >= 0.5:
                failures.append((p["id"], label, overlap))

    print("Per-eval-prompt distinctive-token analysis:")
    for pid, tokens, hit_count, examples in per_prompt_stats:
        tok_preview = ", ".join(sorted(tokens)[:6]) if tokens else "(none)"
        print(f"  {pid}: tokens={tok_preview}  seeds_containing_any={hit_count}/{len(seeds)}")
        if examples:
            for ex in examples:
                print(f"    - {ex[:80]}")

    print()
    if failures:
        print(f"CONTAMINATION FAILURES ({len(failures)} seed/prompt pairs with \u22650.5 token overlap):")
        for pid, label, overlap in failures:
            print(f"  {pid} <-> {label}: {overlap}")
        return 1

    print("PASS: no seed shares >=50% of any eval prompt's distinctive proper nouns.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
