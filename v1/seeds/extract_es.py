#!/usr/bin/env python3
"""Extract ES reference examples from repos/EventSourcing.NodeJS.

Output: v1/seeds/reference_examples.es.jsonl in the v0.7 XState reference
format: {"messages":[{"role":"user","content":"<prompt>"},{"role":"assistant","content":"```typescript\\n<code>\\n```"}]}.

Selection criteria:
- TypeScript file under samples/*/src/
- Size 150-6000 chars (matches verify.py length gate)
- NOT a test file, barrel (index.ts), or top-level bootstrap (app.ts)
- Contains at least one ES idiom token (evolve/decide/when/aggregateStream/...)
- Skip files whose proper nouns overlap with eval prompt aggregates
  (ShoppingCart, BankAccount, CartSummary) to preempt contamination risk
"""
from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Iterable

ROOT = Path("repos/EventSourcing.NodeJS/samples")
SAMPLES = [
    "foundations",
    "snapshots",
    "hotelManagement",
    "closingTheBooks",
    "from_crud_to_eventsourcing",
]

MIN_LEN = 150
MAX_LEN = 6000
TARGET_COUNT = 30

IDIOM_RE = re.compile(
    r"\b(evolve|decide|Decider|CommandHandler|EventStore|Snapshot|"
    r"expectedRevision|appendToStream|readStream|when|aggregateStream|"
    r"projection|aggregate|readFromStream|EventStoreDB)\b"
)

SKIP_NAMES = {"index.ts", "app.ts"}

# Proper nouns used in eval prompts — avoid seeds that lead with them
EVAL_BLOCKLIST = {"ShoppingCart", "BankAccount", "CartSummary", "CartCreated",
                  "ItemAdded", "ItemRemoved", "Deposit", "Withdraw"}


def eligible(path: Path) -> bool:
    if path.suffix != ".ts":
        return False
    name = path.name
    if name in SKIP_NAMES:
        return False
    if ".test." in name or ".spec." in name or path.parts and any(
        p in ("e2e", "testing", "__tests__") for p in path.parts
    ):
        return False
    return True


def describe(path: Path) -> str:
    """Derive a concept label from the file path."""
    stem = path.stem
    parts = [p for p in path.parts if p not in ("samples", "src", "core")]
    trail = "/".join(parts[-3:])
    return f"{stem} ({trail})"


def scan(root: Path) -> Iterable[Path]:
    for sample in SAMPLES:
        base = root / sample / "src"
        if not base.exists():
            continue
        for p in base.rglob("*.ts"):
            if eligible(p):
                yield p


def classify(path: Path, code: str) -> str | None:
    """Return the ES concept category if accepted, else None."""
    if not (MIN_LEN <= len(code) <= MAX_LEN):
        return None
    if not IDIOM_RE.search(code):
        return None
    # Block if file stem matches an eval-prompt proper noun
    stem_lower = path.stem.lower()
    for blocked in EVAL_BLOCKLIST:
        if blocked.lower() in stem_lower:
            return None
    return describe(path)


def build_pair(path: Path, code: str, concept: str) -> dict:
    prompt = (
        f"Write an idiomatic event-sourcing TypeScript example for: "
        f"{concept}."
    )
    assistant = "```typescript\n" + code.rstrip() + "\n```"
    return {"messages": [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": assistant},
    ]}


def main() -> None:
    out_path = Path("v1/seeds/reference_examples.es.jsonl")
    records: list[dict] = []
    seen_stems: set[str] = set()
    for path in scan(ROOT):
        # De-duplicate by stem to get varied coverage instead of
        # three near-identical index-readers, etc.
        if path.stem in seen_stems:
            continue
        try:
            code = path.read_text()
        except (OSError, UnicodeDecodeError):
            continue
        concept = classify(path, code)
        if concept is None:
            continue
        records.append(build_pair(path, code, concept))
        seen_stems.add(path.stem)
        if len(records) >= TARGET_COUNT:
            break

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False))
            fh.write("\n")

    print(f"extracted {len(records)} ES reference examples -> {out_path}")


if __name__ == "__main__":
    main()
