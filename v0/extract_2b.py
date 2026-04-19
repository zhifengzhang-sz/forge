"""Phase 2b: mechanical extraction of XState v5 usage examples.

Walks repos/xstate/examples/ for .ts files containing setup() or createMachine.
Applies one hand-written instruction template (varied by example folder name)
to each completion. Style intentionally matches the curated arm: derived from
the example directory.

Per review.v0.md: 'You're testing the data source, not the instruction generator.'
"""

import json
import re
from pathlib import Path

REPO = Path("/home/zzhang/dev/ai/models/forge/repos/xstate/examples")
OUT = Path(__file__).parent / "data" / "xstate_extracted.jsonl"
OUT.parent.mkdir(exist_ok=True)

SKIP_DIRS = {"node_modules", "dist", "build", ".git"}
SKIP_SUFFIXES = {".d.ts", ".test.ts", ".spec.ts"}


def prettify(name: str) -> str:
    s = re.sub(r"^[0-9]+-?", "", name)
    s = s.replace("-", " ").replace("_", " ")
    return s.strip()


def is_target(text: str) -> bool:
    return ("setup(" in text or "createMachine(" in text) and "from 'xstate'" in text


def make_instruction(example_dir: str, file_path: Path) -> str:
    pretty = prettify(example_dir)
    rel = file_path.name
    # Vary the phrasing slightly based on file type so it's not 100% identical
    if "machine" in rel.lower():
        return f"Write an idiomatic XState v5 machine in TypeScript for the following: {pretty}."
    if "logic" in rel.lower() or "actor" in rel.lower():
        return f"Write the XState v5 actor logic in TypeScript for: {pretty}."
    return f"Implement the XState v5 state machine in TypeScript for: {pretty}."


def main() -> None:
    examples = []
    for example_dir in sorted(REPO.iterdir()):
        if not example_dir.is_dir():
            continue
        for ts in example_dir.rglob("*.ts"):
            if any(part in SKIP_DIRS for part in ts.parts):
                continue
            if any(str(ts).endswith(suffix) for suffix in SKIP_SUFFIXES):
                continue
            try:
                text = ts.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            if not is_target(text):
                continue
            if len(text) > 8000 or len(text) < 100:
                continue
            instruction = make_instruction(example_dir.name, ts)
            examples.append({
                "messages": [
                    {"role": "user", "content": instruction},
                    {"role": "assistant", "content": "```typescript\n" + text.strip() + "\n```"},
                ],
                "_meta": {
                    "example": example_dir.name,
                    "file": str(ts.relative_to(REPO)),
                    "chars": len(text),
                },
            })

    print(f"Found {len(examples)} candidate examples.")

    # Trim to ~50: keep diverse examples (one file per example dir preferred)
    seen_dirs: set[str] = set()
    primary = []
    secondary = []
    for ex in examples:
        d = ex["_meta"]["example"]
        if d in seen_dirs:
            secondary.append(ex)
        else:
            primary.append(ex)
            seen_dirs.add(d)

    # Take up to 50: primary (one-per-dir) first, then top up with secondary
    target = 50
    selected = primary[:target] + secondary[: max(0, target - len(primary))]
    selected = selected[:target]

    print(f"Selected {len(selected)} examples covering {len({s['_meta']['example'] for s in selected})} dirs.")

    with OUT.open("w") as f:
        for ex in selected:
            payload = {"messages": ex["messages"]}
            f.write(json.dumps(payload) + "\n")
    print(f"Wrote {OUT}")

    # Sanity: dump distribution
    from collections import Counter
    c = Counter(s["_meta"]["example"] for s in selected)
    print(f"\nDirs sampled (top 10): {c.most_common(10)}")


if __name__ == "__main__":
    main()
