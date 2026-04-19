#!/usr/bin/env python3
"""Phase B verifier for v0.6 synthesised XState v5 pairs.

Runs each (instruction, completion) pair through three gates:

  1. Compiles under strict TypeScript (`tsc --noEmit` against a probe dir
     that has xstate@5 + @types/node installed).
  2. Uses v5 idioms (regex must match `setup(`; must NOT match a set of
     v4-flavoured patterns like `cond:`, `interpret(`, etc.).
  3. Length sanity (150–6000 chars of extracted code).

Input JSONL shape (one record per line, matching
`v0/data/xstate_curated.jsonl` and `v0/data/xstate_extracted.jsonl`):

    {"messages": [
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "...```typescript\\n...\\n```..."}
    ]}

The verifier extracts the fenced TypeScript block(s) from the assistant
message, runs the gates, and writes:

  - `--out`         : surviving records (same shape as input, one per line)
  - `--rejections`  : CSV with columns `index,reason,detail`
  - stdout          : survival summary

Ordering is preserved by input index. Running the verifier twice overwrites
outputs (idempotent).

Gate 1 (tsc) uses a single batched compilation: all probes are written to
the probe dir, then `npx tsc --noEmit` is invoked once against the
`"include": ["probe-*.ts"]` tsconfig. Per-file errors are attributed back to
records via the `probe-NNNNNN.ts(line,col):` prefix in tsc's output.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Fenced-code extraction
# ---------------------------------------------------------------------------

# Match ```typescript ... ``` or ```ts ... ``` — case-insensitive on the tag,
# non-greedy body, accepts a missing trailing newline before the closing ```.
_FENCE_RE = re.compile(
    r"```(?:typescript|ts)[^\n]*\n(.*?)```",
    re.DOTALL | re.IGNORECASE,
)


def parse_code(assistant_content: str) -> str | None:
    """Return the concatenated content of all ```typescript|ts blocks.

    Returns `None` if the string contains no fenced TS block. When multiple
    fences are present, their bodies are concatenated in order with a single
    newline separator between them.
    """
    if not isinstance(assistant_content, str):
        return None
    matches = _FENCE_RE.findall(assistant_content)
    if not matches:
        return None
    # Strip one trailing newline from each body so the join produces a single
    # blank line between fences, not a double blank line.
    pieces = [m.rstrip("\n") for m in matches]
    return "\n".join(pieces)


# ---------------------------------------------------------------------------
# Comment stripping (for Gate 2 idiom regex — NOT for length gate)
# ---------------------------------------------------------------------------


def strip_comments(code: str) -> str:
    """Remove TS/JS `//...` line comments and `/* ... */` block comments.

    Prose inside comments often discusses v4 patterns (e.g. migration notes
    mentioning `cond:`), which would false-positive the idiom regex. Stripping
    comments before idiom checks sidesteps that.

    This does not try to handle string literals precisely — a real parser is
    needed for that. In practice, v4 keywords in string literals are rare
    enough that we accept that limitation.
    """
    # Block comments first (non-greedy, dotall) so /* // */ doesn't leak.
    code = re.sub(r"/\*.*?\*/", "", code, flags=re.DOTALL)
    # Line comments to end of line.
    code = re.sub(r"//[^\n]*", "", code)
    return code


# ---------------------------------------------------------------------------
# Gate 2 — v5 idiom regexes
# ---------------------------------------------------------------------------

# (name, pattern) — name is used for the rejection reason (`idiom:<name>`).
_MUST_MATCH = re.compile(r"setup\(")

_MUST_NOT_MATCH: list[tuple[str, re.Pattern[str]]] = [
    # `\b` anchors each keyword to a word boundary so we don't match
    # unrelated property names like `second:`, `respond:`, `microservices:`.
    ("cond", re.compile(r"\bcond:")),
    ("interpret", re.compile(r"\binterpret\(")),
    ("services", re.compile(r"\bservices:\s*\{")),
    ("ctx_event_cb", re.compile(r"\(ctx,\s*event\)\s*=>")),
    # v4 `Machine(` factory. Must not fire on `createMachine(` or any other
    # identifier ending in "Machine". Use a negative lookbehind on letters.
    ("Machine", re.compile(r"(?<![A-Za-z])Machine\(")),
]


def check_idiom(code: str) -> tuple[bool, str]:
    """Run the v5 idiom regex gate.

    Returns `(True, "")` if the code passes, otherwise
    `(False, "idiom:<name>:<matched-substring>")` for the first failing
    check. `<matched-substring>` is included so the CSV detail column is
    informative; callers can split on ':' if they want the raw name.

    Comments are stripped from the code before regex matching to avoid
    false positives on v4 keywords that appear in JSDoc / migration-note
    prose.
    """
    stripped = strip_comments(code)
    if not _MUST_MATCH.search(stripped):
        return False, "idiom:missing_setup"
    for name, pat in _MUST_NOT_MATCH:
        m = pat.search(stripped)
        if m is not None:
            return False, f"idiom:{name}:{m.group(0)}"
    return True, ""


# ---------------------------------------------------------------------------
# Gate 3 — length
# ---------------------------------------------------------------------------

MIN_LEN = 150
MAX_LEN = 6000


def check_length(code: str) -> tuple[bool, str]:
    """Reject if `len(code)` is outside [MIN_LEN, MAX_LEN]."""
    n = len(code)
    if n < MIN_LEN:
        return False, f"length:short:{n}"
    if n > MAX_LEN:
        return False, f"length:long:{n}"
    return True, ""


# ---------------------------------------------------------------------------
# Gate 1 — tsc compile (batched)
# ---------------------------------------------------------------------------

_PACKAGE_JSON = {
    "name": "v0-6-verify-probe",
    "private": True,
    "version": "0.0.0",
    "dependencies": {
        "xstate": "^5",
        "@types/node": "^22",
        "typescript": "^5",
    },
}

_TSCONFIG_JSON = {
    "compilerOptions": {
        "strict": True,
        "target": "es2022",
        "module": "esnext",
        "moduleResolution": "bundler",
        "skipLibCheck": True,
        "noEmit": True,
        "esModuleInterop": True,
        "allowSyntheticDefaultImports": True,
    },
    # Batched compile: tsc picks up every probe file in the dir via this
    # glob, so a single compiler startup handles the whole batch.
    "include": ["probe-*.ts"],
}

# Zero-pad probe indices to 6 digits to support runs up to 999_999 records
# without filename collisions or cross-run ordering surprises.
_PROBE_PAD = 6
_PROBE_GLOB = "probe-*.ts"
_PROBE_PATTERN = re.compile(r"^probe-(\d+)\.ts\(")

# Timeout used for the batched tsc call. 10 minutes is plenty for a single
# tsc startup over a few thousand small probe files.
_TSC_TIMEOUT_S = 600

# Timeouts for setup (one-shot at startup).
_NPM_INSTALL_TIMEOUT_S = 600


def _probe_name(idx: int) -> str:
    return f"probe-{idx:0{_PROBE_PAD}d}.ts"


def setup_probe_dir(tmp: Path) -> None:
    """Create the probe directory (if missing) and run `npm install` once.

    Safe to call repeatedly — reinstalls only if `node_modules` is absent.
    Writes `package.json` and `tsconfig.json` every time so changes to the
    config in this source file propagate.
    """
    tmp.mkdir(parents=True, exist_ok=True)
    (tmp / "package.json").write_text(json.dumps(_PACKAGE_JSON, indent=2))
    (tmp / "tsconfig.json").write_text(json.dumps(_TSCONFIG_JSON, indent=2))

    node_modules = tmp / "node_modules"
    if node_modules.exists() and (node_modules / "xstate").exists():
        return
    print(f"[setup] installing npm deps in {tmp} ...", flush=True)
    try:
        res = subprocess.run(
            ["npm", "install", "--silent", "--no-audit", "--no-fund"],
            cwd=tmp,
            capture_output=True,
            text=True,
            timeout=_NPM_INSTALL_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        sys.stderr.write(
            f"npm install timed out after {_NPM_INSTALL_TIMEOUT_S}s\n"
        )
        raise SystemExit(1) from None
    if res.returncode != 0:
        sys.stderr.write("npm install failed:\n")
        sys.stderr.write(res.stdout)
        sys.stderr.write(res.stderr)
        raise SystemExit(1)


def compile_all(
    tmp: Path,
    probes: list[tuple[int, str]],
) -> tuple[dict[int, str], float, bool]:
    """Write all probes, run tsc once, return failures.

    Returns `(failures, elapsed_seconds, timed_out)`:

    - `failures` maps record index → truncated error detail string (≤ 300
      chars) for probes that produced at least one tsc diagnostic.
    - `elapsed_seconds` is wall time spent in the tsc subprocess.
    - `timed_out` is True iff the batched tsc exceeded `_TSC_TIMEOUT_S`.

    On timeout, every probe that produced no parseable error prefix is
    marked as failed with `"tsc:timeout"`. The caller can inspect
    `timed_out` to log a warning or abort.
    """
    # Clean old probe-*.ts files first — prevents stale probes from prior
    # runs polluting the batch compile via the tsconfig `include` glob.
    for p in tmp.glob(_PROBE_GLOB):
        p.unlink()

    # Write all probes for this run.
    for idx, code in probes:
        (tmp / _probe_name(idx)).write_text(code)

    if not probes:
        return {}, 0.0, False

    t0 = time.perf_counter()
    timed_out = False
    try:
        res = subprocess.run(
            ["npx", "--no-install", "tsc", "--noEmit"],
            cwd=tmp,
            capture_output=True,
            text=True,
            timeout=_TSC_TIMEOUT_S,
        )
        output = (res.stdout or "") + (res.stderr or "")
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        # Capture whatever output we got before the timeout.
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        output = stdout + stderr
    elapsed = time.perf_counter() - t0

    failures: dict[int, list[str]] = {}
    for line in output.splitlines():
        m = _PROBE_PATTERN.match(line)
        if m:
            idx = int(m.group(1))
            failures.setdefault(idx, []).append(line)

    # Truncate each idx's error detail to ≤ 300 chars.
    result: dict[int, str] = {
        i: "\n".join(errs)[:300] for i, errs in failures.items()
    }

    if timed_out:
        # Attribute "tsc:timeout" to every probe that wasn't already flagged
        # with a concrete tsc error. This gives downstream consumers a
        # consistent reason and lets survivors-after-timeout be nobody.
        for idx, _code in probes:
            if idx not in result:
                result[idx] = "tsc:timeout"

    return result, elapsed, timed_out


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def _extract_assistant_content(record: dict) -> str | None:
    msgs = record.get("messages")
    if not isinstance(msgs, list):
        return None
    for m in msgs:
        if isinstance(m, dict) and m.get("role") == "assistant":
            c = m.get("content")
            if isinstance(c, str):
                return c
    return None


def _truncate(s: str, limit: int = 300) -> str:
    s = s.replace("\r", " ").replace("\n", " ")
    if len(s) <= limit:
        return s
    return s[: limit - 3] + "..."


def run(
    in_path: Path,
    out_path: Path,
    rejections_path: Path,
    tmp_dir: Path,
) -> dict:
    """Read, verify, and write outputs. Returns summary dict."""
    setup_probe_dir(tmp_dir)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    rejections_path.parent.mkdir(parents=True, exist_ok=True)

    # Read all records up-front so progress counters are accurate.
    records: list[tuple[int, dict]] = []
    with in_path.open("r", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            try:
                records.append((i, json.loads(line)))
            except json.JSONDecodeError as exc:
                # Surface corrupt lines as rejections so the caller sees them.
                records.append((i, {"__parse_error__": str(exc), "__raw__": line}))

    total = len(records)
    rejections: list[tuple[int, str, str]] = []
    reason_counts: dict[str, int] = {}

    # First pass: cheap gates (parse, length, idiom). Records that pass are
    # added to `probes` for the batched tsc run. `survivor_records` preserves
    # input order for final output.
    probes: list[tuple[int, str]] = []
    survivor_records: list[tuple[int, dict]] = []

    def _reject(idx: int, reason: str, detail: str) -> None:
        rejections.append((idx, reason, detail))
        reason_counts[reason] = reason_counts.get(reason, 0) + 1

    for count, (idx, rec) in enumerate(records, start=1):
        if count % 100 == 0 or count == total:
            print(f"[verify] pre-gate {count}/{total} ...", flush=True)

        if "__parse_error__" in rec:
            _reject(idx, "parse", _truncate(rec["__parse_error__"]))
            continue

        content = _extract_assistant_content(rec)
        if content is None:
            _reject(idx, "no_assistant", "")
            continue

        code = parse_code(content)
        if code is None:
            _reject(idx, "no_code_fence", "")
            continue

        # Gate 3 (length) first — cheapest.
        ok, detail = check_length(code)
        if not ok:
            parts = detail.split(":", 2)
            reason = f"{parts[0]}:{parts[1]}"  # "length:short" or "length:long"
            _reject(idx, reason, parts[2] if len(parts) > 2 else "")
            continue

        # Gate 2 (idiom regex) next — also cheap.
        ok, detail = check_idiom(code)
        if not ok:
            parts = detail.split(":", 2)
            if len(parts) >= 2:
                reason = f"{parts[0]}:{parts[1]}"
            else:
                reason = detail
            rej_detail = parts[2] if len(parts) > 2 else ""
            _reject(idx, reason, _truncate(rej_detail))
            continue

        probes.append((idx, code))
        survivor_records.append((idx, rec))

    # Second pass: batched tsc for all candidates that passed cheap gates.
    print(f"[verify] running batched tsc on {len(probes)} candidates ...", flush=True)
    failures, tsc_elapsed, tsc_timed_out = compile_all(tmp_dir, probes)
    print(
        f"[verify] tsc wall time: {tsc_elapsed:.2f}s "
        f"(timeout={'yes' if tsc_timed_out else 'no'})",
        flush=True,
    )

    survivors: list[dict] = []
    for idx, rec in survivor_records:
        if idx in failures:
            detail = failures[idx]
            reason = "tsc:timeout" if detail == "tsc:timeout" else "tsc"
            _reject(idx, reason, _truncate(detail))
            continue
        survivors.append(rec)

    # Write outputs.
    with out_path.open("w", encoding="utf-8") as fh:
        for rec in survivors:
            fh.write(json.dumps(rec, ensure_ascii=False))
            fh.write("\n")

    with rejections_path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, quoting=csv.QUOTE_MINIMAL)
        w.writerow(["index", "reason", "detail"])
        # Sort rejections by index so the CSV mirrors input order.
        for idx, reason, detail in sorted(rejections, key=lambda r: r[0]):
            w.writerow([idx, reason, detail])

    return {
        "total": total,
        "survived": len(survivors),
        "rejected": len(rejections),
        "reasons": reason_counts,
        "tsc_elapsed_s": tsc_elapsed,
        "tsc_timed_out": tsc_timed_out,
    }


def _format_summary(summary: dict) -> str:
    lines = [
        f"Total input records: {summary['total']}",
        f"Survived:            {summary['survived']}",
        f"Rejected:            {summary['rejected']}",
        f"Batched tsc runtime: {summary['tsc_elapsed_s']:.2f}s"
        + (" [TIMED OUT]" if summary.get("tsc_timed_out") else ""),
    ]
    if summary["reasons"]:
        lines.append("Rejection breakdown:")
        for reason, n in sorted(summary["reasons"].items(), key=lambda kv: (-kv[1], kv[0])):
            lines.append(f"  {reason:<24s} {n}")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="verify.py",
        description=(
            "Phase B verifier: filter synthesised XState v5 (instruction, "
            "completion) pairs through compile + idiom + length gates."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--in", dest="in_path",
        default="v0.6/data/synth.raw.jsonl",
        help="Input JSONL of raw synthesised pairs.",
    )
    p.add_argument(
        "--out", dest="out_path",
        default="v0.6/data/synth.verified.jsonl",
        help="Output JSONL of verified (surviving) pairs.",
    )
    p.add_argument(
        "--rejections", dest="rejections_path",
        default="v0.6/data/rejections.csv",
        help="CSV of rejected records (columns: index,reason,detail).",
    )
    p.add_argument(
        "--tmp", dest="tmp_dir",
        default="v0.6/.verify-tmp",
        help="Probe directory for tsc (holds node_modules and probe-*.ts).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    in_path = Path(args.in_path)
    out_path = Path(args.out_path)
    rejections_path = Path(args.rejections_path)
    tmp_dir = Path(args.tmp_dir)

    if not in_path.exists():
        sys.stderr.write(f"error: input file not found: {in_path}\n")
        return 2

    summary = run(in_path, out_path, rejections_path, tmp_dir)
    print()
    print(_format_summary(summary))
    print()
    print(f"Wrote survivors  -> {out_path}")
    print(f"Wrote rejections -> {rejections_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
