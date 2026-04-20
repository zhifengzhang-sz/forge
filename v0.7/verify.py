#!/usr/bin/env python3
"""Phase B verifier for v0.7 multi-domain synthesised TypeScript pairs.

Forked from `v0.6/verify.py`. Extends the XState-only v0.6 design with
per-domain gates so the same verifier can run against XState, fp-ts /
Effect-TS, or RxJS synthesis output.

Each (instruction, completion) pair is run through three gates:

  1. Compiles under strict TypeScript (`tsc --noEmit` against a per-domain
     probe dir that has the domain's runtime deps installed).
  2. Uses domain-appropriate idioms (regex MUST-MATCH at least one positive
     token; MUST-NOT-MATCH any shape-guard or deprecated token).
  3. Length sanity (150-6000 chars of extracted code).

Input JSONL shape (one record per line):

    {"messages": [
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "...```typescript\\n...\\n```..."}
    ], "domain": "fp"}

The `domain` field is optional. Resolution order:

  1. If `--domain` is passed on the CLI, every record is treated as that
     domain (per-record `domain` is ignored).
  2. Else if a record has a `domain` field in {xstate, fp, rx}, that wins.
  3. Else default to `xstate` (v0.6 behaviour).

Per-domain probe dirs live under `--tmp` (default `v0.7/.verify-tmp/`):

  - `v0.7/.verify-tmp/xstate/` — xstate@^5 + @types/node
  - `v0.7/.verify-tmp/fp/`     — fp-ts@^2 + effect@^3 + @types/node
  - `v0.7/.verify-tmp/rx/`     — rxjs@^7 + @types/node

Each probe dir is npm-installed once on first use and reused thereafter
(same pattern as v0.6). tsc is run once per domain batch. The output shape
(survivors JSONL, rejections CSV, stdout summary) mirrors v0.6.
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
# Domains
# ---------------------------------------------------------------------------

DOMAINS = ("xstate", "fp", "rx", "es")
DEFAULT_DOMAIN = "xstate"


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
    pieces = [m.rstrip("\n") for m in matches]
    return "\n".join(pieces)


# ---------------------------------------------------------------------------
# Comment stripping (for Gate 2 idiom regex — NOT for length gate)
# ---------------------------------------------------------------------------


def strip_comments(code: str) -> str:
    """Remove TS/JS `//...` line comments and `/* ... */` block comments.

    Prose inside comments often discusses deprecated patterns (e.g. migration
    notes mentioning `cond:` or `Effect.service(`), which would false-positive
    the idiom regex. Stripping comments before idiom checks sidesteps that.

    This does not try to handle string literals precisely — a real parser is
    needed for that. In practice, library keywords in string literals are rare
    enough that we accept that limitation.
    """
    code = re.sub(r"/\*.*?\*/", "", code, flags=re.DOTALL)
    code = re.sub(r"//[^\n]*", "", code)
    return code


# ---------------------------------------------------------------------------
# Gate 2 — per-domain idiom regexes
# ---------------------------------------------------------------------------

# XState: single MUST-MATCH (setup(), v5-only factory) and the v0.6 negative
# shape guards.
_XSTATE_MUST_MATCH: list[tuple[str, re.Pattern[str]]] = [
    ("setup", re.compile(r"setup\(")),
]

_XSTATE_MUST_NOT_MATCH: list[tuple[str, re.Pattern[str]]] = [
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

# FP: any of a list of Effect / fp-ts tokens satisfies the positive gate.
# Shape guards reject code that mixes in XState patterns or uses deprecated
# Effect APIs.
_FP_MUST_MATCH: list[tuple[str, re.Pattern[str]]] = [
    ("Effect_gen", re.compile(r"Effect\.gen")),
    ("Effect_succeed", re.compile(r"Effect\.succeed")),
    ("Effect_fail", re.compile(r"Effect\.fail")),
    ("Effect_tap", re.compile(r"Effect\.tap")),
    ("Effect_flatMap", re.compile(r"Effect\.flatMap")),
    ("Effect_map", re.compile(r"Effect\.map")),
    ("pipe", re.compile(r"pipe\(")),
    ("Layer", re.compile(r"Layer\.")),
    ("Schedule", re.compile(r"Schedule\.")),
    ("Context_Tag", re.compile(r"Context\.(Generic)?Tag")),
    ("makeContext", re.compile(r"makeContext")),
    ("E_chain", re.compile(r"E\.chain")),
    ("E_right", re.compile(r"E\.right")),
    ("E_left", re.compile(r"E\.left")),
    ("O_chain", re.compile(r"O\.chain")),
    ("O_some", re.compile(r"O\.some")),
    ("O_none", re.compile(r"O\.none")),
    ("TE_chain", re.compile(r"TE\.chain")),
    ("TE_right", re.compile(r"TE\.right")),
    ("TE_left", re.compile(r"TE\.left")),
]

_FP_MUST_NOT_MATCH: list[tuple[str, re.Pattern[str]]] = [
    # XState leakage: any import from 'xstate' in an fp-ts/Effect answer.
    ("xstate_import_in_fp", re.compile(r"from ['\"]xstate['\"]")),
    # XState shape guard: `setup(` shouldn't appear in fp answers.
    ("setup_in_fp", re.compile(r"setup\(")),
    # Deprecated Effect-TS API (removed in effect@3).
    ("Effect_service_deprecated", re.compile(r"Effect\.service\(")),
]

# RX: any of a list of RxJS tokens. Shape guards reject XState leakage and the
# pre-v7 `.do(` operator (renamed to `tap` in rxjs@5.5).
_RX_MUST_MATCH: list[tuple[str, re.Pattern[str]]] = [
    ("pipe", re.compile(r"pipe\(")),
    ("switchMap", re.compile(r"switchMap\(")),
    ("combineLatest", re.compile(r"combineLatest\(")),
    ("mergeMap", re.compile(r"mergeMap\(")),
    ("forkJoin", re.compile(r"forkJoin\(")),
    ("BehaviorSubject", re.compile(r"BehaviorSubject")),
    ("of", re.compile(r"of\(")),
    ("Subject", re.compile(r"Subject")),
    ("Observable_generic", re.compile(r"Observable<")),
    ("ReplaySubject", re.compile(r"ReplaySubject")),
    ("retry", re.compile(r"retry\(")),
    ("shareReplay", re.compile(r"shareReplay")),
    ("tap", re.compile(r"tap\(")),
]

_RX_MUST_NOT_MATCH: list[tuple[str, re.Pattern[str]]] = [
    ("xstate_import_in_rx", re.compile(r"from ['\"]xstate['\"]")),
    ("setup_in_rx", re.compile(r"setup\(")),
    # Pre-v7 `.do(` operator — renamed to `tap` in rxjs@5.5; shouldn't appear
    # in idiomatic v7 code.
    ("rxjs_do_operator", re.compile(r"\.do\(")),
]

# ES: 18-token MUST-MATCH covering both Decider-style (evolve/decide) and
# Oskar-style (when/aggregateStream/reduce<...>) idioms. Shape guards reject
# XState leakage. Effect.gen is NOT blocked — Effect-wrapped event store
# access is a valid ES pattern.
_ES_MUST_MATCH: list[tuple[str, re.Pattern[str]]] = [
    ("evolve", re.compile(r"evolve\(")),
    ("decide", re.compile(r"decide\(")),
    ("dotted_decide", re.compile(r"\.decide\(")),
    ("Decider", re.compile(r"\bDecider\b")),
    ("CommandHandler", re.compile(r"\bCommandHandler\b")),
    ("EventStore", re.compile(r"\bEventStore\b")),
    ("EventStoreDB", re.compile(r"\bEventStoreDB")),
    ("Snapshot", re.compile(r"\bSnapshot\b")),
    ("expectedRevision", re.compile(r"\bexpectedRevision\b")),
    ("appendToStream", re.compile(r"\bappendToStream\b")),
    ("readStream", re.compile(r"\breadStream\b")),
    ("readFromStream", re.compile(r"\breadFromStream\b")),
    ("when_fn", re.compile(r"\bwhen\(")),
    ("aggregateStream", re.compile(r"\baggregateStream\(")),
    ("reduce_typed", re.compile(r"\breduce<")),
    ("projection", re.compile(r"\bprojection")),
    ("aggregate", re.compile(r"\baggregate")),
    ("project_fn", re.compile(r"\bproject\(")),
]

_ES_MUST_NOT_MATCH: list[tuple[str, re.Pattern[str]]] = [
    # Shape guards against XState-leakage back into ES answers.
    ("xstate_import_in_es", re.compile(r"from ['\"]xstate['\"]")),
    ("setup_in_es", re.compile(r"setup\(")),
    # NOTE: Effect.gen is intentionally NOT blocked — ES code may wrap the
    # event-store IO boundary in Effect.gen, and that's a valid idiom.
]


# The per-domain bundles use two shapes:
#
#   MUST_MATCH  — (positive_name, [(token_name, pat), ...]): at least one
#                 pattern in the list must match. `positive_name` is the
#                 rejection reason name used when NONE of them match
#                 (e.g. "missing_fp_positive").
#   MUST_NOT    — [(name, pat), ...]: any match is a rejection
#                 (reason = "idiom:<name>").
_IDIOMS: dict[str, tuple[str, list[tuple[str, re.Pattern[str]]], list[tuple[str, re.Pattern[str]]]]] = {
    "xstate": ("missing_setup", _XSTATE_MUST_MATCH, _XSTATE_MUST_NOT_MATCH),
    "fp": ("missing_fp_positive", _FP_MUST_MATCH, _FP_MUST_NOT_MATCH),
    "rx": ("missing_rx_positive", _RX_MUST_MATCH, _RX_MUST_NOT_MATCH),
    "es": ("missing_es_positive", _ES_MUST_MATCH, _ES_MUST_NOT_MATCH),
}


def check_idiom(code: str, domain: str = DEFAULT_DOMAIN) -> tuple[bool, str]:
    """Run the per-domain idiom regex gate.

    Returns `(True, "")` if the code passes, otherwise
    `(False, "idiom:<name>[:<matched-substring>]")` for the first failing
    check. Callers can split on ':' to recover the reason name.

    Comments are stripped from the code before regex matching to avoid false
    positives on keywords that appear in JSDoc / migration-note prose.
    """
    if domain not in _IDIOMS:
        raise ValueError(f"unknown domain {domain!r}; expected one of {DOMAINS}")

    stripped = strip_comments(code)
    missing_name, must_match, must_not_match = _IDIOMS[domain]

    # MUST-MATCH: at least one positive pattern in the list must match.
    if not any(pat.search(stripped) for _, pat in must_match):
        return False, f"idiom:{missing_name}"

    # MUST-NOT-MATCH: first match wins as rejection reason.
    for name, pat in must_not_match:
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
# Gate 1 — tsc compile (batched, per-domain probe dir)
# ---------------------------------------------------------------------------

# Shared tsconfig across domains (identical compiler options; only the
# dependency set in package.json differs by domain).
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


def _package_json_for(domain: str) -> dict:
    """Return the domain-specific package.json dict.

    typescript is always included (drives tsc); @types/node is always
    included (probes often use `process.env` etc.); the rest are
    domain-specific.
    """
    deps: dict[str, str] = {
        "typescript": "^5",
        "@types/node": "^22",
    }
    if domain == "xstate":
        deps["xstate"] = "^5"
    elif domain == "fp":
        deps["fp-ts"] = "^2"
        deps["effect"] = "^3"
    elif domain == "rx":
        deps["rxjs"] = "^7"
    elif domain == "es":
        # ES probes declare their event-store interface locally (no client
        # dep needed). `effect` IS included because the ES idiom gate
        # explicitly allows Effect-wrapped event store access, and those
        # probes need `effect` at compile time.
        deps["effect"] = "^3"
    else:
        raise ValueError(f"unknown domain {domain!r}")
    return {
        "name": f"v0-7-verify-probe-{domain}",
        "private": True,
        "version": "0.0.0",
        "dependencies": deps,
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


def _marker_package_for_domain(domain: str) -> str:
    """Return the node_modules marker dir that signals install completed."""
    if domain == "xstate":
        return "xstate"
    if domain == "fp":
        # effect is the larger install; check for it.
        return "effect"
    if domain == "rx":
        return "rxjs"
    if domain == "es":
        # ES has no domain-specific dep; the common `typescript` install
        # proves the probe dir is ready.
        return "typescript"
    raise ValueError(f"unknown domain {domain!r}")


def setup_probe_dir(tmp: Path, domain: str = DEFAULT_DOMAIN) -> Path:
    """Create the per-domain probe directory and run `npm install` once.

    `tmp` is the root tmp dir (e.g. `v0.7/.verify-tmp`); the per-domain subdir
    (`tmp/<domain>`) is what gets tsc invoked against. Returns that subdir
    path.

    Safe to call repeatedly — reinstalls only if the marker package is
    absent from node_modules. Writes `package.json` and `tsconfig.json` every
    time so changes to the config in this source file propagate.
    """
    if domain not in DOMAINS:
        raise ValueError(f"unknown domain {domain!r}; expected one of {DOMAINS}")
    probe_dir = tmp / domain
    probe_dir.mkdir(parents=True, exist_ok=True)
    (probe_dir / "package.json").write_text(
        json.dumps(_package_json_for(domain), indent=2)
    )
    (probe_dir / "tsconfig.json").write_text(json.dumps(_TSCONFIG_JSON, indent=2))

    marker = _marker_package_for_domain(domain)
    node_modules = probe_dir / "node_modules"
    if node_modules.exists() and (node_modules / marker).exists():
        return probe_dir
    print(f"[setup] installing npm deps in {probe_dir} ...", flush=True)
    try:
        res = subprocess.run(
            ["npm", "install", "--silent", "--no-audit", "--no-fund"],
            cwd=probe_dir,
            capture_output=True,
            text=True,
            timeout=_NPM_INSTALL_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        sys.stderr.write(
            f"npm install timed out after {_NPM_INSTALL_TIMEOUT_S}s for {domain}\n"
        )
        raise SystemExit(1) from None
    if res.returncode != 0:
        sys.stderr.write(f"npm install failed for {domain}:\n")
        sys.stderr.write(res.stdout)
        sys.stderr.write(res.stderr)
        raise SystemExit(1)
    return probe_dir


def compile_all(
    tmp: Path,
    probes: list[tuple[int, str]],
) -> tuple[dict[int, str], float, bool]:
    """Write all probes, run tsc once, return failures.

    `tmp` must be the per-domain probe dir (as returned by `setup_probe_dir`)
    — NOT the root tmp dir.

    Returns `(failures, elapsed_seconds, timed_out)`:

    - `failures` maps record index → truncated error detail string (<= 300
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

    result: dict[int, str] = {
        i: "\n".join(errs)[:300] for i, errs in failures.items()
    }

    if timed_out:
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


def _resolve_domain(record: dict, cli_domain: str | None) -> str:
    """Return the domain to apply to this record.

    Resolution order (see module docstring):
      1. CLI flag always wins.
      2. Else record's `domain` field if valid.
      3. Else DEFAULT_DOMAIN.
    """
    if cli_domain:
        return cli_domain
    rec_domain = record.get("domain")
    if isinstance(rec_domain, str) and rec_domain in DOMAINS:
        return rec_domain
    return DEFAULT_DOMAIN


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
    cli_domain: str | None,
) -> dict:
    """Read, verify, and write outputs. Returns summary dict.

    `tmp_dir` is the root tmp dir (per-domain subdirs are created under it).
    `cli_domain` is the CLI override; None means per-record resolution.
    """
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
                records.append((i, {"__parse_error__": str(exc), "__raw__": line}))

    total = len(records)
    rejections: list[tuple[int, str, str]] = []
    reason_counts: dict[str, int] = {}

    # Bucket cheap-gate survivors by domain, because tsc needs to run in the
    # domain's probe dir (different deps installed).
    probes_by_domain: dict[str, list[tuple[int, str]]] = {d: [] for d in DOMAINS}
    survivor_records: list[tuple[int, dict, str]] = []

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
            reason = f"{parts[0]}:{parts[1]}"
            _reject(idx, reason, parts[2] if len(parts) > 2 else "")
            continue

        # Resolve the domain AFTER we have a record in hand — per-record
        # override is honoured unless CLI forces.
        domain = _resolve_domain(rec, cli_domain)

        # Gate 2 (idiom regex) — per-domain.
        ok, detail = check_idiom(code, domain=domain)
        if not ok:
            parts = detail.split(":", 2)
            if len(parts) >= 2:
                reason = f"{parts[0]}:{parts[1]}"
            else:
                reason = detail
            rej_detail = parts[2] if len(parts) > 2 else ""
            _reject(idx, reason, _truncate(rej_detail))
            continue

        probes_by_domain[domain].append((idx, code))
        survivor_records.append((idx, rec, domain))

    # Second pass: batched tsc per domain. Only set up probe dirs for
    # domains that actually have candidates to compile — saves one or two
    # npm installs on single-domain runs.
    tsc_elapsed_total = 0.0
    tsc_timed_out_any = False
    tsc_failures: dict[int, tuple[str, bool]] = {}  # idx -> (detail, timed_out)

    for domain in DOMAINS:
        dprobes = probes_by_domain[domain]
        if not dprobes:
            continue
        probe_dir = setup_probe_dir(tmp_dir, domain)
        print(
            f"[verify] running batched tsc on {len(dprobes)} {domain} candidates ...",
            flush=True,
        )
        failures, elapsed, timed_out = compile_all(probe_dir, dprobes)
        tsc_elapsed_total += elapsed
        tsc_timed_out_any = tsc_timed_out_any or timed_out
        print(
            f"[verify] tsc[{domain}] wall time: {elapsed:.2f}s "
            f"(timeout={'yes' if timed_out else 'no'})",
            flush=True,
        )
        for idx, detail in failures.items():
            tsc_failures[idx] = (detail, detail == "tsc:timeout")

    survivors: list[dict] = []
    for idx, rec, _domain in survivor_records:
        if idx in tsc_failures:
            detail, was_timeout = tsc_failures[idx]
            reason = "tsc:timeout" if was_timeout else "tsc"
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
        for idx, reason, detail in sorted(rejections, key=lambda r: r[0]):
            w.writerow([idx, reason, detail])

    return {
        "total": total,
        "survived": len(survivors),
        "rejected": len(rejections),
        "reasons": reason_counts,
        "tsc_elapsed_s": tsc_elapsed_total,
        "tsc_timed_out": tsc_timed_out_any,
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
            lines.append(f"  {reason:<32s} {n}")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="verify.py",
        description=(
            "Phase B verifier (v0.7, multi-domain): filter synthesised "
            "TypeScript (instruction, completion) pairs through per-domain "
            "compile + idiom + length gates. Supports XState v5, fp-ts / "
            "Effect-TS, and RxJS. Domain is selected by --domain (applies "
            "to all records) or read from each record's `domain` field "
            f"(valid values: {', '.join(DOMAINS)}; default: {DEFAULT_DOMAIN})."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--in", dest="in_path",
        default="v0.7/data/synth.raw.jsonl",
        help="Input JSONL of raw synthesised pairs.",
    )
    p.add_argument(
        "--out", dest="out_path",
        default="v0.7/data/synth.verified.jsonl",
        help="Output JSONL of verified (surviving) pairs.",
    )
    p.add_argument(
        "--rejections", dest="rejections_path",
        default="v0.7/data/rejections.csv",
        help="CSV of rejected records (columns: index,reason,detail).",
    )
    p.add_argument(
        "--tmp", dest="tmp_dir",
        default="v0.7/.verify-tmp",
        help=(
            "Root probe directory. Per-domain subdirs (xstate/, fp/, rx/) "
            "hold node_modules and probe-*.ts."
        ),
    )
    p.add_argument(
        "--domain",
        choices=list(DOMAINS),
        default=None,
        help=(
            "Force all records to this domain (overrides per-record "
            "`domain` field). If omitted, each record uses its own `domain` "
            "field, falling back to xstate when absent."
        ),
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    in_path = Path(args.in_path)
    out_path = Path(args.out_path)
    rejections_path = Path(args.rejections_path)
    tmp_dir = Path(args.tmp_dir)
    cli_domain = args.domain  # None or one of DOMAINS

    if not in_path.exists():
        sys.stderr.write(f"error: input file not found: {in_path}\n")
        return 2

    summary = run(in_path, out_path, rejections_path, tmp_dir, cli_domain)
    print()
    print(_format_summary(summary))
    print()
    print(f"Wrote survivors  -> {out_path}")
    print(f"Wrote rejections -> {rejections_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
