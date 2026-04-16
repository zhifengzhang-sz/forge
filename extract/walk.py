"""Walk .ts files and filter by domain focus terms."""

import logging
from pathlib import Path

log = logging.getLogger(__name__)

SKIP_PATTERNS = {
    "node_modules", "dist", "build", ".git", "__tests__",
}

SKIP_SUFFIXES = {".d.ts", ".spec.ts", ".test.ts"}

FOCUS_TERMS: dict[str, list[str]] = {
    "fp":             ["pipe(", "flow(", "Option<", "Either<", "Task<", "Reader"],
    "reactive":       ["Observable<", "Subject", "switchMap(", "mergeMap(", "combineLatest"],
    "xstate":         ["setup(", "createMachine(", "fromPromise(", "fromObservable(", "assign("],
    "eventsourcing":  ["Aggregate", "evolve(", "Command", "EventStore", "append(", "readStream("],
}


def _should_skip(path: Path) -> bool:
    parts = path.parts
    if any(p in SKIP_PATTERNS for p in parts):
        return True
    if any(path.name.endswith(s) for s in SKIP_SUFFIXES):
        return True
    return False


def _has_focus_terms(content: str, domain: str) -> bool:
    terms = FOCUS_TERMS.get(domain, [])
    return any(term in content for term in terms)


def walk_ts_files(repo_path: Path, domain: str) -> list[dict]:
    """Walk a repo and return .ts files matching the domain's focus terms.

    Returns list of dicts with keys: path, content, domain.
    """
    results = []

    for ts_file in repo_path.rglob("*.ts"):
        if _should_skip(ts_file):
            continue

        try:
            content = ts_file.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError) as e:
            log.warning("Skipping %s: %s", ts_file, e)
            continue

        if not _has_focus_terms(content, domain):
            continue

        results.append({
            "path": ts_file,
            "content": content,
            "domain": domain,
            "repo_path": repo_path,
        })

    log.info("Found %d matching .ts files in %s (domain: %s)", len(results), repo_path.name, domain)
    return results
