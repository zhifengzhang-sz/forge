"""Walk .ts files and filter by focus terms. TypeScript-specific."""

import logging
from pathlib import Path

from lib.common.types import TopicConfig

log = logging.getLogger(__name__)

TS_SKIP_DIRS = {"node_modules", "dist", "build", ".git", "__tests__"}
TS_SKIP_SUFFIXES = {".d.ts", ".spec.ts", ".test.ts"}


def _should_skip(path: Path, config: TopicConfig) -> bool:
    skip_dirs = set(config.skip_dirs) if config.skip_dirs else TS_SKIP_DIRS
    skip_suffixes = set(config.skip_suffixes) if config.skip_suffixes else TS_SKIP_SUFFIXES

    if any(p in skip_dirs for p in path.parts):
        return True
    if any(path.name.endswith(s) for s in skip_suffixes):
        return True
    return False


def _has_focus_terms(content: str, config: TopicConfig) -> bool:
    return any(term in content for term in config.focus_terms)


def walk_ts_files(repo_path: Path, config: TopicConfig) -> list[dict]:
    """Walk a repo and return .ts files matching the topic's focus terms.

    Returns list of dicts with keys: path, content, domain, repo_path.
    """
    results = []
    extensions = config.file_extensions or [".ts"]

    for ext in extensions:
        for ts_file in repo_path.rglob(f"*{ext}"):
            if _should_skip(ts_file, config):
                continue

            try:
                content = ts_file.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError) as e:
                log.warning("Skipping %s: %s", ts_file, e)
                continue

            if not _has_focus_terms(content, config):
                continue

            results.append({
                "path": ts_file,
                "content": content,
                "domain": config.name,
                "repo_path": repo_path,
            })

    log.info("Found %d matching files in %s (topic: %s)", len(results), repo_path.name, config.name)
    return results
