"""Clone source repositories for training data extraction."""

import subprocess
import logging
from pathlib import Path

log = logging.getLogger(__name__)

REPOS = [
    {"url": "https://github.com/gcanti/fp-ts.git",                      "domain": "fp",             "name": "fp-ts"},
    {"url": "https://github.com/Effect-TS/effect.git",                   "domain": "fp",             "name": "effect"},
    {"url": "https://github.com/ReactiveX/rxjs.git",                     "domain": "reactive",       "name": "rxjs"},
    {"url": "https://github.com/statelyai/xstate.git",                   "domain": "xstate",         "name": "xstate"},
    {"url": "https://github.com/oskardudycz/EventSourcing.NodeJS.git",   "domain": "eventsourcing",  "name": "EventSourcing.NodeJS"},
]


def clone_repos(repos_dir: Path, full_history: bool = False) -> list[dict]:
    """Clone repos. Returns list of repo dicts with 'path' added.

    Args:
        repos_dir: Directory to clone into.
        full_history: If True, clone full history (needed for diff extraction).
                      If False, use --depth=1 for speed.
    """
    repos_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for repo in REPOS:
        dest = repos_dir / repo["name"]

        if dest.exists():
            log.info("Skipping %s (already exists at %s)", repo["name"], dest)
            results.append({**repo, "path": dest})
            continue

        cmd = ["git", "clone"]
        if not full_history:
            cmd += ["--depth=1"]
        cmd += [repo["url"], str(dest)]

        log.info("Cloning %s -> %s", repo["url"], dest)
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            results.append({**repo, "path": dest})
        except subprocess.CalledProcessError as e:
            log.error("Failed to clone %s: %s", repo["url"], e.stderr.strip())

    return results
