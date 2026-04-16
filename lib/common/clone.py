"""Clone source repositories. Generic across all languages."""

import subprocess
import logging
from pathlib import Path

from lib.common.types import RepoConfig

log = logging.getLogger(__name__)


def clone_repos(repos: list[RepoConfig], repos_dir: Path, full_history: bool = False) -> list[RepoConfig]:
    """Clone repos. Returns list of successfully cloned RepoConfigs.

    Args:
        repos: List of repo configurations.
        repos_dir: Directory to clone into.
        full_history: If True, clone full history (needed for diff extraction).
    """
    repos_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for repo in repos:
        dest = repos_dir / repo.name

        if dest.exists():
            log.info("Skipping %s (already exists at %s)", repo.name, dest)
            results.append(repo)
            continue

        cmd = ["git", "clone"]
        if not full_history:
            cmd += ["--depth=1"]
        cmd += [repo.url, str(dest)]

        log.info("Cloning %s -> %s", repo.url, dest)
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            results.append(repo)
        except subprocess.CalledProcessError as e:
            log.error("Failed to clone %s: %s", repo.url, e.stderr.strip())

    return results
