"""Clone source repositories. Generic across all languages."""

import shutil
import subprocess
import time
import logging
from pathlib import Path

from lib.common.types import RepoConfig

log = logging.getLogger(__name__)

MAX_RETRIES = 3
BASE_DELAY = 2.0


def _clone_with_retry(cmd: list[str], dest: Path, repo_url: str) -> bool:
    for attempt in range(MAX_RETRIES):
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            return True
        except subprocess.CalledProcessError as e:
            log.warning("Clone attempt %d/%d failed for %s: %s", attempt + 1, MAX_RETRIES, repo_url, e.stderr.strip())
            # Clean up partial clone before retry
            if dest.exists():
                shutil.rmtree(dest, ignore_errors=True)
            if attempt < MAX_RETRIES - 1:
                delay = BASE_DELAY * (2 ** attempt)
                log.info("Retrying in %.0fs...", delay)
                time.sleep(delay)

    log.error("Failed to clone %s after %d attempts", repo_url, MAX_RETRIES)
    return False


def clone_repos(repos: list[RepoConfig], repos_dir: Path, full_history: bool = False) -> list[RepoConfig]:
    """Clone repos with retry. Returns list of successfully cloned RepoConfigs."""
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
        if _clone_with_retry(cmd, dest, repo.url):
            results.append(repo)

    return results
