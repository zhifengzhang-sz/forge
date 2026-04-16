"""SHA-256 deduplication of semantic units."""

import hashlib
import json
import re
import logging
from pathlib import Path

log = logging.getLogger(__name__)


def _strip(code: str) -> str:
    """Strip whitespace and comments for fingerprinting."""
    code = re.sub(r"//.*$", "", code, flags=re.MULTILINE)
    code = re.sub(r"/\*.*?\*/", "", code, flags=re.DOTALL)
    code = re.sub(r"\s+", "", code)
    return code


def fingerprint(code: str) -> str:
    """SHA-256 fingerprint of stripped code, truncated to 16 hex chars."""
    return hashlib.sha256(_strip(code).encode()).hexdigest()[:16]


def load_held_out_fingerprints(held_out_dir: Path) -> set[str]:
    """Load SHA-256 fingerprints from the held-out eval set.

    Reads all .json files in held_out_dir. Each file should be an array of
    objects with an 'expected_output' field containing code.
    """
    fps: set[str] = set()
    if not held_out_dir.exists():
        return fps

    for json_file in held_out_dir.glob("*.json"):
        try:
            data = json.loads(json_file.read_text())
            for item in data:
                code = item.get("expected_output", "")
                if code:
                    fps.add(fingerprint(code))
        except (json.JSONDecodeError, OSError) as e:
            log.warning("Could not read held-out file %s: %s", json_file, e)

    log.info("Loaded %d held-out fingerprints from %s", len(fps), held_out_dir)
    return fps


def deduplicate(units: list[dict], held_out_fps: set[str] | None = None) -> list[dict]:
    """Remove exact duplicates and held-out eval examples by SHA-256 fingerprint."""
    seen: set[str] = set()
    if held_out_fps:
        seen.update(held_out_fps)

    result = []
    held_out_excluded = 0

    for unit in units:
        fp = fingerprint(unit["code"])
        if fp in held_out_fps and held_out_fps:
            held_out_excluded += 1
            continue
        if fp in seen:
            continue
        seen.add(fp)
        result.append({**unit, "fingerprint": fp})

    dupes = len(units) - len(result) - held_out_excluded
    log.info("Dedup: %d -> %d units (%d duplicates, %d held-out excluded)", len(units), len(result), dupes, held_out_excluded)
    return result
