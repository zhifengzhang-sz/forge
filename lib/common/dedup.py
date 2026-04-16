"""SHA-256 deduplication. Generic across all languages."""

import hashlib
import json
import re
import logging
from pathlib import Path

from lib.common.types import Unit

log = logging.getLogger(__name__)


def _strip(code: str) -> str:
    """Strip whitespace and comments for fingerprinting."""
    code = re.sub(r"//.*$", "", code, flags=re.MULTILINE)
    code = re.sub(r"/\*.*?\*/", "", code, flags=re.DOTALL)
    code = re.sub(r"#.*$", "", code, flags=re.MULTILINE)  # Python comments
    code = re.sub(r"\s+", "", code)
    return code


def fingerprint(code: str) -> str:
    """SHA-256 fingerprint of stripped code, truncated to 16 hex chars."""
    return hashlib.sha256(_strip(code).encode()).hexdigest()[:16]


def load_held_out_fingerprints(held_out_dir: Path) -> set[str]:
    """Load SHA-256 fingerprints from the held-out eval set."""
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


def deduplicate(units: list[Unit], held_out_fps: set[str] | None = None) -> list[Unit]:
    """Remove exact duplicates and held-out eval examples."""
    excluded = held_out_fps or set()
    seen: set[str] = set(excluded)
    result = []
    held_out_count = 0

    for unit in units:
        fp = fingerprint(unit.code)
        if fp in excluded:
            held_out_count += 1
            continue
        if fp in seen:
            continue
        seen.add(fp)
        unit.fingerprint = fp
        result.append(unit)

    dupes = len(units) - len(result) - held_out_count
    log.info("Dedup: %d -> %d units (%d duplicates, %d held-out excluded)", len(units), len(result), dupes, held_out_count)
    return result
