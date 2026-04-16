"""Quality scoring for extracted semantic units."""

import re
import logging

log = logging.getLogger(__name__)

from extract.walk import FOCUS_TERMS

# TypeScript-specific positive signals
TS_PATTERNS = [
    r"\breadonly\b",
    r":\s*\w+(<",           # type annotations with generics
    r"\bexport\b",
    r"<\w+(\s*,\s*\w+)*>",  # generic parameters
    r"\bas\s+const\b",
]

# Penalty signals
PENALTY_PATTERNS = [
    (r"\bconsole\.\w+\(", 0.05),
    (r"\bTODO\b",         0.05),
    (r"\bFIXME\b",        0.05),
    (r":\s*any\b",        0.10),
]

MIN_LENGTH = 80
QUALITY_THRESHOLD = 0.3


def score_unit(unit: dict) -> float:
    """Score a semantic unit on [0, 1]. Units below QUALITY_THRESHOLD are discarded.

    Score = ts_score + domain_score + diff_bonus - penalties
    """
    code = unit["code"]
    domain = unit["domain"]

    # Length penalty
    if len(code) < MIN_LENGTH:
        return 0.0

    # TypeScript signal score
    ts_score = 0.0
    for pattern in TS_PATTERNS:
        if re.search(pattern, code):
            ts_score += 0.08
    ts_score = min(ts_score, 0.4)

    # Domain signal score
    domain_terms = FOCUS_TERMS.get(domain, [])
    domain_hits = sum(1 for t in domain_terms if t in code)
    domain_score = min(domain_hits * 0.08, 0.4)

    # Diff bonus
    diff_bonus = 0.1 if unit.get("unit_type") == "diff" else 0.0

    # Penalties
    penalty = 0.0
    for pattern, weight in PENALTY_PATTERNS:
        if re.search(pattern, code):
            penalty += weight

    score = ts_score + domain_score + diff_bonus - penalty
    return max(0.0, min(1.0, score))


def filter_by_quality(units: list[dict], threshold: float = QUALITY_THRESHOLD) -> list[dict]:
    """Score and filter units, returning those above threshold sorted by score descending."""
    scored = []
    for unit in units:
        s = score_unit(unit)
        if s >= threshold:
            scored.append({**unit, "quality_score": s})

    scored.sort(key=lambda u: u["quality_score"], reverse=True)
    log.info("Quality filter: %d/%d units passed (threshold=%.2f)", len(scored), len(units), threshold)
    return scored
