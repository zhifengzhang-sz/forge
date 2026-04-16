"""Quality scoring for TypeScript semantic units."""

import re
import logging

from lib.common.types import Unit, TopicConfig

log = logging.getLogger(__name__)

# Default TypeScript-specific positive signals
DEFAULT_TS_SIGNALS = [
    (r"\breadonly\b",             0.08),
    (r":\s*\w+(<",               0.08),  # type annotations with generics
    (r"\bexport\b",              0.08),
    (r"<\w+(\s*,\s*\w+)*>",     0.08),  # generic parameters
    (r"\bas\s+const\b",          0.08),
]

# Default penalties
DEFAULT_PENALTIES = [
    (r"\bconsole\.\w+\(",  0.05),
    (r"\bTODO\b",          0.05),
    (r"\bFIXME\b",         0.05),
    (r":\s*any\b",         0.10),
]

QUALITY_THRESHOLD = 0.3


def score_unit(unit: Unit, config: TopicConfig) -> float:
    """Score a semantic unit on [0, 1]."""
    code = unit.code

    if len(code) < config.min_unit_length:
        return 0.0

    # Language-level signals (from config or defaults)
    signals = config.scoring_signals if config.scoring_signals else DEFAULT_TS_SIGNALS
    ts_score = 0.0
    for pattern, weight in signals:
        if re.search(pattern, code):
            ts_score += weight
    ts_score = min(ts_score, 0.4)

    # Domain focus term signals
    domain_hits = sum(1 for t in config.focus_terms if t in code)
    domain_score = min(domain_hits * 0.08, 0.4)

    # Diff bonus
    diff_bonus = 0.1 if unit.unit_type == "diff" else 0.0

    # Penalties (from config or defaults)
    penalties = config.scoring_penalties if config.scoring_penalties else DEFAULT_PENALTIES
    penalty = 0.0
    for pattern, weight in penalties:
        if re.search(pattern, code):
            penalty += weight

    score = ts_score + domain_score + diff_bonus - penalty
    return max(0.0, min(1.0, score))


def filter_by_quality(units: list[Unit], config: TopicConfig, threshold: float = QUALITY_THRESHOLD) -> list[Unit]:
    """Score and filter units, returning those above threshold sorted by score descending."""
    result = []
    for unit in units:
        s = score_unit(unit, config)
        if s >= threshold:
            unit.quality_score = s
            result.append(unit)

    result.sort(key=lambda u: u.quality_score, reverse=True)
    log.info("Quality filter (%s): %d/%d units passed (threshold=%.2f)", config.name, len(result), len(units), threshold)
    return result
