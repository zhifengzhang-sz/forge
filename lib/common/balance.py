"""Domain balancing. Generic across all languages."""

import logging
from collections import defaultdict

from lib.common.types import Unit

log = logging.getLogger(__name__)

MAX_PER_DOMAIN = 500
MIN_FLOOR = 100  # small domains get all their units; cap is at least this


def balance_domains(units: list[Unit]) -> list[Unit]:
    """Balance units across domains.

    Cap formula: max(2 * median_domain_size, MIN_FLOOR), capped at MAX_PER_DOMAIN.
    Uses median instead of min so one small domain doesn't crush the rest.
    Small domains that fall below the cap contribute all their units.
    """
    by_domain: dict[str, list[Unit]] = defaultdict(list)
    for unit in units:
        by_domain[unit.domain].append(unit)

    for domain in by_domain:
        by_domain[domain].sort(key=lambda u: u.quality_score, reverse=True)

    domain_sizes = sorted(len(us) for us in by_domain.values())
    if not domain_sizes:
        return []

    # Use median instead of min
    median_size = domain_sizes[len(domain_sizes) // 2]
    cap = min(max(2 * median_size, MIN_FLOOR), MAX_PER_DOMAIN)

    result = []
    for domain, domain_units in by_domain.items():
        selected = domain_units[:cap]
        result.extend(selected)
        log.info("Domain '%s': %d available, %d selected (cap=%d)", domain, len(domain_units), len(selected), cap)

    log.info("Balanced dataset: %d total units across %d domains (cap=%d)", len(result), len(by_domain), cap)
    return result
