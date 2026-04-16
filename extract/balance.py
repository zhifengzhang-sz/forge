"""Domain balancing to prevent any single domain from dominating the dataset."""

import logging
from collections import defaultdict

log = logging.getLogger(__name__)

MAX_PER_DOMAIN = 500


def balance_domains(units: list[dict]) -> list[dict]:
    """Balance units across domains.

    Soft cap: no domain exceeds 2x the size of the smallest domain,
    capped at MAX_PER_DOMAIN. Units are selected in descending quality
    score order within each domain.
    """
    by_domain: dict[str, list[dict]] = defaultdict(list)
    for unit in units:
        by_domain[unit["domain"]].append(unit)

    # Sort each domain by quality score descending
    for domain in by_domain:
        by_domain[domain].sort(key=lambda u: u.get("quality_score", 0), reverse=True)

    # Compute cap: 2x smallest domain, capped at MAX_PER_DOMAIN
    domain_sizes = {d: len(us) for d, us in by_domain.items()}
    if not domain_sizes:
        return []

    min_size = min(domain_sizes.values())
    cap = min(2 * min_size, MAX_PER_DOMAIN)

    result = []
    for domain, domain_units in by_domain.items():
        selected = domain_units[:cap]
        result.extend(selected)
        log.info("Domain '%s': %d available, %d selected (cap=%d)", domain, len(domain_units), len(selected), cap)

    log.info("Balanced dataset: %d total units across %d domains", len(result), len(by_domain))
    return result
