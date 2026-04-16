#!/usr/bin/env python3
"""Phase 1-2 orchestrator: extract and curate training data.

Usage:
    python3 extract_pipeline.py                          # all TypeScript topics
    python3 extract_pipeline.py --topics typescript.fp   # single topic
    python3 extract_pipeline.py --dry-run                # estimate API cost only
    python3 extract_pipeline.py --skip-instruct          # skip instruction generation
    python3 extract_pipeline.py --full-history            # include git diffs
"""

import argparse
import logging
import sys
from pathlib import Path

from lib.common.types import TopicConfig
from lib.common.clone import clone_repos
from lib.common.dedup import deduplicate, load_held_out_fingerprints
from lib.common.balance import balance_domains
from lib.common.instruct import generate_instructions

from lib.typescript.walk import walk_ts_files
from lib.typescript.extract import extract_units_from_file, extract_diffs
from lib.typescript.score import filter_by_quality

from app.typescript import ALL_TOPICS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pipeline")

ROOT = Path(__file__).parent
REPOS_DIR = ROOT / "repos"
DATASET_DIR = ROOT / "dataset"
HELD_OUT_DIR = ROOT / "eval" / "held_out"


def load_topics(names: list[str] | None) -> list[TopicConfig]:
    """Load topic configs by name, or all if None."""
    if not names:
        return ALL_TOPICS

    by_name = {t.name: t for t in ALL_TOPICS}
    topics = []
    for name in names:
        if name not in by_name:
            log.error("Unknown topic '%s'. Available: %s", name, list(by_name.keys()))
            sys.exit(1)
        topics.append(by_name[name])
    return topics


def check_held_out():
    if not HELD_OUT_DIR.exists() or not any(HELD_OUT_DIR.iterdir()):
        log.error(
            "Held-out eval set not found at %s. "
            "Create eval/held_out/ JSON files BEFORE running extraction.",
            HELD_OUT_DIR,
        )
        sys.exit(1)
    log.info("Held-out eval set found at %s", HELD_OUT_DIR)


def run_topic(topic: TopicConfig, full_history: bool) -> list:
    """Run extraction and scoring for a single topic. Returns scored units."""
    log.info("--- Topic: %s ---", topic.name)

    # Clone
    repos = clone_repos(topic.repos, REPOS_DIR, full_history=full_history)
    if not repos:
        log.warning("No repos cloned for %s", topic.name)
        return []

    # Walk and extract
    units = []
    for repo in repos:
        repo_path = REPOS_DIR / repo.name
        files = walk_ts_files(repo_path, topic)
        for file_info in files:
            extracted = extract_units_from_file(file_info, topic.name)
            units.extend(extracted)

        if full_history:
            diffs = extract_diffs(repo_path, topic.name)
            units.extend(diffs)

    log.info("Raw units for %s: %d", topic.name, len(units))

    # Score
    scored = filter_by_quality(units, topic)
    return scored


def main():
    parser = argparse.ArgumentParser(description="Extract and curate training data")
    parser.add_argument("--topics", nargs="+", help="Topic names to process (default: all)")
    parser.add_argument("--dry-run", action="store_true", help="Estimate API cost without calling")
    parser.add_argument("--skip-instruct", action="store_true", help="Skip instruction generation")
    parser.add_argument("--full-history", action="store_true", help="Clone full git history for diffs")
    args = parser.parse_args()

    check_held_out()

    topics = load_topics(args.topics)
    log.info("=== Phase 1-2: %d topics ===", len(topics))

    # Extract and score per topic
    all_units = []
    for topic in topics:
        scored = run_topic(topic, args.full_history)
        all_units.extend(scored)

    log.info("Total scored units across all topics: %d", len(all_units))

    # Dedup (with held-out exclusion)
    held_out_fps = load_held_out_fingerprints(HELD_OUT_DIR)
    deduped = deduplicate(all_units, held_out_fps=held_out_fps)

    # Balance
    balanced = balance_domains(deduped)
    log.info("Final dataset size: %d units", len(balanced))

    if args.skip_instruct:
        log.info("Skipping instruction generation (--skip-instruct)")
        return

    # Collect domain validation terms from topic configs
    domain_terms = {}
    for topic in topics:
        module = __import__(f"app.typescript.{topic.name.split('.')[-1]}.config", fromlist=["DOMAIN_TERMS"])
        domain_terms[topic.name] = getattr(module, "DOMAIN_TERMS", [])

    # Generate instructions
    results = generate_instructions(
        units=balanced,
        output_path=DATASET_DIR / "typescript_training.jsonl",
        metadata_path=DATASET_DIR / "metadata.jsonl",
        rejected_path=DATASET_DIR / "rejected.jsonl",
        domain_terms=domain_terms,
        dry_run=args.dry_run,
    )

    if results:
        log.info("Pipeline complete. Training data: %s", DATASET_DIR / "typescript_training.jsonl")


if __name__ == "__main__":
    main()
