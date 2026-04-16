#!/usr/bin/env python3
"""Phase 1-2 orchestrator: extract and curate training data.

Usage:
    python3 extract_pipeline.py                          # all TypeScript topics
    python3 extract_pipeline.py --topics typescript.fp   # single topic
    python3 extract_pipeline.py --dry-run                # estimate API cost only
    python3 extract_pipeline.py --skip-instruct          # skip instruction generation
    python3 extract_pipeline.py --full-history           # include git diffs
"""

import argparse
import logging
import sys
from pathlib import Path

from lib.common.types import TopicConfig, LanguageModule
from lib.common.clone import clone_repos
from lib.common.dedup import deduplicate, load_held_out_fingerprints
from lib.common.balance import balance_domains
from lib.common.instruct import generate_instructions

from lib.typescript import TypeScriptModule
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

# Register language modules here. Add new languages by importing and registering.
LANGUAGES: dict[str, LanguageModule] = {
    "typescript": TypeScriptModule(),
}


def get_language_module(language: str) -> LanguageModule:
    if language not in LANGUAGES:
        log.error("No language module for '%s'. Available: %s", language, list(LANGUAGES.keys()))
        sys.exit(1)
    return LANGUAGES[language]


def load_topics(names: list[str] | None) -> list[TopicConfig]:
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


def check_held_out() -> bool:
    """Check if held-out eval set exists. Returns True if present."""
    if not HELD_OUT_DIR.exists() or not any(HELD_OUT_DIR.iterdir()):
        log.warning(
            "Held-out eval set not found at %s. "
            "Training data will NOT be filtered against eval examples. "
            "Create eval/held_out/ JSON files before training to prevent contamination.",
            HELD_OUT_DIR,
        )
        return False
    log.info("Held-out eval set found at %s", HELD_OUT_DIR)
    return True


def run_topic(topic: TopicConfig, lang: LanguageModule, full_history: bool) -> list:
    """Run extraction and scoring for a single topic."""
    log.info("--- Topic: %s ---", topic.name)

    repos = clone_repos(topic.repos, REPOS_DIR, full_history=full_history)
    if not repos:
        log.warning("No repos cloned for %s", topic.name)
        return []

    units = []
    for repo in repos:
        repo_path = REPOS_DIR / repo.name
        files = lang.walk(repo_path, topic)
        for file_info in files:
            extracted = lang.extract(file_info, topic.name)
            units.extend(extracted)

        if full_history:
            diffs = lang.extract_diffs(repo_path, topic.name)
            units.extend(diffs)

    log.info("Raw units for %s: %d", topic.name, len(units))

    scored = lang.score(units, topic)
    return scored


def main():
    parser = argparse.ArgumentParser(description="Extract and curate training data")
    parser.add_argument("--topics", nargs="+", help="Topic names to process (default: all)")
    parser.add_argument("--dry-run", action="store_true", help="Estimate API cost without calling")
    parser.add_argument("--skip-instruct", action="store_true", help="Skip instruction generation")
    parser.add_argument("--full-history", action="store_true", help="Clone full git history for diffs")
    args = parser.parse_args()

    has_held_out = check_held_out()

    topics = load_topics(args.topics)
    log.info("=== Phase 1-2: %d topics ===", len(topics))

    all_units = []
    for topic in topics:
        lang = get_language_module(topic.language)
        scored = run_topic(topic, lang, args.full_history)
        all_units.extend(scored)

    log.info("Total scored units across all topics: %d", len(all_units))

    held_out_fps = load_held_out_fingerprints(HELD_OUT_DIR) if has_held_out else set()
    deduped = deduplicate(all_units, held_out_fps=held_out_fps)

    balanced = balance_domains(deduped)
    log.info("Final dataset size: %d units", len(balanced))

    if args.skip_instruct:
        log.info("Skipping instruction generation (--skip-instruct)")
        return

    # Collect domain validation terms from topic configs
    domain_terms = {}
    for topic in topics:
        # topic.name is "typescript.fp" -> language="typescript", domain="fp"
        parts = topic.name.split(".")
        module_path = f"app.{'.'.join(parts)}.config"
        module = __import__(module_path, fromlist=["DOMAIN_TERMS"])
        domain_terms[topic.name] = getattr(module, "DOMAIN_TERMS", [])

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
