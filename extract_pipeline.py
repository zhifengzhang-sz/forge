#!/usr/bin/env python3
"""Phase 1-2 orchestrator: extract and curate TypeScript training data.

Usage:
    python extract_pipeline.py                  # full pipeline
    python extract_pipeline.py --dry-run        # estimate API cost only
    python extract_pipeline.py --skip-instruct  # extract + curate, skip instruction generation
"""

import argparse
import logging
import sys
from pathlib import Path

from extract.clone import clone_repos
from extract.walk import walk_ts_files
from extract.extract import extract_units_from_file, extract_diffs
from extract.score import filter_by_quality
from extract.dedup import deduplicate, load_held_out_fingerprints
from extract.balance import balance_domains
from extract.instruct import generate_instructions

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


def check_held_out():
    """Verify that the held-out eval set exists before extraction."""
    if not HELD_OUT_DIR.exists() or not any(HELD_OUT_DIR.iterdir()):
        log.error(
            "Held-out eval set not found at %s. "
            "Create eval/held_out/{fp,reactive,xstate,eventsourcing}.json BEFORE running extraction. "
            "See docs/design.md Section 11.",
            HELD_OUT_DIR,
        )
        sys.exit(1)
    log.info("Held-out eval set found at %s", HELD_OUT_DIR)


def main():
    parser = argparse.ArgumentParser(description="Extract and curate TypeScript training data")
    parser.add_argument("--dry-run", action="store_true", help="Estimate API cost without calling")
    parser.add_argument("--skip-instruct", action="store_true", help="Skip instruction generation")
    parser.add_argument("--full-history", action="store_true", help="Clone full git history for diff extraction")
    args = parser.parse_args()

    check_held_out()

    # Phase 1: Clone
    log.info("=== Phase 1: Data Collection ===")
    repos = clone_repos(REPOS_DIR, full_history=args.full_history)
    if not repos:
        log.error("No repositories cloned successfully")
        sys.exit(1)

    # Phase 1: Walk and extract
    all_units = []
    for repo in repos:
        files = walk_ts_files(repo["path"], repo["domain"])
        for file_info in files:
            units = extract_units_from_file(file_info)
            all_units.extend(units)

        if args.full_history:
            diffs = extract_diffs(repo["path"], repo["domain"])
            all_units.extend(diffs)

    log.info("Total raw units extracted: %d", len(all_units))

    # Phase 2: Score
    log.info("=== Phase 2: Dataset Curation ===")
    scored = filter_by_quality(all_units)

    # Phase 2: Dedup (excluding held-out eval set)
    held_out_fps = load_held_out_fingerprints(HELD_OUT_DIR)
    deduped = deduplicate(scored, held_out_fps=held_out_fps)

    # Phase 2: Balance
    balanced = balance_domains(deduped)

    log.info("Final dataset size: %d units", len(balanced))

    if args.skip_instruct:
        log.info("Skipping instruction generation (--skip-instruct)")
        return

    # Phase 2: Generate instructions
    results = generate_instructions(
        units=balanced,
        output_path=DATASET_DIR / "typescript_training.jsonl",
        metadata_path=DATASET_DIR / "metadata.jsonl",
        rejected_path=DATASET_DIR / "rejected.jsonl",
        dry_run=args.dry_run,
    )

    if results:
        log.info("Pipeline complete. Training data: %s", DATASET_DIR / "typescript_training.jsonl")


if __name__ == "__main__":
    main()
