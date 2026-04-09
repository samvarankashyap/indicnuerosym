#!/usr/bin/env python3
"""
Run All — Full validation pipeline in one command
===================================================

Runs all validation levels sequentially and generates a consolidated report.
Order: Level 1 -> Level 2 -> Level 4 -> Level 3 -> Word TTR -> Report

Level 4 runs before Level 3 because Level 3 is the slowest (model loading).

Usage:
    python run_all.py
    python run_all.py --dataset ../datasets/your_data.json --limit 100
    python run_all.py --skip-level3
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from validation_utils import DEFAULT_DATASET, RESULTS_DIR


def main():
    parser = argparse.ArgumentParser(
        description="Run full validation pipeline (all levels + report)",
    )
    parser.add_argument(
        "--dataset", "-d", default=DEFAULT_DATASET,
        help="Path to dataset JSON",
    )
    parser.add_argument(
        "--output-dir", "-o", default=str(RESULTS_DIR),
        help="Directory for output JSON results",
    )
    parser.add_argument(
        "--limit", "-n", type=int, default=None,
        help="Process only first N records",
    )
    parser.add_argument(
        "--skip-level3", action="store_true",
        help="Skip Level 3 (semantic similarity) for faster runs",
    )
    parser.add_argument(
        "--batch-size", "-b", type=int, default=256,
        help="Batch size for Level 3 model encoding (default: 256)",
    )
    parser.add_argument(
        "--report-output", default=None,
        help="Report output path (default: dataset_validation_scripts/validation_report.txt)",
    )
    args = parser.parse_args()

    start_time = time.time()

    # Build common args for level scripts
    common_args = ["--dataset", args.dataset, "--output-dir", args.output_dir]
    if args.limit:
        common_args.extend(["--limit", str(args.limit)])

    # ── Level 1 ───────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  RUNNING LEVEL 1: CHANDASS SCANNER")
    print("=" * 80 + "\n")
    from level1_chandass_validation import main as run_level1
    run_level1(common_args)

    # ── Level 2 ───────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  RUNNING LEVEL 2: SANITY CHECKS")
    print("=" * 80 + "\n")
    from level2_sanity_checks import main as run_level2
    run_level2(common_args)

    # ── Level 4 (before Level 3 — faster, no large model loading) ────
    print("\n" + "=" * 80)
    print("  RUNNING LEVEL 4: LEXICAL DIVERSITY")
    print("=" * 80 + "\n")
    from level4_lexical_diversity import main as run_level4
    run_level4(common_args)

    # ── Level 3 ───────────────────────────────────────────────────────
    if not args.skip_level3:
        print("\n" + "=" * 80)
        print("  RUNNING LEVEL 3: SEMANTIC SIMILARITY")
        print("=" * 80 + "\n")
        l3_args = common_args + ["--batch-size", str(args.batch_size)]
        from level3_semantic_similarity import main as run_level3
        try:
            run_level3(l3_args)
        except ImportError as e:
            print(f"  SKIPPED Level 3: {e}")
    else:
        print("\n  Skipping Level 3 (--skip-level3 flag)")

    # ── Word TTR ──────────────────────────────────────────────────────
    # Only run if the JSONL dataset exists
    from validation_utils import DATASETS_DIR
    jsonl_dataset = DATASETS_DIR / "dwipada_master_deduplicated.jsonl"
    if jsonl_dataset.exists():
        print("\n" + "=" * 80)
        print("  RUNNING WORD TTR ANALYSIS")
        print("=" * 80 + "\n")
        from word_ttr import main as run_word_ttr
        run_word_ttr([
            "--dataset", str(jsonl_dataset),
            "--output", str(Path(args.output_dir) / "word_ttr_report.md"),
        ])
    else:
        print(f"\n  Skipping Word TTR (JSONL dataset not found: {jsonl_dataset})")

    # ── Report ────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  GENERATING CONSOLIDATED REPORT")
    print("=" * 80 + "\n")
    report_args = ["--results-dir", args.output_dir]
    if args.report_output:
        report_args.extend(["--output", args.report_output])
    from report import main as run_report
    run_report(report_args)

    # ── Summary ───────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    mins, secs = divmod(int(elapsed), 60)
    print("\n" + "=" * 80)
    print(f"  ALL DONE — Total runtime: {mins}m {secs}s")
    print("=" * 80)


if __name__ == "__main__":
    main()
