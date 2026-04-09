#!/usr/bin/env python3
"""
Level 2: Sanity Checks — Field Completeness & Length Ratios
============================================================

Validates field completeness and prose-verse length alignment.
Checks length ratio (telugu_meaning / poem), non-empty telugu_meaning,
and non-empty english_meaning.

Usage:
    python level2_sanity_checks.py
    python level2_sanity_checks.py --dataset ../datasets/your_data.json --limit 100
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
from statistics import mean, median

sys.path.insert(0, str(Path(__file__).resolve().parent))

from validation_utils import (
    DEFAULT_DATASET,
    RESULTS_DIR,
    format_section_header,
    load_dataset,
    save_results,
)

# Thresholds
LENGTH_RATIO_MIN = 0.8
LENGTH_RATIO_MAX = 2.5


def run_level2(data: list[dict]) -> dict:
    """Validate field completeness and prose-verse length alignment.

    Checks per record:
      - Length ratio: len(telugu_meaning) / len(poem) in (0.8, 2.5)
      - telugu_meaning is a non-empty string
      - english_meaning is a non-empty string
    """
    print(format_section_header("LEVEL 2: SANITY CHECKS (Coarse Alignment)"))

    total = len(data)
    ratios = []
    ratio_pass_count = 0
    telugu_pass = 0
    english_pass = 0
    all_pass = 0
    per_record = []
    outliers = []

    for i, record in enumerate(data):
        poem = record.get("poem", "")
        telugu = record.get("telugu_meaning", "")
        english = record.get("english_meaning", "")

        # Strip whitespace for fair character count
        poem_clean = poem.replace(" ", "").replace("\n", "")
        telugu_clean = telugu.replace(" ", "").replace("\n", "")

        # Length ratio
        ratio = len(telugu_clean) / len(poem_clean) if len(poem_clean) > 0 else 0.0
        ratio_ok = LENGTH_RATIO_MIN < ratio < LENGTH_RATIO_MAX
        ratios.append(ratio)

        # Field checks
        telugu_ok = isinstance(telugu, str) and len(telugu.strip()) > 0
        english_ok = isinstance(english, str) and len(english.strip()) > 0

        if ratio_ok:
            ratio_pass_count += 1
        if telugu_ok:
            telugu_pass += 1
        if english_ok:
            english_pass += 1

        passes_all = ratio_ok and telugu_ok and english_ok
        if passes_all:
            all_pass += 1

        rec = {
            "index": i, "length_ratio": ratio, "ratio_pass": ratio_ok,
            "telugu_nonempty": telugu_ok,
            "english_nonempty": english_ok, "all_pass": passes_all,
        }
        per_record.append(rec)

        if not ratio_ok:
            outliers.append((i, ratio, poem[:60]))

    # Sort outliers by distance from midpoint
    outliers.sort(key=lambda x: abs(x[1] - 1.4))
    outliers.reverse()

    stats = {
        "total": total,
        "ratio_pass_count": ratio_pass_count,
        "telugu_pass": telugu_pass,
        "english_pass": english_pass,
        "all_pass": all_pass,
        "ratios": ratios,
        "avg_ratio": mean(ratios) if ratios else 0,
        "median_ratio": median(ratios) if ratios else 0,
        "min_ratio": min(ratios) if ratios else 0,
        "max_ratio": max(ratios) if ratios else 0,
        "outliers": outliers[:20],
        "per_record": per_record,
    }

    print(f"  Done. All pass: {all_pass}/{total}, Ratio outliers: {len(outliers)}")
    return stats


def main(args=None):
    parser = argparse.ArgumentParser(
        description="Level 2: Sanity Checks — Field Completeness & Length Ratios",
    )
    parser.add_argument(
        "--dataset", "-d", default=DEFAULT_DATASET,
        help="Path to dataset JSON",
    )
    parser.add_argument(
        "--output-dir", "-o", default=str(RESULTS_DIR),
        help="Directory for output JSON",
    )
    parser.add_argument(
        "--limit", "-n", type=int, default=None,
        help="Process only first N records",
    )
    args = parser.parse_args(args)

    print(f"Loading dataset: {args.dataset}")
    data = load_dataset(args.dataset)
    if args.limit:
        data = data[:args.limit]
        print(f"Limited to first {args.limit} records.")
    print(f"Records to validate: {len(data):,}")

    results = run_level2(data)

    meta = {
        "level": "level2",
        "dataset": str(Path(args.dataset).resolve()),
        "total_records": len(data),
        "limit": args.limit,
        "timestamp": datetime.now().isoformat(),
    }
    output_path = str(Path(args.output_dir) / "level2.json")
    save_results(results, output_path, meta)


if __name__ == "__main__":
    main()
