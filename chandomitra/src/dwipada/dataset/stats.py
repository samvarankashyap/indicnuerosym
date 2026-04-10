#!/usr/bin/env python3
"""Generate statistics for dwipada analysis on datasets.

Combines per-source stats (from consolidated data) and dataset-level stats
(perfect/valid counts with optional filtered output).
"""

import argparse
import json
import sys

from dwipada.core.analyzer import analyze_dwipada
from dwipada.paths import DATA_DIR, DATASETS_DIR


def run_stats(input_file=None, exclude_sources=None):
    """Print per-source dwipada analysis stats.

    Args:
        input_file: Path to consolidated dataset JSON. Defaults to
            DATA_DIR / "consolidated_dwipada.json".
        exclude_sources: Optional set of source work names to exclude.
    """
    if input_file is None:
        input_file = DATA_DIR / "consolidated_dwipada.json"

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    total = 0
    perfect = 0
    valid = 0
    by_source = {}

    for entry in data:
        source = entry.get("source", {}).get("work", "unknown")
        if exclude_sources and source in exclude_sources:
            continue

        result = analyze_dwipada(entry["poem"])
        total += 1

        if source not in by_source:
            by_source[source] = {"total": 0, "perfect": 0, "valid": 0}
        by_source[source]["total"] += 1

        if result.get("is_valid_dwipada"):
            valid += 1
            by_source[source]["valid"] += 1
        if result.get("match_score", {}).get("overall") == 100.0:
            perfect += 1
            by_source[source]["perfect"] += 1

    # Print results
    print(f"Total couplets: {total}")
    print(f"Perfect dwipada (100% score): {perfect} ({perfect/total*100:.1f}%)")
    print(f"Valid dwipada (is_valid=True): {valid} ({valid/total*100:.1f}%)")
    print()
    print(
        f"{'Source':<30} {'Total':>6} {'Perfect':>8} {'Valid':>8} {'Perf%':>7} {'Valid%':>7}"
    )
    print("-" * 70)
    for src in sorted(by_source.keys()):
        s = by_source[src]
        pp = s["perfect"] / s["total"] * 100 if s["total"] else 0
        vp = s["valid"] / s["total"] * 100 if s["total"] else 0
        print(
            f"{src:<30} {s['total']:>6} {s['perfect']:>8} {s['valid']:>8} {pp:>6.1f}% {vp:>6.1f}%"
        )

    if exclude_sources:
        print(f"\n(Excluded: {', '.join(exclude_sources)})")


def run_dataset_stats(input_file=None, output_file=None):
    """Compute perfect/valid counts and optionally write filtered output.

    Args:
        input_file: Path to master dataset JSON. Defaults to
            DATASETS_DIR / "dwipada_master_dataset.json".
        output_file: If provided, write perfect records to this path.
            If None, writes to DATASETS_DIR / "dwipada_master_filtered_perfect_dataset.json".
    """
    if input_file is None:
        input_file = DATASETS_DIR / "dwipada_master_dataset.json"

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    total = 0
    perfect = 0
    valid = 0
    perfect_records = []

    for entry in data:
        poem = entry["poem"]
        result = analyze_dwipada(poem)
        total += 1

        if result.get("is_valid_dwipada"):
            valid += 1
        if result.get("match_score", {}).get("overall") == 100.0:
            perfect += 1
            perfect_records.append(entry)

    print(f"Total couplets: {total}")
    print(f"Perfect dwipada (100% score): {perfect} ({perfect/total*100:.1f}%)")
    print(f"Valid dwipada (is_valid=True): {valid} ({valid/total*100:.1f}%)")

    if output_file is None:
        output_file = DATASETS_DIR / "dwipada_master_filtered_perfect_dataset.json"

    # Write perfect records to filtered dataset
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(perfect_records, f, ensure_ascii=False, indent=2)
    print(f"\nPerfect records written to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Dwipada dataset statistics."
    )
    parser.add_argument(
        "input", nargs="?", default=None,
        help="Input dataset JSON file (default depends on mode).",
    )
    parser.add_argument(
        "--by-source", action="store_true",
        help="Run per-source stats on consolidated dataset.",
    )
    parser.add_argument(
        "--write-filtered", action="store_true",
        help="Compute perfect/valid counts and write filtered output.",
    )
    parser.add_argument(
        "--exclude", default=None,
        help="Comma-separated source names to exclude (only for --by-source).",
    )
    args = parser.parse_args()

    exclude = None
    if args.exclude:
        exclude = set(args.exclude.split(","))

    if args.by_source:
        run_stats(input_file=args.input, exclude_sources=exclude)
    elif args.write_filtered:
        run_dataset_stats(input_file=args.input)
    else:
        # Default: run both
        print("=== Per-source stats ===\n")
        run_stats(input_file=args.input, exclude_sources=exclude)
        print("\n\n=== Dataset stats (with filtered output) ===\n")
        run_dataset_stats(input_file=args.input)


if __name__ == "__main__":
    main()
