#!/usr/bin/env python3
"""
Level 1: Chandass Scanner — Prosodic Structure Validation
==========================================================

Validates the prosodic structure of every dwipada poem using the
dwipada analyzer. Checks gana sequence, prasa, yati, and overall
match score.

Usage:
    python level1_chandass_validation.py
    python level1_chandass_validation.py --dataset ../datasets/your_data.json --limit 100
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
from statistics import mean

# Ensure local imports work regardless of cwd
sys.path.insert(0, str(Path(__file__).resolve().parent))

from validation_utils import (
    DEFAULT_DATASET,
    RESULTS_DIR,
    format_section_header,
    load_dataset,
    save_results,
)

try:
    from dwipada.core.analyzer import analyze_dwipada
except ImportError:
    sys.exit(
        "ERROR: dwipada package not installed.\n"
        "Run: pip install -e . from the project root"
    )


def run_level1(data: list[dict]) -> dict:
    """Validate prosodic structure of every poem using dwipada_analyzer.

    Checks per record:
      - is_valid_dwipada (all rules satisfied)
      - Gana sequence (3 Indra + 1 Surya per line)
      - Prasa (2nd consonant match across lines)
      - Yati (1st letter of gana 1 == 1st letter of gana 3, per line)
      - Overall match score (0-100)
    """
    print(format_section_header("LEVEL 1: CHANDASS SCANNER (Structural Integrity)"))

    total = len(data)
    valid_count = 0
    overall_scores = []
    prasa_scores = []
    errors = []
    per_record = []

    for i, record in enumerate(data):
        poem = record["poem"]
        try:
            result = analyze_dwipada(poem)
        except Exception as e:
            errors.append({"index": i, "error": str(e), "poem": poem[:60]})
            per_record.append({
                "index": i, "is_valid": False, "overall_score": 0,
                "gana_l1": 0, "gana_l2": 0, "prasa": 0,
                "yati_l1": 0, "yati_l2": 0,
            })
            continue

        is_valid = result.get("is_valid_dwipada", False)
        score = result.get("match_score", {})
        overall = score.get("overall", 0)
        breakdown = score.get("breakdown", {})

        g1 = breakdown.get("gana_line1", 0)
        g2 = breakdown.get("gana_line2", 0)
        pr = breakdown.get("prasa", 0)
        y1 = breakdown.get("yati_line1", 0)
        y2 = breakdown.get("yati_line2", 0)

        if is_valid:
            valid_count += 1
        overall_scores.append(overall)
        prasa_scores.append(pr)

        per_record.append({
            "index": i, "is_valid": is_valid, "overall_score": overall,
            "gana_l1": g1, "gana_l2": g2, "prasa": pr,
            "yati_l1": y1, "yati_l2": y2,
        })

        if (i + 1) % 5000 == 0:
            print(f"  Level 1 progress: {i + 1}/{total}")

    gana_pass = sum(1 for r in per_record if r["gana_l1"] == 100 and r["gana_l2"] == 100)
    prasa_pass = sum(1 for s in prasa_scores if s == 100)
    yati_pass = sum(1 for r in per_record if r["yati_l1"] == 100 and r["yati_l2"] == 100)

    stats = {
        "total": total,
        "valid_count": valid_count,
        "avg_overall_score": mean(overall_scores) if overall_scores else 0,
        "gana_pass": gana_pass,
        "prasa_pass": prasa_pass,
        "yati_pass": yati_pass,
        "overall_scores": overall_scores,
        "errors": errors,
        "per_record": per_record,
    }

    print(f"  Done. Valid: {valid_count}/{total}, Errors: {len(errors)}")
    return stats


def main(args=None):
    parser = argparse.ArgumentParser(
        description="Level 1: Chandass Scanner — Prosodic Structure Validation",
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

    results = run_level1(data)

    meta = {
        "level": "level1",
        "dataset": str(Path(args.dataset).resolve()),
        "total_records": len(data),
        "limit": args.limit,
        "timestamp": datetime.now().isoformat(),
    }
    output_path = str(Path(args.output_dir) / "level1.json")
    save_results(results, output_path, meta)


if __name__ == "__main__":
    main()
