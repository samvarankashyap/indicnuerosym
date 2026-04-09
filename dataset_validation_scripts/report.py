#!/usr/bin/env python3
"""
Report Generator — Consolidate validation results into a report
================================================================

Reads level1-4 JSON results from the results directory and generates
a consolidated validation report matching the original format.

Usage:
    python report.py
    python report.py --results-dir results/ --output validation_report.txt
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from validation_utils import (
    LEVEL3_MODELS,
    RESULTS_DIR,
    SCRIPT_DIR,
    format_histogram,
    format_section_header,
    format_stat_line,
    load_results,
    write_report,
)

# Constants (duplicated from level scripts for report formatting)
LENGTH_RATIO_MIN = 0.8
LENGTH_RATIO_MAX = 2.5
COSINE_SIM_THRESHOLD = 0.65

PAIR_NAMES = [
    ("Poem ↔ Telugu Meaning", "poem", "telugu_meaning"),
    ("Poem ↔ English Meaning", "poem", "english_meaning"),
    ("Telugu Meaning ↔ English Meaning", "telugu_meaning", "english_meaning"),
]


def generate_report(
    l1: dict, l2: dict, l3: dict | None, l4: dict,
    total_records: int, dataset_path: str,
) -> list[str]:
    """Compile all level stats into a formatted text report."""
    lines = []

    # Header
    lines.append("=" * 80)
    lines.append("  DWIPADA DATASET VALIDATION REPORT")
    lines.append(f"  Dataset: {dataset_path}")
    lines.append(f"  Records: {total_records:,}  |  Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("=" * 80)
    lines.append("")

    # ── Level 1 ──────────────────────────────────────────────────────────
    lines.append(format_section_header("LEVEL 1: CHANDASS SCANNER (Structural Integrity)"))
    t = l1["total"]
    lines.append(format_stat_line("Valid Dwipada:", l1["valid_count"], t))
    lines.append(format_stat_line("Avg Overall Score:", l1["avg_overall_score"]))
    lines.append("")
    lines.append("  Sub-check Pass Rates:")
    lines.append(format_stat_line("  Gana Sequence:", l1["gana_pass"], t, label_width=28))
    lines.append(format_stat_line("  Prasa Match:", l1["prasa_pass"], t, label_width=28))
    lines.append(format_stat_line("  Yati (both lines):", l1["yati_pass"], t, label_width=28))
    lines.append("")
    lines.append("  Score Distribution:")
    hist = format_histogram(l1["overall_scores"], [0, 50, 70, 80, 90, 100.01])
    lines.extend(hist)
    if l1["errors"]:
        lines.append(f"\n  Errors ({len(l1['errors'])}):")
        for e in l1["errors"][:10]:
            lines.append(f"    #{e['index']}: {e['error']} — {e['poem']}")
    lines.append("")

    # ── Level 2 ──────────────────────────────────────────────────────────
    lines.append(format_section_header("LEVEL 2: SANITY CHECKS (Coarse Alignment)"))
    t = l2["total"]
    lines.append(format_stat_line("All Sub-checks Pass:", l2["all_pass"], t))
    lines.append("")
    lines.append("  Sub-check Pass Rates:")
    lines.append(format_stat_line(f"  Length Ratio ({LENGTH_RATIO_MIN}-{LENGTH_RATIO_MAX}):", l2["ratio_pass_count"], t, label_width=28))
    lines.append(format_stat_line("  Non-empty Telugu:", l2["telugu_pass"], t, label_width=28))
    lines.append(format_stat_line("  Non-empty English:", l2["english_pass"], t, label_width=28))
    lines.append("")
    lines.append(f"  Length Ratio Stats:")
    lines.append(f"    Mean: {l2['avg_ratio']:.2f}  |  Median: {l2['median_ratio']:.2f}  "
                 f"|  Min: {l2['min_ratio']:.2f}  |  Max: {l2['max_ratio']:.2f}")
    lines.append("")
    lines.append("  Length Ratio Distribution:")
    hist = format_histogram(l2["ratios"], [0, 0.5, 0.8, 1.0, 1.5, 2.0, 5.01])
    lines.extend(hist)
    if l2["outliers"]:
        lines.append(f"\n  Top Outliers ({min(len(l2['outliers']), 10)}):")
        for idx, ratio, poem in l2["outliers"][:10]:
            lines.append(f"    #{idx}  ratio={ratio:.2f}  poem=\"{poem}...\"")
    lines.append("")

    # ── Level 3 ──────────────────────────────────────────────────────────
    if l3 is not None:
        lines.append(format_section_header(
            "LEVEL 3: SEMANTIC FIDELITY (3 Pairs × 3 Models)"))
        t = l3["total"]
        lines.append(format_stat_line(
            f"Pass (LaBSE Te↔En >= {COSINE_SIM_THRESHOLD}):", l3["pass_count"], t))
        lines.append("")

        for model_name, _model_id in LEVEL3_MODELS:
            if model_name not in l3["models"]:
                continue
            model_pairs = l3["models"][model_name]
            lines.append(f"  ┌─ {model_name} ──────────────────────────────────────────")
            for pair_name in [p[0] for p in PAIR_NAMES]:
                p = model_pairs[pair_name]
                lines.append(f"  │")
                lines.append(f"  │  {pair_name}")
                lines.append(f"  │    Pass (>= {COSINE_SIM_THRESHOLD}): "
                             f"{p['pass_count']:,} / {p['total']:,}  "
                             f"({p['pass_count']/p['total']*100:.1f}%)")
                lines.append(f"  │    Mean: {p['avg_sim']:.4f}  |  "
                             f"Median: {p['median_sim']:.4f}  |  "
                             f"Min: {p['min_sim']:.4f}  |  "
                             f"Max: {p['max_sim']:.4f}")
                lines.append(f"  │")
                hist = format_histogram(p["sim_list"], [0, 0.3, 0.5, 0.7, 0.85, 1.01])
                for h in hist:
                    lines.append(f"  │  {h}")
                lines.append(f"  │")
            lines.append(f"  └────────────────────────────────────────────────────")
            lines.append("")

        # Bottom 20 by anchor pair (LaBSE Telugu<->English)
        if "LaBSE" in l3["models"]:
            anchor = l3["models"]["LaBSE"]["Telugu Meaning ↔ English Meaning"]
            if anchor["bottom_20"]:
                lines.append("  Bottom 20 by LaBSE Telugu↔English Similarity:")
                for entry in anchor["bottom_20"]:
                    lines.append(f"    #{entry['index']}  sim={entry['sim']:.4f}  "
                                 f"poem=\"{entry['poem']}...\"")
                lines.append("")
    else:
        lines.append(format_section_header("LEVEL 3: SEMANTIC FIDELITY (Skipped)"))
        lines.append("  Skipped — no level3.json found in results.")
        lines.append("")

    # ── Level 4 ──────────────────────────────────────────────────────────
    lines.append(format_section_header("LEVEL 4: LEXICAL DIVERSITY (Corpus Health)"))
    lines.append(format_stat_line("Total Tokens (Gemma 3):", f"{l4['total_tokens']:,}"))
    lines.append(format_stat_line("Telugu Tokens in Vocab:", f"{l4['total_telugu_tokens']:,}"))
    lines.append(format_stat_line("Telugu Tokens Used:", f"{l4['unique_tokens']:,}"))
    te_cov = l4["telugu_coverage"]
    lines.append(format_stat_line("Telugu Token Coverage:", f"{te_cov:.1%} ({l4['unique_tokens']:,} / {l4['total_telugu_tokens']:,})"))
    lines.append(format_stat_line("Avg Per-Poem TTR:", l4["avg_poem_ttr"]))
    lines.append("")
    lines.append(f"  Exact Duplicates: {l4['exact_dup_poem_count']} poems "
                 f"in {len(l4['exact_dup_groups'])} groups")
    lines.append(f"  Near Duplicates:  {l4['near_dup_poem_count']} poems "
                 f"in {len(l4['near_dup_groups'])} groups")
    if l4["exact_dup_groups"]:
        lines.append(f"\n  Sample Duplicate Groups (top 5):")
        for grp in l4["exact_dup_groups"][:5]:
            lines.append(f"    {grp['count']}x  indices={grp['indices'][:5]}  "
                         f"poem=\"{grp['poem']}...\"")
    lines.append("")

    # ── Cross-Level Summary ──────────────────────────────────────────────
    lines.append(format_section_header("CROSS-LEVEL SUMMARY"))

    l1_pass_set = set(r["index"] for r in l1["per_record"] if r["is_valid"] and r["overall_score"] == 100)
    l2_pass_set = set(r["index"] for r in l2["per_record"] if r["all_pass"])
    if l3 is not None:
        l3_pass_set = set(r["index"] for r in l3["per_record"] if r["pass"])
    else:
        l3_pass_set = set(range(total_records))

    pass_all = l1_pass_set & l2_pass_set & l3_pass_set
    lines.append(format_stat_line("Pass ALL Levels:", len(pass_all), total_records))
    lines.append(format_stat_line("Fail Level 1 only:", len(l2_pass_set & l3_pass_set) - len(pass_all), total_records))
    lines.append(format_stat_line("Fail Level 2 only:", len(l1_pass_set & l3_pass_set) - len(pass_all), total_records))
    if l3 is not None:
        lines.append(format_stat_line("Fail Level 3 only:", len(l1_pass_set & l2_pass_set) - len(pass_all), total_records))
    lines.append("")
    lines.append("=" * 80)

    return lines


def main(args=None):
    parser = argparse.ArgumentParser(
        description="Consolidate validation results into a report",
    )
    parser.add_argument(
        "--results-dir", "-r", default=str(RESULTS_DIR),
        help="Directory containing level JSON results",
    )
    parser.add_argument(
        "--output", "-o", default=str(SCRIPT_DIR / "validation_report.txt"),
        help="Output report path",
    )
    args = parser.parse_args(args)

    results_dir = args.results_dir

    # Load level results
    l1 = load_results(str(Path(results_dir) / "level1.json"))
    l2 = load_results(str(Path(results_dir) / "level2.json"))
    l3 = load_results(str(Path(results_dir) / "level3.json"))
    l4 = load_results(str(Path(results_dir) / "level4.json"))

    if l1 is None or l2 is None or l4 is None:
        missing = []
        if l1 is None:
            missing.append("level1.json")
        if l2 is None:
            missing.append("level2.json")
        if l4 is None:
            missing.append("level4.json")
        print(f"ERROR: Missing required result files: {', '.join(missing)}")
        print(f"Run the corresponding level scripts first.")
        sys.exit(1)

    # Extract metadata from first available level
    meta = l1.get("_meta", {}) or l2.get("_meta", {}) or l4.get("_meta", {})
    dataset_path = meta.get("dataset", "unknown")
    total_records = meta.get("total_records", l1["total"])

    report = generate_report(l1, l2, l3, l4, total_records, dataset_path)
    write_report(report, args.output)


if __name__ == "__main__":
    main()
