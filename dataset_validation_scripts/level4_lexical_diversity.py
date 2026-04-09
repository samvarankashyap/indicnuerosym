#!/usr/bin/env python3
"""
Level 4: Lexical Diversity — TTR, Duplicates, Telugu Token Coverage
====================================================================

Analyses lexical diversity using Gemma 3 tokenizer and duplicate detection.

Metrics:
  - Telugu Token Coverage: fraction of Gemma 3's Telugu tokens used by the dataset
  - Per-poem average TTR (Type-Token Ratio)
  - Exact duplicate poem detection
  - Near-duplicate detection (normalized text comparison)

Usage:
    python level4_lexical_diversity.py
    python level4_lexical_diversity.py --dataset ../datasets/your_data.json --limit 100
"""

import argparse
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parent))

from validation_utils import (
    DEFAULT_DATASET,
    RESULTS_DIR,
    count_telugu_tokens,
    format_section_header,
    gemma_tokenize,
    load_dataset,
    load_gemma_tokenizer,
    save_results,
)


def run_level4(data: list[dict]) -> dict:
    """Analyse lexical diversity using Gemma 3 tokenizer and duplicate detection."""
    print(format_section_header("LEVEL 4: LEXICAL DIVERSITY (Corpus Health)"))

    tokenizer = load_gemma_tokenizer()

    # Count Telugu tokens in vocabulary
    print("  Counting Telugu tokens in Gemma 3 vocabulary...")
    total_telugu_tokens = count_telugu_tokens(tokenizer)
    print(f"  Telugu tokens in vocabulary: {total_telugu_tokens}")

    # Tokenize all poems
    all_token_ids = []
    unique_token_ids = set()
    per_poem_ttrs = []

    total = len(data)
    for i, record in enumerate(data):
        poem = record["poem"]
        ids = gemma_tokenize(tokenizer, poem)

        all_token_ids.extend(ids)
        unique_token_ids.update(ids)

        if len(ids) > 0:
            per_poem_ttrs.append(len(set(ids)) / len(ids))
        else:
            per_poem_ttrs.append(0.0)

        if (i + 1) % 5000 == 0:
            print(f"  Tokenization progress: {i + 1}/{total}")

    telugu_coverage = len(unique_token_ids) / total_telugu_tokens if total_telugu_tokens else 0
    avg_poem_ttr = mean(per_poem_ttrs) if per_poem_ttrs else 0

    # Exact duplicates
    poem_counter = Counter(record["poem"] for record in data)
    exact_dup_groups = []
    exact_dup_total = 0
    for poem_text, count in poem_counter.items():
        if count > 1:
            indices = [i for i, r in enumerate(data) if r["poem"] == poem_text]
            exact_dup_groups.append({"poem": poem_text[:80], "count": count, "indices": indices})
            exact_dup_total += count

    # Near duplicates (normalize: strip spaces, punctuation, newlines)
    def normalize(text: str) -> str:
        text = text.replace("\n", "").replace(" ", "")
        text = re.sub(r"[,.\-;:!?()\"'।॥]", "", text)
        return text

    norm_map: dict[str, list[int]] = {}
    for i, record in enumerate(data):
        key = normalize(record["poem"])
        norm_map.setdefault(key, []).append(i)

    near_dup_groups = []
    near_dup_total = 0
    for norm_text, indices in norm_map.items():
        if len(indices) > 1:
            poem_sample = data[indices[0]]["poem"][:80]
            near_dup_groups.append({"poem": poem_sample, "count": len(indices), "indices": indices})
            near_dup_total += len(indices)

    stats = {
        "total_tokens": len(all_token_ids),
        "unique_tokens": len(unique_token_ids),
        "total_telugu_tokens": total_telugu_tokens,
        "telugu_coverage": telugu_coverage,
        "avg_poem_ttr": avg_poem_ttr,
        "per_poem_ttrs": per_poem_ttrs,
        "exact_dup_groups": sorted(exact_dup_groups, key=lambda x: -x["count"]),
        "exact_dup_poem_count": exact_dup_total,
        "near_dup_groups": sorted(near_dup_groups, key=lambda x: -x["count"]),
        "near_dup_poem_count": near_dup_total,
    }

    print(f"  Done. Telugu Token Coverage: {telugu_coverage:.1%} ({len(unique_token_ids)}/{total_telugu_tokens}), "
          f"Exact dups: {exact_dup_total}, Near dups: {near_dup_total}")
    return stats


def main(args=None):
    parser = argparse.ArgumentParser(
        description="Level 4: Lexical Diversity — TTR, Duplicates, Token Coverage",
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

    results = run_level4(data)

    meta = {
        "level": "level4",
        "dataset": str(Path(args.dataset).resolve()),
        "total_records": len(data),
        "limit": args.limit,
        "timestamp": datetime.now().isoformat(),
    }
    output_path = str(Path(args.output_dir) / "level4.json")
    save_results(results, output_path, meta)


if __name__ == "__main__":
    main()
