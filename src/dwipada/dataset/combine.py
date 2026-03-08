#!/usr/bin/env python3
"""Combine augmented and synthetic datasets with source tags.

Merges dwipada_augmented_perfect_dataset.json and synthetic_dwipada_dataset.json
into dwipada_augmented_dataset.json. Tags each record with its source work
(from consolidated_dwipada.json) and is_synthetic_data flag.
"""

import json
from collections import Counter

from dwipada.core.analyzer import analyze_dwipada
from dwipada.dataset.augment import build_chandassu
from dwipada.paths import DATA_DIR, DATASETS_DIR

CONSOLIDATED_PATH = DATA_DIR / "consolidated_dwipada.json"
AUGMENTED_PATH = DATASETS_DIR / "dwipada_augmented_perfect_dataset.json"
SYNTHETIC_PATH = DATASETS_DIR / "synthetic_dwipada_dataset.json"
OUTPUT_PATH = DATASETS_DIR / "dwipada_augmented_dataset.json"


def normalize_poem(text: str) -> str:
    """Normalize poem text for matching: strip spaces around newlines."""
    lines = [line.strip() for line in text.strip().split("\n")]
    return "\n".join(lines)


def build_source_lookup(consolidated: list[dict]) -> dict[str, str]:
    """Build normalized poem text -> work name lookup from consolidated data."""
    lookup = {}
    for record in consolidated:
        key = normalize_poem(record["poem"])
        work = record["source"]["work"]
        lookup[key] = work
    return lookup


def main():
    # 1. Build source lookup
    print("Loading consolidated_dwipada.json for source lookup...")
    with open(CONSOLIDATED_PATH, encoding="utf-8") as f:
        consolidated = json.load(f)
    source_lookup = build_source_lookup(consolidated)
    print(f"  Built lookup: {len(source_lookup)} unique poems")

    # 2. Load augmented dataset
    print("Loading augmented dataset...")
    with open(AUGMENTED_PATH, encoding="utf-8") as f:
        augmented = json.load(f)
    print(f"  {len(augmented)} records")

    # 3. Load synthetic dataset
    print("Loading synthetic dataset...")
    with open(SYNTHETIC_PATH, encoding="utf-8") as f:
        synthetic = json.load(f)
    print(f"  {len(synthetic)} records")

    output = []
    source_counts = Counter()
    unmatched = 0

    # 4. Process augmented records
    print("\nProcessing augmented records...")
    for record in augmented:
        source = source_lookup.get(normalize_poem(record["poem"]), "unknown")
        if source == "unknown":
            unmatched += 1
        source_counts[source] += 1
        output.append({
            "prompt": record["prompt"],
            "poem": record["poem"],
            "word_to_word_meaning": record["word_to_word_meaning"],
            "telugu_meaning": record["telugu_meaning"],
            "english_meaning": record["english_meaning"],
            "chandassu_analysis": record["chandassu_analysis"],
            "source": source,
            "is_synthetic_data": False,
        })

    # 5. Process synthetic records (regenerate chandassu)
    print("Processing synthetic records (regenerating chandassu)...")
    syn_errors = 0
    for i, record in enumerate(synthetic):
        try:
            analysis = analyze_dwipada(record["poem"])
            chandassu = build_chandassu(analysis)
        except Exception:
            chandassu = record["chandassu_analysis"]  # fallback to LLM version
            syn_errors += 1

        source_counts["synthetic"] += 1
        output.append({
            "prompt": record["prompt"],
            "poem": record["poem"],
            "word_to_word_meaning": record["word_to_word_meaning"],
            "telugu_meaning": record["telugu_meaning"],
            "english_meaning": record["english_meaning"],
            "chandassu_analysis": chandassu,
            "source": "synthetic",
            "is_synthetic_data": True,
        })

    # 6. Write output
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    # 7. Summary
    print(f"\nResults:")
    print(f"  Total records: {len(output)}")
    print(f"  Unmatched sources: {unmatched}")
    if syn_errors:
        print(f"  Synthetic chandassu regen errors (used LLM fallback): {syn_errors}")
    print(f"\n  Per-source breakdown:")
    for source, count in source_counts.most_common():
        print(f"    {source}: {count}")
    print(f"\n  Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
