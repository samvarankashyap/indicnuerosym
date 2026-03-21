#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Count Telugu dwipada poems whose second line starts with a consonant cluster (సంయుక్తాక్షరం).

A consonant cluster is: consonant + halant (్) + consonant (e.g., ప్ర, క్ష, శ్రీ).

Usage:
    python inference_scripts/consonant_cluster_stats.py [path_to_dataset.json]

Defaults to datasets/dwipada_augmented_dataset.json
"""

import json
import sys
import os

# Telugu constants
HALANT = "\u0C4D"  # ్
TELUGU_CONSONANTS = set(
    "కఖగఘఙచఛజఝఞటఠడఢణతథదధనపఫబభమయరలవశషసహళఱ"
)
LONG_VOWEL_MARKS = {"ా", "ీ", "ూ", "ే", "ో", "ౌ", "ౄ"}  # దీర్ఘం
ANUSWARA = "ం"
VISARGA = "ః"
DISQUALIFYING_MARKS = LONG_VOWEL_MARKS | {ANUSWARA, VISARGA}


def first_line_ends_light(poem: str) -> bool:
    """
    Check that the last syllable of the first line does NOT contain
    a deergham (long vowel mark), anuswara (ం), or visarga (ః).
    """
    lines = poem.strip().split("\n")
    if not lines:
        return False

    text = lines[0].rstrip()
    if not text:
        return False

    # Find last syllable: scan backwards to the last consonant
    # (skipping any consonant+halant that is part of a cluster)
    last_syllable_start = None
    i = len(text) - 1
    while i >= 0:
        ch = text[i]
        if ch in TELUGU_CONSONANTS:
            # Check if this consonant is preceded by a halant (part of cluster)
            if i > 0 and text[i - 1] == HALANT:
                i -= 2  # skip halant and continue to preceding consonant
                continue
            last_syllable_start = i
            break
        i -= 1

    if last_syllable_start is None:
        return False

    last_syllable = text[last_syllable_start:]
    return not any(ch in DISQUALIFYING_MARKS for ch in last_syllable)


def second_line_starts_with_consonant_cluster(poem: str) -> bool:
    """
    Check if a poem's second line starts with a consonant cluster.
    A consonant cluster = consonant + halant + consonant at the start.
    Splits poem on \\n.
    """
    lines = poem.strip().split("\n")
    if len(lines) < 2:
        return False

    text = lines[1].strip()
    if len(text) < 3:
        return False

    return (
        text[0] in TELUGU_CONSONANTS
        and text[1] == HALANT
        and text[2] in TELUGU_CONSONANTS
    )


def main():
    # Resolve dataset path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    default_path = os.path.join(project_root, "datasets", "dwipada_augmented_dataset.json")

    dataset_path = sys.argv[1] if len(sys.argv) > 1 else default_path

    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at {dataset_path}")
        sys.exit(1)

    # Check for LFS pointer
    with open(dataset_path, "r", encoding="utf-8") as f:
        start = f.read(50)
    if start.startswith("version https://git-lfs"):
        print(f"Warning: {dataset_path} is a Git LFS pointer (not pulled).")
        print("Run 'git lfs pull' or provide a resolved dataset path.")
        sys.exit(1)

    # Load dataset
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    total = len(data)
    cluster_poems = []

    for entry in data:
        poem = entry.get("poem", "")
        if second_line_starts_with_consonant_cluster(poem) and first_line_ends_light(poem):
            cluster_poems.append(poem)

    count = len(cluster_poems)
    pct = (count / total * 100) if total else 0

    print(f"Dataset: {os.path.basename(dataset_path)}")
    print(f"Total poems: {total}")
    print(f"Poems matching (2nd line consonant cluster + 1st line ends light): {count} ({pct:.1f}%)")
    print()

    # Show some examples
    if cluster_poems:
        print("All matching poems:")
        for i, poem in enumerate(cluster_poems, 1):
            print(f"  {i}. {poem.strip()}")
            print()


if __name__ == "__main__":
    main()
