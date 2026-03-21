#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cross-validation: NFA pipeline vs Analyzer on the full Dwipada dataset.

Compares the three-stage NFA pipeline (SyllableAssembler → GanaMarker → GanaNFA)
against the reference analyzer (split_aksharalu → akshara_ganavibhajana →
find_dwipada_gana_partition) on every line of the dataset.

Reports:
  - Marker agreement (Stage 1+2: syllable splitting & guru/laghu)
  - Gana partition agreement (Stage 3: gana identification)
  - Outliers with details

Usage:
    python inference_scripts/nfa_vs_analyzer_crossval.py [path_to_dataset.json]

Defaults to datasets/dwipada_augmented_dataset.json
"""

import json
import sys
import os

# Resolve paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "nfa_for_dwipada"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from syllable_assembler import SyllableAssembler
from ganana_marker import GanaMarker
from gana_nfa import GanaNFA, format_partition_str
from dwipada.core.analyzer import (
    split_aksharalu,
    akshara_ganavibhajana,
    find_dwipada_gana_partition,
)


def normalize_ref_names(partition):
    """Extract gana names from reference partition, normalized to NFA naming."""
    return tuple(
        "Ha_Gala" if g["name"].startswith("Ha/Gala") else g["name"].split(" ")[0]
        for g in partition["ganas"]
    )


def analyze_line(line):
    """Run both pipelines on a single line, return comparison result.

    Returns dict with keys:
        nfa_markers, ref_markers, markers_match,
        nfa_names, ref_names, nfa_valid, ref_valid, agree
    """
    # NFA pipeline
    asm = SyllableAssembler()
    syllables = asm.process(line)
    gm = GanaMarker()
    markers = gm.process(syllables)
    ui_nfa = [m for m in markers if m in ("U", "I")]

    # Reference pipeline
    aksharalu = split_aksharalu(line)
    ref_markers = akshara_ganavibhajana(aksharalu)
    ui_ref = [m for m in ref_markers if m in ("U", "I")]

    markers_match = ui_nfa == ui_ref

    # Gana partition — NFA
    nfa = GanaNFA()
    nfa_result = nfa.process(ui_nfa)
    nfa_valid = bool(nfa_result and nfa_result[0] is not None)
    nfa_names = tuple(g["name"] for g in nfa_result[0]) if nfa_valid else None

    # Gana partition — Reference
    ref_partition = find_dwipada_gana_partition(ui_ref, aksharalu)
    ref_valid = ref_partition is not None and ref_partition.get("is_fully_valid", False)
    ref_names = normalize_ref_names(ref_partition) if ref_valid else None

    # Agreement
    if nfa_valid and ref_valid:
        agree = nfa_names == ref_names
    elif not nfa_valid and not ref_valid:
        agree = True  # both invalid
    else:
        agree = False

    return {
        "nfa_markers": "".join(ui_nfa),
        "ref_markers": "".join(ui_ref),
        "markers_match": markers_match,
        "nfa_names": nfa_names,
        "ref_names": ref_names,
        "nfa_valid": nfa_valid,
        "ref_valid": ref_valid,
        "agree": agree,
    }


def main():
    default_path = os.path.join(PROJECT_ROOT, "datasets", "dwipada_augmented_dataset.json")
    dataset_path = sys.argv[1] if len(sys.argv) > 1 else default_path

    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at {dataset_path}")
        sys.exit(1)

    with open(dataset_path, "r", encoding="utf-8") as f:
        start = f.read(50)
    if start.startswith("version https://git-lfs"):
        print(f"Warning: {dataset_path} is a Git LFS pointer (not pulled).")
        print("Run 'git lfs pull' or provide a resolved dataset path.")
        sys.exit(1)

    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    total = 0
    both_valid_agree = 0
    both_invalid = 0
    mismatch_partition = 0
    nfa_only_valid = 0
    ref_only_valid = 0
    marker_disagree = 0

    marker_outliers = []
    partition_outliers = []

    for idx, entry in enumerate(data):
        poem = entry.get("poem", "")
        for line_idx, line in enumerate(poem.strip().split("\n")):
            line = line.strip()
            if not line:
                continue
            total += 1

            result = analyze_line(line)

            if not result["markers_match"]:
                marker_disagree += 1
                if len(marker_outliers) < 20:
                    marker_outliers.append((idx, line_idx, line, result))
                continue

            if result["nfa_valid"] and result["ref_valid"]:
                if result["agree"]:
                    both_valid_agree += 1
                else:
                    mismatch_partition += 1
                    if len(partition_outliers) < 20:
                        partition_outliers.append((idx, line_idx, line, result))
            elif result["nfa_valid"]:
                nfa_only_valid += 1
            elif result["ref_valid"]:
                ref_only_valid += 1
            else:
                both_invalid += 1

    # Print report
    w = 70
    print("=" * w)
    print("CROSS-VALIDATION: NFA Pipeline vs Analyzer")
    print("=" * w)
    print(f"Dataset:  {os.path.basename(dataset_path)}")
    print(f"Total lines analyzed: {total}")
    print()
    print(f"  {'Both valid & agree:':<30} {both_valid_agree:>6}  ({both_valid_agree/total*100:.2f}%)")
    print(f"  {'Both invalid:':<30} {both_invalid:>6}  ({both_invalid/total*100:.2f}%)")
    print(f"  {'Partition mismatch:':<30} {mismatch_partition:>6}  ({mismatch_partition/total*100:.2f}%)")
    print(f"  {'NFA valid, REF invalid:':<30} {nfa_only_valid:>6}  ({nfa_only_valid/total*100:.2f}%)")
    print(f"  {'REF valid, NFA invalid:':<30} {ref_only_valid:>6}  ({ref_only_valid/total*100:.2f}%)")
    print(f"  {'Marker disagreement:':<30} {marker_disagree:>6}  ({marker_disagree/total*100:.2f}%)")

    agreement = both_valid_agree + both_invalid
    print()
    print(f"  Total agreement: {agreement}/{total} ({agreement/total*100:.2f}%)")

    if marker_outliers:
        print()
        print("-" * w)
        print(f"MARKER DISAGREEMENTS (showing {len(marker_outliers)}/{marker_disagree}):")
        print("-" * w)
        for i, (pidx, lidx, line, r) in enumerate(marker_outliers, 1):
            print(f"\n  {i}. Poem #{pidx}, Line {lidx+1}")
            print(f"     Text: {line[:60]}")
            print(f"     NFA: {r['nfa_markers']}")
            print(f"     REF: {r['ref_markers']}")

    if partition_outliers:
        print()
        print("-" * w)
        print(f"PARTITION MISMATCHES (showing {len(partition_outliers)}/{mismatch_partition}):")
        print("-" * w)
        for i, (pidx, lidx, line, r) in enumerate(partition_outliers, 1):
            print(f"\n  {i}. Poem #{pidx}, Line {lidx+1}")
            print(f"     Text: {line[:60]}")
            print(f"     NFA ganas: {r['nfa_names']}")
            print(f"     REF ganas: {r['ref_names']}")

    if not marker_outliers and not partition_outliers:
        print()
        print("No outliers found.")

    print()
    print("=" * w)


if __name__ == "__main__":
    main()
