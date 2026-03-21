# -*- coding: utf-8 -*-
"""
Cross-validate YatiNFA against the analyzer (ground truth) on the full dataset.

For each poem, runs the analyzer's analyze_pada() to extract the actual
gana 1 and gana 3 aksharalu, then compares analyzer's yati verdict with
YatiNFA's verdict on the same aksharalu.

Usage:
    python nfa_for_dwipada/yati_nfa_crossval.py
"""

import json
import sys
import os
from collections import Counter

# Setup imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from yati_nfa import YatiNFA, format_yati_result_str
from dwipada.core.analyzer import (
    analyze_pada, check_yati_maitri, check_svara_yati,
    check_samyukta_yati, check_bindu_yati
)

DATASET_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "datasets", "dwipada_augmented_dataset.json"
)


def analyzer_yati_check(pada):
    """Replicate the analyzer's yati logic for a single pada.

    This mirrors analyzer.py lines 1713-1732 exactly:
    first_letter vs third_gana_first_letter, then cascade.

    Returns:
        (match: bool, match_type: str, first_aksharam: str, third_aksharam: str)
        or None if yati cannot be checked (no partition / missing data).
    """
    first_letter = pada.get("first_letter")
    third_gana_first_letter = pada.get("third_gana_first_letter")

    if not first_letter or not third_gana_first_letter:
        return None

    # Step 1: check_yati_maitri on first letters (exact + vyanjana)
    match, group_idx, details = check_yati_maitri(first_letter, third_gana_first_letter)
    match_type = details.get("match_type", "no_match")

    # Step 2: cascade fallbacks on full aksharalu
    aksharam1 = pada.get("first_aksharam")
    aksharam3 = pada.get("third_gana_first_aksharam")
    if not match and aksharam1 and aksharam3:
        if check_svara_yati(aksharam1, aksharam3):
            match = True
            match_type = "svara_yati"
        elif check_samyukta_yati(aksharam1, aksharam3):
            match = True
            match_type = "samyukta_yati"
        elif check_bindu_yati(aksharam1, aksharam3):
            match = True
            match_type = "bindu_yati"

    return match, match_type, aksharam1 or "", aksharam3 or ""


def run_crossval():
    """Run cross-validation: analyzer (ground truth) vs YatiNFA on all poems."""

    print(f"Loading dataset from: {DATASET_PATH}")
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} poems\n")

    nfa = YatiNFA()

    total_lines = 0
    checkable_lines = 0    # lines where analyzer could compute yati
    agree_count = 0
    disagree_count = 0

    # Track disagreement details
    false_positives = []   # NFA says match, analyzer says no
    false_negatives = []   # NFA says no match, analyzer says match
    type_mismatches = []   # Both match, but different match_type

    analyzer_type_counts = Counter()
    nfa_type_counts = Counter()
    agree_type_counts = Counter()

    for poem_idx, entry in enumerate(data):
        poem = entry.get("poem", "")
        lines = [l.strip() for l in poem.strip().split('\n') if l.strip()]
        if len(lines) < 2:
            continue

        for line_idx, line in enumerate(lines[:2]):
            total_lines += 1
            try:
                pada = analyze_pada(line)
            except Exception:
                continue

            # Get analyzer's yati verdict
            analyzer_result = analyzer_yati_check(pada)
            if analyzer_result is None:
                continue  # can't check yati (no valid partition)

            checkable_lines += 1
            ana_match, ana_type, aksharam1, aksharam3 = analyzer_result
            analyzer_type_counts[ana_type] += 1

            # Get YatiNFA's verdict on the same aksharalu
            nfa_results = nfa.process([(aksharam1, aksharam3)])
            nfa_result = nfa_results[0]
            nfa_match = nfa_result["match"]
            nfa_type = nfa_result["match_type"]
            nfa_type_counts[nfa_type] += 1

            # Compare
            if ana_match == nfa_match:
                agree_count += 1
                if ana_match:
                    agree_type_counts[(ana_type, nfa_type)] += 1
                else:
                    agree_type_counts[("no_match", "no_match")] += 1

                # Check if match_type also agrees
                if ana_match and nfa_match and ana_type != nfa_type:
                    type_mismatches.append({
                        "poem_idx": poem_idx,
                        "line": line_idx + 1,
                        "aksharam1": aksharam1,
                        "aksharam3": aksharam3,
                        "ana_type": ana_type,
                        "nfa_type": nfa_type,
                    })
            else:
                disagree_count += 1
                info = {
                    "poem_idx": poem_idx,
                    "line": line_idx + 1,
                    "aksharam1": aksharam1,
                    "aksharam3": aksharam3,
                    "ana_match": ana_match,
                    "ana_type": ana_type,
                    "nfa_match": nfa_match,
                    "nfa_type": nfa_type,
                    "poem_snippet": poem[:80],
                }
                if nfa_match and not ana_match:
                    false_positives.append(info)
                else:
                    false_negatives.append(info)

    # --- Report ---
    print("=" * 70)
    print("YATI NFA vs ANALYZER CROSS-VALIDATION")
    print("=" * 70)
    print(f"Total lines:               {total_lines}")
    print(f"Checkable (valid partition):{checkable_lines}")
    print()
    print(f"Match/No-Match agreement:  {agree_count}  ({100*agree_count/checkable_lines:.4f}%)")
    print(f"Disagreement:              {disagree_count}  ({100*disagree_count/checkable_lines:.4f}%)")
    print(f"  False positives (NFA+):  {len(false_positives)}")
    print(f"  False negatives (NFA-):  {len(false_negatives)}")
    print()

    if type_mismatches:
        print(f"Match type mismatches:     {len(type_mismatches)}  (both agree match=True but differ on type)")
    print()

    print("Analyzer match type distribution:")
    for mt, count in analyzer_type_counts.most_common():
        print(f"  {mt:25s} {count:6d}  ({100*count/checkable_lines:.2f}%)")
    print()

    print("NFA match type distribution:")
    for mt, count in nfa_type_counts.most_common():
        print(f"  {mt:25s} {count:6d}  ({100*count/checkable_lines:.2f}%)")
    print()

    print("Agreement breakdown (analyzer_type, nfa_type):")
    for (at, nt), count in agree_type_counts.most_common():
        print(f"  ({at:20s}, {nt:20s}) {count:6d}")
    print()

    # Show disagreements
    max_show = 20
    if false_positives:
        print(f"--- False Positives (NFA match, Analyzer no match) — first {min(max_show, len(false_positives))} ---")
        for d in false_positives[:max_show]:
            print(f"  Poem #{d['poem_idx']} L{d['line']}: '{d['aksharam1']}' ↔ '{d['aksharam3']}' "
                  f"| Ana: {d['ana_type']} | NFA: {d['nfa_type']}")
        if len(false_positives) > max_show:
            print(f"  ... and {len(false_positives) - max_show} more")
        print()

    if false_negatives:
        print(f"--- False Negatives (NFA no match, Analyzer match) — first {min(max_show, len(false_negatives))} ---")
        for d in false_negatives[:max_show]:
            print(f"  Poem #{d['poem_idx']} L{d['line']}: '{d['aksharam1']}' ↔ '{d['aksharam3']}' "
                  f"| Ana: {d['ana_type']} | NFA: {d['nfa_type']}")
        if len(false_negatives) > max_show:
            print(f"  ... and {len(false_negatives) - max_show} more")
        print()

    if type_mismatches:
        print(f"--- Type Mismatches (both match, different type) — first {min(max_show, len(type_mismatches))} ---")
        for d in type_mismatches[:max_show]:
            print(f"  Poem #{d['poem_idx']} L{d['line']}: '{d['aksharam1']}' ↔ '{d['aksharam3']}' "
                  f"| Ana: {d['ana_type']} | NFA: {d['nfa_type']}")
        if len(type_mismatches) > max_show:
            print(f"  ... and {len(type_mismatches) - max_show} more")
        print()

    print("=" * 70)
    if disagree_count == 0 and len(type_mismatches) == 0:
        print("PERFECT AGREEMENT — YatiNFA matches analyzer on all lines.")
    elif disagree_count == 0:
        print(f"MATCH/NO-MATCH PERFECT — but {len(type_mismatches)} type mismatches.")
    else:
        print(f"DISAGREEMENTS: {disagree_count} match/no-match + {len(type_mismatches)} type mismatches.")
    print("=" * 70)

    return disagree_count == 0


if __name__ == "__main__":
    ok = run_crossval()
    raise SystemExit(0 if ok else 1)
