# -*- coding: utf-8 -*-
"""Analyze the false negative patterns from yati NFA cross-validation."""

import json
import re
import sys
import os
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from yati_nfa import (YATI_MAITRI_GROUPS, MAITRI_GROUP_NAMES,
                       LETTER_TO_MAITRI_GROUP, _analyze_aksharam, _cascade_yati_check)

DATASET_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "datasets", "dwipada_augmented_dataset.json"
)


def parse_yati_check(yati_check_str):
    if not yati_check_str:
        return None
    letters = re.findall(r"'([^']+)'", yati_check_str)
    if len(letters) < 2:
        return None
    letter1, letter2 = letters[0], letters[1]
    expected_match = "match" in yati_check_str.lower() and "no match" not in yati_check_str.lower()
    return letter1, letter2, expected_match


def main():
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Collect all false negative pairs
    fn_pairs = Counter()
    fn_group_pairs = Counter()

    for entry in data:
        chandassu = entry.get("chandassu_analysis", {})
        for line_key in ["line_1", "line_2"]:
            yati_str = chandassu.get(line_key, {}).get("yati_check", "")
            parsed = parse_yati_check(yati_str)
            if parsed is None:
                continue
            letter1, letter2, gt_match = parsed
            info1 = _analyze_aksharam(letter1)
            info2 = _analyze_aksharam(letter2)
            result = _cascade_yati_check(info1, info2)

            if not result["match"] and gt_match:
                # False negative — sort the pair for canonical form
                pair = tuple(sorted([letter1, letter2]))
                fn_pairs[pair] += 1

                g1 = LETTER_TO_MAITRI_GROUP.get(letter1, [-1])
                g2 = LETTER_TO_MAITRI_GROUP.get(letter2, [-1])
                group_pair = tuple(sorted([g1[0] if g1 else -1, g2[0] if g2 else -1]))
                fn_group_pairs[group_pair] += 1

    print("=" * 70)
    print("FALSE NEGATIVE PAIR ANALYSIS")
    print("=" * 70)
    print(f"\nTotal unique letter pairs: {len(fn_pairs)}")
    print(f"\nTop 30 most frequent false negative pairs:")
    print(f"{'Pair':>12s}  {'Count':>6s}  {'Group1':>40s}  {'Group2':>40s}")
    print("-" * 110)

    for (l1, l2), count in fn_pairs.most_common(30):
        g1 = LETTER_TO_MAITRI_GROUP.get(l1, [])
        g2 = LETTER_TO_MAITRI_GROUP.get(l2, [])
        g1_str = f"{g1} {MAITRI_GROUP_NAMES[g1[0]] if g1 else '?'}" if g1 else "not in any group"
        g2_str = f"{g2} {MAITRI_GROUP_NAMES[g2[0]] if g2 else '?'}" if g2 else "not in any group"
        print(f"  '{l1}' ↔ '{l2}'  {count:6d}  {g1_str:>40s}  {g2_str:>40s}")

    print(f"\nGroup pair frequency (cross-group mismatches):")
    print(f"{'Groups':>20s}  {'Count':>6s}  {'Group Names'}")
    print("-" * 80)
    for (g1, g2), count in fn_group_pairs.most_common(20):
        n1 = MAITRI_GROUP_NAMES[g1] if 0 <= g1 < len(MAITRI_GROUP_NAMES) else "?"
        n2 = MAITRI_GROUP_NAMES[g2] if 0 <= g2 < len(MAITRI_GROUP_NAMES) else "?"
        print(f"  ({g1:2d}, {g2:2d})      {count:6d}  {n1} ↔ {n2}")


if __name__ == "__main__":
    main()
