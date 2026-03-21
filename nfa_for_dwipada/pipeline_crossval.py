# -*- coding: utf-8 -*-
"""
Cross-validate FST+NFA Pipeline against the analyzer (ground truth) on the full dataset.

For each poem, runs both:
  1. analyzer.analyze_dwipada() — ground truth
  2. DwipadaPipeline.process() — pipeline under test

Compares: gana validity, prasa match, yati match (per line).

Usage:
    python nfa_for_dwipada/pipeline_crossval.py
"""

import json
import sys
import os
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fst_nfa_pipeline import DwipadaPipeline
from dwipada.core.analyzer import analyze_dwipada

DATASET_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "datasets", "dwipada_augmented_dataset.json"
)


def run_crossval():
    print(f"Loading dataset from: {DATASET_PATH}")
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} poems\n")

    pipeline = DwipadaPipeline()

    total = 0
    skipped = 0

    # Per-check agreement counters
    gana_agree = 0
    gana_disagree = 0
    prasa_agree = 0
    prasa_disagree = 0
    yati_agree = 0
    yati_disagree = 0
    overall_agree = 0
    overall_disagree = 0

    # Collect disagreement samples
    gana_disagreements = []
    prasa_disagreements = []
    yati_disagreements = []

    for poem_idx, entry in enumerate(data):
        poem = entry.get("poem", "")
        lines = [l.strip() for l in poem.strip().split('\n') if l.strip()]
        if len(lines) < 2:
            skipped += 1
            continue

        total += 1

        # --- Ground truth: analyzer ---
        try:
            ana = analyze_dwipada(poem)
        except Exception as e:
            skipped += 1
            total -= 1
            continue

        ana_pada1 = ana.get("pada1", {})
        ana_pada2 = ana.get("pada2", {})
        ana_gana_valid = (
            ana_pada1.get("is_valid_gana_sequence", False)
            and ana_pada2.get("is_valid_gana_sequence", False)
        )
        ana_prasa = ana.get("prasa", {})
        ana_prasa_valid = ana_prasa.get("match", False)

        ana_yati1 = ana.get("yati_line1")
        ana_yati2 = ana.get("yati_line2")
        ana_yati1_match = ana_yati1.get("match", False) if ana_yati1 else False
        ana_yati2_match = ana_yati2.get("match", False) if ana_yati2 else False

        # --- Pipeline ---
        try:
            pipe = pipeline.process(poem)
        except Exception as e:
            skipped += 1
            total -= 1
            continue

        pipe_summary = pipe["validation_summary"]
        pipe_gana_valid = pipe_summary["gana_valid"]
        pipe_prasa_valid = pipe_summary["prasa_valid"]
        pipe_yati1_match = pipe_summary["yati_line1_match"]
        pipe_yati2_match = pipe_summary["yati_line2_match"]

        # --- Compare: Gana ---
        if ana_gana_valid == pipe_gana_valid:
            gana_agree += 1
        else:
            gana_disagree += 1
            if len(gana_disagreements) < 20:
                gana_disagreements.append({
                    "idx": poem_idx,
                    "poem": poem[:80],
                    "ana": ana_gana_valid,
                    "pipe": pipe_gana_valid,
                })

        # --- Compare: Prasa ---
        if ana_prasa_valid == pipe_prasa_valid:
            prasa_agree += 1
        else:
            prasa_disagree += 1
            if len(prasa_disagreements) < 20:
                prasa_disagreements.append({
                    "idx": poem_idx,
                    "poem": poem[:80],
                    "ana": ana_prasa_valid,
                    "pipe": pipe_prasa_valid,
                    "ana_c1": ana_prasa.get("consonant1"),
                    "ana_c2": ana_prasa.get("consonant2"),
                    "pipe_c1": pipe["prasa"].get("line1_consonant"),
                    "pipe_c2": pipe["prasa"].get("line2_consonant"),
                })

        # --- Compare: Yati (both lines) ---
        yati_match = (ana_yati1_match == pipe_yati1_match and ana_yati2_match == pipe_yati2_match)
        if yati_match:
            yati_agree += 1
        else:
            yati_disagree += 1
            if len(yati_disagreements) < 20:
                yati_disagreements.append({
                    "idx": poem_idx,
                    "poem": poem[:80],
                    "ana_y1": ana_yati1_match,
                    "ana_y2": ana_yati2_match,
                    "pipe_y1": pipe_yati1_match,
                    "pipe_y2": pipe_yati2_match,
                    "ana_y1_type": ana_yati1.get("match_type") if ana_yati1 else None,
                    "ana_y2_type": ana_yati2.get("match_type") if ana_yati2 else None,
                    "pipe_y1_type": pipe["yati"]["line1"].get("match_type"),
                    "pipe_y2_type": pipe["yati"]["line2"].get("match_type"),
                })

        # --- Overall ---
        all_same = (ana_gana_valid == pipe_gana_valid
                    and ana_prasa_valid == pipe_prasa_valid
                    and yati_match)
        if all_same:
            overall_agree += 1
        else:
            overall_disagree += 1

        # Progress
        if (poem_idx + 1) % 5000 == 0:
            print(f"  ... processed {poem_idx + 1}/{len(data)} poems")

    # --- Report ---
    print()
    print("=" * 70)
    print("FST+NFA PIPELINE vs ANALYZER CROSS-VALIDATION")
    print("=" * 70)
    print(f"Total poems processed: {total}")
    print(f"Skipped (malformed):   {skipped}")
    print()

    def pct(n, d):
        return f"{100*n/d:.4f}%" if d > 0 else "N/A"

    print(f"{'Check':<20s}  {'Agree':>8s}  {'Disagree':>8s}  {'Agreement':>12s}")
    print("-" * 55)
    print(f"{'Gana':<20s}  {gana_agree:>8d}  {gana_disagree:>8d}  {pct(gana_agree, total):>12s}")
    print(f"{'Prasa':<20s}  {prasa_agree:>8d}  {prasa_disagree:>8d}  {pct(prasa_agree, total):>12s}")
    print(f"{'Yati (both lines)':<20s}  {yati_agree:>8d}  {yati_disagree:>8d}  {pct(yati_agree, total):>12s}")
    print(f"{'Overall (all 3)':<20s}  {overall_agree:>8d}  {overall_disagree:>8d}  {pct(overall_agree, total):>12s}")
    print()

    max_show = 10

    if gana_disagreements:
        print(f"--- Gana Disagreements (first {min(max_show, len(gana_disagreements))}) ---")
        for d in gana_disagreements[:max_show]:
            print(f"  #{d['idx']}: ana={d['ana']}, pipe={d['pipe']} | {d['poem']}")
        print()

    if prasa_disagreements:
        print(f"--- Prasa Disagreements (first {min(max_show, len(prasa_disagreements))}) ---")
        for d in prasa_disagreements[:max_show]:
            print(f"  #{d['idx']}: ana={d['ana']}({d['ana_c1']}↔{d['ana_c2']}), "
                  f"pipe={d['pipe']}({d['pipe_c1']}↔{d['pipe_c2']}) | {d['poem']}")
        print()

    if yati_disagreements:
        print(f"--- Yati Disagreements (first {min(max_show, len(yati_disagreements))}) ---")
        for d in yati_disagreements[:max_show]:
            print(f"  #{d['idx']}: ana=L1:{d['ana_y1']}({d['ana_y1_type']})/L2:{d['ana_y2']}({d['ana_y2_type']}), "
                  f"pipe=L1:{d['pipe_y1']}({d['pipe_y1_type']})/L2:{d['pipe_y2']}({d['pipe_y2_type']}) | {d['poem']}")
        print()

    print("=" * 70)
    if overall_disagree == 0:
        print("PERFECT AGREEMENT on all checks across all poems.")
    else:
        print(f"DISAGREEMENTS FOUND — {overall_disagree} poems differ.")
    print("=" * 70)

    return overall_disagree == 0


if __name__ == "__main__":
    ok = run_crossval()
    raise SystemExit(0 if ok else 1)
