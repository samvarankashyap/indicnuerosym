#!/usr/bin/env python3
"""Augment the master dataset with chandassu analysis from the dwipada analyser.

Reads datasets/dwipada_master_filtered_perfect_dataset.json, runs analyze_dwipada()
on each poem, and writes datasets/dwipada_augmented_perfect_dataset.json with the
chandassu_analysis field added.
"""

import json

from dwipada.core.analyzer import analyze_dwipada
from dwipada.paths import DATASETS_DIR

INPUT_PATH = DATASETS_DIR / "dwipada_master_filtered_perfect_dataset.json"
OUTPUT_PATH = DATASETS_DIR / "dwipada_augmented_perfect_dataset.json"


def short_gana_name(full_name: str) -> str:
    """Extract short gana name: 'Sala (సల)' -> 'Sala'."""
    return full_name.split(" (")[0] if " (" in full_name else full_name


def format_breakdown(pada: dict) -> str:
    """Format gana breakdown string from pada analysis."""
    parts = []
    for gana in pada["partition"]["ganas"]:
        syllable_text = "".join(gana["aksharalu"])
        name = short_gana_name(gana["name"])
        pattern = gana["pattern"]
        gana_type = gana["type"]  # "Indra" or "Surya"
        parts.append(f"{syllable_text} ({name} - {pattern} - {gana_type})")
    return " | ".join(parts)


def format_yati(yati: dict) -> str:
    """Format yati check string."""
    g1 = yati["first_gana_letter"]
    g3 = yati["third_gana_letter"]
    if yati["match"]:
        return f"Matches: '{g1}' and '{g3}'."
    return f"No match: '{g1}' and '{g3}'."


def format_prasa(prasa: dict, line_num: int) -> str:
    """Format prasa check string for a given line."""
    aksharam = prasa[f"line{line_num}_second_aksharam"]
    other = prasa[f"line{3 - line_num}_second_aksharam"]
    if prasa["match"]:
        return f"2nd Letter '{aksharam}' matches '{other}'."
    return f"No match: '{aksharam}' and '{other}'."


def build_chandassu(analysis: dict) -> dict:
    """Build chandassu_analysis dict from analyze_dwipada() result."""
    return {
        "line_1": {
            "breakdown": format_breakdown(analysis["pada1"]),
            "yati_check": format_yati(analysis["yati_line1"]),
            "prasa_check": format_prasa(analysis["prasa"], 1),
        },
        "line_2": {
            "breakdown": format_breakdown(analysis["pada2"]),
            "yati_check": format_yati(analysis["yati_line2"]),
            "prasa_check": format_prasa(analysis["prasa"], 2),
        },
    }


def main():
    with open(INPUT_PATH, encoding="utf-8") as f:
        data = json.load(f)

    print(f"Input: {len(data)} records from {INPUT_PATH.name}")

    output = []
    errors = []

    for i, record in enumerate(data):
        try:
            analysis = analyze_dwipada(record["poem"])
            chandassu = build_chandassu(analysis)
        except Exception as e:
            errors.append((i, str(e)))
            chandassu = None

        augmented = {**record, "chandassu_analysis": chandassu}
        output.append(augmented)

        if (i + 1) % 5000 == 0:
            print(f"  Progress: {i + 1}/{len(data)}")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\nResults:")
    print(f"  Total: {len(output)}")
    print(f"  Errors: {len(errors)}")
    print(f"  Output: {OUTPUT_PATH}")

    if errors:
        print(f"\n  Sample errors (first 5):")
        for idx, reason in errors[:5]:
            print(f"    #{idx}: {reason}")


if __name__ == "__main__":
    main()
