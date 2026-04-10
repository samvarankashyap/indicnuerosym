#!/usr/bin/env python3
"""Convert synthetic dwipada poems to the master dataset format.

Reads synthetic_data/combined_dwipada_poems.json, filters through the
dwipada analyser (100% score only), and writes datasets/synthetic_dwipada_dataset.json.
"""

import json

from dwipada.core.analyzer import analyze_dwipada
from dwipada.paths import DATASETS_DIR, SYNTHETIC_DATA_DIR

INPUT_PATH = SYNTHETIC_DATA_DIR / "combined_dwipada_poems.json"
OUTPUT_PATH = DATASETS_DIR / "synthetic_dwipada_dataset.json"

PROMPT = (
    "Assume role of a telugu and sanskrit scholar and give me bhavam and "
    "prathipadartham of the following dwipada poem. If there are combined "
    "words please break them with + in prathipadartham. Further bhavam "
    "should be in single line in telugu and English. Just give only bhavam "
    "and prathipadartham of the given input. No additional data.\nPoem:"
)


def main():
    with open(INPUT_PATH, encoding="utf-8") as f:
        raw_data = json.load(f)

    print(f"Input: {len(raw_data)} records from {INPUT_PATH.name}")

    output = []
    failed = []

    for i, record in enumerate(raw_data):
        poem = record["poem_lines"][0] + " \n" + record["poem_lines"][1]

        try:
            analysis = analyze_dwipada(poem)
            score = analysis["match_score"]["overall"]
        except Exception as e:
            failed.append((i, record.get("poem_number", i), str(e)))
            continue

        if score != 100.0:
            failed.append((i, record.get("poem_number", i), f"score={score}"))
            continue

        output.append({
            "prompt": PROMPT,
            "poem": poem,
            "word_to_word_meaning": {},
            "telugu_meaning": record["telugu_meaning"],
            "english_meaning": record["english_meaning"],
            "chandassu_analysis": record["chandassu_analysis"],
        })

        if (i + 1) % 500 == 0:
            print(f"  Progress: {i + 1}/{len(raw_data)} processed, {len(output)} passed")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\nResults:")
    print(f"  Total input:  {len(raw_data)}")
    print(f"  Passed (100%): {len(output)}")
    print(f"  Failed/filtered: {len(failed)}")
    print(f"  Output: {OUTPUT_PATH}")

    if failed:
        print(f"\n  Sample failures (first 10):")
        for idx, poem_num, reason in failed[:10]:
            print(f"    #{poem_num}: {reason}")


if __name__ == "__main__":
    main()
