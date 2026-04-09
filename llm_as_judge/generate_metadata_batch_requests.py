#!/usr/bin/env python3
"""
Generate Vertex AI batch request JSONL for topic/keyword metadata extraction.

Reads datasets/dwipada_augmented_dataset.json and writes:
    output/batch_requests_for_metadata.jsonl

Each request asks Gemini to extract topics and keywords (Telugu + English)
from the poem and its meanings.

Usage:
    python generate_metadata_batch_requests.py
"""

import json
import sys
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).parent
INPUT_FILE = ROOT / "datasets" / "dwipada_augmented_dataset.json"
OUTPUT_FILE = ROOT / "output" / "batch_requests_for_metadata.jsonl"

# ─────────────────────────────────────────────────────────────────────────────
# PROMPT
# ─────────────────────────────────────────────────────────────────────────────

PROMPT_TEMPLATE = """\
Analyze the following Telugu poem and its provided meanings. Based strictly on this text, generate a concise list of topics and keywords in both Telugu and English.

Constraints:
Provide exactly 3 Topics and 5 Keywords per language.
Use absolute brevity: Output only the terms or short phrases. Do not include any explanatory text, definitions, or full sentences.

Output strictly in the structured format requested below.

Input Text:

Poem: {poem}

Telugu Meaning: {telugu_meaning}

English Meaning: {english_meaning}

Required Output Structure:

English Topics: [Topic 1], [Topic 2], [Topic 3]

English Keywords: [Keyword 1], [Keyword 2], [Keyword 3], [Keyword 4], [Keyword 5]

Telugu Topics: [Topic 1], [Topic 2], [Topic 3]

Telugu Keywords: [Keyword 1], [Keyword 2], [Keyword 3], [Keyword 4], [Keyword 5]\
"""


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    if not INPUT_FILE.exists():
        print(f"Error: Input file not found: {INPUT_FILE}", file=sys.stderr)
        sys.exit(1)

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out_f:
        for idx, record in enumerate(dataset, start=1):
            poem = record.get("poem", "").strip()
            telugu_meaning = record.get("telugu_meaning", "").strip()
            english_meaning = record.get("english_meaning", "").strip()
            source = record.get("source", "unknown")

            if not poem or not telugu_meaning or not english_meaning:
                skipped += 1
                continue

            prompt_text = PROMPT_TEMPLATE.format(
                poem=poem,
                telugu_meaning=telugu_meaning,
                english_meaning=english_meaning,
            )

            entry = {
                "key": f"{source}__{idx}",
                "request": {
                    "contents": [
                        {
                            "role": "user",
                            "parts": [{"text": prompt_text}],
                        }
                    ]
                },
                "metadata": {
                    "index": idx,
                    "source": source,
                    "poem": poem,
                },
            }

            out_f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            written += 1

    print(f"{'=' * 55}")
    print(f"METADATA BATCH REQUEST GENERATION COMPLETE")
    print(f"{'=' * 55}")
    print(f"  Total records:          {len(dataset)}")
    print(f"  Requests written:       {written}")
    print(f"  Skipped (missing data): {skipped}")
    print(f"  Output:                 {OUTPUT_FILE}")
    print(f"{'=' * 55}")


if __name__ == "__main__":
    main()
