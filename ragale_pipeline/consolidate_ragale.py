# -*- coding: utf-8 -*-
"""
Parse ragale_outputs.txt and consolidate all poems into a single JSON array.
Then run the analyser and filter poems with perfect scores (100%).
"""

import json
import re
import sys

def parse_ragale_outputs(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    # Split on numbered section headers like "1. [", "23.[", "45.  [", "83.JSON\n["
    # Pattern: number followed by dot, optional stuff (whitespace/JSON), then "["
    sections = re.split(r'\n?\d+\.[\s\w]*(?=\[)', text)

    all_poems = []
    errors = []

    for i, section in enumerate(sections):
        section = section.strip()
        if not section:
            continue

        # Find the JSON array in this section
        # It should start with "[" and end with "]"
        start = section.find('[')
        if start == -1:
            continue

        # Find the matching closing bracket
        bracket_count = 0
        end = -1
        for j in range(start, len(section)):
            if section[j] == '[':
                bracket_count += 1
            elif section[j] == ']':
                bracket_count -= 1
                if bracket_count == 0:
                    end = j + 1
                    break

        if end == -1:
            errors.append(f"Section {i}: Could not find matching ']'\n    First 200 chars: {section[:200]}")
            continue

        json_str = section[start:end]

        try:
            poems = json.loads(json_str)
            if isinstance(poems, list):
                all_poems.extend(poems)
            elif isinstance(poems, dict):
                all_poems.append(poems)
        except json.JSONDecodeError as e:
            errors.append(f"Section {i}: JSON parse error: {e}\n    First 200 chars: {json_str[:200]}")

    if errors:
        print(f"Warnings ({len(errors)} sections had issues):")
        for err in errors:
            print(f"  - {err}")

    return all_poems


def main():
    input_file = "ragale_outputs.txt"
    output_file = "ragale_consolidated.json"
    filtered_file = "ragale_consolidated_filtered.json"

    print(f"Parsing {input_file}...")
    poems = parse_ragale_outputs(input_file)
    print(f"Found {len(poems)} poems total.")

    # Write consolidated JSON
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(poems, f, ensure_ascii=False, indent=2)
    print(f"Written consolidated file: {output_file}")

    # Now run the analyser on each poem and filter perfect ones
    from kannada_ragale_analyser import analyze_poem

    perfect_poems = []
    for poem in poems:
        analysis = analyze_poem(poem)
        if "error" in analysis:
            continue
        score = analysis.get("score", {})
        if score.get("overall", 0) == 100.0:
            perfect_poems.append(poem)

    print(f"Found {len(perfect_poems)} poems with perfect score (100%).")

    with open(filtered_file, "w", encoding="utf-8") as f:
        json.dump(perfect_poems, f, ensure_ascii=False, indent=2)
    print(f"Written filtered file: {filtered_file}")


if __name__ == "__main__":
    main()
