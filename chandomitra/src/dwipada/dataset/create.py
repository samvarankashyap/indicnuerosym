#!/usr/bin/env python3
"""
Extract structured dataset from Vertex AI batch response JSONL.

Input:  datasets/100_responses.jsonl  (raw Gemini batch output)
Output: datasets/100_dataset.jsonl    (one record per line, compact)
        datasets/100_dataset_pretty.json (JSON array, indented for review)

Each output record:
  { prompt, poem, word_to_word_meaning, telugu_meaning, english_meaning }
"""

import argparse
import json
import re
import sys
from pathlib import Path

from dwipada.paths import DATASETS_DIR

# ── Section header regexes ───────────────────────────────────────────────────
# Each pattern matches the bold markdown header and captures everything until
# the next bold header (**...**) or end of string.

RE_WORD_MEANING = re.compile(
    r"\*\*ప్రతిపదార్థ(?:ము|ం)\s*(?:\(.*?\))?\s*:?\*\*\s*:?\s*(.*?)(?=\n\*\*|\Z)",
    re.DOTALL,
)

# Matches: **భావం:**, **భావము (Telugu):**, **భావం (తెలుగు):**, **భావం (Bhavam):**,
#          **తెలుగు భావం:**
RE_TELUGU_MEANING = re.compile(
    r"\*\*(?:తెలుగు\s+)?భావ(?:ము|ం)\s*(?:\([^)]*\))?\s*:?\*\*\s*:?\s*(.*?)(?=\n\*\*|\Z)",
    re.DOTALL,
)

# Matches: **Bhavam (English):**, **English:**, **Meaning (English):**, **Meaning:**,
#          **Bhavam:**, **English Bhavam:**, **భావం (English):**, **English Meaning:**,
#          **Purport (English):**, **Gist (English):**
RE_ENGLISH_MEANING = re.compile(
    r"\*\*(?:Bhavam\s*(?:\(English\))?|English(?:\s+(?:Bhavam|Meaning))?|Meaning\s*(?:\(English\))?|(?:Purport|Gist)\s*(?:\(English\))?|భావ(?:ము|ం)\s*\(English\))\s*:?\*\*\s*:?\s*(.*?)(?=\n\*\*|\Z)",
    re.DOTALL,
)

# ── Helpers ──────────────────────────────────────────────────────────────────

def extract_prompt_and_poem(full_text: str) -> tuple[str, str]:
    """Split the request text into (prompt_instruction, poem)."""
    marker = "Poem:\n"
    idx = full_text.find(marker)
    if idx == -1:
        return full_text.strip(), ""
    prompt = full_text[:idx + len("Poem:")].strip()
    poem = full_text[idx + len(marker):].strip()
    # Add space before newline to visually separate the two lines of the couplet
    poem = poem.replace("\n", " \n")
    return prompt, poem


def clean_field(text: str) -> str:
    """Clean extracted markdown field text."""
    text = text.strip()
    # Remove residual bold markers
    text = re.sub(r"\*\*", "", text)
    # Remove leading bullet chars (*, -, •) from each line, keep content
    lines = []
    for line in text.split("\n"):
        line = line.strip()
        line = re.sub(r"^[\*\-•]\s*", "", line)
        if line:
            lines.append(line)
    return "\n".join(lines)


def clean_word_meaning(text: str) -> dict:
    """Clean word_to_word_meaning and return as a dict.

    1. Remove parenthesized content containing English/Latin characters
    2. Normalize ':' -> '=' as separator
    3. Strip semicolons and trailing dots
    4. Return dict: { "word breakdown": "meaning", ... }
    """
    # Remove parenthesized text that contains Latin characters
    text = re.sub(r"\s*\([^)]*[A-Za-z][^)]*\)", "", text)

    # Split into individual entries: first by newline, then by ';' within lines
    entries = []
    for line in text.split("\n"):
        # Normalize ':' to '=' as separator (first occurrence per segment)
        line = re.sub(r"^([^:=]+?)\s*:\s*", r"\1 = ", line, count=1)
        # Split on ';' to handle inline-separated entries like "word1 = meaning1; word2 = meaning2"
        for segment in line.split(";"):
            segment = re.sub(r"  +", " ", segment).strip()
            if segment:
                entries.append(segment)

    result = {}
    for entry in entries:
        if "=" in entry:
            key, _, value = entry.partition("=")
            key = key.strip()
            value = value.strip().rstrip(".").strip()
            if key:
                result[key] = value
        else:
            key = entry.rstrip(".").rstrip(":").strip()
            if key:
                result[key] = ""

    return result


def split_bhavam_block(bhavam_raw: str) -> tuple[str, str]:
    """Split a bhavam block that may contain both Telugu and English inline.

    Some responses put both meanings under one **భావం:** header:
        తెలుగు: <telugu text>
        English: <english text>
    Others use separate bold headers (handled by caller).
    """
    telugu = ""
    english = ""

    # Try inline labels: "తెలుగు:" / "**తెలుగు:**" and "English:" / "**English:**"
    te_inline = re.search(r"(?:\*\*)?(?:తెలుగు|Telugu)\s*:\s*(?:\*\*)?\s*:?\s*(.+?)(?=\n\*?\*?English|\Z)", bhavam_raw, re.DOTALL)
    en_inline = re.search(r"(?:\*\*)?English\s*:\s*(?:\*\*)?\s*:?\s*(.+)", bhavam_raw, re.DOTALL)

    if te_inline:
        telugu = te_inline.group(1).strip()
    if en_inline:
        english = en_inline.group(1).strip()

    # If no inline labels found, try splitting by the last line that starts
    # with an English character (the English meaning tends to be the final
    # sentence written in Latin script).
    if not telugu and not english:
        lines = [l.strip() for l in bhavam_raw.strip().split("\n") if l.strip()]
        telugu_lines = []
        english_lines = []
        for line in lines:
            # If the line starts with a Latin letter (or parenthesized Latin text), treat as English
            if re.match(r"\(?[A-Za-z]", line):
                english_lines.append(line)
            else:
                telugu_lines.append(line)
        telugu = "\n".join(telugu_lines)
        english = "\n".join(english_lines)

    # Final fallback: extract trailing parenthesized English from Telugu text
    if not english and telugu:
        paren_match = re.search(r"\(([A-Za-z][^)]{10,})\)\s*$", telugu)
        if paren_match:
            english = paren_match.group(1).strip()
            telugu = telugu[:paren_match.start()].strip()

    return telugu, english


def extract_sections(response_text: str) -> dict:
    """Parse the 3 sections from the model's markdown response."""
    word_match = RE_WORD_MEANING.search(response_text)

    # Try dedicated English bold header first
    english_match = RE_ENGLISH_MEANING.search(response_text)

    # Telugu bhavam — may contain inline English when no separate header exists
    telugu_match = RE_TELUGU_MEANING.search(response_text)

    word_meaning = clean_word_meaning(clean_field(word_match.group(1))) if word_match else ""
    english_meaning = clean_field(english_match.group(1)) if english_match else ""
    telugu_meaning = ""

    if telugu_match:
        bhavam_raw = telugu_match.group(1)

        # If the bhavam block was cut short by bold sub-headers like **తెలుగు:**,
        # re-capture everything from the bhavam header to end of text
        if not bhavam_raw.strip() and telugu_match.end() < len(response_text):
            rest = response_text[telugu_match.start():]
            # Grab from header to end
            full = re.search(
                r"\*\*భావ(?:ము|ం)\s*(?:\([^)]*\))?\s*:?\*\*\s*:?\s*(.*)",
                rest, re.DOTALL,
            )
            if full:
                bhavam_raw = full.group(1)

        if english_meaning:
            # Separate English header exists, so bhavam block is purely Telugu
            telugu_meaning = clean_field(bhavam_raw)
        else:
            # No separate English header — check for inline labels
            telugu_meaning, english_meaning = split_bhavam_block(clean_field(bhavam_raw))

    return {
        "word_to_word_meaning": word_meaning,
        "telugu_meaning": telugu_meaning,
        "english_meaning": english_meaning,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Extract structured dataset from Vertex AI batch responses.")
    parser.add_argument("input", help="Input JSONL file (batch responses)")
    parser.add_argument("-o", "--output", help="Output JSONL file stem (without extension)")
    args = parser.parse_args()

    input_file = Path(args.input)
    if not input_file.exists():
        print(f"Error: Input file not found: {input_file}", file=sys.stderr)
        sys.exit(1)

    if args.output:
        out_stem = Path(args.output).with_suffix("")
    else:
        out_stem = input_file.with_suffix("")
    output_jsonl = out_stem.with_suffix(".jsonl")
    output_pretty = out_stem.with_suffix(".json")

    records = []
    skipped = []

    with open(input_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            entry = json.loads(line)

            # Extract request text
            try:
                request_text = entry["request"]["contents"][0]["parts"][0]["text"]
            except (KeyError, IndexError):
                skipped.append((line_num, "missing request text"))
                continue

            # Extract response text
            try:
                response_text = entry["response"]["candidates"][0]["content"]["parts"][0]["text"]
            except (KeyError, IndexError):
                skipped.append((line_num, "missing response text"))
                continue

            prompt, poem = extract_prompt_and_poem(request_text)
            if not poem:
                skipped.append((line_num, "could not extract poem from prompt"))
                continue

            sections = extract_sections(response_text)

            missing = [k for k, v in sections.items() if not v]
            if missing:
                skipped.append((line_num, f"empty sections: {', '.join(missing)}"))
                continue

            records.append({
                "prompt": prompt,
                "poem": poem,
                **sections,
            })

    # Write compact JSONL
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(output_jsonl, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Write pretty JSON
    with open(output_pretty, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    # Summary
    print(f"Input:     {input_file.name}")
    print(f"Processed: {len(records)} records")
    print(f"Skipped:   {len(skipped)} records")
    if skipped:
        for ln, reason in skipped[:20]:
            print(f"  Line {ln}: {reason}")
        if len(skipped) > 20:
            print(f"  ... and {len(skipped) - 20} more")
    print(f"\nOutput:")
    print(f"  {output_jsonl}")
    print(f"  {output_pretty}")


if __name__ == "__main__":
    main()
