#!/usr/bin/env python3
"""
Generate Vertex AI batch request JSONL from dwipada couplets.

Reads all .txt files under data/, extracts dwipada couplets, and writes
a JSONL file where each line is a Vertex AI batch prediction request
asking for bhavam and prathipadartham of the couplet.
"""

import json
import os
import re
import sys
from pathlib import Path
from typing import List, Tuple

from dwipada.paths import DATA_DIR, OUTPUT_DIR

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

OUTPUT_FILE = OUTPUT_DIR / "batch_requests.jsonl"

PROMPT_TEMPLATE = (
    "Assume role of a telugu and sanskrit scholar and give me bhavam and "
    "prathipadartham of the following dwipada poem. If there are combined "
    "words please break them with + in prathipadartham. Further bhavam "
    "should be in single line in telugu and English. Just give only bhavam "
    "and prathipadartham of the given input. No additional data."
)

# Pattern to match editorial annotations like [va.ra. sarga 5]
ANNOTATION_PATTERN = re.compile(r"\[.*?\]")

# Pattern to detect damaged/missing text lines
DOT_PATTERN = re.compile(r"…|\.{4,}")


# ─────────────────────────────────────────────────────────────────────────────
# FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def extract_couplets(filepath: Path) -> Tuple[List[Tuple[str, str]], int, int, int]:
    """
    Extract dwipada couplets from a single text file.

    Uses blank lines as couplet boundaries:
        - 2 lines between blanks -> 1 couplet
        - 3 lines between blanks -> 2 overlapping couplets: (1,2) and (2,3)
        - 1 line (singleton) -> skipped
        - Lines with dots/ellipsis -> couplet discarded
        - Lines starting with '#' -> ignored (heading boundary)

    Returns:
        Tuple of (couplets_list, singleton_count, dot_discarded_count, triplet_count)
    """
    with open(filepath, "r", encoding="utf-8") as f:
        raw_lines = f.readlines()

    # Build groups of consecutive verse lines, split by blank lines and # headings
    groups = []
    current_group = []

    for line in raw_lines:
        stripped = line.strip()

        # Stop at footnotes section
        if stripped == "---":
            break

        # Blank lines and # headings end the current group
        if not stripped or stripped.startswith("#"):
            if current_group:
                groups.append(current_group)
                current_group = []
            continue

        # Strip editorial annotations
        cleaned = ANNOTATION_PATTERN.sub("", stripped).strip()
        if cleaned:
            current_group.append(cleaned)

    # Flush last group
    if current_group:
        groups.append(current_group)

    # Process each group into couplets
    valid_couplets = []
    singleton_count = 0
    dot_discarded = 0
    triplet_count = 0

    for group in groups:
        if len(group) == 1:
            singleton_count += 1
        elif len(group) == 2:
            line1, line2 = group
            if DOT_PATTERN.search(line1) or DOT_PATTERN.search(line2):
                dot_discarded += 1
            else:
                valid_couplets.append((line1, line2))
        elif len(group) == 3:
            triplet_count += 1
            # Overlapping couplets: (1,2) and (2,3)
            for i in range(2):
                l1, l2 = group[i], group[i + 1]
                if DOT_PATTERN.search(l1) or DOT_PATTERN.search(l2):
                    dot_discarded += 1
                else:
                    valid_couplets.append((l1, l2))
        else:
            # 4+ lines: use same overlapping sliding window
            for i in range(len(group) - 1):
                l1, l2 = group[i], group[i + 1]
                if DOT_PATTERN.search(l1) or DOT_PATTERN.search(l2):
                    dot_discarded += 1
                else:
                    valid_couplets.append((l1, l2))

    return valid_couplets, singleton_count, dot_discarded, triplet_count


def build_request(line1: str, line2: str, source_file: str, work: str, couplet_num: int) -> dict:
    """
    Build a single Vertex AI batch request dict.

    Args:
        line1: First line of the couplet
        line2: Second line of the couplet
        source_file: Relative path to the source .txt file
        work: Name of the literary work (top-level folder under data/)
        couplet_num: Couplet number within the file (1-based)

    Returns:
        Dict in Vertex AI batch prediction format with metadata
    """
    prompt_text = f"{PROMPT_TEMPLATE}\nPoem:\n{line1}\n{line2}"

    return {
        "request": {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt_text}]
                }
            ]
        },
        "metadata": {
            "source_file": source_file,
            "work": work,
            "couplet_number": couplet_num
        }
    }


def find_txt_files(data_dir: Path) -> List[Path]:
    """Find all .txt files under data_dir, sorted for deterministic order."""
    txt_files = sorted(data_dir.rglob("*.txt"))
    return txt_files


def get_work_name(filepath: Path, data_dir: Path) -> str:
    """Extract the work name (first directory under data/) from a file path."""
    relative = filepath.relative_to(data_dir)
    return relative.parts[0] if relative.parts else "unknown"


def main():
    if not DATA_DIR.exists():
        print(f"Error: Data directory '{DATA_DIR}' not found.", file=sys.stderr)
        sys.exit(1)

    # Find all text files
    txt_files = find_txt_files(DATA_DIR)
    print(f"Found {len(txt_files)} .txt files in {DATA_DIR}/")

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Process all files
    total_couplets = 0
    total_singletons = 0
    total_dot_discarded = 0
    total_triplets = 0

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out_f:
        for filepath in txt_files:
            couplets, singletons, dot_discarded, triplets = extract_couplets(filepath)
            work = get_work_name(filepath, DATA_DIR)
            source = str(filepath)

            for idx, (line1, line2) in enumerate(couplets, start=1):
                request = build_request(line1, line2, source, work, idx)
                out_f.write(json.dumps(request, ensure_ascii=False) + "\n")

            total_couplets += len(couplets)
            total_singletons += singletons
            total_dot_discarded += dot_discarded
            total_triplets += triplets

    # Summary
    print(f"\n{'=' * 60}")
    print(f"BATCH REQUEST GENERATION COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Files processed:        {len(txt_files)}")
    print(f"  Couplets written:       {total_couplets}")
    print(f"  Couplets discarded (...): {total_dot_discarded}")
    print(f"  Singletons skipped:     {total_singletons}")
    print(f"  Triplets expanded:      {total_triplets}")
    print(f"  Output file:            {OUTPUT_FILE}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
