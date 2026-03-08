#!/usr/bin/env python3
"""Convert all dwipada couplets from data/ into a single consolidated JSON file.

Extraction rules (same as generate_batch_requests.py):
  - Blank lines and # headings are group boundaries
  - 2-line group  -> 1 couplet
  - 3+ line group -> overlapping sliding window: (1,2), (2,3), ...
  - 1-line group  -> singleton, skipped
  - Lines with dots/ellipsis (..., ....) -> couplet discarded
  - Lines after '---' -> footnotes, ignored
  - [editorial annotations] stripped from text

Output: data/consolidated_dwipada.json
Each entry contains the couplet text and full provenance metadata.
"""

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

from dwipada.paths import DATA_DIR

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

OUTPUT_FILE = DATA_DIR / "consolidated_dwipada.json"

# Pattern to match editorial annotations like [వా.రా. సర్గ 5]
ANNOTATION_PATTERN = re.compile(r"\[.*?\]")

# Pattern to detect damaged/missing text lines
DOT_PATTERN = re.compile(r"\u2026|\.{4,}")


# ---------------------------------------------------------------------------
# FUNCTIONS
# ---------------------------------------------------------------------------

def parse_headers(raw_lines: List[str]) -> Dict[str, str]:
    """
    Parse # header lines at the top of a file into a metadata dict.

    Example headers:
        # కాండము: బాలకాండము
        # అధ్యాయము: 001
        # శీర్షిక: కృత్యవతరణిక

    Returns dict like {"కాండము": "బాలకాండము", "అధ్యాయము": "001", ...}
    """
    headers = {}
    for line in raw_lines:
        stripped = line.strip()
        if not stripped.startswith("#"):
            break
        # Parse "# key: value"
        content = stripped.lstrip("#").strip()
        if ":" in content:
            key, _, value = content.partition(":")
            headers[key.strip()] = value.strip()
    return headers


def extract_couplets(filepath: Path) -> Tuple[List[Tuple[str, str]], int, int, int]:
    """
    Extract dwipada couplets from a single text file.

    Uses blank lines as couplet boundaries:
        - 2 lines between blanks -> 1 couplet
        - 3+ lines between blanks -> overlapping couplets: (1,2), (2,3), ...
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
        elif len(group) >= 3:
            if len(group) == 3:
                triplet_count += 1
            # Overlapping sliding window: (1,2), (2,3), ...
            for i in range(len(group) - 1):
                l1, l2 = group[i], group[i + 1]
                if DOT_PATTERN.search(l1) or DOT_PATTERN.search(l2):
                    dot_discarded += 1
                else:
                    valid_couplets.append((l1, l2))

    return valid_couplets, singleton_count, dot_discarded, triplet_count


def get_work_name(filepath: Path, data_dir: Path) -> str:
    """Extract the work name (first directory under data/) from a file path."""
    relative = filepath.relative_to(data_dir)
    return relative.parts[0] if relative.parts else "unknown"


def get_file_headers(filepath: Path) -> Dict[str, str]:
    """Read and parse the # headers from a file."""
    with open(filepath, "r", encoding="utf-8") as f:
        raw_lines = f.readlines()
    return parse_headers(raw_lines)


def build_entry(line1: str, line2: str, source_file: str, work: str,
                couplet_idx: int, file_headers: Dict[str, str]) -> Dict[str, Any]:
    """
    Build a single JSON entry for a couplet.

    Args:
        line1: First line of the couplet
        line2: Second line of the couplet
        source_file: Relative path to the source .txt file
        work: Name of the literary work (top-level folder under data/)
        couplet_idx: Couplet index within the file (1-based)
        file_headers: Parsed # headers from the source file
    """
    return {
        "poem": f"{line1}\n{line2}",
        "line1": line1,
        "line2": line2,
        "source": {
            "work": work,
            "file": source_file,
            "couplet_number": couplet_idx,
            **file_headers,
        },
    }


def main():
    if not DATA_DIR.exists():
        print(f"Error: Data directory '{DATA_DIR}' not found.", file=sys.stderr)
        sys.exit(1)

    # Find all text files (sorted for deterministic order)
    txt_files = sorted(DATA_DIR.rglob("*.txt"))
    print(f"Found {len(txt_files)} .txt files in {DATA_DIR}/")

    # Process all files
    all_entries = []
    total_singletons = 0
    total_dot_discarded = 0
    total_triplets = 0

    for filepath in txt_files:
        couplets, singletons, dot_discarded, triplets = extract_couplets(filepath)
        work = get_work_name(filepath, DATA_DIR)
        source = str(filepath.relative_to(DATA_DIR))
        file_headers = get_file_headers(filepath)

        for idx, (line1, line2) in enumerate(couplets, start=1):
            entry = build_entry(line1, line2, source, work, idx, file_headers)
            all_entries.append(entry)

        total_singletons += singletons
        total_dot_discarded += dot_discarded
        total_triplets += triplets

    # Write consolidated JSON
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_entries, f, ensure_ascii=False, indent=2)

    # Summary
    print(f"\n{'=' * 60}")
    print(f"CONSOLIDATED JSON GENERATION COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Files processed:        {len(txt_files)}")
    print(f"  Total couplets:         {len(all_entries)}")
    print(f"  Couplets discarded (\u2026): {total_dot_discarded}")
    print(f"  Singletons skipped:     {total_singletons}")
    print(f"  Triplets expanded:      {total_triplets}")
    print(f"  Output file:            {OUTPUT_FILE}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
