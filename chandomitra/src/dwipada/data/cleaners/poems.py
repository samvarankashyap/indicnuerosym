#!/usr/bin/env python3
"""Clean all poem texts across all datasets.

1. Remove specified punctuation and characters from poem body
2. Preserve # header lines, --- footnotes sections, and .......... gap-markers
"""

import glob
import os
import re

from dwipada.paths import DATA_DIR

# Match lines that are ONLY dots (with optional leading/trailing whitespace)
DOTLINE = re.compile(r'^\s*\.{3,}\s*$')
# Match lines that START with dots then have text (e.g. "..........బూజసేసి")
DOTPREFIX = re.compile(r'^(\.{3,})(.*)')
# Characters to remove from poem body
PUNCTUATION = str.maketrans('', '', ':;".!?\u201c\u201d\u2018\u2019\u0c01')
# [వా.రా. సర్గ ...] references
SARGA_REF = re.compile(r'\[వా\.రా\.[^\]]*\]')


def clean_body_line(line):
    """Clean a single poem body line."""
    # Preserve pure dot-marker lines as-is
    if DOTLINE.match(line):
        return line

    # For lines starting with dots then text (e.g. "..........బూజసేసి"),
    # preserve the dot prefix, clean the text part
    m = DOTPREFIX.match(line)
    if m:
        dot_part = m.group(1)
        text_part = m.group(2)
        text_part = SARGA_REF.sub('', text_part)
        text_part = text_part.translate(PUNCTUATION)
        return dot_part + text_part

    # Regular line: remove sarga refs and punctuation
    line = SARGA_REF.sub('', line)
    line = line.translate(PUNCTUATION)
    return line


def process_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    lines = [line.rstrip('\n') for line in lines]

    # Find where footnotes start (line that is exactly '---')
    footnote_start = None
    for i, line in enumerate(lines):
        if line.strip() == '---':
            footnote_start = i
            break

    # Find where initial header ends (consecutive # lines at the top)
    header_end = 0
    for i, line in enumerate(lines):
        if line.startswith('#'):
            header_end = i + 1
        else:
            break

    # Process only the body (between header and footnotes)
    # Inline # lines in the body (e.g. section titles) are also preserved
    result = []
    for i, line in enumerate(lines):
        if i < header_end:
            # Top header - keep as-is
            result.append(line)
        elif footnote_start is not None and i >= footnote_start:
            # Footnotes - keep as-is
            result.append(line)
        elif line.startswith('#'):
            # Inline section header in body - only remove ఁ
            result.append(line.replace('\u0c01', ''))
        else:
            # Body - clean it
            result.append(clean_body_line(line))

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write('\n'.join(result) + '\n')


def main():
    data_dir = str(DATA_DIR)
    txt_files = sorted(glob.glob(os.path.join(data_dir, '**', '*.txt'), recursive=True))

    print(f"Found {len(txt_files)} files to process")
    for filepath in txt_files:
        rel = os.path.relpath(filepath, data_dir)
        try:
            process_file(filepath)
            print(f"  Cleaned: {rel}")
        except Exception as e:
            print(f"  ERROR: {rel}: {e}")

    print("Done!")


if __name__ == '__main__':
    main()
