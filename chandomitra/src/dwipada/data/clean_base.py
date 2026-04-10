"""Base cleaner with shared utilities for all Telugu poetry data cleaners.

Consolidates the common CHARS_TO_REMOVE list, clean_line(), clean_file(),
and the main processing loop shared across 4 cleaner scripts.
"""

import re
from pathlib import Path

from dwipada.paths import DATA_DIR

# Characters to remove — shared across all cleaners
CHARS_TO_REMOVE = [
    # Single quotes (ASCII and Unicode curly)
    "'",            # U+0027 ASCII apostrophe
    "\u2018",       # U+2018 left single quotation mark
    "\u2019",       # U+2019 right single quotation mark
    # Double quotes (ASCII and Unicode curly)
    '"',            # U+0022 ASCII double quote
    "\u201C",       # U+201C left double quotation mark
    "\u201D",       # U+201D right double quotation mark
    "\u201E",       # U+201E double low-9 quotation mark
    '`', '´',       # Backticks
    '«', '»',       # Guillemets
    # Punctuation
    ',',            # Comma
    '?',            # Question mark
    '!',            # Exclamation
    '–', '—',       # En-dash, Em-dash
    # Telugu specific
    'ఁ',            # Arasunna (chandrabindu)
    # Whitespace
    "\u00A0",       # U+00A0 non-breaking space
    # Other
    ';',            # Semicolon
    ':',            # Colon (but keep in metadata lines)
    '.',            # Period
    '(',  ')',      # Parentheses
    '[', ']',       # Brackets
]


def clean_line(line: str, is_metadata: bool = False,
               extra_chars: list = None,
               remove_trailing_numbers: bool = False) -> str:
    """Clean a single line by removing specified characters.

    Args:
        line: The line to clean.
        is_metadata: If True, preserve colons for metadata format.
        extra_chars: Additional characters to remove beyond CHARS_TO_REMOVE.
        remove_trailing_numbers: If True, strip trailing digits from non-metadata lines.
    """
    chars = list(CHARS_TO_REMOVE)
    if extra_chars:
        chars.extend(extra_chars)
    if is_metadata:
        chars = [c for c in chars if c != ':']

    for char in chars:
        line = line.replace(char, '')

    if remove_trailing_numbers and not is_metadata:
        line = re.sub(r'[0-9]+\s*$', '', line)

    line = re.sub(r'  +', ' ', line)
    return line.strip()


def clean_file(filepath: Path,
               extra_chars: list = None,
               remove_trailing_numbers: bool = False,
               split_on_hyphens: bool = False) -> tuple:
    """Clean a single file in-place.

    Args:
        filepath: Path to the text file.
        extra_chars: Additional characters to remove.
        remove_trailing_numbers: Strip trailing digits from verse lines.
        split_on_hyphens: Split lines containing hyphens into separate verses.

    Returns:
        Tuple of (lines_processed, chars_removed).
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    original_len = len(content)
    lines = content.split('\n')
    cleaned_lines = []

    for line in lines:
        is_metadata = line.startswith('#')
        cleaned = clean_line(
            line, is_metadata,
            extra_chars=extra_chars,
            remove_trailing_numbers=remove_trailing_numbers,
        )

        if split_on_hyphens and not is_metadata and '-' in cleaned:
            parts = cleaned.split('-')
            for part in parts:
                part = part.strip()
                if part:
                    cleaned_lines.append(part)
        else:
            cleaned_lines.append(cleaned)

    cleaned_content = '\n'.join(cleaned_lines)
    chars_removed = original_len - len(cleaned_content)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(cleaned_content)

    return len(lines), chars_removed


def run_cleaner(data_dir: Path, dataset_name: str,
                recursive: bool = True,
                extra_chars: list = None,
                remove_trailing_numbers: bool = False,
                split_on_hyphens: bool = False):
    """Run the standard cleaning loop on all .txt files in a directory.

    Args:
        data_dir: Directory containing text files to clean.
        dataset_name: Human-readable dataset name for logging.
        recursive: If True, search subdirectories.
        extra_chars: Additional characters to remove.
        remove_trailing_numbers: Strip trailing digits from verse lines.
        split_on_hyphens: Split hyphenated lines into separate verses.
    """
    print("=" * 60)
    print(f"Cleaning {dataset_name}")
    print("=" * 60)
    print(f"Directory: {data_dir}")
    print(f"Characters to remove: {len(CHARS_TO_REMOVE)} types")
    print("=" * 60)

    glob_method = data_dir.rglob if recursive else data_dir.glob
    files = sorted(glob_method("*.txt"))
    print(f"Found {len(files)} files to clean\n")

    if not files:
        print("No files found. Run the crawler first.")
        return

    total_lines = 0
    total_chars_removed = 0
    current_folder = None

    for filepath in files:
        if filepath.parent != current_folder:
            current_folder = filepath.parent
            print(f"\n  [{current_folder.name}]")

        lines, chars_removed = clean_file(
            filepath,
            extra_chars=extra_chars,
            remove_trailing_numbers=remove_trailing_numbers,
            split_on_hyphens=split_on_hyphens,
        )
        total_lines += lines
        total_chars_removed += chars_removed
        print(f"    {filepath.name[:45]}... - {chars_removed} chars removed")

    print("\n" + "=" * 60)
    print("CLEANING COMPLETE")
    print("=" * 60)
    print(f"Files processed: {len(files)}")
    print(f"Total lines: {total_lines}")
    print(f"Total characters removed: {total_chars_removed}")
