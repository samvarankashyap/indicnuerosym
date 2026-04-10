#!/usr/bin/env python3
"""Clean the Basava Puranam dataset by removing punctuation marks.

Processes files recursively in nested folders, splitting hyphenated lines
into separate dwipada verses.
"""

from dwipada.data.clean_base import run_cleaner
from dwipada.paths import DATA_DIR


def main():
    run_cleaner(
        DATA_DIR / "basava_puranam",
        "బసవపురాణము (Basava Puranam)",
        recursive=True,
        split_on_hyphens=True,
    )


if __name__ == "__main__":
    main()
