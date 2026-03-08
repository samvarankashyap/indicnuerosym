#!/usr/bin/env python3
"""Clean the Dwipada Bhagavatam dataset by removing punctuation marks.

Processes files recursively in nested folders, removing trailing numbers
from verse lines.
"""

from dwipada.data.clean_base import run_cleaner
from dwipada.paths import DATA_DIR


def main():
    run_cleaner(
        DATA_DIR / "dwipada_bhagavatam",
        "ద్విపద భాగవతము (Dwipada Bhagavatam)",
        recursive=True,
        remove_trailing_numbers=True,
    )


if __name__ == "__main__":
    main()
