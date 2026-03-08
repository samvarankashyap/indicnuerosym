#!/usr/bin/env python3
"""Clean the Sri Rama Parinayamu dataset by removing punctuation marks.

Processes files in the top-level directory only (non-recursive), with
hyphen '-' added to the characters to remove.
"""

from dwipada.data.clean_base import run_cleaner
from dwipada.paths import DATA_DIR


def main():
    run_cleaner(
        DATA_DIR / "srirama_parinayamu",
        "శ్రీరమాపరిణయము (Sri Rama Parinayamu)",
        recursive=False,
        extra_chars=['-'],
    )


if __name__ == "__main__":
    main()
