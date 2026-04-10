#!/usr/bin/env python3
"""Clean the Palanati Veera Charitra dataset by removing punctuation marks.

Processes files in the top-level directory only (non-recursive), removing
trailing numbers from verse lines.
"""

from dwipada.data.clean_base import run_cleaner
from dwipada.paths import DATA_DIR


def main():
    run_cleaner(
        DATA_DIR / "palanati_veera_charitra",
        "పాలనాటి వీర చరిత్ర (Palanati Veera Charitra)",
        recursive=False,
        remove_trailing_numbers=True,
    )


if __name__ == "__main__":
    main()
