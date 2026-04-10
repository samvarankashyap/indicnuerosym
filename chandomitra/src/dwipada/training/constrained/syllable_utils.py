"""Syllable utilities for constrained decoding.

Wraps the analyzer's split_aksharalu + akshara_ganavibhajana with
"stable prefix" logic to handle the look-ahead guru rule (Rule 5).

The look-ahead rule: a syllable becomes Guru if the NEXT syllable starts
with a conjunct or double consonant. During generation, the last syllable's
weight is provisional because we don't yet know what comes next. We handle
this by reporting which weights the last syllable COULD have.
"""

from __future__ import annotations

from typing import Optional

from dwipada.core.analyzer import (
    categorize_aksharam,
    split_aksharalu,
    akshara_ganavibhajana,
    ignorable_chars,
    halant,
)


def get_pure_aksharalu(text: str) -> list[str]:
    """Split text into syllables and filter out ignorable characters."""
    aksharalu = split_aksharalu(text)
    return [ak for ak in aksharalu if ak not in ignorable_chars]


def get_markers_with_stability(
    text: str,
) -> tuple[str, bool, bool]:
    """Compute I/U markers for text, separating stable from provisional.

    The look-ahead guru rule means the LAST syllable's weight can change
    when the next token arrives (if it starts with a conjunct). We split
    markers into a "stable prefix" (definitely correct) and flags for
    the last syllable's possible weights.

    Args:
        text: Current line text (Telugu).

    Returns:
        stable_prefix: I/U string for syllables 0..N-2 (won't change).
        last_could_be_laghu: True if last syllable could remain I.
        last_could_be_guru: True if last syllable could become/stay U.

    If text has 0 or 1 pure syllables, returns ("", True, True) — too
    early to constrain meaningfully.
    """
    aksharalu = split_aksharalu(text)
    markers = akshara_ganavibhajana(aksharalu)

    # Filter to pure (non-ignorable) syllables and their markers
    pure_pairs = [
        (ak, m) for ak, m in zip(aksharalu, markers) if ak not in ignorable_chars
    ]

    if len(pure_pairs) <= 1:
        return "", True, True

    # Stable prefix: all markers except the last
    stable = "".join(m for _, m in pure_pairs[:-1])

    # Last syllable: check its intrinsic weight
    last_marker = pure_pairs[-1][1]
    if last_marker == "U":
        # Already Guru by its own properties (long vowel, anusvara, etc.)
        # Can't become "more Guru" — stays U regardless of next token
        return stable, False, True
    else:
        # Currently Laghu (I), but could become Guru (U) if next token
        # starts with a conjunct/double consonant (look-ahead rule)
        return stable, True, True


def is_conjunct_start(text: str) -> bool:
    """Check if text starts with a conjunct or double consonant.

    Used to determine if appending this token would trigger the look-ahead
    guru rule on the previous syllable.
    """
    aksharalu = split_aksharalu(text)
    pure = [ak for ak in aksharalu if ak not in ignorable_chars]
    if not pure:
        return False
    tags = set(categorize_aksharam(pure[0]))
    return "సంయుక్తాక్షరం" in tags or "ద్విత్వాక్షరం" in tags


def get_full_markers(text: str) -> str:
    """Get the full I/U marker string for text (all syllables, no stability split).

    Useful for final validation after a line is complete (no look-ahead
    ambiguity since all syllables are known).
    """
    aksharalu = split_aksharalu(text)
    markers = akshara_ganavibhajana(aksharalu)
    return "".join(m for m in markers if m)
