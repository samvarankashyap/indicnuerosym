"""
Analyze coverage between the Telugu Unicode block and Gemma 3's Telugu tokens.

1. Lists all characters in the Telugu Unicode block (U+0C00 - U+0C7F).
2. Checks which block characters appear in telugu_tokens.tsv (block -> tokens).
3. Checks which characters in telugu_tokens.tsv fall outside the block (tokens -> block).

Usage:
    python nfa_for_dwipada/analyze_telugu_coverage.py
"""

import unicodedata

TELUGU_RANGE_START = 0x0C00
TELUGU_RANGE_END = 0x0C7F
TOKENS_FILE = "nfa_for_dwipada/telugu_tokens.tsv"


def load_telugu_tokens(path: str) -> list[tuple[int, str]]:
    tokens = []
    with open(path, encoding="utf-8") as f:
        next(f)  # skip header
        for line in f:
            parts = line.rstrip("\n").split("\t", 1)
            if len(parts) == 2:
                tokens.append((int(parts[0]), parts[1]))
    return tokens


def telugu_block_chars() -> list[tuple[int, str, str]]:
    """Return all assigned characters in the Telugu Unicode block."""
    chars = []
    for cp in range(TELUGU_RANGE_START, TELUGU_RANGE_END + 1):
        ch = chr(cp)
        name = unicodedata.name(ch, "")
        if name:  # unassigned codepoints have no name
            chars.append((cp, ch, name))
    return chars


def main():
    block = telugu_block_chars()
    block_codepoints = {cp for cp, _, _ in block}
    block_chars = {ch for _, ch, _ in block}

    tokens = load_telugu_tokens(TOKENS_FILE)

    # Collect all unique characters that appear in the token texts
    chars_in_tokens: dict[str, list[int]] = {}  # char -> [token_ids]
    for token_id, token_text in tokens:
        for ch in token_text:
            chars_in_tokens.setdefault(ch, []).append(token_id)

    # --- Section 1: Full Telugu Unicode block ---
    print("=" * 70)
    print("TELUGU UNICODE BLOCK  (U+0C00 – U+0C7F)")
    print("=" * 70)
    print(f"{'Codepoint':<12} {'Char':<6} {'In tokens?':<12} {'Name'}")
    print("-" * 70)
    for cp, ch, name in block:
        used = ch in chars_in_tokens
        print(f"U+{cp:04X}      {ch!r:<6} {'YES' if used else 'NO':<12} {name}")

    # --- Section 2: Block chars NOT found in any token ---
    unused_in_tokens = [(cp, ch, name) for cp, ch, name in block if ch not in chars_in_tokens]
    print()
    print("=" * 70)
    print(f"BLOCK CHARS NOT USED IN ANY GEMMA TOKEN  ({len(unused_in_tokens)} of {len(block)})")
    print("=" * 70)
    if unused_in_tokens:
        for cp, ch, name in unused_in_tokens:
            print(f"  U+{cp:04X}  {ch!r}  {name}")
    else:
        print("  (all block characters appear in at least one token)")

    # --- Section 3: Chars in tokens that fall OUTSIDE the Telugu block ---
    outside_block: dict[str, list[int]] = {
        ch: tids
        for ch, tids in chars_in_tokens.items()
        if ch not in block_chars
    }
    print()
    print("=" * 70)
    print(f"CHARS IN TOKENS OUTSIDE THE TELUGU BLOCK  ({len(outside_block)} unique chars)")
    print("=" * 70)
    if outside_block:
        print(f"{'Char':<8} {'Codepoint':<12} {'Category':<20} {'Name':<40} {'# tokens'}")
        print("-" * 90)
        for ch in sorted(outside_block, key=ord):
            cp = ord(ch)
            cat = unicodedata.category(ch)
            name = unicodedata.name(ch, "<unknown>")
            n = len(outside_block[ch])
            print(f"{ch!r:<8} U+{cp:04X}      {cat:<20} {name:<40} {n}")
    else:
        print("  (all token characters are within the Telugu Unicode block)")

    # --- Summary ---
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Telugu block assigned codepoints : {len(block)}")
    print(f"  Block chars used in tokens       : {len(block_chars & chars_in_tokens.keys())}")
    print(f"  Block chars NOT in any token     : {len(unused_in_tokens)}")
    print(f"  Token chars outside the block    : {len(outside_block)}")
    print(f"  Total Telugu tokens loaded       : {len(tokens)}")


if __name__ == "__main__":
    main()
