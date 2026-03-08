"""
Extract all Telugu tokens from the Gemma 3 tokenizer vocabulary.

A token is included if it:
  - contains at least one character in the Telugu Unicode block (U+0C00–U+0C7F), OR
  - is a pure whitespace token (spaces and/or newlines only)

Whitespace tokens are needed by the NFA orchestrator for word-boundary and
line-boundary tracking (see design doc section 7).

Output: telugu_tokens.tsv  (token_id <TAB> token_text)

Usage:
    python nfa_for_dwipada/extract_telugu_tokens.py
"""

from transformers import AutoTokenizer

MODEL_ID = "google/gemma-3-4b-it"
OUTPUT_FILE = "nfa_for_dwipada/telugu_tokens.tsv"

TELUGU_RANGE_START = 0x0C00
TELUGU_RANGE_END = 0x0C7F
WHITESPACE_CHARS = {" ", "\n"}


def is_telugu(text: str) -> bool:
    """Return True if text contains at least one Telugu Unicode character."""
    return any(TELUGU_RANGE_START <= ord(ch) <= TELUGU_RANGE_END for ch in text)


def is_pure_whitespace(text: str) -> bool:
    """Return True if text is non-empty and consists only of spaces/newlines."""
    return len(text) > 0 and all(ch in WHITESPACE_CHARS for ch in text)


def main():
    print(f"Loading tokenizer from {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    vocab_size = tokenizer.vocab_size
    print(f"Vocabulary size: {vocab_size}")

    telugu_tokens = []
    whitespace_tokens = []

    for token_id in range(vocab_size):
        token_text = tokenizer.decode([token_id], skip_special_tokens=False)
        if is_telugu(token_text):
            telugu_tokens.append((token_id, token_text))
        elif is_pure_whitespace(token_text):
            whitespace_tokens.append((token_id, token_text))

    all_tokens = telugu_tokens + whitespace_tokens
    print(f"Found {len(telugu_tokens)} Telugu tokens")
    print(f"Found {len(whitespace_tokens)} pure whitespace tokens (space/newline)")
    print(f"Total: {len(all_tokens)}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("token_id\ttoken_text\n")
        for token_id, token_text in all_tokens:
            f.write(f"{token_id}\t{token_text}\n")

    print(f"Written to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
