"""
Tokenize Telugu text using the Gemma 3 tokenizer.

Usage:
    python -m dwipada.training.tokenizer "తెలుగు వచనం ఇక్కడ"
    python -m dwipada.training.tokenizer              # interactive mode
"""

import sys
from transformers import AutoTokenizer

MODEL_ID = "google/gemma-3-4b-it"


def load_tokenizer():
    """Load the Gemma 3 tokenizer from HuggingFace."""
    print(f"Loading tokenizer from {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    return tokenizer


def tokenize_text(tokenizer, text: str):
    """Tokenize text and return detailed breakdown."""
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    decoded_tokens = [tokenizer.decode([tid]) for tid in token_ids]

    return {
        "text": text,
        "token_ids": token_ids,
        "tokens": tokens,
        "decoded_tokens": decoded_tokens,
        "num_tokens": len(token_ids),
    }


def print_result(result: dict):
    """Pretty-print tokenization results."""
    print(f"\nInput text : {result['text']}")
    print(f"Num tokens : {result['num_tokens']}")
    print(f"Token IDs  : {result['token_ids']}")
    print()
    print(f"{'Index':<6} {'Token ID':<10} {'Raw Token':<30} {'Decoded'}")
    print("-" * 80)
    for i, (tid, tok, dec) in enumerate(
        zip(result["token_ids"], result["tokens"], result["decoded_tokens"])
    ):
        print(f"{i:<6} {tid:<10} {tok:<30} {dec}")


def main():
    tokenizer = load_tokenizer()

    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
        result = tokenize_text(tokenizer, text)
        print_result(result)
    else:
        print("\nInteractive mode -- type Telugu text and press Enter (Ctrl+C to quit)\n")
        while True:
            try:
                text = input(">>> ").strip()
                if not text:
                    continue
                result = tokenize_text(tokenizer, text)
                print_result(result)
                print()
            except (KeyboardInterrupt, EOFError):
                print("\nDone.")
                break


if __name__ == "__main__":
    main()
