"""
POS Tagger
===========
Reads the raw Dwipada corpus and adds tags_english + tags_telugu
to every record using spaCy (English) and Stanza (Telugu).

Requirements:
    pip install spacy stanza tqdm
    python -m spacy download en_core_web_sm
    python -c "import stanza; stanza.download('te')"

Usage:
    python pos_tagger.py --input dwipada_augmented_dataset.json
    python pos_tagger.py --input dwipada_augmented_dataset.json --output poems_with_pos.jsonl
"""

import json
import argparse
from pathlib import Path
from tqdm import tqdm


# ──────────────────────────────────────────────
# IO
# ──────────────────────────────────────────────

def load_data(path: str) -> list[dict]:
    """Handles JSON array (.json) and JSONL (.jsonl)."""
    with open(path, encoding="utf-8") as f:
        first = f.read(1); f.seek(0)
        if first == "[":
            data = json.load(f)
            if not isinstance(data, list):
                raise ValueError(f"{path} must be a JSON array.")
            return data
        return [json.loads(l) for l in f if l.strip()]

def save_jsonl(records: list[dict], path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  ✓ {len(records):>6} records → {path}")


# ──────────────────────────────────────────────
# MODEL LOADERS
# ──────────────────────────────────────────────

def load_spacy():
    try:
        import spacy
        return spacy.load("en_core_web_sm")
    except OSError:
        raise RuntimeError(
            "spaCy English model not found.\n"
            "Run: python -m spacy download en_core_web_sm"
        )

def load_stanza():
    try:
        import stanza
        return stanza.Pipeline(lang="te", processors="tokenize,pos", verbose=False)
    except Exception:
        raise RuntimeError(
            "Stanza Telugu model not found.\n"
            "Run: python -c \"import stanza; stanza.download('te')\""
        )


# ──────────────────────────────────────────────
# TAGGERS
# ──────────────────────────────────────────────

KEEP_UPOS = {"NOUN", "VERB", "ADJ", "ADV", "PROPN", "NUM"}

def tag_english(text: str, nlp) -> list[dict]:
    """Content words only — filters stop words, punctuation, determiners."""
    return [
        {"token": t.text, "pos": t.pos_}
        for t in nlp(text)
        if t.pos_ in KEEP_UPOS and not t.is_stop
    ]

def tag_telugu(text: str, nlp) -> list[dict]:
    """Returns Universal POS + Paninian xpos for each content word."""
    results = []
    for sent in nlp(text).sentences:
        for word in sent.words:
            if word.upos in KEEP_UPOS:
                results.append({
                    "token": word.text,
                    "pos":   word.upos,
                    "xpos":  word.xpos,
                })
    return results


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

def run(input_path: str, output_path: str):
    print(f"\n[Loading] {input_path}")
    records = load_data(input_path)
    print(f"  Loaded {len(records):,} records.")

    print("\n[Loading NLP models...]")
    nlp_en = load_spacy()
    nlp_te = load_stanza()
    print("  ✓ spaCy en_core_web_sm")
    print("  ✓ Stanza Telugu pipeline")

    print("\n[Tagging...]")
    empty_en = empty_te = 0
    for rec in tqdm(records, desc="POS Tagging"):
        eng = rec.get("english_meaning", "")
        tel = rec.get("telugu_meaning", "")
        rec["tags_english"] = tag_english(eng, nlp_en) if eng else []
        rec["tags_telugu"]  = tag_telugu(tel, nlp_te)  if tel else []
        if not rec["tags_english"]: empty_en += 1
        if not rec["tags_telugu"]:  empty_te += 1

    total = len(records)
    print(f"\n  tags_english populated : {total - empty_en:,} / {total:,} ({(total-empty_en)/total*100:.1f}%)")
    print(f"  tags_telugu  populated : {total - empty_te:,} / {total:,} ({(total-empty_te)/total*100:.1f}%)")

    print("\n[Saving...]")
    save_jsonl(records, output_path)
    print("\n✅ Done.")


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="POS tagger for Dwipada corpus")
    parser.add_argument("--input",  required=True,
                        help="Raw corpus (dwipada_augmented_dataset.json)")
    parser.add_argument("--output", default="poems_with_pos.jsonl",
                        help="Output path (default: poems_with_pos.jsonl)")
    args = parser.parse_args()
    run(args.input, args.output)