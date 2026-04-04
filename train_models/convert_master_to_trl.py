#!/usr/bin/env python3
"""
Convert dwipada_master_dataset.json → TRL-format JSONL for LoRA fine-tuning.

Generates 4 instruction profiles (matching the original IFT pipeline):
  GN1: Generic Telugu prompt (generate from Telugu bhavam)
  GN2: Generic English prompt (generate from English meaning)
  GN3: Human-style prompt (generate from Telugu bhavam, casual)
  GN4: Scholarly prompt (generate with chandassu rules reminder)

Each record becomes a 3-turn chat: system → user → assistant.
The assistant response always contains the dwipada poem.

Usage:
    python train_models/convert_master_to_trl.py

    # Custom output path:
    python train_models/convert_master_to_trl.py --output train_models/ift_data/master_trl.jsonl
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR.parent / "datasets" / "dwipada_master_dataset.json"
DEFAULT_OUTPUT = SCRIPT_DIR / "ift_data" / "master_generation_trl.jsonl"

# -- System prompt (same as original IFT pipeline) --
SYSTEM_PROMPT = "You are a Telugu and Sanskrit scholar specialising in Dwipada poetry."

# -- Dwipada rules block (embedded in scholarly prompts) --
RULES_BLOCK = (
    "ద్విపద నియమాలు:\n"
    "- ద్విపద = 2 పాదాలు. ప్రతి పాదంలో 4 గణాలు: 3 ఇంద్ర గణాలు + 1 సూర్య గణము.\n"
    "- ఇంద్ర గణాలు: నల (IIII), నగ (IIIU), సల (IIUI), భ (UII), ర (UIU), త (UUI)\n"
    "- సూర్య గణాలు: న (III), హ/గల (UI)\n"
    "- ప్రాస: రెండు పాదాల 2వ అక్షరం హల్లు సమానంగా ఉండాలి\n"
    "- యతి: ప్రతి పాదంలో 1వ గణం మొదటి అక్షరం = 3వ గణం మొదటి అక్షరం\n"
)

# -- Instruction templates --
# Each returns (user_content, profile_id, source_tag)

def make_gn1(entry):
    """Generic Telugu: generate from Telugu bhavam."""
    meaning = entry.get("telugu_meaning", "")
    if not meaning:
        return None
    user = (
        "క్రింది తెలుగు భావానికి అనుగుణంగా ఒక ద్విపద పద్యం రచించండి. "
        "ప్రతి పాదంలో 3 ఇంద్ర గణాలు + 1 సూర్య గణం ఉండాలి.\n\n"
        f"తెలుగు భావం: {meaning}"
    )
    return user, "GN1", "[Telugu_Bhavam]"


def make_gn2(entry):
    """Generic English: generate from English meaning."""
    meaning = entry.get("english_meaning", "")
    if not meaning:
        return None
    user = (
        "క్రింది ఆంగ్ల భావానికి అనుగుణంగా ఒక ద్విపద పద్యం రచించండి. "
        "ప్రతి పాదంలో 3 ఇంద్ర గణాలు + 1 సూర్య గణం ఉండాలి.\n\n"
        f"English meaning: {meaning}"
    )
    return user, "GN2", "[English_Meaning]"


def make_gn3(entry):
    """Human-style: casual Telugu prompt from meaning."""
    meaning = entry.get("telugu_meaning", "")
    if not meaning:
        return None
    user = (
        "క్రింది తెలుగు భావానికి అనుగుణంగా ఒక ద్విపద పద్యం రచించండి. "
        "ప్రతి పాదంలో 3 ఇంద్ర గణాలు + 1 సూర్య గణం ఉండాలి.\n\n"
        f"తెలుగు భావం: {meaning}"
    )
    return user, "GN3", "[Human_Style]"


def make_gn4(entry):
    """Scholarly: includes full chandassu rules in prompt."""
    meaning = entry.get("telugu_meaning", "")
    if not meaning:
        return None
    user = (
        f"{RULES_BLOCK}\n"
        "పై నియమాలను పాటిస్తూ క్రింది భావానికి ద్విపద రచించండి.\n\n"
        f"భావం: {meaning}"
    )
    return user, "GN4", "[Scholarly]"


PROFILE_MAKERS = [make_gn1, make_gn2, make_gn3, make_gn4]


def convert(input_path, output_path, seed=42):
    """Convert master dataset to TRL-format JSONL."""

    print(f"Loading: {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"  {len(data)} poems loaded")

    random.seed(seed)
    records = []

    for entry in data:
        poem = entry.get("poem", "").strip()
        if not poem:
            continue

        source = entry.get("source", "unknown")
        is_synthetic = entry.get("is_synthetic_data", False)

        # Randomly pick a profile for this poem
        maker = random.choice(PROFILE_MAKERS)
        result = maker(entry)
        if result is None:
            # Fallback to GN3 if chosen profile can't generate
            for m in PROFILE_MAKERS:
                result = m(entry)
                if result is not None:
                    break
        if result is None:
            continue

        user_content, profile_id, source_tag = result

        # Build assistant response
        assistant_content = f"ద్విపద:\n{poem}"

        record = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content},
            ],
            "_profile_id": profile_id,
            "_model": "generation",
            "_source_tag": source_tag,
            "_source": source,
        }
        records.append(record)

    # Shuffle for training
    random.shuffle(records)

    # Write JSONL
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"  {len(records)} TRL records written to: {output_path}")
    return len(records)


def main():
    parser = argparse.ArgumentParser(description="Convert master dataset to TRL-format JSONL")
    parser.add_argument("--input", type=str, default=str(DEFAULT_INPUT),
                        help=f"Input JSON path (default: {DEFAULT_INPUT})")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT),
                        help=f"Output JSONL path (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    convert(args.input, args.output, args.seed)


if __name__ == "__main__":
    main()
