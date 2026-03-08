#!/usr/bin/env python3
"""
Prepare training data for Gemma 3 instruction fine-tuning on Telugu Dwipada poetry generation.

Reads from the single source of truth (datasets/dwipada_augmented_dataset.json),
filters poems for 100% metrical purity using dwipada_analyzer, and creates
instruction-completion JSONL files for SFTTrainer.

Output:
  - training_data/train.jsonl   (80%)
  - training_data/val.jsonl     (10%)
  - training_data/test.jsonl    (10%, held-out for post-training evaluation)
  - training_data/data_stats.json
"""

import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

from dwipada.core.analyzer import analyze_dwipada
from dwipada.core.constants import DWIPADA_RULES_BLOCK
from dwipada.paths import DATASETS_DIR, TRAINING_DATA_DIR

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

AUGMENTED_DATASET = DATASETS_DIR / "dwipada_augmented_dataset.json"

SEED = 42
VAL_RATIO = 0.10
TEST_RATIO = 0.10

# Instruction templates for poem generation (Telugu + English variations)
INSTRUCTION_TEMPLATES_GENERIC = [
    "ద్విపదలో ఒక పద్యం వ్రాయండి.",
    "పై నియమాల ప్రకారం ద్విపద రాయండి.",
    "ద్విపద ఛందస్సులో రెండు పాదాలు వ్రాయండి.",
    "Write a Telugu dwipada couplet following the above rules.",
    "Compose a metrically correct Telugu dwipada.",
    "Generate a dwipada poem following the rules above.",
]

INSTRUCTION_TEMPLATES_WORK = [
    "{గ్రంథము} శైలిలో ద్విపద వ్రాయండి.",
    "Compose a dwipada in the style of {work}.",
]

INSTRUCTION_TEMPLATES_BHAVAM_TE = [
    "ఈ భావంతో ద్విపద వ్రాయండి: {bhavam}",
]

INSTRUCTION_TEMPLATES_BHAVAM_EN = [
    "Write a dwipada that expresses: {bhavam}",
    "Compose a dwipada with the following meaning: {bhavam}",
]

# Work name mappings
WORK_NAMES_TELUGU = {
    "basava_puranam": "బసవపురాణము",
    "ranganatha_ramayanam": "రంగనాథ రామాయణము",
    "dwipada_bhagavatam": "ద్విపద భాగవతము",
    "dwipada_bhagavatam2": "ద్విపద భాగవతము",
    "srirama_parinayamu": "శ్రీరమాపరిణయము",
    "palanati_veera_charitra": "పలనాటి వీర చరిత్ర",
}

WORK_NAMES_ENGLISH = {
    "basava_puranam": "Basava Puranam",
    "ranganatha_ramayanam": "Ranganatha Ramayanam",
    "dwipada_bhagavatam": "Dwipada Bhagavatam",
    "dwipada_bhagavatam2": "Dwipada Bhagavatam",
    "srirama_parinayamu": "Srirama Parinayamu",
    "palanati_veera_charitra": "Palanati Veera Charitra",
}


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def format_sample(instruction: str, poem: str) -> dict:
    """Format a single training sample as {input, output} with rules block."""
    return {
        "input": f"{DWIPADA_RULES_BLOCK}\n\n{instruction}",
        "output": poem,
    }


def is_metrically_pure(poem: str) -> bool:
    """Check if a poem scores 100% on the dwipada analyzer."""
    try:
        analysis = analyze_dwipada(poem)
        return analysis["match_score"]["overall"] == 100.0
    except Exception:
        return False


def normalize_poem(poem: str) -> str:
    """Normalize poem text for deduplication."""
    return "\n".join(line.strip() for line in poem.strip().split("\n"))


def pick_instruction(entry: dict) -> str:
    """Pick an instruction template based on available metadata in the entry."""
    options = [random.choice(INSTRUCTION_TEMPLATES_GENERIC)]
    source = entry.get("source", "")

    # Work-style instruction for non-synthetic entries
    if source and source != "synthetic":
        telugu_name = WORK_NAMES_TELUGU.get(source, "")
        english_name = WORK_NAMES_ENGLISH.get(source, "")
        if telugu_name:
            options.append(
                random.choice(INSTRUCTION_TEMPLATES_WORK).format(
                    గ్రంథము=telugu_name, work=english_name,
                )
            )

    # Bhavam-based instruction from meanings
    telugu_meaning = entry.get("telugu_meaning", "")
    english_meaning = entry.get("english_meaning", "")
    if telugu_meaning:
        options.append(
            random.choice(INSTRUCTION_TEMPLATES_BHAVAM_TE).format(
                bhavam=telugu_meaning,
            )
        )
    if english_meaning:
        options.append(
            random.choice(INSTRUCTION_TEMPLATES_BHAVAM_EN).format(
                bhavam=english_meaning,
            )
        )

    return random.choice(options)


# ──────────────────────────────────────────────────────────────────────────────
# Build training samples
# ──────────────────────────────────────────────────────────────────────────────

def build_samples():
    """Build all training samples from the augmented dataset."""
    print(f"Loading {AUGMENTED_DATASET}...")
    if not AUGMENTED_DATASET.exists():
        print(f"ERROR: {AUGMENTED_DATASET} not found.")
        print("Run 'dwipada augment' and 'dwipada combine' first.")
        sys.exit(1)

    data = json.loads(AUGMENTED_DATASET.read_text(encoding="utf-8"))
    print(f"  Loaded {len(data)} entries")

    # Group by source for balanced sampling
    by_source = {}
    for entry in data:
        source = entry.get("source", "unknown")
        by_source.setdefault(source, []).append(entry)

    print(f"  Sources: {', '.join(f'{k} ({len(v)})' for k, v in sorted(by_source.items()))}")

    seen_poems = set()
    samples = []
    stats = {"total": len(data), "by_source": {}, "passed_by_source": {}}

    for source, entries in sorted(by_source.items()):
        stats["by_source"][source] = len(entries)
        passed = 0

        for entry in entries:
            poem = normalize_poem(entry["poem"])
            if poem in seen_poems:
                continue
            if not is_metrically_pure(poem):
                continue
            seen_poems.add(poem)
            passed += 1

            instruction = pick_instruction(entry)
            sample = format_sample(instruction, poem)
            samples.append({**sample, "source": source})

        stats["passed_by_source"][source] = passed
        print(f"  [{source}]: {passed}/{len(entries)} passed (100% purity)")

    print(f"\nTotal samples: {len(samples)}")
    return samples, stats


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    random.seed(SEED)
    os.makedirs(TRAINING_DATA_DIR, exist_ok=True)

    print("=" * 60)
    print("Preparing training data for Gemma 3 Dwipada fine-tuning")
    print("=" * 60)

    samples, stats = build_samples()

    if not samples:
        print("No samples passed the metrical purity filter. Exiting.")
        sys.exit(1)

    # Shuffle and split 80/10/10
    random.shuffle(samples)
    test_size  = max(1, int(len(samples) * TEST_RATIO))
    val_size   = max(1, int(len(samples) * VAL_RATIO))
    test_samples  = samples[:test_size]
    val_samples   = samples[test_size:test_size + val_size]
    train_samples = samples[test_size + val_size:]

    # Write JSONL files
    train_path = TRAINING_DATA_DIR / "train.jsonl"
    val_path   = TRAINING_DATA_DIR / "val.jsonl"
    test_path  = TRAINING_DATA_DIR / "test.jsonl"
    stats_path = TRAINING_DATA_DIR / "data_stats.json"

    for path, split in [(train_path, train_samples), (val_path, val_samples), (test_path, test_samples)]:
        with open(path, "w", encoding="utf-8") as f:
            for s in split:
                f.write(json.dumps({"input": s["input"], "output": s["output"]}, ensure_ascii=False) + "\n")

    # Stats
    stats.update({
        "total_samples": len(samples),
        "train_size": len(train_samples),
        "val_size": len(val_samples),
        "test_size": len(test_samples),
        "train_by_source": dict(Counter(s["source"] for s in train_samples)),
        "val_by_source":   dict(Counter(s["source"] for s in val_samples)),
        "test_by_source":  dict(Counter(s["source"] for s in test_samples)),
    })

    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"\nTrain: {len(train_samples)} samples -> {train_path}")
    print(f"Val:   {len(val_samples)} samples -> {val_path}")
    print(f"Test:  {len(test_samples)} samples -> {test_path}")
    print(f"Stats: {stats_path}")
    print("Done!")


if __name__ == "__main__":
    main()
