#!/usr/bin/env python3
"""
Generate Vertex AI batch request JSONL for LLM-as-Judge evaluation.

Reads datasets/dwipada_augmented_dataset.json and writes:
    llm_as_judge/batch_requests_laj.jsonl

Each request asks the LLM to act as a judge and evaluate the quality of the
Telugu meaning and English meaning provided for a dwipada couplet, using a
structured multi-dimensional rubric.

Rubric design draws on:
  - G-Eval (Liu et al., 2023): chain-of-thought NLG evaluation framework
  - GEMBA (Kocmi & Federmann, 2023): GPT-based translation quality assessment
  - Classical MT evaluation dimensions: Adequacy & Fluency
  - Domain-specific criteria for Telugu classical poetry

Usage:
    python llm_as_judge/generate_batch_requests_for_laj.py [--sample N] [--seed S]
"""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).parent.parent
INPUT_FILE = ROOT / "datasets" / "dwipada_augmented_dataset.json"
LAJ_DIR = Path(__file__).parent
DEFAULT_OUTPUT = LAJ_DIR / "batch_requests_laj.jsonl"

# ─────────────────────────────────────────────────────────────────────────────
# PROMPT
# ─────────────────────────────────────────────────────────────────────────────

JUDGE_PROMPT = """\
You are an expert judge, telugu and english scholar evaluating the quality of meanings (translations/interpretations) \
provided for Telugu classical dwipada poetry. Your evaluation must be rigorous, evidence-based, \
and grounded strictly in the provided text.

---

## Input

**Poem (Telugu Dwipada Couplet):**
{poem}

**Telugu Meaning (Bhavam):**
{telugu_meaning}

**English Meaning:**
{english_meaning}

---

## Evaluation Rubric

Evaluate across the following 5 dimensions. For each dimension, provide:
- A score from 1 to 5
- A one-sentence justification citing specific evidence from the text

### Dimension 1 — Semantic Fidelity (1–5)
Does the meaning accurately and faithfully convey the core message of the poem?
- 5: All key semantic content is captured with no distortions or additions
- 4: Mostly accurate with minor omissions or slight paraphrase
- 3: Partially accurate; one key concept is missing or slightly distorted
- 2: Several inaccuracies or significant concepts are missing
- 1: The meaning does not reflect the poem's content

### Dimension 2 — Completeness (1–5)
Are all poetic elements — imagery, metaphors, comparisons, and descriptive nuances — present in the meaning?
- 5: Every element of the couplet is accounted for
- 4: Nearly complete; one minor element is glossed over
- 3: Moderate gaps; imagery or a key phrase is absent
- 2: Only the surface meaning is given; deeper elements are largely missing
- 1: Grossly incomplete

### Dimension 3 — Cultural & Contextual Accuracy (1–5)
Are mythological names, philosophical concepts, Sanskrit-origin terms, and cultural allusions correctly interpreted?
- 5: All cultural/mythological references are correctly identified and explained
- 4: Mostly correct; one minor reference is imprecise
- 3: One significant reference is incorrect or ambiguous
- 2: Multiple cultural elements are misidentified or absent
- 1: Cultural context is largely wrong or ignored

### Dimension 4 — Telugu Linguistic Quality (1–5)
Is the Telugu meaning grammatically correct, idiomatic, and natural-sounding?
- 5: Fluent, grammatically correct, natural Telugu prose
- 4: Mostly fluent with one awkward phrase
- 3: Understandable but noticeably unnatural or stiff
- 2: Grammatical errors or unidiomatic usage that hinders comprehension
- 1: Poorly constructed Telugu

### Dimension 5 — English Linguistic Quality (1–5)
Is the English meaning grammatically correct, idiomatic, and natural-sounding?
- 5: Fluent, grammatically correct, natural English prose
- 4: Mostly fluent with one awkward phrase
- 3: Understandable but noticeably unnatural or stiff
- 2: Grammatical errors or unidiomatic usage that hinders comprehension
- 1: Poorly constructed English

---

## Required Output Format

Respond ONLY in the exact structure below. Do not add any text outside this structure.

Semantic Fidelity: <score>/5
Justification: <one sentence>

Completeness: <score>/5
Justification: <one sentence>

Cultural & Contextual Accuracy: <score>/5
Justification: <one sentence>

Telugu Linguistic Quality: <score>/5
Justification: <one sentence>

English Linguistic Quality: <score>/5
Justification: <one sentence>

Total Score: <sum>/25
Overall Verdict: <Excellent | Good | Acceptable | Poor>
Critical Issues: <brief note on the most serious problem found, or "None">\
"""


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def stratified_sample(dataset: list, total: int, seed: int) -> list:
    """Return `total` records sampled as evenly as possible across sources."""
    random.seed(seed)

    by_source = defaultdict(list)
    for record in dataset:
        by_source[record.get("source", "unknown")].append(record)

    sources = list(by_source.keys())

    # Iteratively allocate quota: sources that can't fill their share donate
    # the remainder back to sources that still have records.
    allocated = {src: 0 for src in sources}
    remaining = total
    exhausted = set()

    while remaining > 0:
        active = [s for s in sources if s not in exhausted]
        if not active:
            break
        per_source = max(1, remaining // len(active))
        gained = 0
        for src in active:
            if remaining <= 0:
                break
            available = len(by_source[src]) - allocated[src]
            give = min(per_source, available, remaining)
            allocated[src] += give
            remaining -= give
            gained += give
            if allocated[src] >= len(by_source[src]):
                exhausted.add(src)
        if gained == 0:
            break

    sampled = []
    for src in sources:
        n = allocated[src]
        sampled.extend(random.sample(by_source[src], n))

    random.shuffle(sampled)
    return sampled


def build_request(record: dict, index: int) -> dict:
    poem = record.get("poem", "").strip()
    telugu_meaning = record.get("telugu_meaning", "").strip()
    english_meaning = record.get("english_meaning", "").strip()
    source = record.get("source", "unknown")

    prompt_text = JUDGE_PROMPT.format(
        poem=poem,
        telugu_meaning=telugu_meaning,
        english_meaning=english_meaning,
    )

    return {
        "key": f"{source}__{index}",
        "request": {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt_text}],
                }
            ]
        },
        "metadata": {
            "index": index,
            "source": source,
            "poem": poem,
            "is_synthetic": record.get("is_synthetic_data", False),
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate LLM-as-Judge batch requests for dwipada dataset."
    )
    parser.add_argument(
        "--sample", type=int, default=250,
        help="Total number of records to sample, distributed evenly across sources (default: 250)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--output", "-o", default=str(DEFAULT_OUTPUT),
        help=f"Output JSONL file (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()

    if not INPUT_FILE.exists():
        print(f"Error: Input file not found: {INPUT_FILE}", file=sys.stderr)
        sys.exit(1)

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        full_dataset = json.load(f)

    # Sources to exclude from sampling
    EXCLUDED_SOURCES = {"dwipada_bhagavatam"}

    # Drop records with missing required fields or excluded sources before sampling
    valid = [
        r for r in full_dataset
        if r.get("poem", "").strip()
        and r.get("telugu_meaning", "").strip()
        and r.get("english_meaning", "").strip()
        and r.get("source", "unknown") not in EXCLUDED_SOURCES
    ]
    dropped = len(full_dataset) - len(valid)

    dataset = stratified_sample(valid, total=args.sample, seed=args.seed)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    from collections import Counter
    source_counts = Counter()

    with open(out_path, "w", encoding="utf-8") as out_f:
        for idx, record in enumerate(dataset, start=1):
            entry = build_request(record, idx)
            out_f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            source_counts[record.get("source", "unknown")] += 1

    print(f"{'=' * 55}")
    print(f"LLM-AS-JUDGE BATCH REQUEST GENERATION COMPLETE")
    print(f"{'=' * 55}")
    print(f"  Total records in dataset: {len(full_dataset)}")
    print(f"  Dropped (missing data):   {dropped}")
    print(f"  Requested sample size:    {args.sample}")
    print(f"  Requests written:         {sum(source_counts.values())}")
    print(f"  Random seed:              {args.seed}")
    print(f"  Output:                   {out_path}")
    print(f"  Breakdown by source:")
    for src, cnt in sorted(source_counts.items()):
        print(f"    {src}: {cnt}")
    print(f"{'=' * 55}")


if __name__ == "__main__":
    main()
