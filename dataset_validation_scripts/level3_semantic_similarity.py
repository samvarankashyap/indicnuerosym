#!/usr/bin/env python3
"""
Level 3: Semantic Fidelity — Similarity using 3 Models x 3 Pairs
==================================================================

Computes semantic similarity using three sentence embedding models
across three text pairs per record.

Models:
  - LaBSE: best cross-lingual (Telugu<->English)
  - mSBERT-mpnet: strong within-language discrimination
  - L3Cube-IndicSBERT: best Indic within-language discrimination

Pairs:
  1. Poem <-> Telugu Meaning
  2. Poem <-> English Meaning
  3. Telugu Meaning <-> English Meaning (anchor pair for pass/fail)

Usage:
    python level3_semantic_similarity.py
    python level3_semantic_similarity.py --models LaBSE mSBERT-mpnet --batch-size 128
    python level3_semantic_similarity.py --dataset ../datasets/your_data.json --limit 100
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from statistics import mean, median

sys.path.insert(0, str(Path(__file__).resolve().parent))

from validation_utils import (
    DEFAULT_DATASET,
    LEVEL3_MODELS,
    RESULTS_DIR,
    batch_encode,
    cosine_similarity_paired,
    format_section_header,
    load_dataset,
    load_indic_sbert_model,
    load_labse_model,
    load_msbert_mpnet_model,
    save_results,
)

# Pass threshold: LaBSE Telugu<->English meaning >= 0.65
COSINE_SIM_THRESHOLD = 0.65

# The 3 similarity pairs to compute
PAIR_NAMES = [
    ("Poem ↔ Telugu Meaning", "poem", "telugu_meaning"),
    ("Poem ↔ English Meaning", "poem", "english_meaning"),
    ("Telugu Meaning ↔ English Meaning", "telugu_meaning", "english_meaning"),
]

# Model loaders keyed by display name
_MODEL_LOADERS = {
    "LaBSE": load_labse_model,
    "mSBERT-mpnet": load_msbert_mpnet_model,
    "L3Cube-IndicSBERT": load_indic_sbert_model,
}


def _compute_pair_stats(sims_list: list[float], data: list[dict],
                        threshold: float) -> dict:
    """Compute aggregate statistics for one similarity pair."""
    total = len(sims_list)
    pass_count = sum(1 for s in sims_list if s >= threshold)

    indexed = sorted(enumerate(sims_list), key=lambda x: x[1])
    bottom_20 = [
        {"index": idx, "sim": round(sim, 4), "poem": data[idx]["poem"][:60]}
        for idx, sim in indexed[:20]
    ]

    return {
        "total": total,
        "pass_count": pass_count,
        "avg_sim": mean(sims_list) if sims_list else 0,
        "median_sim": median(sims_list) if sims_list else 0,
        "min_sim": min(sims_list) if sims_list else 0,
        "max_sim": max(sims_list) if sims_list else 0,
        "sim_list": sims_list,
        "bottom_20": bottom_20,
    }


def run_level3(data: list[dict], batch_size: int = 256,
               output_dir: str = None,
               models_to_run: list[str] = None) -> dict:
    """Compute semantic similarity using selected models across 3 text pairs.

    Per-model intermediate results are saved to output_dir/level3_{model}.json
    so that a crash mid-way only requires re-running remaining models.
    """
    print(format_section_header(
        "LEVEL 3: SEMANTIC FIDELITY (3 Pairs x 3 Models)"))

    if models_to_run is None:
        models_to_run = [name for name, _ in LEVEL3_MODELS]

    total = len(data)

    # Extract text lists once
    texts = {
        "poem": [r["poem"] for r in data],
        "telugu_meaning": [r["telugu_meaning"] for r in data],
        "english_meaning": [r["english_meaning"] for r in data],
    }

    model_results = {}

    for model_name, _model_id in LEVEL3_MODELS:
        if model_name not in models_to_run:
            continue

        # Check for existing per-model result
        if output_dir:
            per_model_path = Path(output_dir) / f"level3_{model_name}.json"
            if per_model_path.exists():
                with open(per_model_path, "r", encoding="utf-8") as f:
                    model_results[model_name] = json.load(f)
                print(f"  {model_name}: loaded from existing result (skipping)")
                continue

        loader = _MODEL_LOADERS[model_name]
        model = loader()

        embs = {}
        for field_name, field_texts in texts.items():
            print(f"  {model_name} encoding {field_name}...")
            embs[field_name] = batch_encode(model, field_texts, batch_size)

        pairs = {}
        for pair_name, field_a, field_b in PAIR_NAMES:
            sims = cosine_similarity_paired(embs[field_a], embs[field_b])
            pairs[pair_name] = _compute_pair_stats(
                sims.tolist(), data, COSINE_SIM_THRESHOLD)

        model_results[model_name] = pairs
        del model, embs

        # Save per-model result immediately
        if output_dir:
            per_model_path = Path(output_dir) / f"level3_{model_name}.json"
            per_model_path.parent.mkdir(parents=True, exist_ok=True)
            with open(per_model_path, "w", encoding="utf-8") as f:
                json.dump(pairs, f, ensure_ascii=False)
            print(f"  [saved] {per_model_path.name}")

    # Per-record results (pass/fail based on LaBSE anchor pair)
    anchor_pair = "Telugu Meaning ↔ English Meaning"
    per_record = []
    pass_count = 0

    if "LaBSE" in model_results:
        anchor_sims = model_results["LaBSE"][anchor_pair]["sim_list"]
        for i in range(total):
            passes = anchor_sims[i] >= COSINE_SIM_THRESHOLD
            if passes:
                pass_count += 1
            rec = {"index": i, "pass": passes}
            for mn in model_results:
                for pair_name, _, _ in PAIR_NAMES:
                    key = f"{mn}_{pair_name}"
                    rec[key] = round(model_results[mn][pair_name]["sim_list"][i], 4)
            per_record.append(rec)
    else:
        pass_count = total
        for i in range(total):
            per_record.append({"index": i, "pass": True})

    stats = {
        "total": total,
        "pass_count": pass_count,
        "models": model_results,
        "per_record": per_record,
    }

    print(f"  Done. Pass (LaBSE {anchor_pair} >= {COSINE_SIM_THRESHOLD}): "
          f"{pass_count}/{total}")
    return stats


def main(args=None):
    all_model_names = [name for name, _ in LEVEL3_MODELS]

    parser = argparse.ArgumentParser(
        description="Level 3: Semantic Fidelity — 3 Models x 3 Pairs",
    )
    parser.add_argument(
        "--dataset", "-d", default=DEFAULT_DATASET,
        help="Path to dataset JSON",
    )
    parser.add_argument(
        "--output-dir", "-o", default=str(RESULTS_DIR),
        help="Directory for output JSON",
    )
    parser.add_argument(
        "--limit", "-n", type=int, default=None,
        help="Process only first N records",
    )
    parser.add_argument(
        "--batch-size", "-b", type=int, default=256,
        help="Batch size for model encoding (default: 256)",
    )
    parser.add_argument(
        "--models", "-m", nargs="+", choices=all_model_names,
        default=all_model_names,
        help="Which models to run (default: all three)",
    )
    args = parser.parse_args(args)

    print(f"Loading dataset: {args.dataset}")
    data = load_dataset(args.dataset)
    if args.limit:
        data = data[:args.limit]
        print(f"Limited to first {args.limit} records.")
    print(f"Records to validate: {len(data):,}")
    print(f"Models: {', '.join(args.models)}")

    results = run_level3(
        data,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        models_to_run=args.models,
    )

    meta = {
        "level": "level3",
        "dataset": str(Path(args.dataset).resolve()),
        "total_records": len(data),
        "limit": args.limit,
        "models": args.models,
        "timestamp": datetime.now().isoformat(),
    }
    output_path = str(Path(args.output_dir) / "level3.json")
    save_results(results, output_path, meta)


if __name__ == "__main__":
    main()
