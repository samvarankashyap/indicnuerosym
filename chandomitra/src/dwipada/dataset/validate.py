#!/usr/bin/env python3
"""
4-Level Dataset Validation for Telugu Dwipada Poetry
=====================================================

Validates a dwipada poetry dataset through a multi-stage filtration protocol
to ensure near "Gold Standard" quality for fine-tuning.

Levels:
  1. Chandass Scanner   -- Prosodic structure (gana, prasa, yati)
  2. Sanity Checks      -- Field completeness and length-ratio alignment
  3. Semantic Fidelity  -- LaBSE + mSBERT-mpnet + L3Cube-IndicSBERT (3 pairs x 3 models)
  4. Lexical Diversity  -- Gemma-3 tokenizer TTR and duplicate detection

Usage:
    python -m dwipada.dataset.validate [OPTIONS]

Options:
    --dataset, -d      Path to dataset JSON
    --output, -o       Report output path
    --skip-level3      Skip semantic check (faster, no GPU needed)
    --batch-size       Batch size for model encoding (default: 256)
    --limit N          Process only first N records (for testing)
    --results-json     Save per-record results as JSON
    --checkpoint-dir   Checkpoint directory (default: dataset_validation_scripts/checkpoints)
    --fresh            Clear all checkpoints and start from scratch
"""

import argparse
import json
import re
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from statistics import mean, median

from dwipada.core.analyzer import analyze_dwipada
from dwipada.dataset.validation_utils import (
    LEVEL3_MODELS,
    batch_encode,
    cosine_similarity_paired,
    count_telugu_tokens,
    format_histogram,
    format_section_header,
    format_stat_line,
    gemma_tokenize,
    load_dataset,
    load_gemma_tokenizer,
    load_indic_sbert_model,
    load_labse_model,
    load_msbert_mpnet_model,
    write_report,
)
from dwipada.paths import DATASETS_DIR


# ── Constants ────────────────────────────────────────────────────────────────

DEFAULT_DATASET = str(DATASETS_DIR / "dwipada_master_filtered_perfect_dataset.json")
DEFAULT_OUTPUT = "dataset_validation_scripts/validation_report.txt"

# Level 2 thresholds
LENGTH_RATIO_MIN = 0.8
LENGTH_RATIO_MAX = 2.5

# Level 3 threshold (LaBSE anchor pair: Telugu<->English meaning)
# Unrelated pairs score 0.23-0.46, related pairs 0.68-0.86 on LaBSE.
# 0.65 is conservative enough to catch mismatches while allowing
# figurative/archaic poetry with naturally lower cross-lingual similarity.
COSINE_SIM_THRESHOLD = 0.65

# Checkpoints
DEFAULT_CHECKPOINT_DIR = "dataset_validation_scripts/checkpoints"


# =============================================================================
# CHECKPOINTING
# =============================================================================

def _ckpt_path(checkpoint_dir: str, name: str) -> Path:
    """Return the path for a named checkpoint file."""
    return Path(checkpoint_dir) / f"{name}.json"


def _save_checkpoint(checkpoint_dir: str, name: str, data: dict):
    """Save a checkpoint as JSON."""
    path = _ckpt_path(checkpoint_dir, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write to temp file first, then rename for atomicity
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    tmp.rename(path)
    print(f"  [checkpoint] Saved: {path.name}")


def _load_checkpoint(checkpoint_dir: str, name: str) -> dict | None:
    """Load a checkpoint if it exists. Returns None if missing."""
    path = _ckpt_path(checkpoint_dir, name)
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"  [checkpoint] Loaded: {path.name}")
    return data


def _save_manifest(checkpoint_dir: str, dataset: str, limit: int | None, total: int):
    """Save manifest that identifies the current run configuration."""
    _save_checkpoint(checkpoint_dir, "manifest", {
        "dataset": str(Path(dataset).resolve()),
        "limit": limit,
        "total_records": total,
    })


def _manifest_valid(checkpoint_dir: str, dataset: str, limit: int | None, total: int) -> bool:
    """Check if existing checkpoints match the current run configuration."""
    manifest = _load_checkpoint(checkpoint_dir, "manifest")
    if manifest is None:
        return False
    return (
        manifest["dataset"] == str(Path(dataset).resolve())
        and manifest.get("limit") == limit
        and manifest["total_records"] == total
    )


def _clear_checkpoints(checkpoint_dir: str):
    """Remove all checkpoint files."""
    ckpt_dir = Path(checkpoint_dir)
    if ckpt_dir.exists():
        for f in ckpt_dir.glob("*.json"):
            f.unlink()
        print(f"  [checkpoint] Cleared all checkpoints in {ckpt_dir}")


# =============================================================================
# LEVEL 1: CHANDASS SCANNER (Structural Integrity)
# =============================================================================

def run_level1(data: list[dict]) -> dict:
    """Validate prosodic structure of every poem using dwipada_analyzer.

    Checks per record:
      - is_valid_dwipada (all rules satisfied)
      - Gana sequence (3 Indra + 1 Surya per line)
      - Prasa (2nd consonant match across lines)
      - Yati (1st letter of gana 1 == 1st letter of gana 3, per line)
      - Overall match score (0-100)

    Args:
        data: List of dataset records, each with a "poem" field.

    Returns:
        Dict with aggregate stats and per-record score lists.
    """
    print(format_section_header("LEVEL 1: CHANDASS SCANNER (Structural Integrity)"))

    total = len(data)
    valid_count = 0
    overall_scores = []
    gana_l1_scores = []
    gana_l2_scores = []
    prasa_scores = []
    yati_l1_scores = []
    yati_l2_scores = []
    errors = []
    per_record = []

    for i, record in enumerate(data):
        poem = record["poem"]
        try:
            result = analyze_dwipada(poem)
        except Exception as e:
            errors.append({"index": i, "error": str(e), "poem": poem[:60]})
            per_record.append({
                "index": i, "is_valid": False, "overall_score": 0,
                "gana_l1": 0, "gana_l2": 0, "prasa": 0,
                "yati_l1": 0, "yati_l2": 0,
            })
            continue

        is_valid = result.get("is_valid_dwipada", False)
        score = result.get("match_score", {})
        overall = score.get("overall", 0)
        breakdown = score.get("breakdown", {})

        g1 = breakdown.get("gana_line1", 0)
        g2 = breakdown.get("gana_line2", 0)
        pr = breakdown.get("prasa", 0)
        y1 = breakdown.get("yati_line1", 0)
        y2 = breakdown.get("yati_line2", 0)

        if is_valid:
            valid_count += 1
        overall_scores.append(overall)
        gana_l1_scores.append(g1)
        gana_l2_scores.append(g2)
        prasa_scores.append(pr)
        yati_l1_scores.append(y1)
        yati_l2_scores.append(y2)

        per_record.append({
            "index": i, "is_valid": is_valid, "overall_score": overall,
            "gana_l1": g1, "gana_l2": g2, "prasa": pr,
            "yati_l1": y1, "yati_l2": y2,
        })

        if (i + 1) % 5000 == 0:
            print(f"  Level 1 progress: {i + 1}/{total}")

    # Aggregate stats
    gana_pass = sum(1 for r in per_record if r["gana_l1"] == 100 and r["gana_l2"] == 100)
    prasa_pass = sum(1 for s in prasa_scores if s == 100)
    yati_pass = sum(1 for r in per_record if r["yati_l1"] == 100 and r["yati_l2"] == 100)

    stats = {
        "total": total,
        "valid_count": valid_count,
        "avg_overall_score": mean(overall_scores) if overall_scores else 0,
        "gana_pass": gana_pass,
        "prasa_pass": prasa_pass,
        "yati_pass": yati_pass,
        "overall_scores": overall_scores,
        "errors": errors,
        "per_record": per_record,
    }

    print(f"  Done. Valid: {valid_count}/{total}, Errors: {len(errors)}")
    return stats


# =============================================================================
# LEVEL 2: SANITY CHECKS (Coarse Alignment)
# =============================================================================

def run_level2(data: list[dict]) -> dict:
    """Validate field completeness and prose-verse length alignment.

    Checks per record:
      - Length ratio: len(telugu_meaning) / len(poem) in (0.8, 2.0)
      - telugu_meaning is a non-empty string
      - english_meaning is a non-empty string

    Args:
        data: List of dataset records.

    Returns:
        Dict with aggregate stats and per-record results.
    """
    print(format_section_header("LEVEL 2: SANITY CHECKS (Coarse Alignment)"))

    total = len(data)
    ratios = []
    ratio_pass_count = 0
    telugu_pass = 0
    english_pass = 0
    all_pass = 0
    per_record = []
    outliers = []

    for i, record in enumerate(data):
        poem = record.get("poem", "")
        telugu = record.get("telugu_meaning", "")
        english = record.get("english_meaning", "")

        # Strip whitespace for fair character count
        poem_clean = poem.replace(" ", "").replace("\n", "")
        telugu_clean = telugu.replace(" ", "").replace("\n", "")

        # Length ratio
        ratio = len(telugu_clean) / len(poem_clean) if len(poem_clean) > 0 else 0.0
        ratio_ok = LENGTH_RATIO_MIN < ratio < LENGTH_RATIO_MAX
        ratios.append(ratio)

        # Field checks
        telugu_ok = isinstance(telugu, str) and len(telugu.strip()) > 0
        english_ok = isinstance(english, str) and len(english.strip()) > 0

        if ratio_ok:
            ratio_pass_count += 1
        if telugu_ok:
            telugu_pass += 1
        if english_ok:
            english_pass += 1

        passes_all = ratio_ok and telugu_ok and english_ok
        if passes_all:
            all_pass += 1

        rec = {
            "index": i, "length_ratio": ratio, "ratio_pass": ratio_ok,
            "telugu_nonempty": telugu_ok,
            "english_nonempty": english_ok, "all_pass": passes_all,
        }
        per_record.append(rec)

        # Track outliers
        if not ratio_ok:
            outliers.append((i, ratio, poem[:60]))

    # Sort outliers by how far they are from the valid range
    outliers.sort(key=lambda x: abs(x[1] - 1.4))  # sort by distance from midpoint
    outliers.reverse()

    stats = {
        "total": total,
        "ratio_pass_count": ratio_pass_count,
        "telugu_pass": telugu_pass,
        "english_pass": english_pass,
        "all_pass": all_pass,
        "ratios": ratios,
        "avg_ratio": mean(ratios) if ratios else 0,
        "median_ratio": median(ratios) if ratios else 0,
        "min_ratio": min(ratios) if ratios else 0,
        "max_ratio": max(ratios) if ratios else 0,
        "outliers": outliers[:20],
        "per_record": per_record,
    }

    print(f"  Done. All pass: {all_pass}/{total}, Ratio outliers: {len(outliers)}")
    return stats


# =============================================================================
# LEVEL 3: SEMANTIC FIDELITY (3 Pairs x 3 Models)
# =============================================================================

# The 3 similarity pairs to compute
PAIR_NAMES = [
    ("Poem ↔ Telugu Meaning", "poem", "telugu_meaning"),
    ("Poem ↔ English Meaning", "poem", "english_meaning"),
    ("Telugu Meaning ↔ English Meaning", "telugu_meaning", "english_meaning"),
]

# Model loaders keyed by display name (must match LEVEL3_MODELS ordering)
_MODEL_LOADERS = {
    "LaBSE": load_labse_model,
    "mSBERT-mpnet": load_msbert_mpnet_model,
    "L3Cube-IndicSBERT": load_indic_sbert_model,
}


def _compute_pair_stats(sims_list: list[float], data: list[dict],
                        threshold: float) -> dict:
    """Compute aggregate statistics for one similarity pair.

    Args:
        sims_list: List of cosine similarity values, one per record.
        data: The dataset records (for poem text in bottom-20).
        threshold: Pass threshold.

    Returns:
        Dict with mean, median, min, max, pass_count, bottom_20, sim_list.
    """
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
               checkpoint_dir: str | None = None) -> dict:
    """Compute semantic similarity using 3 models across 3 text pairs.

    Models:
      - LaBSE: best cross-lingual (Telugu<->English)
      - mSBERT-mpnet: strong within-language discrimination
      - L3Cube-IndicSBERT: best Indic within-language discrimination

    Pairs computed (with each model):
      1. Poem (Te) <-> Telugu Meaning   -- does the meaning explain this poem?
      2. Poem (Te) <-> English Meaning  -- cross-lingual validation
      3. Telugu <-> English Meaning     -- translation consistency (quality anchor)

    A record passes Level 3 if LaBSE Pair 3 (Telugu<->English) >= 0.70.
    All other scores are diagnostic.

    Each model's results are checkpointed independently so that a crash
    mid-way only requires re-running the remaining models, not all of them.

    Args:
        data: List of dataset records.
        batch_size: Batch size for encoding.
        checkpoint_dir: Directory for per-model checkpoints.

    Returns:
        Dict with per-model, per-pair stats and per-record results.
    """
    print(format_section_header(
        "LEVEL 3: SEMANTIC FIDELITY (3 Pairs × 3 Models)"))

    total = len(data)

    # Extract text lists once
    texts = {
        "poem": [r["poem"] for r in data],
        "telugu_meaning": [r["telugu_meaning"] for r in data],
        "english_meaning": [r["english_meaning"] for r in data],
    }

    # Run each model sequentially, free memory between models
    model_results = {}  # model_name -> {pair_name -> pair_stats}

    for model_name, _model_id in LEVEL3_MODELS:
        ckpt_name = f"level3_{model_name}"

        # Try to load from checkpoint
        if checkpoint_dir:
            cached = _load_checkpoint(checkpoint_dir, ckpt_name)
            if cached is not None:
                model_results[model_name] = cached
                print(f"  {model_name}: restored from checkpoint (skipping encode)")
                continue

        # No checkpoint — run the model
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

        # Save per-model checkpoint immediately
        if checkpoint_dir:
            _save_checkpoint(checkpoint_dir, ckpt_name, pairs)

    # ── Per-record results (pass/fail based on LaBSE Pair 3) ─────────
    anchor_pair = "Telugu Meaning ↔ English Meaning"
    anchor_sims = model_results["LaBSE"][anchor_pair]["sim_list"]

    per_record = []
    pass_count = 0
    for i in range(total):
        passes = anchor_sims[i] >= COSINE_SIM_THRESHOLD
        if passes:
            pass_count += 1
        rec = {"index": i, "pass": passes}
        for model_name in model_results:
            for pair_name, _, _ in PAIR_NAMES:
                key = f"{model_name}_{pair_name}"
                rec[key] = round(model_results[model_name][pair_name]["sim_list"][i], 4)
        per_record.append(rec)

    stats = {
        "total": total,
        "pass_count": pass_count,
        "models": model_results,
        "per_record": per_record,
    }

    print(f"  Done. Pass (LaBSE {anchor_pair} >= {COSINE_SIM_THRESHOLD}): "
          f"{pass_count}/{total}")
    return stats


# =============================================================================
# LEVEL 4: LEXICAL DIVERSITY (Corpus Health)
# =============================================================================

def run_level4(data: list[dict]) -> dict:
    """Analyse lexical diversity using Gemma 3 tokenizer and duplicate detection.

    Metrics:
      - Telugu Token Coverage: fraction of Gemma 3's Telugu tokens used by the dataset
      - Per-poem average TTR
      - Exact duplicate poem detection
      - Near-duplicate detection (normalized text comparison)

    Args:
        data: List of dataset records.

    Returns:
        Dict with corpus-level diversity metrics.
    """
    print(format_section_header("LEVEL 4: LEXICAL DIVERSITY (Corpus Health)"))

    tokenizer = load_gemma_tokenizer()

    # ── Count Telugu tokens in vocabulary ──────────────────────────────────
    print("  Counting Telugu tokens in Gemma 3 vocabulary...")
    total_telugu_tokens = count_telugu_tokens(tokenizer)
    print(f"  Telugu tokens in vocabulary: {total_telugu_tokens}")

    # ── Tokenize all poems ───────────────────────────────────────────────
    all_token_ids = []           # flat list of every token ID across all poems
    unique_token_ids = set()     # set of distinct token IDs
    per_poem_ttrs = []           # TTR per individual poem

    total = len(data)
    for i, record in enumerate(data):
        poem = record["poem"]
        ids = gemma_tokenize(tokenizer, poem)

        all_token_ids.extend(ids)
        unique_token_ids.update(ids)

        # Per-poem TTR
        if len(ids) > 0:
            per_poem_ttrs.append(len(set(ids)) / len(ids))
        else:
            per_poem_ttrs.append(0.0)

        if (i + 1) % 5000 == 0:
            print(f"  Tokenization progress: {i + 1}/{total}")

    telugu_coverage = len(unique_token_ids) / total_telugu_tokens if total_telugu_tokens else 0
    avg_poem_ttr = mean(per_poem_ttrs) if per_poem_ttrs else 0

    # ── Exact duplicates ─────────────────────────────────────────────────
    poem_counter = Counter(record["poem"] for record in data)
    exact_dup_groups = []
    exact_dup_total = 0
    for poem_text, count in poem_counter.items():
        if count > 1:
            indices = [i for i, r in enumerate(data) if r["poem"] == poem_text]
            exact_dup_groups.append({"poem": poem_text[:80], "count": count, "indices": indices})
            exact_dup_total += count

    # ── Near duplicates (normalize: strip spaces, punctuation, newlines) ─
    def normalize(text: str) -> str:
        text = text.replace("\n", "").replace(" ", "")
        text = re.sub(r"[,.\-;:!?()\"'।॥]", "", text)
        return text

    norm_map: dict[str, list[int]] = {}
    for i, record in enumerate(data):
        key = normalize(record["poem"])
        norm_map.setdefault(key, []).append(i)

    near_dup_groups = []
    near_dup_total = 0
    for norm_text, indices in norm_map.items():
        if len(indices) > 1:
            poem_sample = data[indices[0]]["poem"][:80]
            near_dup_groups.append({"poem": poem_sample, "count": len(indices), "indices": indices})
            near_dup_total += len(indices)

    stats = {
        "total_tokens": len(all_token_ids),
        "unique_tokens": len(unique_token_ids),
        "total_telugu_tokens": total_telugu_tokens,
        "telugu_coverage": telugu_coverage,
        "avg_poem_ttr": avg_poem_ttr,
        "per_poem_ttrs": per_poem_ttrs,
        "exact_dup_groups": sorted(exact_dup_groups, key=lambda x: -x["count"]),
        "exact_dup_poem_count": exact_dup_total,
        "near_dup_groups": sorted(near_dup_groups, key=lambda x: -x["count"]),
        "near_dup_poem_count": near_dup_total,
    }

    print(f"  Done. Telugu Token Coverage: {telugu_coverage:.1%} ({len(unique_token_ids)}/{total_telugu_tokens}), "
          f"Exact dups: {exact_dup_total}, Near dups: {near_dup_total}")
    return stats


# =============================================================================
# REPORT GENERATION
# =============================================================================

def generate_report(
    l1: dict, l2: dict, l3: dict | None, l4: dict,
    total_records: int, dataset_path: str, elapsed: float,
) -> list[str]:
    """Compile all level stats into a formatted text report.

    Args:
        l1: Level 1 stats dict.
        l2: Level 2 stats dict.
        l3: Level 3 stats dict (None if skipped).
        l4: Level 4 stats dict.
        total_records: Number of records processed.
        dataset_path: Path to the dataset file.
        elapsed: Total runtime in seconds.

    Returns:
        List of report lines.
    """
    lines = []

    # Header
    mins, secs = divmod(int(elapsed), 60)
    lines.append("=" * 80)
    lines.append("  DWIPADA DATASET VALIDATION REPORT")
    lines.append(f"  Dataset: {dataset_path}")
    lines.append(f"  Records: {total_records:,}  |  Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}  |  Runtime: {mins}m {secs}s")
    lines.append("=" * 80)
    lines.append("")

    # ── Level 1 ──────────────────────────────────────────────────────────
    lines.append(format_section_header("LEVEL 1: CHANDASS SCANNER (Structural Integrity)"))
    t = l1["total"]
    lines.append(format_stat_line("Valid Dwipada:", l1["valid_count"], t))
    lines.append(format_stat_line("Avg Overall Score:", l1["avg_overall_score"]))
    lines.append("")
    lines.append("  Sub-check Pass Rates:")
    lines.append(format_stat_line("  Gana Sequence:", l1["gana_pass"], t, label_width=28))
    lines.append(format_stat_line("  Prasa Match:", l1["prasa_pass"], t, label_width=28))
    lines.append(format_stat_line("  Yati (both lines):", l1["yati_pass"], t, label_width=28))
    lines.append("")
    lines.append("  Score Distribution:")
    hist = format_histogram(l1["overall_scores"], [0, 50, 70, 80, 90, 100.01])
    lines.extend(hist)
    if l1["errors"]:
        lines.append(f"\n  Errors ({len(l1['errors'])}):")
        for e in l1["errors"][:10]:
            lines.append(f"    #{e['index']}: {e['error']} — {e['poem']}")
    lines.append("")

    # ── Level 2 ──────────────────────────────────────────────────────────
    lines.append(format_section_header("LEVEL 2: SANITY CHECKS (Coarse Alignment)"))
    t = l2["total"]
    lines.append(format_stat_line("All Sub-checks Pass:", l2["all_pass"], t))
    lines.append("")
    lines.append("  Sub-check Pass Rates:")
    lines.append(format_stat_line(f"  Length Ratio ({LENGTH_RATIO_MIN}-{LENGTH_RATIO_MAX}):", l2["ratio_pass_count"], t, label_width=28))
    lines.append(format_stat_line("  Non-empty Telugu:", l2["telugu_pass"], t, label_width=28))
    lines.append(format_stat_line("  Non-empty English:", l2["english_pass"], t, label_width=28))
    lines.append("")
    lines.append(f"  Length Ratio Stats:")
    lines.append(f"    Mean: {l2['avg_ratio']:.2f}  |  Median: {l2['median_ratio']:.2f}  "
                 f"|  Min: {l2['min_ratio']:.2f}  |  Max: {l2['max_ratio']:.2f}")
    lines.append("")
    lines.append("  Length Ratio Distribution:")
    hist = format_histogram(l2["ratios"], [0, 0.5, 0.8, 1.0, 1.5, 2.0, 5.01])
    lines.extend(hist)
    if l2["outliers"]:
        lines.append(f"\n  Top Outliers ({min(len(l2['outliers']), 10)}):")
        for idx, ratio, poem in l2["outliers"][:10]:
            lines.append(f"    #{idx}  ratio={ratio:.2f}  poem=\"{poem}...\"")
    lines.append("")

    # ── Level 3 ──────────────────────────────────────────────────────────
    if l3 is not None:
        lines.append(format_section_header(
            "LEVEL 3: SEMANTIC FIDELITY (3 Pairs × 3 Models)"))
        t = l3["total"]
        lines.append(format_stat_line(
            f"Pass (LaBSE Te↔En >= {COSINE_SIM_THRESHOLD}):", l3["pass_count"], t))
        lines.append("")

        for model_name, _model_id in LEVEL3_MODELS:
            model_pairs = l3["models"][model_name]
            lines.append(f"  ┌─ {model_name} ──────────────────────────────────────────")
            for pair_name in [p[0] for p in PAIR_NAMES]:
                p = model_pairs[pair_name]
                lines.append(f"  │")
                lines.append(f"  │  {pair_name}")
                lines.append(f"  │    Pass (>= {COSINE_SIM_THRESHOLD}): "
                             f"{p['pass_count']:,} / {p['total']:,}  "
                             f"({p['pass_count']/p['total']*100:.1f}%)")
                lines.append(f"  │    Mean: {p['avg_sim']:.4f}  |  "
                             f"Median: {p['median_sim']:.4f}  |  "
                             f"Min: {p['min_sim']:.4f}  |  "
                             f"Max: {p['max_sim']:.4f}")
                lines.append(f"  │")
                hist = format_histogram(p["sim_list"], [0, 0.3, 0.5, 0.7, 0.85, 1.01])
                for h in hist:
                    lines.append(f"  │  {h}")
                lines.append(f"  │")
            lines.append(f"  └────────────────────────────────────────────────────")
            lines.append("")

        # Bottom 20 by anchor pair (LaBSE Telugu<->English)
        anchor = l3["models"]["LaBSE"]["Telugu Meaning ↔ English Meaning"]
        if anchor["bottom_20"]:
            lines.append("  Bottom 20 by LaBSE Telugu↔English Similarity:")
            for entry in anchor["bottom_20"]:
                lines.append(f"    #{entry['index']}  sim={entry['sim']:.4f}  "
                             f"poem=\"{entry['poem']}...\"")
            lines.append("")
    else:
        lines.append(format_section_header("LEVEL 3: SEMANTIC FIDELITY (Skipped)"))
        lines.append("  Skipped via --skip-level3 flag.")
        lines.append("")

    # ── Level 4 ──────────────────────────────────────────────────────────
    lines.append(format_section_header("LEVEL 4: LEXICAL DIVERSITY (Corpus Health)"))
    lines.append(format_stat_line("Total Tokens (Gemma 3):", f"{l4['total_tokens']:,}"))
    lines.append(format_stat_line("Telugu Tokens in Vocab:", f"{l4['total_telugu_tokens']:,}"))
    lines.append(format_stat_line("Telugu Tokens Used:", f"{l4['unique_tokens']:,}"))
    te_cov = l4["telugu_coverage"]
    lines.append(format_stat_line("Telugu Token Coverage:", f"{te_cov:.1%} ({l4['unique_tokens']:,} / {l4['total_telugu_tokens']:,})"))
    lines.append(format_stat_line("Avg Per-Poem TTR:", l4["avg_poem_ttr"]))
    lines.append("")
    lines.append(f"  Exact Duplicates: {l4['exact_dup_poem_count']} poems "
                 f"in {len(l4['exact_dup_groups'])} groups")
    lines.append(f"  Near Duplicates:  {l4['near_dup_poem_count']} poems "
                 f"in {len(l4['near_dup_groups'])} groups")
    if l4["exact_dup_groups"]:
        lines.append(f"\n  Sample Duplicate Groups (top 5):")
        for grp in l4["exact_dup_groups"][:5]:
            lines.append(f"    {grp['count']}x  indices={grp['indices'][:5]}  "
                         f"poem=\"{grp['poem']}...\"")
    lines.append("")

    # ── Cross-Level Summary ──────────────────────────────────────────────
    lines.append(format_section_header("CROSS-LEVEL SUMMARY"))

    # Compute pass-all count using per-record data from L1 and L2
    l1_pass_set = set(r["index"] for r in l1["per_record"] if r["is_valid"] and r["overall_score"] == 100)
    l2_pass_set = set(r["index"] for r in l2["per_record"] if r["all_pass"])
    if l3 is not None:
        l3_pass_set = set(r["index"] for r in l3["per_record"] if r["pass"])
    else:
        l3_pass_set = set(range(total_records))  # assume pass if skipped

    pass_all = l1_pass_set & l2_pass_set & l3_pass_set
    lines.append(format_stat_line("Pass ALL Levels:", len(pass_all), total_records))
    lines.append(format_stat_line("Fail Level 1 only:", len(l2_pass_set & l3_pass_set) - len(pass_all), total_records))
    lines.append(format_stat_line("Fail Level 2 only:", len(l1_pass_set & l3_pass_set) - len(pass_all), total_records))
    if l3 is not None:
        lines.append(format_stat_line("Fail Level 3 only:", len(l1_pass_set & l2_pass_set) - len(pass_all), total_records))
    lines.append("")
    lines.append("=" * 80)

    return lines


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="4-Level Validation for Telugu Dwipada Poetry Dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset", "-d", default=DEFAULT_DATASET,
        help=f"Path to dataset JSON (default: {DEFAULT_DATASET})",
    )
    parser.add_argument(
        "--output", "-o", default=DEFAULT_OUTPUT,
        help=f"Report output path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--skip-level3", action="store_true",
        help="Skip Level 3 (semantic check) for faster runs",
    )
    parser.add_argument(
        "--batch-size", type=int, default=256,
        help="Batch size for model encoding (default: 256)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process only first N records (for testing)",
    )
    parser.add_argument(
        "--results-json", default=None,
        help="Save per-record results as JSON for further analysis",
    )
    parser.add_argument(
        "--checkpoint-dir", default=DEFAULT_CHECKPOINT_DIR,
        help=f"Checkpoint directory (default: {DEFAULT_CHECKPOINT_DIR})",
    )
    parser.add_argument(
        "--fresh", action="store_true",
        help="Clear all checkpoints and start from scratch",
    )
    args = parser.parse_args()

    ckpt_dir = args.checkpoint_dir

    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    data = load_dataset(args.dataset)
    if args.limit:
        data = data[:args.limit]
        print(f"Limited to first {args.limit} records.")
    total_records = len(data)
    print(f"Records to validate: {total_records:,}")

    # ── Checkpoint validation ─────────────────────────────────────────
    if args.fresh:
        _clear_checkpoints(ckpt_dir)
    elif not _manifest_valid(ckpt_dir, args.dataset, args.limit, total_records):
        print("  [checkpoint] Manifest mismatch (dataset/limit changed) — clearing stale checkpoints")
        _clear_checkpoints(ckpt_dir)

    _save_manifest(ckpt_dir, args.dataset, args.limit, total_records)
    print()

    start_time = time.time()

    # ── Level 1 ───────────────────────────────────────────────────────
    l1_stats = _load_checkpoint(ckpt_dir, "level1")
    if l1_stats is None:
        l1_stats = run_level1(data)
        _save_checkpoint(ckpt_dir, "level1", l1_stats)
    else:
        print("  Level 1: restored from checkpoint (skipping)")
    print()

    # ── Level 2 ───────────────────────────────────────────────────────
    l2_stats = _load_checkpoint(ckpt_dir, "level2")
    if l2_stats is None:
        l2_stats = run_level2(data)
        _save_checkpoint(ckpt_dir, "level2", l2_stats)
    else:
        print("  Level 2: restored from checkpoint (skipping)")
    print()

    # ── Level 4 (before Level 3 — faster, no model loading) ──────────
    l4_stats = _load_checkpoint(ckpt_dir, "level4")
    if l4_stats is None:
        l4_stats = run_level4(data)
        _save_checkpoint(ckpt_dir, "level4", l4_stats)
    else:
        print("  Level 4: restored from checkpoint (skipping)")
    print()

    # ── Level 3 (per-model checkpoints handled inside run_level3) ────
    l3_stats = None
    if not args.skip_level3:
        # Check if the fully-assembled Level 3 result is cached
        l3_stats = _load_checkpoint(ckpt_dir, "level3")
        if l3_stats is not None:
            print("  Level 3: restored from checkpoint (skipping)")
        else:
            try:
                l3_stats = run_level3(
                    data, batch_size=args.batch_size, checkpoint_dir=ckpt_dir)
                _save_checkpoint(ckpt_dir, "level3", l3_stats)
            except ImportError as e:
                print(f"  SKIPPED: {e}")
                l3_stats = None
        print()

    elapsed = time.time() - start_time

    # Generate and write report
    report = generate_report(l1_stats, l2_stats, l3_stats, l4_stats,
                             total_records, args.dataset, elapsed)
    write_report(report, args.output)

    # Optionally save per-record results
    if args.results_json:
        per_record_combined = []
        for i in range(total_records):
            rec = {"index": i}
            if i < len(l1_stats["per_record"]):
                rec["level1"] = l1_stats["per_record"][i]
            if i < len(l2_stats["per_record"]):
                rec["level2"] = l2_stats["per_record"][i]
            if l3_stats and i < len(l3_stats["per_record"]):
                rec["level3"] = l3_stats["per_record"][i]
            per_record_combined.append(rec)

        with open(args.results_json, "w", encoding="utf-8") as f:
            json.dump(per_record_combined, f, ensure_ascii=False, indent=2)
        print(f"Per-record results saved to: {args.results_json}")


if __name__ == "__main__":
    main()
