#!/usr/bin/env python3
"""
Ragale Benchmark — 3 models x 3 strategies
===========================================
Runs all 9 combinations of model and decoding strategy,
generates 20+ poems per combination (4 topics x 5 seeds),
and outputs a summary table with timing and validity metrics.

Models:
    1. gemma3-1b-base  — google/gemma-3-1b-it (no fine-tuning)
    2. gemma3-1b-lora  — google/gemma-3-1b-it + ragale LoRA adapter
    3. gemma4-e2b-base — google/gemma-4-E2B-it (4-bit quantized)

Strategies:
    1. masking_only      — pure logit masking
    2. masking_backtrack  — masking + checkpoint backtracking
    3. hybrid            — masking + NFA rejection sampling on top-100

Usage:
    cd ragale_pipeline
    python ragale_inference_scripts/benchmark.py
    python ragale_inference_scripts/benchmark.py --seeds 10
    python ragale_inference_scripts/benchmark.py --models gemma3-1b-base gemma3-1b-lora
    python ragale_inference_scripts/benchmark.py --strategies masking_only hybrid
"""

import argparse
import gc
import json
import os
import sys
import time

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from shared_utils import (
    load_model,
    load_model_lora,
    load_model_quantized,
    precompute_token_data,
    print_result,
    BENCHMARK_TOPICS,
)

from generate_masking_only import generate_poem as gen_masking_only
from generate_masking_backtrack import generate_poem as gen_masking_backtrack
from generate_hybrid import generate_poem as gen_hybrid


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_CONFIGS = {
    "gemma3-1b-base": {
        "loader": "base",
        "model_name": "google/gemma-3-1b-it",
        "desc": "Gemma 3 1B IT (base)",
    },
    "gemma3-1b-lora": {
        "loader": "lora",
        "model_name": "google/gemma-3-1b-it",
        "adapter_path": os.path.join(
            os.path.dirname(SCRIPT_DIR), "ragale_checkpoints", "checkpoint-336"
        ),
        "desc": "Gemma 3 1B IT + Ragale LoRA (best checkpoint)",
    },
    "gemma4-e2b-base": {
        "loader": "quantized",
        "model_name": "google/gemma-4-E2B-it",
        "desc": "Gemma 4 E2B IT (4-bit NF4)",
    },
}

STRATEGY_FUNCS = {
    "masking_only": gen_masking_only,
    "masking_backtrack": gen_masking_backtrack,
    "hybrid": gen_hybrid,
}


# ---------------------------------------------------------------------------
# Model lifecycle
# ---------------------------------------------------------------------------

def load_model_by_config(cfg):
    """Load model+tokenizer based on config dict."""
    if cfg["loader"] == "base":
        return load_model(cfg["model_name"])
    elif cfg["loader"] == "lora":
        return load_model_lora(cfg["model_name"], cfg["adapter_path"])
    elif cfg["loader"] == "quantized":
        return load_model_quantized(cfg["model_name"])
    else:
        raise ValueError(f"Unknown loader: {cfg['loader']}")


def unload_model():
    """Free GPU memory between model runs.

    Must be called AFTER the caller has deleted its own references
    to model and tokenizer.
    """
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()
    print("  Model unloaded, GPU memory freed.\n")


# ---------------------------------------------------------------------------
# Run one (model, strategy) combination
# ---------------------------------------------------------------------------

def run_combination(model, tokenizer, strategy_name, strategy_fn,
                    model_name, topics, num_seeds,
                    kannada_ids, kannada_texts, static_mask, newline_token_id):
    """Run a single model+strategy combination across all topics and seeds.

    Returns list of result dicts.
    """
    results = []
    combo_start = time.time()

    for topic in topics:
        for i in range(num_seeds):
            seed = 42 + i * 7
            r = strategy_fn(
                model, tokenizer, topic,
                kannada_ids, kannada_texts, static_mask, newline_token_id,
                seed=seed,
            )
            # Tag with model name for the summary
            r["model"] = model_name
            results.append(r)
            status = "V" if r["all_valid"] else "X"
            print(f"    {status} {topic:10s} seed={seed:3d}  "
                  f"{r['elapsed']:.1f}s  {r['tokens_generated']} tok", flush=True)

    combo_elapsed = time.time() - combo_start
    print(f"  [{model_name} / {strategy_name}] "
          f"{len(results)} poems in {combo_elapsed:.1f}s")

    return results


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def compute_summary_row(model_name, strategy_name, results):
    """Compute one summary row from a list of results."""
    total = len(results)
    valid = sum(1 for r in results if r["all_valid"])
    avg_time = sum(r["elapsed"] for r in results) / total if total else 0
    avg_tok = sum(r["tokens_generated"] for r in results) / total if total else 0
    avg_masks = sum(r.get("mask_computations", 0) for r in results) / total if total else 0
    avg_bt = sum(r.get("backtracks", 0) for r in results) / total if total else 0
    pct = valid / total * 100 if total else 0

    return {
        "model": model_name,
        "strategy": strategy_name,
        "valid": valid,
        "total": total,
        "valid_pct": pct,
        "avg_time": avg_time,
        "avg_tokens": avg_tok,
        "avg_masks": avg_masks,
        "avg_backtracks": avg_bt,
    }


def format_summary_table(rows):
    """Format summary rows as a markdown table string."""
    header = (
        "| Model              | Strategy          | Valid | Total | Valid% "
        "| Avg Time | Avg Tokens | Avg Masks | Avg Backtracks |"
    )
    sep = (
        "|:-------------------|:------------------|------:|------:|-------:"
        "|---------:|-----------:|----------:|---------------:|"
    )
    lines = [header, sep]

    for r in rows:
        line = (
            f"| {r['model']:18s} "
            f"| {r['strategy']:17s} "
            f"| {r['valid']:5d} "
            f"| {r['total']:5d} "
            f"| {r['valid_pct']:5.1f}% "
            f"| {r['avg_time']:7.1f}s "
            f"| {r['avg_tokens']:10.0f} "
            f"| {r['avg_masks']:9.0f} "
            f"| {r['avg_backtracks']:14.1f} |"
        )
        lines.append(line)

    return "\n".join(lines)


def print_and_save_summary(rows, all_results):
    """Print summary table and save results to disk."""
    table = format_summary_table(rows)

    print(f"\n{'='*100}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*100}")
    print(table)
    print(f"{'='*100}")

    # Save full results
    results_path = os.path.join(SCRIPT_DIR, "benchmark_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nFull results saved to: {results_path}")

    # Save summary table
    summary_path = os.path.join(SCRIPT_DIR, "benchmark_summary.md")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("# Ragale Benchmark Results\n\n")
        f.write(f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(table + "\n")
    print(f"Summary table saved to: {summary_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Ragale Benchmark: 3 models x 3 strategies")
    p.add_argument(
        "--models", nargs="*",
        default=list(MODEL_CONFIGS.keys()),
        choices=list(MODEL_CONFIGS.keys()),
        help="Models to benchmark (default: all 3)",
    )
    p.add_argument(
        "--strategies", nargs="*",
        default=list(STRATEGY_FUNCS.keys()),
        choices=list(STRATEGY_FUNCS.keys()),
        help="Strategies to benchmark (default: all 3)",
    )
    p.add_argument(
        "--seeds", type=int, default=5,
        help="Seeds per topic (default: 5, gives 4 topics x 5 seeds = 20 poems)",
    )
    p.add_argument(
        "--topics", type=int, default=4,
        help="Number of topics to use (default: 4)",
    )
    args = p.parse_args()

    topics = BENCHMARK_TOPICS[:args.topics]
    total_combos = len(args.models) * len(args.strategies)
    total_poems = total_combos * len(topics) * args.seeds

    print("=" * 80)
    print("RAGALE BENCHMARK")
    print("=" * 80)
    print(f"  Models:     {args.models}")
    print(f"  Strategies: {args.strategies}")
    print(f"  Topics:     {topics}")
    print(f"  Seeds:      {args.seeds} per topic")
    print(f"  Total:      {total_combos} combinations x {len(topics) * args.seeds} poems = {total_poems} poems")
    print("=" * 80)

    all_results = []
    summary_rows = []
    bench_start = time.time()

    for model_key in args.models:
        cfg = MODEL_CONFIGS[model_key]
        print(f"\n{'#'*80}")
        print(f"  MODEL: {cfg['desc']} ({model_key})")
        print(f"{'#'*80}")

        # Load model
        model, tokenizer = load_model_by_config(cfg)
        print("  Pre-computing Kannada token data...")
        kannada_ids, kannada_texts, static_mask, newline_token_id = (
            precompute_token_data(tokenizer)
        )

        # Run each strategy
        for strat_key in args.strategies:
            strat_fn = STRATEGY_FUNCS[strat_key]
            print(f"\n  --- Strategy: {strat_key} ---")

            results = run_combination(
                model, tokenizer, strat_key, strat_fn,
                model_key, topics, args.seeds,
                kannada_ids, kannada_texts, static_mask, newline_token_id,
            )
            all_results.extend(results)

            row = compute_summary_row(model_key, strat_key, results)
            summary_rows.append(row)

        # Unload model before loading next — delete all references first
        del model, tokenizer, kannada_ids, kannada_texts, static_mask, newline_token_id
        unload_model()

    bench_elapsed = time.time() - bench_start
    print(f"\nTotal benchmark time: {bench_elapsed:.0f}s ({bench_elapsed/60:.1f}m)")

    # Print and save
    print_and_save_summary(summary_rows, all_results)


if __name__ == "__main__":
    main()
