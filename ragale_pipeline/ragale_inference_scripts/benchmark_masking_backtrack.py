#!/usr/bin/env python3
"""
Benchmark: Logit Masking + Backtracking — Ragale.

Mirrors domino/benchmark_masking_backtrack.py for Kannada Utsaha Ragale.

Approach:
    1. Same logit masking as masking-only (static + dynamic Gana NFA mask).
    2. Force newline when NFA reaches ACCEPT at valid line length.
    3. If masking leaves < MIN_VALID_TOKENS, backtrack to a checkpoint:
       - Checkpoints saved every 3 tokens.
       - On backtrack: pop checkpoint, restore CompositeState, bump temp, reseed.
    4. Escalating temperature + RNG reseed on consecutive backtracks.

Expected validity: ~70-90%.

Usage:
    python ragale_inference_scripts/benchmark_masking_backtrack.py --model gemma3-1b-base --seeds 34
    python ragale_inference_scripts/benchmark_masking_backtrack.py --model gemma3-1b-lora --seeds 34
"""

import argparse
import json
import os
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from shared_utils import (
    load_model_by_choice, precompute_token_data,
    MODEL_CHOICES, BENCHMARK_TOPICS_DEFAULT, BENCHMARK_SEEDS_DEFAULT, is_lora_model,
)
from generate_masking_backtrack import generate_poem

METHOD_NAME = "masking_backtrack"


###############################################################################
# EXPERIMENT RUNNER
###############################################################################

def run_experiment(model, tokenizer, topic, kannada_ids, kannada_texts,
                   static_mask, newline_token_id, num_seeds=34, model_name=""):
    print(f"\n{'='*72}")
    print(f"  {METHOD_NAME.upper()} -- {model_name}")
    print(f"  Topic: {topic}")
    print(f"  Seeds: {num_seeds}")
    print(f"{'='*72}")

    results = []
    for i in range(num_seeds):
        seed = 42 + i * 7
        print(f"\n  Seed {seed}:", end=" ", flush=True)
        r = generate_poem(model, tokenizer, topic,
                          kannada_ids, kannada_texts, static_mask, newline_token_id,
                          seed=seed, model_name=model_name)
        results.append(r)
        status = "V VALID" if r["all_valid"] else "X INVALID"
        bt = r.get("backtracks", 0)
        bt_str = f", {bt} bt" if bt > 0 else ""
        print(f"{status} ({r['elapsed']:.1f}s, {r['tokens_generated']} tok, "
              f"{r['mask_computations']} masks{bt_str})")
        for vl in r.get("valid_lines", []):
            mark = "V" if vl["valid"] else "X"
            print(f"    {mark} {vl['line'][:65]}")
            if vl["valid"]:
                print(f"      {vl['partition']}")

    _print_stats(results, model_name, topic, num_seeds)
    return results


def _print_stats(results, model_name, topic, num_seeds):
    valid = sum(1 for r in results if r["all_valid"])
    total_time = sum(r["elapsed"] for r in results)
    avg_time = total_time / len(results) if results else 0
    avg_tok = sum(r["tokens_generated"] for r in results) / len(results) if results else 0
    avg_masks = sum(r["mask_computations"] for r in results) / len(results) if results else 0
    avg_bt = sum(r.get("backtracks", 0) for r in results) / len(results) if results else 0

    print(f"\n{'='*50}")
    print(f"  BENCHMARK STATS -- {METHOD_NAME}")
    print(f"{'='*50}")
    print(f"  Model:             {model_name}")
    print(f"  Topic:             {topic}")
    print(f"  Seeds:             {num_seeds}")
    print(f"  Valid poems:       {valid}/{num_seeds} ({valid/num_seeds*100:.1f}%)")
    print(f"  Avg time/poem:     {avg_time:.1f}s")
    print(f"  Avg tokens/poem:   {avg_tok:.0f}")
    print(f"  Avg masks/poem:    {avg_masks:.0f}")
    print(f"  Avg backtracks:    {avg_bt:.1f}")
    print(f"  Total time:        {total_time:.1f}s")
    print(f"{'='*50}")


###############################################################################
# SUMMARY
###############################################################################

def _build_summary(all_results, model_name, total_time):
    """Build a summary dict matching the domino benchmark JSON format."""
    total_poems = len(all_results)
    total_valid = sum(1 for r in all_results if r["all_valid"])
    total_lines = sum(len(r.get("valid_lines", [])) for r in all_results)
    valid_lines = sum(
        sum(1 for vl in r.get("valid_lines", []) if vl["valid"])
        for r in all_results
    )
    per_topic = {}
    for r in all_results:
        t = r["topic"]
        if t not in per_topic:
            per_topic[t] = {"total": 0, "valid": 0}
        per_topic[t]["total"] += 1
        if r["all_valid"]:
            per_topic[t]["valid"] += 1
    for v in per_topic.values():
        v["rate"] = v["valid"] / v["total"] * 100 if v["total"] else 0

    return {
        "label": f"{model_name} -- {METHOD_NAME}",
        "method": METHOD_NAME,
        "model": model_name,
        "constrained": True,
        "total_poems": total_poems,
        "total_valid": total_valid,
        "poem_accuracy": total_valid / total_poems * 100 if total_poems else 0,
        "total_lines": total_lines,
        "valid_lines": valid_lines,
        "line_accuracy": valid_lines / total_lines * 100 if total_lines else 0,
        "total_time": total_time,
        "per_topic": per_topic,
        "poems": all_results,
    }


###############################################################################
# CLI
###############################################################################

def main():
    p = argparse.ArgumentParser(description=f"Ragale benchmark: {METHOD_NAME}")
    p.add_argument("--model", type=str, default="gemma3-1b-base", choices=MODEL_CHOICES)
    p.add_argument("--seeds", type=int, default=BENCHMARK_SEEDS_DEFAULT,
                   help=f"Seeds per topic (default: {BENCHMARK_SEEDS_DEFAULT}, "
                        f"gives 3 topics x {BENCHMARK_SEEDS_DEFAULT} = "
                        f"{3 * BENCHMARK_SEEDS_DEFAULT} poems)")
    p.add_argument("--topic", type=str, default=None,
                   help="Single topic to benchmark (default: first 3)")
    p.add_argument("--output", type=str, default=None,
                   help="Path to save results JSON")
    args = p.parse_args()

    model, tokenizer = load_model_by_choice(args.model)
    print(f"\n  Pre-computing Kannada token data...")
    kannada_ids, kannada_texts, static_mask, newline_token_id = precompute_token_data(tokenizer)
    topics = [args.topic] if args.topic else BENCHMARK_TOPICS_DEFAULT

    all_results = []
    total_start = time.time()
    for topic in topics:
        results = run_experiment(model, tokenizer, topic,
                                kannada_ids, kannada_texts, static_mask, newline_token_id,
                                num_seeds=args.seeds, model_name=args.model)
        all_results.extend(results)
    total_time = time.time() - total_start

    output_path = args.output or os.path.join(
        SCRIPT_DIR, f"benchmark_{METHOD_NAME}_{args.model}.json"
    )
    summary = _build_summary(all_results, args.model, total_time)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n  Results saved to: {output_path}")


if __name__ == "__main__":
    main()
