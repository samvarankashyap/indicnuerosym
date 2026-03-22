#!/usr/bin/env python3
"""
Base Model Benchmark — Constrained & Unconstrained Dwipada Generation.

Runs the same benchmark as constrained_generate_v2.py but on the BASE
google/gemma-3-1b-it model (no fine-tuning). This establishes a baseline
to measure the contribution of:
  - Fine-tuning alone (base unconstrained vs finetuned unconstrained)
  - Constrained decoding alone (base unconstrained vs base constrained)
  - Both combined (base unconstrained vs finetuned constrained)

Uses identical prompts, seeds, and generation parameters as the
fine-tuned benchmark for apples-to-apples comparison.

Output files (separate from fine-tuned results):
  - domino/benchmark_base_constrained_1000.json
  - domino/benchmark_base_unconstrained_1000.json

Usage:
    # Full run (1000 poems, both modes):
    python domino/constrained_generate_base.py

    # Quick test:
    python domino/constrained_generate_base.py --num-poems 50

    # Constrained only:
    python domino/constrained_generate_base.py --constrained-only

    # Unconstrained only:
    python domino/constrained_generate_base.py --baseline-only
"""

import argparse
import json
import os
import sys
from collections import Counter

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

import constrained_generate
from constrained_generate import generate_poem, validate_poem, TOP_K
from constrained_generate_v2 import (
    build_diverse_prompts,
    run_large_benchmark,
    analyze_results,
    save_results,
)

BASE_MODEL_ID = "google/gemma-3-1b-it"

# ── Override prompt for base model ───────────────────────────────────────────
# The fine-tuned model uses the IFT training prompt (in constrained_generate.py).
# The base model needs a more detailed prompt with explicit dwipada rules.

BASE_SYSTEM_PROMPT = """You are an expert Telugu poet (Maha Kavi) specialising in Dwipada poetry (ద్విపద పద్యం).

### DWIPADA STRUCTURE
Each line (పాదం) has exactly: 3 Indra ganas (ఇంద్ర గణాలు) + 1 Surya gana (సూర్య గణం).

Guru (గురువు, U) = long syllable. Laghu (లఘువు, I) = short syllable.

Indra ganas (3-4 syllables each):
  నల (Nala): I I I I
  నగ (Naga): I I I U
  సల (Sala): I I U I
  భ (Bha):    U I I
  ర (Ra):     U I U
  త (Ta):     U U I

Surya ganas (2-3 syllables):
  న (Na):     I I I
  హ/గల (Ha): U I

### RULES
1. **Length:** Exactly 2 lines. No more, no less.
2. **Prasa (ప్రాస):** The 2nd syllable (అక్షరం) of Line 1 must have the same base consonant as the 2nd syllable of Line 2.
3. **Language:** Use simple, natural Telugu (సరళమైన తెలుగు).
4. **Meter:** Each line = 3 Indra ganas + 1 Surya gana = 11-15 syllables total.

### OUTPUT CONSTRAINTS
Output ONLY the two lines of the Telugu poem.
No titles, no introductory text, no JSON, no English, no markdown, no explanations.
Just two lines of Telugu poetry."""


def _base_build_prompt(topic, tokenizer):
    """Build prompt optimized for the base (non-fine-tuned) model."""
    user_prompt = (
        f"క్రింది అంశంపై ఒక ద్విపద పద్యం రాయండి.\n\n"
        f"అంశం: {topic}\n\n"
        f"రెండు పాదాల తెలుగు ద్విపద పద్యం మాత్రమే రాయండి. "
        f"ఇతర వచనం రాయకూడదు."
    )
    messages = [
        {"role": "system", "content": BASE_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


# Monkey-patch: base model uses detailed prompt, fine-tuned uses IFT prompt
constrained_generate.build_prompt = _base_build_prompt


def load_finetuned_stats():
    """Load existing fine-tuned benchmark results for comparison."""
    stats = {}
    for mode in ["constrained", "unconstrained"]:
        path = os.path.join(SCRIPT_DIR, f"benchmark_{mode}_1000.json")
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            stats[mode] = {
                "poem_accuracy": data["poem_accuracy"],
                "line_accuracy": data["line_accuracy"],
                "elapsed": data["elapsed"],
                "total_poems": data["total_poems"],
            }
    return stats


def print_comparison(base_stats, finetuned_stats):
    """Print 4-way comparison table."""
    print(f"\n{'='*72}")
    print(f"  4-WAY COMPARISON: Base vs Fine-tuned × Constrained vs Unconstrained")
    print(f"{'='*72}")
    print(f"\n  {'':25s} {'Constrained':>15s} {'Unconstrained':>15s}")
    print(f"  {'':25s} {'─'*15} {'─'*15}")

    # Base model row
    bc = base_stats.get("constrained", {})
    bu = base_stats.get("unconstrained", {})
    bc_acc = f"{bc['poem_accuracy']:.1f}%" if bc else "—"
    bu_acc = f"{bu['poem_accuracy']:.1f}%" if bu else "—"
    print(f"  {'Base (gemma-3-1b-it)':25s} {bc_acc:>15s} {bu_acc:>15s}")

    # Fine-tuned row
    fc = finetuned_stats.get("constrained", {})
    fu = finetuned_stats.get("unconstrained", {})
    fc_acc = f"{fc['poem_accuracy']:.1f}%" if fc else "—"
    fu_acc = f"{fu['poem_accuracy']:.1f}%" if fu else "—"
    print(f"  {'Fine-tuned (LoRA)':25s} {fc_acc:>15s} {fu_acc:>15s}")

    # Line accuracy
    print(f"\n  Line-level accuracy:")
    print(f"  {'':25s} {'Constrained':>15s} {'Unconstrained':>15s}")
    print(f"  {'':25s} {'─'*15} {'─'*15}")
    bc_la = f"{bc['line_accuracy']:.1f}%" if bc else "—"
    bu_la = f"{bu['line_accuracy']:.1f}%" if bu else "—"
    fc_la = f"{fc['line_accuracy']:.1f}%" if fc else "—"
    fu_la = f"{fu['line_accuracy']:.1f}%" if fu else "—"
    print(f"  {'Base (gemma-3-1b-it)':25s} {bc_la:>15s} {bu_la:>15s}")
    print(f"  {'Fine-tuned (LoRA)':25s} {fc_la:>15s} {fu_la:>15s}")

    # Timing
    print(f"\n  Avg time per poem:")
    print(f"  {'':25s} {'Constrained':>15s} {'Unconstrained':>15s}")
    print(f"  {'':25s} {'─'*15} {'─'*15}")
    bc_t = f"{bc['elapsed']/bc['total_poems']:.2f}s" if bc else "—"
    bu_t = f"{bu['elapsed']/bu['total_poems']:.2f}s" if bu else "—"
    fc_t = f"{fc['elapsed']/fc['total_poems']:.2f}s" if fc else "—"
    fu_t = f"{fu['elapsed']/fu['total_poems']:.2f}s" if fu else "—"
    print(f"  {'Base (gemma-3-1b-it)':25s} {bc_t:>15s} {bu_t:>15s}")
    print(f"  {'Fine-tuned (LoRA)':25s} {fc_t:>15s} {fu_t:>15s}")


def main():
    p = argparse.ArgumentParser(description="Base Model Dwipada Benchmark")
    p.add_argument("--num-poems", type=int, default=1000,
                   help="Total poems to generate (default 1000)")
    p.add_argument("--seeds-per-prompt", type=int, default=5,
                   help="Seeds per prompt (default 5)")
    p.add_argument("--baseline-only", action="store_true",
                   help="Run ONLY unconstrained")
    p.add_argument("--constrained-only", action="store_true",
                   help="Run ONLY constrained")
    p.add_argument("--top-k", type=int, default=50)
    args = p.parse_args()

    constrained_generate.TOP_K = args.top_k

    num_prompts = args.num_poems // args.seeds_per_prompt
    if num_prompts < 1:
        num_prompts = 1

    print("=" * 72)
    print("Base Model Dwipada Benchmark (google/gemma-3-1b-it)")
    print(f"  Target poems:      {num_prompts * args.seeds_per_prompt}")
    print(f"  Prompts:           {num_prompts}")
    print(f"  Seeds per prompt:  {args.seeds_per_prompt}")
    print(f"  Top-K:             {args.top_k}")
    print("=" * 72)

    # Build prompts — same seed as v2 for identical prompt set
    print("\n  Building diverse prompt set...")
    prompts_with_cats = build_diverse_prompts(num_prompts=num_prompts, seed=42)
    categories = Counter(cat for _, cat in prompts_with_cats)
    print(f"  Selected {len(prompts_with_cats)} prompts:")
    for cat, count in sorted(categories.items()):
        print(f"    {cat:15s}: {count}")

    # Load BASE model (no fine-tuning)
    print(f"\n  Loading BASE model: {BASE_MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    )
    model.eval()

    prompts_texts = [p for p, _ in prompts_with_cats]
    base_stats = {}

    # ── Constrained ──────────────────────────────────────────────────
    if not args.baseline_only:
        results_c, stats_c = run_large_benchmark(
            model, tokenizer, prompts_with_cats,
            seeds_per_prompt=args.seeds_per_prompt,
            constrained=True, label="BASE MODEL — CONSTRAINED",
        )
        per_prompt_c = analyze_results(results_c, stats_c, prompts_texts)
        save_results(results_c, stats_c, per_prompt_c,
                     os.path.join(SCRIPT_DIR, "benchmark_base_constrained_1000.json"))
        base_stats["constrained"] = stats_c

    # ── Unconstrained ────────────────────────────────────────────────
    if not args.constrained_only:
        results_u, stats_u = run_large_benchmark(
            model, tokenizer, prompts_with_cats,
            seeds_per_prompt=args.seeds_per_prompt,
            constrained=False, label="BASE MODEL — UNCONSTRAINED",
        )
        per_prompt_u = analyze_results(results_u, stats_u, prompts_texts)
        save_results(results_u, stats_u, per_prompt_u,
                     os.path.join(SCRIPT_DIR, "benchmark_base_unconstrained_1000.json"))
        base_stats["unconstrained"] = stats_u

    # ── Base model comparison ────────────────────────────────────────
    if "constrained" in base_stats and "unconstrained" in base_stats:
        print(f"\n{'='*72}")
        print(f"  BASE MODEL COMPARISON")
        print(f"{'='*72}")
        print(f"                        Constrained    Unconstrained")
        print(f"  Poem accuracy:        {base_stats['constrained']['poem_accuracy']:6.1f}%"
              f"        {base_stats['unconstrained']['poem_accuracy']:6.1f}%")
        print(f"  Line accuracy:        {base_stats['constrained']['line_accuracy']:6.1f}%"
              f"        {base_stats['unconstrained']['line_accuracy']:6.1f}%")

    # ── 4-way comparison with fine-tuned results ─────────────────────
    finetuned_stats = load_finetuned_stats()
    if finetuned_stats:
        print_comparison(base_stats, finetuned_stats)

    print("\n" + "=" * 72)
    print("Done.")


if __name__ == "__main__":
    main()
