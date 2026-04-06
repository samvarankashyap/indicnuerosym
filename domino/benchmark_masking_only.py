#!/usr/bin/env python3
"""
Benchmark: Logit Masking Only (no rejection, no backtracking).

Approach:
    1. Pre-compute a static mask killing all non-Telugu tokens.
    2. At each step, build a dynamic mask using CompositeState (incremental
       NFA) — only tokens that keep the NFA alive are allowed.
    3. Apply both masks to logits, then do standard top-p sampling.
    4. If no valid tokens remain, fall back to the most probable token.

Expected validity: ~40-50% on gemma3-1b-merged.

This is the simplest constrained decoding approach. Each individual token
is guaranteed to keep the NFA reachable, but the model can still paint
itself into a corner because the mask doesn't ensure the *sequence* of
tokens leads to a valid gana completion.

Usage:
    python domino/benchmark_masking_only.py --model gemma3-1b-merged --seeds 20 --topic "శ్రీరాముడు"
"""

import argparse
import os
import sys
import time

import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
NFA_DIR = os.path.join(PROJECT_DIR, "nfa_for_dwipada")
sys.path.insert(0, NFA_DIR)
sys.path.insert(0, SCRIPT_DIR)

from composite_state import (
    CompositeState, build_gana_mask, get_telugu_token_set,
    VALID_LINE_LENGTHS, MAX_LINE_LENGTH,
)
from constrained_generate import (
    validate_poem, strip_poem_prefix, BENCHMARK_PROMPTS,
)
from constrained_generate_v2 import MODEL_CHOICES, load_model, build_prompt_for_model, is_finetuned_model

METHOD_NAME = "masking_only"


###############################################################################
# PRE-COMPUTATION
###############################################################################

def precompute_token_data(tokenizer):
    """Build static mask and Telugu token set for logit masking."""
    telugu_ids, telugu_texts = get_telugu_token_set(tokenizer)
    vocab_size = tokenizer.vocab_size
    static_mask = torch.full((vocab_size,), float("-inf"))
    for tid in telugu_ids:
        static_mask[tid] = 0.0
    if tokenizer.eos_token_id is not None:
        static_mask[tokenizer.eos_token_id] = 0.0
    newline_ids = tokenizer.encode("\n", add_special_tokens=False)
    newline_token_id = newline_ids[0] if newline_ids else None
    print(f"  Telugu tokens: {len(telugu_ids)}")
    print(f"  Newline token ID: {newline_token_id}")
    return telugu_ids, telugu_texts, static_mask, newline_token_id


###############################################################################
# GENERATION — MASKING ONLY
###############################################################################

def generate_poem(
    model, tokenizer, topic,
    telugu_ids, telugu_texts, static_mask, newline_token_id,
    max_new_tokens=150, temperature=0.7, top_p=0.9, seed=42,
    model_choice="gemma3-1b-merged",
):
    """
    Generate a dwipada poem using logit masking only.

    At each step:
      1. Forward pass → logits
      2. Apply static mask (kill non-Telugu)
      3. Apply dynamic Gana NFA mask (kill tokens that make NFA unreachable)
      4. Top-p sampling from the masked logits
      5. No backtracking — once a token is chosen, it's final

    Returns dict with poem, validity, timing, and statistics.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    input_text = build_prompt_for_model(topic, tokenizer, model_choice)
    input_ids = tokenizer(input_text, return_tensors="pt").to(model.device)

    generated_ids = []
    start_time = time.time()
    lines_done = 0
    mask_computations = 0

    device = model.device
    device_static_mask = static_mask.to(device)
    composite = CompositeState()
    # Fine-tuned models output "ద్విపద:" prefix; base models output poem directly
    prefix_done = not is_finetuned_model(model_choice)

    with torch.no_grad():
        for step in range(max_new_tokens):
            if lines_done >= 2:
                break

            # Forward pass
            outputs = model(**input_ids)
            logits = outputs.logits[:, -1, :]
            if temperature > 0:
                logits = logits / temperature

            # Static mask: kill non-Telugu tokens
            logits = logits + device_static_mask.unsqueeze(0)

            # Detect prefix ("ద్విపద:" or similar)
            if not prefix_done:
                current_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
                stripped = strip_poem_prefix(current_text)
                if stripped != current_text or ":" in current_text:
                    prefix_done = True
                    composite = CompositeState()
                    for ch in stripped:
                        composite.feed_char(ch)

            # Dynamic Gana NFA mask
            if prefix_done:
                # Force newline if line has valid completion
                flushed = CompositeState.from_snapshot(composite.snapshot())
                flushed.flush()
                if flushed.has_accept():
                    if newline_token_id is not None:
                        force_mask = torch.full_like(logits, float("-inf"))
                        force_mask[0, newline_token_id] = 0.0
                        logits = logits + force_mask
                        chosen_id = newline_token_id
                        composite.feed_char("\n")
                        lines_done = composite.lines_complete
                        generated_ids.append(chosen_id)
                        next_input = torch.tensor([[chosen_id]], device=device)
                        input_ids = {
                            "input_ids": torch.cat([input_ids["input_ids"], next_input], dim=1),
                            "attention_mask": torch.cat([input_ids["attention_mask"],
                                torch.ones((1, 1), dtype=torch.long, device=device)], dim=1),
                        }
                        continue

                # Build dynamic mask over Telugu tokens
                snap = composite.snapshot()
                valid_ids = build_gana_mask(snap, telugu_ids, telugu_texts)
                mask_computations += 1

                # Filter newline: only allow if it completes a valid line
                if newline_token_id in valid_ids:
                    clone = CompositeState.from_snapshot(snap)
                    clone.feed_char("\n")
                    if clone.lines_complete <= lines_done:
                        valid_ids.discard(newline_token_id)

                # Apply dynamic mask
                dynamic_mask = torch.full_like(logits, float("-inf"))
                for tid in valid_ids:
                    dynamic_mask[0, tid] = 0.0
                logits = logits + dynamic_mask

            # Handle poem completion: only allow EOS
            if lines_done >= 2:
                eos_mask = torch.full_like(logits, float("-inf"))
                if tokenizer.eos_token_id is not None:
                    eos_mask[0, tokenizer.eos_token_id] = 0.0
                logits = logits + eos_mask

            # Top-p sampling
            probs = torch.softmax(logits, dim=-1)
            sorted_probs, sorted_indices = torch.sort(probs[0], descending=True)
            cumsum = torch.cumsum(sorted_probs, dim=0)
            remove = cumsum - sorted_probs > top_p
            sorted_probs[remove] = 0
            if sorted_probs.sum() > 0:
                sorted_probs = sorted_probs / sorted_probs.sum()
            else:
                sorted_probs[0] = 1.0  # fallback: force most probable token
            choice = torch.multinomial(sorted_probs, 1)
            chosen_id = sorted_indices[choice].item()

            if chosen_id == tokenizer.eos_token_id:
                break

            generated_ids.append(chosen_id)

            # Advance composite state
            if prefix_done:
                chosen_text = tokenizer.decode([chosen_id])
                for ch in chosen_text:
                    composite.feed_char(ch)
                lines_done = composite.lines_complete

            # Advance model input
            next_input = torch.tensor([[chosen_id]], device=device)
            input_ids = {
                "input_ids": torch.cat([input_ids["input_ids"], next_input], dim=1),
                "attention_mask": torch.cat([input_ids["attention_mask"],
                    torch.ones((1, 1), dtype=torch.long, device=device)], dim=1),
            }

    elapsed = time.time() - start_time
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    poem_lines = _extract_poem_lines(generated_text)
    valid_lines = validate_poem("\n".join(poem_lines))

    return {
        "topic": topic, "seed": seed, "constrained": True,
        "method": METHOD_NAME,
        "generated_text": generated_text, "poem_lines": poem_lines,
        "valid_lines": valid_lines,
        "all_valid": len(valid_lines) == 2 and all(v["valid"] for v in valid_lines),
        "elapsed": elapsed,
        "tokens_generated": len(generated_ids),
        "backtracks": 0,
        "mask_computations": mask_computations,
    }


###############################################################################
# HELPERS
###############################################################################

def _extract_poem_lines(text):
    """Extract poem lines from model output, stripping prefixes."""
    raw = [l.strip() for l in text.split("\n") if l.strip()]
    lines = []
    for l in raw:
        if l.startswith("ద్విపద:") or l.startswith("పూర్తి ద్విపద:"):
            after = l.split(":", 1)[1].strip()
            if after:
                lines.append(after)
            continue
        if l.endswith(":") or l in ("ద్విపద", "పూర్తి ద్విపద"):
            continue
        lines.append(l)
    return lines[:2]


###############################################################################
# EXPERIMENT RUNNER
###############################################################################

def run_experiment(model, tokenizer, topic, telugu_ids, telugu_texts,
                   static_mask, newline_token_id, num_seeds=10, model_name=""):
    print(f"\n{'='*72}")
    print(f"  {METHOD_NAME.upper()} — {model_name}")
    print(f"  Topic: {topic}")
    print(f"  Seeds: {num_seeds}")
    print(f"{'='*72}")

    results = []
    for i in range(num_seeds):
        seed = 42 + i * 7
        print(f"\n  Seed {seed}:", end=" ", flush=True)
        r = generate_poem(model, tokenizer, topic,
                          telugu_ids, telugu_texts, static_mask, newline_token_id,
                          seed=seed, model_choice=model_name)
        results.append(r)
        status = "V VALID" if r["all_valid"] else "X INVALID"
        print(f"{status} ({r['elapsed']:.1f}s, {r['tokens_generated']} tok, "
              f"{r['mask_computations']} masks)")
        for vl in r["valid_lines"]:
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
    print(f"  BENCHMARK STATS — {METHOD_NAME}")
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
# CLI
###############################################################################

def main():
    p = argparse.ArgumentParser(description=f"Dwipada benchmark: {METHOD_NAME}")
    p.add_argument("--model", type=str, default="gemma3-1b-merged", choices=MODEL_CHOICES)
    p.add_argument("--seeds", type=int, default=20)
    p.add_argument("--topic", type=str, default=None)
    args = p.parse_args()

    model, tokenizer = load_model(args.model)
    print(f"\n  Pre-computing Telugu token data...")
    telugu_ids, telugu_texts, static_mask, newline_token_id = precompute_token_data(tokenizer)
    topics = [args.topic] if args.topic else BENCHMARK_PROMPTS[:3]

    for topic in topics:
        run_experiment(model, tokenizer, topic,
                       telugu_ids, telugu_texts, static_mask, newline_token_id,
                       num_seeds=args.seeds, model_name=args.model)


if __name__ == "__main__":
    main()
