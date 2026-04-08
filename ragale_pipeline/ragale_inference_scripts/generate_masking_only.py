#!/usr/bin/env python3
"""
Approach 2: Logit Masking Only (no backtracking).

Algorithm:
    1. Pre-compute static mask killing all non-Kannada tokens.
    2. At each step, build a dynamic mask using CompositeState (incremental
       NFA) — only tokens that keep the NFA alive are allowed.
    3. Apply both masks to logits, then do standard top-p sampling.
    4. Force newline when the NFA reaches ACCEPT at 12 syllables.
    5. If no valid tokens remain, fall back to the most probable token.

Constraint enforcement: Local only. Each token is guaranteed to keep the NFA
alive, but there's no mechanism to force line completion beyond the forced
newline at ACCEPT.

Usage:
    python ragale_inference_scripts/generate_masking_only.py --seeds 5 --topic "Bubbles"
"""

import argparse
import os
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import torch
from shared_utils import (
    load_model, build_prompt, precompute_token_data,
    validate_poem, extract_poem_lines, print_result, print_summary,
    BENCHMARK_TOPICS,
)

NFA_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "nfa_pipeline")
sys.path.insert(0, NFA_DIR)
from composite_state import CompositeState, build_gana_mask

METHOD_NAME = "masking_only"


def generate_poem(
    model, tokenizer, topic,
    kannada_ids, kannada_texts, static_mask, newline_token_id,
    max_new_tokens=150, temperature=0.7, top_p=0.9, seed=42,
):
    """Generate a Ragale poem using logit masking only."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    input_text = build_prompt(topic, tokenizer)
    input_ids = tokenizer(input_text, return_tensors="pt").to(model.device)

    generated_ids = []
    start_time = time.time()
    lines_done = 0
    mask_computations = 0
    device = model.device
    device_static_mask = static_mask.to(device)
    composite = CompositeState()

    with torch.no_grad():
        for step in range(max_new_tokens):
            if lines_done >= 2:
                break

            outputs = model(**input_ids)
            logits = outputs.logits[:, -1, :]
            if temperature > 0:
                logits = logits / temperature

            # Static mask: kill non-Kannada tokens
            logits = logits + device_static_mask.unsqueeze(0)

            # Force newline on valid line completion
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

            # Dynamic Gana NFA mask
            snap = composite.snapshot()
            valid_ids = build_gana_mask(snap, kannada_ids, kannada_texts)
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

            # Handle poem completion
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
                sorted_probs[0] = 1.0
            choice = torch.multinomial(sorted_probs, 1)
            chosen_id = sorted_indices[choice].item()

            if chosen_id == tokenizer.eos_token_id:
                break

            generated_ids.append(chosen_id)

            # Advance composite state
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
    poem_lines = extract_poem_lines(generated_text)
    valid_lines = validate_poem("\n".join(poem_lines)) if len(poem_lines) >= 2 else []

    return {
        "topic": topic, "seed": seed, "method": METHOD_NAME,
        "generated_text": generated_text, "poem_lines": poem_lines,
        "valid_lines": valid_lines,
        "all_valid": len(valid_lines) == 2 and all(v["valid"] for v in valid_lines),
        "elapsed": elapsed, "tokens_generated": len(generated_ids),
        "backtracks": 0, "mask_computations": mask_computations,
    }


def main():
    p = argparse.ArgumentParser(description=f"Ragale: {METHOD_NAME}")
    p.add_argument("--model", type=str, default="google/gemma-3-1b-it")
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--topic", type=str, default=None)
    args = p.parse_args()

    model, tokenizer = load_model(args.model)
    print("  Pre-computing Kannada token data...")
    kannada_ids, kannada_texts, static_mask, newline_token_id = precompute_token_data(tokenizer)

    topics = [args.topic] if args.topic else BENCHMARK_TOPICS[:2]
    all_results = []

    for topic in topics:
        print(f"\n{'='*60}")
        print(f"  Topic: {topic} | Seeds: {args.seeds}")
        print(f"{'='*60}")

        for i in range(args.seeds):
            seed = 42 + i * 7
            print(f"\n  Seed {seed}:", end=" ", flush=True)
            r = generate_poem(model, tokenizer, topic,
                              kannada_ids, kannada_texts, static_mask, newline_token_id,
                              seed=seed)
            all_results.append(r)
            print_result(r)

    print_summary(all_results, METHOD_NAME)

    output_path = os.path.join(SCRIPT_DIR, f"results_{METHOD_NAME}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    import json
    main()
