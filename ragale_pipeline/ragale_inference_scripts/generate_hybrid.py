#!/usr/bin/env python3
"""
Approach 4: Hybrid (Logit Masking + NFA Rejection Sampling + Forced Newlines).

Algorithm:
    1. Same logit masking as masking-only (static + dynamic Gana NFA mask)
       narrows vocab to valid Kannada tokens.
    2. From the masked logits, take top-SEARCH_K candidates by probability.
    3. For each candidate, clone CompositeState, feed the token, flush, check
       is_alive() + has_accept() via the incremental NFA.
    4. Prefer "accept candidates" (tokens that reach valid line completion)
       over merely "alive" candidates.
    5. Weighted sample from the best candidate set.
    6. Falls back to backtracking only if NO valid candidate in top-K.

Constraint enforcement: Local (masking) + per-token verification (rejection) +
global (forced newlines). Masking eliminates ~95% of tokens cheaply, rejection
validates the remaining ~5%.

Usage:
    python ragale_inference_scripts/generate_hybrid.py --seeds 5 --topic "Rain"
"""

import argparse
import os
import sys
import time
import json

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
from composite_state import CompositeState, build_gana_mask, MAX_LINE_LENGTH

METHOD_NAME = "hybrid_mask_reject"

# Rejection sampling
SEARCH_K = 100

# Backtracking (fallback — rarely needed)
CHECKPOINT_INTERVAL = 3
MAX_BACKTRACKS_DEFAULT = 30
TEMP_BUMP = 0.15
TEMP_DECAY_STEPS = 6
MAX_TEMP = 1.2
MAX_CHECKPOINTS = 15


def generate_poem(
    model, tokenizer, topic,
    kannada_ids, kannada_texts, static_mask, newline_token_id,
    max_new_tokens=150, temperature=0.7, top_p=0.9, seed=42,
    max_backtracks=MAX_BACKTRACKS_DEFAULT,
    model_name=None,
):
    """Generate a Ragale poem using hybrid masking + NFA rejection sampling."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    input_text = build_prompt(topic, tokenizer, model_name=model_name)
    input_ids = tokenizer(input_text, return_tensors="pt").to(model.device)

    generated_ids = []
    start_time = time.time()
    lines_done = 0
    mask_computations = 0
    device = model.device
    device_static_mask = static_mask.to(device)
    composite = CompositeState()

    # Backtracking state (fallback)
    checkpoints = []
    backtrack_count = 0
    consecutive_backtracks = 0
    current_temp = temperature
    temp_boost_remaining = 0

    with torch.no_grad():
        step = 0
        while step < max_new_tokens:
            if lines_done >= 2:
                break

            outputs = model(**input_ids)
            logits = outputs.logits[:, -1, :]
            if current_temp > 0:
                logits = logits / current_temp

            # Static mask
            logits = logits + device_static_mask.unsqueeze(0)

            # Save checkpoint
            if step % CHECKPOINT_INTERVAL == 0:
                checkpoints.append((
                    step, composite.snapshot(), len(generated_ids),
                    input_ids["input_ids"].shape[1], lines_done, current_temp,
                ))
                if len(checkpoints) > MAX_CHECKPOINTS:
                    checkpoints = checkpoints[-MAX_CHECKPOINTS:]

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
                    step += 1
                    consecutive_backtracks = 0
                    continue

            # Dynamic Gana NFA mask
            snap = composite.snapshot()
            valid_ids = build_gana_mask(snap, kannada_ids, kannada_texts)
            mask_computations += 1

            # Filter newline
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

            # --- Hybrid sampling: mask + NFA rejection ---
            probs = torch.softmax(logits, dim=-1)
            sorted_probs, sorted_indices = torch.sort(probs[0], descending=True)

            chosen_id = None
            valid_candidates = []
            accept_candidates = []

            for k in range(min(SEARCH_K, len(sorted_indices))):
                if sorted_probs[k].item() < 1e-8:
                    break

                cand_id = sorted_indices[k].item()
                if cand_id == tokenizer.eos_token_id:
                    if lines_done >= 2:
                        chosen_id = cand_id
                        break
                    continue

                # Clone NFA, feed candidate, flush, check validity
                cand_text = tokenizer.decode([cand_id])
                clone = CompositeState.from_snapshot(snap)
                for ch in cand_text:
                    clone.feed_char(ch)
                flushed_clone = CompositeState.from_snapshot(clone.snapshot())
                flushed_clone.flush()

                if not flushed_clone.is_alive():
                    continue

                # Reject newlines that don't complete valid line
                if "\n" in cand_text:
                    if clone.lines_complete <= lines_done:
                        continue

                # Reject overshoot
                if flushed_clone.syllable_count > MAX_LINE_LENGTH and not flushed_clone.has_accept():
                    continue

                # Track accept candidates
                if flushed_clone.has_accept():
                    accept_candidates.append((cand_id, sorted_probs[k].item()))

                valid_candidates.append((cand_id, sorted_probs[k].item()))

                # Early stop
                if len(accept_candidates) >= 3:
                    break
                if len(valid_candidates) >= 10 and not accept_candidates:
                    break

            # Prefer accept candidates
            use_candidates = accept_candidates if accept_candidates else valid_candidates

            if use_candidates and chosen_id is None:
                cand_probs = torch.tensor([c[1] for c in use_candidates], device=device)
                cand_probs = cand_probs / cand_probs.sum()
                choice_idx = torch.multinomial(cand_probs, 1).item()
                chosen_id = use_candidates[choice_idx][0]
            elif chosen_id is None:
                # No valid candidate — backtrack
                if checkpoints and backtrack_count < max_backtracks:
                    backtrack_count += 1
                    consecutive_backtracks += 1
                    pop_count = min(consecutive_backtracks, len(checkpoints))
                    for _ in range(pop_count):
                        ckpt = checkpoints.pop()
                    ckpt_step, ckpt_snap, ckpt_gen_len, ckpt_ids_len, ckpt_lines, ckpt_temp = ckpt
                    composite = CompositeState.from_snapshot(ckpt_snap)
                    generated_ids = generated_ids[:ckpt_gen_len]
                    input_ids = {
                        "input_ids": input_ids["input_ids"][:, :ckpt_ids_len],
                        "attention_mask": input_ids["attention_mask"][:, :ckpt_ids_len],
                    }
                    lines_done = ckpt_lines
                    step = ckpt_step
                    current_temp = min(ckpt_temp + TEMP_BUMP * consecutive_backtracks, MAX_TEMP)
                    temp_boost_remaining = TEMP_DECAY_STEPS
                    new_seed = seed + backtrack_count * 1337
                    torch.manual_seed(new_seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed(new_seed)
                    continue
                else:
                    chosen_id = sorted_indices[0].item()

            if chosen_id == tokenizer.eos_token_id:
                break

            generated_ids.append(chosen_id)

            chosen_text = tokenizer.decode([chosen_id])
            for ch in chosen_text:
                composite.feed_char(ch)
            lines_done = composite.lines_complete

            next_input = torch.tensor([[chosen_id]], device=device)
            input_ids = {
                "input_ids": torch.cat([input_ids["input_ids"], next_input], dim=1),
                "attention_mask": torch.cat([input_ids["attention_mask"],
                    torch.ones((1, 1), dtype=torch.long, device=device)], dim=1),
            }

            consecutive_backtracks = 0
            if temp_boost_remaining > 0:
                temp_boost_remaining -= 1
                if temp_boost_remaining == 0:
                    current_temp = temperature

            step += 1

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
        "backtracks": backtrack_count, "mask_computations": mask_computations,
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
    main()
