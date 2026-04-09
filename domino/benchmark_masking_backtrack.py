#!/usr/bin/env python3
"""
Benchmark: Logit Masking + Backtracking.

Approach:
    1. Same logit masking as masking_only (static + dynamic Gana NFA mask).
    2. Saves CompositeState checkpoints every 3 tokens.
    3. Backtrack triggers:
       a) Mask produces < 3 valid tokens (NFA dead end).
       b) Syllable count exceeds MAX_LINE_LENGTH without ACCEPT.
       c) Post-newline validation: line completed but invalid gana partition.
    4. On backtrack: restore checkpoint, bump temperature, reseed RNG.
    5. Escalating depth: consecutive backtracks pop more checkpoints.

Expected validity: ~70% on gemma3-1b-merged (up from ~50% masking-only).

The remaining ~30% failures are seeds where the 1B model cannot produce
valid gana patterns regardless of temperature/seed, because the masking
still allows tokens that are individually reachable but collectively
paint the NFA into a corner.

Usage:
    python domino/benchmark_masking_backtrack.py --model gemma3-1b-merged --seeds 20 --topic "శ్రీరాముడు"
"""

import argparse
import json
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

METHOD_NAME = "masking_backtrack"

# Backtracking constants
CHECKPOINT_INTERVAL = 3
MIN_VALID_TOKENS = 3
MAX_BACKTRACKS_DEFAULT = 30
TEMP_BUMP = 0.15
TEMP_DECAY_STEPS = 6
MAX_TEMP = 1.2
MAX_CHECKPOINTS = 15


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
# GENERATION — MASKING + BACKTRACKING
###############################################################################

def generate_poem(
    model, tokenizer, topic,
    telugu_ids, telugu_texts, static_mask, newline_token_id,
    max_new_tokens=150, temperature=0.7, top_p=0.9, seed=42,
    max_backtracks=MAX_BACKTRACKS_DEFAULT,
    model_choice="gemma3-1b-merged",
):
    """
    Generate a dwipada poem using logit masking with backtracking.

    Same masking as masking_only, but recovers from dead ends:
      - Saves checkpoints every CHECKPOINT_INTERVAL tokens.
      - If mask is nearly empty or line overshoots, restores a checkpoint.
      - After newline, validates the completed line; backtracks if invalid.
      - Temperature bump + reseed on backtrack to diversify.

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
    prefix_done = not is_finetuned_model(model_choice)

    # Backtracking state
    checkpoints = []
    backtrack_count = 0
    consecutive_backtracks = 0
    current_temp = temperature
    temp_boost_remaining = 0
    prefix_step = 0

    def _do_backtrack():
        """Pop checkpoint(s), restore state, bump temp, reseed."""
        nonlocal composite, generated_ids, input_ids, lines_done
        nonlocal backtrack_count, consecutive_backtracks, current_temp
        nonlocal temp_boost_remaining

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
        current_temp = min(ckpt_temp + TEMP_BUMP * consecutive_backtracks, MAX_TEMP)
        temp_boost_remaining = TEMP_DECAY_STEPS
        new_seed = seed + backtrack_count * 1337
        torch.manual_seed(new_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(new_seed)
        return ckpt_step

    with torch.no_grad():
        step = 0
        while step < max_new_tokens:
            if lines_done >= 2:
                break

            # Forward pass
            outputs = model(**input_ids)
            logits = outputs.logits[:, -1, :]
            if current_temp > 0:
                logits = logits / current_temp

            # Static mask
            logits = logits + device_static_mask.unsqueeze(0)

            # Detect prefix
            if not prefix_done:
                current_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
                stripped = strip_poem_prefix(current_text)
                if stripped != current_text or ":" in current_text:
                    prefix_done = True
                    prefix_step = step
                    composite = CompositeState()
                    for ch in stripped:
                        composite.feed_char(ch)

            # Save checkpoint periodically
            if (prefix_done and step > prefix_step
                    and (step - prefix_step) % CHECKPOINT_INTERVAL == 0):
                checkpoints.append((
                    step, composite.snapshot(), len(generated_ids),
                    input_ids["input_ids"].shape[1], lines_done, current_temp,
                ))
                if len(checkpoints) > MAX_CHECKPOINTS:
                    checkpoints = checkpoints[-MAX_CHECKPOINTS:]

            # Dynamic Gana NFA mask
            if prefix_done:
                flushed = CompositeState.from_snapshot(composite.snapshot())
                flushed.flush()

                # Force newline on valid line completion
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

                # Build dynamic mask
                snap = composite.snapshot()
                valid_ids = build_gana_mask(snap, telugu_ids, telugu_texts)
                mask_computations += 1

                # Filter newline
                if newline_token_id in valid_ids:
                    clone = CompositeState.from_snapshot(snap)
                    clone.feed_char("\n")
                    if clone.lines_complete <= lines_done:
                        valid_ids.discard(newline_token_id)

                # Backtrack trigger: mask exhaustion or line overshoot
                mask_empty = len(valid_ids) < MIN_VALID_TOKENS
                overshot = flushed.syllable_count > MAX_LINE_LENGTH
                if (mask_empty or overshot) and checkpoints and backtrack_count < max_backtracks:
                    step = _do_backtrack()
                    continue

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
            if prefix_done:
                chosen_text = tokenizer.decode([chosen_id])
                for ch in chosen_text:
                    composite.feed_char(ch)
                lines_done = composite.lines_complete

                # Post-newline validation: backtrack if line is invalid
                if "\n" in chosen_text and checkpoints and backtrack_count < max_backtracks:
                    gen_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
                    stripped = strip_poem_prefix(gen_text)
                    completed = [l.strip() for l in stripped.split("\n") if l.strip()]
                    if completed:
                        vresult = validate_poem(completed[-1])
                        if vresult and not vresult[0]["valid"]:
                            step = _do_backtrack()
                            continue

            # Advance model input
            next_input = torch.tensor([[chosen_id]], device=device)
            input_ids = {
                "input_ids": torch.cat([input_ids["input_ids"], next_input], dim=1),
                "attention_mask": torch.cat([input_ids["attention_mask"],
                    torch.ones((1, 1), dtype=torch.long, device=device)], dim=1),
            }

            # Successful commit: reset escalation
            consecutive_backtracks = 0
            if temp_boost_remaining > 0:
                temp_boost_remaining -= 1
                if temp_boost_remaining == 0:
                    current_temp = temperature

            step += 1

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
        "backtracks": backtrack_count,
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
        bt = r.get("backtracks", 0)
        bt_str = f", {bt} bt" if bt > 0 else ""
        print(f"{status} ({r['elapsed']:.1f}s, {r['tokens_generated']} tok, "
              f"{r['mask_computations']} masks{bt_str})")
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

def _build_summary(all_results, model_name, total_time):
    """Build a summary dict matching the format of existing benchmark JSONs."""
    total_poems = len(all_results)
    total_valid = sum(1 for r in all_results if r["all_valid"])
    total_lines = sum(len(r["valid_lines"]) for r in all_results)
    valid_lines = sum(
        sum(1 for vl in r["valid_lines"] if vl["valid"])
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
        "label": f"{model_name} — {METHOD_NAME}",
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


def main():
    p = argparse.ArgumentParser(description=f"Dwipada benchmark: {METHOD_NAME}")
    p.add_argument("--model", type=str, default="gemma3-1b-merged", choices=MODEL_CHOICES)
    p.add_argument("--seeds", type=int, default=20)
    p.add_argument("--topic", type=str, default=None)
    p.add_argument("--output", type=str, default=None,
                   help="Path to save results JSON (default: benchmark_{METHOD_NAME}_{model}.json)")
    args = p.parse_args()

    model, tokenizer = load_model(args.model)
    print(f"\n  Pre-computing Telugu token data...")
    telugu_ids, telugu_texts, static_mask, newline_token_id = precompute_token_data(tokenizer)
    topics = [args.topic] if args.topic else BENCHMARK_PROMPTS[:3]

    all_results = []
    total_start = time.time()
    for topic in topics:
        results = run_experiment(model, tokenizer, topic,
                       telugu_ids, telugu_texts, static_mask, newline_token_id,
                       num_seeds=args.seeds, model_name=args.model)
        all_results.extend(results)
    total_time = time.time() - total_start

    # Save results to JSON
    output_path = args.output or os.path.join(
        SCRIPT_DIR, f"benchmark_{METHOD_NAME}_{args.model}.json"
    )
    summary = _build_summary(all_results, args.model, total_time)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n  Results saved to: {output_path}")


if __name__ == "__main__":
    main()
