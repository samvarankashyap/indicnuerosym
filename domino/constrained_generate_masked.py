#!/usr/bin/env python3
"""
Constrained Dwipada Generation — Logit Masking (Gana NFA) with Backtracking.

Alternative to the rejection-sampling approach in constrained_generate.py.
Pre-computes a boolean mask over ALL Telugu tokens and applies it to logits
before sampling. When the mask produces too few valid tokens (dead end),
backtracks to a saved checkpoint with bumped temperature and retries.

Properties:
  - Complete coverage (no missed valid tokens via masking)
  - Backtracking on dead ends (recovers from bad sampling paths)
  - Constant per-step cost (independent of poem length)

Usage:
    python domino/constrained_generate_masked.py --finetuned --seeds 10

The existing rejection-sampling code (constrained_generate.py) is untouched.
"""

import argparse
import os
import sys
import time

import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Gemma3ForCausalLM,
    Gemma3ForConditionalGeneration,
)

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
    analyze_text, build_prompt, validate_poem, strip_poem_prefix,
    BENCHMARK_PROMPTS, VALID_LINE_LENGTHS as CG_VALID_LINE_LENGTHS,
)
from constrained_generate_v2 import MODEL_CHOICES, load_model


###############################################################################
# 1) PRE-COMPUTATION
###############################################################################


def precompute_token_data(tokenizer):
    """
    Build the static data structures needed for logit masking.

    Returns:
        telugu_ids: list of int — Telugu token IDs (~2,346)
        telugu_texts: list of str — decoded text for each Telugu token
        static_mask: Tensor [vocab_size] — 0.0 for Telugu+EOS, -inf elsewhere
        newline_token_id: int — the newline token ID
    """
    telugu_ids, telugu_texts = get_telugu_token_set(tokenizer)
    telugu_set = set(telugu_ids)

    vocab_size = tokenizer.vocab_size
    static_mask = torch.full((vocab_size,), float("-inf"))
    for tid in telugu_ids:
        static_mask[tid] = 0.0
    # Also allow EOS
    if tokenizer.eos_token_id is not None:
        static_mask[tokenizer.eos_token_id] = 0.0

    # Find newline token
    newline_ids = tokenizer.encode("\n", add_special_tokens=False)
    newline_token_id = newline_ids[0] if newline_ids else None

    print(f"  Telugu tokens: {len(telugu_ids)}")
    print(f"  Newline token ID: {newline_token_id}")

    return telugu_ids, telugu_texts, static_mask, newline_token_id


###############################################################################
# 2) LOGIT-MASKED GENERATION (with backtracking)
###############################################################################

# Backtracking constants
CHECKPOINT_INTERVAL = 3    # save state every 3 tokens
MIN_VALID_TOKENS = 3       # trigger backtrack below this threshold
MAX_BACKTRACKS_DEFAULT = 30
TEMP_BUMP = 0.15           # temperature increase per backtrack
TEMP_DECAY_STEPS = 6       # steps before temperature returns to base
MAX_TEMP = 1.2             # temperature ceiling
MAX_CHECKPOINTS = 15       # max checkpoints to keep in ring buffer


def generate_poem_masked(
    model, tokenizer, topic,
    telugu_ids, telugu_texts, static_mask, newline_token_id,
    max_new_tokens=150, temperature=0.7, top_p=0.9, seed=42,
    max_backtracks=MAX_BACKTRACKS_DEFAULT, checkpoint_interval=CHECKPOINT_INTERVAL,
    min_valid_tokens=MIN_VALID_TOKENS,
):
    """
    Generate a dwipada poem using logit masking (Gana NFA) with backtracking.

    When the NFA mask produces too few valid tokens (dead end), backtracks
    to a saved checkpoint, bumps temperature, and retries from there.

    Returns the same dict format as constrained_generate.generate_poem()
    for easy A/B comparison.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    input_text = build_prompt(topic, tokenizer)
    input_ids = tokenizer(input_text, return_tensors="pt").to(model.device)

    generated_ids = []
    start_time = time.time()
    lines_done = 0
    mask_computations = 0

    # Move static mask to device
    device = model.device
    device_static_mask = static_mask.to(device)

    # Composite state — tracks the incremental FST+NFA pipeline
    composite = CompositeState()

    # Track whether we've passed the "ద్విపద:" prefix
    prefix_done = False

    # Backtracking state
    # Each checkpoint: (step, composite_snap, num_generated, input_ids_len, lines_done, temp)
    checkpoints = []
    backtrack_count = 0
    consecutive_backtracks = 0
    current_temp = temperature
    temp_boost_remaining = 0
    prefix_step = 0  # step at which prefix was first detected

    with torch.no_grad():
        step = 0
        while step < max_new_tokens:
            if lines_done >= 2:
                break

            # --- Forward pass ---
            outputs = model(**input_ids)
            logits = outputs.logits[:, -1, :]

            if current_temp > 0:
                logits = logits / current_temp

            # --- Apply static mask (kill non-Telugu tokens) ---
            logits = logits + device_static_mask.unsqueeze(0)

            # --- Detect prefix completion ---
            if not prefix_done:
                current_text = tokenizer.decode(
                    generated_ids, skip_special_tokens=True
                )
                stripped = strip_poem_prefix(current_text)
                if stripped != current_text or ":" in current_text:
                    # Prefix has appeared — initialize composite state
                    # with the post-prefix text
                    prefix_done = True
                    prefix_step = step
                    composite = CompositeState()
                    for ch in stripped:
                        composite.feed_char(ch)

            # --- Save checkpoint periodically ---
            if prefix_done and step > prefix_step and (step - prefix_step) % checkpoint_interval == 0:
                checkpoints.append((
                    step,
                    composite.snapshot(),
                    len(generated_ids),
                    input_ids["input_ids"].shape[1],
                    lines_done,
                    current_temp,
                ))
                # Ring buffer: keep only recent checkpoints
                if len(checkpoints) > MAX_CHECKPOINTS:
                    checkpoints = checkpoints[-MAX_CHECKPOINTS:]

            # --- Build dynamic Gana mask ---
            if prefix_done:
                # Check if current line has ACCEPT at valid length → force newline
                # Use a flushed clone for accurate syllable count
                flushed = CompositeState.from_snapshot(composite.snapshot())
                flushed.flush()
                if flushed.has_accept():
                    if newline_token_id is not None:
                        # Force newline: mask everything except newline
                        force_mask = torch.full_like(
                            logits, float("-inf")
                        )
                        force_mask[0, newline_token_id] = 0.0
                        logits = logits + force_mask

                        # Sample (will be newline)
                        chosen_id = newline_token_id

                        # Feed newline through composite state
                        composite.feed_char("\n")
                        lines_done = composite.lines_complete

                        generated_ids.append(chosen_id)

                        # Advance model input
                        next_input = torch.tensor(
                            [[chosen_id]], device=device
                        )
                        input_ids = {
                            "input_ids": torch.cat(
                                [input_ids["input_ids"], next_input], dim=1
                            ),
                            "attention_mask": torch.cat(
                                [
                                    input_ids["attention_mask"],
                                    torch.ones(
                                        (1, 1),
                                        dtype=torch.long,
                                        device=device,
                                    ),
                                ],
                                dim=1,
                            ),
                        }
                        step += 1
                        consecutive_backtracks = 0
                        continue

                # Build mask over Telugu tokens
                snap = composite.snapshot()
                valid_ids = build_gana_mask(snap, telugu_ids, telugu_texts)
                mask_computations += 1

                # Handle newline: only valid if has_accept at valid length
                # (already handled above by forcing, but also filter here)
                if newline_token_id in valid_ids:
                    # Check if newline would complete a valid line
                    clone = CompositeState.from_snapshot(snap)
                    clone.feed_char("\n")
                    if clone.lines_complete <= lines_done:
                        valid_ids.discard(newline_token_id)

                # --- BACKTRACK if dead end detected ---
                # Trigger 1: too few valid tokens in mask
                mask_empty = len(valid_ids) < min_valid_tokens
                # Trigger 2: line overshot max length — no way to complete
                overshot = flushed.syllable_count > MAX_LINE_LENGTH
                need_backtrack = mask_empty or overshot
                if need_backtrack and checkpoints and backtrack_count < max_backtracks:
                    backtrack_count += 1
                    consecutive_backtracks += 1

                    # Escalate: pop more checkpoints on consecutive backtracks
                    pop_count = min(consecutive_backtracks, len(checkpoints))
                    for _ in range(pop_count):
                        ckpt = checkpoints.pop()

                    # Restore from checkpoint
                    ckpt_step, ckpt_snap, ckpt_gen_len, ckpt_ids_len, ckpt_lines, ckpt_temp = ckpt
                    composite = CompositeState.from_snapshot(ckpt_snap)
                    generated_ids = generated_ids[:ckpt_gen_len]
                    input_ids = {
                        "input_ids": input_ids["input_ids"][:, :ckpt_ids_len],
                        "attention_mask": input_ids["attention_mask"][:, :ckpt_ids_len],
                    }
                    lines_done = ckpt_lines
                    step = ckpt_step

                    # Bump temperature and reseed to diversify away from dead end
                    current_temp = min(ckpt_temp + TEMP_BUMP * consecutive_backtracks, MAX_TEMP)
                    temp_boost_remaining = TEMP_DECAY_STEPS
                    new_seed = seed + backtrack_count * 1337
                    torch.manual_seed(new_seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed(new_seed)
                    continue

                # Apply dynamic mask
                dynamic_mask = torch.full_like(logits, float("-inf"))
                for tid in valid_ids:
                    dynamic_mask[0, tid] = 0.0
                logits = logits + dynamic_mask

            # --- Handle lines_done >= 2: only allow EOS ---
            if lines_done >= 2:
                eos_mask = torch.full_like(logits, float("-inf"))
                if tokenizer.eos_token_id is not None:
                    eos_mask[0, tokenizer.eos_token_id] = 0.0
                logits = logits + eos_mask

            # --- Hybrid sampling: mask + rejection ---
            # Step 1: Get top-K from masked logits (sorted by probability)
            probs = torch.softmax(logits, dim=-1)
            sorted_probs, sorted_indices = torch.sort(
                probs[0], descending=True
            )

            if not prefix_done:
                # Before prefix: simple top-p sampling (no rejection needed)
                cumsum = torch.cumsum(sorted_probs, dim=0)
                remove = cumsum - sorted_probs > top_p
                sorted_probs[remove] = 0
                if sorted_probs.sum() > 0:
                    sorted_probs = sorted_probs / sorted_probs.sum()
                else:
                    sorted_probs[0] = 1.0
                choice = torch.multinomial(sorted_probs, 1)
                chosen_id = sorted_indices[choice].item()
            else:
                # Step 2: Rejection sample from top-K using incremental NFA
                # Clone composite state for each candidate, check validity
                chosen_id = None
                valid_candidates = []
                accept_candidates = []
                snap = composite.snapshot()

                search_k = 100  # search deeper than typical top-K
                for k in range(min(search_k, len(sorted_indices))):
                    if sorted_probs[k].item() < 1e-8:
                        break  # negligible probability

                    cand_id = sorted_indices[k].item()

                    if cand_id == tokenizer.eos_token_id:
                        if lines_done >= 2:
                            chosen_id = cand_id
                            break
                        continue

                    # NFA test: clone state, feed candidate token, flush, check
                    cand_text = tokenizer.decode([cand_id])
                    clone = CompositeState.from_snapshot(snap)
                    for ch in cand_text:
                        clone.feed_char(ch)
                    flushed_clone = CompositeState.from_snapshot(clone.snapshot())
                    flushed_clone.flush()

                    if not flushed_clone.is_alive():
                        continue  # dead branch

                    # Reject newlines that don't complete a valid line
                    if "\n" in cand_text:
                        if clone.lines_complete <= lines_done:
                            continue

                    # Reject if overshot without accept
                    if flushed_clone.syllable_count > MAX_LINE_LENGTH and not flushed_clone.has_accept():
                        continue

                    # Track accept candidates (tokens reaching valid line completion)
                    if flushed_clone.has_accept():
                        accept_candidates.append((cand_id, sorted_probs[k].item()))

                    valid_candidates.append((cand_id, sorted_probs[k].item()))

                    # Stop early if we have accept candidates
                    if len(accept_candidates) >= 3:
                        break
                    if len(valid_candidates) >= 10 and not accept_candidates:
                        break

                # Prefer accept candidates, then valid candidates
                use_candidates = accept_candidates if accept_candidates else valid_candidates

                if use_candidates and chosen_id is None:
                    # Weighted sample from candidates
                    cand_probs = torch.tensor(
                        [c[1] for c in use_candidates], device=device
                    )
                    cand_probs = cand_probs / cand_probs.sum()
                    choice_idx = torch.multinomial(cand_probs, 1).item()
                    chosen_id = use_candidates[choice_idx][0]
                elif chosen_id is None:
                    # No valid candidate found — backtrack
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
                        # Exhausted — force first masked token
                        chosen_id = sorted_indices[0].item()

            if chosen_id == tokenizer.eos_token_id:
                break

            generated_ids.append(chosen_id)

            # --- Advance composite state with chosen token ---
            if prefix_done:
                chosen_text = tokenizer.decode([chosen_id])
                for ch in chosen_text:
                    composite.feed_char(ch)
                lines_done = composite.lines_complete

            # --- Advance model input ---
            next_input = torch.tensor([[chosen_id]], device=device)
            input_ids = {
                "input_ids": torch.cat(
                    [input_ids["input_ids"], next_input], dim=1
                ),
                "attention_mask": torch.cat(
                    [
                        input_ids["attention_mask"],
                        torch.ones(
                            (1, 1), dtype=torch.long, device=device
                        ),
                    ],
                    dim=1,
                ),
            }

            # --- Successful token committed: reset backtrack escalation ---
            consecutive_backtracks = 0
            if temp_boost_remaining > 0:
                temp_boost_remaining -= 1
                if temp_boost_remaining == 0:
                    current_temp = temperature  # restore base temperature

            step += 1

    elapsed = time.time() - start_time
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    # Extract poem lines (same logic as constrained_generate.py)
    raw_lines = [l.strip() for l in generated_text.split("\n") if l.strip()]
    poem_lines = []
    for l in raw_lines:
        if l.startswith("ద్విపద:") or l.startswith("పూర్తి ద్విపద:"):
            after_colon = l.split(":", 1)[1].strip()
            if after_colon:
                poem_lines.append(after_colon)
            continue
        if l.endswith(":") or l == "ద్విపద" or l == "పూర్తి ద్విపద":
            continue
        poem_lines.append(l)
    poem_lines = poem_lines[:2]

    valid_lines = validate_poem("\n".join(poem_lines))

    return {
        "topic": topic,
        "seed": seed,
        "constrained": True,
        "method": "hybrid_mask_reject",
        "generated_text": generated_text,
        "poem_lines": poem_lines,
        "valid_lines": valid_lines,
        "all_valid": len(valid_lines) == 2 and all(v["valid"] for v in valid_lines),
        "elapsed": elapsed,
        "tokens_generated": len(generated_ids),
        "backtracks": backtrack_count,
        "mask_computations": mask_computations,
    }


###############################################################################
# 3) EXPERIMENT RUNNER
###############################################################################


def run_experiment_masked(
    model, tokenizer, topic,
    telugu_ids, telugu_texts, static_mask, newline_token_id,
    num_seeds=10, label="",
):
    print(f"\n{'='*72}")
    print(f"  {label}")
    print(f"  Topic: {topic}")
    print(f"  Mode: CONSTRAINED (logit masking, Gana NFA)")
    print(f"  Seeds: {num_seeds}")
    print(f"{'='*72}")

    results = []
    valid_count = 0

    for i in range(num_seeds):
        seed = 42 + i * 7
        print(f"\n  Seed {seed}:", end=" ", flush=True)

        result = generate_poem_masked(
            model, tokenizer, topic,
            telugu_ids, telugu_texts, static_mask, newline_token_id,
            seed=seed, temperature=0.7, top_p=0.9, max_new_tokens=150,
        )
        results.append(result)

        if result["all_valid"]:
            valid_count += 1
            status = "V VALID"
        else:
            status = "X INVALID"

        bt = result.get("backtracks", 0)
        bt_str = f", {bt} bt" if bt > 0 else ""
        print(
            f"{status} ({result['elapsed']:.1f}s, "
            f"{result['tokens_generated']} tok, "
            f"{result['mask_computations']} masks{bt_str})"
        )

        for vl in result["valid_lines"]:
            mark = "V" if vl["valid"] else "X"
            print(f"    {mark} {vl['line'][:65]}")
            if vl["valid"]:
                print(f"      {vl['partition']}")

    print(f"\n  {'_'*60}")
    print(f"  SUMMARY: {valid_count}/{num_seeds} fully valid "
          f"({valid_count/num_seeds*100:.0f}%)")
    if results:
        avg_time = sum(r["elapsed"] for r in results) / len(results)
        avg_masks = sum(r["mask_computations"] for r in results) / len(results)
        avg_bt = sum(r.get("backtracks", 0) for r in results) / len(results)
        print(f"  Avg time: {avg_time:.1f}s")
        print(f"  Avg mask computations: {avg_masks:.0f}")
        if avg_bt > 0:
            print(f"  Avg backtracks: {avg_bt:.1f}")

    return results


###############################################################################
# 4) CLI
###############################################################################

def main():
    parser = argparse.ArgumentParser(
        description="Constrained Dwipada generation with logit masking"
    )
    parser.add_argument(
        "--model", type=str, default="gemma3-1b-merged", choices=MODEL_CHOICES,
        help="Model to use (default: gemma3-1b-merged)",
    )
    parser.add_argument(
        "--seeds", type=int, default=5,
        help="Number of seeds per prompt",
    )
    parser.add_argument(
        "--topic", type=str, default=None,
        help="Single topic (default: use BENCHMARK_PROMPTS)",
    )
    args = parser.parse_args()

    model, tokenizer = load_model(args.model)

    print(f"\n  Pre-computing Telugu token data...")
    telugu_ids, telugu_texts, static_mask, newline_token_id = (
        precompute_token_data(tokenizer)
    )

    if args.topic:
        topics = [args.topic]
    else:
        topics = BENCHMARK_PROMPTS[:3]  # default: first 3 prompts

    for topic in topics:
        run_experiment_masked(
            model, tokenizer, topic,
            telugu_ids, telugu_texts, static_mask, newline_token_id,
            num_seeds=args.seeds,
            label=f"LOGIT MASKING — {args.model}",
        )


if __name__ == "__main__":
    main()
