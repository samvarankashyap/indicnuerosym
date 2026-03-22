#!/usr/bin/env python3
"""
Constrained Dwipada Generation — Stateless Re-syllabification.

Algorithm:
  1. Model produces token probabilities
  2. Pick most probable token
  3. Decode FULL text so far + candidate → fresh FST + NFA check
  4. Path exists? → accept. Dead? → try next most probable.
  5. When syllable count reaches a valid line length (11-15) AND NFA has
     ACCEPT branch → force a newline to complete the line.
  6. Stop when 2 lines are complete.

Key insight: dwipada lines are 11-15 syllables. We check for ACCEPT at
each of these counts and force line completion when possible.

Usage:
    python domino/constrained_generate.py --finetuned --seeds 10
    python domino/constrained_generate.py --finetuned --seeds 10 --unconstrained
"""

import argparse
import os
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
NFA_DIR = os.path.join(PROJECT_DIR, "nfa_for_dwipada")
sys.path.insert(0, NFA_DIR)

from syllable_assembler import SyllableAssembler
from guru_laghu_classifier import GuruLaghuClassifier
from gana_nfa import (
    GanaNFA, _advance, _spawn_slot, SLOT_ACCEPT, format_partition_str,
    INDRA_GANAS, SURYA_GANAS,
)


def is_valid_telugu_token(token_text):
    """Check if a token contains only Telugu characters and minimal formatting.

    Allows: Telugu script, spaces, newlines, colon (model outputs "ద్విపద:").
    Rejects: Arabic, Latin, CJK, Devanagari, digits, other scripts.
    """
    for ch in token_text:
        cp = ord(ch)
        if (0x0C00 <= cp <= 0x0C7F   # Telugu block
            or ch in " \n:"):         # space, newline, colon
            continue
        return False
    return True


###############################################################################
# 1) STATELESS PREFIX CHECK
###############################################################################

# Valid dwipada line lengths: 3 Indra (3-4 syl each) + 1 Surya (2-3 syl) = 11-15
VALID_LINE_LENGTHS = {11, 12, 13, 14, 15}
MAX_LINE_LENGTH = 15


def _min_to_accept(slot, gana_name, sub_pos):
    """Minimum syllables needed from this branch state to reach ACCEPT."""
    if slot == SLOT_ACCEPT:
        return 0
    pool = INDRA_GANAS if slot <= 2 else SURYA_GANAS
    pattern = pool[gana_name]
    remaining = len(pattern) - sub_pos
    if slot <= 2:
        # Need (2 - slot) more Indra ganas (min 3 each) + 1 Surya (min 2)
        return remaining + (2 - slot) * 3 + 2
    return remaining


def _max_to_accept(slot, gana_name, sub_pos):
    """Maximum syllables from this branch state to reach ACCEPT."""
    if slot == SLOT_ACCEPT:
        return 0
    pool = INDRA_GANAS if slot <= 2 else SURYA_GANAS
    pattern = pool[gana_name]
    remaining = len(pattern) - sub_pos
    if slot <= 2:
        # Need (2 - slot) more Indra (max 4 each) + 1 Surya (max 3)
        return remaining + (2 - slot) * 4 + 3
    return remaining


def _is_reachable(branches, syllable_count):
    """Check if ANY branch can reach ACCEPT within the remaining syllable budget.

    A branch is reachable if:
        syllable_count + min_to_accept <= MAX_LINE_LENGTH
    AND:
        syllable_count + max_to_accept >= min(VALID_LINE_LENGTHS)

    In other words: the branch needs at least `min_to_accept` more syllables,
    and the total must land within 11-15.
    """
    for branch in branches:
        slot, gana_name, sub_pos, matched = branch
        if slot == SLOT_ACCEPT:
            if syllable_count in VALID_LINE_LENGTHS:
                return True
            continue

        lo = _min_to_accept(slot, gana_name, sub_pos)
        hi = _max_to_accept(slot, gana_name, sub_pos)

        # Total syllable count if this branch reaches accept
        total_lo = syllable_count + lo
        total_hi = syllable_count + hi

        # Does the range [total_lo, total_hi] overlap with [11, 15]?
        if total_lo <= MAX_LINE_LENGTH and total_hi >= min(VALID_LINE_LENGTHS):
            return True

    return False


def analyze_text(text):
    """Run fresh FST + NFA on full text. Completely stateless.

    Includes REACHABILITY check: not just "branches alive?" but
    "can any branch reach ACCEPT within 11-15 total syllables?"

    Returns:
        alive: at least one branch can reach ACCEPT within budget
        lines_complete: how many lines reached ACCEPT
        has_accept: current line has an ACCEPT branch at valid length
        branches: number of active branches
        syllable_count: syllables in current line
        markers: U/I markers for current line
        raw_branches: the actual branch set (for completion zone logic)
    """
    asm = SyllableAssembler()
    clf = GuruLaghuClassifier()
    raw_items = asm.process(text)

    lines_complete = 0
    branches = _spawn_slot(0, ())
    current_markers = []

    for item in raw_items:
        if item == "\n":
            for syl, label in clf.flush():
                branches = _advance(branches, label)
                current_markers.append(label)
            if any(b[0] == SLOT_ACCEPT for b in branches):
                lines_complete += 1
            branches = _spawn_slot(0, ())
            current_markers = []
        elif item == " ":
            for syl, label in clf._on_boundary():
                branches = _advance(branches, label)
                current_markers.append(label)
        else:
            for syl, label in clf._on_syllable(item):
                branches = _advance(branches, label)
                current_markers.append(label)

    # Flush remaining
    for syl, label in clf.flush():
        branches = _advance(branches, label)
        current_markers.append(label)

    syl_count = len(current_markers)
    has_accept = any(
        b[0] == SLOT_ACCEPT and syl_count in VALID_LINE_LENGTHS
        for b in branches
    )

    # Reachability: can any branch reach ACCEPT within [11, 15] total syllables?
    reachable = _is_reachable(branches, syl_count)

    return {
        "alive": reachable or lines_complete >= 2,
        "lines_complete": lines_complete,
        "has_accept": has_accept,
        "branches": len(branches),
        "syllable_count": syl_count,
        "markers": current_markers,
        "raw_branches": branches,
    }


def is_in_completion_zone(result):
    """Check if we're close to completing a line (in Surya gana slot).

    When branches are in slot 3 (Surya) and need only 1-3 more syllables,
    we're in the "completion zone". Tokens must produce the EXACT U/I
    pattern the branch needs, otherwise the line will fail.
    """
    for branch in result["raw_branches"]:
        slot, gana_name, sub_pos, matched = branch
        if slot == 3:
            # In Surya slot — need very specific remaining symbols
            return True
        if slot == SLOT_ACCEPT:
            return True
    return False


def get_required_next_symbols(branches):
    """Get the set of U/I symbols that at least one branch accepts next.

    Used in the completion zone to be strict about which tokens are valid.
    """
    accepted = set()
    for branch in branches:
        slot, gana_name, sub_pos, matched = branch
        if slot == SLOT_ACCEPT:
            continue
        pool = INDRA_GANAS if slot <= 2 else SURYA_GANAS
        pattern = pool[gana_name]
        if sub_pos < len(pattern):
            accepted.add(pattern[sub_pos])
    return accepted


def validate_poem(text):
    """Validate a complete poem. Returns per-line results."""
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    results = []
    for line in lines[:2]:
        asm = SyllableAssembler()
        clf = GuruLaghuClassifier()
        nfa = GanaNFA()
        syls = asm.process(line)
        labels = clf.process(syls)
        markers = [l for _, l in labels]
        result = nfa.process(markers)
        partition = result[0] if result else None
        results.append({
            "line": line,
            "markers": " ".join(markers),
            "valid": partition is not None,
            "partition": format_partition_str(partition) if partition else "INVALID",
        })
    return results


def strip_poem_prefix(text):
    """Strip 'ద్విపద:' or 'పూర్తి ద్విపద:' prefix from generated text.

    The model outputs 'ద్విపద: <line1>\\n<line2>' — the prefix must be
    removed before NFA analysis, otherwise its syllables (ద్వి, ప, ద)
    corrupt the gana count for line 1.
    """
    for prefix in ["పూర్తి ద్విపద:", "ద్విపద:"]:
        if text.startswith(prefix):
            return text[len(prefix):].lstrip()
        # Also handle with space before colon
        prefix_no_colon = prefix[:-1]
        if text.startswith(prefix_no_colon):
            idx = text.index(":") + 1 if ":" in text else len(prefix_no_colon)
            return text[idx:].lstrip()
    return text


###############################################################################
# 2) CONSTRAINED GENERATION
###############################################################################

# IFT training prompt — the fine-tuned model expects this exact format
SYSTEM_PROMPT = "You are a Telugu and Sanskrit scholar specialising in Dwipada poetry."

TOP_K = 50


def build_prompt(topic, tokenizer):
    user_prompt = (
        "క్రింది తెలుగు భావానికి అనుగుణంగా ఒక ద్విపద పద్యం రచించండి. "
        "ప్రతి పాదంలో 3 ఇంద్ర గణాలు + 1 సూర్య గణం ఉండాలి.\n\n"
        f"తెలుగు భావం: {topic}"
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def generate_poem(
    model, tokenizer, topic,
    max_new_tokens=150, temperature=0.7, top_p=0.9,
    seed=42, constrained=True, top_k=TOP_K,
):
    """Generate a dwipada poem with stateless constrained decoding."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    input_text = build_prompt(topic, tokenizer)
    input_ids = tokenizer(input_text, return_tensors="pt").to(model.device)

    generated_ids = []
    start_time = time.time()
    backtracks = 0
    checks = 0
    lines_done = 0

    with torch.no_grad():
        for step in range(max_new_tokens):
            if constrained and lines_done >= 2:
                break

            outputs = model(**input_ids)
            logits = outputs.logits[:, -1, :]

            if temperature > 0:
                scaled = logits / temperature
            else:
                scaled = logits

            probs = torch.softmax(scaled, dim=-1)
            sorted_probs, sorted_indices = torch.sort(probs[0], descending=True)

            if not constrained:
                # Unconstrained: sample with top-p
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
            else:
                # ── Check current state before picking next token ────────
                current_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
                current_state = analyze_text(strip_poem_prefix(current_text)) if generated_ids else {
                    "alive": True, "lines_complete": 0, "has_accept": False,
                    "syllable_count": 0, "markers": [], "branches": 6,
                }
                lines_done = current_state["lines_complete"]

                if lines_done >= 2:
                    break

                # ── If current line has ACCEPT and is at valid length,
                #    force a newline to complete the line ──────────────────
                if (current_state["has_accept"]
                    and current_state["syllable_count"] in VALID_LINE_LENGTHS):
                    # Force newline
                    newline_ids = tokenizer.encode("\n", add_special_tokens=False)
                    if newline_ids:
                        # Verify the newline keeps things valid
                        test_ids = generated_ids + newline_ids
                        test_text = tokenizer.decode(test_ids, skip_special_tokens=True)
                        checks += 1
                        test_result = analyze_text(strip_poem_prefix(test_text))

                        if test_result["lines_complete"] > lines_done:
                            # Line completed successfully
                            generated_ids.extend(newline_ids)
                            lines_done = test_result["lines_complete"]

                            # Advance model input for each newline token
                            for nid in newline_ids:
                                next_input = torch.tensor([[nid]], device=model.device)
                                input_ids = {
                                    "input_ids": torch.cat([input_ids["input_ids"], next_input], dim=1),
                                    "attention_mask": torch.cat([
                                        input_ids["attention_mask"],
                                        torch.ones((1, 1), dtype=torch.long, device=model.device)
                                    ], dim=1),
                                }
                            continue  # skip normal token selection this step

                # ── If no branch can reach ACCEPT (reachability failed),
                #    the line is doomed. Force newline and start fresh.
                #    This is the fallback — shouldn't happen often with
                #    proper reachability checking at each step.
                if (current_state["syllable_count"] > MAX_LINE_LENGTH
                    and not current_state["has_accept"]):
                    newline_ids = tokenizer.encode("\n", add_special_tokens=False)
                    if newline_ids:
                        generated_ids.extend(newline_ids)
                        for nid in newline_ids:
                            next_input = torch.tensor([[nid]], device=model.device)
                            input_ids = {
                                "input_ids": torch.cat([input_ids["input_ids"], next_input], dim=1),
                                "attention_mask": torch.cat([
                                    input_ids["attention_mask"],
                                    torch.ones((1, 1), dtype=torch.long, device=model.device)
                                ], dim=1),
                            }
                        continue

                # ── Normal token selection with constraint checking ──────
                chosen_id = None
                valid_candidates = []
                accept_candidates = []  # candidates that reach has_accept

                # Expand search range when close to completion
                search_k = top_k
                if current_state["syllable_count"] >= 8:
                    search_k = min(top_k * 2, len(sorted_indices))  # search wider

                for k in range(min(search_k, len(sorted_indices))):
                    candidate_id = sorted_indices[k].item()

                    if sorted_probs[k].item() < 1e-8:
                        break

                    if candidate_id == tokenizer.eos_token_id:
                        if lines_done >= 2:
                            chosen_id = candidate_id
                            break
                        continue

                    # Reject non-Telugu tokens (Arabic, Latin, etc.)
                    candidate_text = tokenizer.decode([candidate_id])
                    if not is_valid_telugu_token(candidate_text):
                        continue

                    test_ids = generated_ids + [candidate_id]
                    test_text = tokenizer.decode(test_ids, skip_special_tokens=True)
                    checks += 1
                    result = analyze_text(strip_poem_prefix(test_text))

                    if not result["alive"]:
                        continue

                    syl = result["syllable_count"]

                    # Check if this candidate TOKEN itself introduces a newline.
                    # If so, the line before it must have reached ACCEPT.
                    candidate_text = tokenizer.decode([candidate_id])
                    if "\n" in candidate_text:
                        if result["lines_complete"] <= lines_done:
                            continue  # token has newline but line isn't complete — reject

                    # Past max line length without accept — dead end
                    if syl > MAX_LINE_LENGTH and not result["has_accept"]:
                        continue

                    # If this candidate achieves has_accept at valid length,
                    # it's a HIGH-PRIORITY candidate — the line can complete here
                    if result["has_accept"] and syl in VALID_LINE_LENGTHS:
                        accept_candidates.append((candidate_id, result, sorted_probs[k].item()))

                    valid_candidates.append((candidate_id, result, sorted_probs[k].item()))

                    # If we have accept candidates, that's enough
                    if len(accept_candidates) >= 5:
                        break

                    # Otherwise collect more
                    if len(valid_candidates) >= 8 and not accept_candidates:
                        # Keep searching for accept candidates up to search_k
                        continue

                # Prefer accept candidates if available
                use_candidates = accept_candidates if accept_candidates else valid_candidates

                if use_candidates:
                    # Sample from candidates weighted by probability
                    cand_probs = torch.tensor(
                        [c[2] for c in use_candidates], device=model.device
                    )
                    # Top-p filter
                    cand_sorted, cand_order = torch.sort(cand_probs, descending=True)
                    cand_cumsum = torch.cumsum(cand_sorted, dim=0)
                    cand_remove = cand_cumsum - cand_sorted > top_p
                    cand_sorted[cand_remove] = 0
                    if cand_sorted.sum() > 0:
                        cand_sorted = cand_sorted / cand_sorted.sum()
                    else:
                        cand_sorted = torch.ones_like(cand_sorted) / len(cand_sorted)

                    choice_idx = cand_order[torch.multinomial(cand_sorted, 1)].item()
                    chosen_id = use_candidates[choice_idx][0]

                    if accept_candidates and use_candidates == accept_candidates:
                        backtracks += 0
                    elif choice_idx > 0:
                        backtracks += 1
                else:
                    # No valid candidate in top-K — exhaustive search over ALL vocab
                    # This is the safety net that prevents invalid poems.
                    backtracks += 1
                    for k in range(min(len(sorted_indices), 500)):
                        fb_id = sorted_indices[k].item()
                        if fb_id == tokenizer.eos_token_id:
                            continue
                        fb_text = tokenizer.decode([fb_id])
                        if not is_valid_telugu_token(fb_text):
                            continue
                        test_ids = generated_ids + [fb_id]
                        test_text = tokenizer.decode(test_ids, skip_special_tokens=True)
                        checks += 1
                        fb_result = analyze_text(strip_poem_prefix(test_text))
                        if fb_result["alive"]:
                            syl = fb_result["syllable_count"]
                            # Reject newlines that don't complete a line
                            if "\n" in fb_text and fb_result["lines_complete"] <= lines_done:
                                continue
                            if syl <= MAX_LINE_LENGTH or fb_result["has_accept"]:
                                chosen_id = fb_id
                                break

                if chosen_id is None or chosen_id == tokenizer.eos_token_id:
                    # Truly exhausted — should be extremely rare
                    break

                generated_ids.append(chosen_id)

            # Advance model input
            if generated_ids:
                last_id = generated_ids[-1]
                next_input = torch.tensor([[last_id]], device=model.device)
                input_ids = {
                    "input_ids": torch.cat([input_ids["input_ids"], next_input], dim=1),
                    "attention_mask": torch.cat([
                        input_ids["attention_mask"],
                        torch.ones((1, 1), dtype=torch.long, device=model.device)
                    ], dim=1),
                }

    elapsed = time.time() - start_time
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    # Extract poem lines — handle "ద్విపద: <line1>" format
    raw_lines = [l.strip() for l in generated_text.split("\n") if l.strip()]
    poem_lines = []
    for l in raw_lines:
        # Strip "ద్విపద:" or "పూర్తి ద్విపద:" prefix if on same line as poem text
        if l.startswith("ద్విపద:") or l.startswith("పూర్తి ద్విపద:"):
            after_colon = l.split(":", 1)[1].strip()
            if after_colon:
                poem_lines.append(after_colon)
            continue
        # Skip standalone labels
        if l.endswith(":") or l == "ద్విపద" or l == "పూర్తి ద్విపద":
            continue
        poem_lines.append(l)
    poem_lines = poem_lines[:2]

    valid_lines = validate_poem("\n".join(poem_lines))

    return {
        "topic": topic,
        "seed": seed,
        "constrained": constrained,
        "generated_text": generated_text,
        "poem_lines": poem_lines,
        "valid_lines": valid_lines,
        "all_valid": len(valid_lines) == 2 and all(v["valid"] for v in valid_lines),
        "elapsed": elapsed,
        "tokens_generated": len(generated_ids),
        "backtracks": backtracks,
        "nfa_checks": checks,
    }


###############################################################################
# 3) EXPERIMENT RUNNER
###############################################################################


def load_model(model_path, device="auto"):
    print(f"  Loading model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        attn_implementation="sdpa",
    )
    model.eval()
    return model, tokenizer


def run_experiment(model, tokenizer, topic, num_seeds=10, constrained=True, label=""):
    print(f"\n{'='*72}")
    print(f"  {label}")
    print(f"  Topic: {topic}")
    print(f"  Mode: {'CONSTRAINED (stateless + line-length aware)' if constrained else 'UNCONSTRAINED'}")
    print(f"  Seeds: {num_seeds}")
    print(f"{'='*72}")

    results = []
    valid_count = 0

    for i in range(num_seeds):
        seed = 42 + i * 7
        print(f"\n  Seed {seed}:", end=" ", flush=True)

        result = generate_poem(
            model, tokenizer, topic,
            seed=seed, constrained=constrained,
            temperature=0.7, top_p=0.9, max_new_tokens=150,
        )
        results.append(result)

        if result["all_valid"]:
            valid_count += 1
            status = "✓ VALID"
        else:
            status = "✗ INVALID"

        extra = ""
        if constrained:
            extra = f", {result['backtracks']} bt, {result['nfa_checks']} chk"

        print(f"{status} ({result['elapsed']:.1f}s, {result['tokens_generated']} tok{extra})")

        for vl in result["valid_lines"]:
            mark = "✓" if vl["valid"] else "✗"
            print(f"    {mark} {vl['line'][:65]}")
            if vl["valid"]:
                print(f"      {vl['partition']}")

    print(f"\n  {'─'*60}")
    print(f"  SUMMARY: {valid_count}/{num_seeds} fully valid ({valid_count/num_seeds*100:.0f}%)")
    if results:
        avg_time = sum(r["elapsed"] for r in results) / len(results)
        print(f"  Avg time: {avg_time:.1f}s")
        if constrained:
            avg_bt = sum(r["backtracks"] for r in results) / len(results)
            avg_chk = sum(r["nfa_checks"] for r in results) / len(results)
            print(f"  Avg backtracks: {avg_bt:.1f}, Avg checks: {avg_chk:.0f}")

    return results


###############################################################################
# 4) BENCHMARK — 10 PROMPTS × MULTIPLE SEEDS
###############################################################################

BENCHMARK_PROMPTS = [
    "తల్లి ప్రేమ అన్నింటికంటే గొప్పది",
    "హనుమంతుడు సముద్రమును దాటి లంకను చేరెను",
    "శ్రీరాముడు సీతాదేవిని రక్షించుటకు లంకకు వెళ్ళెను",
    "గురువు శిష్యునకు విద్యను బోధించెను",
    "సూర్యోదయమున ప్రకృతి అందముగా వెలుగును",
    "కృష్ణుడు అర్జునునకు గీతోపదేశము చేసెను",
    "నదులు పర్వతముల నుండి సముద్రమునకు ప్రవహించును",
    "విజయమునకు కఠోర పరిశ్రమ అవసరము",
    "రావణుడు సీతను అపహరించి లంకకు తీసుకొనిపోయెను",
    "భగవంతుడు సర్వ ప్రాణులను రక్షించును",
]


def run_benchmark(model, tokenizer, prompts, seeds_per_prompt=5,
                  constrained=True, label=""):
    """Run multiple prompts × multiple seeds and produce aggregate stats."""
    import json as json_mod

    print(f"\n{'='*72}")
    print(f"  BENCHMARK: {label}")
    print(f"  Prompts: {len(prompts)}, Seeds per prompt: {seeds_per_prompt}")
    print(f"  Total poems: {len(prompts) * seeds_per_prompt}")
    print(f"  Mode: {'CONSTRAINED' if constrained else 'UNCONSTRAINED'}")
    print(f"{'='*72}")

    all_results = []
    per_prompt_stats = []
    total_valid = 0
    total_poems = 0
    total_lines = 0
    valid_lines = 0
    start_time = time.time()

    for pi, topic in enumerate(prompts):
        prompt_valid = 0
        prompt_results = []

        print(f"\n  [{pi+1}/{len(prompts)}] {topic[:50]}...")

        for si in range(seeds_per_prompt):
            seed = 42 + si * 7
            result = generate_poem(
                model, tokenizer, topic,
                seed=seed, constrained=constrained,
                temperature=0.7, top_p=0.9, max_new_tokens=150,
            )
            prompt_results.append(result)
            all_results.append(result)
            total_poems += 1

            if result["all_valid"]:
                prompt_valid += 1
                total_valid += 1

            for vl in result["valid_lines"]:
                total_lines += 1
                if vl["valid"]:
                    valid_lines += 1

            # Print compact result
            status = "✓" if result["all_valid"] else "✗"
            bt = result.get("backtracks", 0)
            print(f"    Seed {seed}: {status} ({result['elapsed']:.1f}s, {bt} bt)", end="")
            if result["all_valid"]:
                print(f" — {result['poem_lines'][0][:40]}...")
            else:
                print()

        prompt_rate = prompt_valid / seeds_per_prompt * 100
        per_prompt_stats.append({
            "topic": topic,
            "valid": prompt_valid,
            "total": seeds_per_prompt,
            "rate": prompt_rate,
        })
        print(f"    → {prompt_valid}/{seeds_per_prompt} valid ({prompt_rate:.0f}%)")

    total_elapsed = time.time() - start_time

    # ── Aggregate Report ─────────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"  BENCHMARK RESULTS: {label}")
    print(f"{'='*72}")
    print(f"\n  Per-prompt accuracy:")
    for ps in per_prompt_stats:
        bar = "█" * int(ps["rate"] / 10) + "░" * (10 - int(ps["rate"] / 10))
        print(f"    {bar} {ps['rate']:5.1f}%  {ps['topic'][:45]}")

    print(f"\n  {'─'*60}")
    print(f"  POEM-LEVEL ACCURACY:  {total_valid}/{total_poems} ({total_valid/total_poems*100:.1f}%)")
    print(f"  LINE-LEVEL ACCURACY:  {valid_lines}/{total_lines} ({valid_lines/total_lines*100:.1f}%)")
    print(f"  Total time:           {total_elapsed:.1f}s")
    print(f"  Avg time per poem:    {total_elapsed/total_poems:.1f}s")

    if constrained and all_results:
        avg_bt = sum(r.get("backtracks", 0) for r in all_results) / len(all_results)
        avg_chk = sum(r.get("nfa_checks", 0) for r in all_results) / len(all_results)
        print(f"  Avg backtracks:       {avg_bt:.1f}")
        print(f"  Avg NFA checks:       {avg_chk:.0f}")

    # Save results to JSON (separate file per mode)
    mode_tag = "constrained" if constrained else "unconstrained"
    output_path = os.path.join(SCRIPT_DIR, f"benchmark_{mode_tag}.json")
    save_data = {
        "label": label,
        "constrained": constrained,
        "total_poems": total_poems,
        "total_valid": total_valid,
        "poem_accuracy": total_valid / total_poems * 100,
        "total_lines": total_lines,
        "valid_lines": valid_lines,
        "line_accuracy": valid_lines / total_lines * 100,
        "total_time": total_elapsed,
        "per_prompt": per_prompt_stats,
        "poems": [
            {
                "topic": r["topic"],
                "seed": r["seed"],
                "all_valid": r["all_valid"],
                "lines": [
                    {"text": vl["line"], "valid": vl["valid"],
                     "markers": vl["markers"], "partition": vl["partition"]}
                    for vl in r["valid_lines"]
                ],
                "elapsed": r["elapsed"],
            }
            for r in all_results
        ],
    }
    with open(output_path, "w") as f:
        json_mod.dump(save_data, f, ensure_ascii=False, indent=2)
    print(f"\n  Results saved: {output_path}")

    return all_results


###############################################################################
# 5) MAIN
###############################################################################


def main():
    global TOP_K

    p = argparse.ArgumentParser(description="Constrained Dwipada Generation")
    p.add_argument("--base", action="store_true", help="Use base Gemma model")
    p.add_argument("--finetuned", action="store_true", help="Use fine-tuned model")
    p.add_argument("--both", action="store_true", help="Run both models")
    p.add_argument("--unconstrained", action="store_true", help="Also run unconstrained")
    p.add_argument("--benchmark", action="store_true",
                   help="Run benchmark: 10 prompts × 5 seeds = 50 poems")
    p.add_argument("--topic", type=str,
                   default="తల్లి ప్రేమ అన్నింటికంటే గొప్పది",
                   help="Telugu topic for generation (single-topic mode)")
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--top-k", type=int, default=50)
    args = p.parse_args()

    TOP_K = args.top_k

    if not args.base and not args.finetuned and not args.both:
        args.both = True

    print("=" * 72)
    print("Constrained Dwipada Generation")
    print("  Stateless re-syllabification + line-length aware + backtracking")
    print(f"  Top-K: {TOP_K}, Valid line lengths: {sorted(VALID_LINE_LENGTHS)}")
    if args.benchmark:
        print(f"  Mode: BENCHMARK ({len(BENCHMARK_PROMPTS)} prompts × {args.seeds} seeds)")
    print("=" * 72)

    if args.benchmark:
        # ── Benchmark mode: 10 prompts × N seeds ────────────────────
        if args.base or args.both:
            model, tokenizer = load_model("google/gemma-3-1b-it")

            run_benchmark(model, tokenizer, BENCHMARK_PROMPTS,
                         seeds_per_prompt=args.seeds, constrained=True,
                         label="BASE — CONSTRAINED")

            if args.unconstrained:
                run_benchmark(model, tokenizer, BENCHMARK_PROMPTS,
                             seeds_per_prompt=args.seeds, constrained=False,
                             label="BASE — UNCONSTRAINED")

            del model
            torch.cuda.empty_cache()

        if args.finetuned or args.both:
            merged_path = os.path.join(PROJECT_DIR, "train_models", "dwipada_merged_model")
            if os.path.exists(merged_path):
                model, tokenizer = load_model(merged_path)
            else:
                from peft import PeftModel
                model, tokenizer = load_model("google/gemma-3-1b-it")
                adapter_path = os.path.join(PROJECT_DIR, "train_models", "dwipada_lora_adapter")
                model = PeftModel.from_pretrained(model, adapter_path)
                model.eval()

            run_benchmark(model, tokenizer, BENCHMARK_PROMPTS,
                         seeds_per_prompt=args.seeds, constrained=True,
                         label="FINE-TUNED — CONSTRAINED")

            if args.unconstrained:
                run_benchmark(model, tokenizer, BENCHMARK_PROMPTS,
                             seeds_per_prompt=args.seeds, constrained=False,
                             label="FINE-TUNED — UNCONSTRAINED")
    else:
        # ── Single-topic mode ────────────────────────────────────────
        topic = args.topic

        if args.base or args.both:
            model, tokenizer = load_model("google/gemma-3-1b-it")

            run_experiment(model, tokenizer, topic,
                          num_seeds=args.seeds, constrained=True,
                          label="BASE MODEL — CONSTRAINED")

            if args.unconstrained:
                run_experiment(model, tokenizer, topic,
                              num_seeds=args.seeds, constrained=False,
                              label="BASE MODEL — UNCONSTRAINED")

            del model
            torch.cuda.empty_cache()

        if args.finetuned or args.both:
            merged_path = os.path.join(PROJECT_DIR, "train_models", "dwipada_merged_model")
            if os.path.exists(merged_path):
                model, tokenizer = load_model(merged_path)
            else:
                from peft import PeftModel
                model, tokenizer = load_model("google/gemma-3-1b-it")
                adapter_path = os.path.join(PROJECT_DIR, "train_models", "dwipada_lora_adapter")
                model = PeftModel.from_pretrained(model, adapter_path)
                model.eval()

            run_experiment(model, tokenizer, topic,
                          num_seeds=args.seeds, constrained=True,
                          label="FINE-TUNED MODEL — CONSTRAINED")

            if args.unconstrained:
                run_experiment(model, tokenizer, topic,
                              num_seeds=args.seeds, constrained=False,
                              label="FINE-TUNED MODEL — UNCONSTRAINED")

    print("\n" + "=" * 72)
    print("Done.")


if __name__ == "__main__":
    main()
