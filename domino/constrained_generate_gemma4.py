#!/usr/bin/env python3
"""
Constrained Dwipada Generation — Gemma 4 E4B (4-bit quantized).

Uses the same rejection-sampling pipeline as constrained_generate.py but
targets google/gemma-4-4b-it loaded in 4-bit via bitsandbytes.
No fine-tuning — base Gemma 4 with constrained decoding only.

Requirements:
    pip install bitsandbytes accelerate

Usage:
    python domino/constrained_generate_gemma4.py --seeds 5
    python domino/constrained_generate_gemma4.py --seeds 5 --unconstrained
    python domino/constrained_generate_gemma4.py --benchmark --seeds 3
"""

import argparse
import json
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

MODEL_ID = "unsloth/gemma-4-E4B-it-unsloth-bnb-4bit"

###############################################################################
# 1) TOKEN FILTER
###############################################################################


def is_valid_telugu_token(token_text):
    """Allow only Telugu script, spaces, newlines, colon."""
    for ch in token_text:
        cp = ord(ch)
        if (0x0C00 <= cp <= 0x0C7F or ch in " \n:"):
            continue
        return False
    return True


###############################################################################
# 2) STATELESS PREFIX CHECK (identical to constrained_generate.py)
###############################################################################

VALID_LINE_LENGTHS = {11, 12, 13, 14, 15}
MAX_LINE_LENGTH = 15


def _min_to_accept(slot, gana_name, sub_pos):
    if slot == SLOT_ACCEPT:
        return 0
    pool = INDRA_GANAS if slot <= 2 else SURYA_GANAS
    pattern = pool[gana_name]
    remaining = len(pattern) - sub_pos
    if slot <= 2:
        return remaining + (2 - slot) * 3 + 2
    return remaining


def _max_to_accept(slot, gana_name, sub_pos):
    if slot == SLOT_ACCEPT:
        return 0
    pool = INDRA_GANAS if slot <= 2 else SURYA_GANAS
    pattern = pool[gana_name]
    remaining = len(pattern) - sub_pos
    if slot <= 2:
        return remaining + (2 - slot) * 4 + 3
    return remaining


def _is_reachable(branches, syllable_count):
    for branch in branches:
        slot, gana_name, sub_pos, matched = branch
        if slot == SLOT_ACCEPT:
            if syllable_count in VALID_LINE_LENGTHS:
                return True
            continue
        lo = _min_to_accept(slot, gana_name, sub_pos)
        hi = _max_to_accept(slot, gana_name, sub_pos)
        total_lo = syllable_count + lo
        total_hi = syllable_count + hi
        if total_lo <= MAX_LINE_LENGTH and total_hi >= min(VALID_LINE_LENGTHS):
            return True
    return False


def analyze_text(text):
    """Run fresh FST + NFA on full text. Completely stateless."""
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

    for syl, label in clf.flush():
        branches = _advance(branches, label)
        current_markers.append(label)

    syl_count = len(current_markers)
    has_accept = any(
        b[0] == SLOT_ACCEPT and syl_count in VALID_LINE_LENGTHS
        for b in branches
    )
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


def validate_poem(text):
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
    for prefix in ["పూర్తి ద్విపద:", "ద్విపద:"]:
        if text.startswith(prefix):
            return text[len(prefix):].lstrip()
        prefix_no_colon = prefix[:-1]
        if text.startswith(prefix_no_colon):
            idx = text.index(":") + 1 if ":" in text else len(prefix_no_colon)
            return text[idx:].lstrip()
    return text


###############################################################################
# 3) PROMPT BUILDING — Gemma 4 base (no fine-tuning)
###############################################################################

SYSTEM_PROMPT = (
    "You are a Telugu and Sanskrit scholar specialising in Dwipada poetry. "
    "A Dwipada has exactly 2 lines. Each line has 3 Indra ganas + 1 Surya gana. "
    "Indra ganas: Nala(IIII), Naga(IIIU), Sala(IIUI), Bha(UII), Ra(UIU), Ta(UUI). "
    "Surya ganas: Na(III), Ha/Gala(UI). "
    "Output ONLY the 2-line poem, nothing else."
)

TOP_K = 50


def build_prompt(topic, tokenizer):
    user_prompt = (
        "క్రింది తెలుగు భావానికి అనుగుణంగా ఒక ద్విపద పద్యం రచించండి.\n\n"
        f"భావం: {topic}\n\n"
        "ద్విపద:"
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


###############################################################################
# 4) CONSTRAINED GENERATION (rejection sampling)
###############################################################################


def generate_poem(
    model, tokenizer, topic,
    max_new_tokens=150, temperature=0.7, top_p=0.9,
    seed=42, constrained=True, top_k=TOP_K,
):
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
                # ── Check current state ─────────────────────────────────
                current_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
                current_state = analyze_text(strip_poem_prefix(current_text)) if generated_ids else {
                    "alive": True, "lines_complete": 0, "has_accept": False,
                    "syllable_count": 0, "markers": [], "branches": 6,
                }
                lines_done = current_state["lines_complete"]

                if lines_done >= 2:
                    break

                # ── Force newline if line is complete ───────────────────
                if (current_state["has_accept"]
                    and current_state["syllable_count"] in VALID_LINE_LENGTHS):
                    newline_ids = tokenizer.encode("\n", add_special_tokens=False)
                    if newline_ids:
                        test_ids = generated_ids + newline_ids
                        test_text = tokenizer.decode(test_ids, skip_special_tokens=True)
                        checks += 1
                        test_result = analyze_text(strip_poem_prefix(test_text))

                        if test_result["lines_complete"] > lines_done:
                            generated_ids.extend(newline_ids)
                            lines_done = test_result["lines_complete"]
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

                # ── Fallback: line exceeded max without accept ──────────
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

                # ── Rejection sampling over top-K tokens ────────────────
                chosen_id = None
                valid_candidates = []
                accept_candidates = []

                search_k = top_k
                if current_state["syllable_count"] >= 8:
                    search_k = min(top_k * 2, len(sorted_indices))

                for k in range(min(search_k, len(sorted_indices))):
                    candidate_id = sorted_indices[k].item()

                    if sorted_probs[k].item() < 1e-8:
                        break

                    if candidate_id == tokenizer.eos_token_id:
                        if lines_done >= 2:
                            chosen_id = candidate_id
                            break
                        continue

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

                    if "\n" in candidate_text:
                        if result["lines_complete"] <= lines_done:
                            continue

                    if syl > MAX_LINE_LENGTH and not result["has_accept"]:
                        continue

                    if result["has_accept"] and syl in VALID_LINE_LENGTHS:
                        accept_candidates.append((candidate_id, result, sorted_probs[k].item()))

                    valid_candidates.append((candidate_id, result, sorted_probs[k].item()))

                    if len(accept_candidates) >= 5:
                        break
                    if len(valid_candidates) >= 8 and not accept_candidates:
                        continue

                use_candidates = accept_candidates if accept_candidates else valid_candidates

                if use_candidates:
                    cand_probs = torch.tensor(
                        [c[2] for c in use_candidates], device=model.device
                    )
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

                    if not accept_candidates and choice_idx > 0:
                        backtracks += 1
                else:
                    # Exhaustive fallback
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
                            if "\n" in fb_text and fb_result["lines_complete"] <= lines_done:
                                continue
                            if syl <= MAX_LINE_LENGTH or fb_result["has_accept"]:
                                chosen_id = fb_id
                                break

                if chosen_id is None or chosen_id == tokenizer.eos_token_id:
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
# 5) MODEL LOADING — 4-bit quantized
###############################################################################


def load_model(model_id=MODEL_ID):
    print(f"  Loading model: {model_id} (pre-quantized 4-bit)")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
    )
    model.eval()

    # Report memory usage
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1e9
        print(f"  GPU memory used: {alloc:.1f} GB")

    return model, tokenizer


###############################################################################
# 6) EXPERIMENT RUNNERS
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


def run_experiment(model, tokenizer, topic, num_seeds=5, constrained=True, label=""):
    print(f"\n{'='*72}")
    print(f"  {label}")
    print(f"  Topic: {topic}")
    print(f"  Mode: {'CONSTRAINED (rejection sampling)' if constrained else 'UNCONSTRAINED'}")
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
            status = "VALID"
        else:
            status = "INVALID"

        extra = ""
        if constrained:
            extra = f", {result['backtracks']} bt, {result['nfa_checks']} chk"

        print(f"{status} ({result['elapsed']:.1f}s, {result['tokens_generated']} tok{extra})")

        for vl in result["valid_lines"]:
            mark = "V" if vl["valid"] else "X"
            print(f"    {mark} {vl['line'][:65]}")
            if vl["valid"]:
                print(f"      {vl['partition']}")

    print(f"\n  {'='*60}")
    print(f"  SUMMARY: {valid_count}/{num_seeds} fully valid ({valid_count/num_seeds*100:.0f}%)")
    if results:
        avg_time = sum(r["elapsed"] for r in results) / len(results)
        print(f"  Avg time: {avg_time:.1f}s")
        if constrained:
            avg_bt = sum(r["backtracks"] for r in results) / len(results)
            avg_chk = sum(r["nfa_checks"] for r in results) / len(results)
            print(f"  Avg backtracks: {avg_bt:.1f}, Avg checks: {avg_chk:.0f}")

    return results


def run_benchmark(model, tokenizer, prompts, seeds_per_prompt=5,
                  constrained=True, label=""):
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
        print(f"\n  [{pi+1}/{len(prompts)}] {topic[:50]}...")

        for si in range(seeds_per_prompt):
            seed = 42 + si * 7
            result = generate_poem(
                model, tokenizer, topic,
                seed=seed, constrained=constrained,
                temperature=0.7, top_p=0.9, max_new_tokens=150,
            )
            all_results.append(result)
            total_poems += 1

            if result["all_valid"]:
                prompt_valid += 1
                total_valid += 1

            for vl in result["valid_lines"]:
                total_lines += 1
                if vl["valid"]:
                    valid_lines += 1

            status = "V" if result["all_valid"] else "X"
            bt = result.get("backtracks", 0)
            print(f"    Seed {seed}: {status} ({result['elapsed']:.1f}s, {bt} bt)", end="")
            if result["all_valid"]:
                print(f" -- {result['poem_lines'][0][:40]}...")
            else:
                print()

        prompt_rate = prompt_valid / seeds_per_prompt * 100
        per_prompt_stats.append({
            "topic": topic,
            "valid": prompt_valid,
            "total": seeds_per_prompt,
            "rate": prompt_rate,
        })
        print(f"    -> {prompt_valid}/{seeds_per_prompt} valid ({prompt_rate:.0f}%)")

    total_elapsed = time.time() - start_time

    print(f"\n{'='*72}")
    print(f"  BENCHMARK RESULTS: {label}")
    print(f"{'='*72}")
    print(f"\n  Per-prompt accuracy:")
    for ps in per_prompt_stats:
        pct = int(ps["rate"] / 10)
        bar = "#" * pct + "." * (10 - pct)
        print(f"    [{bar}] {ps['rate']:5.1f}%  {ps['topic'][:45]}")

    print(f"\n  {'='*60}")
    print(f"  POEM-LEVEL ACCURACY:  {total_valid}/{total_poems} ({total_valid/total_poems*100:.1f}%)")
    print(f"  LINE-LEVEL ACCURACY:  {valid_lines}/{total_lines} ({valid_lines/total_lines*100:.1f}%)")
    print(f"  Total time:           {total_elapsed:.1f}s")
    print(f"  Avg time per poem:    {total_elapsed/total_poems:.1f}s")

    if constrained and all_results:
        avg_bt = sum(r.get("backtracks", 0) for r in all_results) / len(all_results)
        avg_chk = sum(r.get("nfa_checks", 0) for r in all_results) / len(all_results)
        print(f"  Avg backtracks:       {avg_bt:.1f}")
        print(f"  Avg NFA checks:       {avg_chk:.0f}")

    # Save results
    mode_tag = "constrained" if constrained else "unconstrained"
    output_path = os.path.join(SCRIPT_DIR, f"benchmark_gemma4_{mode_tag}.json")
    save_data = {
        "model": MODEL_ID,
        "quantization": "4-bit NF4",
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
        json.dump(save_data, f, ensure_ascii=False, indent=2)
    print(f"\n  Results saved: {output_path}")

    return all_results


###############################################################################
# 7) MAIN
###############################################################################


def main():
    global TOP_K

    p = argparse.ArgumentParser(
        description="Constrained Dwipada Generation — Gemma 4 E4B (4-bit)")
    p.add_argument("--model", type=str, default=MODEL_ID,
                   help=f"HuggingFace model ID (default: {MODEL_ID})")
    p.add_argument("--unconstrained", action="store_true",
                   help="Run unconstrained generation (no NFA)")
    p.add_argument("--benchmark", action="store_true",
                   help="Run benchmark: 10 prompts x N seeds")
    p.add_argument("--topic", type=str,
                   default="తల్లి ప్రేమ అన్నింటికంటే గొప్పది",
                   help="Telugu topic for single-topic mode")
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--top-k", type=int, default=50)
    args = p.parse_args()

    TOP_K = args.top_k

    print("=" * 72)
    print("Constrained Dwipada Generation — Gemma 4 E4B (4-bit NF4)")
    print(f"  Model: {args.model}")
    print(f"  Top-K: {TOP_K}, Valid line lengths: {sorted(VALID_LINE_LENGTHS)}")
    print("=" * 72)

    model, tokenizer = load_model(args.model)

    if args.benchmark:
        run_benchmark(model, tokenizer, BENCHMARK_PROMPTS,
                     seeds_per_prompt=args.seeds, constrained=True,
                     label="GEMMA-4-E4B — CONSTRAINED")
        if args.unconstrained:
            run_benchmark(model, tokenizer, BENCHMARK_PROMPTS,
                         seeds_per_prompt=args.seeds, constrained=False,
                         label="GEMMA-4-E4B — UNCONSTRAINED")
    else:
        run_experiment(model, tokenizer, args.topic,
                      num_seeds=args.seeds, constrained=True,
                      label="GEMMA-4-E4B — CONSTRAINED")
        if args.unconstrained:
            run_experiment(model, tokenizer, args.topic,
                          num_seeds=args.seeds, constrained=False,
                          label="GEMMA-4-E4B — UNCONSTRAINED")


if __name__ == "__main__":
    main()
