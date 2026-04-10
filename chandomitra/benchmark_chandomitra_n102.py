#!/usr/bin/env python3
"""Chandomitra constrained-decoding benchmark — n=102 (3 prompts x 34 seeds).

Mirrors the n=102 setup of domino/benchmark_*.py for direct comparison
against the FST+NFA enforcer described in the paper.

Models supported (skip the 4-bit Gemma 4 E2B variant):
  - gemma3-1b-base    : google/gemma-3-1b-it (no fine-tuning, bf16)
  - gemma3-1b-merged  : ./dwipada_merged_model (LoRA-merged, bf16)

Constraint stack: chandomitra DwipadaConstrainedLogitsProcessor with all three
constraints active (gana + prasa + yati), soft mode (no hard_constraints).

Sampling: temperature=0.7, top_p=0.9, max_new_tokens=150, seeds 42 + 7k for
k in [0, 33]. Same prompts and seed schedule as domino/benchmark_*.py.

Usage:
    python benchmark_chandomitra_n102.py --model gemma3-1b-base
    python benchmark_chandomitra_n102.py --model gemma3-1b-merged
"""

import argparse
import json
import os
import sys
import time

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "src"))

from dwipada.core.analyzer import analyze_dwipada
from dwipada.training.constrained.logits_processor import DwipadaConstrainedLogitsProcessor


###############################################################################
# CONFIG
###############################################################################

# Same 3 prompts used by domino/benchmark_*.py (BENCHMARK_PROMPTS[:3]).
BENCHMARK_PROMPTS = [
    "తల్లి ప్రేమ అన్నింటికంటే గొప్పది",
    "హనుమంతుడు సముద్రమును దాటి లంకను చేరెను",
    "శ్రీరాముడు సీతాదేవిని రక్షించుటకు లంకకు వెళ్ళెను",
]

# Same prompt template as the existing chandomitra benchmark.
STRUCTURED_PROMPT = """\
### ROLE
You are an expert Telugu Scholar and Maha Kavi. You possess deep knowledge of Telugu Chandassu (Prosody). Gana boundaries do not need to align with word boundaries.

### TASK
Compose a Dwipada (ద్విపద) poem in Telugu. Output ONLY the two-line Telugu poem, nothing else.

{user_instruction}

### STRUCTURAL RULES
* Exactly 2 lines (Padas).
* Each line: Indra - Indra - Indra - Surya.
* Indra Ganas: Nala (IIII), Naga (IIIU), Sala (IIUI), Bha (UII), Ra (UIU), Ta (UUI)
* Surya Ganas: Na (III), Ha/Gala (UI)
* Guru (U): Long vowel, Anusvara (ం), Visarga (ః), before conjunct/double consonant.
* Laghu (I): Short vowel without the above.
* Prasa: 2nd syllable consonant must match between lines.
* Yati: 1st letter of Gana 1 = 1st letter of Gana 3 (Varga Maitri).

### STYLE
* సరళమైన వాడుక భాష (Simple colloquial Telugu).
* Natural flow — do not force word breaks for gana alignment.

### OUTPUT
Output ONLY the two Telugu lines of the Dwipada poem. No titles, no explanations, no markdown."""


MODEL_CHOICES = ["gemma3-1b-base", "gemma3-1b-merged"]

MODEL_CONFIGS = {
    "gemma3-1b-base": {
        "model_path": "google/gemma-3-1b-it",
        "tokenizer_path": "google/gemma-3-1b-it",
        "desc": "Gemma 3 1B IT (base, bf16)",
    },
    "gemma3-1b-merged": {
        "model_path": os.path.join(SCRIPT_DIR, "dwipada_merged_model"),
        "tokenizer_path": os.path.join(SCRIPT_DIR, "dwipada_merged_model"),
        "desc": "Gemma 3 1B IT + Dwipada LoRA merged (bf16)",
    },
}

NUM_SEEDS = 34
METHOD_NAME = "chandomitra_constrained"


###############################################################################
# MODEL LOADING
###############################################################################

def load_model(model_choice):
    cfg = MODEL_CONFIGS[model_choice]
    print(f"\n  Loading model: {cfg['desc']}")
    print(f"  Path: {cfg['model_path']}")

    tokenizer = AutoTokenizer.from_pretrained(cfg["tokenizer_path"])
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_path"],
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    model.eval()

    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1e9
        print(f"  GPU memory used: {alloc:.1f} GB")

    return model, tokenizer


###############################################################################
# PROMPT + GENERATION
###############################################################################

def build_prompt(topic, tokenizer):
    instruction = STRUCTURED_PROMPT.format(user_instruction=f"భావం: {topic}")
    messages = [{"role": "user", "content": instruction}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def validate_poem(poem_text):
    try:
        analysis = analyze_dwipada(poem_text)
        score = analysis["match_score"]["overall"]
        is_valid = analysis.get("is_valid_dwipada", False)
        breakdown = analysis.get("match_score", {}).get("breakdown", {})
        return score, is_valid, breakdown
    except Exception:
        return 0.0, False, {}


def extract_poem_lines(text):
    """Pull the first two non-empty Telugu lines out of the model output."""
    lines = []
    for raw in text.split("\n"):
        line = raw.strip()
        if not line:
            continue
        # Skip obvious meta lines (markdown headers, English fragments, etc.)
        if line.startswith("#") or line.startswith("```") or line.startswith("---"):
            continue
        # Require at least one Telugu codepoint to count as a poem line.
        if any(0x0C00 <= ord(ch) <= 0x0C7F for ch in line):
            lines.append(line)
        if len(lines) == 2:
            break
    return lines


def generate_one(model, tokenizer, topic, seed, top_k=25, max_k=200,
                 temperature=0.7, top_p=0.9, max_new_tokens=150):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    formatted = build_prompt(topic, tokenizer)
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
    prompt_len = inputs["input_ids"].shape[1]

    processor = DwipadaConstrainedLogitsProcessor(
        tokenizer=tokenizer,
        initial_k=top_k,
        max_k=max_k,
        enable_prasa=True,
        enable_yati=True,
        hard_constraints=False,
    )
    processor.prompt_length = prompt_len

    start = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            logits_processor=LogitsProcessorList([processor]),
        )
    elapsed = time.time() - start

    generated_ids = outputs[0][prompt_len:]
    tokens_generated = len(generated_ids)
    raw_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    poem_lines = extract_poem_lines(raw_text)
    poem = "\n".join(poem_lines)

    score, is_valid, breakdown = validate_poem(poem)

    return {
        "topic": topic,
        "seed": seed,
        "method": METHOD_NAME,
        "generated_text": raw_text,
        "poem_lines": poem_lines,
        "poem": poem,
        "score": score,
        "is_valid": is_valid,
        "all_valid": is_valid,  # alias for cross-benchmark consistency
        "breakdown": breakdown,
        "elapsed": elapsed,
        "tokens_generated": tokens_generated,
    }


###############################################################################
# EXPERIMENT RUNNER
###############################################################################

def run_topic(model, tokenizer, topic, num_seeds, model_name):
    print(f"\n{'='*72}")
    print(f"  {METHOD_NAME.upper()} -- {model_name}")
    print(f"  Topic: {topic}")
    print(f"  Seeds: {num_seeds}")
    print(f"{'='*72}")

    results = []
    for i in range(num_seeds):
        seed = 42 + i * 7
        print(f"\n  Seed {seed}:", end=" ", flush=True)
        r = generate_one(model, tokenizer, topic, seed=seed)
        results.append(r)
        status = "V VALID" if r["is_valid"] else f"X INVALID ({r['score']:.0f}%)"
        print(f"{status} ({r['elapsed']:.1f}s, {r['tokens_generated']} tok)")
        for line in r["poem_lines"][:2]:
            print(f"    {line[:65]}")

    valid = sum(1 for r in results if r["is_valid"])
    total_time = sum(r["elapsed"] for r in results)
    avg_time = total_time / len(results) if results else 0
    avg_tok = sum(r["tokens_generated"] for r in results) / len(results) if results else 0
    avg_score = sum(r["score"] for r in results) / len(results) if results else 0

    print(f"\n{'='*50}")
    print(f"  TOPIC STATS -- {METHOD_NAME}")
    print(f"{'='*50}")
    print(f"  Model:             {model_name}")
    print(f"  Topic:             {topic}")
    print(f"  Valid poems:       {valid}/{num_seeds} ({valid/num_seeds*100:.1f}%)")
    print(f"  Avg score:         {avg_score:.1f}")
    print(f"  Avg time/poem:     {avg_time:.1f}s")
    print(f"  Avg tokens/poem:   {avg_tok:.0f}")
    print(f"  Total time:        {total_time:.1f}s")
    print(f"{'='*50}")

    return results


###############################################################################
# SUMMARY
###############################################################################

def build_summary(all_results, model_name, total_time):
    total_poems = len(all_results)
    total_valid = sum(1 for r in all_results if r["is_valid"])
    avg_score = sum(r["score"] for r in all_results) / total_poems if total_poems else 0
    avg_time = sum(r["elapsed"] for r in all_results) / total_poems if total_poems else 0
    avg_tokens = sum(r["tokens_generated"] for r in all_results) / total_poems if total_poems else 0

    per_topic = {}
    for r in all_results:
        t = r["topic"]
        if t not in per_topic:
            per_topic[t] = {"total": 0, "valid": 0, "score_sum": 0.0}
        per_topic[t]["total"] += 1
        per_topic[t]["score_sum"] += r["score"]
        if r["is_valid"]:
            per_topic[t]["valid"] += 1
    for v in per_topic.values():
        v["rate"] = v["valid"] / v["total"] * 100 if v["total"] else 0
        v["avg_score"] = v["score_sum"] / v["total"] if v["total"] else 0
        del v["score_sum"]

    return {
        "label": f"{model_name} -- {METHOD_NAME}",
        "method": METHOD_NAME,
        "model": model_name,
        "constrained": True,
        "constraints": {"gana": True, "prasa": True, "yati": True, "hard_constraints": False},
        "decoding": "sampling (temp=0.7, top_p=0.9)",
        "total_poems": total_poems,
        "total_valid": total_valid,
        "poem_accuracy": total_valid / total_poems * 100 if total_poems else 0,
        "avg_score": avg_score,
        "avg_time_per_poem": avg_time,
        "avg_tokens_per_poem": avg_tokens,
        "total_time": total_time,
        "per_topic": per_topic,
        "poems": all_results,
    }


###############################################################################
# CLI
###############################################################################

def main():
    p = argparse.ArgumentParser(description="Chandomitra n=102 benchmark")
    p.add_argument("--model", type=str, default="gemma3-1b-base", choices=MODEL_CHOICES)
    p.add_argument("--seeds", type=int, default=NUM_SEEDS,
                   help=f"Seeds per topic (default {NUM_SEEDS})")
    p.add_argument("--topic", type=str, default=None,
                   help="Single topic to benchmark (default: all 3)")
    p.add_argument("--output", type=str, default=None,
                   help="Path to save results JSON")
    args = p.parse_args()

    model, tokenizer = load_model(args.model)
    topics = [args.topic] if args.topic else BENCHMARK_PROMPTS

    all_results = []
    total_start = time.time()
    for topic in topics:
        results = run_topic(model, tokenizer, topic, args.seeds, args.model)
        all_results.extend(results)
    total_time = time.time() - total_start

    output_path = args.output or os.path.join(
        SCRIPT_DIR, f"benchmark_chandomitra_n102_{args.model}.json"
    )
    summary = build_summary(all_results, args.model, total_time)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n  Results saved to: {output_path}")
    print(f"  Total: {summary['total_valid']}/{summary['total_poems']} valid "
          f"({summary['poem_accuracy']:.1f}%) | "
          f"avg score {summary['avg_score']:.1f} | "
          f"total time {total_time:.1f}s")


if __name__ == "__main__":
    main()
