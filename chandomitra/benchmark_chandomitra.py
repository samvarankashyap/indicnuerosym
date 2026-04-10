#!/usr/bin/env python3
"""Benchmark Chandomitra constrained decoding — 20 poems, detailed metrics.

Captures: valid poems, avg time/poem, avg tokens/poem, avg relaxations, total time.
Uses the same Gemma 4 E2B model with 4-bit quantization.
"""

import json
import os
import time

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, LogitsProcessorList

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from dwipada.core.analyzer import analyze_dwipada, format_analysis_report
from dwipada.training.constrained.logits_processor import DwipadaConstrainedLogitsProcessor

MODEL_ID = "google/gemma-4-E2B-it"

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
    "పుస్తకం జ్ఞానాన్ని ఇచ్చే నేస్తం",
    "ప్రకృతి అందాలు మనసును రంజింపజేయును",
    "విద్య వల్ల జ్ఞానం లభించును",
    "నది ప్రవాహం అందమైనది",
    "సముద్రము విశాలమైనది అనంతమైనది",
    "చంద్రుడు రాత్రిని వెలిగించును",
    "దేశభక్తి మనకు ధర్మము",
    "స్నేహం జీవితంలో అమూల్యమైనది",
    "కాలము ఎవరికొరకు ఆగదు",
    "ధర్మము సత్యము కలిసి నడచును",
]

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


def load_model():
    import transformers.modeling_utils as _mu
    if hasattr(_mu, "caching_allocator_warmup"):
        _mu.caching_allocator_warmup = lambda *a, **kw: None

    print(f"Loading model: {MODEL_ID} (4-bit NF4)")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        llm_int8_enable_fp32_cpu_offload=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        max_memory={0: "7GiB", "cpu": "16GiB"},
        low_cpu_mem_usage=True,
    )
    model.eval()

    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1e9
        print(f"GPU memory used: {alloc:.1f} GB")

    return model, tokenizer


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
    except Exception as e:
        return 0.0, False, {}


def generate_one(model, tokenizer, topic, top_k=25, max_k=200):
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
            max_new_tokens=256,
            do_sample=False,
            logits_processor=LogitsProcessorList([processor]),
        )
    elapsed = time.time() - start

    generated_ids = outputs[0][prompt_len:]
    tokens_generated = len(generated_ids)
    poem = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    score, is_valid, breakdown = validate_poem(poem)

    return {
        "topic": topic,
        "poem": poem,
        "score": score,
        "is_valid": is_valid,
        "breakdown": breakdown,
        "elapsed": elapsed,
        "tokens_generated": tokens_generated,
    }


def main():
    model, tokenizer = load_model()

    print(f"\n{'='*72}")
    print(f"  BENCHMARK: Chandomitra Constrained Decoding (Masking)")
    print(f"  Model: {MODEL_ID} (4-bit NF4)")
    print(f"  Prompts: {len(BENCHMARK_PROMPTS)}")
    print(f"  Method: LogitsProcessor masking, greedy decoding")
    print(f"{'='*72}\n")

    results = []
    valid_count = 0
    total_start = time.time()

    for i, topic in enumerate(BENCHMARK_PROMPTS):
        print(f"  [{i+1:2d}/{len(BENCHMARK_PROMPTS)}] {topic[:50]}...", end=" ", flush=True)

        result = generate_one(model, tokenizer, topic)
        results.append(result)

        if result["is_valid"]:
            valid_count += 1
            status = "VALID"
        else:
            status = f"INVALID ({result['score']:.0f}%)"

        print(f"{status}  ({result['elapsed']:.1f}s, {result['tokens_generated']} tok)")

        lines = result["poem"].split("\n")
        for line in lines[:2]:
            print(f"      {line[:70]}")

    total_time = time.time() - total_start

    # Compute stats
    avg_time = sum(r["elapsed"] for r in results) / len(results)
    avg_tokens = sum(r["tokens_generated"] for r in results) / len(results)
    avg_score = sum(r["score"] for r in results) / len(results)

    print(f"\n{'='*72}")
    print(f"  RESULTS: Chandomitra Constrained Decoding")
    print(f"{'='*72}")
    print(f"  Valid poems:      {valid_count}/{len(results)} ({valid_count/len(results)*100:.0f}%)")
    print(f"  Avg score:        {avg_score:.1f}%")
    print(f"  Avg time/poem:    {avg_time:.1f}s")
    print(f"  Avg tokens/poem:  {avg_tokens:.0f}")
    print(f"  Total time:       {total_time:.1f}s")
    print(f"{'='*72}")

    # Save results
    output_data = {
        "model": MODEL_ID,
        "method": "chandomitra_constrained_masking",
        "quantization": "4-bit NF4",
        "total_poems": len(results),
        "valid_poems": valid_count,
        "valid_rate": valid_count / len(results) * 100,
        "avg_score": round(avg_score, 1),
        "avg_time_per_poem": round(avg_time, 1),
        "avg_tokens_per_poem": round(avg_tokens, 0),
        "total_time": round(total_time, 1),
        "poems": results,
    }
    output_path = "benchmark_chandomitra_20poems.json"
    with open(output_path, "w") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print(f"\n  Results saved: {output_path}")


if __name__ == "__main__":
    main()
