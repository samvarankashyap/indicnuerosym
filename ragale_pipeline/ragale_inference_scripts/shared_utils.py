# -*- coding: utf-8 -*-
"""
Shared utilities for Ragale inference scripts.

Contains: model loading, prompt building, Kannada token validation,
poem validation, and result formatting.

All three inference approaches (masking-only, masking+backtrack, hybrid)
import from this module.
"""

import os
import sys
import json

# ---------------------------------------------------------------------------
# Path setup — import from nfa_pipeline and analyser
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
NFA_DIR = os.path.join(PROJECT_DIR, "nfa_pipeline")
sys.path.insert(0, NFA_DIR)
sys.path.insert(0, PROJECT_DIR)

from composite_state import (
    CompositeState, build_gana_mask, get_kannada_token_set,
    VALID_LINE_LENGTHS, MAX_LINE_LENGTH,
)
from ragale_pipeline import RagalePipeline, format_pipeline_result
from gana_nfa import format_partition_str


###############################################################################
# MODEL LOADING
###############################################################################

DEFAULT_MODEL = "google/gemma-3-1b-it"


def load_model(model_name=DEFAULT_MODEL):
    """Load model + tokenizer. Supports base Gemma models."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\n  Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    )
    print(f"  Model loaded on: {model.device}")
    return model, tokenizer


###############################################################################
# PROMPT BUILDING
###############################################################################

SYSTEM_PROMPT = (
    "You are an expert Kannada Scholar and Maha Kavi. "
    "You possess deep knowledge of Kannada Chandassu (Prosody). "
    "Generate Utsaha Ragale poems with exactly 12 syllables per line, "
    "4 ganas (III or IIU pattern), ending on Guru, with Adi Prasa."
)


def build_prompt(topic, tokenizer):
    """Build generation prompt for a Kannada Ragale poem."""
    user_prompt = (
        f"Write a 2-line Utsaha Ragale poem in Kannada about: {topic}\n\n"
        "Rules:\n"
        "- Each line: exactly 12 syllables (aksharas)\n"
        "- 4 ganas per line: III (short-short-short) or IIU (short-short-long)\n"
        "- Both lines end on Guru (long syllable)\n"
        "- 2nd syllable consonant must match between lines (Adi Prasa)\n\n"
        "Poem:"
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


###############################################################################
# KANNADA TOKEN VALIDATION
###############################################################################

def is_valid_kannada_token(token_text):
    """Check if a token contains only Kannada characters and formatting."""
    for ch in token_text:
        cp = ord(ch)
        if (0x0C80 <= cp <= 0x0CFF   # Kannada Unicode block
                or ch in " \n:"):      # space, newline, colon
            continue
        return False
    return True


###############################################################################
# PRECOMPUTATION
###############################################################################

def precompute_token_data(tokenizer):
    """Build static mask and Kannada token set for logit masking.

    Returns:
        (kannada_ids, kannada_texts, static_mask, newline_token_id)
    """
    import torch

    kannada_ids, kannada_texts = get_kannada_token_set(tokenizer)
    vocab_size = tokenizer.vocab_size
    static_mask = torch.full((vocab_size,), float("-inf"))
    for tid in kannada_ids:
        static_mask[tid] = 0.0
    if tokenizer.eos_token_id is not None:
        static_mask[tokenizer.eos_token_id] = 0.0
    newline_ids = tokenizer.encode("\n", add_special_tokens=False)
    newline_token_id = newline_ids[0] if newline_ids else None
    print(f"  Kannada tokens: {len(kannada_ids)}")
    print(f"  Newline token ID: {newline_token_id}")
    return kannada_ids, kannada_texts, static_mask, newline_token_id


###############################################################################
# POEM VALIDATION (using NFA pipeline)
###############################################################################

def validate_poem(poem_text):
    """Validate a generated poem using the NFA pipeline.

    Returns list of per-line validation dicts.
    """
    pipeline = RagalePipeline()
    result = pipeline.process(poem_text)
    valid_lines = []

    for lk in ["line1", "line2"]:
        gl = result["guru_laghu"].get(lk, [])
        markers = " ".join(l for _, l in gl)
        gana = result["gana"].get(lk)
        partition_str = format_partition_str(gana) if gana else "INVALID"
        syls = result["syllables"].get(lk, [])

        valid_lines.append({
            "line": " ".join(s for s, _ in gl) if gl else "",
            "syllables": syls,
            "markers": markers,
            "valid": result["gana"].get(f"{lk}_valid", False),
            "partition": partition_str,
            "syllable_count": len(syls),
            "guru_ending": result["guru_ending"].get(lk, False),
        })

    return valid_lines


def validate_poem_full(poem_text):
    """Full validation returning the complete pipeline result dict."""
    pipeline = RagalePipeline()
    return pipeline.process(poem_text)


###############################################################################
# LINE EXTRACTION
###############################################################################

def extract_poem_lines(text):
    """Extract poem lines from model output, stripping any prefix."""
    raw = [l.strip() for l in text.split("\n") if l.strip()]
    lines = []
    for l in raw:
        # Skip common prefixes
        if l.endswith(":") or l.startswith("Poem:"):
            continue
        # Check if line contains Kannada characters
        has_kannada = any(0x0C80 <= ord(ch) <= 0x0CFF for ch in l)
        if has_kannada:
            lines.append(l)
    return lines[:2]


###############################################################################
# RESULT FORMATTING
###############################################################################

def print_result(result, index=None):
    """Pretty-print a generation result."""
    prefix = f"[{index}] " if index is not None else ""
    status = "VALID" if result["all_valid"] else "INVALID"
    print(f"\n{prefix}{status} — {result['method']} "
          f"({result['elapsed']:.1f}s, {result['tokens_generated']} tok, "
          f"{result.get('mask_computations', 0)} masks, "
          f"{result.get('backtracks', 0)} backtracks)")

    for vl in result["valid_lines"]:
        mark = "V" if vl["valid"] else "X"
        print(f"  {mark} [{vl['syllable_count']} syls] {vl['line'][:70]}")
        if vl["valid"]:
            print(f"    Gana: {vl['partition']}")
        print(f"    Markers: {vl['markers']}")


def print_summary(results, method_name):
    """Print aggregate statistics."""
    total = len(results)
    valid = sum(1 for r in results if r["all_valid"])
    avg_time = sum(r["elapsed"] for r in results) / total if total else 0
    avg_tok = sum(r["tokens_generated"] for r in results) / total if total else 0
    avg_masks = sum(r.get("mask_computations", 0) for r in results) / total if total else 0
    avg_bt = sum(r.get("backtracks", 0) for r in results) / total if total else 0

    print(f"\n{'='*60}")
    print(f"  {method_name.upper()} SUMMARY")
    print(f"{'='*60}")
    print(f"  Poems:         {total}")
    print(f"  Valid:          {valid}/{total} ({valid/total*100:.1f}%)")
    print(f"  Avg time:       {avg_time:.1f}s")
    print(f"  Avg tokens:     {avg_tok:.0f}")
    print(f"  Avg masks:      {avg_masks:.0f}")
    print(f"  Avg backtracks: {avg_bt:.1f}")
    print(f"{'='*60}")


###############################################################################
# BENCHMARK TOPICS
###############################################################################

BENCHMARK_TOPICS = [
    "Bubbles",
    "Moon",
    "Rain",
    "Birds",
    "Flowers",
]
