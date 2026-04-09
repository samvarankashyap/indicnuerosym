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


def load_model_lora(base_name=DEFAULT_MODEL, adapter_path=None):
    """Load base model + LoRA adapter via peft."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    if adapter_path is None:
        adapter_path = os.path.join(PROJECT_DIR, "ragale_checkpoints", "checkpoint-336")

    print(f"\n  Loading base model: {base_name}")
    tokenizer = AutoTokenizer.from_pretrained(base_name)
    model = AutoModelForCausalLM.from_pretrained(
        base_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    )
    print(f"  Loading LoRA adapter: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)
    print(f"  Model + LoRA loaded on: {model.device}")
    return model, tokenizer


def load_model_quantized(model_name="google/gemma-4-E2B-it"):
    """Load model with 4-bit NF4 quantization (for large models like Gemma 4 E2B)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    import transformers.modeling_utils as _mu

    print(f"\n  Loading quantized model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Monkey-patch to avoid OOM during caching allocator warmup
    _mu.caching_allocator_warmup = lambda *args, **kwargs: None

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        llm_int8_enable_fp32_cpu_offload=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        max_memory={0: "7GiB", "cpu": "16GiB"},
        low_cpu_mem_usage=True,
    )
    print(f"  Quantized model loaded")
    return model, tokenizer


###############################################################################
# PROMPT BUILDING
###############################################################################

# System prompt for base models — strong "poem only" directive
_SYSTEM_PROMPT_BASE = (
    "You are a Kannada Maha Kavi. You output ONLY the poem — no greetings, "
    "no explanations, no titles, no confirmations, no English text. "
    "You write Utsaha Ragale poems: exactly 2 lines, 12 syllables per line, "
    "4 ganas (III or IIU), ending on Guru, with Adi Prasa."
)

# System prompt for fine-tuned models — matches IFT training format
_SYSTEM_PROMPT_LORA = (
    "You are an expert Kannada Scholar and Maha Kavi. "
    "You possess deep knowledge of Kannada Chandassu (Prosody). "
    "Generate Utsaha Ragale poems with exactly 12 syllables per line, "
    "4 ganas (III or IIU pattern), ending on Guru, with Adi Prasa."
)

# Rules block matching IFT training data (English — used for LoRA models)
_RAGALE_RULES_EN = (
    "Utsaha Ragale rules:\n"
    "- 2 lines, each with exactly 12 syllables (aksharas)\n"
    "- 4 ganas per line: each gana is III (laghu-laghu-laghu) "
    "or IIU (laghu-laghu-guru)\n"
    "- 4th gana must be IIU (line must end on Guru)\n"
    "- Adi Prasa: the 2nd syllable's base consonant must match "
    "between both lines"
)


def build_prompt(topic, tokenizer, model_name=None):
    """Build generation prompt for a Kannada Ragale poem.

    Uses English prompt for LoRA models (matching IFT training format)
    and Kannada prompt for base models (to avoid conversational bleed).
    """
    is_lora = model_name and "lora" in model_name

    if is_lora:
        # Match the IFT training format (G1 profile)
        system_prompt = _SYSTEM_PROMPT_LORA
        user_prompt = (
            f"{_RAGALE_RULES_EN}\n\n"
            f"Write a 2-line Utsaha Ragale poem in Kannada about: {topic}"
        )
    else:
        # Kannada prompt for base models — avoids ನಮಸ್ಕಾರ/ಖಚಿತ bleed
        system_prompt = _SYSTEM_PROMPT_BASE
        user_prompt = (
            f"'{topic}' ಕುರಿತು ಉತ್ಸಾಹ ರಗಳೆ ಬರೆಯಿರಿ.\n\n"
            "ನಿಯಮಗಳು:\n"
            "- ಪ್ರತಿ ಸಾಲು: ನಿಖರವಾಗಿ 12 ಅಕ್ಷರಗಳು\n"
            "- 4 ಗಣಗಳು: III (ಲಘು-ಲಘು-ಲಘು) ಅಥವಾ IIU (ಲಘು-ಲಘು-ಗುರು)\n"
            "- ಎರಡೂ ಸಾಲುಗಳು ಗುರುವಿನಲ್ಲಿ ಕೊನೆಗೊಳ್ಳಬೇಕು\n"
            "- ಆದಿ ಪ್ರಾಸ: 2ನೇ ಅಕ್ಷರದ ವ್ಯಂಜನ ಎರಡೂ ಸಾಲುಗಳಲ್ಲಿ ಹೊಂದಬೇಕು\n\n"
            "ಕೇವಲ ಕನ್ನಡ ಪದ್ಯವನ್ನು ಮಾತ್ರ ಬರೆಯಿರಿ. "
            "ಯಾವುದೇ ವಿವರಣೆ, ಶೀರ್ಷಿಕೆ, ಅಥವಾ ಇಂಗ್ಲಿಷ್ ಬೇಡ."
        )

    messages = [
        {"role": "system", "content": system_prompt},
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
        # Skip lines that are labels, titles, or meta-text
        if l.endswith(":") or l.startswith("Poem:"):
            continue
        # Strip known conversational/meta prefixes from base models
        for prefix in ("ಖಚಿತತೆ:", "ಖಚಿತ:", "ನಮಸ್ಕಾರ", "ಶಸ್:"):
            if l.startswith(prefix):
                l = l[len(prefix):].strip()
        if not l:
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
