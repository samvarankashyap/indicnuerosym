#!/usr/bin/env python3
"""Generate Telugu Dwipada poems with constrained decoding.

Uses a custom LogitsProcessor (adapted from Chandomitra paper, Algorithm 1)
to enforce metrical constraints during autoregressive generation:
  - Gana structure: 3 Indra + 1 Surya per line (via prefix trie)
  - Prasa: 2nd syllable consonant rhyme between lines
  - Yati: alliteration between gana 1 and gana 3

Usage:
    # Single prompt (greedy, constrained):
    python -m dwipada.training.generate_constrained "ద్విపదలో ఒక పద్యం వ్రాయండి."

    # Interactive mode:
    python -m dwipada.training.generate_constrained --interactive

    # Batch mode:
    python -m dwipada.training.generate_constrained --batch prompts.txt

    # Disable specific constraints:
    python -m dwipada.training.generate_constrained --no-prasa --no-yati "prompt"

    # Use sampling instead of greedy:
    python -m dwipada.training.generate_constrained --do-sample --temperature 0.7 "prompt"
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, LogitsProcessorList

from dwipada.core.analyzer import analyze_dwipada, format_analysis_report
from dwipada.core.constants import DWIPADA_RULES_BLOCK
from dwipada.paths import CHECKPOINTS_DIR
from dwipada.training.constrained.logits_processor import DwipadaConstrainedLogitsProcessor

DEFAULT_BASE_MODEL = "google/gemma-3-1b-it"
DEFAULT_ADAPTER = str(CHECKPOINTS_DIR / "gemma3-1b-dwipada-lora" / "final")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate Telugu Dwipada poems with constrained decoding"
    )
    parser.add_argument(
        "prompt", nargs="?", default=None, help="Prompt for poem generation"
    )
    parser.add_argument(
        "--interactive", action="store_true", help="Interactive mode (REPL)"
    )
    parser.add_argument(
        "--batch", type=str, default=None, help="File with one prompt per line"
    )
    # Model options
    parser.add_argument(
        "--base-model",
        type=str,
        default=DEFAULT_BASE_MODEL,
        help=f"Base model ID (default: {DEFAULT_BASE_MODEL})",
    )
    parser.add_argument(
        "--adapter", type=str, default=None, help="Path to LoRA adapter directory"
    )
    parser.add_argument(
        "--merged-model",
        type=str,
        default=None,
        help="Path to merged model (skips base + adapter loading)",
    )
    # Constraint options
    parser.add_argument(
        "--top-k-constraint",
        type=int,
        default=25,
        help="Initial k for constraint filtering (default: 25)",
    )
    parser.add_argument(
        "--max-k-constraint",
        type=int,
        default=200,
        help="Max k before fallback to unconstrained (default: 200)",
    )
    parser.add_argument(
        "--no-prasa",
        action="store_true",
        help="Disable prasa (rhyme) constraint",
    )
    parser.add_argument(
        "--no-yati",
        action="store_true",
        help="Disable yati (alliteration) constraint",
    )
    parser.add_argument(
        "--hard-constraints",
        action="store_true",
        help="Enforce all constraints as hard (no relaxation — scans full vocabulary)",
    )
    # Decoding options
    parser.add_argument(
        "--do-sample",
        action="store_true",
        help="Use sampling instead of greedy decoding",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7, only with --do-sample)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum new tokens to generate (default: 256)",
    )
    # Validation & output
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip post-generation metrical validation",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Save results to JSON file (poems, scores, analysis)",
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Load model in 4-bit quantization (saves VRAM for larger models)",
    )
    return parser.parse_args()


def load_model(args):
    """Load model and tokenizer."""
    import os
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

    # Disable the caching allocator warmup that tries to pre-allocate
    # the full unquantized model size on GPU (OOM on small GPUs)
    import transformers.modeling_utils as _mu
    if hasattr(_mu, "caching_allocator_warmup"):
        _mu.caching_allocator_warmup = lambda *args, **kwargs: None

    load_kwargs = {
        "device_map": "auto",
    }

    if args.load_in_4bit:
        print("Using 4-bit quantization")
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            llm_int8_enable_fp32_cpu_offload=True,
        )
        load_kwargs["max_memory"] = {0: "7GiB", "cpu": "16GiB"}
        load_kwargs["low_cpu_mem_usage"] = True
    else:
        load_kwargs["dtype"] = torch.bfloat16

    if args.merged_model:
        print(f"Loading merged model from {args.merged_model}")
        model = AutoModelForCausalLM.from_pretrained(
            args.merged_model, **load_kwargs
        )
        tokenizer = AutoTokenizer.from_pretrained(args.merged_model)
    else:
        print(f"Loading base model: {args.base_model}")
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model, **load_kwargs
        )
        if args.adapter:
            print(f"Loading LoRA adapter from {args.adapter}")
            model = PeftModel.from_pretrained(model, args.adapter)
            tokenizer = AutoTokenizer.from_pretrained(args.adapter)
        else:
            tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    model.eval()
    return model, tokenizer


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


def build_prompt(prompt: str, tokenizer=None) -> str:
    """Format prompt with structured Dwipada instructions using chat template."""
    instruction = STRUCTURED_PROMPT.format(user_instruction=prompt)
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": instruction}]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    # Fallback to Gemma 3 format
    return (
        f"<start_of_turn>user\n"
        f"{instruction}<end_of_turn>\n"
        f"<start_of_turn>model\n"
    )


def generate_poem_constrained(model, tokenizer, prompt, args):
    """Generate a single Dwipada poem with constrained decoding."""
    formatted = build_prompt(prompt, tokenizer)
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

    # Create a fresh processor for each poem
    processor = DwipadaConstrainedLogitsProcessor(
        tokenizer=tokenizer,
        initial_k=args.top_k_constraint,
        max_k=args.max_k_constraint,
        enable_prasa=not args.no_prasa,
        enable_yati=not args.no_yati,
        hard_constraints=args.hard_constraints,
    )
    processor.prompt_length = inputs["input_ids"].shape[1]

    generate_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "logits_processor": LogitsProcessorList([processor]),
    }

    if args.do_sample:
        generate_kwargs.update(
            {
                "do_sample": True,
                "temperature": args.temperature,
                "top_p": 0.9,
            }
        )
    else:
        # Greedy decoding (best per Chandomitra ablation study)
        generate_kwargs["do_sample"] = False

    with torch.no_grad():
        outputs = model.generate(**inputs, **generate_kwargs)

    # Decode generated tokens (excluding prompt), stripping special tokens
    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1] :],
        skip_special_tokens=True,
    )

    return response.strip()


def validate_poem(poem_text):
    """Validate a poem using dwipada analyzer and return score + report."""
    try:
        analysis = analyze_dwipada(poem_text)
        score = analysis["match_score"]["overall"]
        report = format_analysis_report(analysis)
        return score, report, analysis
    except Exception as e:
        return None, f"Validation error: {e}", None


def display_result(poem, args, index=None):
    """Display a generated poem with optional validation."""
    prefix = f"[Poem {index}] " if index is not None else ""
    print(f"\n{prefix}Generated Dwipada (Constrained):")
    print("-" * 50)
    print(poem)
    print("-" * 50)

    if not args.no_validate:
        score, report, analysis = validate_poem(poem)
        if score is not None:
            print(f"Metrical Score: {score:.1f}%")
            if analysis:
                breakdown = analysis["match_score"]["breakdown"]
                print(
                    f"  Gana: {breakdown.get('gana', 'N/A')}%  "
                    f"Prasa: {breakdown.get('prasa', 'N/A')}%  "
                    f"Yati: {breakdown.get('yati_average', 'N/A')}%"
                )
            if score < 100.0:
                print(f"\n{report}")
        else:
            print(report)


def run_interactive(model, tokenizer, args):
    """Interactive REPL mode with constrained decoding."""
    print("\nInteractive Dwipada Generator (Constrained Decoding)")
    print("Type a prompt to generate. Type 'quit' or 'exit' to stop.\n")

    while True:
        try:
            prompt = input("Prompt> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not prompt or prompt.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break

        poem = generate_poem_constrained(model, tokenizer, prompt, args)
        display_result(poem, args)
        print()


def run_batch(model, tokenizer, args):
    """Batch mode: read prompts from file, generate with constraints."""
    batch_path = Path(args.batch)
    if not batch_path.exists():
        print(f"Error: batch file not found: {args.batch}")
        sys.exit(1)

    prompts = [
        line.strip()
        for line in batch_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    print(f"Processing {len(prompts)} prompts with constrained decoding\n")

    scores = []
    results = []
    for i, prompt in enumerate(prompts, 1):
        print(f"[{i}/{len(prompts)}] Prompt: {prompt}")
        poem = generate_poem_constrained(model, tokenizer, prompt, args)
        display_result(poem, args, index=i)

        result_entry = {"prompt": prompt, "poem": poem}
        if not args.no_validate:
            score, report, analysis = validate_poem(poem)
            if score is not None:
                scores.append(score)
            result_entry["score"] = score
            result_entry["report"] = report
            if analysis:
                result_entry["is_valid"] = analysis.get("is_valid_dwipada", False)
                result_entry["breakdown"] = analysis.get("match_score", {}).get("breakdown", {})
                # Line-level gana info
                for line_key in ("pada1", "pada2"):
                    pada = analysis.get(line_key, {})
                    partition = pada.get("partition")
                    if partition:
                        ganas = []
                        for g in partition.get("ganas", []):
                            ganas.append({
                                "name": g.get("name", ""),
                                "pattern": g.get("pattern", ""),
                                "aksharalu": "".join(g.get("aksharalu", [])),
                                "valid": g.get("valid", False),
                            })
                        result_entry[f"{line_key}_ganas"] = ganas
                        result_entry[f"{line_key}_valid"] = partition.get("is_fully_valid", False)
        results.append(result_entry)
        print()

    if scores:
        avg = sum(scores) / len(scores)
        perfect = sum(1 for s in scores if s == 100.0)
        print(f"\n{'=' * 50}")
        print(f"Batch Summary ({len(scores)} poems):")
        print(f"  Average score: {avg:.1f}%")
        print(f"  Perfect (100%): {perfect}/{len(scores)} ({100*perfect/len(scores):.1f}%)")
        print(f"{'=' * 50}")

    # Save results to file
    if args.output:
        output_data = {
            "config": {
                "model": args.base_model,
                "adapter": args.adapter,
                "merged_model": args.merged_model,
                "constraints": {
                    "gana": True,
                    "prasa": not args.no_prasa,
                    "yati": not args.no_yati,
                    "hard_constraints": args.hard_constraints,
                },
                "decoding": "sampling" if args.do_sample else "greedy",
                "top_k_constraint": args.top_k_constraint,
            },
            "summary": {
                "total": len(results),
                "average_score": round(avg, 1) if scores else None,
                "perfect_count": perfect if scores else 0,
            },
            "results": results,
        }
        output_path = Path(args.output)
        output_path.write_text(
            json.dumps(output_data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\nResults saved to {args.output}")


def main():
    args = parse_args()

    if not args.prompt and not args.interactive and not args.batch:
        print("Error: provide a prompt, use --interactive, or use --batch <file>")
        sys.exit(1)

    model, tokenizer = load_model(args)

    constraint_info = []
    if not args.no_prasa:
        constraint_info.append("prasa")
    if not args.no_yati:
        constraint_info.append("yati")
    mode = "HARD (no relaxation)" if args.hard_constraints else "soft (with relaxation)"
    print(f"Constraints: gana (always) + {', '.join(constraint_info) or 'none'} [{mode}]")
    print(f"Decoding: {'sampling (T={})'.format(args.temperature) if args.do_sample else 'greedy'}")
    print(f"Top-k constraint: {args.top_k_constraint} (max: {args.max_k_constraint})\n")

    if args.interactive:
        run_interactive(model, tokenizer, args)
    elif args.batch:
        run_batch(model, tokenizer, args)
    else:
        poem = generate_poem_constrained(model, tokenizer, args.prompt, args)
        display_result(poem, args)


if __name__ == "__main__":
    main()
