#!/usr/bin/env python3
"""
Inference script for generating Telugu Dwipada poems using a fine-tuned Gemma 3 model.

Usage:
    # Single prompt:
    python -m dwipada.training.generate "ద్విపదలో ఒక పద్యం వ్రాయండి."

    # Interactive mode:
    python -m dwipada.training.generate --interactive

    # Batch mode (one prompt per line):
    python -m dwipada.training.generate --batch prompts.txt

    # Use merged model instead of base + adapter:
    python -m dwipada.training.generate --merged-model ./checkpoints/gemma3-1b-dwipada-merged "Write a dwipada."

    # Skip metrical validation:
    python -m dwipada.training.generate --no-validate "ద్విపద వ్రాయండి."
"""

import argparse
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from dwipada.core.constants import DWIPADA_RULES_BLOCK
from dwipada.paths import CHECKPOINTS_DIR

DEFAULT_BASE_MODEL = "google/gemma-3-1b-it"
DEFAULT_ADAPTER = str(CHECKPOINTS_DIR / "gemma3-1b-dwipada-lora" / "final")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate Telugu Dwipada poems using fine-tuned Gemma 3"
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        default=None,
        help="Prompt for poem generation",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Interactive mode (REPL)",
    )
    parser.add_argument(
        "--batch",
        type=str,
        default=None,
        help="File with one prompt per line for batch generation",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=DEFAULT_BASE_MODEL,
        help=f"Base model ID (default: {DEFAULT_BASE_MODEL})",
    )
    parser.add_argument(
        "--adapter",
        type=str,
        default=DEFAULT_ADAPTER,
        help="Path to LoRA adapter directory",
    )
    parser.add_argument(
        "--merged-model",
        type=str,
        default=None,
        help="Path to merged model (skips loading base + adapter separately)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Generation temperature (default: 0.7)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p nucleus sampling (default: 0.9)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum new tokens to generate (default: 256)",
    )
    parser.add_argument(
        "--num-poems",
        type=int,
        default=1,
        help="Number of poems to generate per prompt (default: 1)",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip metrical validation with dwipada_analyzer",
    )
    return parser.parse_args()


def load_model(args):
    """Load model and tokenizer."""
    if args.merged_model:
        print(f"Loading merged model from {args.merged_model}")
        model = AutoModelForCausalLM.from_pretrained(
            args.merged_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="sdpa",
        )
        tokenizer = AutoTokenizer.from_pretrained(args.merged_model)
    else:
        print(f"Loading base model: {args.base_model}")
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="sdpa",
        )
        print(f"Loading LoRA adapter from {args.adapter}")
        model = PeftModel.from_pretrained(model, args.adapter)
        tokenizer = AutoTokenizer.from_pretrained(args.adapter)

    model.eval()
    return model, tokenizer


def generate_poem(model, tokenizer, prompt, args):
    """Generate a dwipada poem from a prompt."""
    formatted = (
        f"<start_of_turn>user\n"
        f"{DWIPADA_RULES_BLOCK}\n\n"
        f"{prompt}<end_of_turn>\n"
        f"<start_of_turn>model\n"
    )

    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=50,
            repetition_penalty=1.1,
            do_sample=True,
            num_return_sequences=args.num_poems,
        )

    poems = []
    for seq in outputs:
        response = tokenizer.decode(
            seq[inputs["input_ids"].shape[1]:],
            skip_special_tokens=False,
        )
        # Strip trailing markers
        if "<end_of_turn>" in response:
            response = response[:response.index("<end_of_turn>")]
        poems.append(response.strip())

    return poems


def validate_poem(poem_text):
    """Validate a poem using dwipada_analyzer and return the report."""
    try:
        from dwipada.core.analyzer import analyze_dwipada, format_analysis_report
        analysis = analyze_dwipada(poem_text)
        score = analysis["match_score"]["overall"]
        report = format_analysis_report(analysis)
        return score, report
    except Exception as e:
        return None, f"Validation error: {e}"


def display_result(poem, args, index=None):
    """Display a generated poem with optional validation."""
    prefix = f"[Poem {index}] " if index is not None else ""
    print(f"\n{prefix}Generated Dwipada:")
    print("-" * 40)
    print(poem)
    print("-" * 40)

    if not args.no_validate:
        score, report = validate_poem(poem)
        if score is not None:
            print(f"Metrical Score: {score:.1f}%")
            if score < 100.0:
                print(report)
        else:
            print(report)


def run_interactive(model, tokenizer, args):
    """Interactive REPL mode."""
    print("\nInteractive Dwipada Generator")
    print("Type a prompt to generate a poem. Type 'quit' or 'exit' to stop.\n")

    while True:
        try:
            prompt = input("Prompt> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not prompt or prompt.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break

        poems = generate_poem(model, tokenizer, prompt, args)
        for i, poem in enumerate(poems, 1):
            display_result(poem, args, index=i if len(poems) > 1 else None)
        print()


def run_batch(model, tokenizer, args):
    """Batch mode: read prompts from file."""
    batch_path = Path(args.batch)
    if not batch_path.exists():
        print(f"Error: batch file not found: {args.batch}")
        sys.exit(1)

    prompts = [
        line.strip()
        for line in batch_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    print(f"Processing {len(prompts)} prompts from {args.batch}\n")

    for i, prompt in enumerate(prompts, 1):
        print(f"[{i}/{len(prompts)}] Prompt: {prompt}")
        poems = generate_poem(model, tokenizer, prompt, args)
        for j, poem in enumerate(poems, 1):
            display_result(poem, args, index=j if len(poems) > 1 else None)
        print()


def main():
    args = parse_args()

    if not args.prompt and not args.interactive and not args.batch:
        print("Error: provide a prompt, use --interactive, or use --batch <file>")
        sys.exit(1)

    model, tokenizer = load_model(args)

    if args.interactive:
        run_interactive(model, tokenizer, args)
    elif args.batch:
        run_batch(model, tokenizer, args)
    else:
        poems = generate_poem(model, tokenizer, args.prompt, args)
        for i, poem in enumerate(poems, 1):
            display_result(poem, args, index=i if len(poems) > 1 else None)


if __name__ == "__main__":
    main()
