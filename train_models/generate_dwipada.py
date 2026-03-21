#!/usr/bin/env python3
"""
Generate dwipada poems using the merged fine-tuned model.

Usage:
    # Interactive mode (type your own prompts):
    python generate_dwipada.py

    # With a specific topic:
    python generate_dwipada.py --topic "శ్రీరాముడు సీతను రక్షించుట"

    # Run built-in sample prompts:
    python generate_dwipada.py --samples

    # Use adapter instead of merged model:
    python generate_dwipada.py --adapter
"""

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

SCRIPT_DIR = Path(__file__).resolve().parent
MERGED_DIR = SCRIPT_DIR / "dwipada_merged_model"
ADAPTER_DIR = SCRIPT_DIR / "dwipada_lora_adapter"
BASE_MODEL = "google/gemma-3-1b-it"

SYSTEM_PROMPT = "You are a Telugu and Sanskrit scholar specialising in Dwipada poetry."

SAMPLE_TOPICS = [
    "శ్రీరాముడు సీతాదేవిని రక్షించుటకు లంకకు వెళ్ళెను.",
    "భగవంతుడు సర్వ ప్రాణులను రక్షించును.",
    "హనుమంతుడు సముద్రమును దాటి లంకను చేరెను.",
    "సూర్యోదయమున ప్రకృతి అందముగా వెలుగును.",
    "గురువు శిష్యునకు విద్యను బోధించెను.",
    "తల్లి ప్రేమ అన్నింటికంటే గొప్పది.",
    "రావణుడు సీతను అపహరించి లంకకు తీసుకొనిపోయెను.",
    "కృష్ణుడు అర్జునునకు గీతోపదేశము చేసెను.",
    "నదులు పర్వతముల నుండి సముద్రమునకు ప్రవహించును.",
    "విజయమునకు కఠోర పరిశ్రమ అవసరము.",
]


def load_model(use_adapter=False):
    print(f"\nLoading model...")
    if use_adapter:
        print(f"  Base: {BASE_MODEL}")
        print(f"  Adapter: {ADAPTER_DIR}")
        tokenizer = AutoTokenizer.from_pretrained(str(ADAPTER_DIR))
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="sdpa",
        )
        model = PeftModel.from_pretrained(model, str(ADAPTER_DIR))
    else:
        print(f"  Merged model: {MERGED_DIR}")
        tokenizer = AutoTokenizer.from_pretrained(str(MERGED_DIR))
        model = AutoModelForCausalLM.from_pretrained(
            str(MERGED_DIR),
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="sdpa",
        )

    model.eval()
    print("  Model loaded.\n")
    return model, tokenizer


def generate(model, tokenizer, topic, temperature=0.5, top_p=0.9, max_new_tokens=128, repetition_penalty=1.3):
    user_prompt = (
        "క్రింది తెలుగు భావానికి అనుగుణంగా ఒక ద్విపద పద్యం రచించండి. "
        "ప్రతి పాదంలో 3 ఇంద్ర గణాలు + 1 సూర్య గణం ఉండాలి.\n\n"
        f"తెలుగు భావం: తెలుగు: {topic}"
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            repetition_penalty=repetition_penalty,
            pad_token_id=tokenizer.pad_token_id,
        )

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()

    # Extract only the two-line poem (skip "ద్విపద:" prefix if present)
    lines = [l.strip() for l in response.split("\n") if l.strip()]
    # Skip header like "ద్విపద:" or "పూర్తి ద్విపద:"
    poem_lines = [l for l in lines if not l.endswith(":") and not l.startswith("ద్విపద")]
    if not poem_lines and lines:
        poem_lines = lines[1:] if len(lines) > 1 else lines
    # Take exactly 2 lines
    poem_lines = poem_lines[:2]
    return "\n".join(poem_lines)


def run_samples(model, tokenizer):
    print("=" * 70)
    print("  Dwipada Poem Generation — Sample Outputs")
    print("=" * 70)

    for i, topic in enumerate(SAMPLE_TOPICS, 1):
        print(f"\n  [{i}] భావం: {topic}")
        print(f"  " + "-" * 60)
        response = generate(model, tokenizer, topic)
        print(f"  {response}")
        print()


def run_interactive(model, tokenizer):
    print("=" * 70)
    print("  Dwipada Poem Generator — Interactive Mode")
    print("  Type a Telugu meaning/topic, press Enter to generate.")
    print("  Type 'quit' or 'exit' to stop.")
    print("=" * 70)

    while True:
        try:
            topic = input("\n  భావం: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nDone.")
            break

        if not topic or topic.lower() in ("quit", "exit", "q"):
            print("Done.")
            break

        print(f"  " + "-" * 60)
        response = generate(model, tokenizer, topic)
        print(f"  {response}")


def main():
    p = argparse.ArgumentParser(description="Generate dwipada poems")
    p.add_argument("--topic", type=str, default=None,
                   help="Telugu meaning/topic for poem generation")
    p.add_argument("--samples", action="store_true",
                   help="Run built-in sample prompts")
    p.add_argument("--adapter", action="store_true",
                   help="Use LoRA adapter instead of merged model")
    p.add_argument("--temperature", type=float, default=0.5)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--repetition_penalty", type=float, default=1.3)
    args = p.parse_args()

    model, tokenizer = load_model(use_adapter=args.adapter)

    if args.samples:
        run_samples(model, tokenizer)
    elif args.topic:
        print(f"\n  భావం: {args.topic}")
        print(f"  " + "-" * 60)
        response = generate(model, tokenizer, args.topic,
                          temperature=args.temperature, top_p=args.top_p,
                          repetition_penalty=args.repetition_penalty)
        print(f"  {response}\n")
    else:
        run_interactive(model, tokenizer)


if __name__ == "__main__":
    main()
