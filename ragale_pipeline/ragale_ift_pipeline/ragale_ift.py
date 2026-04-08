"""
Ragale IFT Pipeline — Unified Tool
=====================================
Data preparation, format conversion, and fine-tuning for
Kannada Utsaha Ragale poem generation.

All generation profiles output **poem only** — no meanings, no analysis.

Subcommands:
    prepare   — Assign generation profiles + train/val/test split
    convert   — Convert training_ready.jsonl to alpaca/sharegpt/trl formats
    finetune  — QLoRA fine-tune Gemma 3 1B IT on ift_alpaca.jsonl (local only)

Usage:
    python ragale_ift.py prepare  --input ../ragale_consolidated_filtered.json --outdir ./ift_data
    python ragale_ift.py convert  --input ./ift_data/training_ready.jsonl --outdir ./ift_data
    python ragale_ift.py finetune --dataset ./ift_data/ift_alpaca.jsonl

Requirements:
    Core     : (none — stdlib only)
    Finetune : pip install peft transformers trl datasets
"""

import json
import random
import re
import os
import argparse
from pathlib import Path


# ──────────────────────────────────────────────
# SECTION 0 ▸ SHARED UTILITIES
# ──────────────────────────────────────────────

def load_jsonl(path: str) -> list[dict]:
    """Loads either a JSON array (.json) or JSONL (.jsonl) file."""
    with open(path, encoding="utf-8") as f:
        first_char = f.read(1)
        f.seek(0)
        if first_char == "[":
            data = json.load(f)
            if not isinstance(data, list):
                raise ValueError(f"{path} is a JSON object, expected a list/array.")
            return data
        else:
            return [json.loads(line) for line in f if line.strip()]


def save_jsonl(records: list[dict], path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"  Saved {len(records)} records -> {path}")


# ──────────────────────────────────────────────
# SECTION 1 ▸ RAGALE-SPECIFIC HELPERS
# ──────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are an expert Kannada Scholar and Maha Kavi "
    "specializing in Utsaha Ragale poetry."
)

RAGALE_RULES = (
    "Utsaha Ragale rules:\n"
    "- 2 lines, each with exactly 12 syllables (aksharas)\n"
    "- 4 ganas per line: each gana is III (laghu-laghu-laghu) "
    "or IIU (laghu-laghu-guru)\n"
    "- 4th gana must be IIU (line must end on Guru)\n"
    "- Adi Prasa: the 2nd syllable's base consonant must match "
    "between both lines"
)

# Generic prompts for G5 profile — varied phrasing to prevent overfitting
GENERIC_PROMPTS = [
    "Write a 2-line Utsaha Ragale poem in Kannada.",
    "Compose a Kannada Ragale poem following the Utsaha meter.",
    "Generate a Kannada Utsaha Ragale couplet.",
    "Create a Ragale poem in Kannada with proper meter and Adi Prasa.",
    "Write a metrically valid Utsaha Ragale in Kannada.",
    "Produce a Kannada Ragale poem with 12 syllables per line.",
    "Compose a Ragale couplet in Kannada following all metrical rules.",
    "Write a Kannada poem in the Utsaha Ragale form.",
]


def _extract_prasa(rec: dict) -> str:
    """Extract prasa consonant from prasa_syllable field.
    e.g. 'l (ಲ್)' -> 'ಲ್', 'ಮ' -> 'ಮ'
    """
    ps = rec.get("prasa_syllable", "")
    # Try to extract Kannada character from parentheses
    m = re.search(r'\(([\u0C80-\u0CFF\u0CCD]+)\)', ps)
    if m:
        return m.group(1)
    # If no parentheses, check if the field itself is Kannada
    if re.search(r'[\u0C80-\u0CFF]', ps):
        return ps.strip()
    return ps.strip()


def _line1(rec: dict) -> str:
    lines = rec.get("poem_kannada", "").split("\n")
    return lines[0].strip() if lines else ""


def _line2(rec: dict) -> str:
    lines = rec.get("poem_kannada", "").split("\n")
    return lines[1].strip() if len(lines) > 1 else ""


# ──────────────────────────────────────────────
# SECTION 2 ▸ GENERATION PROFILES (5 profiles)
# ──────────────────────────────────────────────
# All profiles output ONLY the poem — nothing else.

def profile_g1(rec, rng):
    """G1: Theme -> poem"""
    theme = rec.get("theme", "Nature")
    instruction = (
        f"{RAGALE_RULES}\n\n"
        f"Write a 2-line Utsaha Ragale poem in Kannada about: {theme}"
    )
    output = rec["poem_kannada"]
    return instruction, output


def profile_g2(rec, rng):
    """G2: Theme + prasa constraint -> poem"""
    theme = rec.get("theme", "Nature")
    prasa = _extract_prasa(rec)
    instruction = (
        f"{RAGALE_RULES}\n\n"
        f"Compose a Ragale poem about: {theme}. "
        f"The Adi Prasa consonant (2nd syllable) must be '{prasa}'."
    )
    output = rec["poem_kannada"]
    return instruction, output


def profile_g3(rec, rng):
    """G3: English meaning -> poem"""
    meaning = rec.get("meaning_english", "")
    instruction = (
        f"{RAGALE_RULES}\n\n"
        f"The following is the meaning of a Ragale poem in English. "
        f"Generate the Kannada Ragale poem.\n\n"
        f"English Meaning: {meaning}"
    )
    output = rec["poem_kannada"]
    return instruction, output


def profile_g4(rec, rng):
    """G4: Kannada meaning -> poem"""
    meaning = rec.get("meaning_kannada", "")
    instruction = (
        f"{RAGALE_RULES}\n\n"
        f"ಕೆಳಗಿನ ಅರ್ಥಕ್ಕೆ ಅನುಗುಣವಾಗಿ ಉತ್ಸಾಹ ರಗಳೆ ಬರೆಯಿರಿ.\n\n"
        f"ಅರ್ಥ: {meaning}"
    )
    output = rec["poem_kannada"]
    return instruction, output


def profile_g5(rec, rng):
    """G5: Generic prompt -> poem"""
    prompt = rng.choice(GENERIC_PROMPTS)
    instruction = f"{RAGALE_RULES}\n\n{prompt}"
    output = rec["poem_kannada"]
    return instruction, output


PROFILES = {
    "G1": {"weight": 0.20, "builder": profile_g1, "desc": "Theme -> poem"},
    "G2": {"weight": 0.15, "builder": profile_g2, "desc": "Theme + prasa -> poem"},
    "G3": {"weight": 0.20, "builder": profile_g3, "desc": "English meaning -> poem"},
    "G4": {"weight": 0.20, "builder": profile_g4, "desc": "Kannada meaning -> poem"},
    "G5": {"weight": 0.25, "builder": profile_g5, "desc": "Generic prompt -> poem"},
}


# ──────────────────────────────────────────────
# SECTION 3 ▸ PROFILE ASSIGNMENT + EXPANSION
# ──────────────────────────────────────────────

def assign_profiles(rec: dict, rng: random.Random,
                    num_profiles: int = 3) -> list[dict]:
    """
    Assign multiple profiles to a single poem record.
    Returns a list of training examples (one per assigned profile).
    """
    profile_ids = list(PROFILES.keys())
    weights = [PROFILES[pid]["weight"] for pid in profile_ids]

    # Sample without replacement (up to available profiles)
    n = min(num_profiles, len(profile_ids))
    selected = []
    remaining_ids = list(profile_ids)
    remaining_weights = list(weights)

    for _ in range(n):
        chosen = rng.choices(remaining_ids, weights=remaining_weights, k=1)[0]
        selected.append(chosen)
        idx = remaining_ids.index(chosen)
        remaining_ids.pop(idx)
        remaining_weights.pop(idx)

    examples = []
    for pid in selected:
        builder = PROFILES[pid]["builder"]
        instruction, output = builder(rec, rng)
        examples.append({
            "input": instruction,
            "output": output,
            "profile_id": pid,
            "theme": rec.get("theme", ""),
        })

    return examples


# ──────────────────────────────────────────────
# SECTION 4 ▸ TRAIN / VAL / TEST SPLIT
# ──────────────────────────────────────────────

def split_data(records: list[dict], seed: int = 42,
               train_ratio: float = 0.80,
               val_ratio: float = 0.10):
    """
    Split poem records 80/10/10 BEFORE profile expansion.
    Returns (train_poems, val_poems, test_poems).
    """
    rng = random.Random(seed)
    shuffled = list(records)
    rng.shuffle(shuffled)

    n = len(shuffled)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    return shuffled[:train_end], shuffled[train_end:val_end], shuffled[val_end:]


# ──────────────────────────────────────────────
# SECTION 5 ▸ FORMAT CONVERTERS
# ──────────────────────────────────────────────

def to_alpaca(rec: dict) -> dict:
    """Alpaca format for Unsloth / TRL."""
    return {
        "instruction": rec.get("input", ""),
        "input": "",
        "output": rec.get("output", ""),
        "_profile_id": rec.get("profile_id"),
        "_theme": rec.get("theme", ""),
    }


def to_sharegpt(rec: dict) -> dict:
    """ShareGPT conversation format for Axolotl / LLaMA-Factory."""
    return {
        "conversations": [
            {"from": "system", "value": SYSTEM_PROMPT},
            {"from": "human", "value": rec.get("input", "")},
            {"from": "gpt", "value": rec.get("output", "")},
        ],
        "_profile_id": rec.get("profile_id"),
        "_theme": rec.get("theme", ""),
    }


def to_trl(rec: dict) -> dict:
    """TRL SFTTrainer native messages format."""
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": rec.get("input", "")},
            {"role": "assistant", "content": rec.get("output", "")},
        ],
        "_profile_id": rec.get("profile_id"),
        "_theme": rec.get("theme", ""),
    }


def print_stats(records: list[dict]):
    """Print profile distribution stats."""
    profile_counts = {}
    for r in records:
        pid = r.get("_profile_id") or r.get("profile_id", "?")
        profile_counts[pid] = profile_counts.get(pid, 0) + 1

    total = len(records)
    print(f"\n-- Profile Distribution ({total} records) --")
    for pid in sorted(profile_counts):
        count = profile_counts[pid]
        desc = PROFILES.get(pid, {}).get("desc", "")
        bar = "#" * int(count / total * 40)
        print(f"  {pid:>3}: {count:>5} ({count/total*100:4.1f}%)  {bar}  {desc}")
    print("-" * 60)


# ──────────────────────────────────────────────
# SECTION 6 ▸ FINETUNE (LoRA with peft + trl)
# ──────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_DIR = os.path.dirname(SCRIPT_DIR)  # ragale_pipeline/

DEFAULT_MODEL = "google/gemma-3-1b-it"


def format_chat(example):
    """Convert alpaca {instruction, input, output} into Gemma chat format."""
    user_content = example["instruction"]
    if example.get("input", "").strip():
        user_content += "\n" + example["input"]
    return (
        f"<start_of_turn>user\n"
        f"{user_content}<end_of_turn>\n"
        f"<start_of_turn>model\n"
        f"{example['output']}<end_of_turn>"
    )


def run_finetune(dataset_path, max_seq_len=512, lora_rank=16,
                 lora_alpha=32, batch_size=2, grad_accum=8,
                 epochs=4, lr=1e-4, merge=False,
                 resume_from=None):
    """
    Local LoRA fine-tuning for Gemma 3 1B IT on Ragale data.
    Uses peft + transformers + trl (same stack as dwipada training).
    """
    import torch
    from datasets import load_dataset
    from peft import LoraConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    checkpoint_dir = os.path.join(PIPELINE_DIR, "ragale_checkpoints")
    adapter_dir = os.path.join(PIPELINE_DIR, "ragale_lora_adapter")
    log_dir = os.path.join(PIPELINE_DIR, "ragale_logs")

    print("=" * 60)
    print("Gemma 3 LoRA Fine-Tuning on Kannada Ragale IFT Dataset")
    print("=" * 60)
    print(f"  Model:          {DEFAULT_MODEL}")
    print(f"  Dataset:        {dataset_path}")
    print(f"  Output:         {checkpoint_dir}")
    print(f"  Epochs:         {epochs}")
    print(f"  Batch size:     {batch_size} (effective: {batch_size * grad_accum})")
    print(f"  Learning rate:  {lr}")
    print(f"  LoRA rank:      {lora_rank}")
    print(f"  LoRA alpha:     {lora_alpha}")
    print(f"  Max seq length: {max_seq_len}")
    print()

    # -- Load and split dataset --
    raw = load_dataset("json", data_files=dataset_path, split="train")
    split = raw.train_test_split(test_size=0.1, seed=42)
    train_dataset = split["train"]
    val_dataset = split["test"]
    print(f"  Dataset loaded: {len(train_dataset)} train, {len(val_dataset)} val")

    # -- Load model and tokenizer --
    print(f"\n  Loading model: {DEFAULT_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        DEFAULT_MODEL,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    )
    model.config.use_cache = False

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model loaded: {total_params / 1e6:.1f}M parameters")

    # -- LoRA configuration --
    peft_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        bias="none",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        task_type="CAUSAL_LM",
    )

    # -- Training arguments --
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    training_args = SFTConfig(
        output_dir=checkpoint_dir,

        # Sequence
        max_length=max_seq_len,
        packing=True,

        # Batch
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,

        # Duration
        num_train_epochs=epochs,

        # Optimizer
        optim="adamw_torch_fused",
        learning_rate=lr,
        weight_decay=0.01,
        max_grad_norm=1.0,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",

        # Precision
        bf16=True,
        fp16=False,

        # Memory
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},

        # Logging
        logging_steps=25,
        logging_dir=log_dir,
        report_to="tensorboard",

        # Saving
        save_strategy="steps",
        save_steps=500,
        save_total_limit=3,

        # Evaluation
        eval_strategy="steps",
        eval_steps=500,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",

        # Misc
        seed=42,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
    )

    # -- Train --
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
        formatting_func=format_chat,
    )

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {trainable_params / 1e6:.2f}M "
          f"({100 * trainable_params / total_params:.2f}%)")
    print("\n  Starting training...")

    trainer.train(resume_from_checkpoint=resume_from)

    # -- Save LoRA adapter --
    final_dir = os.path.join(adapter_dir, "final")
    print(f"\n  Saving LoRA adapter to {final_dir}")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)

    # -- Optional merge --
    if merge:
        print("\n  Merging LoRA adapter into base model...")
        merged_dir = os.path.join(PIPELINE_DIR, "ragale_merged_model")

        base_model = AutoModelForCausalLM.from_pretrained(
            DEFAULT_MODEL,
            dtype=torch.bfloat16,
            device_map="cpu",
        )
        peft_model = PeftModel.from_pretrained(base_model, final_dir)
        merged_model = peft_model.merge_and_unload()
        merged_model.save_pretrained(merged_dir)
        tokenizer.save_pretrained(merged_dir)
        print(f"  Merged model saved to {merged_dir}")

    print("\n  Training complete!")


# ──────────────────────────────────────────────
# SECTION 7 ▸ CLI SUBCOMMANDS
# ──────────────────────────────────────────────

def cmd_prepare(args):
    """Prepare IFT dataset from ragale_consolidated_filtered.json."""
    print(f"\n[Loading] {args.input}")
    records = load_jsonl(args.input)
    print(f"  Loaded {len(records)} poems.")

    # Validate required fields
    required = {"poem_kannada", "meaning_kannada", "meaning_english",
                "theme", "prasa_syllable"}
    missing = required - set(records[0].keys()) if records else set()
    if missing:
        print(f"  WARNING: Missing fields in data: {missing}")

    # Split poems before expansion
    train_poems, val_poems, test_poems = split_data(records, seed=42)
    print(f"\n  Split: {len(train_poems)} train / "
          f"{len(val_poems)} val / {len(test_poems)} test poems")

    # Expand with profiles
    rng = random.Random(42)
    num_profiles = args.num_profiles

    train_examples = []
    for rec in train_poems:
        for ex in assign_profiles(rec, rng, num_profiles):
            ex["split"] = "train"
            train_examples.append(ex)

    val_examples = []
    for rec in val_poems:
        for ex in assign_profiles(rec, rng, num_profiles):
            ex["split"] = "val"
            val_examples.append(ex)

    test_examples = []
    for rec in test_poems:
        # Test set: 1 profile per poem (G1 — theme to poem)
        instruction, output = profile_g1(rec, rng)
        test_examples.append({
            "input": instruction,
            "output": output,
            "profile_id": "G1",
            "theme": rec.get("theme", ""),
            "split": "test",
        })

    all_examples = train_examples + val_examples + test_examples
    print(f"\n  Expanded: {len(train_examples)} train / "
          f"{len(val_examples)} val / {len(test_examples)} test examples")

    # Save
    outdir = Path(args.outdir)
    save_jsonl(all_examples, str(outdir / "training_ready.jsonl"))
    save_jsonl(train_examples, str(outdir / "train.jsonl"))
    save_jsonl(val_examples, str(outdir / "val.jsonl"))
    save_jsonl(test_examples, str(outdir / "test.jsonl"))

    print_stats(all_examples)

    # Save data stats
    stats = {
        "total_poems": len(records),
        "train_poems": len(train_poems),
        "val_poems": len(val_poems),
        "test_poems": len(test_poems),
        "profiles_per_poem": num_profiles,
        "train_examples": len(train_examples),
        "val_examples": len(val_examples),
        "test_examples": len(test_examples),
        "total_examples": len(all_examples),
    }
    stats_path = str(outdir / "data_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Stats saved to: {stats_path}")
    print("\nDone.")


def cmd_convert(args):
    """Convert training_ready.jsonl to alpaca/sharegpt/trl formats."""
    print(f"\n[Loading] {args.input}")
    records = load_jsonl(args.input)

    # Filter to train+val only (exclude test)
    train_val = [r for r in records if r.get("split") != "test"]
    print(f"  Loaded {len(records)} records, using {len(train_val)} (train+val)")

    alpaca_records = [to_alpaca(r) for r in train_val]
    sharegpt_records = [to_sharegpt(r) for r in train_val]
    trl_records = [to_trl(r) for r in train_val]

    outdir = Path(args.outdir)
    print("\n[Saving IFT files...]")
    save_jsonl(alpaca_records, str(outdir / "ift_alpaca.jsonl"))
    save_jsonl(sharegpt_records, str(outdir / "ift_sharegpt.jsonl"))
    save_jsonl(trl_records, str(outdir / "ift_trl.jsonl"))

    print_stats(alpaca_records)

    print("\n-- Framework -> File mapping --")
    print("  Unsloth                 ->  ift_alpaca.jsonl")
    print("  Axolotl (alpaca)        ->  ift_alpaca.jsonl")
    print("  Axolotl (sharegpt)      ->  ift_sharegpt.jsonl")
    print("  LLaMA-Factory           ->  ift_sharegpt.jsonl")
    print("  HuggingFace TRL         ->  ift_trl.jsonl")
    print("\nDone.")


def cmd_finetune(args):
    """Run LoRA fine-tuning locally."""
    run_finetune(
        dataset_path=args.dataset,
        max_seq_len=args.max_seq_len,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        epochs=args.epochs,
        lr=args.lr,
        merge=args.merge,
        resume_from=args.resume_from,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Ragale IFT Pipeline — Unified Tool"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # -- prepare --
    p_prep = sub.add_parser("prepare",
                            help="Prepare IFT dataset from filtered poems")
    p_prep.add_argument("--input", required=True,
                        help="Path to ragale_consolidated_filtered.json")
    p_prep.add_argument("--outdir", default="./ift_data",
                        help="Output directory (default: ./ift_data)")
    p_prep.add_argument("--num-profiles", type=int, default=3,
                        help="Profiles per poem for expansion (default: 3)")

    # -- convert --
    p_conv = sub.add_parser("convert",
                            help="Convert training_ready.jsonl to IFT formats")
    p_conv.add_argument("--input", required=True,
                        help="Path to training_ready.jsonl")
    p_conv.add_argument("--outdir", default="./ift_data",
                        help="Output directory (default: ./ift_data)")

    # -- finetune --
    p_ft = sub.add_parser("finetune",
                          help="LoRA fine-tune Gemma 3 1B IT (local)")
    p_ft.add_argument("--dataset", required=True,
                      help="Path to ift_alpaca.jsonl")
    p_ft.add_argument("--max-seq-len", type=int, default=512,
                      help="Max sequence length (default: 512)")
    p_ft.add_argument("--lora-rank", type=int, default=16,
                      help="LoRA rank (default: 16)")
    p_ft.add_argument("--lora-alpha", type=int, default=32,
                      help="LoRA alpha (default: 32)")
    p_ft.add_argument("--batch-size", type=int, default=2,
                      help="Per-device batch size (default: 2)")
    p_ft.add_argument("--grad-accum", type=int, default=8,
                      help="Gradient accumulation steps (default: 8)")
    p_ft.add_argument("--epochs", type=int, default=6,
                      help="Training epochs (default: 6)")
    p_ft.add_argument("--lr", type=float, default=1e-4,
                      help="Learning rate (default: 1e-4)")
    p_ft.add_argument("--merge", action="store_true",
                      help="Merge LoRA adapter into base model after training")
    p_ft.add_argument("--resume-from", type=str, default=None,
                      help="Resume from checkpoint directory")

    args = parser.parse_args()

    if args.command == "prepare":
        cmd_prepare(args)
    elif args.command == "convert":
        cmd_convert(args)
    elif args.command == "finetune":
        cmd_finetune(args)


if __name__ == "__main__":
    main()
