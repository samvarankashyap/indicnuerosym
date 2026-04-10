#!/usr/bin/env python3
"""
LoRA fine-tuning script for Gemma 3 1B (PT or IT) on Telugu Dwipada poetry generation.

Usage:
    # Default (gemma-3-1b-pt):
    python -m dwipada.training.train

    # Use instruction-tuned variant:
    python -m dwipada.training.train --model google/gemma-3-1b-it

    # Smoke test (10 steps):
    python -m dwipada.training.train --max_steps 10

    # Merge LoRA into base model after training:
    python -m dwipada.training.train --merge
"""

import argparse
import os
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from dwipada.paths import TRAINING_DATA_DIR, CHECKPOINTS_DIR, LOGS_DIR

DEFAULT_OUTPUT_DIR = CHECKPOINTS_DIR / "gemma3-1b-dwipada-lora"
DEFAULT_LOG_DIR = LOGS_DIR / "gemma3-1b-dwipada-lora"


def parse_args():
    parser = argparse.ArgumentParser(
        description="LoRA fine-tune Gemma 3 1B for Telugu Dwipada poetry generation"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="google/gemma-3-1b-pt",
        help="Base model ID (default: google/gemma-3-1b-pt). Also supports google/gemma-3-1b-it.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default=str(DEFAULT_LOG_DIR),
        help="Directory for TensorBoard logs",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Per-device training batch size (default: 2)",
    )
    parser.add_argument(
        "--grad_accum",
        type=int,
        default=8,
        help="Gradient accumulation steps (default: 8, effective batch = batch_size * grad_accum)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-4,
        help="Learning rate (default: 2e-4)",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=512,
        help="Maximum sequence length (default: 512)",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=16,
        help="LoRA rank (default: 16)",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
        help="LoRA alpha (default: 32)",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=-1,
        help="Max training steps (-1 = use epochs). Set to e.g. 10 for smoke tests.",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge LoRA adapter into base model after training",
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Resume training from a checkpoint directory",
    )
    return parser.parse_args()


def format_chat(example):
    """Convert {input, output} fields into Gemma chat-formatted text."""
    return (
        f"<start_of_turn>user\n"
        f"{example['input']}<end_of_turn>\n"
        f"<start_of_turn>model\n"
        f"{example['output']}<end_of_turn>"
    )


def main():
    args = parse_args()

    print("=" * 60)
    print("Gemma 3 LoRA Fine-Tuning for Telugu Dwipada Poetry")
    print("=" * 60)
    print(f"Model:          {args.model}")
    print(f"Output:         {args.output_dir}")
    print(f"Epochs:         {args.epochs}")
    print(f"Batch size:     {args.batch_size} (effective: {args.batch_size * args.grad_accum})")
    print(f"Learning rate:  {args.lr}")
    print(f"LoRA rank:      {args.lora_rank}")
    print(f"Max seq length: {args.max_seq_length}")
    if args.max_steps > 0:
        print(f"Max steps:      {args.max_steps} (smoke test)")
    print()

    # -- Load dataset --
    train_path = str(TRAINING_DATA_DIR / "train.jsonl")
    val_path = str(TRAINING_DATA_DIR / "val.jsonl")

    if not os.path.exists(train_path):
        print(f"Error: {train_path} not found. Run prepare_training_data.py first.")
        return

    dataset = load_dataset("json", data_files={
        "train": train_path,
        "validation": val_path,
    })
    print(f"Dataset loaded: {len(dataset['train'])} train, {len(dataset['validation'])} val")

    # -- Load model and tokenizer --
    print(f"\nLoading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    )
    model.config.use_cache = False  # Required for gradient checkpointing

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded: {total_params / 1e6:.1f}M parameters")

    # -- LoRA configuration --
    peft_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        task_type="CAUSAL_LM",
    )

    # -- Training arguments --
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    training_args = SFTConfig(
        output_dir=args.output_dir,

        # Sequence
        max_length=args.max_seq_length,
        packing=True,

        # Batch
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,

        # Duration
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,

        # Optimizer
        optim="adamw_torch_fused",
        learning_rate=args.lr,
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
        logging_dir=args.log_dir,
        report_to="tensorboard",

        # Saving
        save_strategy="steps",
        save_steps=500,
        save_total_limit=3,

        # Evaluation
        eval_strategy="steps",
        eval_steps=500,

        # Misc
        seed=42,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
    )

    # -- Train --
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        peft_config=peft_config,
        processing_class=tokenizer,
        formatting_func=format_chat,
    )

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params / 1e6:.2f}M "
          f"({100 * trainable_params / total_params:.2f}%)")
    print("\nStarting training...")

    trainer.train(resume_from_checkpoint=args.resume_from)

    # -- Save --
    final_dir = os.path.join(args.output_dir, "final")
    print(f"\nSaving LoRA adapter to {final_dir}")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)

    # -- Optional merge --
    if args.merge:
        print("\nMerging LoRA adapter into base model...")
        merged_dir = os.path.join(
            str(Path(args.output_dir).parent),
            "gemma3-1b-dwipada-merged"
        )

        # Reload on CPU for merging
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model,
            dtype=torch.bfloat16,
            device_map="cpu",
        )
        peft_model = PeftModel.from_pretrained(base_model, final_dir)
        merged_model = peft_model.merge_and_unload()
        merged_model.save_pretrained(merged_dir)
        tokenizer.save_pretrained(merged_dir)
        print(f"Merged model saved to {merged_dir}")

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
