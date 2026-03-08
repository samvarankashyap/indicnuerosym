#!/usr/bin/env python3
"""
LoRA fine-tuning script for Gemma 3 1B IT on Telugu Dwipada using TRL messages format.

Trains on datasets/ift_trl_data.jsonl (TRL messages format with system/user/assistant)
using plain LoRA + bfloat16, identical setup to train_ift.py but with TRL messages dataset.

Usage:
    # Default run (5 epochs):
    python -m dwipada.training.train_trl

    # Smoke test (10 steps):
    python -m dwipada.training.train_trl --max_steps 10

    # Resume from checkpoint:
    python -m dwipada.training.train_trl --resume_from checkpoints/gemma3-dwipada-trl/checkpoint-500

    # Merge LoRA adapter into base model after training:
    python -m dwipada.training.train_trl --merge
"""

import argparse
import os
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from dwipada.paths import DATASETS_DIR, CHECKPOINTS_DIR, LOGS_DIR

DEFAULT_MODEL      = "google/gemma-3-1b-it"
DEFAULT_OUTPUT_DIR = CHECKPOINTS_DIR / "gemma3-dwipada-trl"
DEFAULT_LOG_DIR    = LOGS_DIR / "gemma3-dwipada-trl"
TRL_DATASET_PATH   = DATASETS_DIR / "ift_trl_data.jsonl"


def parse_args():
    parser = argparse.ArgumentParser(
        description="LoRA fine-tune Gemma 3 1B IT on Telugu Dwipada (TRL messages format)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Base model ID (default: {DEFAULT_MODEL})",
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
        default=5,
        help="Number of training epochs (default: 5)",
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
        default=1024,
        help="Maximum sequence length (default: 1024)",
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
    parser.add_argument(
        "--restart_lr",
        action="store_true",
        help="Reset the LR scheduler when resuming (fresh cosine schedule for remaining steps)",
    )
    return parser.parse_args()


def strip_metadata(example):
    """Keep only the 'messages' field, dropping underscore-prefixed metadata keys."""
    return {"messages": example["messages"]}


def main():
    args = parse_args()

    print("=" * 60)
    print("Gemma 3 LoRA Fine-Tuning on Telugu Dwipada (TRL format)")
    print("=" * 60)
    print(f"Model:          {args.model}")
    print(f"Dataset:        {TRL_DATASET_PATH}")
    print(f"Output:         {args.output_dir}")
    print(f"Epochs:         {args.epochs}")
    print(f"Batch size:     {args.batch_size} (effective: {args.batch_size * args.grad_accum})")
    print(f"Learning rate:  {args.lr}")
    print(f"LoRA rank:      {args.lora_rank}")
    print(f"Max seq length: {args.max_seq_length}")
    if args.max_steps > 0:
        print(f"Max steps:      {args.max_steps} (smoke test)")
    print()

    # -- Load and split dataset --
    if not TRL_DATASET_PATH.exists():
        print(f"Error: {TRL_DATASET_PATH} not found.")
        return

    raw = load_dataset("json", data_files=str(TRL_DATASET_PATH), split="train")
    raw = raw.map(strip_metadata, remove_columns=[c for c in raw.column_names if c != "messages"])
    split = raw.train_test_split(test_size=0.1, seed=42)
    train_dataset = split["train"]
    val_dataset   = split["test"]
    print(f"Dataset loaded: {len(train_dataset)} train, {len(val_dataset)} val")

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
    model.config.use_cache = False

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
    )

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params / 1e6:.2f}M "
          f"({100 * trainable_params / total_params:.2f}%)")
    print("\nStarting training...")

    if args.restart_lr and args.resume_from:
        # First, do a normal resume to restore model weights + optimizer state
        # but we need to intercept after checkpoint loading to reset the scheduler.
        # We achieve this by subclassing the callback mechanism:
        # 1. Load checkpoint to get global_step
        # 2. Reset scheduler for remaining steps with fresh cosine warmup
        import json
        state_file = Path(args.resume_from) / "trainer_state.json"
        with open(state_file) as f:
            saved_state = json.load(f)
        resumed_step = saved_state["global_step"]
        total_steps = trainer.args.max_steps if trainer.args.max_steps > 0 else int(
            len(trainer.get_train_dataloader()) * trainer.args.num_train_epochs
        )
        remaining_steps = total_steps - resumed_step
        print(f"\n--restart_lr: Will reset cosine LR schedule over {remaining_steps} remaining steps")
        print(f"  Resumed at step {resumed_step}, total steps {total_steps}")

        original_create_scheduler = trainer.create_scheduler

        def patched_create_scheduler(num_training_steps, optimizer=None):
            original_create_scheduler(num_training_steps=remaining_steps, optimizer=optimizer)

        trainer.create_scheduler = patched_create_scheduler

    trainer.train(resume_from_checkpoint=args.resume_from)

    # -- Save --
    final_dir = os.path.join(args.output_dir, "final")
    print(f"\nSaving LoRA adapter to {final_dir}")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)

    # -- Optional merge --
    if args.merge:
        print("\nMerging LoRA adapter into base model...")
        merged_dir = str(Path(args.output_dir).parent / "gemma3-dwipada-trl-merged")

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
