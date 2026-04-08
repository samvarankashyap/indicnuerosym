"""
Ragale Gemma 3 1B — LoRA Fine-tuning Script
=============================================
Standalone fine-tuning script for Kannada Utsaha Ragale poem generation.
Uses peft + transformers + trl (same stack as dwipada training).
All outputs stay local — no HuggingFace Hub push.

Requirements:
    pip install peft transformers trl datasets

Usage:
    python ragale_gemma3_finetune.py --dataset ./ift_data/ift_alpaca.jsonl
    python ragale_gemma3_finetune.py --dataset ./ift_data/ift_alpaca.jsonl \
        --epochs 4 --lora-rank 16 --max-seq-len 512

    # Resume from checkpoint:
    python ragale_gemma3_finetune.py --dataset ./ift_data/ift_alpaca.jsonl \
        --resume-from ../ragale_checkpoints/checkpoint-500

    # Merge LoRA into base model after training:
    python ragale_gemma3_finetune.py --dataset ./ift_data/ift_alpaca.jsonl --merge
"""

import os
import argparse
from pathlib import Path


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


def run_finetune(dataset_path, max_seq_len=512, lora_rank=32,
                 lora_alpha=64, batch_size=2, grad_accum=8,
                 epochs=6, lr=5e-5, merge=False,
                 resume_from=None):
    """
    Full LoRA fine-tuning pipeline for Gemma 3 1B IT on Ragale data.
    Uses peft + transformers + trl.
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
        warmup_ratio=0.06,
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ragale Gemma 3 1B — LoRA Fine-tuning (Local)"
    )
    parser.add_argument(
        "--dataset", required=True,
        help="Path to ift_alpaca.jsonl"
    )
    parser.add_argument(
        "--max-seq-len", type=int, default=512,
        help="Max sequence length (default: 512)"
    )
    parser.add_argument(
        "--lora-rank", type=int, default=32,
        help="LoRA rank (default: 32)"
    )
    parser.add_argument(
        "--lora-alpha", type=int, default=64,
        help="LoRA alpha (default: 64)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=2,
        help="Per-device batch size (default: 2)"
    )
    parser.add_argument(
        "--grad-accum", type=int, default=8,
        help="Gradient accumulation steps (default: 8)"
    )
    parser.add_argument(
        "--epochs", type=int, default=6,
        help="Number of training epochs (default: 6)"
    )
    parser.add_argument(
        "--lr", type=float, default=5e-5,
        help="Learning rate (default: 5e-5)"
    )
    parser.add_argument(
        "--merge", action="store_true",
        help="Merge LoRA adapter into base model after training"
    )
    parser.add_argument(
        "--resume-from", type=str, default=None,
        help="Resume from checkpoint directory"
    )
    args = parser.parse_args()

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
