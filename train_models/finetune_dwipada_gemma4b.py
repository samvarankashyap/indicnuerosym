#!/usr/bin/env python3
"""
Standalone QLoRA fine-tuning script for Gemma 3 4B IT on Telugu Dwipada poem generation.

Optimized for NVIDIA RTX 5050 (8GB VRAM) with 16 CPU cores and 23GB RAM.
Uses 4-bit quantization (NF4) + LoRA + gradient checkpointing + sequence packing.

Usage:
    # Full training run (8 epochs):
    python finetune_dwipada_gemma4b.py

    # Smoke test (10 steps):
    python finetune_dwipada_gemma4b.py --max_steps 10

    # Resume from checkpoint:
    python finetune_dwipada_gemma4b.py --resume_from checkpoints_gemma4b/checkpoint-500

    # Train and merge adapter into base model:
    python finetune_dwipada_gemma4b.py --merge

    # Custom hyperparameters:
    python finetune_dwipada_gemma4b.py --epochs 12 --lr 1e-4 --lora_rank 32 --batch_size 2
"""

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
from transformers import AutoTokenizer, BitsAndBytesConfig, Gemma3ForCausalLM, Gemma3ForConditionalGeneration
from trl import SFTConfig, SFTTrainer

# ── Paths (all relative to this script's directory) ──────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "ift_data"
CHECKPOINT_DIR = SCRIPT_DIR / "checkpoints_gemma4b"
ADAPTER_DIR = SCRIPT_DIR / "dwipada_lora_adapter_gemma4b"
MERGED_DIR = SCRIPT_DIR / "dwipada_merged_model_gemma4b"
LOG_DIR = SCRIPT_DIR / "logs_gemma4b"

DEFAULT_MODEL = "google/gemma-3-4b-it"
DEFAULT_DATASET = DATA_DIR / "ift_generation_trl.jsonl"


def parse_args():
    p = argparse.ArgumentParser(
        description="QLoRA fine-tune Gemma 3 4B IT for Telugu Dwipada poem generation"
    )

    # Model & data
    p.add_argument("--model", type=str, default=DEFAULT_MODEL,
                   help=f"Base model HF ID (default: {DEFAULT_MODEL})")
    p.add_argument("--dataset", type=str, default=str(DEFAULT_DATASET),
                   help="Path to TRL-format JSONL dataset")

    # Training duration
    p.add_argument("--epochs", type=int, default=8,
                   help="Number of training epochs (default: 8)")
    p.add_argument("--max_steps", type=int, default=-1,
                   help="Max steps (-1 = use epochs). Set to 10 for smoke tests.")

    # Batch & sequence
    p.add_argument("--batch_size", type=int, default=1,
                   help="Per-device train batch size (default: 1)")
    p.add_argument("--grad_accum", type=int, default=32,
                   help="Gradient accumulation steps (default: 32, effective batch=32)")
    p.add_argument("--max_seq_length", type=int, default=256,
                   help="Max sequence length (default: 256)")

    # Optimizer
    p.add_argument("--lr", type=float, default=2e-4,
                   help="Learning rate (default: 2e-4)")
    p.add_argument("--weight_decay", type=float, default=0.01,
                   help="Weight decay (default: 0.01)")
    p.add_argument("--warmup_steps", type=int, default=120,
                   help="Warmup steps (default: 120)")

    # LoRA
    p.add_argument("--lora_rank", type=int, default=16,
                   help="LoRA rank (default: 16)")
    p.add_argument("--lora_alpha", type=int, default=32,
                   help="LoRA alpha (default: 32)")
    p.add_argument("--lora_dropout", type=float, default=0.05,
                   help="LoRA dropout (default: 0.05)")

    # Checkpointing
    p.add_argument("--save_steps", type=int, default=50,
                   help="Save checkpoint every N steps (default: 50)")
    p.add_argument("--eval_steps", type=int, default=50,
                   help="Evaluate every N steps (default: 50)")
    p.add_argument("--save_total_limit", type=int, default=10,
                   help="Keep best N checkpoints (default: 10)")

    # Resume & merge
    p.add_argument("--resume_from", type=str, default=None,
                   help="Resume from checkpoint directory")
    p.add_argument("--merge", action="store_true",
                   help="Merge LoRA adapter into base model after training")

    # Workers
    p.add_argument("--num_workers", type=int, default=4,
                   help="Dataloader workers (default: 4)")

    return p.parse_args()


def print_banner(args):
    eff_batch = args.batch_size * args.grad_accum
    print("=" * 70)
    print("  Dwipada Poem Generation — QLoRA Fine-Tuning (Gemma 3 4B IT)")
    print("=" * 70)
    print(f"  Model:            {args.model}")
    print(f"  Quantization:     4-bit NF4 (double quant)")
    print(f"  Dataset:          {args.dataset}")
    print(f"  Epochs:           {args.epochs}")
    print(f"  Batch size:       {args.batch_size} x {args.grad_accum} accum = {eff_batch} effective")
    print(f"  Max seq length:   {args.max_seq_length}")
    print(f"  Learning rate:    {args.lr}")
    print(f"  LoRA:             r={args.lora_rank}, alpha={args.lora_alpha}")
    print(f"  Checkpoints:      {CHECKPOINT_DIR}")
    print(f"  Final adapter:    {ADAPTER_DIR}")
    print(f"  Logs:             {LOG_DIR}")
    if args.max_steps > 0:
        print(f"  Max steps:        {args.max_steps} (smoke test)")
    print("=" * 70)


def print_gpu_info():
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        used = torch.cuda.memory_allocated() / 1e9
        total = props.total_memory / 1e9
        print(f"\n  GPU:    {props.name}")
        print(f"  VRAM:   {used:.2f} / {total:.2f} GB")
    else:
        print("\n  WARNING: No CUDA GPU detected. Training will be very slow on CPU.")


def get_quantization_config():
    """4-bit NF4 quantization config for QLoRA."""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def load_data(dataset_path, seed=42):
    """Load TRL-format JSONL, strip metadata columns, split 90/10."""
    print(f"\nLoading dataset: {dataset_path}")
    if not Path(dataset_path).exists():
        print(f"ERROR: Dataset not found: {dataset_path}")
        sys.exit(1)

    raw = load_dataset("json", data_files=str(dataset_path), split="train")

    # Strip underscore-prefixed metadata keys, keep only 'messages'
    metadata_cols = [c for c in raw.column_names if c.startswith("_")]
    if metadata_cols:
        raw = raw.remove_columns(metadata_cols)

    split = raw.train_test_split(test_size=0.1, seed=seed)
    print(f"  Train: {len(split['train']):,} samples")
    print(f"  Val:   {len(split['test']):,} samples")
    return split["train"], split["test"]


def load_model(model_id):
    """Load base model in 4-bit quantization with SDPA attention.

    Gemma 3 4B is a vision-language model.  We load the full checkpoint but
    cap GPU memory so the vision tower spills to CPU, then extract only the
    text-only language model (Gemma3ForCausalLM) for fine-tuning.
    """
    import gc

    print(f"\nLoading model: {model_id} (4-bit quantized, text-only)")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = get_quantization_config()

    # Load full multimodal model; cap GPU so vision tower stays on CPU
    print("  Loading full checkpoint (vision tower → CPU, language model → GPU)...")
    full_model = Gemma3ForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        max_memory={0: "5GiB", "cpu": "16GiB"},
        attn_implementation="sdpa",
    )

    # Build a text-only Gemma3ForCausalLM from the loaded multimodal model.
    # Structure: full_model.model.language_model (text backbone) + full_model.lm_head
    text_config = full_model.config.text_config
    model = Gemma3ForCausalLM(text_config)
    model.model = full_model.model.language_model
    model.lm_head = full_model.lm_head

    # Discard vision components to free memory
    full_model.model.language_model = None
    full_model.lm_head = None
    del full_model
    gc.collect()
    torch.cuda.empty_cache()
    print("  Vision tower discarded — using text-only language model.")

    # Point config back to the base model ID so PEFT can find the tokenizer config
    model.config.name_or_path = model_id

    # Prepare model for k-bit training (freeze base, enable gradient for LoRA)
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params / 1e6:.1f}M (quantized to 4-bit)")
    print_gpu_info()

    return model, tokenizer, total_params


def get_lora_config(args):
    """Create LoRA configuration targeting all linear projection layers."""
    return LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        task_type="CAUSAL_LM",
    )


def get_training_args(args):
    """Create SFTConfig optimized for RTX 5050 8GB with 4-bit quantized 4B model."""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    return SFTConfig(
        output_dir=str(CHECKPOINT_DIR),

        # ── Sequence ──
        max_length=args.max_seq_length,
        packing=False,

        # ── Batch ──
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,

        # ── Duration ──
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,

        # ── Optimizer (fused for speed) ──
        optim="adamw_torch_fused",
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        max_grad_norm=1.0,
        warmup_steps=args.warmup_steps,
        lr_scheduler_type="cosine",

        # ── Precision ──
        bf16=True,
        fp16=False,
        tf32=True,

        # ── Memory efficiency ──
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},

        # ── Logging ──
        logging_steps=25,
        logging_dir=str(LOG_DIR),  # TODO: migrate to TENSORBOARD_LOGGING_DIR env var when upgrading to transformers v5.2
        logging_first_step=True,
        report_to="tensorboard",

        # ── Saving (keep best 10) ──
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,

        # ── Evaluation ──
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        # ── Data loading ──
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=True,
        dataloader_prefetch_factor=2,

        # ── Reproducibility ──
        seed=42,
        data_seed=42,
    )


def test_inference(model, tokenizer):
    """Generate a sample dwipada poem to verify the adapter works."""
    print("\n" + "=" * 70)
    print("  Post-Training Inference Test")
    print("=" * 70)

    model.eval()

    test_prompts = [
        "క్రింది తెలుగు భావానికి అనుగుణంగా ఒక ద్విపద పద్యం రచించండి. "
        "ప్రతి పాదంలో 3 ఇంద్ర గణాలు + 1 సూర్య గణం ఉండాలి.\n\n"
        "తెలుగు భావం: శ్రీరాముడు సీతాదేవిని రక్షించుటకు లంకకు వెళ్ళెను.",
        "క్రింది తెలుగు భావానికి అనుగుణంగా ఒక ద్విపద పద్యం రచించండి. "
        "ప్రతి పాదంలో 3 ఇంద్ర గణాలు + 1 సూర్య గణం ఉండాలి.\n\n"
        "తెలుగు భావం: భగవంతుడు సర్వ ప్రాణులను రక్షించును.",
    ]

    for i, prompt in enumerate(test_prompts, 1):
        messages = [
            {"role": "system", "content": "You are a Telugu and Sanskrit scholar specialising in Dwipada poetry."},
            {"role": "user", "content": prompt},
        ]

        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )

        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        print(f"\n  Sample {i}:")
        print(f"  Prompt: {prompt[:80]}...")
        print(f"  Response:\n  {response.strip()}")

    print("\n" + "=" * 70)


def train(args):
    """Full training pipeline."""
    print_banner(args)

    # ── Enable TF32 for Ampere+ GPUs ──
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # ── Load data ──
    train_dataset, val_dataset = load_data(args.dataset)

    # ── Load model (quantized) ──
    model, tokenizer, total_params = load_model(args.model)

    # ── Configure LoRA ──
    peft_config = get_lora_config(args)

    # ── Configure training ──
    training_args = get_training_args(args)

    # ── Build trainer ──
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Trainable:  {trainable_params / 1e6:.2f}M / {total_params / 1e6:.1f}M "
          f"({100 * trainable_params / total_params:.2f}%)")

    # ── Train ──
    print("\nStarting training...\n")
    trainer.train(resume_from_checkpoint=args.resume_from)

    # ── Save final (best) adapter ──
    os.makedirs(ADAPTER_DIR, exist_ok=True)
    print(f"\nSaving best LoRA adapter to {ADAPTER_DIR}")
    trainer.save_model(str(ADAPTER_DIR))
    tokenizer.save_pretrained(str(ADAPTER_DIR))

    # ── VRAM after training ──
    if torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"  Peak VRAM usage: {peak:.2f} GB")

    # ── Inference test ──
    test_inference(trainer.model, tokenizer)

    # ── Optional merge ──
    if args.merge:
        merge_adapter(args, tokenizer)

    print("\nTraining complete!")
    print(f"  Adapter saved to:     {ADAPTER_DIR}")
    print(f"  Checkpoints:          {CHECKPOINT_DIR}")
    print(f"  TensorBoard logs:     {LOG_DIR}")
    if args.merge:
        print(f"  Merged model:         {MERGED_DIR}")
    print(f"\n  View training curves: tensorboard --logdir {LOG_DIR}")


def merge_adapter(args, tokenizer=None):
    """Merge LoRA adapter into base model and save (dequantized to bf16)."""
    print(f"\nMerging LoRA adapter into base model...")
    os.makedirs(MERGED_DIR, exist_ok=True)

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(str(ADAPTER_DIR))

    # Load full model on CPU, extract text-only language model for merging
    full_model = Gemma3ForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )
    text_config = full_model.config.text_config
    base_model = Gemma3ForCausalLM(text_config)
    base_model.model = full_model.model.language_model
    base_model.lm_head = full_model.lm_head
    del full_model
    peft_model = PeftModel.from_pretrained(base_model, str(ADAPTER_DIR))
    merged_model = peft_model.merge_and_unload()

    merged_model.save_pretrained(str(MERGED_DIR))
    tokenizer.save_pretrained(str(MERGED_DIR))
    print(f"  Merged model saved to {MERGED_DIR}")


def main():
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
