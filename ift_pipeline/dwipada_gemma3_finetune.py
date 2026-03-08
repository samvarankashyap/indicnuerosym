"""
Dwipada Gemma 3 4B — QLoRA Fine-tuning Script
===============================================
Converted from dwipada_gemma3_finetune.ipynb for standalone use.

Trains a QLoRA adapter on Gemma 3 4B using Unsloth for Dwipada poetry.
Works on Colab (T4 GPU) or any CUDA machine with >= 15 GB VRAM.

Requirements:
    pip install unsloth transformers trl datasets huggingface_hub

Usage:
    python dwipada_gemma3_finetune.py \
        --dataset ift_alpaca.jsonl \
        --hf-token $HF_TOKEN \
        --hf-repo  your-username/dwipada-gemma3-4b

    # Or use environment variables:
    export HF_TOKEN=hf_xxxx
    export HF_REPO=your-username/dwipada-gemma3-4b
    python dwipada_gemma3_finetune.py --dataset ift_alpaca.jsonl
"""

import json
import os
import argparse
import subprocess


ALPACA_PROMPT = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{}

### Response:
{}"""


def check_gpu():
    """Print GPU info if available."""
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        print(result.stdout)
    except FileNotFoundError:
        print("nvidia-smi not found — ensure a CUDA GPU is available.")


def load_alpaca_jsonl(path):
    """Load alpaca-format JSONL file."""
    records = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def run_finetune(dataset_path, hf_token, hf_repo,
                 max_seq_len=1024, lora_rank=32, epochs=2,
                 push_to_hub=True):
    """
    Full QLoRA fine-tuning pipeline for Gemma 3 4B on Dwipada data.

    Args:
        dataset_path: Path to ift_alpaca.jsonl
        hf_token:     HuggingFace token with WRITE scope
        hf_repo:      HuggingFace repo name (e.g. user/dwipada-gemma3-4b)
        max_seq_len:  Max sequence length (1024 for T4, 2048 for A100)
        lora_rank:    LoRA rank (32 conservative for T4)
        epochs:       Number of training epochs
        push_to_hub:  Whether to push adapter to HuggingFace Hub
    """
    # Lazy imports — these are heavy and not needed for other subcommands
    import torch
    from datasets import Dataset
    from unsloth import FastLanguageModel, is_bfloat16_supported
    from trl import SFTTrainer
    from transformers import TrainingArguments
    from huggingface_hub import login, create_repo

    # ── GPU Check ──────────────────────────────────
    check_gpu()

    # ── HuggingFace Login ──────────────────────────
    login(token=hf_token, add_to_git_credential=False)
    if push_to_hub:
        create_repo(hf_repo, repo_type='model', exist_ok=True, private=True)
    print(f"Logged in. Adapter will be pushed to: https://huggingface.co/{hf_repo}")

    # ── Load Model ─────────────────────────────────
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name     = "google/gemma-3-4b-it",
        max_seq_length = max_seq_len,
        dtype          = None,
        load_in_4bit   = True,
    )
    print(f"Model loaded. Dtype: {model.dtype}")
    print(f"VRAM used: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # ── Attach LoRA Adapters ───────────────────────
    model = FastLanguageModel.get_peft_model(
        model,
        r                           = lora_rank,
        lora_alpha                  = lora_rank * 2,
        lora_dropout                = 0.05,
        target_modules              = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias                        = "none",
        use_gradient_checkpointing  = "unsloth",
        random_state                = 42,
        use_rslora                  = False,
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable/1e6:.1f}M / {total/1e6:.0f}M "
          f"({100*trainable/total:.2f}%)")

    # ── Load & Order Dataset ───────────────────────
    raw = load_alpaca_jsonl(dataset_path)
    synthetic = [r for r in raw if r.get('_source_tag') == '[Synthetic]']
    human     = [r for r in raw if r.get('_source_tag') == '[Human_Style]']
    ordered   = synthetic + human

    print(f"Total records : {len(ordered)}")
    print(f"Synthetic     : {len(synthetic)} (first in training order)")
    print(f"Human         : {len(human)} (second in training order)")

    dataset = Dataset.from_list([
        {"instruction": r["instruction"], "output": r["output"]}
        for r in ordered
    ])

    # ── Apply Alpaca Prompt Template ───────────────
    eos_token = tokenizer.eos_token

    def format_prompts(examples):
        texts = []
        for instr, out in zip(examples["instruction"], examples["output"]):
            text = ALPACA_PROMPT.format(instr, out) + eos_token
            texts.append(text)
        return {"text": texts}

    dataset = dataset.map(format_prompts, batched=True)
    print(f"Dataset ready: {dataset}")

    # ── VRAM Check ─────────────────────────────────
    gpu_stats  = torch.cuda.get_device_properties(0)
    used_vram  = torch.cuda.memory_allocated() / 1e9
    total_vram = gpu_stats.total_memory / 1e9
    print(f"GPU           : {gpu_stats.name}")
    print(f"VRAM used     : {used_vram:.2f} GB")
    print(f"VRAM total    : {total_vram:.2f} GB")
    print(f"VRAM free     : {total_vram - used_vram:.2f} GB")

    # ── Configure Trainer ──────────────────────────
    training_args_kwargs = dict(
        per_device_train_batch_size  = 1,
        gradient_accumulation_steps  = 8,
        warmup_steps                 = 100,
        num_train_epochs             = epochs,
        learning_rate                = 2e-4,
        fp16                         = not is_bfloat16_supported(),
        bf16                         = is_bfloat16_supported(),
        logging_steps                = 50,
        optim                        = 'adamw_8bit',
        weight_decay                 = 0.01,
        lr_scheduler_type            = 'cosine',
        seed                         = 42,
        output_dir                   = './dwipada_checkpoints',
        save_strategy                = 'steps',
        save_steps                   = 500,
        save_total_limit             = 2,
        report_to                    = 'none',
    )

    if push_to_hub:
        training_args_kwargs.update(
            push_to_hub   = True,
            hub_model_id  = hf_repo,
            hub_token     = hf_token,
            hub_strategy  = 'checkpoint',
        )

    trainer = SFTTrainer(
        model              = model,
        tokenizer          = tokenizer,
        train_dataset      = dataset,
        dataset_text_field = 'text',
        max_seq_length     = max_seq_len,
        dataset_num_proc   = 2,
        packing            = True,
        args               = TrainingArguments(**training_args_kwargs),
    )

    print('Trainer configured.')
    print(f'Effective batch size : {1 * 8}')
    print(f'Max sequence length  : {max_seq_len}')
    print(f'Epochs               : {epochs}')

    # ── Train ──────────────────────────────────────
    trainer_stats = trainer.train()
    print(f"\nTraining complete.")
    print(f"Total steps     : {trainer_stats.global_step}")
    print(f"Training loss   : {trainer_stats.training_loss:.4f}")
    print(f"Runtime         : {trainer_stats.metrics['train_runtime']/3600:.2f} hours")

    # ── Quick Inference Test ───────────────────────
    FastLanguageModel.for_inference(model)

    test_poem = "భువనత్రయాధారభూతమయుండు \nపవనుండు లేకున్న బడు శరీరములు"
    test_prompt = ALPACA_PROMPT.format(
        f"[Human_Style] [Minimalist]\n"
        f"Assume role of a Telugu and Sanskrit scholar and give me bhavam and "
        f"prathipadartham of the following dwipada poem. If there are combined "
        f"words please break them with + in prathipadartham. Further bhavam "
        f"should be in single line in telugu and English. Just give only bhavam "
        f"and prathipadartham of the given input. No additional data.\n"
        f"Poem:\n\n{test_poem}",
        ""
    )

    inputs = tokenizer(test_prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(
        **inputs,
        max_new_tokens  = 512,
        temperature     = 0.7,
        top_p           = 0.9,
        do_sample       = True,
        pad_token_id    = tokenizer.eos_token_id,
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response_only = response[len(test_prompt):].strip()
    print("── Model Response ───────────────────────────")
    print(response_only)

    # ── Save & Push ────────────────────────────────
    save_dir = './dwipada_lora_adapter'
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f'Adapter saved locally to {save_dir}')

    if push_to_hub:
        print(f'Pushing to https://huggingface.co/{hf_repo} ...')
        model.push_to_hub(hf_repo, token=hf_token,
                          commit_message='Add Dwipada QLoRA adapter — final')
        tokenizer.push_to_hub(hf_repo, token=hf_token,
                              commit_message='Add tokenizer')
        print(f'\nAdapter pushed to: https://huggingface.co/{hf_repo}')

    # ── List local files ───────────────────────────
    print('\nLocal adapter files:')
    for f in sorted(os.listdir(save_dir)):
        size = os.path.getsize(f'{save_dir}/{f}') / 1e6
        print(f'  {f:<45} {size:.1f} MB')

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Dwipada Gemma 3 4B — QLoRA Fine-tuning"
    )
    parser.add_argument(
        "--dataset", required=True,
        help="Path to ift_alpaca.jsonl"
    )
    parser.add_argument(
        "--hf-token", default=os.environ.get("HF_TOKEN"),
        help="HuggingFace token (default: $HF_TOKEN env var)"
    )
    parser.add_argument(
        "--hf-repo", default=os.environ.get("HF_REPO"),
        help="HuggingFace repo name (default: $HF_REPO env var)"
    )
    parser.add_argument(
        "--max-seq-len", type=int, default=1024,
        help="Max sequence length (default: 1024, use 2048 for A100)"
    )
    parser.add_argument(
        "--lora-rank", type=int, default=32,
        help="LoRA rank (default: 32)"
    )
    parser.add_argument(
        "--epochs", type=int, default=2,
        help="Number of training epochs (default: 2)"
    )
    parser.add_argument(
        "--no-push", action="store_true",
        help="Skip pushing adapter to HuggingFace Hub"
    )
    args = parser.parse_args()

    if not args.hf_token:
        parser.error("--hf-token is required (or set HF_TOKEN env var)")
    if not args.hf_repo:
        parser.error("--hf-repo is required (or set HF_REPO env var)")

    run_finetune(
        dataset_path = args.dataset,
        hf_token     = args.hf_token,
        hf_repo      = args.hf_repo,
        max_seq_len  = args.max_seq_len,
        lora_rank    = args.lora_rank,
        epochs       = args.epochs,
        push_to_hub  = not args.no_push,
    )
