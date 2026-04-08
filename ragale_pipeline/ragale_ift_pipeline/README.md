# Ragale IFT Pipeline

Instruction Fine-Tuning pipeline for Kannada Utsaha Ragale poem generation using QLoRA on Gemma 3 1B IT.

## Quick Start

```bash
# 1. Prepare IFT dataset (assigns profiles, splits train/val/test)
python ragale_ift.py prepare --input ../ragale_consolidated_filtered.json --outdir ./ift_data

# 2. Convert to training formats (alpaca/sharegpt/trl)
python ragale_ift.py convert --input ./ift_data/training_ready.jsonl --outdir ./ift_data

# 3. Fine-tune (local, no HF push)
python ragale_ift.py finetune --dataset ./ift_data/ift_alpaca.jsonl
```

## Generation Profiles

All profiles output **poem only** — no meanings, no analysis.

| ID | Weight | Input |
|----|--------|-------|
| G1 | 20% | Theme -> poem |
| G2 | 15% | Theme + prasa constraint -> poem |
| G3 | 20% | English meaning -> poem |
| G4 | 20% | Kannada meaning -> poem |
| G5 | 25% | Generic prompt -> poem |

Each poem is assigned to 3 profiles, expanding the dataset ~3x.

## Training Configuration

| Parameter | Default |
|-----------|---------|
| Model | google/gemma-3-1b-it |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| Epochs | 4 |
| Learning rate | 1e-4 |
| Max seq length | 512 |
| Batch size | 1 (grad accum 8) |

## Output Directories

```
ragale_pipeline/
    ragale_checkpoints/     # Training checkpoints
    ragale_lora_adapter/    # Final LoRA adapter
    ragale_ift_pipeline/
        ift_data/           # Generated datasets
```

## Requirements

```
# Core (prepare/convert) — no dependencies
# Fine-tuning:
pip install unsloth transformers trl datasets
```
