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

(Defaults match `ragale_gemma3_finetune.py` and the paper's
`tab:training-config`.)

| Parameter | Default |
|-----------|---------|
| Model | `google/gemma-3-1b-it` |
| LoRA rank *r* | **32** |
| LoRA α | **64** |
| LoRA dropout | 0.05 |
| Target modules | all linear (`q,k,v,o,gate,up,down`) |
| Optimizer | AdamW (fused) |
| Learning rate | **5e-5** |
| LR schedule | cosine |
| Warmup ratio | 0.06 |
| Weight decay | 0.01 |
| Per-device batch | **2** |
| Grad accum steps | 8 |
| Effective batch | 16 |
| Max seq length | 512 |
| Epochs | **6** |
| Precision | bf16 |
| Hardware | RTX 5050 8 GB |
| Seed | 42 |

The Telugu Dvipada counterpart of this trainer
(`../../train_models/finetune_dwipada.py`) uses different defaults
(rank 16, lr 2e-4, 8 epochs) — see the paper's Section 6.1 "Training
Setup" for the rationale.

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
