# train_models — Dvipada LoRA Fine-Tuning

LoRA fine-tuning scripts for Telugu *dvipada* generation. This is the
folder that produces the `gemma3-1b-merged` Telugu model used by the
`domino/` benchmark suite.

## Scripts

| Script | Base model | Purpose |
| --- | --- | --- |
| `finetune_dwipada.py` | `google/gemma-3-1b-it` | Main LoRA fine-tuning script for Gemma 3 1B IT on the Dvipada IFT dataset (8 epochs, rank 16, α 32, lr 2e-4, effective batch 16). The merged model produced here is the `gemma3-1b-merged` checkpoint benchmarked in the paper. |
| `finetune_dwipada_gemma4b.py` | `google/gemma-3-4b-it` | Larger 4B variant of the same flow (used for early scaling experiments) |
| `generate_dwipada.py` | merged model | Stand-alone interactive / batch generation script for sanity-checking a trained checkpoint |
| `convert_master_to_trl.py` | — | Convert `dwipada_master_dataset.json` into the TRL chat-messages format consumed by the SFTTrainer |

## Hyperparameter table (matches paper §6.1 / `tab:training-config`)

| Hyperparameter | Telugu Dvipada |
| --- | --- |
| Base model | `google/gemma-3-1b-it` |
| LoRA rank *r* | 16 |
| LoRA α | 32 |
| LoRA dropout | 0.05 |
| Target modules | all linear (`q,k,v,o,gate,up,down`) |
| Optimizer | AdamW (fused) |
| Learning rate | 2e-4 |
| LR schedule | cosine |
| Warmup ratio | 0.03 |
| Weight decay | 0.01 |
| Per-device batch | 1 |
| Grad accum steps | 16 |
| Effective batch | 16 |
| Max seq length | 384 |
| Epochs | 8 |
| Precision | bf16 + TF32 |
| Hardware | RTX 5050 8 GB |
| Seed | 42 |

The Kannada *ragale* counterpart of this script lives in
`../ragale_pipeline/ragale_ift_pipeline/ragale_gemma3_finetune.py` and
uses different defaults (rank 32, lr 5e-5, 6 epochs) — see paper
Section 6.1 for the rationale.

## Subdirectories (gitignored model artifacts)

| Directory | Content |
| --- | --- |
| `checkpoints/` | Per-step training checkpoints from `finetune_dwipada.py` |
| `checkpoints_gemma4b/` | Per-step checkpoints from the 4B variant |
| `dwipada_lora_adapter/` | Final LoRA adapter weights |
| `dwipada_merged_model/` | Adapter merged back into the base model — this is the `gemma3-1b-merged` benchmarked in `domino/` |
| `_text_only_gemma3_4b/` | Text-only-tokens variant of the 4B model (early experiment) |
| `logs/`, `logs_gemma4b/` | TensorBoard logs |
| `ift_data/` | The IFT JSONL files consumed by the trainer |

## Usage

```bash
# Full Dvipada training run (8 epochs, defaults)
python finetune_dwipada.py

# Smoke test
python finetune_dwipada.py --max_steps 10

# Resume from checkpoint
python finetune_dwipada.py --resume_from checkpoints/checkpoint-500

# Train and merge adapter into base model
python finetune_dwipada.py --merge

# Custom hyperparameters
python finetune_dwipada.py --epochs 12 --lr 1e-4 --lora_rank 32 --batch_size 2

# Generate from a trained model
python generate_dwipada.py --topic "శ్రీరాముడు సీతాదేవిని రక్షించుటకు లంకకు వెళ్ళెను"

# Convert master dataset to TRL format
python convert_master_to_trl.py
```

## Related folders

- `../datasets/` — source `dwipada_master_dataset.json` (27,881 couplets)
  consumed by `convert_master_to_trl.py`
- `../training_data/` — final 80/10/10 train/val/test JSONL splits
- `../domino/` — the benchmark suite that consumes the merged model
- `../ragale_pipeline/ragale_ift_pipeline/` — Kannada counterpart
- Paper Section 6 ("Fine-Tuning Dataset Preparation" + Training Setup
  subsection) and `tab:training-config`
