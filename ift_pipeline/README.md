# Dwipada IFT Pipeline

Unified tool for preprocessing, format conversion, profile assignment, and QLoRA fine-tuning of Telugu Dwipada poetry data.

## Data Flow

```
dwipada_augmented_dataset.json
        │
        ▼
┌──────────────────────────┐
│  pipeline                │   POS tagging (spaCy + Stanza)
│  (Section 1 + 2)         │   → 10 V1 prompt profiles
│                          │   → curriculum split
└──────────┬───────────────┘
           │
           ▼
    training_ready.jsonl
           │
     ┌─────┴──────┐
     ▼            ▼
┌──────────┐ ┌──────────────┐
│ convert  │ │  reassign    │   Re-assign 22 V2 profiles
│          │ │  (G1-G12 +   │   (skips POS tagging)
│          │ │   A1-A10)    │
└────┬─────┘ └──────┬───────┘
     │               │
     ▼               ▼
 ift_alpaca.jsonl   training_ready_v2.jsonl
 ift_sharegpt.jsonl training_ready_v2_stage1.jsonl
 ift_trl.jsonl      training_ready_v2_stage2.jsonl
     │
     ▼
┌──────────────────────────┐
│  finetune                │   QLoRA on Gemma 3 4B
│  (Unsloth + TRL)         │   → LoRA adapter → HuggingFace Hub
└──────────────────────────┘
```

## Files

| File | Description |
|---|---|
| `dwipada_ift.py` | Unified modular script (all subcommands) |
| `dwipada_gemma3_finetune.py` | Standalone fine-tuning script (converted from notebook) |
| `dwipada_pipeline.py` | Original: POS tagging + V1 profiles |
| `dwipada_ift_converter.py` | Original: format converter |
| `reassign_profiles.py` | Original: V2 profile reassignment |
| `dwipada_gemma3_finetune.ipynb` | Original: Colab fine-tuning notebook |

## Requirements

**Core** (all subcommands except `finetune`):
```bash
pip install tqdm
```

**Pipeline** subcommand (POS tagging):
```bash
pip install spacy stanza
python -m spacy download en_core_web_sm
python -c "import stanza; stanza.download('te')"
```

**Finetune** subcommand:
```bash
pip install unsloth transformers trl datasets huggingface_hub
```

## Usage

All subcommands are available through the unified `dwipada_ift.py`:

```bash
# Show all subcommands
python dwipada_ift.py --help
```

### 1. Pipeline — POS tag + V1 profile assignment

```bash
python dwipada_ift.py pipeline \
    --input  dwipada_augmented_dataset.json \
    --output training_ready.jsonl \
    --pos-output poems_with_pos.jsonl
```

**Outputs:**
- `training_ready.jsonl` — all records with V1 profiles assigned
- `training_ready_stage1_synthetic.jsonl` — synthetic only (Stage 1)
- `training_ready_stage2_human.jsonl` — 3x human + 10% synthetic (Stage 2)
- `poems_with_pos.jsonl` — POS-tagged checkpoint

### 2. Convert — training_ready → IFT formats

```bash
python dwipada_ift.py convert \
    --input  training_ready.jsonl \
    --outdir ./ift_data
```

**Outputs:**

| File | Framework |
|---|---|
| `ift_alpaca.jsonl` | Unsloth, Axolotl (alpaca), TRL alpaca |
| `ift_sharegpt.jsonl` | Axolotl (sharegpt), LLaMA-Factory |
| `ift_trl.jsonl` | HuggingFace TRL SFTTrainer |

### 3. Reassign — V2 profiles (22 profiles)

```bash
python dwipada_ift.py reassign \
    --input  training_ready.jsonl \
    --outdir ./v2
```

Skips POS tagging — works directly on `training_ready.jsonl`. Falls back to WTW keys if `tags_telugu` is missing.

**Outputs:**
- `training_ready_v2.jsonl` — all records with V2 profiles
- `training_ready_v2_stage1.jsonl` — synthetic only
- `training_ready_v2_stage2.jsonl` — 3x human + 10% synthetic

### 4. Finetune — QLoRA on Gemma 3 4B

```bash
# Using CLI arguments
python dwipada_ift.py finetune \
    --dataset    ift_alpaca.jsonl \
    --hf-token   hf_xxxxxxxxxxxx \
    --hf-repo    your-username/dwipada-gemma3-4b

# Using environment variables
export HF_TOKEN=hf_xxxxxxxxxxxx
export HF_REPO=your-username/dwipada-gemma3-4b
python dwipada_ift.py finetune --dataset ift_alpaca.jsonl

# Optional flags
python dwipada_ift.py finetune \
    --dataset    ift_alpaca.jsonl \
    --max-seq-len 2048 \
    --lora-rank   64 \
    --epochs      3 \
    --no-push
```

**Hardware:** T4 GPU (15 GB VRAM) minimum. Use `--max-seq-len 2048` on A100.

The standalone `dwipada_gemma3_finetune.py` script has the same interface and can be used independently.

## Profile Systems

### V1 — 10 Analysis Profiles

| ID | Type | Weight | Description |
|---|---|---|---|
| 1 | Educational | 14% | Word breakdown + meanings |
| 2 | Scholarly | 12% | Chandassu analysis (Ganas, Yati, Prasa) |
| 3 | Creative | 10% | Poem + English meaning |
| 4 | Linguistic | 10% | Sandhi deconstruction |
| 5 | Constraint | 10% | Prasa verification with proof |
| 6 | Comparative | 13% | Meanings only |
| 7 | Debugger | 8% | Error-fixing task |
| 8 | Modern | 8% | English meaning first, then poem + analysis |
| 9 | Minimalist | 13% | JSON-only output |
| 10 | GanaVariety | 2% | Detailed rhythmic scansion |

Records with fewer than 3 word-to-word meaning entries ("thin" records) are restricted to profiles 3, 6, 9.

### V2 — 22 Profiles (Generation + Analysis)

**Generation (G1-G12)** — 70% total weight:
Theme-to-poem, meaning-to-poem, line completion, keyword-constrained generation, Prasa/Yati-constrained generation.

**Analysis (A1-A10)** — 30% total weight:
Same 10 analysis styles as V1, applied to existing poems.

Records with `needs_wtw=True` profiles are excluded for thin records.

## Curriculum Strategy

Both V1 and V2 use a two-stage curriculum:

- **Stage 1** — Synthetic data only (structural tutor)
- **Stage 2** — Human data 3x upsampled + 10% synthetic rehearsal (artistic master)

The `finetune` subcommand also orders data with synthetic first, human second, to simulate curriculum learning within a single training pass.
