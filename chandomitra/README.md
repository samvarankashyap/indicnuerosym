# Chandomitra

**Telugu Dwipada Poetry Generation with Constrained Decoding**

Chandomitra is a neuro-symbolic system for generating metrically correct Telugu Dwipada (ద్విపద) poetry using Small Language Models. It combines classical Telugu prosody analysis with modern constrained decoding techniques to produce couplets that satisfy strict metrical rules — gana structure, prasa (rhyme), and yati (alliteration).

---

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [CLI Reference](#cli-reference)
  - [Data Collection](#1-data-collection)
  - [Dataset Preparation](#2-dataset-preparation)
  - [Training](#3-training)
  - [Generation](#4-generation)
  - [Analysis](#5-analysis)
  - [Batch Processing](#6-batch-processing)
- [End-to-End Workflow](#end-to-end-workflow)
- [Constrained Decoding](#constrained-decoding)
- [Prosody Rules](#prosody-rules)
- [Evaluation](#evaluation)

---

## Installation

**Requirements:** Python >= 3.10

```bash
# Clone the repository
git clone <repo-url> && cd chandomitra

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install in editable mode
pip install -e .
```

Key dependencies: `transformers`, `peft`, `trl`, `torch`, `datasets`, `beautifulsoup4`, `requests`, `sentence-transformers`.

---

## Quick Start

```bash
# Analyze an existing dwipada poem
dwipada analyze "రామ రామ యనరాదా రసన పారగ\nరామ నామ మహిమ దెలుపరాదా"

# Generate a poem with constrained decoding (using the pre-trained merged model)
dwipada generate-constrained "ద్విపదలో ఒక పద్యం వ్రాయండి." \
    --merged-model ./dwipada_merged_model

# Generate interactively
dwipada generate-constrained --interactive --merged-model ./dwipada_merged_model

# Or use the standalone script
python generate_dwipada.py --topic "ప్రకృతి అందాలు"
```

---

## Project Structure

```
chandomitra/
├── src/dwipada/                        # Main Python package
│   ├── __main__.py                     # python -m dwipada entry point
│   ├── cli.py                          # Unified CLI dispatcher (15 commands)
│   ├── paths.py                        # Centralized path resolution
│   │
│   ├── core/                           # Prosody analysis engine
│   │   ├── analyzer.py                 # Dwipada analyzer (match scoring 0-100%)
│   │   ├── aksharanusarika.py          # Telugu syllable splitter
│   │   └── constants.py                # Dwipada rules & constants
│   │
│   ├── data/                           # Web crawlers & text cleaners
│   │   ├── crawl_base.py              # Base crawler with shared utilities
│   │   ├── consolidate.py             # Extract couplets from raw text → JSON
│   │   ├── crawlers/                  # Source-specific crawlers
│   │   │   ├── basava_puranam.py
│   │   │   ├── dwipada_bhagavatam.py
│   │   │   ├── palanati_veera_charitra.py
│   │   │   ├── ranganatha_ramayanam.py
│   │   │   └── srirama_parinayamu.py
│   │   └── cleaners/                  # Source-specific text cleaners
│   │       ├── basava_puranam.py
│   │       ├── dwipada_bhagavatam.py
│   │       ├── palanati_veera_charitra.py
│   │       ├── poems.py
│   │       └── srirama_parinayamu.py
│   │
│   ├── dataset/                        # Dataset preparation & validation
│   │   ├── create.py                  # Extract dataset from batch responses
│   │   ├── augment.py                 # Add chandassu (prosodic) analysis
│   │   ├── combine.py                 # Merge real + synthetic datasets
│   │   ├── stats.py                   # Dataset statistics & filtering
│   │   ├── validate.py                # 4-level validation pipeline
│   │   ├── validation_utils.py        # Semantic checking (LaBSE, mSBERT)
│   │   ├── prepare_synthetic.py       # Prepare synthetic dataset
│   │   └── word_ttr.py                # Word-level type-token ratio analysis
│   │
│   ├── batch/                          # Cloud batch processing
│   │   ├── generate_requests.py       # Create Vertex AI batch requests
│   │   ├── gemini.py                  # Gemini batch API operations
│   │   ├── vertex.py                  # Vertex AI batch operations
│   │   ├── config.py                  # Batch configuration
│   │   └── client.py                  # Batch client utilities
│   │
│   └── training/                       # Model training & inference
│       ├── train.py                   # LoRA fine-tuning with SFTTrainer
│       ├── train_trl.py               # TRL-based fine-tuning variant
│       ├── train_ift.py               # Instruction fine-tuning variant
│       ├── generate.py                # Unconstrained generation
│       ├── generate_constrained.py    # Constrained generation (Algorithm 1)
│       ├── prepare_data.py            # Prepare JSONL training data
│       ├── tokenizer.py               # Tokenizer utilities
│       └── constrained/               # Constrained decoding components
│           ├── logits_processor.py    # DwipadaConstrainedLogitsProcessor
│           ├── generation_state.py    # State tracking during generation
│           ├── pattern_trie.py        # Trie of 432 valid gana patterns
│           └── syllable_utils.py      # Syllable weight computation
│
├── dwipada_merged_model/               # Pre-trained merged model (Gemma 3-1B)
├── generate_dwipada.py                 # Standalone interactive generation script
├── benchmark_chandomitra.py            # Benchmarking script (20 poems)
├── benchmark_chandomitra_n102.py       # n=102 benchmark used as the paper's external baseline (3 prompts × 34 seeds)
├── benchmark_chandomitra_n102_gemma3-1b-base.json    # n=102 results, base model
├── benchmark_chandomitra_n102_gemma3-1b-merged.json  # n=102 results, merged model
├── eval_prompts.txt                    # Evaluation prompts (6 topics)
├── eval_results*.json                  # Various evaluation results
├── chandomitra.pdf                     # Original Chandomitra paper (Jagadeeshan et al. 2026, arXiv:2506.00815)
├── dwipada_augmented_dataset.json      # Master dataset with chandassu analysis
└── pyproject.toml                      # Package metadata
```

> **Citation note.** This folder is our port of the constrained-decoding
> algorithm from Jagadeeshan et al. 2026 ("Chandomitra: Towards Generating
> Structured Sanskrit Poetry from Natural Language Inputs", arXiv:2506.00815)
> to the Telugu Dvipada rule set. The original Chandomitra targets Sanskrit
> Anuṣṭubh (99.86% on its native task); our port replaces the fixed-length
> per-pāda regex with a 432-pattern prefix trie and adds *prāsa* and *yati*
> checks. Our paper benchmarks the port as the "adapted Chandomitra"
> external baseline (1.0% / 13.7% poem accuracy on the n=102 runs above).

**Data directories** (created during pipeline execution):

| Directory | Purpose |
|-----------|---------|
| `data/` | Raw crawled text files |
| `datasets/` | Consolidated JSON datasets |
| `synthetic_data/` | Synthetic poem datasets |
| `training_data/` | JSONL train/val/test splits |
| `checkpoints/` | Model checkpoints & LoRA adapters |
| `output/` | Batch processing output |
| `logs/` | TensorBoard training logs |

---

## CLI Reference

All commands are available through the unified CLI:

```bash
dwipada <command> [options]
# or
python -m dwipada <command> [options]
```

### 1. Data Collection

**Crawl** poetry from online Telugu literature sources:

```bash
# Available sources: basava_puranam, dwipada_bhagavatam,
#   ranganatha_ramayanam, palanati_veera_charitra, srirama_parinayamu

dwipada crawl dwipada_bhagavatam
dwipada crawl ranganatha_ramayanam
```

**Clean** crawled HTML into plain text:

```bash
# Available: basava_puranam, dwipada_bhagavatam,
#   palanati_veera_charitra, srirama_parinayamu, poems

dwipada clean dwipada_bhagavatam
dwipada clean poems
```

**Consolidate** cleaned text files into structured JSON couplets:

```bash
dwipada consolidate
# Output: data/consolidated_dwipada.json
```

### 2. Dataset Preparation

**View statistics** and filter the dataset:

```bash
dwipada stats                              # Overall statistics
dwipada stats --by-source                  # Per-source breakdown
dwipada stats --write-filtered perfect.json # Export only perfect couplets
dwipada stats --exclude basava_puranam     # Exclude specific sources
```

**Augment** dataset with chandassu (prosodic) analysis:

```bash
dwipada augment
# Adds gana patterns, prasa, yati analysis to each couplet
```

**Combine** real and synthetic datasets:

```bash
dwipada combine
```

**Validate** dataset quality with a 4-level pipeline:

```bash
dwipada validate                           # Full validation
dwipada validate --skip-levels 3           # Skip semantic checking (slow)
dwipada validate --limit 100               # Validate first 100 records
dwipada validate --dataset path/to/data.json --fresh  # Custom dataset, no checkpoints
```

Validation levels:
1. **Chandass Scanner** — Prosodic structure (gana, prasa, yati)
2. **Sanity Checks** — Field completeness, length-ratio alignment
3. **Semantic Fidelity** — Cross-lingual similarity (LaBSE + mSBERT)
4. **Lexical Diversity** — Type-token ratio + duplicate detection

**Prepare** training data (80/10/10 split):

```bash
dwipada prepare
# Output: training_data/train.jsonl, val.jsonl, test.jsonl, data_stats.json
```

### 3. Training

**Fine-tune** Gemma 3-1B with LoRA:

```bash
# Default settings
dwipada train

# Custom configuration
dwipada train \
    --model google/gemma-3-1b-it \
    --epochs 3 \
    --batch-size 2 \
    --lora-rank 16

# Train and merge adapter into base model
dwipada train --merge
```

Training defaults:
- Base model: `google/gemma-3-1b-it`
- LoRA rank: 16, alpha: 32
- Epochs: 3, batch size: 2, learning rate: 2e-4
- Gradient accumulation: 8 steps
- Max sequence length: 512
- Output: `checkpoints/gemma3-1b-dwipada-lora/final`

### 4. Generation

#### Unconstrained Generation

```bash
# Single prompt
dwipada generate "ద్విపదలో ఒక పద్యం వ్రాయండి."

# Multiple poems
dwipada generate "ప్రకృతి గురించి ద్విపద వ్రాయండి." --num-poems 5

# Interactive mode
dwipada generate --interactive

# Batch mode from file
dwipada generate --batch prompts.txt

# Skip validation
dwipada generate "prompt" --no-validate
```

#### Constrained Generation

Generates poems while enforcing prosodic rules at the token level:

```bash
# Basic constrained generation
dwipada generate-constrained "ద్విపదలో ఒక పద్యం వ్రాయండి." \
    --merged-model ./dwipada_merged_model

# With LoRA adapter instead of merged model
dwipada generate-constrained "prompt" \
    --base-model google/gemma-3-1b-it \
    --adapter ./checkpoints/gemma3-1b-dwipada-lora/final

# Relax specific constraints
dwipada generate-constrained "prompt" --no-prasa    # Disable rhyme constraint
dwipada generate-constrained "prompt" --no-yati     # Disable alliteration constraint

# Adjust constraint strictness
dwipada generate-constrained "prompt" --top-k-constraint 50

# Enable sampling with temperature
dwipada generate-constrained "prompt" --do-sample --temperature 0.8

# Batch mode
dwipada generate-constrained --batch eval_prompts.txt --merged-model ./dwipada_merged_model

# Interactive
dwipada generate-constrained --interactive --merged-model ./dwipada_merged_model
```

#### Standalone Script

```bash
# Interactive generation
python generate_dwipada.py

# Topic-based
python generate_dwipada.py --topic "శ్రీరాముడు గొప్పతనం"

# Run built-in sample prompts
python generate_dwipada.py --samples
```

### 5. Analysis

Analyze any dwipada poem for metrical correctness:

```bash
# Inline (use \n to separate lines)
dwipada analyze "రామ రామ యనరాదా రసన పారగ\nరామ నామ మహిమ దెలుపరాదా"

# From file
dwipada analyze --file poem.txt

# Interactive (type lines, press Enter on empty line to finish)
dwipada analyze
```

The analyzer reports:
- Syllable breakdown (aksharalu)
- Guru/Laghu markers (U/I)
- Gana identification (Indra + Surya)
- Prasa (rhyme) check
- Yati (alliteration) check
- Overall match score (0-100%)

### 6. Batch Processing

For large-scale annotation using cloud APIs:

```bash
# Prepare batch requests
dwipada batch --prepare 100          # First 100 couplets
dwipada batch --prepare 50-150       # Range of couplets

# Submit to Gemini API
dwipada batch --submit output/batch_requests.jsonl

# Check status
dwipada batch --status <job-id>

# Download results
dwipada batch --results <job-id>
```

---

## End-to-End Workflow

The complete pipeline from raw data to poem generation:

```
1. Crawl          dwipada crawl <source>
       |
2. Clean          dwipada clean <source>
       |
3. Consolidate    dwipada consolidate
       |
4. Augment        dwipada augment
       |
5. Validate       dwipada validate
       |
6. Combine        dwipada combine
       |
7. Prepare        dwipada prepare
       |
8. Train          dwipada train --merge
       |
9. Generate       dwipada generate-constrained "prompt" --merged-model ./dwipada_merged_model
       |
10. Analyze       dwipada analyze "generated poem"
```

---

## Constrained Decoding

The constrained decoding system (Algorithm 1 from the paper) enforces Dwipada prosody rules during token generation using a custom `LogitsProcessor`. It operates in four constraint layers with soft relaxation:

1. **Telugu-only tokens** — Hard constraint: only Telugu script characters and spaces
2. **Gana structure** — Soft: valid prefix in a pre-computed trie of 432 patterns (6 Indra x 6 Indra x 6 Indra x 2 Surya)
3. **Prasa (rhyme)** — Soft: 2nd syllable consonant of line 2 must match line 1
4. **Yati (alliteration)** — Soft: 1st letter of gana 3 must match 1st letter of gana 1

If no token satisfies all constraints, the system relaxes from layer 4 upward until a valid token is found.

Key components:
- `constrained/pattern_trie.py` — Pre-computed trie of all valid I/U patterns
- `constrained/logits_processor.py` — Token filtering with layered constraints
- `constrained/generation_state.py` — Tracks position within the metrical structure
- `constrained/syllable_utils.py` — Syllable weight computation with look-ahead

---

## Prosody Rules

A **Dwipada** (ద్విపద) is a 2-line Telugu couplet. Each line must contain exactly **3 Indra ganas + 1 Surya gana**.

### Gana Types

**Indra Ganas** (3-4 syllables):

| Gana | Pattern | Example |
|------|---------|---------|
| Nala | I I I I | 4 light syllables |
| Naga | I I I U | 3 light + 1 heavy |
| Sala | I I U I | 2 light + 1 heavy + 1 light |
| Bha  | U I I   | 1 heavy + 2 light |
| Ra   | U I U   | heavy-light-heavy |
| Ta   | U U I   | 2 heavy + 1 light |

**Surya Ganas** (2-3 syllables):

| Gana | Pattern |
|------|---------|
| Na   | I I I   |
| Ha/Gala | U I |

Where **I** = Laghu (light syllable, short vowel) and **U** = Guru (heavy syllable, long vowel/anusvara/visarga).

### Constraints

- **Gana Structure**: Each line = 3 Indra + 1 Surya (432 valid combinations, 11-15 syllables per line)
- **Prasa (ప్రాస)**: The consonant of the 2nd syllable must be the same in both lines
- **Yati (యతి)**: The 1st letter of gana 1 must match the 1st letter of gana 3 (with phonetic equivalence classes)

---

## Evaluation

Run the benchmark suite:

```bash
python benchmark_chandomitra.py
# Output: benchmark_chandomitra_20poems.json
```

Evaluation prompts are in `eval_prompts.txt` (6 topics):

```
Topic: పుస్తకం. Bhaavam: జ్ఞానాన్ని ఇచ్చే నేస్తం.
Topic: ప్రకృతి. Bhaavam: ప్రకృతి అందాలు.
Topic: తల్లి. Bhaavam: తల్లి ప్రేమ గొప్పది.
Topic: విద్య. Bhaavam: విద్య వల్ల జ్ఞానం.
Topic: నది. Bhaavam: నది ప్రవాహం అందమైనది.
```

Metrics tracked: valid poem rate, average generation time, tokens per poem, constraint relaxations.

---

## License

See repository for license details.
