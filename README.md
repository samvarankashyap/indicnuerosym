# Indic-NeuroSym

**A Neuro-Symbolic Logits Processor for Strict Metrical Generation in Low-Resource SLMs**

Tools and datasets for enabling strict metrical poetry generation in Telugu using neuro-symbolic approaches with Small Language Models (SLMs). The focus is on **Dwipada** — a classical Telugu couplet form governed by Chandassu (prosodic) rules.

---

## Overview

Generating metrically correct poetry in low-resource Indic languages is challenging due to:
- Complex prosodic rules (Chandassu) governing syllable patterns
- Lack of annotated datasets for training
- Limited tokenizer support for Indic scripts in mainstream LLMs

**Indic-NeuroSym** addresses these by providing:

1. **Rule-based prosody analyzers** that validate and constrain LLM outputs
2. **Web crawlers** for building Telugu poetry corpora from public sources
3. **Symbolic constraint systems** for Dwipada meter validation (Gana, Yati, Prasa)
4. **Fine-tuning pipelines** for LoRA/IFT/TRL training on Gemma 3 1B
5. **NFA-based constrained decoding** design for meter-aware token masking
6. **LLM-as-Judge evaluation** for dataset quality assessment
7. **Unified CLI** (`dwipada`) for all pipeline stages

---

## Repository Structure

```
inlp_project/
├── pyproject.toml                         # Package config & dependencies
├── requirements.txt                       # Pip dependency list
├── README.md
│
├── src/dwipada/                           # Main Python package
│   ├── __init__.py
│   ├── __main__.py                        # python -m dwipada support
│   ├── cli.py                             # Unified CLI dispatcher
│   ├── paths.py                           # Centralized path resolution
│   │
│   ├── core/                              # Prosody analysis engine
│   │   ├── analyzer.py                    # Dwipada meter analyzer (gana/yati/prasa)
│   │   ├── aksharanusarika.py             # Telugu syllable & guru/laghu analysis
│   │   └── constants.py                   # DWIPADA_RULES_BLOCK and shared constants
│   │
│   ├── data/                              # Data acquisition & cleaning
│   │   ├── crawl_base.py                  # Shared crawler utilities (retry, HTML cleaning)
│   │   ├── clean_base.py                  # Shared cleaner utilities (char removal, line cleaning)
│   │   ├── consolidate.py                 # Text files -> consolidated JSON
│   │   ├── crawlers/                      # Per-source web crawlers
│   │   │   ├── basava_puranam.py
│   │   │   ├── dwipada_bhagavatam.py
│   │   │   ├── ranganatha_ramayanam.py
│   │   │   ├── palanati_veera_charitra.py
│   │   │   └── srirama_parinayamu.py
│   │   └── cleaners/                      # Per-source text cleaners
│   │       ├── basava_puranam.py
│   │       ├── dwipada_bhagavatam.py
│   │       ├── palanati_veera_charitra.py
│   │       ├── srirama_parinayamu.py
│   │       └── poems.py                   # General cleaner (unique logic)
│   │
│   ├── dataset/                           # Dataset preparation & validation
│   │   ├── create.py                      # Extract structured dataset from API responses
│   │   ├── augment.py                     # Add chandassu analysis metadata
│   │   ├── combine.py                     # Merge real + synthetic datasets
│   │   ├── prepare_synthetic.py           # Prepare synthetic training data
│   │   ├── stats.py                       # Metrical purity statistics
│   │   ├── validate.py                    # 4-level quality validation
│   │   └── validation_utils.py            # Embedding models & helpers
│   │
│   ├── batch/                             # Gemini / Vertex AI batch processing
│   │   ├── config.py                      # API config loading
│   │   ├── gemini.py                      # Gemini Batch API orchestrator
│   │   ├── vertex.py                      # Vertex AI batch processing
│   │   ├── client.py                      # Gemini API client
│   │   └── generate_requests.py           # Batch request JSONL generator
│   │
│   └── training/                          # Model training & inference
│       ├── prepare_data.py                # Filter & format training JSONL
│       ├── train.py                       # LoRA fine-tuning (Gemma 3 1B)
│       ├── train_ift.py                   # IFT with alpaca format
│       ├── train_trl.py                   # TRL messages format training
│       ├── generate.py                    # Poem generation with validation
│       └── tokenizer.py                   # Gemma tokenizer analysis
│
├── tests/                                 # Test suite
│   ├── test_analyzer.py                   # Prosody analysis tests
│   └── test_imports.py                    # Package import smoke tests
│
├── nfa_for_dwipada/                       # NFA constrained decoding (research)
│   ├── architecture.md                    # Full system architecture
│   ├── guru_laghu_rules.md                # Guru/Laghu classification rules
│   ├── design_codepoint_classifier.md     # FST Stage 1 design
│   ├── design_syllable_assembler.md       # FST Stage 2 design
│   ├── design_ganana_marker.md            # FST Stage 3 design
│   ├── nfa_constrained_decoding_design.md # Overall NFA design
│   ├── guru_laghu_classifier.py           # Guru/Laghu classifier implementation
│   ├── syllable_assembler.py              # Telugu syllable assembler
│   ├── ganana_marker.py                   # Gana marker implementation
│   ├── analyze_telugu_coverage.py         # Token coverage analysis
│   ├── extract_telugu_tokens.py           # Gemma Telugu token extraction
│   └── telugu_tokens.tsv                  # Extracted Telugu tokens
│
├── llm_as_judge/                          # LLM-as-Judge evaluation
│   └── generate_batch_requests_for_laj.py # Generate evaluation batch requests
│
├── dataset_validation_scripts/            # Dataset validation utilities
│   └── validation_report.txt              # Validation results
│
├── report/                                # Phase reports
│   ├── phase1.md                          # Data collection & analysis
│   ├── phase3.md                          # Dataset augmentation
│   ├── phase4.md                          # IFT dataset preparation
│   └── phase6.md                          # Training & generation
│
├── datasets/                              # Processed datasets (gitignored except below)
│   └── dwipada_augmented_dataset.json     # Final augmented dataset (29,343 entries)
│
├── generate_metadata_batch_requests.py    # Metadata batch request generator
└── run_batch_vertex.py                    # Vertex AI batch runner
```

Directories excluded from version control (see `.gitignore`):
- `data/` — Raw crawled text files (~33K couplets)
- `datasets/intermediate/` — Intermediate processed datasets
- `synthetic_data/` — LLM-generated poems
- `training_data/` — Final train/val/test JSONL splits
- `output/` — Batch API I/O files
- `checkpoints/` — Model weights and adapters
- `logs/` — TensorBoard logs
- `venv/` — Python virtual environment

---

## Installation

### Prerequisites
- Python 3.10+
- NVIDIA GPU with 8+ GB VRAM (for fine-tuning)

### Setup

```bash
git clone https://github.com/samvarankashyap/indic-neurosym.git
cd indic-neurosym

python -m venv venv
source venv/bin/activate

# Install as editable package with dev dependencies
pip install -e ".[dev]"
```

### Configuration (for Gemini/Vertex AI features)

Create a `config.yaml` in the project root:

```yaml
api_key: "your-gemini-api-key"
vertex:
  project_id: "your-gcp-project"
  location: "us-central1"
  gcs_bucket: "your-bucket"
```

---

## CLI Usage

All tools are accessible through the `dwipada` CLI:

```bash
dwipada --help
# or: python -m dwipada --help
```

### Full Pipeline

| Stage | Command | Description |
|-------|---------|-------------|
| 1. Crawl | `dwipada crawl <source>` | Fetch poems from web (e.g., `basava_puranam`, `ranganatha_ramayanam`) |
| 2. Clean | `dwipada clean <source>` | Remove punctuation, noise, and formatting artifacts |
| 3. Consolidate | `dwipada consolidate` | Merge all text files into `data/consolidated_dwipada.json` |
| 4. Stats | `dwipada stats --by-source` | Show metrical purity statistics per source |
| 5. Augment | `dwipada augment` | Add chandassu (meter) analysis to each entry |
| 6. Combine | `dwipada combine` | Merge real + synthetic datasets |
| 7. Validate | `dwipada validate` | Run 4-level quality validation |
| 8. Prepare | `dwipada prepare` | Filter metrically pure poems and format train/val/test splits |
| 9. Train | `dwipada train` | LoRA fine-tune Gemma 3 1B |
| 10. Generate | `dwipada generate "prompt"` | Generate poems with metrical validation |

### Available Sources

| Source | CLI Name | Origin |
|--------|----------|--------|
| రంగనాథ రామాయణము | `ranganatha_ramayanam` | AndhaBharati.com |
| బసవపురాణము | `basava_puranam` | te.wikisource.org |
| ద్విపద భాగవతము | `dwipada_bhagavatam` | te.wikisource.org |
| పలనాటి వీర చరిత్ర | `palanati_veera_charitra` | te.wikisource.org |
| శ్రీరమాపరిణయము | `srirama_parinayamu` | te.wikisource.org |

### Examples

```bash
# Analyze a single poem
dwipada analyze "సౌధాగ్రముల యందు సదనంబు లందు\nవీధుల యందును వెఱవొప్ప నిలిచి"

# Show stats with per-source breakdown
dwipada stats --by-source

# Prepare training data (filter 100% metrically pure, create 80/10/10 splits)
dwipada prepare

# Train with custom parameters
dwipada train --epochs 5 --batch-size 4 --lora-rank 32

# Generate poems interactively
dwipada generate --interactive

# Generate with a specific prompt
dwipada generate "ద్విపదలో ఒక పద్యం వ్రాయండి."

# Batch operations
dwipada batch --prepare 100
dwipada batch --submit output/batch_requests_100.jsonl
```

### Alternative Training Scripts

Beyond `dwipada train`, two standalone training scripts are available:

```bash
# IFT training with alpaca-format dataset
python -m dwipada.training.train_ift --max_steps 10  # smoke test
python -m dwipada.training.train_ift                  # full run

# TRL training with messages-format dataset
python -m dwipada.training.train_trl --max_steps 10
python -m dwipada.training.train_trl --merge          # merge adapter after training
```

---

## Python API

```python
from dwipada.core import analyze_dwipada, format_analysis_report, DWIPADA_RULES_BLOCK

poem = """సౌధాగ్రముల యందు సదనంబు లందు
వీధుల యందును వెఱవొప్ప నిలిచి"""

result = analyze_dwipada(poem)
print(f"Valid: {result['is_valid_dwipada']}")
print(f"Score: {result['match_score']['overall']}%")
print(f"Prasa: {result['prasa']['match']}")
print(f"Yati L1: {result['yati_line1']['match']}")

# Human-readable report
print(format_analysis_report(result))
```

### Key Functions

| Module | Function | Description |
|--------|----------|-------------|
| `dwipada.core` | `analyze_dwipada(poem)` | Full metrical analysis with 0-100% score |
| `dwipada.core` | `format_analysis_report(analysis)` | Human-readable report |
| `dwipada.core` | `analyze_pada(line)` | Single line (pada) analysis |
| `dwipada.core` | `check_prasa(line1, line2)` | Rhyme (prasa) matching |
| `dwipada.core` | `check_yati_maitri(l1, l2)` | Yati group matching |
| `dwipada.core` | `split_aksharalu(text)` | Telugu syllable splitting |
| `dwipada.core.aksharanusarika` | `akshara_ganavibhajana(...)` | Guru/Laghu marking |

---

## Data Pipeline

```
1. RAW TEXT (data/*.txt)
   ↓ crawl → clean
2. CONSOLIDATED JSON (data/consolidated_dwipada.json)
   ↓ stats → filter 100% metrical purity
3. FILTERED DATASET (datasets/intermediate/dwipada_master_filtered_perfect_dataset.json)
   ↓ augment → add chandassu analysis
4. AUGMENTED DATASET (datasets/dwipada_augmented_dataset.json)
   ↓ prepare → instruction templates + rules block + 80/10/10 split
5. TRAINING DATA (training_data/train.jsonl + val.jsonl + test.jsonl)
   ↓ train → LoRA fine-tune Gemma 3 1B
6. MODEL (checkpoints/gemma3-1b-dwipada-*/final/)
   ↓ generate → produce new poems with metrical validation
7. GENERATED POEMS (with metrical scores)
```

---

## Dwipada Meter Rules

Each Dwipada couplet = **2 lines (padas)**, each with **3 Indra ganas + 1 Surya gana**.

### Gana Types

| Type | Pattern | Name | Telugu |
|------|---------|------|--------|
| Indra | IIII | Nala | నల |
| Indra | IIIU | Naga | నగ |
| Indra | IIUI | Sala | సల |
| Indra | UII | Bha | భ |
| Indra | UIU | Ra | ర |
| Indra | UUI | Ta | త |
| Surya | III | Na | న |
| Surya | UI | Ha/Gala | హ/గల |

Where **U** = Guru (heavy syllable: long vowel, anusvara, visarga, or followed by conjunct) and **I** = Laghu (light syllable: short vowel).

### Metrical Constraints
- **Prasa**: 2nd syllable's consonant must match between both lines
- **Yati**: 1st letter of gana 1 must match 1st letter of gana 3 (maitri groups allowed)

### Scoring (0-100%)
- **Gana** (40%): 25% per valid gana × 4 ganas
- **Prasa** (35%): 2nd syllable consonant match between lines
- **Yati** (25%): 1st syllable match between gana 1 and gana 3

---

## NFA Constrained Decoding

The `nfa_for_dwipada/` directory contains the design and implementation of an NFA-based constrained decoding system that intercepts Gemma's generation at each step and masks out tokens that would violate Dwipada meter.

**Architecture (three chained FSTs → three parallel NFAs):**

1. **Codepoint Classifier** — maps Unicode characters to linguistic categories
2. **Syllable Assembler** — groups characters into complete aksharalu (syllables)
3. **Ganana Marker** — assigns Guru/Laghu labels to each syllable
4. **Gana NFA / Prasa NFA / Yati NFA** — three parallel automata enforcing metrical rules

See [nfa_for_dwipada/architecture.md](nfa_for_dwipada/architecture.md) for the full design.

---

## Dataset

### Telugu Poetry Corpus

| Dataset | Source | Couplets | 100% Pure | Purity |
|---------|--------|--------:|--------:|-------:|
| రంగనాథ రామాయణము | AndhaBharati.com | 26,296 | 21,947 | 83.5% |
| బసవపురాణము | te.wikisource.org | 2,454 | 1,872 | 76.3% |
| ద్విపద భాగవతము | te.wikisource.org | 3,157 | 2,649 | 83.9% |
| పలనాటి వీర చరిత్ర | te.wikisource.org | 783 | 66 | 8.4% |
| శ్రీరమాపరిణయము | te.wikisource.org | 392 | 377 | 96.2% |

The final augmented dataset (`datasets/dwipada_augmented_dataset.json`) contains **29,343 entries** with poem text, Telugu/English meanings, source attribution, and chandassu analysis.

---

## Running Tests

```bash
pytest tests/ -v
```

---

## LLM-as-Judge Evaluation

The `llm_as_judge/` directory contains scripts for evaluating dataset quality using an LLM as a judge. It generates Vertex AI batch requests that ask the LLM to assess Telugu and English meaning quality using a structured rubric inspired by G-Eval and GEMBA frameworks.

```bash
python llm_as_judge/generate_batch_requests_for_laj.py --sample 50
```

---

## Research Context

This project supports research in:

1. **Constrained Text Generation** — symbolic rules guiding neural language models
2. **Low-Resource NLP** — building tools for Telugu and other Indic languages
3. **Neuro-Symbolic AI** — combining neural generation with logical constraints
4. **Computational Poetics** — automated analysis and generation of metrical poetry

---

## Acknowledgments

- **Telugu Wikisource** (te.wikisource.org) for ద్విపద భాగవతము, బసవపురాణము, శ్రీరమాపరిణయము
- **AndhaBharati.com** for రంగనాథ రామాయణము
- Traditional Telugu prosody scholars for Chandassu documentation
