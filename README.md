# Indic-NeuroSym

**A Neuro-Symbolic Logits Processor for Strict Metrical Generation in Low-Resource SLMs**

This repository contains foundational tools and datasets for enabling strict metrical poetry generation in Telugu using neuro-symbolic approaches with Small Language Models (SLMs).

---

## Overview

Generating metrically correct poetry in low-resource Indic languages is challenging due to:
- Complex prosodic rules (Chandassu) governing syllable patterns
- Lack of annotated datasets for training
- Limited tokenizer support for Indic scripts in mainstream LLMs

**Indic-NeuroSym** addresses these challenges by providing:

1. **Rule-based prosody analyzers** that can validate and constrain LLM outputs
2. **Web crawlers** for building Telugu poetry corpora from public sources
3. **Symbolic constraint systems** for Dwipada meter validation (Gana, Yati, Prasa rules)
4. **Fine-tuning pipeline** for LoRA training on Gemma 3 1B
5. **Unified CLI** for all pipeline stages

---

## Project Structure

```
inlp_project/
├── pyproject.toml                         # Package metadata & dependencies
├── requirements.txt                       # Legacy dependency list
├── config.yaml                            # API keys (gitignored)
├── README.md
│
├── src/dwipada/                           # Main Python package
│   ├── __init__.py                        # Version & top-level exports
│   ├── __main__.py                        # python -m dwipada support
│   ├── cli.py                             # Unified CLI dispatcher
│   ├── paths.py                           # Centralized path resolution
│   │
│   ├── core/                              # Prosody analysis (zero external deps)
│   │   ├── analyzer.py                    # Dwipada meter analyzer (gana/yati/prasa)
│   │   ├── aksharanusarika.py             # Telugu linguistic analysis library
│   │   └── constants.py                   # Shared constants (DWIPADA_RULES_BLOCK)
│   │
│   ├── data/                              # Data acquisition & cleaning
│   │   ├── crawl_base.py                  # Shared crawler utilities
│   │   ├── clean_base.py                  # Shared cleaner utilities
│   │   ├── consolidate.py                 # Text files -> consolidated JSON
│   │   ├── crawlers/                      # Per-source crawlers
│   │   │   ├── basava_puranam.py
│   │   │   ├── dwipada_bhagavatam.py
│   │   │   ├── ranganatha_ramayanam.py
│   │   │   ├── palanati_veera_charitra.py
│   │   │   └── srirama_parinayamu.py
│   │   └── cleaners/                      # Per-source cleaners
│   │       ├── basava_puranam.py
│   │       ├── dwipada_bhagavatam.py
│   │       ├── palanati_veera_charitra.py
│   │       ├── srirama_parinayamu.py
│   │       └── poems.py                   # General cleaner (unique logic)
│   │
│   ├── dataset/                           # Dataset preparation & validation
│   │   ├── stats.py                       # Metrical purity statistics
│   │   ├── create.py                      # Extract structured dataset from API responses
│   │   ├── augment.py                     # Augment with chandassu analysis
│   │   ├── combine.py                     # Merge real + synthetic datasets
│   │   ├── prepare_synthetic.py           # Prepare synthetic training data
│   │   ├── validate.py                    # 4-level quality validation
│   │   └── validation_utils.py            # Embedding models & helpers
│   │
│   ├── batch/                             # Gemini/Vertex AI batch processing
│   │   ├── config.py                      # Shared API config loading
│   │   ├── gemini.py                      # Gemini Batch API orchestrator
│   │   ├── vertex.py                      # Vertex AI batch processing
│   │   ├── client.py                      # Gemini API client
│   │   └── generate_requests.py           # Batch request JSONL generator
│   │
│   └── training/                          # Model training & inference
│       ├── prepare_data.py                # Filter & format training JSONL
│       ├── train.py                       # LoRA fine-tuning (Gemma 3 1B)
│       ├── generate.py                    # Poem generation with validation
│       └── tokenizer.py                   # Gemma tokenizer utilities
│
├── tests/                                 # Test suite
│   ├── test_analyzer.py                   # 34 prosody analysis tests
│   └── test_imports.py                    # Package import smoke tests
│
├── data/                                  # Raw crawled text (33K+ couplets)
├── datasets/                              # Processed JSON/JSONL datasets
├── synthetic_data/                        # LLM-generated poems
├── output/                                # Batch API I/O files
├── training_data/                         # Final train/val JSONL
├── checkpoints/                           # Model artifacts
└── logs/                                  # TensorBoard logs
```

---

## Installation

### Prerequisites
- Python 3.10+
- NVIDIA GPU with 8+ GB VRAM (for fine-tuning)

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd inlp_project

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install as editable package (includes all dependencies)
pip install -e ".[dev]"
```

### Configuration (Optional - for Gemini API)

Create a `config.yaml` file with your API key:

```yaml
api_key: "your-gemini-api-key-here"
vertex:
  project_id: "your-gcp-project"
  location: "us-central1"
  gcs_bucket: "your-bucket"
```

---

## CLI Usage

All tools are accessible through a unified CLI:

```bash
# Show all commands
python -m dwipada --help
# or simply:
dwipada --help
```

### Pipeline Commands

| Stage | Command | Description |
|-------|---------|-------------|
| 1. Crawl | `dwipada crawl basava_puranam` | Fetch poems from web sources |
| 2. Clean | `dwipada clean basava_puranam` | Remove punctuation and noise |
| 3. Consolidate | `dwipada consolidate` | Combine all text files into JSON |
| 4. Stats | `dwipada stats --by-source` | Metrical purity statistics |
| 5. Augment | `dwipada augment` | Add chandassu analysis to dataset |
| 6. Combine | `dwipada combine` | Merge real + synthetic datasets |
| 7. Validate | `dwipada validate` | 4-level quality validation |
| 8. Prepare | `dwipada prepare` | Filter & format training data |
| 9. Train | `dwipada train` | LoRA fine-tune Gemma 3 1B |
| 10. Generate | `dwipada generate "ద్విపద వ్రాయండి."` | Generate poems |

### Quick Examples

```bash
# Analyze a poem
dwipada analyze "సౌధాగ్రముల యందు సదనంబు లందు\nవీధుల యందును వెఱవొప్ప నిలిచి"

# Generate poems interactively
dwipada generate --interactive

# Run batch annotation
dwipada batch --prepare 100
dwipada batch --submit output/batch_requests_100.jsonl
```

---

## Python API

```python
from dwipada.core import analyze_dwipada, DWIPADA_RULES_BLOCK

# Analyze a Dwipada couplet
poem = """సౌధాగ్రముల యందు సదనంబు లందు
వీధుల యందును వెఱవొప్ప నిలిచి"""

result = analyze_dwipada(poem)
print(f"Valid: {result['is_valid_dwipada']}")
print(f"Score: {result['match_score']['overall']}%")
print(f"Prasa: {result['prasa']['match']}")
print(f"Yati L1: {result['yati_line1']['match']}")
```

### Key Functions

| Module | Function | Description |
|--------|----------|-------------|
| `dwipada.core` | `analyze_dwipada(poem)` | Full analysis with 0-100% score |
| `dwipada.core` | `format_analysis_report(analysis)` | Human-readable report |
| `dwipada.core` | `analyze_pada(line)` | Single line analysis |
| `dwipada.core` | `check_prasa(line1, line2)` | Rhyme matching |
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
3. FILTERED DATASET (datasets/dwipada_master_filtered_perfect_dataset.json)
   ↓ augment → add chandassu analysis
4. AUGMENTED DATASET (datasets/dwipada_augmented_perfect_dataset.json)
   ↓ combine → merge with synthetic data
5. COMBINED DATASET (datasets/dwipada_augmented_dataset.json)
   ↓ validate → 4-level quality checks
6. TRAINING DATA (training_data/train.jsonl + val.jsonl)
   ↓ train → LoRA fine-tune Gemma 3 1B
7. MODEL (checkpoints/gemma3-1b-dwipada-lora/final/)
   ↓ generate → produce new poems with validation
8. GENERATED POEMS (with metrical scores)
```

### Training Data Tiers

| Tier | Source | Description |
|------|--------|-------------|
| 1 | Gemini-annotated | 100 poems with bhavam (meaning) |
| 2 | Synthetic | ~3,496 poems with titles/meanings |
| 3 | Classical corpus | ~33K poems from 5 literary works |

All poems pass through `analyze_dwipada()` -- only 100% metrically pure poems are included.

---

## Dwipada Meter Rules

### Structure
Each Dwipada = 2 lines (padas), each with 3 Indra ganas + 1 Surya gana.

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

### Scoring (0-100%)
- **Gana** (40%): 25% per valid gana x 4 ganas
- **Prasa** (35%): 2nd syllable consonant must match between lines
- **Yati** (25%): 1st letter of gana 1 must match 1st letter of gana 3

---

## Datasets

### Telugu Poetry Corpus

| Dataset | Source | Couplets | 100% Pure | Purity |
|---------|--------|--------:|--------:|-------:|
| రంగనాథ రామాయణము | AndhaBharati.com | 26,296 | 21,947 | 83.5% |
| బసవపురాణము | te.wikisource.org | 2,454 | 1,872 | 76.3% |
| ద్విపద భాగవతము | te.wikisource.org | 737 | 645 | 87.5% |
| ద్విపద భాగవతము 2 | te.wikisource.org | 2,420 | 2,004 | 82.8% |
| పలనాటి వీర చరిత్ర | te.wikisource.org | 783 | 66 | 8.4% |
| శ్రీరమాపరిణయము | te.wikisource.org | 392 | 377 | 96.2% |
| **Total** | | **33,082** | **26,911** | **81.3%** |

---

## Running Tests

```bash
# Run full test suite (34 analyzer + 6 import smoke tests)
pytest tests/ -v

# Run only analyzer tests
pytest tests/test_analyzer.py -v
```

---

## Architecture Notes

### Shared Utilities

The `crawl_base.py` and `clean_base.py` modules consolidate code that was previously duplicated across 5 crawlers and 4 cleaners:

- **`crawl_base`**: HTTP fetching with retry/backoff, HTML cleaning (pagenum/reference/ws-noexport removal), filename sanitization, content extraction
- **`clean_base`**: 25-character removal set, metadata-aware line cleaning, configurable trailing number removal and hyphen splitting

### Key Design Decisions

- **`dwipada.paths`** resolves `PROJECT_ROOT` once from `pyproject.toml` location -- all data paths derive from it, eliminating `Path(__file__).parent` hacks
- **Backward-compatible wrappers** at root level (`dwipada_analyzer.py`, `aksharanusarika.py`) re-export from `dwipada.core` for scripts that haven't been updated
- **`clean_poems.py`** uses fundamentally different cleaning logic (footnote preservation, dot-marker handling) and is kept as a standalone cleaner rather than forced into `BaseCleaner`

---

## Research Context

This project supports research in:

1. **Constrained Text Generation**: Symbolic rules guiding neural language models
2. **Low-Resource NLP**: Building tools for Telugu and other Indic languages
3. **Neuro-Symbolic AI**: Combining neural generation with logical constraints
4. **Computational Poetics**: Automated analysis and generation of metrical poetry

---

## Citation

```bibtex
@software{indic_neurosym,
  title = {Indic-NeuroSym: A Neuro-Symbolic Logits Processor for Strict Metrical Generation in Low-Resource SLMs},
  author = {[Your Name]},
  year = {2025},
  url = {[repository-url]}
}
```

---

## Acknowledgments

- **Telugu Wikisource** (te.wikisource.org) for ద్విపద భాగవతము, బసవపురాణము, శ్రీరమాపరిణయము
- **AndhaBharati.com** for రంగనాథ రామాయణము
- Traditional Telugu prosody scholars for Chandassu documentation
