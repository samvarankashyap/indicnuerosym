# dwipada

Main Python package for Indic-NeuroSym. Provides prosody analysis, data pipelines, and training tools for Telugu Dwipada poetry.

## Modules

| Module | Description |
|---|---|
| `cli.py` | Unified CLI dispatcher (`dwipada` command) |
| `paths.py` | Centralized path resolution for all project directories |
| `__main__.py` | Enables `python -m dwipada` invocation |

## Subpackages

| Subpackage | Description |
|---|---|
| `core/` | Prosody analysis engine — syllable splitting, guru/laghu classification, gana/yati/prasa validation |
| `data/` | Web crawlers (5 Telugu sources) and text cleaners |
| `dataset/` | Dataset preparation, augmentation, deduplication, validation, and statistics |
| `batch/` | Gemini and Vertex AI batch processing integration |
| `training/` | Model training (LoRA, IFT, TRL) and inference scripts |

## Key Files in `core/`

| File | Description |
|---|---|
| `analyzer.py` | Dwipada meter analyzer — gana partitioning, yati/prasa checks, scoring |
| `aksharanusarika.py` | Telugu syllable splitting and guru/laghu analysis (v0.0.7a) |
| `constants.py` | `DWIPADA_RULES_BLOCK` and shared prosody constants |

## Usage

```bash
# CLI
dwipada analyze "poem text"
dwipada crawl ranganatha_ramayanam
dwipada prepare --split 80 10 10

# Python
from dwipada.core import analyze_dwipada, format_analysis_report
```
