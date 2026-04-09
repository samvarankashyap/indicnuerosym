# Dataset Validation Scripts

Independent CLI scripts for multi-level quality validation of the Dwipada dataset.
Each level runs as a standalone script, outputs JSON results, and a report generator consolidates everything.

## Prerequisites

- **dwipada package** installed (`pip install -e .` from project root) — required for Level 1 only
- **sentence-transformers** — required for Level 3 (`pip install sentence-transformers`)
- **transformers** — required for Level 4 (`pip install transformers`)
- **numpy** — required for Level 3

## Folder Structure

```
dataset_validation_scripts/
├── validation_utils.py              # Shared utilities (path resolution, model loaders, formatting)
├── level1_chandass_validation.py    # Level 1: Prosodic structure (gana, prasa, yati)
├── level2_sanity_checks.py          # Level 2: Field completeness & length ratios
├── level3_semantic_similarity.py    # Level 3: Semantic similarity (3 models × 3 pairs)
├── level4_lexical_diversity.py      # Level 4: TTR, duplicates, Telugu token coverage
├── word_ttr.py                      # Word-level TTR analysis (literary metric)
├── report.py                        # Consolidate results into a validation report
├── run_all.py                       # Run all levels + report in one command
├── README.md
├── results/                         # JSON outputs from each level
│   ├── level1.json
│   ├── level2.json
│   ├── level3.json
│   ├── level3_LaBSE.json
│   ├── level3_mSBERT-mpnet.json
│   ├── level3_L3Cube-IndicSBERT.json
│   ├── level4.json
│   └── word_ttr_report.md
└── checkpoints/                     # (Legacy) checkpoint files
```

## Validation Levels

| Level | Script | What it checks | Dependencies |
|-------|--------|----------------|--------------|
| 1 | `level1_chandass_validation.py` | Prosodic structure: gana sequence, prasa, yati | `dwipada` package |
| 2 | `level2_sanity_checks.py` | Field completeness, length ratio (telugu/poem) | stdlib only |
| 3 | `level3_semantic_similarity.py` | Semantic similarity (LaBSE, mSBERT-mpnet, L3Cube-IndicSBERT) | `sentence-transformers`, `numpy` |
| 4 | `level4_lexical_diversity.py` | TTR, exact/near duplicates, Telugu token coverage | `transformers` |
| — | `word_ttr.py` | Word-level TTR analysis per source | stdlib only |
| — | `report.py` | Consolidate all levels into a report | stdlib only |

## Usage

### Run everything at once

```bash
cd dataset_validation_scripts

# Full pipeline
python run_all.py

# With options
python run_all.py --dataset ../datasets/your_data.json --limit 100
python run_all.py --skip-level3  # skip the slow semantic similarity step
```

### Run individual levels

```bash
cd dataset_validation_scripts

# Level 1: Prosodic validation
python level1_chandass_validation.py
python level1_chandass_validation.py --dataset ../datasets/your_data.json --limit 100

# Level 2: Field completeness
python level2_sanity_checks.py -d ../datasets/your_data.json

# Level 3: Semantic similarity (slow — loads 3 models)
python level3_semantic_similarity.py
python level3_semantic_similarity.py --models LaBSE mSBERT-mpnet --batch-size 128

# Level 4: Lexical diversity
python level4_lexical_diversity.py

# Word TTR (uses JSONL dataset)
python word_ttr.py --dataset ../datasets/dwipada_master_deduplicated.jsonl --top-n 30
```

### Generate report from existing results

```bash
python report.py
python report.py --results-dir results/ --output validation_report.txt
```

## Common Options

All level scripts support these options:

| Option | Description |
|--------|-------------|
| `--dataset, -d` | Path to dataset JSON (default: `datasets/dwipada_augmented_dataset.json`) |
| `--output-dir, -o` | Directory for output JSON (default: `results/`) |
| `--limit, -n` | Process only first N records (for quick testing) |

Level 3 additionally supports:

| Option | Description |
|--------|-------------|
| `--batch-size, -b` | Batch size for model encoding (default: 256) |
| `--models, -m` | Select which models to run: `LaBSE`, `mSBERT-mpnet`, `L3Cube-IndicSBERT` |

## Output Format

Each level script writes a JSON file to `results/` with a `_meta` block:

```json
{
  "_meta": {
    "level": "level1",
    "dataset": "/absolute/path/to/dataset.json",
    "total_records": 29343,
    "limit": null,
    "timestamp": "2026-04-08T14:30:00"
  },
  "total": 29343,
  "valid_count": 29343,
  ...
}
```

The `report.py` script reads these JSON files and generates a human-readable validation report.
