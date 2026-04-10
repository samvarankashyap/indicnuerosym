# dwipada.dataset — Dataset Preparation, Augmentation, Combination

Turns the consolidated raw corpus from `../data/` into the
training-ready datasets that live in `../../../datasets/` and
`../../../training_data/`. Pipeline order: `create` → `augment` →
`combine` → `prepare_synthetic`, with `stats` available at any stage.

## Files

| File | Purpose |
| --- | --- |
| `create.py` | Extract the structured base dataset from the raw consolidated JSON. Per-couplet records get a stable ID, source attribution, line text, and (where available) the prose-meaning fields from earlier batch annotation runs. |
| `augment.py` | Add chandassu (prosodic) analysis metadata to every record by running `dwipada.core.analyzer.analyze_dwipada()` over the couplet. Output is the *augmented* dataset (`datasets/dwipada_augmented_dataset.json`) used by the validation pipeline. |
| `combine.py` | Merge real (classical) and synthetic (LLM-generated) datasets into a single unified file. |
| `prepare_synthetic.py` | Prepare a synthetic-data-only variant of the dataset (used for ablation studies on training-data composition). |
| `stats.py` | Compute corpus-level statistics (per-source counts, prosodic purity, length histograms) and optionally write a filtered "metrically perfect" subset to disk. |

## Outputs

| File written | Where |
| --- | --- |
| `dwipada_master_dataset.json` | `../../../datasets/` (the 27,881-couplet master dataset benchmarked in the paper) |
| `dwipada_augmented_dataset.json` | `../../../datasets/` (29,343-record augmented dataset with chandassu analysis) |
| `dwipada_master_filtered_perfect_dataset.json` | `../../../datasets/` (filtered subset of 100% prosodically pure couplets) |
| `dwipada_master_deduplicated.jsonl` | `../../../datasets/` (deduplicated JSONL) |
| `ift_alpaca.jsonl`, `ift_trl_data.jsonl` | `../../../datasets/` (IFT-format training data) |

## Usage

```bash
# Build the structured dataset from consolidated raw text
dwipada create

# Add chandassu analysis to every record
dwipada augment

# Merge real + synthetic
dwipada combine

# Inspect statistics
dwipada stats                              # overall
dwipada stats --by-source                  # per-source breakdown
dwipada stats --write-filtered perfect.json # export 100% pure couplets
```

## Validation

The chandomitra fork of this folder
(`../../../chandomitra/src/dwipada/dataset/`) additionally contains
`validate.py`, `validation_utils.py`, and `word_ttr.py` — the same
multi-level validation pipeline that lives standalone at
`../../../dataset_validation_scripts/`. The standalone copy is the
canonical entry point used by the paper; the chandomitra-bundled
versions are reproductions kept inside the chandomitra benchmark
folder for self-containment.

## Related

- `../data/` — upstream sub-package (consolidated raw corpus)
- `../../../datasets/` — output directory
- `../../../dataset_validation_scripts/` — multi-level validation
  pipeline applied to `dwipada_master_dataset.json` (Levels 1–4
  there + LLM-judge in `../../../llm_as_judge/` as Level 5)
- Paper Section 4 (Dataset Construction) and Section 5 (Data Validation)
