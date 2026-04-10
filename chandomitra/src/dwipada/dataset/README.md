# dwipada.dataset (chandomitra fork) — Dataset + Validation

Dataset preparation, augmentation, and validation, snapshotted into
the chandomitra benchmark folder. Adds three validation files
(`validate.py`, `validation_utils.py`, `word_ttr.py`) that the
canonical `../../../../src/dwipada/dataset/` does not have, so the
chandomitra benchmark can run the full multi-level validation
pipeline self-contained.

## Files

| File | Purpose |
| --- | --- |
| `create.py` | Extract the structured base dataset from raw consolidated JSON. |
| `augment.py` | Add chandassu (prosodic) analysis metadata via `dwipada.core.analyzer`. Output: `dwipada_augmented_dataset.json`. |
| `combine.py` | Merge real (classical) and synthetic (LLM-generated) datasets. |
| `prepare_synthetic.py` | Prepare a synthetic-only variant for ablation studies. |
| `stats.py` | Corpus statistics + filtered "metrically perfect" subset writer. |
| `validate.py` | **Fork-only.** Multi-level validation entry point — runs Levels 1–4 (chandass, sanity, semantic similarity, lexical diversity) over an input dataset. Mirrors `../../../../dataset_validation_scripts/run_all.py`. |
| `validation_utils.py` | **Fork-only.** Shared validation utilities — model loaders for LaBSE / mSBERT-mpnet / L3Cube-IndicSBERT, length-ratio computation, Telugu token counter, formatting helpers. |
| `word_ttr.py` | **Fork-only.** Word-level type-token ratio analysis (literary metric) — corpus + per-poem TTR, MATTR, Yule's K, Honoré's H, Sichel's S, hapax/dis-legomena ratios, near-duplicate detection. |

## Usage

```bash
# Build / augment
dwipada create
dwipada augment
dwipada combine

# Validate (fork-only entry point)
python -m dwipada.dataset.validate
python -m dwipada.dataset.validate --skip-level3   # skip slow semantic checking
python -m dwipada.dataset.validate --limit 100
python -m dwipada.dataset.validate --dataset path/to/data.json

# Word-level TTR analysis
python -m dwipada.dataset.word_ttr --dataset ../../../../datasets/dwipada_master_deduplicated.jsonl --top-n 30
```

## Validation levels

| Level | Script | Checks |
| --- | --- | --- |
| 1 | `validate.py` (chandass) | Prosodic structure (gana, prasa, yati) |
| 2 | `validate.py` (sanity) | Field completeness, length-ratio bounds |
| 3 | `validate.py` (semantic) | Semantic similarity via LaBSE, mSBERT-mpnet, L3Cube-IndicSBERT |
| 4 | `validate.py` (lexical) | TTR, MATTR, Yule's K, Honoré's H, near-duplicate detection |
| 5 | (separate) | LLM-as-Judge — see `../../../../llm_as_judge/` |

## Related

- Canonical version (no `validate.py`/`validation_utils.py`/`word_ttr.py`):
  `../../../../src/dwipada/dataset/`
- Standalone validation pipeline: `../../../../dataset_validation_scripts/`
- LLM-as-Judge: `../../../../llm_as_judge/` (Level 5)
- Paper Section 5 (Data Validation, Levels 1–5)
