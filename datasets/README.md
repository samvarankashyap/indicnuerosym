# Datasets

Processed datasets in various formats for training, evaluation, and analysis.

## Files

| File | Description |
|---|---|
| `dwipada_master_dataset.json` | **Master dataset benchmarked in the paper — 27,881 couplets** after the chandass scanner inclusion criterion. Each record has the poem, Telugu and English meanings, source attribution, and `chandassu_analysis`. This is the file consumed by `dataset_validation_scripts/`, `train_models/`, `domino/`, and `chandomitra/`. |
| `dwipada_master_filtered_perfect_dataset.json` | Strict-perfect filter applied on top of the master dataset (~61 MB). |
| `dwipada_master_deduplicated.jsonl` | Deduplicated JSONL variant of the master (~57 MB). |
| `dwipada_augmented_dataset.json` | **Predates** the final purity filter — 29,343 records with chandassu analysis. Used by the paper's Level 3 (semantic) and the LaBSE-distribution table where the larger record set was needed. Stored in Git LFS. |
| `ift_alpaca.jsonl` | Alpaca-format `{instruction, input, output}` training data for instruction fine-tuning (~35 MB). |
| `ift_trl_data.jsonl` | TRL messages-format `{messages: [{role, content}]}` training data (~39 MB). |
| `intermediate/` | Intermediate processing stages (perfect dataset, filtered, synthetic, etc.). |

> **Master vs augmented.** The 27,881-couplet `dwipada_master_dataset.json`
> is the **canonical** dataset benchmarked in the paper's Section 5
> validation pipeline, Section 6 fine-tuning, and Section 9 results
> tables. The 29,343-record `dwipada_augmented_dataset.json` is an
> **earlier** snapshot computed before final purity filtering and
> survives only because the paper's Level 3 semantic-similarity
> tables (`tab:msbert`, `tab:l3cube`, `tab:labse_dist`) are
> reported on it.

## Record Structure

Each record in the augmented dataset contains:

```json
{
  "poem": "line 1\nline 2",
  "word_to_word_meaning": { "word": "meaning" },
  "telugu_meaning": "...",
  "english_meaning": "...",
  "chandassu_analysis": {
    "line_1": { "breakdown": "...", "yati_check": "...", "prasa_check": "..." },
    "line_2": { ... }
  }
}
```
