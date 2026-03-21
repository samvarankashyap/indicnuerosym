# Datasets

Processed datasets in various formats for training, evaluation, and analysis.

## Files

| File | Description |
|---|---|
| `dwipada_augmented_dataset.json` | Final augmented dataset (29,343 entries with metadata, chandassu analysis, meanings). Stored in Git LFS. |
| `dwipada_master_deduplicated.jsonl` | Deduplicated master dataset (~57MB) |
| `ift_alpaca.jsonl` | Alpaca-format training data for instruction fine-tuning (~35MB) |
| `ift_trl_data.jsonl` | TRL messages-format training data (~39MB) |
| `intermediate/` | Intermediate processing stages (perfect dataset, filtered, synthetic, etc.) |

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
