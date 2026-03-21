# Inference Scripts

Tools for model evaluation, benchmarking, and linguistic analysis of the Dwipada dataset.

## Files

| File | Description |
|---|---|
| `benchmark.py` | Model generation benchmarking — runs multiple prompts across models with metrical scoring |
| `benchmark_report.md` | Detailed benchmark results (3 models x 5 prompts) |
| `benchmark_results.json` | Structured benchmark results in JSON |
| `consonant_cluster_stats.py` | Count poems whose 2nd line starts with a consonant cluster |
| `consonant_cluster_results.txt` | Full output of consonant cluster analysis (771 poems) |
| `nfa_vs_analyzer_crossval.py` | Cross-validation script: NFA pipeline vs analyzer on full dataset |
| `nfa_crossvalidation.txt` | Cross-validation results output |

## Usage

```bash
# Cross-validate NFA pipeline vs analyzer (defaults to datasets/dwipada_augmented_dataset.json)
python inference_scripts/nfa_vs_analyzer_crossval.py datasets/intermediate/dwipada_augmented_perfect_dataset.json

# Run consonant cluster analysis
python inference_scripts/consonant_cluster_stats.py datasets/intermediate/dwipada_augmented_perfect_dataset.json

# Run model benchmarks
python inference_scripts/benchmark.py
```
