# domino — Dvipada Constrained Decoding Benchmarks

End-to-end benchmark scripts and result JSONs for the three FST+NFA
constrained-decoding strategies on Telugu *dvipada*. This is the
folder behind the headline numbers in the paper's Section 9 results
tables.

## Benchmark scripts

| Script | Strategy | Description |
| --- | --- | --- |
| `benchmark_masking_only.py` | Masking-only | Per-step NFA alive-predicate filter via `BuildGanaMask` over the ~1,822 Telugu tokens, then top-*p* sampling. No recovery. |
| `benchmark_masking_backtrack.py` | Masking + Backtracking | Same alive-predicate filter, plus checkpoint rollback and temperature escalation when the alive set drops below 3 candidates or the line overshoots 15 syllables. |
| `benchmark_hybrid.py` | Hybrid Mask + Accept Preference | Same alive-predicate filter, plus a second NFA pass over the top-100 masked candidates that adds a `has_accept()` check and prefers tokens that complete a valid line. |

All three scripts share the same prompts (3 mythological/devotional
themes) × 34 seeds = 102 generations per (model, method) cell, and
both models — `gemma3-1b-base` and `gemma3-1b-merged` — are tested by
each script.

## Result JSONs

Each `(strategy, model)` cell produces one result file:

```
benchmark_masking_only_gemma3-1b-base.json
benchmark_masking_only_gemma3-1b-merged.json
benchmark_masking_backtrack_gemma3-1b-base.json
benchmark_masking_backtrack_gemma3-1b-merged.json
benchmark_hybrid_mask_reject_gemma3-1b-base.json
benchmark_hybrid_mask_reject_gemma3-1b-merged.json
```

Each JSON has top-level fields `total_poems`, `total_valid`,
`poem_accuracy`, `total_lines`, `valid_lines`, `line_accuracy`,
`total_time`, `per_topic`, and a `poems[]` array with per-generation
records (`topic`, `seed`, `generated_text`, `poem_lines`,
`valid_lines[*].markers`, `all_valid`, `elapsed`, `tokens_generated`,
`backtracks`, `mask_computations`).

Older exploratory result files (`benchmark_constrained_*.json`,
`benchmark_unconstrained_*.json`, `benchmark_base_*.json`) are kept
for reference but are superseded by the per-cell files above.

## Constrained-generation prototype scripts

| Script | Purpose |
| --- | --- |
| `constrained_generate.py` | Earliest single-prompt prototype |
| `constrained_generate_v2.py` | Second-iteration prototype |
| `constrained_generate_base.py` | Base-model variant |
| `constrained_generate_masked.py` | Masking-only standalone |
| `constrained_generate_gemma4.py` | Gemma-4B variant |

These predate the three benchmark scripts above and remain only for
provenance; the benchmark scripts are the canonical entry points.

## Related folders

- `../nfa_for_dwipada/` — the FST + NFA pipeline that
  `BuildGanaMask`, `CompositeState`, and `IsReachable` come from
- `../ragale_pipeline/ragale_inference_scripts/` — parallel set of
  benchmark scripts for Kannada *utsaha ragale*
- `../chandomitra/` — adapted Chandomitra port used as the external
  baseline in the same Section 9 results table

## Paper

These benchmarks back Tables `tab:dvipada-results`,
`tab:dvipada-pertopic`, and `tab:efficiency-stats` of the paper. Each
cell is exactly 102 generations (3 prompts × 34 seeds; seed schedule
`42 + 7k` for *k* ∈ [0, 33]).

## Usage

```bash
# Run a single benchmark script (writes its own JSON)
python benchmark_hybrid.py --model gemma3-1b-merged

# All three for both models
for s in masking_only masking_backtrack hybrid; do
  for m in gemma3-1b-base gemma3-1b-merged; do
    python benchmark_${s}.py --model $m
  done
done
```
