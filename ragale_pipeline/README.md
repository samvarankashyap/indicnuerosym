# ragale_pipeline — Kannada *Utsaha Ragale* End-to-End

The Kannada-side counterpart to the Telugu *dvipada* pipeline that
lives in `../src/dwipada/`. This folder contains everything needed to
analyse, validate, fine-tune, and benchmark Kannada *utsaha ragale*
poetry generation: the analyser, the FST + NFA constraint engine, the
IFT (instruction fine-tuning) data preparation, the LoRA fine-tuning
script, and the three constrained-decoding benchmark scripts.

## Top-level files

| File | Purpose |
| --- | --- |
| `kannada_ragale_analyser.py` | Stand-alone Kannada *utsaha ragale* analyser (analogue of `dwipada/core/analyzer.py`); checks gana, ādi prāsa, and 12-syllable line constraints |
| `consolidate_ragale.py` | Consolidate raw synthetic ragale poems into a single JSON corpus |
| `generate_prompts.py` | Generate the topic prompts used for benchmark generation |
| `prompts_list.txt` | The 100 topic prompts used to seed synthetic generation |
| `topics_list.txt` | Short topic list for the inference benchmarks (Bubbles, Moon, Rain) |
| `requirements.txt` | Python dependencies for the analyser, NFA pipeline, IFT, and benchmarks |

## Data files

| File | Description |
| --- | --- |
| `ragale_consolidated.json` | Raw consolidated ragale poems |
| `ragale_consolidated_filtered.json` | Filtered to only metrically-pure couplets (the 1,010-poem corpus referenced in the paper) |
| `ragale_consolidated_filtered_nfa_results.json` | Per-poem NFA validation output |
| `ragale_outputs.txt` | Plain-text generation log |

## Design documents

| File | Topic |
| --- | --- |
| `ragale_prosodic_framework.md` | High-level prosodic framework (gana, prāsa, syllable rules) |
| `ragale_formal_theory.txt` | Formal-language treatment of *utsaha ragale* constraints |
| `ragale_fst_design.txt` | FST pipeline design for the syllable assembler and guru/laghu classifier |
| `ragale_nfa_design.txt` | NFA design for the gana and prāsa NFAs (parallel to `nfa_for_dwipada/`) |

## Subdirectories

| Subdirectory | What it contains |
| --- | --- |
| `nfa_pipeline/` | The FST + NFA constraint engine for Kannada (`syllable_assembler.py`, `guru_laghu_classifier.py`, `gana_nfa.py`, `prasa_nfa.py`, `composite_state.py`, `ragale_pipeline.py`) — parallels `../nfa_for_dwipada/` |
| `ragale_ift_pipeline/` | Instruction fine-tuning data preparation and Gemma 3 1B IT LoRA training script |
| `ragale_inference_scripts/` | Three constrained-decoding benchmark scripts (`benchmark_masking_only.py`, `benchmark_masking_backtrack.py`, `benchmark_hybrid.py`) and their result JSONs |
| `ragale_checkpoints/` | LoRA training checkpoints (auto-saved by the trainer; gitignored) |
| `ragale_lora_adapter/` | Final LoRA adapter weights produced by the IFT pipeline |
| `ragale_logs/` | TensorBoard logs from the IFT training runs |

## Pipeline order

```
1. consolidate_ragale.py            → ragale_consolidated.json
2. kannada_ragale_analyser.py        → ragale_consolidated_filtered.json
                                       (1,010 metrically-pure couplets)
3. ragale_ift_pipeline/ragale_ift.py prepare → ift_data/training_ready.jsonl
4. ragale_ift_pipeline/ragale_ift.py convert → ift_alpaca.jsonl, ift_trl.jsonl
5. ragale_ift_pipeline/ragale_gemma3_finetune.py → LoRA adapter
6. ragale_inference_scripts/benchmark_*.py → result JSONs
```

## Related folders

- `../nfa_for_dwipada/` — Telugu *dvipada* counterpart of `nfa_pipeline/`
- `../domino/` — Telugu *dvipada* counterpart of `ragale_inference_scripts/`
- `../train_models/` — Telugu *dvipada* counterpart of `ragale_ift_pipeline/`

## Paper

This folder backs the Kannada half of the paper:
prosodic framework (Section 3.2), constrained-decoding adaptation
(Section 7.8 + Appendix G), and the Ragale results table
(`tab:ragale-results`) plus per-topic and efficiency tables.
