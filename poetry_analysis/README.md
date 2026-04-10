# poetry_analysis — POS-tagging + Ablation Pipeline

A standalone numbered-script pipeline that performs POS tagging on the
Telugu corpus, audits the data, builds 4-gana / 5-gana ablation
variants, fine-tunes Gemma 4B with Unsloth, and runs analysis-mode
inference. The scripts are intentionally numbered so they run in
order.

This pipeline is **separate from** the main `train_models/` and
`domino/` flows; it explores augmentation and analysis-task variants
that are referenced in the paper's MetricalArgs taxonomy framing
(Kranti and Vajjala 2024) but are not part of the headline benchmark.

## Pipeline (in order)

| # | Script | Stage |
| --- | --- | --- |
| 1 | `1.pos_tagger.py` | Add part-of-speech tags (English + Telugu) to every couplet in the consolidated corpus |
| 2 | `2.data_audit.py` | Audit the tagged corpus (counts, balance, gana distribution, missing fields) |
| 3a | `3.1an4_an5_builder.py` | Build the 4-gana and 5-gana ablation datasets (drop one Indra slot) |
| 3b | `3.metricalargs_builder.py` | Build instruction-formatted variants for the MetricalArgs Generation/Analysis taxonomy |
| 4a | `4.FinetuneGemmaUnslothModal.py` | Fine-tune Gemma 4B on Modal with Unsloth (cloud) |
| 4b | `4finetune_gemma4_analysis.py` | Local fine-tune for the Analysis task variant |
| 4c | `4finetune_gemma4_retrieval.py` | Local fine-tune for the Retrieval task variant |
| 5 | `5infer_analysis.py` | Inference + scoring on the held-out analysis split |

## Inputs / outputs

Inputs are read from `../datasets/` (the master and augmented JSON
files); outputs (POS-tagged corpora, ablation splits, analysis
predictions) are written into the same folder. There is no shared
`requirements.txt` — each script declares its dependencies in the
imports.

## Related folders

- `../datasets/` — source corpus the scripts read
- `../train_models/` — the main `dwipada` LoRA fine-tuning flow that
  produces the merged model used by the rest of the project
- `../inference_scripts/` — separate single-prompt inference suite

## Paper

These scripts are the experimental scaffolding for the
Generation-vs-Analysis taxonomy of Kranti and Vajjala 2024 referenced
in the paper's Section 1 contributions list. The headline benchmark
results in the paper come from `domino/` and
`ragale_pipeline/ragale_inference_scripts/`, not from this folder.
