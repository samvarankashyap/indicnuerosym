# dwipada.training (chandomitra fork) — Training, Generation, Constrained Decoding

LoRA fine-tuning, generation, and the constrained-decoding port of
Chandomitra's Algorithm 1 — bundled inside the chandomitra benchmark
folder so the benchmark scripts in `../../../` can run end-to-end
without external paths. **This entire sub-package exists only in the
chandomitra fork**; the canonical `../../../../src/dwipada/` does not
have a `training/` sub-package.

## Top-level files

| File | Purpose |
| --- | --- |
| `prepare_data.py` | Filter the augmented dataset to 100% metrically pure couplets and emit `training_data/{train,val,test}.jsonl` (80/10/10 split). |
| `train.py` | Default LoRA fine-tuning script (Gemma 3 1B; expects `training_data/{train,val}.jsonl` in `{input, output}` format). |
| `train_ift.py` | IFT variant for alpaca-format `{instruction, input, output}` data. |
| `train_trl.py` | TRL variant for messages-format `{messages: [{role, content}]}` data. |
| `generate.py` | Unconstrained generation entry point (sanity-check a trained checkpoint). |
| `generate_constrained.py` | Constrained generation entry point that wires the `constrained/` components into a HuggingFace `LogitsProcessor`. |
| `tokenizer.py` | Tokenizer utilities (Telugu token extraction, vocab analysis). |

## `constrained/` — Chandomitra's Algorithm 1 ported to Dvipada

The `constrained/` subdirectory implements our port of Chandomitra's
top-*k* constrained-decoding algorithm to the Telugu Dvipada rule set.
The published Chandomitra system targets Sanskrit Anuṣṭubh; our port
preserves the outer top-*k* masking loop verbatim, replaces the
fixed-length per-*pāda* regex with a prefix trie of all 432 valid
Indra³ · Surya foot sequences, and adds the *prāsa* and *yati* checks
that Anuṣṭubh does not require.

| File | Purpose |
| --- | --- |
| `constrained/logits_processor.py` | `DwipadaConstrainedLogitsProcessor` — the HuggingFace `LogitsProcessor` that masks invalid tokens at every generation step using the top-*k* + 4-level relaxation cascade. |
| `constrained/pattern_trie.py` | Prefix trie of all 432 valid `Indra³ · Surya` foot sequences, built once at startup. |
| `constrained/generation_state.py` | Position-tracking state passed between steps (which line, which slot, current syllable count, line-1 prāsa consonant). |
| `constrained/syllable_utils.py` | Syllable-weight computation with one-step lookahead for the conjunct-induced *guru* promotion rule. |

This is the implementation that the paper benchmarks as the
"adapted Chandomitra" external baseline. Its results live in
`../../../benchmark_chandomitra_n102_gemma3-1b-base.json` and
`../../../benchmark_chandomitra_n102_gemma3-1b-merged.json`.

## Why this is fork-only

The canonical `../../../../src/dwipada/` is intentionally
constrained-decoder-free: it provides only the offline analyser, the
data pipeline, and the dataset preparation. The constrained decoder
in this fork (and the `nfa_for_dwipada/` + `domino/` flow that the
paper actually uses) live outside the canonical package because they
import the analyser as a library rather than the other way around.

## Usage

```bash
# Default training
python -m dwipada.training.train
python -m dwipada.training.train --max_steps 10        # smoke test
python -m dwipada.training.train --merge                # merge adapter after

# IFT (alpaca format)
python -m dwipada.training.train_ift

# TRL (messages format)
python -m dwipada.training.train_trl

# Constrained generation
dwipada generate-constrained "ద్విపదలో ఒక పద్యం వ్రాయండి." \
    --merged-model ../../../dwipada_merged_model
```

## Related

- `../../../README.md` — full chandomitra benchmark documentation
- `../../../benchmark_chandomitra_n102_*.json` — n=102 benchmark
  result JSONs produced by this code
- `../../../../nfa_for_dwipada/` — the FST + NFA pipeline used by
  the FST+NFA enforcer (paper Section 7), which is mechanically
  different from the top-*k* approach in this folder
- `../../../../domino/` — the FST+NFA enforcer's benchmark suite
- `../../../chandomitra.pdf` — the original Chandomitra paper
  (Jagadeeshan et al. 2026, arXiv:2506.00815)
- Paper Section 2 (Related Work, Computational prosody for Indic
  languages) and Section 9 (Comparison to Chandomitra)
