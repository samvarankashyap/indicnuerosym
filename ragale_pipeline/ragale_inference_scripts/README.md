# Ragale Inference Scripts

Three constrained decoding approaches for generating Kannada Utsaha Ragale poems using a base Gemma model + the NFA pipeline from `nfa_pipeline/`.

## Directory Structure

```
ragale_inference_scripts/
├── README.md                          # This file
├── shared_utils.py                    # Shared: model loading, prompts, validation
├── generate_masking_only.py           # Approach 2: Logit masking only
├── generate_masking_backtrack.py      # Approach 3: Masking + backtracking
├── generate_hybrid.py                 # Approach 4: Masking + NFA rejection + backtracking
└── results_*.json                     # Output results (auto-generated)
```

## Dependencies

- `nfa_pipeline/` — FST+NFA pipeline (syllable assembler, guru/laghu classifier, gana NFA, prasa NFA, composite state)
- `kannada_ragale_analyser.py` — standalone analyser for validation
- `transformers`, `torch` — model loading and inference
- Model: `google/gemma-3-1b-it` (base, no fine-tuning)

## Three Approaches

### Approach 2: Masking Only (`generate_masking_only.py`)

**How:** Before each token, simulate all Kannada tokens through CompositeState (FST -> Guru/Laghu -> GanaNFA). Set `logits[invalid] = -inf`. Model samples only from surviving tokens. Force newline when NFA has ACCEPT at 12 syllables.

**Constraint enforcement:** Local only — each token keeps the NFA alive, but no recovery from dead ends.

```bash
python ragale_inference_scripts/generate_masking_only.py --seeds 5 --topic "Bubbles"
```

### Approach 3: Masking + Backtrack (`generate_masking_backtrack.py`)

**How:** Same masking as above, plus: (a) forced newline at ACCEPT, (b) if masking leaves < 3 valid tokens, backtrack to a saved checkpoint with temperature bump + RNG reseed.

**Constraint enforcement:** Local (masking) + global (forced newlines) + recovery (checkpoint backtracking).

```bash
python ragale_inference_scripts/generate_masking_backtrack.py --seeds 5 --topic "Moon"
```

### Approach 4: Hybrid Mask + Accept Preference (`generate_hybrid.py`)

**How:** All three approaches share the same per-step NFA
alive-predicate filter via `BuildGanaMask`. Hybrid additionally runs
a *second* NFA pass over the top-100 surviving candidates that adds
a `has_accept()` check and weighted-samples preferentially from
candidates that would complete a valid line. Backtracking is
retained as a rare fallback.

**Constraint enforcement:** Same NFA alive-predicate filter as the
other two strategies, plus a top-100 *accept-state* re-ranking pass.
The "rejection" terminology in the paper's earlier drafts referred
to this second `has_accept` pass — not to the underlying alive
filter, which all three strategies share.

```bash
python ragale_inference_scripts/generate_hybrid.py --seeds 5 --topic "Rain"
```

## How It Works

### Pipeline Flow

```
User Topic
    |
    v
[Prompt Builder] -> chat template with Ragale rules
    |
    v
[Model Forward Pass] -> logits (256K vocab)
    |
    v
[Static Mask] -> kill non-Kannada tokens (~256K -> ~1500 Kannada)
    |
    v
[Dynamic Gana NFA Mask] -> kill tokens that make NFA unreachable (~1500 -> ~200)
    |
    v
[Forced Newline Check] -> if NFA has ACCEPT at 12 syllables, force \n
    |
    v
[Sampling] -> top-p (masking) or NFA rejection (hybrid) from remaining tokens
    |
    v
[Advance CompositeState] -> feed chosen token chars through FST+NFA
    |
    v
[Repeat until 2 lines complete]
```

### NFA Constraints Enforced

1. **12 syllables per line** — NFA tracks syllable count, masks tokens that overshoot
2. **4 ganas of III or IIU** — GanaNFA prunes branches that don't match valid gana patterns
3. **Guru ending** — Slot 3 only allows IIU pattern (ends on U)
4. **Adi Prasa** — 2nd syllable consonant matching tracked by CompositeState

### Validation

Generated poems are validated by the NFA pipeline (`ragale_pipeline.py`) which runs the full FST+NFA chain and reports per-line gana validity, prasa match, and guru ending.

## CLI Options

All three scripts share the same CLI:

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `google/gemma-3-1b-it` | HuggingFace model ID |
| `--seeds` | `5` | Number of generation seeds per topic |
| `--topic` | `None` | Single topic; if omitted, runs benchmark topics |

## Output Format

Results are saved to `results_<method>.json` with structure:

```json
{
  "topic": "Bubbles",
  "seed": 42,
  "method": "masking_only",
  "generated_text": "...",
  "poem_lines": ["line1", "line2"],
  "valid_lines": [
    {"line": "...", "markers": "I I U ...", "valid": true, "partition": "III+IIU+..."},
    ...
  ],
  "all_valid": true,
  "elapsed": 12.3,
  "tokens_generated": 45,
  "backtracks": 0,
  "mask_computations": 20
}
```
