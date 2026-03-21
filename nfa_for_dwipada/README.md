# NFA for Dwipada

NFA-based constrained decoding system for enforcing Dwipada metrical rules during LLM generation. Implements a three-stage FST pipeline that feeds parallel NFAs for gana, prasa, and yati validation.

## Pipeline

```
Raw text
   |
   v
[SyllableAssembler FST]   -- Stage 1: Unicode chars -> syllables
   |
   v
[GanaMarker FST]          -- Stage 2: syllables -> U/I markers
   |
   v
[Gana NFA]                -- Stage 3: U/I stream -> gana partition
```

## Implementations

| File | Description |
|---|---|
| `syllable_assembler.py` | FST Stage 1: Unicode codepoints to Telugu syllables (4 states) |
| `ganana_marker.py` | FST Stage 2: syllables to Guru (U) / Laghu (I) markers (3 states) |
| `guru_laghu_classifier.py` | Alternative Stage 2 returning (syllable, label) tuples (2 states) |
| `gana_nfa.py` | NFA Stage 3: U/I markers to gana partition — 70 states, 3 Indra + 1 Surya (non-deterministic) |

## Analysis Tools

| File | Description |
|---|---|
| `extract_telugu_tokens.py` | Extract Telugu tokens from Gemma tokenizer |
| `analyze_telugu_coverage.py` | Analyze tokenizer coverage of Telugu script |
| `telugu_tokens.tsv` | Extracted Gemma tokenizer Telugu tokens |

## Design Documents

| File | Description |
|---|---|
| `architecture.md` | Full system architecture (3 FSTs + 3 NFAs) |
| `nfa_constrained_decoding_design.md` | Theory: gana definitions, dead state detection, formal language |
| `guru_laghu_rules.md` | The 5 guru/laghu classification rules with examples |
| `design_syllable_assembler.md` | FST design for syllable assembly |
| `design_ganana_marker.md` | FST design for guru/laghu marking |
| `design_codepoint_classifier.md` | Codepoint classification design |

## Usage

```python
from syllable_assembler import SyllableAssembler
from ganana_marker import GanaMarker
from gana_nfa import GanaNFA, format_partition_str

# Full pipeline
asm = SyllableAssembler()
syllables = asm.process("రణములోపల రఘురాముచే నీల్గి")

gm = GanaMarker()
markers = gm.process(syllables)

nfa = GanaNFA()
ui = [m for m in markers if m in ('U', 'I')]
result = nfa.process(ui)
print(format_partition_str(result[0]))
# INDRA - I I I U  INDRA - I I I I  INDRA - U I U  SURYA - U I
```

## Running Tests

Each implementation has inline tests:

```bash
python nfa_for_dwipada/syllable_assembler.py
python nfa_for_dwipada/ganana_marker.py
python nfa_for_dwipada/gana_nfa.py
```
