# dwipada.core — Prosody Analysis Engine

The Telugu *dvipada* prosody analysis engine. Implements syllable
splitting, *guru*/*laghu* classification, *gaṇa* partitioning,
*prāsa* (rhyme) and *yati* (caesura alliteration) checking, and
overall metrical scoring on a 0–100% scale. Pure Python, no
external ML dependencies.

## Files

| File | Purpose |
| --- | --- |
| `analyzer.py` | Top-level analyser. Exports `analyze_dwipada(poem)`, `analyze_pada(line)`, `check_prasa(l1, l2)`, `check_yati_maitri(l1, l2)`, and `format_analysis_report(result)`. Handles foot partitioning over the 6 Indra + 2 Surya gana set, scoring, and human-readable report formatting. |
| `aksharanusarika.py` | Telugu syllable splitter and *guru*/*laghu* classifier. Implements the 5-rule classification from *Chandhodarpanamu* (long vowels, diphthongs, anusvara/visarga, virama, and conjunct lookahead with the word-boundary constraint). |
| `constants.py` | `DWIPADA_RULES_BLOCK` (the canonical rule prompt prefixed to every training example) and shared prosody constants (Indra/Surya gana lists, prāsa equivalence groups, yati maitri groups). |

## Public API

```python
from dwipada.core import (
    analyze_dwipada,
    analyze_pada,
    check_prasa,
    check_yati_maitri,
    format_analysis_report,
    split_aksharalu,
    DWIPADA_RULES_BLOCK,
)

result = analyze_dwipada("సౌధాగ్రముల యందు సదనంబు లందు\nవీధుల యందును వెఱవొప్ప నిలిచి")
print(result["is_valid_dwipada"])         # bool
print(result["match_score"]["overall"])    # 0–100
print(result["prasa"]["match"])            # bool
print(result["yati_line1"]["match"])       # bool
print(format_analysis_report(result))      # human-readable
```

## Scoring

| Component | Weight |
| --- | --- |
| *Gaṇa* (foot partition validity) | 40% (10% per foot × 4 feet per line, averaged over 2 lines) |
| *Prāsa* (second-syllable rhyme between lines) | 35% |
| *Yati* (foot-1 ↔ foot-3 alliteration within each line) | 25% |

## Related

- Sister sub-packages: `../data/` (crawl/clean), `../dataset/` (build), `../batch/` (LLM annotation), `../training/` (LoRA fine-tune)
- The same logic is mirrored as a streaming FST + NFA pipeline in
  `../../../nfa_for_dwipada/` for use in constrained decoding
- Tests: `../../../tests/test_analyzer.py`
- Paper Section 3.1 (Telugu *Dvipada* prosodic framework) and
  Appendix A (complete rule tables)
