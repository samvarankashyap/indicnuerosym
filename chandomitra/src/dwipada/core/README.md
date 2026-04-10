# dwipada.core (chandomitra fork) — Prosody Analysis Engine

The Telugu *dvipada* prosody analysis engine, snapshotted into the
chandomitra benchmark folder. Same purpose and same public API as the
canonical version at `../../../../src/dwipada/core/`; the chandomitra
copy is kept here so the chandomitra benchmark scripts in
`../../../` can run self-contained.

## Files

| File | Purpose |
| --- | --- |
| `analyzer.py` | Top-level analyser. Exports `analyze_dwipada(poem)`, `analyze_pada(line)`, `check_prasa(l1, l2)`, `check_yati_maitri(l1, l2)`, and `format_analysis_report(result)`. |
| `aksharanusarika.py` | Telugu syllable splitter and *guru*/*laghu* classifier (5-rule classification from *Chandhodarpanamu*). |
| `constants.py` | `DWIPADA_RULES_BLOCK` and shared prosody constants (Indra/Surya gana lists, prāsa equivalence groups, yati maitri groups). |

This fork's `analyzer.py` differs slightly from the canonical version
(see `git diff` against `../../../../src/dwipada/core/analyzer.py`).
The differences are minor (additional helpers used by the
chandomitra benchmark) and the public API is identical.

## Public API

```python
from dwipada.core import analyze_dwipada, format_analysis_report
result = analyze_dwipada("సౌధాగ్రముల యందు సదనంబు లందు\nవీధుల యందును వెఱవొప్ప నిలిచి")
print(format_analysis_report(result))
```

## Scoring

| Component | Weight |
| --- | --- |
| *Gaṇa* (foot partition validity) | 40% |
| *Prāsa* (second-syllable rhyme between lines) | 35% |
| *Yati* (foot-1 ↔ foot-3 alliteration within each line) | 25% |

## Related

- Canonical version: `../../../../src/dwipada/core/`
- Fork parent: `../README.md`
- Sister sub-packages in this fork: `../data/`, `../dataset/`,
  `../batch/`, `../training/` (the last two have extra files vs
  the canonical versions)
- Streaming FST + NFA mirror: `../../../../nfa_for_dwipada/`
- Paper Section 3.1 (Telugu *Dvipada* prosodic framework) and
  Appendix A
