# nfa_pipeline — Kannada Ragale FST + NFA Constraint Engine

The Kannada *utsaha ragale* counterpart of `../../nfa_for_dwipada/`.
Same architecture (chained Mealy FSTs feeding parallel NFAs), simpler
constraints: only two foot patterns (III, IIU), exactly 12 syllables
per line, no *yati*.

## Files

| File | Type | States | Purpose |
| --- | --- | --- | --- |
| `syllable_assembler.py` | Mealy FST | 4 | Stage 1: Kannada Unicode (U+0C80–U+0CFF) → complete *akṣaras* |
| `guru_laghu_classifier.py` | Mealy FST | 2 | Stage 2: *akṣara* stream → U/I (*guru*/*laghu*) markers, with the same 5 phonological rules used for Telugu |
| `gana_nfa.py` | NFA | 22 | Stage 3: U/I marker stream → foot partition over the regular language `(III \| IIU)³ · IIU` |
| `prasa_nfa.py` | NFA | 7 | Ādi prāsa: second-syllable consonant of line 1 must match line 2, with two Kannada equivalence groups (laterals, sibilants) |
| `composite_state.py` | Composite | — | Maintains all of the above incrementally; exposes `is_alive()`, `has_accept()`, `feed_token_text()`, `snapshot()`, `from_snapshot()`, and `build_gana_mask()` |
| `ragale_pipeline.py` | Driver | — | Top-level entry point that wires the FSTs and NFAs together for offline validation of a poem (used by `kannada_ragale_analyser.py` and the inference scripts) |

## Differences from the Telugu Dvipada NFA pipeline

| Aspect | Dvipada (`../../nfa_for_dwipada/`) | Ragale (this folder) |
| --- | --- | --- |
| Gana patterns | 8 (6 Indra + 2 Surya) | 2 (III, IIU) |
| Gana NFA size | ~69 states | 22 states |
| Syllables / line | 11–15 (variable) | 12 (fixed) |
| Prasa equivalence groups | 3 (laterals, sibilants, rhotics) | 2 (laterals, sibilants — no rhotic group in Kannada) |
| Yati | 4-state NFA (optional) | not used |
| Guru ending | implicit | structurally enforced (slot 4 must be IIU) |
| Total state budget | ~80 | 35 |
| Equivalent product DFA | 1,932 states | 154 states |

## Related

- Parent: `../README.md` (the full Ragale pipeline)
- Telugu counterpart: `../../nfa_for_dwipada/README.md`
- Consumed by: `../ragale_inference_scripts/benchmark_*.py`,
  `../kannada_ragale_analyser.py`

## Paper

This folder is described in Section 7.8 of the paper (the Ragale
adaptation of the Symbolic Enforcer) with full construction details
in Appendix G (`app:ragale_construction`). The state budget table
appears as `tab:ragale_states`.
