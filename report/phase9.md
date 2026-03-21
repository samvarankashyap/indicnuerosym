# Phase 9: Kannada↔Telugu Script Converter for Dwipada Dataset

## Context

Phases 1–6 established a complete pipeline for Telugu dwipada analysis, dataset preparation, and constrained decoding — all grounded in Telugu Unicode codepoints. Telugu and Kannada share a common Dravidian heritage and nearly identical phonological inventories. Their scripts, while visually distinct, encode the same set of vowels, consonants, vowel signs, and diacritics in a structurally parallel Unicode layout.

This phase implements a **bidirectional character-level Kannada↔Telugu converter** based on a published comparative study, and uses it to transliterate the Telugu dwipada dataset into Kannada script. The converter also computes per-entry quality scores based on the paper's resemblance classifications.

---

## 1. Motivation

If a bijective character mapping between the two scripts preserves all phonological distinctions, the converted Kannada text could potentially be fed back through the existing Telugu dwipada analyzer (`analyze_dwipada`) to test whether the metrical rules transfer across scripts. **This phase builds the converter; metrical validation is future work.**

---

## 2. Script Mapping: Theoretical Basis

The mapping is grounded in Nidamarthy's 2021 comparative study:

> Nidamarthy, S. (2021). *Comparative study of Kannada and Telugu consonants, consonant sub forms, vowels, signs.* Shikshan Sanshodhan, 4(9).

Key findings from the paper:

| Category | Count | Resemblance |
|----------|------:|-------------|
| High-resemblance consonants | 17 | 99% (గ, ఠ, డ, ద, న, ర, స, ళ, etc.) |
| Medium-resemblance consonants | 5 | 70–99% (ఖ, ఙ, ప, ఝ, య) |
| Low/no-resemblance consonants | 5 | Visually different but phonetically identical (చ, శ, త, ష, హ) |
| Vowels | 16 | 99% (అ–ఔ, ఌ, ౡ) |
| Vowel signs (matras) | 13 | 95–99% |
| Special symbols | 4 | anusvara, visarga, halant, chandrabindu |

The critical insight: **visual resemblance is irrelevant for metrical analysis**. What matters is that the mapping is **phonologically bijective** — every Kannada phoneme maps to exactly one Telugu phoneme and vice versa. Since both scripts encode the same phonological inventory via Unicode, a codepoint-to-codepoint mapping preserves all information needed for prosodic analysis.

---

## 3. Implementation

The converter is implemented in `kannada2telugu/kannada_telugu_converter_paper_based.py` as `EnhancedKannadaTeluguConverter`.

### 3.1 Architecture

```
Kannada Unicode Text
        │
        ▼
┌─ Character-level mapping ────────────────────────────┐
│   For each character:                                 │
│     1. Try 3-char sequence match (conjuncts like క్ష) │
│     2. Try 2-char sequence match                      │
│     3. Try single-char match                          │
│     4. Pass through unmapped chars (punctuation, etc.) │
└───────────────────────────────────────────────────────┘
        │
        ▼
Telugu Unicode Text
```

### 3.2 Mapping Categories

The mapping covers 6 character classes:

| Class | Kannada Range | Telugu Range | Count |
|-------|--------------|-------------|------:|
| Consonants (high resemblance) | ಗ ಠ ಡ ದ ನ ರ ಸ ಳ ... | గ ఠ డ ద న ర స ళ ... | 17 |
| Consonants (medium resemblance) | ಖ ಙ ಪ ಝ ಯ | ఖ ఙ ప ఝ య | 5 |
| Consonants (low resemblance) | ಚ ಶ ತ ಷ ಹ | చ శ త ష హ | 5 |
| Remaining consonants | ಕ ಛ ಟ ಫ ಮ ವ ಞ ಘ | క ఛ ట ఫ మ వ ఞ ఘ | 8 |
| Vowels + signs + special | ಅ–ಔ, ಾ–ೌ, ಂ ಃ ್ ಁ | అ–ఔ, ా–ౌ, ం ః ్ ఁ | 33 |
| Numbers | ೦–೯ | ౦–౯ | 10 |
| **Total** | | | **78+** |

### 3.3 Multi-Character Handling

The converter processes text with a longest-match-first strategy to handle conjunct consonants:

```python
# Priority: 3-char > 2-char > 1-char
while i < text_len:
    if text[i:i+3] in mapping:   # conjuncts (e.g., ಕ್ಷ → క్ష)
        ...
    elif text[i:i+2] in mapping: # consonant + sign pairs
        ...
    elif text[i] in mapping:     # single character
        ...
    else:
        pass_through(text[i])    # punctuation, spaces, etc.
```

### 3.4 Bidirectional Conversion

The mapping is constructed as Telugu→Kannada, then automatically inverted:

```python
self.telugu_to_kannada_map = self._create_enhanced_mapping()
self.kannada_to_telugu_map = {v: k for k, v in self.telugu_to_kannada_map.items()}
```

This supports both directions:
- `kannada_to_telugu(text)` — for feeding Kannada poems into the Telugu analyzer
- `telugu_to_kannada(text)` — for converting the Telugu dwipada dataset to Kannada

---

## 4. Dataset Conversion

The converter includes a `convert_dataset()` method that processes the entire dwipada corpus:

```
dwipada_master_filtered_perfect_dataset.json (Telugu)
        │
        ▼
┌─ Per-entry conversion ───────────────────────────────┐
│   • poem → kannada_poem                               │
│   • telugu_meaning → kannada_meaning                  │
│   • word_to_word_meaning → word_to_word_meaning_kannada│
│   • english_meaning preserved as-is                   │
│   • conversion_quality_score computed per entry        │
└───────────────────────────────────────────────────────┘
        │
        ▼
dwipada_kannada_paper_based_dataset.json
```

### 4.1 Quality Scoring

Each converted entry receives a quality score based on the resemblance categories from the paper:

```
quality_score = (high × 0.99 + medium × 0.85 + low × 0.70) / total_classified
```

This provides a per-poem confidence measure for the transliteration fidelity, weighted by the paper's resemblance findings.

### 4.2 Output Schema

```json
{
  "original_telugu_poem": "మా రాముబాణనిర్మథితమాంసముల \nకీ రాదె నీ నాక మేల యిచ్చెదవు",
  "kannada_poem": "ಮಾ ರಾಮುಬಾಣನಿರ್ಮಥಿತಮಾಂಸಮುಲ \nಕೀ ರಾದೆ ನೀ ನಾಕ ಮೇಲ ಯಿಚ್ಚೆದವು",
  "conversion_quality_score": 0.95,
  "high_resemblance_chars": 12,
  "medium_resemblance_chars": 2,
  "low_resemblance_chars": 1,
  "original_telugu_meaning": "...",
  "kannada_meaning": "...",
  "english_meaning": "...",
  "word_to_word_meaning_kannada": { "ಮಾ": "our", ... },
  "source_index": 0
}
```

---

## 5. Next Steps

The converter enables several follow-up experiments not yet implemented:

- **Metrical validation**: Feed converted Kannada→Telugu text through `analyze_dwipada()` to test whether gana, prasa, and yati rules produce correct results on transliterated text.
- **Constrained decoding**: If metrical validation succeeds, the Phase 6 FST+NFA pipeline could be extended to Kannada via a transliteration pre/post-processing layer.
- **Kannada IFT dataset**: The converted dataset could be used to train a Kannada dwipada generation model using the same IFT pipeline (Phase 4).

---

## 7. Limitations

| Limitation | Impact | Mitigation |
|-----------|--------|------------|
| **Mapping is purely character-level** | Cannot handle script-specific ligature conventions or rendering differences | Acceptable — metrical analysis operates on phonological structure, not visual rendering |
| **No Kannada-specific prosody rules** | Assumes Kannada dwipada follows identical metrical rules to Telugu | Valid for shared Dravidian metres; would need verification for Kannada-specific forms |
| **Meanings translated character-by-character** | Telugu prose meanings converted to Kannada script may be unnatural | Quality score flags these; meanings serve as conditioning prompts, not linguistic targets |
| **Hardcoded paths in script** | `__main__` block uses absolute paths | Trivially fixable; converter class itself is path-independent |

---

## 8. Files

| File | Description |
|------|-------------|
| `kannada2telugu/kannada_telugu_converter_paper_based.py` | Bidirectional Kannada↔Telugu converter with quality analysis |

---

## 9. References

- Nidamarthy, S. (2021). *Comparative study of Kannada and Telugu consonants, consonant sub forms, vowels, signs.* Shikshan Sanshodhan, 4(9).

---

*Kannada↔Telugu converter documented 2026-03-08.*
