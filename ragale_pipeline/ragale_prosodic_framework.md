# The Utsaha Ragale Prosodic Framework

Utsaha Ragale metre is governed by a hierarchy of rules operating at the syllable, foot (gana), and line levels. Complete rule tables appear below.

## 1. Structure

A Utsaha Ragale stanza contains exactly **two lines** (padalu). Each line comprises **four metrical feet** (ganalu), each of exactly 3 syllables, yielding **12 syllables per line**.

```
Line 1:  [Gana 1] [Gana 2] [Gana 3] [Gana 4]
Line 2:  [Gana 1] [Gana 2] [Gana 3] [Gana 4]
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
          3 syl    3 syl    3 syl    3 syl  = 12 syllables
```

## 2. Syllable Weight

Each syllable (aksharam) is classified as **heavy** (guru, **U**) or **light** (laghu, **I**) according to six classification rules. Rules 1-4 are intrinsic (decided from the syllable itself). Rule 5 requires a 1-syllable lookahead.

| Rule | Name | Condition | Result |
|------|------|-----------|--------|
| 1a | Deergham (long matra) | Syllable contains a long dependent vowel: {ಾ, ೀ, ೂ, ೇ, ೋ, ೌ} | **U** (Guru) |
| 1b | Deergham (long independent vowel) | Syllable is a standalone long vowel: {ಆ, ಈ, ಊ, ಏ, ಓ} | **U** (Guru) |
| 2 | Pluta (diphthong) | Syllable contains ಐ / ಔ (independent) or ೈ / ೌ (dependent) | **U** (Guru) |
| 3a | Diacritic (anusvara) | Syllable contains ಂ (anusvara) | **U** (Guru) |
| 3b | Diacritic (visarga) | Syllable contains ಃ (visarga) | **U** (Guru) |
| 4 | Pollu (halant ending) | Syllable ends with virama ್ | **U** (Guru) |
| 5 | Sandhi (conjunct) | **Next** syllable *in same word* contains a C+್+C conjunct or doubled consonant | **U** (Guru) |
| -- | Default | None of the above | **I** (Laghu) |

**Word-boundary blocking**: Rule 5 is blocked by word boundaries (space or newline). A syllable that is intrinsically Laghu becomes Guru only if the immediately following syllable *within the same word* begins with a conjunct cluster.

### 2.1 Syllable Weight Examples

| Word | Syllables | Markers | Rules Applied |
|------|-----------|---------|---------------|
| ಕರಿ | ಕ \| ರಿ | I I | Default for both |
| ಕಾಲ | ಕಾ \| ಲ | U I | Rule 1a (long matra ಾ) |
| ಆಗ | ಆ \| ಗ | U I | Rule 1b (long independent vowel ಆ) |
| ಐದು | ಐ \| ದು | U I | Rule 2 (diphthong ಐ) |
| ಸಂತಸ | ಸಂ \| ತ \| ಸ | U I I | Rule 3a (anusvara ಂ) |
| ದುಃಖ | ದುಃ \| ಖ | U I | Rule 3b (visarga ಃ) |
| ಕೃಷ್ಣ | ಕೃ \| ಷ್ಣ | U I | Rule 5 (next syllable has conjunct ಷ್ಣ) |
| ಕನ್ನಡ | ಕ \| ನ್ನ \| ಡ | U I I | Rule 5 (next syllable has doubled ನ್ನ) |
| ಕನ ಕೃಷಿ | ಕ \| ನ \| ಕೃ \| ಷಿ | I I I I | Rule 5 blocked by word boundary |

## 3. Metrical Feet (Ganalu)

Each line is partitioned into exactly **4 ganas**. The following table lists the permitted and forbidden gana patterns:

### 3.1 Permitted Gana Patterns

| Gana Pattern | Syllable Count | Matras | Positions Allowed | Kannada Name | Description |
|-------------|----------------|--------|-------------------|--------------|-------------|
| **III** | 3 | 3 | Ganas 1, 2, 3 | ಲಘು-ಲಘು-ಲಘು | Three consecutive Laghu syllables |
| **IIU** | 3 | 4 | Ganas 1, 2, 3, 4 | ಲಘು-ಲಘು-ಗುರು | Two Laghu followed by one Guru |

### 3.2 Forbidden Gana Pattern

| Pattern | Syllable Count | Reason for Prohibition |
|---------|----------------|----------------------|
| **IU** | 2 | Breaks rhythmic flow; a Laghu-Guru pair disrupts the triple-syllable gana structure |

### 3.3 Gana 4 Constraint

Gana 4 (the final foot of each line) **must be IIU**. This structurally enforces the guru ending rule -- the last syllable of every line is always Guru (U). Ganas 1-3 may be either III or IIU.

### 3.4 Formal Language

```
L_line = (III | IIU)^3 . IIU
```

Each line must match three free-choice ganas (III or IIU) followed by one mandatory IIU gana.

### 3.5 Valid Line Patterns

All 8 valid gana combinations for a single line:

| # | Gana 1 | Gana 2 | Gana 3 | Gana 4 | Marker String |
|---|--------|--------|--------|--------|---------------|
| 1 | III | III | III | IIU | I I I I I I I I I I I U |
| 2 | III | III | IIU | IIU | I I I I I I I I U I I U |
| 3 | III | IIU | III | IIU | I I I I I U I I I I I U |
| 4 | III | IIU | IIU | IIU | I I I I I U I I U I I U |
| 5 | IIU | III | III | IIU | I I U I I I I I I I I U |
| 6 | IIU | III | IIU | IIU | I I U I I I I I U I I U |
| 7 | IIU | IIU | III | IIU | I I U I I U I I I I I U |
| 8 | IIU | IIU | IIU | IIU | I I U I I U I I U I I U |

## 4. Guru Ending

Both lines must end on a **Guru** syllable (U). This is structurally enforced by the Gana 4 constraint: since Gana 4 must be IIU, the last syllable is always U.

## 5. Adi Prasa (Second-Syllable Rhyme)

The **base consonant** of the **2nd syllable** (aksharam) of line 1 must match that of line 2. "Base consonant" is the first Kannada consonant character of the syllable, ignoring vowel marks (matras), conjunct extensions, anusvara, and visarga.

### 5.1 Consonant Equivalence Groups

The NFA pipeline optionally relaxes exact matching with these equivalence groups:

| Group | Equivalent Consonants | Class Name |
|-------|----------------------|------------|
| Laterals | ಲ (la) ↔ ಳ (la) | lateral |
| Sibilants | ಶ (sha) ↔ ಷ (sha) ↔ ಸ (sa) | sibilant |

**Strict mode**: Only exact consonant match is accepted (used by `kannada_ragale_analyser.py`).

**Relaxed mode**: Consonants within the same equivalence group are treated as matching (supported by `prasa_nfa.py`).

### 5.2 Adi Prasa Examples

| Line 1 (2nd syl) | Line 2 (2nd syl) | Base C1 | Base C2 | Match? | Type |
|-------------------|-------------------|---------|---------|--------|------|
| ಲದಾ | ಲದೇ | ಲ | ಲ | Yes | Exact |
| ರೆಯಾ | ರಗೀ | ರ | ರ | Yes | Exact |
| ಲದಾ | ಳಿಯೂ | ಲ | ಳ | Yes (relaxed) | Lateral equivalence |
| ಶರೀ | ಸರೂ | ಶ | ಸ | Yes (relaxed) | Sibilant equivalence |
| ಲದಾ | ರಗೀ | ಲ | ರ | No | Mismatch |

## 6. Scoring Weights

The analyser scores poems with the following weighted breakdown:

| Component | Weight | Description |
|-----------|--------|-------------|
| Syllable count | 20% | Must be exactly 12 per line; each deviation of 1 syllable costs 20 points |
| Gana pattern validity | 30% | Fraction of valid ganas (out of 4 per line), averaged across both lines |
| Guru ending | 15% | 100% if both lines end Guru, 50% if one, 0% if neither |
| Adi Prasa | 35% | 100% if 2nd-syllable consonants match, 0% otherwise |

## Appendix A: Kannada Character Sets

### A.1 Consonants (Vyanjana) -- 34 characters

| Varga | Characters |
|-------|-----------|
| Ka-varga (velar) | ಕ ಖ ಗ ಘ ಙ |
| Cha-varga (palatal) | ಚ ಛ ಜ ಝ ಞ |
| Ta-varga (retroflex) | ಟ ಠ ಡ ಢ ಣ |
| Ta-varga (dental) | ತ ಥ ದ ಧ ನ |
| Pa-varga (labial) | ಪ ಫ ಬ ಭ ಮ |
| Antastha (approximants) | ಯ ರ ಲ ವ |
| Ushma (fricatives) | ಶ ಷ ಸ ಹ |
| Other | ಳ |

### A.2 Independent Vowels (Svara) -- 13 characters

| Type | Characters |
|------|-----------|
| Short | ಅ ಇ ಉ ಋ ಎ ಒ |
| Long | ಆ ಈ ಊ ಏ ಓ |
| Diphthong | ಐ ಔ |

### A.3 Dependent Vowel Signs (Matra) -- 12 characters

| Matra | Vowel | Weight |
|-------|-------|--------|
| ಾ | aa | Guru |
| ಿ | i | Laghu |
| ೀ | ii | Guru |
| ು | u | Laghu |
| ೂ | uu | Guru |
| ೃ | ri | Laghu |
| ೆ | e | Laghu |
| ೇ | ee | Guru |
| ೈ | ai | Guru |
| ೊ | o | Laghu |
| ೋ | oo | Guru |
| ೌ | au | Guru |

### A.4 Guru-Triggering Characters Summary

| Category | Characters | Rule |
|----------|-----------|------|
| Long matras | ಾ ೀ ೂ ೇ ೋ ೌ | Rule 1a |
| Long independent vowels | ಆ ಈ ಊ ಏ ಓ | Rule 1b |
| Diphthongs (independent) | ಐ ಔ | Rule 2 |
| Diphthongs (dependent) | ೈ ೌ | Rule 2 |
| Anusvara | ಂ | Rule 3a |
| Visarga | ಃ | Rule 3b |
| Virama/Halant | ್ (when ending syllable) | Rule 4 |
