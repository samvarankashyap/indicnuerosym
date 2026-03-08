# Guru / Laghu Rules — Telugu Chandassu

Every Telugu syllable (akshara) is classified as either:

- **Guru (U)** — heavy / long syllable
- **Laghu (I)** — light / short syllable

The classification uses 5 rules, applied in order. The first rule that matches
wins. If none match, the syllable is Laghu by default (it carries only the
short inherent vowel అ).

---

## Rule 1 — Long Vowel (దీర్ఘ స్వరం)

**Condition:** The syllable contains a long vowel matra or an independent long vowel.

| Type | Characters |
|------|-----------|
| Long matras (attached) | ా ీ ూ ే ో ౌ ౄ |
| Independent long vowels | ఆ ఈ ఊ ౠ ఏ ఓ |

**Examples:**

| Syllable | Reason | Class |
|----------|--------|-------|
| రా | contains ా (long-a matra) | U |
| నీ | contains ీ (long-i matra) | U |
| భూ | contains ూ (long-u matra) | U |
| ఆ | independent long vowel | U |
| కే | contains ే (long-e matra) | U |
| రో | contains ో (long-o matra) | U |

> **Note:** ృ (U+0C43, short vocalic-R matra) is NOT a long vowel.
> Only ౄ (U+0C44, long vocalic-R matra) qualifies under this rule.

---

## Rule 2 — Diphthong (సంధ్యక్షరం)

**Condition:** The syllable contains a diphthong matra or an independent diphthong vowel.

| Type | Characters |
|------|-----------|
| Diphthong matras | ై ౌ |
| Independent diphthongs | ఐ ఔ |

**Examples:**

| Syllable | Reason | Class |
|----------|--------|-------|
| గై | contains ై | U |
| వౌ | contains ౌ | U |
| ఐ | independent diphthong | U |
| ఔ | independent diphthong | U |

> **Note:** ౌ appears in both Rule 1 and Rule 2 — it matches either way.
> Rule 1 is checked first, so it is caught there.

---

## Rule 3 — Anusvara or Visarga (అనుస్వారం / విసర్గ)

**Condition:** The syllable contains anusvara (ం) or visarga (ః).

| Character | Name | Unicode |
|-----------|------|---------|
| ం | Anusvara | U+0C02 |
| ః | Visarga | U+0C03 |

**Examples:**

| Syllable | Reason | Class |
|----------|--------|-------|
| సం | contains anusvara ం | U |
| రాం | contains anusvara ం (also Rule 1, but Rule 3 catches it too) | U |
| దుః | contains visarga ః | U |
| నమః | contains visarga ః | U |

---

## Rule 4 — Pollu Hallu / Trailing Virama (పొల్లు హల్లు)

**Condition:** The syllable ends with a virama ్ (halant), meaning it is a bare
consonant with no following vowel.

| Character | Name | Unicode |
|-----------|------|---------|
| ్ | Virama (halant) | U+0C4D |

**Examples:**

| Syllable | Reason | Class |
|----------|--------|-------|
| న్ | bare consonant, ends with ్ | U |
| క్ | bare consonant, ends with ్ | U |
| స్ | bare consonant, ends with ్ | U |
| త్ | bare consonant, ends with ్ | U |

> These are "half-consonants" that appear at the end of words or as the
> first half of a conjunct cluster.

---

## Rule 5 — Sandhi Lookahead (సంధి)

**Condition:** The **next** syllable (immediately following, within the same word)
begins with a conjunct consonant (C + ్ + C).

This rule is unique — it is decided by what comes **after** the current syllable,
not by the syllable's own contents.

**Why it works:** When a conjunct consonant follows, the first half of that
cluster is phonetically "borrowed back" to close the preceding syllable,
making it heavy.

**Conjunct start check:**
```
is_conjunct_start(s) = (
    len(s) >= 3
    and s[0] is a Telugu consonant
    and s[1] == '్'   ← virama at position 1
    and s[2] is a Telugu consonant
)
```

Covers:
- Distinct conjuncts: `స్క`, `త్య`, `ప్ర`, `స్త్ర`
- Doubled consonants: `మ్మ`, `ళ్ళ`, `న్న`, `క్క`

**Examples:**

| Sequence | Next syllable | Rule 5 fires? | Current class |
|----------|--------------|---------------|---------------|
| స + త్య | త్య (conjunct) | YES | U |
| స + య | య (simple) | no | I |
| క + ష్ణ | ష్ణ (conjunct) | YES | U |
| రా + మ్మ | మ్మ (doubled) | YES | U (already U by Rule 1, but Rule 5 also applies) |

### Word Boundary Suppression

Rule 5 applies **only within the same word**. A space between words blocks it.
A newline is transparent (ignored for this rule).

```
Same word:      స + త్య    →  స = U   (Rule 5 fires)
Across space:   స + ' ' + త్య  →  స = I   (Rule 5 blocked by space)
Across newline: స + '\n' + త్య  →  స = U   (newline is transparent, Rule 5 fires)
```

**Concrete example:**

```
తనుమ ళ్ళరాస్తుంది

తు  ను  మ   ళ్ళ  రా  స్తుం  ది
I   I   I   I    U   U     I

మ is in a different word from ళ్ళ (space separates them).
Even though ళ్ళ is a conjunct start, Rule 5 is suppressed.
మ stays I.
```

---

## Default — Laghu (లఘువు)

If none of the 5 rules match, the syllable is **Laghu (I)**.

This covers syllables with:
- A short inherent vowel: క కి కు కె కొ
- A short vocalic-R matra: కృ (ృ is short, not long)
- Any plain consonant + short vowel combination

---

## Summary Table

| Rule | Name | Trigger | Result |
|------|------|---------|--------|
| 1 | Long Vowel | ా ీ ూ ే ో ౌ ౄ or ఆ ఈ ఊ ౠ ఏ ఓ inside syllable | U |
| 2 | Diphthong | ై ౌ or ఐ ఔ inside syllable | U |
| 3 | Anusvara/Visarga | ం or ః inside syllable | U |
| 4 | Trailing Virama | syllable ends with ్ | U |
| 5 | Sandhi Lookahead | next syllable (same word) starts with C+్+C | U |
| — | Default | none of the above | I |

---

## Implementation Reference

- `nfa_for_dwipada/ganana_marker.py` — FST implementing all 5 rules
- `src/dwipada/core/aksharanusarika.py` — original reference: `akshara_ganavibhajana()`
- `src/dwipada/core/analyzer.py` — full dwipada analyser (ground truth for testing)
