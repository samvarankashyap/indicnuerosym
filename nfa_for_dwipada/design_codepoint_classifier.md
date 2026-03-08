# Design: Telugu Codepoint Classifier FST

A theoretical design for a Finite State Transducer (FST) that accepts a stream
of Unicode characters from a Telugu string and emits a stream of classified
codepoints — the first stage feeding into the syllable assembler.

---

## The Problem

Gemma's BPE tokens are variable-length chunks of text. Before any syllable or
gana logic can run, we need to reduce every token — or any raw Telugu string —
to a flat stream of **typed codepoints**, one character at a time.

Given:
```
Input  : "నమస్కారం"   (a single Telugu word, 1 Gemma token or raw text)
Output : [(న, CONSONANT), (మ, CONSONANT), (స, CONSONANT), (్, VIRAMA),
          (క, CONSONANT), (ా, MATRA), (ర, CONSONANT), (ం, ANUSVARA)]
```

This is purely a **classification** problem — no state needed to classify a
single codepoint. But it is the foundation on which the stateful syllable
assembler FST (Stage 2) operates.

---

## Codepoint Categories

Based on the Telugu Unicode block (U+0C00–U+0C7F) and the characters confirmed
present in Gemma's vocabulary:

```
Category        Symbol    Codepoint Range / Values          Examples
---------------------------------------------------------------------------
CONSONANT       C         U+0C15–U+0C39, U+0C31, U+0C33    క ఖ గ ... హ ళ ఱ
INDEPENDENT_V   V         U+0C05–U+0C14                    అ ఆ ఇ ఈ ఉ ఊ ఋ
                                                            ఎ ఏ ఐ ఒ ఓ ఔ
MATRA           M         U+0C3E–U+0C4C                    ా ి ీ ు ూ ృ
                                                            ె ే ై ొ ో ౌ
VIRAMA          V̄         U+0C4D                           ్
ANUSVARA        A         U+0C02                           ం
VISARGA         VS        U+0C03                           ః
AI_LENGTH       AL        U+0C56                           ౖ   (rare)
SPACE           SP        U+0020                           ' '
NEWLINE         NL        U+000A                           '\n'
ZWNJ            ZW        U+200C                           ‌
OTHER           OT        anything else                    digits, punct, ASCII
```

The mapping is a **pure function** — no state required:

```
classify(codepoint) -> Category
```

---

## FST Design: The Classifier

### Machine Definition

```
Type       : Finite State Transducer (FST)
Input      : a single Unicode codepoint
Output     : (codepoint, Category)
States     : 1  (it is stateless — one universal state q0)
Transitions: one per codepoint range (lookup table)
```

Because classification is stateless, this is technically a **Mealy machine**
with a single state that loops on every input symbol and emits an output pair.

```
                  ┌─────────────────────────────────────┐
                  │                                     │
              ┌───┴───┐   classify(cp)                  │
──── cp ────> │  q0   │ ─────────────────> (cp, Cat) ───┘
              └───────┘
```

### Lookup Table (ranges)

```
Codepoint(s)              Category
------------------------------------------
0x0C15 – 0x0C39          CONSONANT
0x0C31                   CONSONANT  (ఱ, within above range)
0x0C33                   CONSONANT  (ళ, within above range)
0x0C05 – 0x0C14          INDEPENDENT_V
0x0C3E – 0x0C4C          MATRA
0x0C4D                   VIRAMA
0x0C02                   ANUSVARA
0x0C03                   VISARGA
0x0C56                   AI_LENGTH
0x0020                   SPACE
0x000A                   NEWLINE
0x200C                   ZWNJ
*                        OTHER
```

---

## Input: Any Telugu String

The classifier accepts input from three possible sources without any change:

```
Source 1: Raw Telugu string
    "నమస్కారం"
    → iterate char by char → classify each

Source 2: Gemma token (already decoded to text)
    tokenizer.decode([token_id])  →  "స్కా"
    → same iteration

Source 3: Multiple sentences
    "నమస్కారం\nఏమిటి?"
    → same iteration; NEWLINE codepoints appear naturally in the stream
```

All three reduce to the same flat codepoint stream:

```
for ch in text:
    emit (ch, classify(ch))
```

---

## Output Stream Examples

### Single word: "నమస్కారం"

```
Position   Char   Codepoint   Category
---------------------------------------
0          న      U+0C28      CONSONANT
1          మ      U+0C2E      CONSONANT
2          స      U+0C38      CONSONANT
3          ్      U+0C4D      VIRAMA
4          క      U+0C15      CONSONANT
5          ా      U+0C3E      MATRA
6          ర      U+0C30      CONSONANT
7          ం      U+0C02      ANUSVARA
```

### Two words with space: "తెలుగు భాష"

```
Position   Char   Codepoint   Category
---------------------------------------
0          త      U+0C24      CONSONANT
1          ె      U+0C46      MATRA
2          ల      U+0C32      CONSONANT
3          ు      U+0C41      MATRA
4          గ      U+0C17      CONSONANT
5          ు      U+0C41      MATRA
6          ' '    U+0020      SPACE         ← word boundary
7          భ      U+0C2D      CONSONANT
8          ా      U+0C3E      MATRA
9          ష      U+0C37      CONSONANT
```

### Two lines: "ఆదిన్\nమదిన్"

```
Position   Char   Category
---------------------------
...        ...    ...
5          ్      VIRAMA
6          '\n'   NEWLINE       ← line boundary
7          మ      CONSONANT
...
```

---

## What the Next Stage Receives

The codepoint classifier's output stream is consumed directly by the
**syllable assembler FST** (Stage 2, designed separately). The assembler
reads Category signals — not raw codepoints — to decide syllable boundaries:

```
Classifier output stream
    (న, CONSONANT) (మ, CONSONANT) (స, CONSONANT) (్, VIRAMA) (క, CONSONANT) ...
                │
                ▼
    Syllable Assembler FST
        reads: CONSONANT CONSONANT CONSONANT VIRAMA CONSONANT ...
        emits: syllable "న"  syllable "మ"  syllable "స్క" ...
```

The assembler only needs Categories, not the actual codepoints, for its
state transitions — except when storing the actual character for Prasa/Yati
comparison.

---

## Properties

| Property              | Value                                              |
|-----------------------|----------------------------------------------------|
| Machine type          | Mealy FST (stateless transducer)                   |
| Number of states      | 1                                                  |
| Input alphabet        | All Unicode codepoints (~1.1M possible)            |
| Output alphabet       | Category enum (10 values)                          |
| Transition function   | Range lookup table (O(1) per codepoint)            |
| Memory                | None — no state carried between codepoints         |
| Composable?           | Yes — pipes directly into syllable assembler FST   |

---

## Relation to the Full Pipeline

```
Raw Telugu string / Gemma token text
          │
          ▼
┌─────────────────────┐
│  Codepoint          │   ← THIS DOCUMENT
│  Classifier FST     │   1 state, O(1) lookup per char
│  (stateless)        │
└──────────┬──────────┘
           │  stream of (char, Category)
           ▼
┌─────────────────────┐
│  Syllable Assembler │   ~6 states, groups codepoints into aksharalu
│  FST                │   (design: Stage 2)
└──────────┬──────────┘
           │  stream of complete syllables
           ▼
┌─────────────────────┐
│  Guru/Laghu         │   classifies each syllable as U or I
│  Classifier         │   with 1-syllable lookahead for Rule 5
└──────────┬──────────┘
           │  stream of U/I markers
           ▼
┌─────────────────────┐
│  Gana / Prasa /     │   the three NFAs that validate dwipada structure
│  Yati NFAs          │
└─────────────────────┘
```
