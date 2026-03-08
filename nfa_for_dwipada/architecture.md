# NFA Constrained Decoding — Full Architecture

End-to-end architecture from Gemma token stream to valid token mask,
showing the three FST stages and three parallel NFA streams.

---

## Overview

The system intercepts Gemma's generation process at each decoding step and
masks out any token that would violate Telugu Dwipada metre. It does this by
maintaining a set of finite automata that track the metrical state of the poem
being generated in real time.

The architecture has two layers:

**Layer 1 — The Orchestrator (three chained FSTs)**

Before any metrical judgement can be made, each raw Gemma token (a variable-
length Unicode string) must be reduced to a stream of classified syllables with
U/I labels. This is a pure data transformation — no acceptance or rejection —
so it is implemented as a pipeline of three Finite State Transducers (FSTs):

1. **Codepoint Classifier** — stateless; maps each Unicode character to a
   linguistic category (CONSONANT, MATRA, VIRAMA, DIACRITIC, SPACE, etc.).
   Takes raw text, produces a typed character stream.

2. **Syllable Assembler** — 4 states; groups the typed character stream into
   complete aksharalu (syllables). Handles conjuncts (C + ్ + C), trailing
   virama (pollu) merge, independent vowels, and SPACE/NEWLINE boundaries.
   Produces a syllable stream.

3. **Guru/Laghu Classifier** — 3 states with a 1-syllable delay buffer;
   classifies each syllable as Guru (U) or Laghu (I). Applies all six
   classification rules including Rule 5 (syllable before a conjunct becomes
   Guru) with word-boundary awareness (a SPACE between syllables blocks Rule 5).
   Produces a U/I stream.

**Layer 2 — The Position Tracker and Three Parallel NFAs**

Once syllables are labelled U or I, the Position Tracker knows exactly where
in the dwipada structure each syllable falls (which gana, which line). It
routes three different signals to three independent NFAs running in parallel:

- **Gana NFA** (~60 states) — validates the metrical gana pattern of each
  line. Each line must be exactly three Indra ganas followed by one Surya gana:
  `(IIII|IIIU|IIUI|UII|UIU|UUI)³ · (III|UI)`. This NFA is non-deterministic
  because after seeing the first syllable of a gana (e.g., "I"), multiple gana
  types remain possible simultaneously. It receives a U/I signal for every
  syllable.

- **Prasa NFA** (~35 states) — enforces the rhyme constraint. It records the
  base consonant class of the 2nd syllable of line 1, then verifies that the
  2nd syllable of line 2 shares the same consonant class (with equivalences:
  ల↔ళ, శ↔స, ఱ↔ర). It only receives a signal at the 2nd syllable of each line.

- **Yati NFA** (~28 states) — enforces the alliteration constraint. It records
  the maitri group of the 1st syllable of gana 1, then verifies that the 1st
  syllable of gana 3 belongs to the same maitri group (~11 equivalence classes).
  It only receives a signal at the start of gana 1 and gana 3.

All three NFAs run on the same syllable stream but consume different projections
of it at different positions. After each token, each NFA's active state set is
pruned against a precomputed co-reachability set (states that can still reach
an accept state). If any NFA's active set becomes empty, the token is invalid.

**The Mask**

At each generation step, before sampling, the system computes a boolean mask
over all 262,144 vocabulary tokens. A token is valid only if feeding its
Unicode characters through all three stages leaves all three NFAs alive. Invalid
tokens are set to −∞ logits so Gemma can never sample them. The result is a
model that generates Telugu text constrained to valid Dwipada metre at every
single token step.

---

---

```mermaid
flowchart TD
    %% ── INPUT ──────────────────────────────────────────────────────────────
    GEM["🔤 Gemma Model\nlogits over 262,144 tokens"]
    TOK["tokenizer.decode(token_id)\nraw Unicode text"]

    GEM -->|"next token id"| TOK

    %% ── FST PIPELINE ────────────────────────────────────────────────────────
    subgraph FST ["Orchestrator FST  (all three stages are FSTs — deterministic transducers)"]
        direction TB

        CP["Stage 1 · Codepoint Classifier\n―――――――――――――――――――――――――――\nstateless · 1 state\nchar → Category\n{CONSONANT, MATRA, VIRAMA,\nINDEP_VOWEL, DIACRITIC,\nSPACE, NEWLINE, SKIP}"]

        SA["Stage 2 · Syllable Assembler\n―――――――――――――――――――――――――――\n4 states\nIDLE → CONSONANT_CLUSTER\n→ PENDING_VIRAMA → VOWEL\nCategory stream → aksharalu\n+ SPACE + NEWLINE boundaries"]

        GL["Stage 3 · Guru / Laghu Classifier\n―――――――――――――――――――――――――――\n3 states · 1-syllable delay buffer\nPENDING_I · PENDING_U · EMPTY\nword_boundary flag (blocks Rule 5)\nsyllable → U or I"]

        CP -->|"(char, Category) stream"| SA
        SA -->|"syllable / SPACE / NEWLINE stream"| GL
    end

    TOK -->|"char by char"| CP

    %% ── POSITION TRACKER ────────────────────────────────────────────────────
    PT["Position Tracker\n―――――――――――――――――――――――――――\nknows: which syllable #\nwhich gana {1,2,3,surya}\nwhich line {1,2}\nroutes signals selectively"]

    GL -->|"U / I  +  syllable metadata"| PT

    %% ── THREE PARALLEL NFA STREAMS ──────────────────────────────────────────
    subgraph NFAs ["Three Parallel NFA Streams  (run simultaneously, independently)"]
        direction LR

        subgraph G ["Gana NFA"]
            GS["~60 states\nvalidates gana pattern\nper line:\n(IIII|IIIU|IIUI|\nUII|UIU|UUI)³\n· (III|UI)"]
        end

        subgraph P ["Prasa NFA"]
            PS["~35 states\nline 1: store base\nconsonant of 2nd syllable\nline 2: check it matches\n(ల↔ళ · శ↔స · ఱ↔ర)"]
        end

        subgraph Y ["Yati NFA"]
            YS["~28 states\nstore maitri group of\ngana 1's 1st syllable\ncheck it matches\ngana 3's 1st syllable\n(~11 equivalence groups)"]
        end
    end

    PT -->|"U/I\nevery syllable"| G
    PT -->|"base consonant\nonly at 2nd syllable\nof each line"| P
    PT -->|"maitri group\nonly at gana 1\nand gana 3 start"| Y

    %% ── DEAD STATE DETECTION ─────────────────────────────────────────────────
    DS["Dead State Detection\n―――――――――――――――――――――――――――\nper NFA:\nactive_states ∩ can_reach_accept\nif empty → NFA is dead\n(precomputed via backward BFS)"]

    G --> DS
    P --> DS
    Y --> DS

    %% ── MASK COMBINATION ────────────────────────────────────────────────────
    AND{"AND\nall three\nalive?"}
    DS --> AND

    %% ── OUTPUT ──────────────────────────────────────────────────────────────
    MASK["valid_mask\n(boolean vector\nover vocab)"]
    FILTER["Token Filter\nlogits[~valid_mask] = −∞"]
    SAMPLE["Sampling\nnext_token_id"]

    AND -->|"yes → token is valid"| MASK
    MASK --> FILTER
    GEM -->|"raw logits"| FILTER
    FILTER --> SAMPLE
    SAMPLE -->|"advance NFA states\nfeed next token"| TOK

    %% ── SELECTIVE ENFORCEMENT ───────────────────────────────────────────────
    SE["Selective Enforcement\n―――――――――――――――――――――――――――\nGana NFA   → always enforced\nPrasa NFA  → enforced on line 2\nYati NFA   → optional (strict mode)"]

    AND -.->|"configured by"| SE

    %% ── STYLING ─────────────────────────────────────────────────────────────
    classDef fst    fill:#dbeafe,stroke:#3b82f6,color:#1e3a5f
    classDef nfa    fill:#dcfce7,stroke:#16a34a,color:#14532d
    classDef ctrl   fill:#fef9c3,stroke:#ca8a04,color:#713f12
    classDef io     fill:#f3e8ff,stroke:#9333ea,color:#3b0764
    classDef gate   fill:#fee2e2,stroke:#dc2626,color:#7f1d1d

    class CP,SA,GL fst
    class GS,PS,YS nfa
    class PT,DS,SE ctrl
    class GEM,TOK,MASK,FILTER,SAMPLE io
    class AND gate
```

---

## Signal Routing Summary

The Position Tracker is the router — it knows exactly which syllable of which
gana of which line is being processed, and selectively forwards signals:

| Signal | From | To | When |
|--------|------|----|------|
| U / I marker | Guru/Laghu FST | **Gana NFA** | Every syllable |
| Base consonant class | Syllable metadata | **Prasa NFA** | 2nd syllable of line 1 and line 2 only |
| Maitri group | Syllable metadata | **Yati NFA** | 1st syllable of gana 1 and gana 3 only |

## Why Three Separate NFAs

The three NFAs consume **different projections** of the same syllable stream
at **different positions**. Keeping them separate gives:

- **Debuggability** — know which rule failed
- **Partial scoring** — gana 40%, prasa 35%, yati 25%
- **Selective enforcement** — enforce gana strictly, relax yati
- **Smaller state space** — 3 × ~40 states vs one product automaton of ~46,000

## Constrained Decoding Loop

```python
for each generation step:
    logits     = gemma.forward(input_ids)          # model output
    valid_mask = get_valid_tokens(nfa_states)       # AND of all 3 NFAs
    logits[~valid_mask] = float('-inf')             # kill invalid tokens
    next_token = sample(logits)                     # sample from valid only
    nfa_states = advance(nfa_states, next_token)    # update all 3 NFA states
```
