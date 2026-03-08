# NFA-Based Constrained Decoding for Dwipada Poetry Generation

A theoretical design for using Non-deterministic Finite Automata to constrain
Gemma's token generation to produce metrically valid Telugu Dwipada poetry.

---

## Table of Contents

1. [Can Dwipada Rules Be Modeled as an NFA?](#1-can-dwipada-rules-be-modeled-as-an-nfa)
2. [Gemma Tokens Flowing Into an NFA](#2-gemma-tokens-flowing-into-an-nfa)
3. [Tracking Prasa and Yati — Do We Need a Turing Machine?](#3-tracking-prasa-and-yati--do-we-need-a-turing-machine)
4. [Three Separate NFAs](#4-three-separate-nfas)
5. [Decomposing Tokens to Unicode Codepoints](#5-decomposing-tokens-to-unicode-codepoints)
6. [Dead State Detection](#6-dead-state-detection)
7. [Space Tracking for Word Boundaries](#7-space-tracking-for-word-boundaries)
8. [Constrained Decoding Architecture](#8-constrained-decoding-architecture)

---

## 1. Can Dwipada Rules Be Modeled as an NFA?

### Dwipada Rules Summary

From `src/dwipada/core/constants.py` and `src/dwipada/core/analyzer.py`:

- **Alphabet**: Each syllable is classified as **Guru (U)** or **Laghu (I)** — so the input alphabet is Sigma = {U, I}
- **Structure per line**: 3 Indra ganas + 1 Surya gana (read left to right)
- **Indra ganas** (each is a pattern of U/I):
  - Nala (నల): `IIII`
  - Naga (నగ): `IIIU`
  - Sala (సల): `IIUI`
  - Bha (భ): `UII`
  - Ra (ర): `UIU`
  - Ta (త): `UUI`
- **Surya ganas**:
  - Na (న): `III`
  - Ha/Gala (హ/గల): `UI`

### NFA Construction (for one line/pada)

**States** represent progress through the 4 gana slots:

```
q0 --[Indra1]--> q1 --[Indra2]--> q2 --[Indra3]--> q3 --[Surya]--> q_accept
```

Each `--[Indra]-->` transition expands into sub-states for each possible Indra gana
pattern. Since there are 6 Indra gana options and 2 Surya gana options, the NFA
non-deterministically "guesses" which gana is being read and verifies the U/I sequence.

For example, from state `q0`, reading the first syllable:
- On `I` -> could be Nala, Naga, or Sala (need more symbols to decide)
- On `U` -> could be Bha, Ra, or Ta

This is exactly where **non-determinism** is useful — the NFA branches into all valid
possibilities and accepts if any branch reaches the accept state.

Each gana slot needs at most 4 sub-states (Nala is the longest at 4 symbols). With
4 gana slots x ~4 sub-states x 6 branches for Indra (or 2 for Surya), the NFA for
one line has roughly **50-70 states**.

### What the NFA Validates vs. What It Can't

| Rule                          | NFA-encodable? | Why                                                        |
|-------------------------------|----------------|------------------------------------------------------------|
| Gana pattern (U/I sequence)   | **Yes**        | Regular language over {U, I}                               |
| Total syllable count per line | **Yes**        | Follows from gana patterns (12-15 syllables)               |
| Prasa (2nd syl consonant)     | **Partially**  | Requires comparing across two lines — product automaton     |
| Yati (gana 1 = gana 3 start) | **Partially**  | Requires remembering a symbol — finite alphabet expansion   |

### The Key Insight

The **gana pattern matching** for a single line is a straightforward **regular language**:

```
L_line = (Nala | Naga | Sala | Bha | Ra | Ta)^3 . (Na | Ha)
```

Which expands to:

```
L_line = (IIII | IIIU | IIUI | UII | UIU | UUI)^3 . (III | UI)
```

This is a **finite union of concatenations of finite strings** — trivially regular.

For a full dwipada (two lines): `L_dwipada = L_line . L_line` — still regular.

### Prasa and Yati Require Alphabet Expansion

- **Prasa**: Build a product automaton that tracks the 2nd syllable's consonant from
  line 1 and checks it matches in line 2. Since the consonant set is finite (~35),
  this multiplies states by ~35 — still finite.
- **Yati**: Track the 1st syllable of gana 1 and compare with the 1st syllable of
  gana 3. With Yati Maitri groups (~11 groups), this adds ~11x state multiplication.

### Conclusion

A complete dwipada validator can be expressed as an NFA (and therefore also a DFA).
The language of valid dwipada poems over the Telugu syllable alphabet is **regular**.

All dwipada constraints are:
1. **Fixed-length patterns** (gana sequences)
2. **Bounded look-back comparisons** (Prasa, Yati)
3. Over a **finite alphabet** (Telugu syllables)

No unbounded memory is needed — no counting, no nesting, no recursion — so finite
automata suffice.

---

## 2. Gemma Tokens Flowing Into an NFA

### The Problem: Token-Syllable Misalignment

The current analysis pipeline is:

```
Telugu text -> split_aksharalu() -> aksharalu list -> akshara_ganavibhajana() -> U/I markers -> gana matching
```

Gemma's BPE tokenizer splits Telugu text based on **byte-level statistical patterns**,
not linguistic syllable boundaries. A single Gemma token might:
- Contain **multiple syllables**: "సౌధా" -> 2 aksharalu (సౌ, ధా)
- Split a syllable **across tokens**: a conjunct like త్య could be split as త్ | య
- Include **whitespace + text** mixed in one token

### Solution: Two-Stage Automaton

#### Stage 1: Token-to-Syllable Transducer (Finite State Transducer / FST)

Each Gemma token is a fixed Unicode string from a finite vocabulary (~256K tokens).
Precompute for every Telugu-containing token:

```
token_id -> (prefix_fragment, [complete_syllables...], suffix_fragment)
```

Where:
- `prefix_fragment`: leftover characters that complete the previous token's trailing syllable
- `complete_syllables`: fully contained aksharalu within this token
- `suffix_fragment`: trailing characters that need the next token to form a complete syllable

This is a finite-state transducer because:
- **State** = the current pending syllable fragment (finite — bounded by max Telugu syllable length)
- **Input** = Gemma token ID (finite alphabet)
- **Output** = stream of complete aksharalu

#### Stage 2: Syllable-to-Guru/Laghu NFA

Once you have a syllable stream, the guru/laghu classification is **almost** a
finite-state problem, with one subtlety:

**Rule 5 (sandhi/lookahead)**: A syllable is Guru if the *next* syllable starts with
a conjunct. This requires **1-syllable lookahead**.

Handled by making the NFA state include the **current pending syllable** waiting for
the next one to determine its U/I classification — a 1-element buffer, still finite state.

#### Stage 3: Gana Pattern NFA

```
L_line = (IIII | IIIU | IIUI | UII | UIU | UUI)^3 . (III | UI)
```

### Combined Pipeline

```
Gemma token stream -> [FST: reassemble syllables] -> [NFA: classify U/I with 1-lookahead] -> [NFA: match gana patterns]
```

All three stages are finite-state, so **their composition is also finite-state**.

### Practical State Count Estimate

| Component                              | States              |
|----------------------------------------|----------------------|
| Syllable reassembly (pending fragment) | ~50-100              |
| Guru/Laghu lookahead buffer            | x2                   |
| Gana pattern progress                  | ~70                  |
| **Total (product)**                    | **~7,000-14,000**    |

### Application: Constrained Generation

This means you could **constrain Gemma's decoding at inference time** using this NFA —
at each generation step, mask out token IDs that would lead to an invalid dwipada state.
This is exactly how **guided/constrained generation** works (similar to grammar-constrained
decoding in tools like Outlines or guidance).

---

## 3. Tracking Prasa and Yati — Do We Need a Turing Machine?

### What Prasa and Yati Need to Track

**Prasa** (from `analyzer.py:153-161`):
> The base consonant of the 2nd syllable of line 1 must equal the base consonant
> of the 2nd syllable of line 2 (with equivalence: ల<->ళ, శ<->స, ఱ<->ర).

**Yati** (from `analyzer.py:110-125`):
> The 1st letter of gana 1 must match the 1st letter of gana 3 (within the same line),
> with Yati Maitri group equivalence (~11 groups).

### All Quantities Are Finite

| What to remember                 | Possible values                  | Size |
|----------------------------------|----------------------------------|------|
| Prasa consonant from line 1      | Base consonants (equiv. classes) | ~30  |
| Yati: gana 1's 1st letter group  | Yati Maitri groups               | ~11  |
| Current line (1 or 2)            | {line1, line2}                   | 2    |
| Current gana position            | {gana1, gana2, gana3, surya}     | 4    |
| Syllable counter within gana     | {0, 1, 2, 3}                     | 4    |

Everything is **bounded and finite**. No stack, no tape, no unbounded memory.

### Chomsky Hierarchy Perspective

| Machine                          | Memory             | Needed when                              | Dwipada?        |
|----------------------------------|--------------------|------------------------------------------|-----------------|
| **Finite Automaton (DFA/NFA)**   | Finite states only | Fixed patterns, bounded comparisons      | **Sufficient**  |
| **Pushdown Automaton (PDA)**     | Stack (unbounded)  | Nested matching like a^n b^n             | Not needed      |
| **Turing Machine**               | Infinite tape      | Arbitrary computation                    | Overkill        |

A Turing machine would be needed if dwipada had rules like:
- "The number of Guru syllables in line 1 must equal the number in line 2" (unbounded counting)
- "Recursively nested verse structures" (context-free)
- "The consonant pattern of line 1 must be the reverse of line 2" (requires full tape)

Dwipada has **none of these**. Every constraint involves comparing **specific fixed positions**
from a **finite alphabet**.

### Conclusion

```
Dwipada validation (including Prasa + Yati) = Regular Language

    Finite Automaton    <-- sufficient, this is the right tool
    Pushdown Automaton  <-- unnecessary (no nesting/recursion)
    Turing Machine      <-- massive overkill
```

---

## 4. Three Separate NFAs

### Why Separate?

Prasa and Yati are different NFAs from Gana because they consume **different projections**
of the same syllable stream at **different positions**:

| NFA   | Reads what          | When                                     |
|-------|---------------------|------------------------------------------|
| Gana  | Every U/I marker    | Every syllable                           |
| Prasa | Base consonant      | Only at 2nd syllable of each line        |
| Yati  | Maitri group        | Only at 1st syllable of gana 1 and gana 3 |

### Architecture with Orchestrator

The three NFAs don't talk to each other or to the token stream directly. A **fourth
component** — the orchestrator — sits in between:

```
                        Gemma Token Stream
                              |
                              v
                   +-------------------------+
                   |                         |
                   |     ORCHESTRATOR        |
                   |                         |
                   |  - reassembles          |
                   |    syllables            |
                   |  - classifies U/I       |
                   |  - tracks position      |
                   |    (which syllable,     |
                   |     which gana,         |
                   |     which line)         |
                   |                         |
                   +--+--------+----------+--+
                      |        |          |
                 U/I  | consonant    maitri
                every | at 2nd       group at
                syl   | syl only     gana 1,3
                      |        |     only
                      v        v          v
                 +------+ +------+  +----------+
                 | Gana | |Prasa |  |   Yati   |
                 | NFA  | | NFA  |  |   NFA    |
                 +--+---+ +--+---+  +----+-----+
                    |        |           |
                    v        v           v
                  alive?   alive?      alive?
                    |        |           |
                    +----+---+-----------+
                         |
                    ALL alive? --> token is valid
```

### Why Separate Is Better Than One Giant Product

1. **Debuggability** — when a poem fails, you know *which* rule failed
2. **Partial scoring** — matches existing scoring: gana 40%, prasa 35%, yati 25%
3. **State space** — three small NFAs (~70 + ~30 + ~22) vs one product (~46,000)
4. **Selective enforcement** — enforce gana strictly, allow yati relaxation

### The Orchestrator Is an FST, Not an NFA

| Component    | Machine type                     | Accepts/Rejects? |
|-------------|----------------------------------|-------------------|
| Orchestrator | **Finite State Transducer (FST)** | **No** — transforms, doesn't judge |
| Gana NFA    | NFA (acceptor)                   | **Yes** — valid gana pattern? |
| Prasa NFA   | NFA (acceptor)                   | **Yes** — consonants match? |
| Yati NFA    | NFA (acceptor)                   | **Yes** — groups match? |

The formal composition:

```
L_dwipada = L_gana  intersection  pi1_inverse(L_prasa)  intersection  pi2_inverse(L_yati)
```

Where pi1 and pi2 are projections that extract the relevant symbols at the relevant
positions. Regular languages are closed under intersection and inverse homomorphism,
so the combined machine is still an NFA.

### For Constrained Decoding

```python
valid_mask = gana_mask                           # always
if on_line_2:
    valid_mask &= prasa_mask                     # enforce rhyme
if strict_yati:
    valid_mask &= yati_mask                      # optional
```

---

## 5. Decomposing Tokens to Unicode Codepoints

### Why Codepoints Are Better Than Tokens

Gemma tokens are variable-length chunks of unicode. Decomposing first:

```
Gemma token "సౌధా" (1 token)
    | decompose
Unicode codepoints: స ౌ ధ ా (4 codepoints)
```

The orchestrator's input alphabet becomes **individual Telugu unicode codepoints** —
a fixed set of ~130 characters instead of ~5K variable-length Telugu tokens.

### Problems That Disappear

| Before (token-level input)                   | After (codepoint-level input)                |
|----------------------------------------------|----------------------------------------------|
| Syllable splits across tokens -> fragment buf | Syllable boundary = pattern match on types   |
| Multiple syllables in one token              | One codepoint at a time, one transition      |
| Arbitrary token boundaries -> precompute 5K  | ~130 well-understood symbols                 |
| Rule 5 lookahead across tokens               | Buffer one syllable (cleaner)                |

### Telugu Unicode Codepoint Categories

```
Consonant (hallu):        క ఖ గ ... హ ళ ఱ         (~35 codepoints)
Independent vowel:        అ ఆ ఇ ఈ ... ఔ           (~14 codepoints)
Dependent vowel (maatra): ా ి ీ ు ూ ... ౌ         (~13 codepoints)
Halant:                   ్                         (1 codepoint)
Anusvara:                 ం                         (1 codepoint)
Visarga:                  ః                         (1 codepoint)
Space/newline:            ' ' '\n'                  (2 codepoints)
```

### Orchestrator as Character-Level FST

Syllable assembly becomes a clean FST with ~5-6 states:

```
        +--------------------------------------------------+
        |                                                  |
        v                                                  |
   +---------+  consonant   +----------+  halant   +----------+
   |  START  |------------>| HAVE_C   |--------->| HAVE_C్  |
   +---------+              +----------+           +----------+
        |                        |                      |
        | indep.vowel            | dep.vowel/anusv/vis  | consonant
        |                        |                      | (conjunct!)
        v                        v                      |
   +---------+              +----------+                |
   | HAVE_V  |              | COMPLETE |<---------------+
   |(emit syl)|             |(emit syl)|
   +---------+              +----------+
```

### For Constrained Decoding: Map Back to Tokens

```python
def get_valid_tokens(fst_state, gana_state, prasa_state, yati_state):
    valid = []

    for token_id in telugu_tokens:
        codepoints = list(tokenizer.decode([token_id]))

        # Simulate: feed each codepoint through the FST + NFAs
        s_fst, s_g, s_p, s_y = fst_state, gana_state, prasa_state, yati_state
        alive = True

        for cp in codepoints:
            s_fst, signals = fst_transition(s_fst, cp)
            if signals.guru_laghu:
                s_g = gana_nfa.advance(s_g, signals.guru_laghu)
            if signals.prasa_consonant:
                s_p = prasa_nfa.advance(s_p, signals.prasa_consonant)
            if signals.yati_group:
                s_y = yati_nfa.advance(s_y, signals.yati_group)

            if is_dead(s_g) or is_dead(s_p) or is_dead(s_y):
                alive = False
                break

        if alive:
            valid.append(token_id)

    return valid
```

### State Count Comparison

| Approach              | Orchestrator states | Complexity                     |
|-----------------------|---------------------|--------------------------------|
| Token-level input     | ~100 (fragment buf) | Precompute table for ~5K tokens |
| **Codepoint-level**   | **~6**              | **Just unicode character classes** |

---

## 6. Dead State Detection

### Two Kinds of Dead

#### Kind 1: Immediately Dead — Active State Set Is Empty

After processing a codepoint, no NFA branch survived:

```python
active_states = gana_nfa.advance(active_states, "U")
if len(active_states) == 0:
    # DEAD -- no branch can continue
```

Easy to detect: `len(active_states) == 0`.

#### Kind 2: Eventually Dead — Alive but Doomed

The NFA has active states, but **no path from any of them reaches an accept state**.

Example:
```
Line 1 has consumed 14 syllables so far.
Gana NFA says: "still in gana 3, expecting 2 more syllables"
But 14 + 2 = 16 for Indra ganas alone, plus Surya (2-3 more) = 18-19.
Maximum valid line length is 15 (4+4+4+3).
NFA branches are alive but can NEVER reach accept.
```

Example for Prasa:
```
Line 2, past the 2nd syllable position.
Consonant was "చ" but line 1 stored "క".
Prasa NFA is in a non-accepting state permanently.
No future input can fix this.
```

### Solution: Precompute Co-Reachability

For each NFA state, precompute: **can this state reach any accept state?**

```python
# One-time precomputation using backward BFS from accept states
can_reach_accept = set()

queue = list(accept_states)
can_reach_accept = set(accept_states)

while queue:
    state = queue.pop()
    for pred in reverse_transitions[state]:
        if pred not in can_reach_accept:
            can_reach_accept.add(pred)
            queue.append(pred)

# At runtime:
def is_dead(active_states):
    live = active_states & can_reach_accept
    return len(live) == 0
```

After every transition, prune doomed branches:

```python
active_states = gana_nfa.advance(active_states, marker)
active_states = active_states & can_reach_accept   # prune doomed branches
if len(active_states) == 0:
    # Truly dead, no hope
```

### Per-NFA Dead State Detection

#### Gana NFA

```
States encode: (gana_index, symbols_consumed, pattern_branch)

Dead when:
- Empty active set (immediate)
- All active states need more syllables than line can hold (doomed)
- Overshot: consumed too many syllables (doomed)

Co-reachability handles all of these automatically.
```

#### Prasa NFA

```
States: (phase, stored_consonant)
  phase 1 (line 1): recording -- never dead
  phase 2 (line 2):
    - before 2nd syllable: alive
    - at 2nd syllable: alive only if consonant matches
    - after 2nd syllable with mismatch: DOOMED
```

#### Yati NFA

```
States: (stored_group, phase)
  phase: {recording_gana1, between, checking_gana3, done}

Dead when:
- At gana 3's 1st syllable, maitri group doesn't match stored group
- Past gana 3 start with mismatch -- DOOMED
```

### Gana-Yati Coupling

The Gana NFA's non-determinism means different branches have different gana boundary
positions, which affects where Yati checks happen:

```python
active_states = {
    (gana_state_A, yati_state_A),
    (gana_state_B, yati_state_B),
    (gana_state_C, yati_state_C),
}

# Dead only if ALL pairs are dead
def is_dead(active_states, can_reach_accept):
    return all(
        (g, y) not in can_reach_accept
        for g, y in active_states
    )
```

### Detection at Decode Time

```
On each codepoint:

1. Orchestrator FST produces signals
2. Advance each NFA
3. Prune: active_states &= can_reach_accept
4. Check: any NFA's active set empty?

   +---------+     +---------+     +---------+
   |Gana NFA |     |Prasa NFA|     |Yati NFA |
   |         |     |         |     |(per gana|
   | advance |     | advance |     | branch) |
   | prune   |     | prune   |     | advance |
   | empty?  |     | empty?  |     | prune   |
   +----+----+     +----+----+     | empty?  |
        |               |          +----+----+
        v               v               v
       dead?           dead?          dead?
        |               |               |
        +-------+-------+---------------+
                |
          ANY dead? --> token is invalid, mask it out
```

| Dead type                 | Detection                     | Cost                 |
|---------------------------|-------------------------------|----------------------|
| Immediate (empty set)     | `len(active) == 0`            | O(1)                 |
| Doomed (alive but hopeless)| `active & can_reach_accept == empty` | O(1) per state |
| Gana-Yati coupling        | Per-branch yati tracking      | Multiplies pairs     |

The `can_reach_accept` set is **precomputed once** via backward BFS — at runtime it's
just a set intersection.

---

## 7. Space Tracking for Word Boundaries

### Where Spaces Matter

Rule 5 from `analyzer.py:969-990`:

> If the next syllable starts with a conjunct or double consonant, the current syllable
> becomes Guru. **But this does NOT cross word boundaries.**

```
Within a word:     స + త్యము  ->  స becomes U (conjunct త్య follows)
Across a space:    స + " " + త్యము  ->  స stays I (space blocks Rule 5)
```

Same syllable, same following syllable, **different guru/laghu classification**
depending on whether there's a space.

### Updated Orchestrator FST

Add a `word_boundary` signal to the state:

```
orchestrator_state = (
    syllable_assembly: {~6 states},
    pending_syllable:  {syllable or null},
    pending_self_class: {U_by_self, I_by_self},
)
```

### Transition Logic

```
On completing a syllable S:

  1. Classify S by its own properties (Rules 1-4)
     -> self_class = U or I

  2. If there's a pending syllable P:

     If NO space since P:
       -> Check if S is conjunct/double
       -> If yes: P becomes U (Rule 5)
       -> If no: P keeps P_self_class

     If SPACE since P:
       -> P keeps P_self_class (Rule 5 blocked)

  3. Emit P's final classification to Gana NFA
  4. Buffer S as new pending syllable

On seeing SPACE:
  -> Set word_boundary flag
  -> Do NOT emit pending syllable yet
```

### Rule 5 + Spaces Transition Table

```
 Pending P    Space?    Next S is conjunct?    P's final class
-----------------------------------------------------------------
 I_by_self     no         yes                  U  (Rule 5 fires)
 I_by_self     no         no                   I  (stays laghu)
 I_by_self     yes        (irrelevant)         I  (space blocks)
 U_by_self     no         (irrelevant)         U  (already guru)
 U_by_self     yes        (irrelevant)         U  (already guru)
 null          -          -                    nothing to emit
```

Only one cell changes anything: `I_by_self + no space + conjunct follows -> U`.

### Example

```
తెలుగు స్త్రీ    vs    తెలుగుస్త్రీ
(two words)            (one compound word)

Two words:    గు -> SPACE -> I    (space blocks, స్త్రీ doesn't affect గు)
One word:     గు -> స్త్రీ(conjunct!) -> U   (Rule 5 fires)
```

That one space changes గు from I to U, shifting the entire gana pattern.

### Impact on Constrained Decoding

```python
# Considering space token
if pending_syllable.self_class == "I":
    # Space would LOCK this as I (blocking potential Rule 5)
    gana_with_I = gana_nfa.try_advance(state, "I")
    if is_dead(gana_with_I):
        # Space would kill the meter -- mask it out
        # Model MUST continue the word (emit conjunct to make it U)
        mask_out(space_token)
```

The space tracking gives the constrained decoder **control over word boundaries as a
meter tool** — the model can choose to merge words or split them to hit the right
guru/laghu pattern. This is exactly what human poets do.

### Updated State Counts

```
Before:  ~6 states x pending_class{U,I,null}           = ~18 states
After:   ~6 states x pending_class{U,I,null} x space{T,F} = ~36 states
```

---

## 8. Constrained Decoding Architecture

### Full System Diagram

```
+------------------------------------------------------------+
|                    Gemma Model                              |
|                                                             |
|  input_ids --> [Transformer] --> logits (256K tokens)       |
|                                                             |
+------------------------+-----------------------------------+
                         | raw logits
                         v
                +-----------------+
                |  Token Filter   |<---- valid_mask
                +---------+-------+
                          | masked logits
                          v
                +-----------------+
                |   Sampling      |--> next_token_id
                +---------+-------+
                          |
              +-----------+-----------+
              |    Decompose to       |
              |  unicode codepoints   |
              +-----------+-----------+
                          |
              +-----------+-----------+
              |    Orchestrator FST   |    ~36 states
              |  (syllable assembly   |
              |   + space tracking    |
              |   + position counter) |
              +--+--------+--------+-+
                 |        |        |
            U/I  | consonant  maitri
           every | at 2nd     group at
           syl   | syl only   gana 1,3
                 |        |   only
                 v        v        v
            +------+ +------+ +------+
            | Gana | |Prasa | | Yati |    ~60 + ~35 + ~28 states
            | NFA  | | NFA  | | NFA  |
            +--+---+ +--+---+ +--+---+
               |        |        |
               v        v        v
             alive?   alive?   alive?
               |        |        |
               +----+---+--------+
                    |
               ALL alive? --> valid_mask for next step
```

### Core Decode Loop

```python
for each generation step:
    logits = gemma.forward(input_ids)              # raw model output
    valid_mask = get_valid_tokens(nfa_states)       # from our 3 NFAs
    logits[~valid_mask] = -infinity                 # kill invalid tokens
    next_token = sample(logits)                     # sample from valid only
    nfa_states = advance(nfa_states, next_token)    # update NFA states
```

### Prasa Enforcement Strategy

```
Phase 1 (line 1): OPEN -- accept any consonant at 2nd syllable, store it
Phase 2 (line 2): CONSTRAINED -- only accept tokens whose 2nd syllable
                   consonant matches the stored one
```

### Yati Enforcement Strategy

```
Gana 1 processing: OPEN -- store 1st letter's maitri group
Gana 2 processing: no constraint from yati
Gana 3 processing: CONSTRAINED -- 1st letter must be in stored maitri group
```

### Selective Enforcement

```python
valid_mask = gana_mask                           # always enforce
if on_line_2:
    valid_mask &= prasa_mask                     # enforce rhyme on line 2
if strict_yati:
    valid_mask &= yati_mask                      # optionally enforce
```

### State Summary

| Component          | Type | States   | Complexity |
|--------------------|------|----------|------------|
| Orchestrator       | FST  | ~36      | Syllable assembly + space + pending |
| Gana NFA           | NFA  | ~60      | Non-deterministic gana matching |
| Prasa NFA          | NFA  | ~35      | Store-and-compare consonant class |
| Yati NFA           | NFA  | ~28      | Store-and-compare maitri group |
| **Total**          |      | **~159** | |

### Prior Art

This approach is proven in practice:
- **Outlines** (dottxt-ai/outlines) — uses FSMs for JSON/regex-constrained generation
- **guidance** (microsoft/guidance) — grammar-constrained LLM decoding
- **LMQL** — query language with constraints on LLM output

The novelty here is applying constrained decoding to **Telugu poetic meter** using
prosody-specific NFAs.

---

## Key Files Referenced

- `src/dwipada/core/constants.py` — Dwipada rules block, gana definitions
- `src/dwipada/core/analyzer.py` — Full analyzer with guru/laghu classification, prasa, yati
- `src/dwipada/training/tokenizer.py` — Gemma tokenizer integration

## Theoretical Classification

```
Dwipada (all rules including Prasa + Yati) = Regular Language

  1 FST  (orchestrator -- transducer)  +  3 NFAs (gana, prasa, yati -- acceptors)

  Finite Automaton   <-- sufficient
  Pushdown Automaton <-- unnecessary
  Turing Machine     <-- overkill
```
