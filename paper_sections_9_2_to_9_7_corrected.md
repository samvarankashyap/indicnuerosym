# Corrected Paper Sections 9.2--9.7

> Changes from the original are marked with `[CORRECTED]` annotations.
> Deletions are shown in ~~strikethrough~~. Additions are shown in **bold**.

---

## 9.2 System Overview

Figure 1 illustrates the IndicNeuroSym constrained
decoding pipeline. At each generation step, raw
logits from Gemma 3 ~~4B~~ **1B** pass through the Enforcer
before sampling.

`[CORRECTED: Model size -- verify which model size the final experiments used. The codebase targets Gemma 3.1 1B IT and Gemma 4 E4B across different scripts.]`

## 9.3 Layer 1: Orchestrator FST Pipeline

~~Three~~ **Two** chained Mealy FSTs transform raw Unicode
token strings into a classified U/I syllable stream.
**A stateless character classifier precedes them.**

`[CORRECTED: The Codepoint Classifier is a stateless function inlined into the Syllable Assembler, not a separate FST. The pipeline has two stateful transducers plus one stateless classifier.]`

**Stage 0: Codepoint Classifier (stateless).** A
stateless ~~transducer~~ **classification function** ~~classifying~~ **classifies** each Unicode
codepoint into ~~11~~ **9** categories via an O(1) ~~range
lookup table~~ **set-membership check**. Categories cover consonants
(U+0C15--U+0C39), independent vowels
(U+0C05--U+0C14), matras (U+0C3E--U+0C4C),
virama (U+0C4D), ~~anusvara (U+0C02), visarga
(U+0C03),~~ **diacritics (anusvara U+0C02, visarga U+0C03),** space, newline, **skip (ZWNJ, ZWSP, candrabindu),** and other. ~~Full category
table in Appendix B~~ **This function is inlined into the Syllable Assembler rather than implemented as a separate transducer.**

`[CORRECTED: 9 categories in the implementation, not 11. The design doc splits diacritics into anusvara+visarga and adds separate ZWNJ/AI_LENGTH categories to reach 11, but the code groups them as DIACRITIC and SKIP respectively. Also not a separate FST -- it is the classify() function at syllable_assembler.py:49.]`

Stage ~~2~~ **1**: Syllable Assembler (4 states). Groups
the category stream into complete Telugu syllables
using states IDLE, CONSONANT\_CLUSTER,
PENDING\_VIRAMA, and VOWEL. A one-slot
prev\_syllable buffer enables retroactive pollu
(syllable-final consonant) merging. The
PENDING\_VIRAMA state distinguishes conjunct
consonants from pollu marks via lookahead.

`[No changes needed -- confirmed correct.]`

Stage ~~3~~ **2**: Guru/Laghu Classifier (~~3~~ **2**
states). Classifies each syllable as U
or I using five phonological rules from
Chandhodarpanamu (Chirravuri Srirama Murthy,
1998), via states ~~IDLE, PENDING\_CLEAR, and
PENDING\_BOUNDARY~~ **EMPTY and PENDING\_I**. A one-syllable delay
buffer **in PENDING\_I** enables Rule 5 (conjunct-induced weight
promotion), which applies within words but is
blocked by intervening spaces. ~~Full transition table
in Appendix B.~~

`[CORRECTED: 2 states, not 3. The state names IDLE, PENDING_CLEAR, and PENDING_BOUNDARY do not exist in the codebase. The actual states are STATE_EMPTY (no pending syllable) and STATE_PENDING_I (one intrinsically Laghu syllable buffered, awaiting next-syllable lookahead for Rule 5). See guru_laghu_classifier.py:63-64.]`

## 9.4 Layer 2: Position Tracker and Parallel NFAs

The Position Tracker routes syllable metadata to
three NFAs based on the current position within
the couplet: line index (1 or 2), ~~gan.a index (1--4),
and syllable index within the gan.a~~ **syllable index within the line, and line completion count**.

`[CORRECTED: The Position Tracker is not a separate state-machine component. It is counter-based tracking (line_syllable_index, lines_complete, syllable_count) embedded directly in the CompositeState class. A 3-valued line-phase discriminator (LINE1/LINE2/DECIDED) routes signals to the Prasa and Yati logic.]`

**Gana NFA (~~\~60~~ **69** states).** Validates each line
against the regular pattern (nala | naga | sala |
bha | ra | ta)^3 . (na | ha) over {U, I}. The NFA
branches non-deterministically after each syllable.
**Each branch is a tuple (slot, gana\_name, sub\_pos, matched\_ganas), where slot indexes the current foot (0--2 for Indra ganas, 3 for Surya gana). Valid line lengths range from 11 to 15 syllables.**

`[CORRECTED: Exact count is 69, not ~60. Derivation: each Indra slot has 6 gana candidates with sub_pos values 0..len-1: Nala(4) + Naga(4) + Sala(4) + Bha(3) + Ra(3) + Ta(3) = 21 states per slot × 3 Indra slots = 63. Surya slot: Na(3) + Ha_Gala(2) = 5. Plus 1 ACCEPT state. Total = 63 + 5 + 1 = 69. The matched_ganas tuple is carried data for output formatting and does not affect transitions. See gana_nfa.py:88-100 for patterns, :121-132 for _spawn_slot, :142-179 for _advance.]`

**Prasa NFA (~~\~35~~ **7** states).** Phase 1 records the
~~consonant equivalence class~~ **base consonant** of the second syllable
of line 1. Phase 2 verifies that the second syllable
of line 2 ~~belongs to the same class~~ **has a matching base consonant (checked via equivalence lookup)**; a mismatch
transitions to the ~~dead~~ **REJECT** state. ~~Approximately 30
consonant classes x 2 phases yields the state count.~~
**The 7 states are: LINE1\_SYL0, LINE1\_SYL1, LINE1\_REST, LINE2\_SYL0, LINE2\_SYL1, ACCEPT, and REJECT. Consonant identity is stored as a data variable, not encoded in NFA states. Three equivalence pairs (laterals: l/L, sibilants: sh/Sh/s, rhotics: R/r) are checked via lookup.**

`[CORRECTED: The ~35 count described a hypothetical product construction (consonant classes x phases) that was never built. The actual implementation is a 7-state machine where the matched consonant is stored as a string variable, and equivalence is checked via are_prasa_equivalent(). See prasa_nfa.py:130-134.]`

**Yati NFA (~~\~28~~ **4** states).** Phase 1 records the
~~maitri group~~ **phonetic information** of the first syllable of ~~foot~~ **gana** 1. Phase 2
passes through ~~foot~~ **gana** 2 silently. Phase 3 verifies the
first syllable of ~~foot~~ **gana** 3 belongs to the same ~~group~~ **phonetic class via a 5-level cascade (exact match, vyanjana yati, svara yati, samyukta yati, bindu yati)**.
~~Eleven groups x 3 phases yields the state count.~~
**The 4 phases are: IDLE (awaiting gana 1 first syllable), RECORDED (gana 1 info stored, awaiting gana 3), ACCEPTED (yati check passed), and REJECTED (yati check failed). Phonetic group membership (across 11 Yati Maitri groups) is computed on demand rather than encoded in NFA states.**
Yati enforcement is optional; it can be disabled for
manjari variant generation.

`[CORRECTED: 4 phases, not 3 (the paper omitted the REJECTED terminal phase). The ~28 state count (~11 groups x 3 phases = 33, which itself doesn't equal 28) described a hypothetical product that was never built. Like Prasa, phonetic info is stored as data. The 5-level cascade check is a significant implementation detail worth mentioning. See yati_nfa.py:185-188.]`

## 9.5 Dead State Detection

For each NFA, ~~the co-reachability set (states from
which an accept state is reachable) is precomputed
via backward BFS. At runtime, the active state
set is intersected with this set in O(1) per
state. An empty intersection signals a doomed
configuration~~ **closed-form arithmetic bounds determine whether any active branch can reach an accept state within the valid line-length window [11, 15]. For each branch, two functions compute the minimum and maximum syllables needed to reach ACCEPT from its current position (slot, gana\_name, sub\_pos). A branch is reachable if `syllable_count + min_to_accept <= 15` and `syllable_count + max_to_accept >= 11`. An empty reachable set signals a doomed configuration**, enabling aggressive token pruning
before generating further output.

`[CORRECTED: The implementation does not use backward BFS or precomputed co-reachability sets. Instead, _is_reachable() at composite_state.py:88 uses _min_to_accept() and _max_to_accept() (lines 64-85) to compute closed-form bounds. This is O(1) per branch via arithmetic -- simpler and more efficient than the described BFS approach. The design doc (nfa_constrained_decoding_design.md:477) mentions backward BFS as a concept, but the implementation chose the arithmetic approach.]`

## 9.6 State Budget and Design Rationale

Table 8 compares the separate-NFA design against
the equivalent product automaton. Using three
independent NFAs achieves a ~~435x~~ **24x** state reduction
(~~60 x 35 x 28 = 58,800~~ **69 x 7 x 4 = 1,932** states versus ~~\~135~~ **85**),
while enabling partial scoring, selective constraint
enforcement, and independent debugging per
constraint.

**Table 8: State budget for the constrained decoding system.**

| Component | Type | States |
|---|---|---|
| ~~Codepoint Classifier~~ **Character Classifier** | ~~Mealy FST~~ **Stateless function** | ~~1~~ **0 (inlined)** |
| Syllable Assembler | Mealy FST | 4 |
| Guru/Laghu Classifier | Mealy FST | ~~3~~ **2** |
| Position Tracker | Controller | ~~\~4~~ **3 (counter-based)** |
| Gana NFA | NFA | ~~\~60~~ **69** |
| Prasa NFA | NFA | ~~\~35~~ **7** |
| Yati NFA | NFA | ~~\~28~~ **4** |
| **Total (separate NFAs)** | | ~~\~135~~ **85** |
| Product automaton (equivalent) | DFA | ~~\~58,800~~ **1,932** |

`[CORRECTED: All NFA state counts are now exact, not approximate. (1) Gana NFA: 69 states = 3 Indra slots × (Nala[4] + Naga[4] + Sala[4] + Bha[3] + Ra[3] + Ta[3] = 21) + Surya slot (Na[3] + Ha_Gala[2] = 5) + 1 ACCEPT. (2) Prasa NFA: 7 states (LINE1_SYL0, LINE1_SYL1, LINE1_REST, LINE2_SYL0, LINE2_SYL1, ACCEPT, REJECT). Consonant identity stored as data variable, not encoded in states. (3) Yati NFA: 4 states (IDLE, RECORDED, ACCEPTED, REJECTED). Phonetic group membership computed on demand, not encoded in states. (4) Guru/Laghu: 2 states (STATE_EMPTY, STATE_PENDING_I). (5) Position Tracker: 3 values (LINE1, LINE2, DECIDED) as a counter-based discriminator. Total: 0 + 4 + 2 + 3 + 69 + 7 + 4 = 89, but Codepoint Classifier and Position Tracker are not automata so excluding them: 4 + 2 + 69 + 7 + 4 = 86. Product automaton: 69 × 7 × 4 = 1,932. Note: the product concept is somewhat misleading because Prasa and Yati store constraint data in variables rather than NFA states.]`

## 9.7 Constrained Decoding Loop

**The system implements three decoding strategies:**

**Strategy 1: Rejection Sampling.** At each generation step: (1) Gemma produces
logits z_t; (2) **the top-K candidates (default K=50) are tested sequentially**; (3) **each candidate token is decoded to
Unicode and the full text is passed through a fresh FST+NFA
pipeline**; (4) tokens causing any NFA to
enter a dead or doomed state ~~receive logit -inf~~ **are rejected**; (5)
~~the surviving distribution is sampled normally~~ **the first valid candidate is accepted, preferring tokens that reach a line-completion (ACCEPT) state. If no valid candidate is found in top-K, an exhaustive vocabulary search is performed.**

**Strategy 2: Logit Masking with Backtracking.** At each generation step: (1) Gemma produces logits z_t; (2) a static mask eliminates all non-Telugu tokens (~256K vocabulary reduced to ~2,346 Telugu tokens); (3) a dynamic Gana NFA mask tests each Telugu token against a snapshot of the current CompositeState, setting invalid tokens to logit -inf; (4) the surviving distribution is sampled normally. **If the valid token count drops below a threshold (default 3) or the line overshoots 15 syllables without reaching ACCEPT, the system backtracks to a saved checkpoint. Checkpoints are stored in a ring buffer (max 15) saved every 3 tokens. On backtrack, the generation temperature is escalated by +0.15 (capped at 1.2) and decays back to the base over 6 steps.**

**Strategy 3: Hybrid Mask + Rejection.** Combines both approaches: (1) logit masking narrows the vocabulary to ~1,000 valid Telugu tokens; (2) a dynamic Gana NFA mask further prunes invalid tokens; (3) from the top-100 candidates, each is tested via CompositeState cloning; (4) candidates reaching line-completion (ACCEPT) are preferred over merely alive candidates; (5) backtracking is triggered only if no valid candidate exists in the top-100.

`[CORRECTED: The original described only a single simplified loop. The codebase implements three distinct strategies with significantly different algorithmic properties. Backtracking with temperature escalation is a major feature affecting generation quality and was entirely omitted. See constrained_generate.py (rejection), constrained_generate_masked.py (masking+backtrack), and benchmark_hybrid.py (hybrid).]`
