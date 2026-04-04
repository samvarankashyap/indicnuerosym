# -*- coding: utf-8 -*-
"""
CFG Representation Document Generator for Dwipada Pipeline.
=============================================================

Generates a comprehensive text document converting all 6 automata
(3 FSTs + 3 NFAs) in the Dwipada constrained-decoding pipeline
into their equivalent Context-Free Grammar (CFG) representations.

The script imports constants directly from the implementation modules
to ensure the CFGs stay faithful to the code. For FSTs whose transitions
are encoded in if/elif logic, transition tables are defined here matching
the implementation exactly.

Usage:
    cd nfa_for_dwipada/
    python generate_cfg_doc.py

Output:
    cfg_representations.txt
"""

import os
import sys
import textwrap
from datetime import datetime

# ---------------------------------------------------------------------------
# Path setup — ensure we can import sibling modules
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "..", "src"))

# ---------------------------------------------------------------------------
# Import constants from the actual implementation modules
# ---------------------------------------------------------------------------
from syllable_assembler import (
    TELUGU_CONSONANTS as SA_CONSONANTS,
    INDEPENDENT_VOWELS as SA_VOWELS,
    MATRAS as SA_MATRAS,
    VIRAMA as SA_VIRAMA,
    DIACRITICS as SA_DIACRITICS,
    SKIP_CHARS as SA_SKIP,
)

from guru_laghu_classifier import (
    LONG_MATRAS as GLC_LONG_MATRAS,
    INDEPENDENT_LONG_VOWELS as GLC_INDEP_LONG,
    BOUNDARIES as GLC_BOUNDARIES,
)

from ganana_marker import (
    TELUGU_CONSONANTS as GM_CONSONANTS,
    LONG_MATRAS as GM_LONG_MATRAS,
    DIPHTHONG_MATRAS as GM_DIPHTHONG_MATRAS,
    INDEPENDENT_LONG_VOWELS as GM_INDEP_LONG,
    INDEPENDENT_DIPHTHONGS as GM_INDEP_DIPHTHONGS,
    DIACRITICS as GM_DIACRITICS,
    VIRAMA as GM_VIRAMA,
)

from gana_nfa import INDRA_GANAS, SURYA_GANAS, GANA_DISPLAY_NAMES

from prasa_nfa import (
    PRASA_EQUIVALENTS,
    TELUGU_CONSONANTS as PRASA_CONSONANTS,
    CLASS_NAMES as PRASA_CLASS_NAMES,
)

from yati_nfa import (
    YATI_MAITRI_GROUPS,
    MAITRI_GROUP_NAMES,
    SVARA_YATI_GROUPS,
    SVARA_GROUP_NAMES,
    VARGA_NASALS,
    NASAL_TO_VARGA,
    TELUGU_CONSONANTS as YATI_CONSONANTS,
    INDEPENDENT_VOWELS as YATI_VOWELS,
    DEPENDENT_TO_INDEPENDENT,
)


# ===================================================================
# DOCUMENT SECTIONS
# ===================================================================

def generate_preamble():
    """Generate the preamble explaining CFG basics and the conversion rationale."""
    return textwrap.dedent("""\
    ╔══════════════════════════════════════════════════════════════════════════════╗
    ║        CONTEXT-FREE GRAMMAR (CFG) REPRESENTATIONS                          ║
    ║        Dwipada Constrained Decoding Pipeline                               ║
    ║                                                                            ║
    ║        Auto-generated from implementation source code                       ║
    ╚══════════════════════════════════════════════════════════════════════════════╝

    Generated: {date}
    Source: nfa_for_dwipada/ (3 FSTs + 3 NFAs)

    ════════════════════════════════════════════════════════════════════════════════
    SECTION 0: WHAT IS A CONTEXT-FREE GRAMMAR (CFG)?
    ════════════════════════════════════════════════════════════════════════════════

    A Context-Free Grammar (CFG) is a formal system for describing languages.
    It consists of four components:

        G = (V, Σ, R, S)

    where:

        V   = Set of NON-TERMINAL symbols (variables)
              These are placeholders that get replaced during derivation.
              Written in angle brackets: <IDLE>, <Line>, <Indra>, etc.

        Σ   = Set of TERMINAL symbols (alphabet)
              These are the actual symbols that appear in the final string.
              Examples: U, I, క, ా, SPACE, NEWLINE, etc.

        R   = Set of PRODUCTION RULES
              Each rule has the form:  <NonTerminal> → body
              The body can contain terminals, non-terminals, or both.
              Multiple alternatives are separated by |
              Example:  <Indra> → I I I I | U I I | U I U

        S   = START SYMBOL (a distinguished non-terminal)
              Derivation always begins from S.

    ── Special Symbols ──

        ε   = The empty string (epsilon). A production A → ε means the
              non-terminal A can be replaced by nothing.

        →   = "produces" or "rewrites to". Left side rewrites to right side.

        |   = "or". Separates alternative productions for the same non-terminal.


    ── WHY CAN WE CONVERT NFAs AND FSTs TO CFGs? ──

    Both NFAs (Non-deterministic Finite Automata) and FSTs (Finite State
    Transducers) recognize/define REGULAR LANGUAGES. Regular languages are a
    strict subset of context-free languages. Therefore, every NFA/FST can be
    expressed as a CFG — specifically, a RIGHT-LINEAR grammar where every
    production has one of these forms:

        A → a B     (terminal followed by one non-terminal)
        A → ε       (empty production for accepting states)

    The conversion is mechanical:
        1. Each STATE of the automaton becomes a NON-TERMINAL in the grammar
        2. The START STATE becomes the START SYMBOL
        3. Each TRANSITION δ(q, a) → r becomes the production  <q> → a <r>
        4. Each ACCEPTING STATE q gets the production  <q> → ε

    For FSTs (which produce output), we convert the INPUT LANGUAGE (the set
    of valid inputs the FST accepts). The output transformation is noted
    separately in each section.

    For NFAs with cross-referencing constraints (like Prasa and Yati, which
    store a value from one position and compare it at another), we use
    PARAMETERIZED NON-TERMINALS — one version per possible stored value.
    This "unrolls" the memory into the grammar, keeping it context-free.


    ── RIGHT-LINEAR vs GENERAL CFG ──

    A right-linear grammar is a CFG where every production is of the form:
        A → w B   or   A → w     (w is a string of terminals)

    Right-linear grammars are exactly as powerful as regular expressions and
    finite automata. They cannot express nested structures like balanced
    parentheses, but they suffice for all constraints in the Dwipada pipeline
    because Telugu metrical rules involve bounded, position-based checks —
    no unbounded nesting or recursion.

    In some sections below (especially GanaNFA), we use a more readable
    CONCATENATION form like:
        <Line> → <Indra> <Indra> <Indra> <Surya>

    This is still a regular language (finite concatenation of regular
    sub-languages), but written in the more intuitive CFG notation.


    ════════════════════════════════════════════════════════════════════════════════
    NOTATION GUIDE (used throughout this document)
    ════════════════════════════════════════════════════════════════════════════════

    Symbol          Meaning
    ──────          ───────
    <NAME>          Non-terminal (variable) — will be expanded by a rule
    lowercase/      Terminal (actual symbol in the language)
      Telugu chars
    →               "produces" / "rewrites to"
    |               "or" (alternative productions)
    ε               Empty string (epsilon)
    ·               Concatenation (when explicit clarity is needed)
    CONSONANT       Any character from the Telugu consonant set (35 chars)
    VOWEL           Any independent Telugu vowel (14 chars)
    MATRA           Any dependent vowel sign (13 chars)
    DIACRITIC       Anusvara (ం) or Visarga (ః)
    SPACE           Word boundary (single space character)
    NEWLINE         Line boundary (newline character)
    syl             Any complete Telugu syllable
    syl_U           A syllable classified as Guru (heavy)
    syl_I           A syllable classified as Laghu (light)
    syl_conj        A syllable starting with a conjunct (C+్+C pattern)
    syl_simple      A syllable NOT starting with a conjunct
    U               Guru marker (heavy syllable)
    I               Laghu marker (light syllable)
    c               Any single Telugu consonant character

    For parameterized non-terminals:
    <STATE_c>       State parameterized by stored consonant c
    <STATE_g>       State parameterized by stored group index g

    """).format(date=datetime.now().strftime("%Y-%m-%d"))


# ===================================================================
# FST 1: SYLLABLE ASSEMBLER
# ===================================================================

def generate_syllable_assembler_cfg():
    """Generate CFG for SyllableAssembler FST."""

    # Build the terminal set description from imported constants
    consonants_str = " ".join(sorted(SA_CONSONANTS))
    vowels_str = " ".join(sorted(SA_VOWELS))
    matras_str = " ".join(sorted(SA_MATRAS))

    section = textwrap.dedent("""\
    ════════════════════════════════════════════════════════════════════════════════
    SECTION 1: SYLLABLE ASSEMBLER FST → CFG
    ════════════════════════════════════════════════════════════════════════════════

    Source: syllable_assembler.py
    Purpose: Converts raw Unicode characters into complete Telugu syllables
             (aksharalu), emitting spaces and newlines as boundary markers.
    Type: Finite State Transducer (FST) — transforms character stream to
          syllable stream.

    ── ORIGINAL AUTOMATON SUMMARY ──

    States (4):
        IDLE               — No syllable buffer active
        CONSONANT_CLUSTER  — Building a consonant-based syllable
        PENDING_VIRAMA     — Buffer ends with C+్, awaiting conjunct or pollu
        VOWEL              — Building an independent-vowel syllable

    Start state:     IDLE
    Accept states:   IDLE (after flush — all buffered content emitted)

    Input alphabet (Σ):  Unicode characters classified into 9 categories:
        CONSONANT          — {consonants}
                             ({n_consonants} characters)
        INDEPENDENT_VOWEL  — {vowels}
                             ({n_vowels} characters)
        MATRA              — {matras}
                             ({n_matras} characters)
        VIRAMA             — {virama}
        DIACRITIC          — {diacritics}
        SPACE              — ' '
        NEWLINE            — '\\n'
        SKIP               — {{ZWNJ, ZWSP, candrabindu}} (transparent, ignored)
        OTHER              — anything else (ignored)

    Output: Complete syllable strings + SPACE + NEWLINE boundary markers

    ── CFG REPRESENTATION ──

    This CFG describes the set of valid Telugu character sequences that the
    SyllableAssembler FST accepts and segments into syllables. The grammar
    captures when the FST transitions between states based on character
    categories.

    G = (V, Σ, R, S)

    V = {{ <IDLE>, <CC>, <PV>, <VOW> }}

    Σ = {{ CONSONANT, VOWEL, MATRA, VIRAMA, DIACRITIC, SPACE, NEWLINE, SKIP }}
        (Each terminal represents a CHARACTER CATEGORY, not a single character.
         CONSONANT stands for any of the {n_consonants} Telugu consonants, etc.)

    S = <IDLE>

    R = {{
        ── From IDLE ──
        (1)  <IDLE>  →  CONSONANT <CC>          ;; consonant starts a syllable cluster
        (2)  <IDLE>  →  VOWEL <VOW>             ;; independent vowel starts vowel syllable
        (3)  <IDLE>  →  SPACE <IDLE>            ;; space is a boundary, stay idle
        (4)  <IDLE>  →  NEWLINE <IDLE>          ;; newline is a boundary, stay idle
        (5)  <IDLE>  →  DIACRITIC <IDLE>        ;; diacritic attaches to prev syllable
        (6)  <IDLE>  →  SKIP <IDLE>             ;; transparent characters ignored
        (7)  <IDLE>  →  ε                       ;; accept (end of input)

        ── From CONSONANT_CLUSTER ──
        (8)  <CC>    →  VIRAMA <PV>             ;; virama after consonant → pending
        (9)  <CC>    →  MATRA <IDLE>            ;; matra completes syllable, emit
        (10) <CC>    →  DIACRITIC <IDLE>        ;; diacritic completes syllable, emit
        (11) <CC>    →  CONSONANT <CC>          ;; new consonant → emit current, start new
        (12) <CC>    →  VOWEL <VOW>             ;; vowel → emit current, start vowel
        (13) <CC>    →  SPACE <IDLE>            ;; boundary → emit current, idle
        (14) <CC>    →  NEWLINE <IDLE>          ;; boundary → emit current, idle
        (15) <CC>    →  SKIP <CC>               ;; transparent, stay in cluster
        (16) <CC>    →  ε                       ;; accept (flush emits buffer)

        ── From PENDING_VIRAMA ──
        (17) <PV>    →  CONSONANT <CC>          ;; conjunct confirmed: C+్+C
        (18) <PV>    →  MATRA <IDLE>            ;; unusual: virama+matra, emit
        (19) <PV>    →  DIACRITIC <IDLE>        ;; unusual: virama+diacritic, emit
        (20) <PV>    →  SPACE <IDLE>            ;; pollu merge: C+్ merges to prev syl
        (21) <PV>    →  NEWLINE <IDLE>          ;; pollu merge at line end
        (22) <PV>    →  VOWEL <VOW>             ;; pollu merge, then start vowel
        (23) <PV>    →  SKIP <PV>               ;; transparent, stay pending
        (24) <PV>    →  ε                       ;; accept (flush: pollu merge)

        ── From VOWEL ──
        (25) <VOW>   →  DIACRITIC <IDLE>        ;; diacritic completes vowel syllable
        (26) <VOW>   →  CONSONANT <CC>          ;; emit vowel, start consonant cluster
        (27) <VOW>   →  VOWEL <VOW>             ;; emit current vowel, start new vowel
        (28) <VOW>   →  SPACE <IDLE>            ;; emit vowel, boundary
        (29) <VOW>   →  NEWLINE <IDLE>          ;; emit vowel, boundary
        (30) <VOW>   →  SKIP <VOW>              ;; transparent, stay in vowel
        (31) <VOW>   →  ε                       ;; accept (flush emits buffer)
    }}

    Total: 31 production rules

    ── OUTPUT SEMANTICS (not captured by the CFG) ──

    The CFG above describes the INPUT language — which character sequences the
    FST can process. The FST's OUTPUT (syllable segmentation) involves:
      • Buffering characters and emitting complete syllables at transitions
      • Pollu merge: when PENDING_VIRAMA sees a boundary, the C+్ merges
        backward onto the previous syllable
      • Conjunct formation: C+్+C continues building the same syllable

    These output actions are side-effects of the transducer, not expressible
    in a pure CFG.

    ── WORKED EXAMPLE ──

    Input: "నమస్కారం" (namaskārāṁ)
    Characters: న(C) మ(C) స(C) ్(V) క(C) ా(M) ర(C) ం(D)

    Derivation:
        <IDLE>
        → C <CC>                                    [rule 1: న]
        → C C <CC>                                  [rule 11: మ, emits "న"]
        → C C C <CC>                                [rule 11: స, emits "మ"]
        → C C C VIRAMA <PV>                         [rule 8: ్]
        → C C C VIRAMA C <CC>                       [rule 17: క, conjunct స్క]
        → C C C VIRAMA C MATRA <IDLE>               [rule 9: ా, emits "స్కా"]
        → C C C VIRAMA C MATRA C <CC>               [rule 1 from IDLE: ర starts new cluster]
        → C C C VIRAMA C MATRA C DIAC <IDLE>        [rule 10: ం, emits "రం"]
        → C C C VIRAMA C MATRA C DIAC ε             [rule 7: end]

    Result: ["న", "మ", "స్కా", "రం"] ✓

    """).format(
        consonants=consonants_str,
        n_consonants=len(SA_CONSONANTS),
        vowels=vowels_str,
        n_vowels=len(SA_VOWELS),
        matras=matras_str,
        n_matras=len(SA_MATRAS),
        virama=SA_VIRAMA,
        diacritics=" ".join(sorted(SA_DIACRITICS)),
    )
    return section


# ===================================================================
# FST 2: GURU/LAGHU CLASSIFIER
# ===================================================================

def generate_guru_laghu_cfg():
    """Generate CFG for GuruLaghuClassifier FST."""

    long_matras_str = " ".join(sorted(GLC_LONG_MATRAS))
    indep_long_str = " ".join(sorted(GLC_INDEP_LONG))

    section = textwrap.dedent("""\
    ════════════════════════════════════════════════════════════════════════════════
    SECTION 2: GURU/LAGHU CLASSIFIER FST → CFG
    ════════════════════════════════════════════════════════════════════════════════

    Source: guru_laghu_classifier.py
    Purpose: Classifies each syllable as Guru (U/heavy) or Laghu (I/light)
             using 5 classification rules. Emits (syllable, label) pairs.
    Type: Finite State Transducer (FST) — transforms syllable stream to
          labeled syllable stream.

    ── ORIGINAL AUTOMATON SUMMARY ──

    States (2):
        EMPTY      — Nothing buffered, ready for next syllable
        PENDING_I  — One intrinsically Laghu syllable in buffer, waiting
                     for next syllable to check Rule 5 (conjunct lookahead)

    Start state:     EMPTY
    Accept states:   EMPTY (after flush — buffered syllable emitted as Laghu)

    Input alphabet (Σ):
        syl_U       — A syllable intrinsically classified as Guru by Rules 1-4:
                       Rule 1: Contains long matra {{ {long_matras} }}
                               or is long indep. vowel {{ {indep_long} }}
                       Rule 2: Contains diphthong (ఐ, ఔ, ై, ౌ)
                       Rule 3: Contains anusvara (ం) or visarga (ః)
                       Rule 4: Ends with virama (్)
        syl_I_conj  — Intrinsically Laghu AND contains a conjunct (C+్+C)
                       ANYWHERE in the syllable. Triggers Rule 5 on preceding.
                       Example: "త్య" (conjunct at start).
        syl_I_simple— Intrinsically Laghu, no conjunct pattern anywhere.
                       Example: "మ" (bare consonant with inherent short vowel).
        syl_U_conj  — Intrinsically Guru AND contains a conjunct (C+్+C).
                       Also triggers Rule 5 on the preceding buffered syllable.
                       Example: "స్కా" (conjunct + long matra).
        syl_U_simple— Intrinsically Guru, no conjunct pattern.
                       Example: "రా" (long matra, no conjunct).
        BOUNDARY    — Space or newline (blocks Rule 5, flushes buffer)

    IMPORTANT — CONJUNCT DETECTION PREDICATE:

    This FST uses is_conjunct_trigger(syl) which checks for C+్+C pattern
    at ANY position in the syllable (not just the start). This differs from
    GanaMarker's is_conjunct_start() which only checks indices 0,1,2.

    Example of divergence: syllable "రాత్య"
      • is_conjunct_trigger("రాత్య") → True  (finds త్య at indices 2,3,4)
      • is_conjunct_start("రాత్య")  → False (indices 0,1,2 are ర,ా,త — not C+్+C)

    IMPORTANT — NEWLINE HANDLING:

    In this FST, BOTH space and newline block Rule 5 (unified as BOUNDARY).
    This differs from GanaMarker where ONLY space blocks Rule 5 and newline
    is transparent. See Section 3 for details.

    Output: (syllable_text, "U"|"I") pairs

    ── CFG REPRESENTATION ──

    This CFG describes valid sequences of classified syllables and boundaries.
    The key insight is that Rule 5 creates a 1-syllable lookahead dependency:
    a Laghu syllable might become Guru if the NEXT syllable contains a
    conjunct — and this check applies to ALL incoming syllables, including
    those that are intrinsically Guru.

    The conjunct-trigger property only matters at the PENDING_I state (where
    a buffered Laghu syllable awaits its fate). At the EMPTY state, the
    conjunct property of an incoming syllable is irrelevant — there is no
    buffered syllable to promote. Therefore, at EMPTY we use the simpler
    classification (syl_U vs syl_I_conj/syl_I_simple).

    G = (V, Σ, R, S)

    V = {{ <EMPTY>, <PENDING_I> }}

    Σ = {{ syl_U, syl_I_conj, syl_I_simple, syl_U_conj, syl_U_simple, BOUNDARY }}
        (Note: syl_U_conj and syl_U_simple are subsets of syl_U.
         The split is only needed at PENDING_I where the distinction matters.)

    S = <EMPTY>

    R = {{
        ── From EMPTY ──
        At EMPTY, no syllable is buffered, so the conjunct property of the
        incoming syllable is irrelevant (nothing to promote via Rule 5).
        (1)  <EMPTY>     →  syl_U <EMPTY>               ;; Guru: emit as U, stay EMPTY
        (2)  <EMPTY>     →  syl_I_conj <PENDING_I>      ;; Laghu: buffer it (might be promoted)
        (3)  <EMPTY>     →  syl_I_simple <PENDING_I>    ;; Laghu: buffer it
        (4)  <EMPTY>     →  BOUNDARY <EMPTY>             ;; boundary with nothing buffered
        (5)  <EMPTY>     →  ε                            ;; accept

        ── From PENDING_I (one Laghu syllable buffered) ──
        Here the conjunct property of the incoming syllable determines whether
        the BUFFERED syllable gets promoted to Guru via Rule 5.
        (6)  <PENDING_I> →  syl_U_conj <EMPTY>          ;; RULE 5: emit buffered as U (promoted!),
                                                         ;;         then emit new Guru as U
        (7)  <PENDING_I> →  syl_U_simple <EMPTY>        ;; No Rule 5: emit buffered as I,
                                                         ;;            then emit new Guru as U
        (8)  <PENDING_I> →  syl_I_conj <PENDING_I>      ;; RULE 5: emit buffered as U (promoted!),
                                                         ;;         buffer new Laghu syllable
        (9)  <PENDING_I> →  syl_I_simple <PENDING_I>    ;; No Rule 5: emit buffered as I,
                                                         ;;            buffer new Laghu
        (10) <PENDING_I> →  BOUNDARY <EMPTY>             ;; emit buffered as I (boundary blocks Rule 5)
        (11) <PENDING_I> →  ε                            ;; accept (flush: emit buffered as I)
    }}

    Total: 11 production rules

    ── RULE 5 FIRING LOGIC (from implementation) ──

    In the code (_on_syllable), is_conjunct_trigger(syl) is computed for
    EVERY incoming syllable before its intrinsic label is examined:

      • At EMPTY: is_conjunct_trigger is computed but UNUSED — _flush_buffer()
        does nothing when no syllable is buffered. The Guru/Laghu label alone
        determines the transition. (Rules 1-3)

      • At PENDING_I: is_conjunct_trigger determines Rule 5:
        - Rule 6 (syl_U_conj):  is_conjunct_trigger = True → flush_buffer(rule5=True)
          → buffered Laghu promoted to U. Then the Guru itself is emitted as U.
        - Rule 7 (syl_U_simple): is_conjunct_trigger = False → flush_buffer(rule5=False)
          → buffered Laghu stays I. Then the Guru is emitted as U.
        - Rule 8 (syl_I_conj):  is_conjunct_trigger = True → flush_buffer(rule5=True)
          → buffered Laghu promoted to U. Then new Laghu is buffered.
        - Rule 9 (syl_I_simple): is_conjunct_trigger = False → flush_buffer(rule5=False)
          → buffered Laghu stays I. Then new Laghu is buffered.

    BOUNDARY (space/newline) always flushes with rule5=False (rule 10).

    ── WORKED EXAMPLE ──

    Input syllables: ["స", "త్య", "ము"]  (from "సత్యము")
    Classification:
        "స"    → intrinsic I, no conjunct → syl_I_simple
        "త్య" → intrinsic I, has conjunct (త+్+య) → syl_I_conj
        "ము"   → intrinsic I, no conjunct → syl_I_simple

    Derivation:
        <EMPTY>
        → syl_I_simple <PENDING_I>                  [rule 3: buffer "స"]
        → syl_I_simple syl_I_conj <PENDING_I>       [rule 8: emit "స" as U (Rule 5!),
                                                              buffer "త్య"]
        → syl_I_simple syl_I_conj syl_I_simple <PENDING_I>  [rule 9: emit "త్య" as I,
                                                                       buffer "ము"]
        → syl_I_simple syl_I_conj syl_I_simple ε    [rule 11: emit "ము" as I]

    Result: [("స", U), ("త్య", I), ("ము", I)] ✓
    """).format(
        long_matras=long_matras_str,
        indep_long=indep_long_str,
    )
    return section


# ===================================================================
# FST 3: GANA MARKER
# ===================================================================

def generate_gana_marker_cfg():
    """Generate CFG for GanaMarker FST."""

    section = textwrap.dedent("""\
    ════════════════════════════════════════════════════════════════════════════════
    SECTION 3: GANA MARKER FST → CFG
    ════════════════════════════════════════════════════════════════════════════════

    Source: ganana_marker.py
    Purpose: Alternative to GuruLaghuClassifier. Takes syllable stream and
             emits U/I markers (strings, not tuples), passing spaces and
             newlines through unchanged. Same 5 classification rules.
    Type: Finite State Transducer (FST) — transforms syllable stream to
          U/I marker stream.

    ── ORIGINAL AUTOMATON SUMMARY ──

    States (3):
        IDLE              — No syllable buffered
        PENDING_CLEAR     — One syllable buffered, no boundary seen since
        PENDING_BOUNDARY  — One syllable buffered, boundary seen → Rule 5 blocked

    Start state:     IDLE
    Accept states:   IDLE (after flush)

    Input alphabet (Σ):
        syl_conj     — Any syllable (Guru or Laghu) that starts with a conjunct
                       (C+్+C pattern). Triggers Rule 5 on the preceding syllable.
                       Examples: "త్య" (Laghu+conjunct), "స్కా" (Guru+conjunct)
        syl_simple   — Any syllable that does NOT start with a conjunct.
                       Examples: "మ" (Laghu), "రా" (Guru), "సం" (Guru)
        SPACE        — Word boundary (blocks Rule 5)
        NEWLINE      — Line boundary (does NOT block Rule 5 — see note below)

    IMPORTANT — CONJUNCT DETECTION PREDICATE:

    This FST uses is_conjunct_start(syl) which checks for C+్+C pattern
    ONLY at the START of the syllable (indices 0,1,2). This differs from
    GuruLaghuClassifier's is_conjunct_trigger() which scans the ENTIRE
    syllable. See Section 2 for the divergence example.

    Like GuruLaghuClassifier, the conjunct check is applied to ALL incoming
    syllables regardless of Guru/Laghu classification. A Guru syllable
    starting with a conjunct ALSO triggers Rule 5 on the buffered syllable.

    IMPORTANT — NEWLINE HANDLING (differs from GuruLaghuClassifier!):

    In this FST, ONLY space blocks Rule 5. NEWLINE is transparent — it stays
    in PENDING_CLEAR and Rule 5 can still fire across line boundaries. This
    matches the reference analyser behavior. Compare with GuruLaghuClassifier
    (Section 2) where BOTH space and newline block Rule 5.

    Output: U, I, SPACE, NEWLINE markers

    ── CFG REPRESENTATION ──

    The GanaMarker has the same Rule 5 lookahead logic as GuruLaghuClassifier,
    but with an additional PENDING_BOUNDARY state that explicitly tracks when
    a SPACE boundary has intervened (blocking Rule 5).

    CRITICAL DESIGN DETAIL: Only SPACE transitions to PENDING_BOUNDARY.
    NEWLINE does NOT block Rule 5 — it stays in PENDING_CLEAR. This matches
    the reference analyser (akshara_ganavibhajana) which skips newlines when
    searching for the next syllable's conjunct status. A conjunct at the start
    of the next LINE can still upgrade the last syllable of the current line.

    G = (V, Σ, R, S)

    V = {{ <IDLE>, <PC>, <PB> }}
        where PC = PENDING_CLEAR, PB = PENDING_BOUNDARY

    Σ = {{ syl_conj, syl_simple, SPACE, NEWLINE }}

    S = <IDLE>

    R = {{
        ── From IDLE ──
        (1)  <IDLE>  →  syl_conj <PC>             ;; buffer conjunct-start syllable
        (2)  <IDLE>  →  syl_simple <PC>            ;; buffer non-conjunct syllable
        (3)  <IDLE>  →  SPACE <IDLE>               ;; emit space, stay idle
        (4)  <IDLE>  →  NEWLINE <IDLE>             ;; emit newline, stay idle
        (5)  <IDLE>  →  ε                          ;; accept

        ── From PENDING_CLEAR (syllable buffered, no SPACE boundary since) ──
        (6)  <PC>    →  syl_conj <PC>              ;; RULE 5 FIRES: emit buffered as U,
                                                    ;;   emit any buffered newlines,
                                                    ;;   buffer new syllable
        (7)  <PC>    →  syl_simple <PC>            ;; No Rule 5: emit buffered as self_class,
                                                    ;;   emit any buffered newlines,
                                                    ;;   buffer new syllable
        (8)  <PC>    →  SPACE <PB>                 ;; buffer space, block Rule 5
        (9)  <PC>    →  NEWLINE <PC>               ;; buffer newline, Rule 5 STILL POSSIBLE
                                                    ;; (stays PENDING_CLEAR, NOT PENDING_BOUNDARY)
        (10) <PC>    →  ε                          ;; accept (flush: emit self_class of buffered)

        ── From PENDING_BOUNDARY (syllable buffered + SPACE seen) ──
        (11) <PB>    →  syl_conj <PC>              ;; Rule 5 BLOCKED by space!
                                                    ;; emit self_class of buffered,
                                                    ;; emit all buffered boundaries, buffer new
        (12) <PB>    →  syl_simple <PC>            ;; same: emit buffered+boundaries, buffer new
        (13) <PB>    →  SPACE <PB>                 ;; accumulate another boundary
        (14) <PB>    →  NEWLINE <PB>               ;; accumulate another boundary
        (15) <PB>    →  ε                          ;; accept (flush all)
    }}

    Total: 15 production rules

    ── KEY DIFFERENCES FROM GURU/LAGHU CLASSIFIER ──

    1. NEWLINE handling (rule 9 vs rule 8):
       • Rule 9: NEWLINE at PENDING_CLEAR → stays PENDING_CLEAR (not PB!)
       • Rule 8: SPACE at PENDING_CLEAR → transitions to PENDING_BOUNDARY
       • Only SPACE blocks Rule 5. NEWLINE is transparent to Rule 5.

    2. Rule 5 trigger (rules 6 & 11):
       • Rule 6 [PENDING_CLEAR + conjunct-start]: Rule 5 fires → buffered becomes U
       • Rule 11 [PENDING_BOUNDARY + conjunct-start]: Rule 5 is BLOCKED by the
         intervening space → buffered stays at its intrinsic class

    This three-state design makes the boundary suppression of Rule 5 explicit,
    which is important for the streaming constrained-decoding use case where
    syllables arrive one at a time.

    ── WORKED EXAMPLE ──

    Input syllables: ["మ", " ", "ళ్ళ"]  (word boundary between మ and ళ్ళ)
    Classification:
        "మ"    → syl_simple (Laghu, no conjunct start)
        " "    → SPACE
        "ళ్ళ" → syl_conj (starts with ళ+్+ళ conjunct)

    Derivation:
        <IDLE>
        → syl_simple <PC>                           [rule 2: buffer "మ"]
        → syl_simple SPACE <PB>                     [rule 8: space blocks Rule 5]
        → syl_simple SPACE syl_conj <PC>            [rule 11: Rule 5 BLOCKED!
                                                              emit "మ" as I (self_class),
                                                              emit " ",
                                                              buffer "ళ్ళ"]
        → syl_simple SPACE syl_conj ε               [rule 10: emit "ళ్ళ" as I]

    Result: I, " ", I  (మ stays I because space blocked Rule 5) ✓

    """)
    return section


# ===================================================================
# NFA 1: GANA NFA
# ===================================================================

def generate_gana_nfa_cfg():
    """Generate CFG for GanaNFA, directly from INDRA_GANAS and SURYA_GANAS."""

    lines = []
    lines.append(textwrap.dedent("""\
    ════════════════════════════════════════════════════════════════════════════════
    SECTION 4: GANA NFA → CFG
    ════════════════════════════════════════════════════════════════════════════════

    Source: gana_nfa.py
    Purpose: Partitions each Dwipada line into exactly 3 Indra ganas + 1 Surya
             gana. Validates that the U/I marker stream matches a valid metrical
             pattern.
    Type: Non-deterministic Finite Automaton (NFA) — multiple gana assignments
          are explored in parallel; dead branches are pruned.

    ── ORIGINAL AUTOMATON SUMMARY ──

    The NFA processes U/I markers and non-deterministically guesses which gana
    pattern is being read at each position. The state is a SET of active
    branches, where each branch is:

        (slot, gana_name, sub_position, matched_ganas_so_far)

        slot:     0, 1, 2 (Indra positions) or 3 (Surya position)
        sub_pos:  position within the current gana pattern

    Start: All 6 Indra gana branches spawned for slot 0
    Accept: Any branch reaches (ACCEPT, _, _, (g1, g2, g3, g4))

    Input alphabet (Σ): {{ U, I, NEWLINE }}

    ── GANA PATTERNS (from source code) ──
    """))

    # Indra ganas
    lines.append("    Indra Ganas (ఇంద్ర గణములు) — used in slots 0, 1, 2:\n")
    for name, pattern in sorted(INDRA_GANAS.items()):
        display = GANA_DISPLAY_NAMES.get(name, name)
        pattern_str = " ".join(pattern)
        lines.append(f"        {display:<20s}  {pattern_str:<12s}  ({len(pattern)} syllables)\n")

    lines.append("\n")

    # Surya ganas
    lines.append("    Surya Ganas (సూర్య గణములు) — used in slot 3:\n")
    for name, pattern in sorted(SURYA_GANAS.items()):
        display = GANA_DISPLAY_NAMES.get(name, name)
        pattern_str = " ".join(pattern)
        lines.append(f"        {display:<20s}  {pattern_str:<12s}  ({len(pattern)} syllables)\n")

    lines.append("\n")

    # Formal language
    indra_alternatives = []
    for name, pattern in sorted(INDRA_GANAS.items()):
        indra_alternatives.append(" ".join(pattern))
    indra_str = " | ".join(indra_alternatives)

    surya_alternatives = []
    for name, pattern in sorted(SURYA_GANAS.items()):
        surya_alternatives.append(" ".join(pattern))
    surya_str = " | ".join(surya_alternatives)

    # Calculate valid line lengths
    indra_lens = sorted(set(len(p) for p in INDRA_GANAS.values()))
    surya_lens = sorted(set(len(p) for p in SURYA_GANAS.values()))
    min_line = min(indra_lens) * 3 + min(surya_lens)
    max_line = max(indra_lens) * 3 + max(surya_lens)

    lines.append(textwrap.dedent("""\
    ── FORMAL LANGUAGE ──

    Each valid Dwipada line belongs to:

        L_line = (Indra)³ · (Surya)

    where Indra and Surya are defined by the gana patterns above.
    Valid line length: {min_line} to {max_line} syllables.

    ── CFG REPRESENTATION ──

    This is the most elegant conversion. Because the GanaNFA partitions a line
    into a fixed sequence of 4 slots (3 Indra + 1 Surya), the CFG is simply
    a concatenation of sub-grammars — one for each slot.

    G = (V, Σ, R, S)

    V = {{ <Dwipada>, <Line>, <Indra>, <Surya> }}

    Σ = {{ U, I, NEWLINE }}

    S = <Dwipada>

    R = {{
        ── Top-level: a Dwipada is 1 or 2 lines ──
        (1)  <Dwipada>  →  <Line>
        (2)  <Dwipada>  →  <Line> NEWLINE <Line>

        ── Each line is 3 Indra ganas + 1 Surya gana ──
        (3)  <Line>     →  <Indra> <Indra> <Indra> <Surya>

        ── Indra gana alternatives (from INDRA_GANAS) ──
    """).format(min_line=min_line, max_line=max_line))

    rule_num = 4
    for name, pattern in sorted(INDRA_GANAS.items()):
        display = GANA_DISPLAY_NAMES.get(name, name)
        pattern_str = " ".join(pattern)
        lines.append(f"        ({rule_num})  <Indra>   →  {pattern_str:<12s}  ;; {display}\n")
        rule_num += 1

    lines.append("\n        ── Surya gana alternatives (from SURYA_GANAS) ──\n")
    for name, pattern in sorted(SURYA_GANAS.items()):
        display = GANA_DISPLAY_NAMES.get(name, name)
        pattern_str = " ".join(pattern)
        lines.append(f"        ({rule_num})  <Surya>   →  {pattern_str:<12s}  ;; {display}\n")
        rule_num += 1

    lines.append(f"    }}\n\n    Total: {rule_num - 1} production rules\n\n")

    # Compact form
    lines.append(textwrap.dedent("""\
    ── COMPACT FORM ──

    <Dwipada>  →  <Line> | <Line> NEWLINE <Line>
    <Line>     →  <Indra> <Indra> <Indra> <Surya>
    <Indra>    →  {indra}
    <Surya>    →  {surya}

    ── NOTE ON NON-DETERMINISM ──

    In the original NFA, the non-determinism arises because after seeing the
    first few symbols, multiple gana types remain possible. For example, after
    seeing "I", the NFA has branches for Nala(IIII), Naga(IIIU), and Sala(IIUI).
    The CFG captures this naturally through the alternative productions for
    <Indra> — the parser (or derivation) simply picks the correct alternative
    that matches the full input.

    ── WORKED EXAMPLE ──

    Input: I I I U  I I I I  U I U  U I

    Derivation:
        <Dwipada>
        → <Line>                                              [rule 1]
        → <Indra> <Indra> <Indra> <Surya>                    [rule 3]
        → I I I U  <Indra> <Indra> <Surya>                   [Naga]
        → I I I U  I I I I  <Indra> <Surya>                  [Nala]
        → I I I U  I I I I  U I U  <Surya>                   [Ra]
        → I I I U  I I I I  U I U  U I                       [Ha/Gala]

    Partition: Naga(నగ) | Nala(నల) | Ra(ర) | Ha/Gala(హ/గల) ✓

    """).format(indra=indra_str, surya=surya_str))

    return "".join(lines)


# ===================================================================
# NFA 2: PRASA NFA
# ===================================================================

def generate_prasa_nfa_cfg():
    """Generate CFG for PrasaNFA with parameterized non-terminals."""

    lines = []

    # Build equivalence group descriptions
    equiv_lines = []
    for group in PRASA_EQUIVALENTS:
        members = sorted(group)
        name = PRASA_CLASS_NAMES.get(group, "unknown")
        equiv_lines.append(f"            {' ↔ '.join(members)}  ({name})")
    equiv_str = "\n".join(equiv_lines)

    # Get all consonants sorted
    all_consonants = sorted(PRASA_CONSONANTS)

    # Build equivalence class mapping
    # Group consonants into: those in equivalence groups + singletons
    equiv_classes = {}
    for c in all_consonants:
        found = False
        for group in PRASA_EQUIVALENTS:
            if c in group:
                class_name = PRASA_CLASS_NAMES[group]
                if class_name not in equiv_classes:
                    equiv_classes[class_name] = sorted(group)
                found = True
                break
        if not found:
            equiv_classes[c] = [c]

    lines.append(textwrap.dedent("""\
    ════════════════════════════════════════════════════════════════════════════════
    SECTION 5: PRASA NFA → CFG
    ════════════════════════════════════════════════════════════════════════════════

    Source: prasa_nfa.py
    Purpose: Validates the prasa (ప్రాస) rhyme constraint — the 2nd syllable
             of line 1 and line 2 must share the same base consonant.
    Type: Deterministic Finite Automaton (named NFA for pipeline consistency)

    ── ORIGINAL AUTOMATON SUMMARY ──

    States (7):
        LINE1_SYL0  — Waiting for 1st syllable of line 1
        LINE1_SYL1  — Waiting for 2nd syllable of line 1 (prasa position)
        LINE1_REST  — Consuming remaining syllables of line 1
        LINE2_SYL0  — Waiting for 1st syllable of line 2
        LINE2_SYL1  — Waiting for 2nd syllable of line 2 (check position)
        ACCEPT      — Prasa matched
        REJECT      — Prasa did not match

    Start state:     LINE1_SYL0
    Accept states:   {{ ACCEPT }}

    Input alphabet (Σ):
        syl   — Any Telugu syllable
        SPACE — Word boundary (skipped, not counted as syllable)
        NEWLINE — Line separator

    Prasa equivalence groups (from PRASA_EQUIVALENTS):
    {equiv}

    All other consonants must match exactly.

    ── THE CHALLENGE: CROSS-REFERENCING CONSTRAINT ──

    The Prasa NFA STORES the consonant from line 1's 2nd syllable and COMPARES
    it against line 2's 2nd syllable. A pure CFG cannot "remember" a value and
    compare it later — that requires context-sensitivity.

    SOLUTION: We use PARAMETERIZED NON-TERMINALS. Instead of one <LINE1_REST>
    state, we create one version for EACH possible stored consonant:
        <LINE1_REST_క>, <LINE1_REST_ఖ>, <LINE1_REST_గ>, ...

    This "unrolls" the finite memory into the grammar. Since the set of
    possible consonants is finite ({n_consonants} consonants collapsed into
    {n_classes} equivalence classes), this remains a finite (though larger) CFG.

    """).format(
        equiv=equiv_str,
        n_consonants=len(all_consonants),
        n_classes=len(equiv_classes),
    ))

    # Generate the parameterized CFG
    lines.append("    ── CFG REPRESENTATION ──\n\n")
    lines.append("    G = (V, Σ, R, S)\n\n")

    # Non-terminals
    lines.append("    V = { <L1S0>, <L1S1>,\n")
    lines.append("          <L1R_[class]> for each consonant class,\n")
    lines.append("          <L2S0_[class]> for each consonant class,\n")
    lines.append("          <L2S1_[class]> for each consonant class,\n")
    lines.append("          <ACCEPT> }\n\n")

    lines.append("    Consonant classes (each [] is one of these):\n")
    for class_name, members in sorted(equiv_classes.items()):
        if len(members) > 1:
            lines.append(f"        [{class_name}] = {{ {', '.join(members)} }}  (equivalent for prasa)\n")
        else:
            lines.append(f"        [{members[0]}] = {{ {members[0]} }}  (singleton)\n")

    lines.append(f"\n    Total classes: {len(equiv_classes)}\n\n")

    lines.append("    Σ = { syl, syl_with_base_c (for each consonant c), SPACE, NEWLINE }\n\n")
    lines.append("    S = <L1S0>\n\n")

    lines.append("    R = {\n")
    lines.append("        ── Phase 1: Line 1, counting to 2nd syllable ──\n")
    lines.append("        (1)  <L1S0>        →  syl <L1S1>             ;; 1st syl of line 1\n")
    lines.append("        (2)  <L1S0>        →  SPACE <L1S0>           ;; skip spaces\n\n")

    lines.append("        ── Phase 2: Store consonant from 2nd syllable ──\n")
    lines.append("             For each consonant class [c]:\n")
    lines.append("        (3c) <L1S1>        →  syl_with_base_[c] <L1R_[c]>  ;; store class\n")
    lines.append("        (3') <L1S1>        →  SPACE <L1S1>           ;; skip spaces\n\n")

    lines.append("        ── Phase 3: Consume rest of line 1 ──\n")
    lines.append("             For each consonant class [c]:\n")
    lines.append("        (4c) <L1R_[c]>     →  syl <L1R_[c]>         ;; consume syllable\n")
    lines.append("        (5c) <L1R_[c]>     →  SPACE <L1R_[c]>       ;; skip space\n")
    lines.append("        (6c) <L1R_[c]>     →  NEWLINE <L2S0_[c]>    ;; line break → line 2\n\n")

    lines.append("        ── Phase 4: Line 2, counting to 2nd syllable ──\n")
    lines.append("             For each consonant class [c]:\n")
    lines.append("        (7c) <L2S0_[c]>    →  syl <L2S1_[c]>        ;; 1st syl of line 2\n")
    lines.append("        (8c) <L2S0_[c]>    →  SPACE <L2S0_[c]>      ;; skip spaces\n\n")

    lines.append("        ── Phase 5: Check consonant of 2nd syllable ──\n")
    lines.append("             For each consonant class [c]:\n")
    lines.append("        (9c) <L2S1_[c]>    →  syl_with_base_[c] <ACCEPT>   ;; MATCH! ✓\n")
    lines.append("             (Any syl with base NOT in class [c] → no production → REJECT)\n")
    lines.append("        (9') <L2S1_[c]>    →  SPACE <L2S1_[c]>      ;; skip spaces\n\n")

    lines.append("        ── Phase 6: Accept (consume remaining input) ──\n")
    lines.append("        (10) <ACCEPT>       →  syl <ACCEPT>          ;; consume rest\n")
    lines.append("        (11) <ACCEPT>       →  SPACE <ACCEPT>        ;; consume rest\n")
    lines.append("        (12) <ACCEPT>       →  NEWLINE <ACCEPT>      ;; consume rest\n")
    lines.append("        (13) <ACCEPT>       →  ε                     ;; done\n")
    lines.append("    }\n\n")

    # Count rules
    n_classes = len(equiv_classes)
    # Fixed rules: 1,2,3' = 3, per-class: 3c,4c,5c,6c,7c,8c,9c,9' = 8 each, accept: 4
    total = 3 + (8 * n_classes) + 4
    lines.append(f"    Total production rules: 3 + (8 × {n_classes} classes) + 4 = {total}\n\n")

    lines.append(textwrap.dedent("""\
    ── EQUIVALENCE CLASS DETAIL ──

    The parameterization uses equivalence classes, NOT individual consonants.
    Consonants in the same Prasa equivalence group share a single non-terminal:

"""))

    for class_name, members in sorted(equiv_classes.items()):
        if len(members) > 1:
            lines.append(f"        <L1R_{class_name}> accepts base consonant ∈ {{ {', '.join(members)} }}\n")

    lines.append(textwrap.dedent("""
    All other consonants are their own class (singleton).

    ── WORKED EXAMPLE ──

    Input: "సౌ ధా గ్ర ము ల NEWLINE వీ ధు ల"
    (Line 1 prasa position = "ధా", Line 2 prasa position = "ధు")

    Base consonant of "ధా" = ధ
    Base consonant of "ధు" = ధ
    ధ is a singleton class (not in any equivalence group)

    Derivation:
        <L1S0>
        → syl <L1S1>                           [rule 1: "సౌ"]
        → syl syl_with_base_ధ <L1R_ధ>          [rule 3ధ: "ధా", store ధ]
        → syl syl syl <L1R_ధ>                  [rule 4ధ×3: "గ్ర","ము","ల"]
        → ... NEWLINE <L2S0_ధ>                  [rule 6ధ: line break]
        → ... syl <L2S1_ధ>                     [rule 7ధ: "వీ"]
        → ... syl_with_base_ధ <ACCEPT>          [rule 9ధ: "ధు" matches! ✓]
        → ... ε                                 [rule 13: done]

    Result: ACCEPT (exact match on ధ) ✓

    """))

    return "".join(lines)


# ===================================================================
# NFA 3: YATI NFA
# ===================================================================

def generate_yati_nfa_cfg():
    """Generate CFG for YatiNFA with parameterized non-terminals."""

    lines = []

    lines.append(textwrap.dedent("""\
    ════════════════════════════════════════════════════════════════════════════════
    SECTION 6: YATI NFA → CFG
    ════════════════════════════════════════════════════════════════════════════════

    Source: yati_nfa.py
    Purpose: Validates the Yati (యతి) alliteration constraint — the 1st syllable
             of gana 1 must phonetically match the 1st syllable of gana 3 within
             each line, under Yati Maitri group equivalence.
    Type: NFA with cascading match checks (5 yati types in priority order).

    ── ORIGINAL AUTOMATON SUMMARY ──

    Phases (per line):
        IDLE       — Waiting for gana 1's first syllable
        RECORDED   — Gana 1's info stored, waiting for gana 3's first syllable
        ACCEPTED   — Yati check passed
        REJECTED   — Yati check failed

    The NFA receives PAIRS: (gana1_first_syllable, gana3_first_syllable)
    per line. The Position Tracker in the pipeline extracts these from the
    syllable stream based on the GanaNFA's partition.

    Input: Pairs of aksharalu (syllables) — one pair per line.

    ── YATI MAITRI GROUPS (from source code) ──

    11 phonetic equivalence classes. Letters in the same group satisfy
    Vyanjana Yati (వ్యంజన యతి):

    """))

    for idx, (group, name) in enumerate(zip(YATI_MAITRI_GROUPS, MAITRI_GROUP_NAMES)):
        members = sorted(group)
        lines.append(f"        Group {idx:2d}: {name:<35s} {{ {', '.join(members)} }}\n")

    lines.append(textwrap.dedent("""
    ── SVARA YATI GROUPS (vowel family harmony) ──

    """))

    for idx, (group, name) in enumerate(zip(SVARA_YATI_GROUPS, SVARA_GROUP_NAMES)):
        members = sorted(group)
        lines.append(f"        Group {idx}: {name:<25s} {{ {', '.join(members)} }}\n")

    lines.append(textwrap.dedent("""
    ── BINDU YATI MAPPING (anusvara → varga nasal) ──

    """))
    for consonant, nasal in sorted(VARGA_NASALS.items()):
        lines.append(f"        {consonant}ం → {nasal}\n")

    lines.append(textwrap.dedent("""
    ── CASCADE CHECK ORDER ──

    The 5 yati checks are tried in priority order. First match wins:

        Priority 1: EXACT MATCH       — Same first letter (e.g., క ↔ క)
        Priority 2: VYANJANA YATI     — Same Maitri group (e.g., క ↔ గ, both Velars)
        Priority 3: SVARA YATI        — Same vowel family (e.g., కా ↔ అ, both అ-family)
        Priority 4: SAMYUKTA YATI     — Any consonant in a conjunct matches via maitri
                                        (e.g., ప్ర ↔ ర, because ర is in the conjunct)
        Priority 5: BINDU YATI        — Anusvara maps to varga nasal
                                        (e.g., కం ↔ ఙ, because కం → velar nasal ఙ)

    ── THE CHALLENGE: MULTI-DIMENSIONAL MATCHING ──

    Unlike Prasa (which compares one dimension — base consonant), Yati has
    FIVE different matching criteria across multiple phonetic features. A full
    CFG unrolling would need parameterized non-terminals for every combination
    of features that could be stored from gana 1.

    However, since the NFA receives pre-extracted PAIRS (not a raw stream),
    the CFG is simpler: it describes which pairs are accepted.

    ── CFG REPRESENTATION ──

    Since the YatiNFA receives (gana1_syl, gana3_syl) pairs per line, the
    CFG defines the LANGUAGE OF VALID PAIRS.

    G = (V, Σ, R, S)

    V = { <Yati>, <LinePair>, <ValidPair> }

    Σ = { (a, b) for all aksharalu a, b — represented as pair terminals }

    S = <Yati>

    R = {
        ── Top level ──
        (1)  <Yati>       →  <LinePair>
        (2)  <Yati>       →  <LinePair> <LinePair>

        ── Each line has one pair to validate ──
        (3)  <LinePair>   →  <ValidPair>

        ── A pair is valid if ANY of the 5 yati checks pass ──
        (4)  <ValidPair>  →  <ExactMatch>
        (5)  <ValidPair>  →  <VyanjanaMaitri>
        (6)  <ValidPair>  →  <SvaraYati>
        (7)  <ValidPair>  →  <SamyuktaYati>
        (8)  <ValidPair>  →  <BinduYati>

    """))

    # Generate exact match rules
    lines.append("        ── Exact Match: same first letter ──\n")
    all_letters = sorted(YATI_CONSONANTS | YATI_VOWELS)
    lines.append(f"        (9)  <ExactMatch>  →  (x, x)  for any x ∈ Σ_letters\n")
    lines.append(f"             where Σ_letters = all {len(all_letters)} Telugu consonants + vowels\n\n")

    # Generate Vyanjana Maitri rules
    lines.append("        ── Vyanjana Maitri: same Yati Maitri group ──\n")
    rule_num = 10
    for idx, (group, name) in enumerate(zip(YATI_MAITRI_GROUPS, MAITRI_GROUP_NAMES)):
        members = sorted(group)
        lines.append(f"        ({rule_num})  <VyanjanaMaitri>  →  (a, b)  where a, b ∈ Group {idx} ({name})\n")
        lines.append(f"              Group {idx} = {{ {', '.join(members)} }}\n")
        rule_num += 1

    lines.append("\n")

    # Generate Svara Yati rules
    lines.append("        ── Svara Yati: same vowel family ──\n")
    for idx, (group, name) in enumerate(zip(SVARA_YATI_GROUPS, SVARA_GROUP_NAMES)):
        members = sorted(group)
        lines.append(f"        ({rule_num})  <SvaraYati>  →  (a, b)  where vowel(a), vowel(b) ∈ {name}\n")
        lines.append(f"              Vowels: {{ {', '.join(members)} }}\n")
        rule_num += 1

    lines.append("\n")

    # Samyukta Yati
    lines.append("        ── Samyukta Yati: conjunct consonant matching ──\n")
    lines.append(f"        ({rule_num})  <SamyuktaYati>  →  (a, b)  where there EXISTS some\n")
    lines.append(f"              consonant c1 in a's conjunct cluster and some consonant c2\n")
    lines.append(f"              in b's conjunct cluster such that c1 and c2 are in the same\n")
    lines.append(f"              Maitri group (or c1 == c2).\n")
    lines.append(f"              (Existential match: the code returns True on the FIRST\n")
    lines.append(f"              matching pair found. E.g., ప్ర ↔ ర matches because ర\n")
    lines.append(f"              appears in the conjunct ప్ర and equals ర directly.)\n")
    rule_num += 1
    lines.append("\n")

    # Bindu Yati
    lines.append("        ── Bindu Yati: anusvara → varga nasal mapping ──\n")
    lines.append(f"        ({rule_num})  <BinduYati>  →  (a, b)  where:\n")
    lines.append(f"              a contains anusvara (ం) AND base consonant of a maps\n")
    lines.append(f"              to a varga nasal that matches b's consonant, or vice versa.\n")
    lines.append(f"              Varga nasal mapping:\n")
    for consonant, nasal in sorted(VARGA_NASALS.items()):
        lines.append(f"                {consonant}ం → {nasal}  (matches {nasal} or any in its varga)\n")
    rule_num += 1

    lines.append(f"\n    }}\n\n    Total: {rule_num - 1} production rule groups\n\n")

    lines.append(textwrap.dedent("""\
    ── NOTE ON EXPRESSIVENESS ──

    The Yati CFG is best understood as a CONSTRAINT GRAMMAR rather than a
    generative grammar. Each <ValidPair> production defines a PREDICATE on
    the pair (gana1_syl, gana3_syl). The cascade priority (exact > vyanjana >
    svara > samyukta > bindu) determines which match_type is reported, but
    for acceptance/rejection, any single match suffices.

    Because the set of Telugu aksharalu is finite (bounded by the Unicode
    block), the complete expansion of all valid pairs is finite, making
    this a regular (and therefore context-free) language.

    ── WORKED EXAMPLE ──

    Input pairs: [(("కా", "గు")), (("సా", "సి"))]

    Line 1: gana1_syl = "కా", gana3_syl = "గు"
      • First letters: క, గ
      • Exact match? క ≠ గ → No
      • Vyanjana Maitri? క ∈ Group 3 (Velars), గ ∈ Group 3 (Velars) → Yes! ✓
      • Result: ACCEPTED (vyanjana_maitri)

    Line 2: gana1_syl = "సా", gana3_syl = "సి"
      • First letters: స, స
      • Exact match? స = స → Yes! ✓
      • Result: ACCEPTED (exact)

    Both lines pass → Yati constraint satisfied ✓

    """))

    return "".join(lines)


# ===================================================================
# SUMMARY TABLE
# ===================================================================

def generate_summary_table():
    """Generate the summary table at the end."""

    n_indra = len(INDRA_GANAS)
    n_surya = len(SURYA_GANAS)
    n_prasa_classes = len(set(
        PRASA_CLASS_NAMES.get(g, sorted(g)[0])
        for g in PRASA_EQUIVALENTS
    )) + (len(PRASA_CONSONANTS) - sum(len(g) for g in PRASA_EQUIVALENTS))
    n_maitri = len(YATI_MAITRI_GROUPS)

    return textwrap.dedent("""\
    ════════════════════════════════════════════════════════════════════════════════
    SUMMARY TABLE
    ════════════════════════════════════════════════════════════════════════════════

    ┌─────────────────────────┬──────────┬────────┬────────────────┬──────────────┐
    │ Component               │ Type     │ States │ CFG Rules      │ CFG Type     │
    ├─────────────────────────┼──────────┼────────┼────────────────┼──────────────┤
    │ SyllableAssembler       │ FST      │ 4      │ 31             │ Right-linear │
    │ GuruLaghuClassifier     │ FST      │ 2      │ 11             │ Right-linear │
    │ GanaMarker              │ FST      │ 3      │ 15             │ Right-linear │
    │ GanaNFA                 │ NFA      │ ~60    │ {gana_rules:<14s}│ Concatenation│
    │ PrasaNFA                │ DFA/NFA  │ 7      │ ~{prasa_rules:<13s}│ Parameterized│
    │ YatiNFA                 │ NFA      │ 4      │ ~{yati_rules:<13s}│ Constraint   │
    └─────────────────────────┴──────────┴────────┴────────────────┴──────────────┘

    Key:
      Right-linear    — Standard NFA→grammar conversion (A → a B | ε)
      Concatenation   — Regular sub-grammars composed by concatenation
      Parameterized   — Non-terminals parameterized by stored values
      Constraint      — Predicate-based: defines valid input pairs

    All grammars are CONTEXT-FREE and describe REGULAR LANGUAGES.
    The Dwipada metrical system requires no context-sensitive or
    Turing-complete power — finite automata suffice for all constraints.

    ════════════════════════════════════════════════════════════════════════════════
    END OF DOCUMENT
    ════════════════════════════════════════════════════════════════════════════════
    """).format(
        gana_rules=str(3 + n_indra + n_surya),
        prasa_rules=str(3 + 8 * n_prasa_classes + 4),
        yati_rules=str(8 + 1 + n_maitri + len(SVARA_YATI_GROUPS) + 1 + 1),  # 8 top + 1 exact + 11 vyanjana + 3 svara + 1 samyukta + 1 bindu
    )


# ===================================================================
# MAIN: ASSEMBLE AND WRITE
# ===================================================================

def main():
    """Generate the complete CFG representations document."""

    sections = [
        generate_preamble(),
        generate_syllable_assembler_cfg(),
        generate_guru_laghu_cfg(),
        generate_gana_marker_cfg(),
        generate_gana_nfa_cfg(),
        generate_prasa_nfa_cfg(),
        generate_yati_nfa_cfg(),
        generate_summary_table(),
    ]

    document = "\n".join(sections)

    output_path = os.path.join(_HERE, "cfg_representations.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(document)

    print(f"Generated: {output_path}")
    print(f"Size: {len(document):,} characters, {document.count(chr(10)):,} lines")

    # Quick stats
    print(f"\nData sourced from implementation:")
    print(f"  INDRA_GANAS:        {len(INDRA_GANAS)} patterns")
    print(f"  SURYA_GANAS:        {len(SURYA_GANAS)} patterns")
    print(f"  PRASA_EQUIVALENTS:  {len(PRASA_EQUIVALENTS)} groups")
    print(f"  YATI_MAITRI_GROUPS: {len(YATI_MAITRI_GROUPS)} groups")
    print(f"  SVARA_YATI_GROUPS:  {len(SVARA_YATI_GROUPS)} groups")
    print(f"  VARGA_NASALS:       {len(VARGA_NASALS)} mappings")


if __name__ == "__main__":
    main()
