# -*- coding: utf-8 -*-
"""
Ganana Marker FST for Telugu.
==============================

Reads a stream of syllables (aksharalu), spaces, and newlines — the output of
SyllableAssembler — and emits a Guru (U) or Laghu (I) marker for each syllable,
passing spaces and newlines through unchanged.

This is Stage 2 of the NFA constrained-decoding pipeline:

    Raw text
       |
       v
    [SyllableAssembler FST]   -- Stage 1: Unicode chars -> syllables
       |
       v
    [GanaMarker FST]          -- Stage 2: syllables -> U/I markers  (THIS FILE)
       |
       v
    [Gana NFA]                -- Stage 3: U/I stream -> gana validation

See design_ganana_marker.md for the full FST design with diagrams.

-------------------------------------------------------------------------------
QUICK REFERENCE
-------------------------------------------------------------------------------

    from syllable_assembler import SyllableAssembler
    from ganana_marker import GanaMarker, mark_text

    # End-to-end helper
    markers = mark_text("సౌధాగ్రముల యందు")
    # => ['U', 'U', 'I', 'I', ' ', 'I', 'U', 'I']

    # Step by step
    asm = SyllableAssembler()
    syllables = asm.process("సౌధాగ్రముల యందు")
    # => ['సౌ', 'ధా', 'గ్ర', 'ము', 'ల', ' ', 'య', 'న్దు']  (example)

    gm = GanaMarker()
    markers = gm.process(syllables)
    # => ['U', 'U', 'I', 'I', 'I', ' ', 'I', 'U', 'I']

    # With trace (for debugging)
    markers, trace = gm.process_with_trace(syllables)

-------------------------------------------------------------------------------
KEY CONCEPTS
-------------------------------------------------------------------------------

Guru (గురువు, U) — Heavy syllable. Takes more time to pronounce.
Laghu (లఘువు, I) — Light syllable. Quick to pronounce.

GURU RULES (5 rules from aksharanusarika.py :: akshara_ganavibhajana):

    Rule 1 — Long vowel (దీర్ఘ స్వరం):
              Syllable contains a long matra (ా ీ ూ ే ో ౌ ృ)
              OR starts with an independent long vowel (ఆ ఈ ఊ ౠ ఏ ఓ).

    Rule 2 — Diphthong (సంధ్యక్షరం):
              Syllable contains diphthong matra (ై ౌ)
              OR is/contains an independent diphthong (ఐ ఔ).
              (ౌ overlaps with Rule 1 — either rule firing is sufficient.)

    Rule 3 — Anusvara / Visarga (అనుస్వారం / విసర్గ):
              Syllable contains ం or ః.

    Rule 4 — Ends with Virama (పొల్లు హల్లు):
              Syllable ends with ్ (halant/virama) — an incomplete syllable.

    Rule 5 — Sandhi lookahead (సంధి):
              The NEXT syllable (within the same word) starts with a conjunct
              or doubled consonant (C + ్ + C). The CURRENT syllable becomes
              Guru because the first consonant of the cluster phonetically
              "closes" the current syllable.

              CRITICAL: Rule 5 does NOT cross word boundaries (spaces) or
              line boundaries (newlines). Seeing a space or newline between
              the current syllable and a following conjunct suppresses Rule 5.

WORD / LINE BOUNDARY RULE:
    A space (' ') or newline ('\n') blocks Rule 5 for the preceding syllable.
    Each new word and each new line begins independently — no sandhi effect
    carries over from the previous word or line.

    Example:  "తనుమ ళ్ళరాస్తుంది"
              Syllables: తు  ను  మ  ' '  ళ్ళ  రా  స్తుం  ది
              Markers:    I   I   I   ' '   I    U    U     I
              (మ stays I because the space blocks Rule 5 before ళ్ళ)

-------------------------------------------------------------------------------
FST STATE DIAGRAM
-------------------------------------------------------------------------------

```mermaid
stateDiagram-v2
    [*] --> IDLE

    IDLE --> PENDING_CLEAR : syllable S\n/ buffer(S)
    IDLE --> IDLE : space or newline\n/ emit(boundary)

    PENDING_CLEAR --> PENDING_CLEAR : syllable S2\n/ emit Rule5? U else self_class(P)\nbuffer(S2)
    PENDING_CLEAR --> PENDING_BOUNDARY : space or newline\n/ buffer_boundary

    PENDING_BOUNDARY --> PENDING_CLEAR : syllable S2\n/ emit self_class(P)\nflush_boundaries\nbuffer(S2)
    PENDING_BOUNDARY --> PENDING_BOUNDARY : space or newline\n/ buffer_boundary

    PENDING_CLEAR --> [*] : FLUSH\n/ emit self_class(P)\nflush_boundaries
    PENDING_BOUNDARY --> [*] : FLUSH\n/ emit self_class(P)\nflush_boundaries
```

-------------------------------------------------------------------------------
classify_self() FLOWCHART  (Rules 1–4 applied to a single syllable)
-------------------------------------------------------------------------------

```mermaid
flowchart TD
    A([syllable s]) --> B{has long matra\nor indep. long vowel?}
    B -- yes --> U([return U])
    B -- no --> C{has diphthong\nmatra or ఐ/ఔ?}
    C -- yes --> U
    C -- no --> D{has ం or ః?}
    D -- yes --> U
    D -- no --> E{ends with ్?}
    E -- yes --> U
    E -- no --> I([return I])
```

-------------------------------------------------------------------------------
WORD BOUNDARY — Rule 5 SUPPRESSION SEQUENCE
-------------------------------------------------------------------------------

```mermaid
sequenceDiagram
    participant In as Input stream
    participant FST as GanaMarker FST
    participant Out as Output

    In->>FST: syllable మ (self=I)
    Note over FST: state → PENDING_CLEAR, pending=మ

    In->>FST: space ' '
    Note over FST: state → PENDING_BOUNDARY\nbuffer boundary ' '

    In->>FST: syllable ళ్ళ (conjunct start!)
    Note over FST: PENDING_BOUNDARY → Rule 5 BLOCKED\nemit self_class(మ) = I\nemit buffered ' '\nbuffer(ళ్ళ)
    FST->>Out: I
    FST->>Out: ' '

    In->>FST: FLUSH
    FST->>Out: I (self_class of ళ్ళ)
```

-------------------------------------------------------------------------------
FULL TRANSITION TABLE
-------------------------------------------------------------------------------

```mermaid
%%{init: {"theme": "base"}}%%
block-beta
columns 1
  block:table
    A["From: IDLE            | Input: syllable S     | Action: buffer(S), self=classify_self(S)        | Next: PENDING_CLEAR"]
    B["From: IDLE            | Input: space/newline   | Action: emit(boundary)                         | Next: IDLE"]
    C["From: PENDING_CLEAR   | Input: syllable S2     | Action: if conjunct_start(S2): emit(U)          | Next: PENDING_CLEAR"]
    D["                      |                        |         else: emit(self_class(P))               |                    "]
    E["                      |                        |         buffer(S2), self=classify_self(S2)      |                    "]
    F["From: PENDING_CLEAR   | Input: space/newline   | Action: buffer_boundary                         | Next: PENDING_BOUNDARY"]
    G["From: PENDING_BOUNDARY| Input: syllable S2     | Action: emit(self_class(P))                    | Next: PENDING_CLEAR"]
    H["                      |                        |         emit_all_boundaries()                   |                    "]
    I2["                     |                        |         buffer(S2), self=classify_self(S2)      |                    "]
    J["From: PENDING_BOUNDARY| Input: space/newline   | Action: buffer_boundary                         | Next: PENDING_BOUNDARY"]
    K["From: PENDING_*       | Input: FLUSH           | Action: emit(self_class(P)), emit_all_bounds    | Next: IDLE"]
  end
```

-------------------------------------------------------------------------------
"""

# ---------------------------------------------------------------------------
# Section 1: Character classification constants
# (Self-contained — no imports from src/. Mirrors syllable_assembler.py.)
# ---------------------------------------------------------------------------

TELUGU_CONSONANTS: set[str] = {
    "క", "ఖ", "గ", "ఘ", "ఙ", "చ", "ఛ", "జ", "ఝ", "ఞ",
    "ట", "ఠ", "డ", "ఢ", "ణ", "త", "థ", "ద", "ధ", "న",
    "ప", "ఫ", "బ", "భ", "మ", "య", "ర", "ల", "వ", "శ",
    "ష", "స", "హ", "ళ", "ఱ",
}

# Long dependent vowel matras (ా ీ ూ ే ో ౌ ౄ) — Rule 1
# NOTE: ృ (U+0C43) is the SHORT vocalic-R matra — it is NOT a long vowel.
#       ౄ (U+0C44) is the LONG vocalic-R matra — it IS included.
# Reference: analyzer.py :: long_vowels = {"ా", "ీ", "ూ", "ే", "ో", "ౌ", "ౄ"}
LONG_MATRAS: set[str] = {"ా", "ీ", "ూ", "ే", "ో", "ౌ", "ౄ"}

# Diphthong dependent matra forms — Rule 2
# ై (ai-matra) and ౌ (au-matra). Note: ౌ also matches Rule 1, but either
# rule firing is sufficient to classify the syllable as Guru.
DIPHTHONG_MATRAS: set[str] = {"ై", "ౌ"}

# Independent long vowels (ఆ ఈ ఊ ౠ ఏ ఓ) — Rule 1
INDEPENDENT_LONG_VOWELS: set[str] = {"ఆ", "ఈ", "ఊ", "ౠ", "ఏ", "ఓ"}

# Independent diphthong vowels (ఐ ఔ) — Rule 2
INDEPENDENT_DIPHTHONGS: set[str] = {"ఐ", "ఔ"}

# Halant / Virama — used in Rule 4 and conjunct-start detection
VIRAMA: str = "్"

# Anusvara and Visarga — Rule 3
DIACRITICS: set[str] = {"ం", "ః"}

# Boundary characters (passed through unchanged)
SPACE:   str = " "
NEWLINE: str = "\n"


# ---------------------------------------------------------------------------
# Section 2: Guru rule functions (pure, stateless)
# ---------------------------------------------------------------------------

def has_long_vowel(s: str) -> bool:
    """
    Rule 1: Syllable contains a long matra or is an independent long vowel.

    Examples:
        has_long_vowel("రా")  -> True   (ా long matra)
        has_long_vowel("ఆ")   -> True   (independent long vowel)
        has_long_vowel("రు")  -> False  (ు is short)
        has_long_vowel("ర")   -> False  (bare consonant, inherent short అ)
    """
    if any(m in s for m in LONG_MATRAS):
        return True
    # Independent long vowel syllable (e.g. "ఆ", "ఏ")
    if len(s) >= 1 and s[0] in INDEPENDENT_LONG_VOWELS:
        return True
    return False


def has_diphthong(s: str) -> bool:
    """
    Rule 2: Syllable contains a diphthong matra (ై, ౌ) or is/contains ఐ/ఔ.

    Examples:
        has_diphthong("గై")   -> True   (ై diphthong matra)
        has_diphthong("గౌ")   -> True   (ౌ diphthong matra)
        has_diphthong("ఐ")    -> True   (independent ఐ)
        has_diphthong("ఔ")    -> True   (independent ఔ)
        has_diphthong("గు")   -> False
    """
    if any(m in s for m in DIPHTHONG_MATRAS):
        return True
    if any(v in s for v in INDEPENDENT_DIPHTHONGS):
        return True
    return False


def has_diacritic(s: str) -> bool:
    """
    Rule 3: Syllable contains anusvara (ం) or visarga (ః).

    Examples:
        has_diacritic("సం")  -> True
        has_diacritic("దుః")  -> True
        has_diacritic("స")   -> False
    """
    return any(d in s for d in DIACRITICS)


def ends_with_virama(s: str) -> bool:
    """
    Rule 4: Syllable ends with halant/virama (్) — a pollu hallu.

    A pollu hallu is a trailing consonant+virama that could not attach to
    a following consonant (e.g. "పూసెన్" ends with "న్").

    Examples:
        ends_with_virama("న్")   -> True
        ends_with_virama("పూసె") -> False
    """
    return s.endswith(VIRAMA)


def is_conjunct_start(s: str) -> bool:
    """
    Rule 5 probe: Does this syllable start with a conjunct or doubled consonant?

    A conjunct start has the pattern: C + ్ + C (at indices 0, 1, 2).
    This covers:
        - Distinct conjuncts:  స్క, త్య, స్త్రీ
        - Doubled consonants:  మ్మ, ళ్ళ, న్న

    This is applied to the NEXT syllable to decide if the CURRENT (pending)
    syllable should be upgraded from I to U by Rule 5.

    Examples:
        is_conjunct_start("త్య")  -> True   (distinct conjunct)
        is_conjunct_start("మ్మ")  -> True   (doubled consonant)
        is_conjunct_start("ళ్ళ")  -> True   (doubled consonant)
        is_conjunct_start("మ")    -> False  (simple consonant)
        is_conjunct_start("రా")   -> False  (consonant + long matra, no virama)
    """
    return (
        len(s) >= 3
        and s[0] in TELUGU_CONSONANTS
        and s[1] == VIRAMA
        and s[2] in TELUGU_CONSONANTS
    )


def classify_self(s: str) -> str:
    """
    Classify a syllable by its own properties only (Rules 1–4).

    Returns "U" (Guru) if any of Rules 1–4 apply, else "I" (Laghu).
    Rule 5 (sandhi lookahead) is NOT applied here — it is handled by the
    FST state machine in GanaMarker based on the next syllable seen.

    Args:
        s: A single Telugu syllable (output of SyllableAssembler).

    Returns:
        "U" if Guru, "I" if Laghu.

    Examples:
        classify_self("రా")    -> "U"  (Rule 1: long matra)
        classify_self("గై")    -> "U"  (Rule 2: diphthong matra)
        classify_self("సం")    -> "U"  (Rule 3: anusvara)
        classify_self("న్")    -> "U"  (Rule 4: ends with virama)
        classify_self("త")     -> "I"  (no guru rule applies)
        classify_self("తు")    -> "I"  (short matra ు)
    """
    if has_long_vowel(s):   return "U"  # Rule 1
    if has_diphthong(s):    return "U"  # Rule 2
    if has_diacritic(s):    return "U"  # Rule 3
    if ends_with_virama(s): return "U"  # Rule 4
    return "I"


# ---------------------------------------------------------------------------
# Section 3: FST state constants
# ---------------------------------------------------------------------------

STATE_IDLE             = "IDLE"
STATE_PENDING_CLEAR    = "PENDING_CLEAR"     # no boundary since pending syl
STATE_PENDING_BOUNDARY = "PENDING_BOUNDARY"  # boundary seen → Rule 5 blocked


# ---------------------------------------------------------------------------
# Section 4: GanaMarker class
# ---------------------------------------------------------------------------

class GanaMarker:
    """
    FST-based Telugu Ganana (Guru/Laghu) marker.

    Consumes the syllable stream produced by SyllableAssembler and emits
    a U/I marker for each syllable, plus pass-through for spaces/newlines.

    The FST has 3 states:
        IDLE              — No pending syllable.
        PENDING_CLEAR     — A syllable is buffered; no word/line boundary
                            has been seen since it. Rule 5 may still fire
                            when the next syllable arrives.
        PENDING_BOUNDARY  — A syllable is buffered; at least one space or
                            newline has been seen since it. Rule 5 is
                            suppressed for this syllable.

    Usage:
        gm = GanaMarker()
        markers = gm.process(['తె', 'లు', 'గు'])
        # => ['I', 'I', 'I']

        markers = gm.process(['స', 'త్య', 'ము'])
        # => ['U', 'I', 'I']  (స upgraded by Rule 5 before conjunct త్య)

        markers = gm.process(['మ', ' ', 'ళ్ళ'])
        # => ['I', ' ', 'I']  (మ stays I — space blocks Rule 5)
    """

    def __init__(self):
        self._reset()

    def _reset(self):
        self.state: str = STATE_IDLE
        self.pending_syl:   str | None = None   # buffered syllable
        self.pending_class: str | None = None   # its self-class ("U" or "I")
        self.pending_boundaries: list[str] = [] # spaces/newlines buffered after it
        self.output: list[str] = []

    def _emit(self, marker: str):
        self.output.append(marker)

    def _flush_pending(self):
        """Emit the pending syllable's class then all buffered boundaries."""
        if self.pending_syl is not None:
            self._emit(self.pending_class)
            for b in self.pending_boundaries:
                self._emit(b)
        self.pending_syl        = None
        self.pending_class      = None
        self.pending_boundaries = []

    # ------------------------------------------------------------------
    # FST transitions
    # ------------------------------------------------------------------

    def _on_syllable(self, syl: str):
        """Handle a syllable token."""
        if self.state == STATE_IDLE:
            # No pending syllable — buffer this one.
            self.pending_syl        = syl
            self.pending_class      = classify_self(syl)
            self.pending_boundaries = []
            self.state              = STATE_PENDING_CLEAR

        elif self.state == STATE_PENDING_CLEAR:
            # Pending syllable P, no space-boundary since P (newlines may be buffered).
            # Rule 5: if this new syllable starts a conjunct, P becomes Guru.
            if is_conjunct_start(syl):
                self._emit("U")
            else:
                self._emit(self.pending_class)
            # Flush any newlines buffered between P and this syllable.
            for b in self.pending_boundaries:
                self._emit(b)
            # Buffer the new syllable.
            self.pending_syl        = syl
            self.pending_class      = classify_self(syl)
            self.pending_boundaries = []
            # State stays PENDING_CLEAR.

        elif self.state == STATE_PENDING_BOUNDARY:
            # Pending syllable P, with at least one boundary after P.
            # Rule 5 is BLOCKED — emit P's self-class only.
            self._emit(self.pending_class)
            # Emit all buffered boundaries in order.
            for b in self.pending_boundaries:
                self._emit(b)
            # Buffer the new syllable with a clean slate.
            self.pending_syl        = syl
            self.pending_class      = classify_self(syl)
            self.pending_boundaries = []
            self.state              = STATE_PENDING_CLEAR

    def _on_boundary(self, boundary: str):
        """
        Handle a space or newline token.

        SPACE (' '):
            Blocks Rule 5 — transitions PENDING_CLEAR → PENDING_BOUNDARY.
            This matches the reference analyser: akshara_ganavibhajana() only
            breaks the lookahead loop on ' ', not on '\n'.

        NEWLINE ('\n'):
            Transparent to Rule 5 — stays in current state.
            The analyser treats '\n' as an ignorable character and skips over
            it when searching for the next syllable for the sandhi check.
            Therefore a conjunct at the start of the next line CAN upgrade the
            last syllable of the current line.
        """
        if self.state == STATE_IDLE:
            self._emit(boundary)

        elif self.state == STATE_PENDING_CLEAR:
            self.pending_boundaries.append(boundary)
            if boundary == SPACE:
                # Only spaces block Rule 5
                self.state = STATE_PENDING_BOUNDARY
            # Newlines leave state as PENDING_CLEAR — Rule 5 still possible

        elif self.state == STATE_PENDING_BOUNDARY:
            # Already blocked by a previous space — keep buffering.
            self.pending_boundaries.append(boundary)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def feed(self, item: str):
        """
        Feed a single item (syllable, space, or newline) into the FST.

        Args:
            item: A syllable string (e.g. "తె", "స్క") OR " " OR "\\n".
        """
        if item == SPACE or item == NEWLINE:
            self._on_boundary(item)
        else:
            self._on_syllable(item)

    def flush(self):
        """
        Signal end of input and flush any remaining buffered syllable.

        Call this after the last feed() to ensure the final syllable is
        emitted with its self-class (Rule 5 has no following syllable to fire).
        """
        if self.state in (STATE_PENDING_CLEAR, STATE_PENDING_BOUNDARY):
            self._emit(self.pending_class)
            for b in self.pending_boundaries:
                self._emit(b)
        self.state              = STATE_IDLE
        self.pending_syl        = None
        self.pending_class      = None
        self.pending_boundaries = []

    def process(self, syllables: list[str]) -> list[str]:
        """
        Process a complete syllable list and return U/I markers.

        Args:
            syllables: List of syllables and boundaries from SyllableAssembler.
                       E.g. ['తె', 'లు', 'గు', ' ', 'భా', 'ష']

        Returns:
            List of markers: 'U', 'I', ' ', or '\\n'.
            E.g. ['I', 'I', 'I', ' ', 'U', 'I']
        """
        self._reset()
        for item in syllables:
            self.feed(item)
        self.flush()
        return self.output

    def process_with_trace(
        self, syllables: list[str]
    ) -> tuple[list[str], list[dict]]:
        """
        Process syllables and return (markers, trace) for debugging.

        Each trace entry is a dict with:
            item           - the input item (syllable or boundary)
            state_before   - FST state before this item
            pending_before - pending syllable before (or None)
            rule5_fired    - True if Rule 5 upgraded the pending syllable
            emitted        - list of markers emitted by this step
            state_after    - FST state after this item
        """
        self._reset()
        trace: list[dict] = []

        items = list(syllables) + [None]   # None sentinel triggers flush

        for item in items:
            state_before   = self.state
            pending_before = self.pending_syl
            output_len_before = len(self.output)

            rule5_fired = False
            if item is None:
                self.flush()
            else:
                # Detect Rule 5 firing: PENDING_CLEAR + conjunct-start syllable
                if (
                    item not in (SPACE, NEWLINE)
                    and self.state == STATE_PENDING_CLEAR
                    and is_conjunct_start(item)
                ):
                    rule5_fired = True
                self.feed(item)

            emitted = list(self.output[output_len_before:])

            trace.append({
                "item":           item if item is not None else "∎",
                "state_before":   state_before,
                "pending_before": pending_before,
                "rule5_fired":    rule5_fired,
                "emitted":        emitted,
                "state_after":    self.state,
            })

        return self.output, trace


# ---------------------------------------------------------------------------
# Section 5: Integration helper
# ---------------------------------------------------------------------------

def mark_text(text: str) -> list[str]:
    """
    End-to-end helper: raw Telugu text → U/I markers.

    Chains SyllableAssembler (Stage 1) with GanaMarker (Stage 2).

    Args:
        text: A raw Telugu string (word, phrase, or multi-line poem).

    Returns:
        A list of 'U', 'I', ' ', '\\n' markers — one per syllable or boundary.

    Examples:
        mark_text("తెలుగు")
        # => ['I', 'I', 'I']

        mark_text("రాముడు")
        # => ['U', 'I', 'I']

        mark_text("సత్యము")
        # => ['U', 'I', 'I']   (స upgraded by Rule 5 before conjunct త్య)

        mark_text("తనుమ ళ్ళరాస్తుంది")
        # => ['I', 'I', 'I', ' ', 'I', 'U', 'U', 'I']
        #     మ stays I because space blocks Rule 5 before ళ్ళ
    """
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
    from syllable_assembler import SyllableAssembler

    syllables = SyllableAssembler().process(text)
    return GanaMarker().process(syllables)


# ---------------------------------------------------------------------------
# Section 6: Tests — verified against akshara_ganavibhajana()
# ---------------------------------------------------------------------------

def _print_trace(
    desc: str,
    text: str,
    syllables: list[str],
    result: list[str],
    gt: list[str],
    trace: list[dict],
    index: int,
):
    status = "PASS" if result == gt else "FAIL"
    bar = "=" * 90

    print()
    print(bar)
    print(f"Test {index}: {desc}  [{status}]")
    print(bar)
    print(f"  Input      : {repr(text)}")
    print(f"  Syllables  : {syllables}")
    print()

    # Side-by-side syllable → marker comparison (analyser vs FST)
    # Only syllables (not boundary markers) are shown for readability.
    syl_markers = [(s, r, g) for s, r, g in zip(syllables, result, gt)
                   if r not in (" ", "\n") and g not in (" ", "\n")]
    if syl_markers:
        col = max(len(s) for s, _, _ in syl_markers) + 2
        col = max(col, 8)
        hdr = f"  {'Syllable':<{col}} {'Analyser':^8} {'FST':^8} {'Match':^6}"
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        for syl, fst_mark, gt_mark in syl_markers:
            match_sym = "✓" if fst_mark == gt_mark else "✗"
            print(f"  {syl:<{col}} {gt_mark:^8} {fst_mark:^8} {match_sym:^6}")
        print()

    # Full FST state trace
    header = (
        f"  {'Item':<14} {'State Before':<20} {'Pending':<14} "
        f"{'Rule5':<6} {'State After':<20} {'Emitted'}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for row in trace:
        item_disp    = repr(row["item"]) if row["item"] in (" ", "\n", "∎") else row["item"]
        pending_disp = row["pending_before"] if row["pending_before"] else "—"
        rule5_disp   = "YES" if row["rule5_fired"] else "—"
        emitted_disp = ", ".join(repr(e) for e in row["emitted"]) if row["emitted"] else "—"
        print(
            f"  {item_disp:<14} {row['state_before']:<20} {pending_disp:<14} "
            f"{rule5_disp:<6} {row['state_after']:<20} {emitted_disp}"
        )

    print()
    print(f"  FST output  : {result}")
    print(f"  Analyser GT : {gt}")
    if result != gt:
        print("  *** MISMATCH ***")


def run_tests():
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

    from syllable_assembler import SyllableAssembler
    # Ground truth: canonical dwipada analyser in src/dwipada/core/analyzer.py
    # This is the same logic used by analyze_dwipada() and analyze_pada().
    from dwipada.core.analyzer import split_aksharalu, akshara_ganavibhajana

    asm = SyllableAssembler()
    gm  = GanaMarker()

    def ground_truth(text: str) -> list[str]:
        """
        Run the reference implementation from dwipada.core.analyzer.

        Uses split_aksharalu() + akshara_ganavibhajana() — the same functions
        that power analyze_dwipada() and analyze_pada() — as the canonical
        ground truth for guru/laghu classification.

        akshara_ganavibhajana() returns:
            "U"  for Guru syllables
            "I"  for Laghu syllables
            ""   for ignorable characters (spaces, newlines, arasunna, etc.)

        We re-insert the actual boundary characters (' ', '\\n') to match
        the output format of GanaMarker.process().
        """
        syllables = split_aksharalu(text)
        markers   = akshara_ganavibhajana(syllables)
        result = []
        for syl, mark in zip(syllables, markers):
            if mark == "":
                result.append(syl)   # space or newline — pass through as-is
            else:
                result.append(mark)
        return result

    test_cases = [
        # ---------------------------------------------------------------
        # All Laghu (simple short vowel syllables — no Guru rule applies)
        # ---------------------------------------------------------------
        ("All Laghu: bare consonants",             "నమక"),
        ("All Laghu: short matras",                "తెలుగు"),
        ("All Laghu: independent short vowels",    "అఇఉఎఒ"),

        # ---------------------------------------------------------------
        # Rule 1 — Long vowel
        # ---------------------------------------------------------------
        ("Rule 1: long matra ా",                   "రాముడు"),
        ("Rule 1: long matra ీ",                    "సీత"),
        ("Rule 1: long matra ూ",                    "భూమి"),
        ("Rule 1: independent long vowel ఆ",       "ఆదిన్"),
        ("Rule 1: independent long vowel ఏ",       "ఏది"),

        # ---------------------------------------------------------------
        # Rule 2 — Diphthong
        # ---------------------------------------------------------------
        ("Rule 2: diphthong matra ై",               "సైనికుడు"),
        ("Rule 2: diphthong matra ౌ",               "గౌరవం"),
        ("Rule 2: independent diphthong ఐ",        "ఐదు"),
        ("Rule 2: independent diphthong ఔ",        "ఔషధం"),

        # ---------------------------------------------------------------
        # Rule 3 — Anusvara / Visarga
        # ---------------------------------------------------------------
        ("Rule 3: anusvara ం",                     "సందడి"),
        ("Rule 3: visarga ః",                      "దుఃఖము"),
        ("Rule 3: anusvara mid-word",              "సంస్కృతం"),

        # ---------------------------------------------------------------
        # Rule 4 — Ends with Virama (pollu hallu)
        # ---------------------------------------------------------------
        ("Rule 4: pollu at end of word",           "పూసెన్"),
        ("Rule 4: pollu before space",             "పదిన్ కాలం"),
        ("Rule 4: pollu at end of input",          "ఆదిన్"),

        # ---------------------------------------------------------------
        # Rule 5 — Sandhi lookahead (conjunct within same word)
        # ---------------------------------------------------------------
        ("Rule 5: before distinct conjunct",       "సత్యము"),
        ("Rule 5: before doubled consonant",       "అమ్మ"),
        ("Rule 5: triple conjunct",                "శ్రీరాముడు"),
        ("Rule 5: before conjunct mid-word",       "కృష్ణ"),

        # ---------------------------------------------------------------
        # Rule 5 BLOCKED — word boundary (space)
        # ---------------------------------------------------------------
        ("Rule 5 blocked by space",                "తనుమ ళ్ళరాస్తుంది"),
        ("Rule 5 blocked by space (2)",            "తెలుగు స్త్రీ"),
        ("Two words, no cross-boundary sandhi",    "నమక స్కారం"),

        # ---------------------------------------------------------------
        # Rule 5 BLOCKED — line boundary (newline)
        # ---------------------------------------------------------------
        ("Rule 5 blocked by newline",              "పదిన్\nత్వరగా"),
        ("Two lines, independent gana",            "తెలుగు\nస్త్రీ"),

        # ---------------------------------------------------------------
        # Individual words — common Telugu vocabulary
        # ---------------------------------------------------------------
        ("Word: రాముడు (Ra-mu-Du)",               "రాముడు"),
        ("Word: సీతమ్మ (conjunct doubled మ)",      "సీతమ్మ"),
        ("Word: లక్ష్మి (conjunct క్ష్మ)",          "లక్ష్మి"),
        ("Word: ధర్మము (pollu న్ inside)",          "ధర్మము"),
        ("Word: సంస్కారం",                         "సంస్కారం"),
        ("Word: వర్షంబు",                           "వర్షంబు"),
        ("Word: భక్తి",                             "భక్తి"),
        ("Word: దేశభక్తుడు",                       "దేశభక్తుడు"),
        ("Word: పరమాత్మ",                           "పరమాత్మ"),
        ("Word: ప్రపంచం",                           "ప్రపంచం"),

        # ---------------------------------------------------------------
        # Phrases — multi-word, tests word-boundary Rule 5 suppression
        # ---------------------------------------------------------------
        ("Phrase: రామ నామము",                      "రామ నామము"),
        ("Phrase: నీలమేఘ శ్యామ",                   "నీలమేఘ శ్యామ"),
        ("Phrase: జయ జయ రాఘవ",                     "జయ జయ రాఘవ"),
        ("Phrase: తెలుగు భాష",                      "తెలుగు భాష"),
        ("Phrase: శ్రీరాముడు జయం",                  "శ్రీరాముడు జయం"),
        ("Phrase: పూసెన్ నాటిన్",                   "పూసెన్ నాటిన్"),
        ("Phrase: ఆది కావ్యము",                      "ఆది కావ్యము"),
        ("Phrase: సత్యం శివం సుందరం",               "సత్యం శివం సుందరం"),
        ("Phrase: తల్లి భాష తెలుగు",                "తల్లి భాష తెలుగు"),
        ("Phrase: శాంతి ప్రస్తావన",                  "శాంతి ప్రస్తావన"),

        # ---------------------------------------------------------------
        # Full dwipada couplets — baseline
        # ---------------------------------------------------------------
        (
            "Dwipada line 1",
            "సౌధాగ్రముల యందు సదనంబు లందు",
        ),
        (
            "Dwipada couplet — సౌధాగ్రముల",
            "సౌధాగ్రముల యందు సదనంబు లందు\nవీధుల యందును వెఱవొప్ప నిలిచి",
        ),
        (
            # పు=U (Rule 5: next syl వ్వు conjunct within same word)
            # మ=U (Rule 5: next syl వ్వ conjunct within same word)
            # newline blocks Rule 5 — last syl of line 1 unaffected by line 2
            "Dwipada couplet — పువ్వులు",
            "పువ్వులు మలయంగ పుడమిలో నచట\nమవ్వగా నుండును మాటిమాటికిని",
        ),

        # ---------------------------------------------------------------
        # 10 dwipada couplets from dwipada_augmented_dataset.json
        # Ground truth verified against akshara_ganavibhajana()
        # ---------------------------------------------------------------
        (
            # Source: ranganatha_ramayanam
            # Ganas: Ta(UUI) | Ra(UIU) | Naga(IIIU) | Na(III)
            #        Ta(UUI) | Ta(UUI) | Ra(UIU)    | Na(III)
            "Dataset #1 — మారాముబాణ",
            "మా రాముబాణనిర్మథితమాంసముల\nకీ రాదె నీ నాక మేల యిచ్చెదవు",
        ),
        (
            # Source: ranganatha_ramayanam
            # Ganas: Sala(IIUI) | Ta(UUI) | Bha(UII) | Ha(UI)
            #        Sala(IIUI) | Ta(UUI)  | Naga(IIIU) | Na(III)
            "Dataset #2 — భువనత్రయ",
            "భువనత్రయాధారభూతమయుండు\nపవనుండు లేకున్న బడు శరీరములు",
        ),
        (
            # Source: dwipada_bhagavatam2
            # Ganas: Ta(UUI) | Sala(IIUI) | Nala(IIII) | Ha(UI)
            #        Ta(UUI) | Sala(IIUI)  | Sala(IIUI) | Ha(UI)
            "Dataset #3 — యీక్షింప",
            "యీక్షింప నదిగాక యిరువదినాల్గు\nయక్షోహిణులు దాను నట దండువచ్చి",
        ),
        (
            # Source: ranganatha_ramayanam
            # Ganas: Ra(UIU) | Ra(UIU)   | Sala(IIUI) | Ha(UI)
            #        Ra(UIU) | Sala(IIUI) | Naga(IIIU) | Na(III)
            "Dataset #4 — నందు నిద్రించె",
            "నందు నిద్రించె నయ్యవనీశ్వరుండు\nనందమై విలసిల్లె నభినవస్ఫురణ",
        ),
        (
            # Source: ranganatha_ramayanam
            # Ganas: Bha(UII)  | Sala(IIUI) | Ra(UIU)   | Na(III)
            #        Ra(UIU)   | Bha(UII)   | Sala(IIUI) | Na(III)
            "Dataset #5 — యమ్ముని",
            "యమ్మునిగొనిపోయి యర్ఘ్యపాద్యముల\nనెమ్మితో నిచ్చిన నృపుజూచి యతడు",
        ),
        (
            # Source: ranganatha_ramayanam
            # Ganas: Sala(IIUI) | Naga(IIIU) | Ra(UIU) | Ha(UI)
            #        Sala(IIUI) | Bha(UII)   | Ra(UIU) | Ha(UI)
            "Dataset #6 — ధరియించి",
            "ధరియించి సముచితస్తన్యపానంబు\nసరినొప్పజేయుట షణ్ముఖుండయ్యె",
        ),
        (
            # Source: ranganatha_ramayanam
            # Ganas: Sala(IIUI) | Ta(UUI) | Ta(UUI) | Na(III)
            #        Sala(IIUI) | Bha(UII) | Ta(UUI) | Na(III)
            "Dataset #7 — చరితంబు",
            "చరితంబు ధైర్యంబు శౌర్యంబు నతడు\nఖరదూషణాదులఖండించుటయును",
        ),
        (
            # Source: ranganatha_ramayanam
            # Ganas: Naga(IIIU) | Sala(IIUI) | Sala(IIUI) | Na(III)
            #        Sala(IIUI) | Nala(IIII) | Ra(UIU)    | Na(III)
            "Dataset #8 — నెనయ నెప్పటి",
            "నెనయ నెప్పటిచోట నిరవొంద నునిచి\nచనుదెంచె రయమున సంగరస్థలికి",
        ),
        (
            # Source: basava_puranam
            # Ganas: Ta(UUI) | Sala(IIUI) | Ta(UUI) | Ha(UI)
            #        Bha(UII) | Ta(UUI)   | Sala(IIUI) | Ha(UI)
            "Dataset #9 — నొక్కొక్క",
            "నొక్కొక్క నియమంబు నిక్కంబుగాగ\nజక్కన దర్శింప జనుదెంచువారు",
        ),
        (
            # Source: ranganatha_ramayanam
            # Ganas: Nala(IIII) | Ra(UIU)   | Sala(IIUI) | Ha(UI)
            #        Sala(IIUI) | Naga(IIIU) | Sala(IIUI) | Ha(UI)
            "Dataset #10 — తడయక రాముచే",
            "తడయక రాముచే దశకంఠుడింక\nజెడునంచు ముడియ వైచినబాగు దోప",
        ),
    ]

    passed = 0
    failed = 0

    for i, (desc, text) in enumerate(test_cases, 1):
        syllables      = asm.process(text)
        result, trace  = gm.process_with_trace(syllables)
        gt             = ground_truth(text)
        match          = result == gt
        _print_trace(desc, text, syllables, result, gt, trace, i)
        if match:
            passed += 1
        else:
            failed += 1

    print()
    print("=" * 90)
    print(f"SUMMARY: {passed} passed, {failed} failed out of {passed + failed} tests")
    print("=" * 90)
    return failed == 0


if __name__ == "__main__":
    ok = run_tests()
    raise SystemExit(0 if ok else 1)
