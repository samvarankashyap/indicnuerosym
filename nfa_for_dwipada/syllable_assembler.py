"""
Syllable Assembler FST for Telugu.

Reads a stream of Unicode characters and emits complete syllables (aksharalu),
spaces, and newlines — matching the behaviour of split_aksharalu() in
aksharanusarika.py.

States:
    IDLE               - No syllable buffer active
    CONSONANT_CLUSTER  - Building a consonant-based syllable
    PENDING_VIRAMA     - Buffer ends with [C + ్], waiting to see if conjunct or pollu
    VOWEL              - Building an independent-vowel syllable

See design_syllable_assembler.md for the full FST design.
"""

# ---------------------------------------------------------------------------
# Codepoint categories (from design_codepoint_classifier.md)
# ---------------------------------------------------------------------------

TELUGU_CONSONANTS = {
    "క", "ఖ", "గ", "ఘ", "ఙ", "చ", "ఛ", "జ", "ఝ", "ఞ",
    "ట", "ఠ", "డ", "ఢ", "ణ", "త", "థ", "ద", "ధ", "న",
    "ప", "ఫ", "బ", "భ", "మ", "య", "ర", "ల", "వ", "శ",
    "ష", "స", "హ", "ళ", "ఱ",
}
INDEPENDENT_VOWELS = {
    "అ", "ఆ", "ఇ", "ఈ", "ఉ", "ఊ", "ఋ", "ౠ",
    "ఎ", "ఏ", "ఐ", "ఒ", "ఓ", "ఔ",
}
MATRAS = {"ా", "ి", "ీ", "ు", "ూ", "ృ", "ె", "ే", "ై", "ొ", "ో", "ౌ", "ౖ"}
VIRAMA = "్"
DIACRITICS = {"ం", "ః"}   # anusvara, visarga
SKIP_CHARS = {"\u200c", "\u200b", "ఁ"}  # ZWNJ, ZWSP, candrabindu


# Category constants
CAT_CONSONANT   = "CONSONANT"
CAT_INDEP_VOWEL = "INDEPENDENT_VOWEL"
CAT_MATRA       = "MATRA"
CAT_VIRAMA      = "VIRAMA"
CAT_DIACRITIC   = "DIACRITIC"
CAT_SPACE       = "SPACE"
CAT_NEWLINE     = "NEWLINE"
CAT_SKIP        = "SKIP"
CAT_OTHER       = "OTHER"


def classify(ch: str) -> str:
    if ch in TELUGU_CONSONANTS:  return CAT_CONSONANT
    if ch in INDEPENDENT_VOWELS: return CAT_INDEP_VOWEL
    if ch in MATRAS:             return CAT_MATRA
    if ch == VIRAMA:             return CAT_VIRAMA
    if ch in DIACRITICS:         return CAT_DIACRITIC
    if ch == " ":                return CAT_SPACE
    if ch == "\n":               return CAT_NEWLINE
    if ch in SKIP_CHARS:         return CAT_SKIP
    return CAT_OTHER


# ---------------------------------------------------------------------------
# FST States
# ---------------------------------------------------------------------------

STATE_IDLE              = "IDLE"
STATE_CONSONANT_CLUSTER = "CONSONANT_CLUSTER"
STATE_PENDING_VIRAMA    = "PENDING_VIRAMA"
STATE_VOWEL             = "VOWEL"


class SyllableAssembler:
    """
    FST-based Telugu syllable assembler.

    Usage:
        asm = SyllableAssembler()
        syllables = asm.process("నమస్కారం")
    """

    def __init__(self):
        self._reset()

    def _reset(self):
        self.state = STATE_IDLE
        self.buffer: list[str] = []
        self.prev_syllable: str | None = None
        self.output: list[str] = []

    def _buffer_str(self) -> str:
        return "".join(self.buffer)

    def _emit_syllable(self):
        """Emit the current buffer as a complete syllable."""
        s = self._buffer_str()
        if s:
            self.output.append(s)
            self.prev_syllable = s
        self.buffer = []

    def _emit_pollu_merge(self):
        """
        Pollu merge: append buffer (C + ్) onto prev_syllable.
        If no prev_syllable, emit buffer on its own.
        """
        pollu = self._buffer_str()
        if self.prev_syllable is not None:
            # Replace the last emitted syllable with the merged form
            if self.output and self.output[-1] == self.prev_syllable:
                merged = self.prev_syllable + pollu
                self.output[-1] = merged
                self.prev_syllable = merged
            else:
                # prev_syllable was separated by a boundary — emit pollu standalone
                self.output.append(pollu)
                self.prev_syllable = pollu
        else:
            # No previous syllable (pollu at very start — unusual)
            self.output.append(pollu)
            self.prev_syllable = pollu
        self.buffer = []

    def _emit_boundary(self, ch: str):
        """Emit a SPACE or NEWLINE boundary character."""
        self.output.append(ch)
        # Boundaries break the prev_syllable chain for pollu purposes
        self.prev_syllable = None

    # ------------------------------------------------------------------
    # Transition handlers
    # ------------------------------------------------------------------

    def _from_idle(self, ch: str, cat: str):
        if cat == CAT_CONSONANT:
            self.buffer = [ch]
            self.state = STATE_CONSONANT_CLUSTER
        elif cat == CAT_INDEP_VOWEL:
            self.buffer = [ch]
            self.state = STATE_VOWEL
        elif cat in (CAT_SPACE, CAT_NEWLINE):
            self._emit_boundary(ch)
        elif cat == CAT_DIACRITIC:
            # Diacritic (anusvara/visarga) immediately after a complete syllable
            # (e.g. ు then ః in "దుఃఖ") — attach to the previous syllable.
            if self.prev_syllable is not None and self.output and self.output[-1] == self.prev_syllable:
                merged = self.prev_syllable + ch
                self.output[-1] = merged
                self.prev_syllable = merged
            else:
                # No previous syllable to attach to — emit standalone
                self.output.append(ch)
                self.prev_syllable = ch
        elif cat in (CAT_SKIP, CAT_OTHER, CAT_MATRA, CAT_VIRAMA):
            pass  # ignore

    def _from_consonant_cluster(self, ch: str, cat: str):
        if cat == CAT_VIRAMA:
            self.buffer.append(ch)
            self.state = STATE_PENDING_VIRAMA
        elif cat in (CAT_MATRA, CAT_DIACRITIC):
            self.buffer.append(ch)
            self._emit_syllable()
            self.state = STATE_IDLE
        elif cat == CAT_CONSONANT:
            # New consonant starts a new syllable — emit current first
            self._emit_syllable()
            self.buffer = [ch]
            self.state = STATE_CONSONANT_CLUSTER
        elif cat == CAT_INDEP_VOWEL:
            self._emit_syllable()
            self.buffer = [ch]
            self.state = STATE_VOWEL
        elif cat in (CAT_SPACE, CAT_NEWLINE):
            self._emit_syllable()
            self._emit_boundary(ch)
            self.state = STATE_IDLE
        elif cat == CAT_SKIP:
            pass  # transparent
        else:  # OTHER
            self._emit_syllable()
            self.state = STATE_IDLE

    def _from_pending_virama(self, ch: str, cat: str):
        if cat == CAT_CONSONANT:
            # Conjunct confirmed — extend the cluster
            self.buffer.append(ch)
            self.state = STATE_CONSONANT_CLUSTER
        elif cat in (CAT_MATRA, CAT_DIACRITIC):
            # Virama + matra is unusual but handle gracefully — emit as-is
            self.buffer.append(ch)
            self._emit_syllable()
            self.state = STATE_IDLE
        elif cat in (CAT_SPACE, CAT_NEWLINE):
            # Pollu — merge [C + ్] back into previous syllable
            self._emit_pollu_merge()
            self._emit_boundary(ch)
            self.state = STATE_IDLE
        elif cat == CAT_INDEP_VOWEL:
            # Pollu before a new word starting with vowel
            self._emit_pollu_merge()
            self.buffer = [ch]
            self.state = STATE_VOWEL
        elif cat == CAT_SKIP:
            pass  # transparent
        else:  # OTHER, or unexpected
            self._emit_pollu_merge()
            self.state = STATE_IDLE

    def _from_vowel(self, ch: str, cat: str):
        if cat == CAT_DIACRITIC:
            self.buffer.append(ch)
            self._emit_syllable()
            self.state = STATE_IDLE
        elif cat == CAT_CONSONANT:
            self._emit_syllable()
            self.buffer = [ch]
            self.state = STATE_CONSONANT_CLUSTER
        elif cat == CAT_INDEP_VOWEL:
            self._emit_syllable()
            self.buffer = [ch]
            self.state = STATE_VOWEL
        elif cat in (CAT_SPACE, CAT_NEWLINE):
            self._emit_syllable()
            self._emit_boundary(ch)
            self.state = STATE_IDLE
        elif cat == CAT_SKIP:
            pass  # transparent
        else:
            self._emit_syllable()
            self.state = STATE_IDLE

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def feed(self, ch: str):
        """Feed a single character into the FST."""
        cat = classify(ch)
        if self.state == STATE_IDLE:
            self._from_idle(ch, cat)
        elif self.state == STATE_CONSONANT_CLUSTER:
            self._from_consonant_cluster(ch, cat)
        elif self.state == STATE_PENDING_VIRAMA:
            self._from_pending_virama(ch, cat)
        elif self.state == STATE_VOWEL:
            self._from_vowel(ch, cat)

    def flush(self):
        """Signal end of input and flush any remaining buffer."""
        if self.state == STATE_CONSONANT_CLUSTER:
            self._emit_syllable()
        elif self.state == STATE_PENDING_VIRAMA:
            self._emit_pollu_merge()
        elif self.state == STATE_VOWEL:
            self._emit_syllable()
        self.state = STATE_IDLE

    def process(self, text: str) -> list[str]:
        """
        Process a complete Telugu string and return the list of syllables
        (plus spaces and newlines as boundary markers).
        """
        self._reset()
        for ch in text:
            self.feed(ch)
        self.flush()
        return self.output

    def process_with_trace(self, text: str) -> tuple[list[str], list[dict]]:
        """
        Process text and return (syllables, trace).

        Each trace entry is a dict with:
            char          - the input character
            codepoint     - e.g. "U+0C28"
            category      - classifier category string
            state_before  - FST state before this character
            buffer_before - buffer contents before
            state_after   - FST state after
            buffer_after  - buffer contents after
            emitted       - list of syllables/boundaries emitted by this step
        """
        self._reset()
        trace: list[dict] = []

        chars = list(text) + [None]  # None sentinel triggers flush
        for ch in chars:
            state_before   = self.state
            buffer_before  = self._buffer_str()
            output_len_before  = len(self.output)
            last_out_before    = self.output[-1] if self.output else None

            if ch is None:
                self.flush()
                cat = "END"
                cp  = "—"
            else:
                self.feed(ch)
                cat = classify(ch)
                cp  = f"U+{ord(ch):04X}"

            # Newly appended entries
            new_entries = list(self.output[output_len_before:])

            # Detect in-place pollu merge: output length unchanged but last entry changed
            if (output_len_before > 0
                    and len(self.output) == output_len_before
                    and self.output[-1] != last_out_before):
                emitted_label = [f"MERGE → {repr(self.output[-1])}"]
            else:
                emitted_label = [repr(e) for e in new_entries]

            trace.append({
                "char":          ch if ch is not None else "∎",
                "codepoint":     cp,
                "category":      cat,
                "state_before":  state_before,
                "buffer_before": buffer_before,
                "state_after":   self.state,
                "buffer_after":  self._buffer_str(),
                "emitted":       emitted_label,
            })

        return self.output, trace


# ---------------------------------------------------------------------------
# Tests — verified against split_aksharalu() from aksharanusarika.py
# ---------------------------------------------------------------------------

def _print_trace(desc: str, text: str, trace: list[dict], result: list[str],
                 gt: list[str], index: int):
    status = "PASS" if result == gt else "FAIL"
    bar = "=" * 80

    print()
    print(bar)
    print(f"Test {index}: {desc}  [{status}]")
    print(bar)

    # --- Unicode codepoint breakdown ---
    print(f"  Input       : {repr(text)}")
    cp_parts = []
    for ch in text:
        label = repr(ch) if ch in (" ", "\n") else ch
        cp_parts.append(f"{label}(U+{ord(ch):04X})")
    print(f"  Codepoints  : {' · '.join(cp_parts)}")
    print()

    # --- State trace table ---
    COL = {
        "char":   6,
        "cp":     10,
        "cat":    18,
        "s_bef":  20,
        "buf_b":  16,
        "s_aft":  20,
        "buf_a":  16,
        "emit":   20,
    }
    header = (
        f"  {'Char':<{COL['char']}} {'Codepoint':<{COL['cp']}} "
        f"{'Category':<{COL['cat']}} {'State Before':<{COL['s_bef']}} "
        f"{'Buf Before':<{COL['buf_b']}} {'State After':<{COL['s_aft']}} "
        f"{'Buf After':<{COL['buf_a']}} {'Emitted'}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))

    for row in trace:
        char_disp = repr(row["char"]) if row["char"] in (" ", "\n", "∎") else row["char"]
        emitted_disp = ", ".join(row["emitted"]) if row["emitted"] else "—"
        print(
            f"  {char_disp:<{COL['char']}} {row['codepoint']:<{COL['cp']}} "
            f"{row['category']:<{COL['cat']}} {row['state_before']:<{COL['s_bef']}} "
            f"{repr(row['buffer_before']):<{COL['buf_b']}} {row['state_after']:<{COL['s_aft']}} "
            f"{repr(row['buffer_after']):<{COL['buf_a']}} {emitted_disp}"
        )

    # --- Final result ---
    print()
    print(f"  Syllables   : {result}")
    print(f"  Ground truth: {gt}")
    if result != gt:
        print(f"  *** MISMATCH ***")


def run_tests():
    import sys
    sys.path.insert(0, "src")
    from dwipada.core.aksharanusarika import split_aksharalu

    assembler = SyllableAssembler()

    test_cases = [
        # Basic syllables
        ("Simple consonants",                "నమక"),
        ("Consonant + matra",                "నమస్కారం"),
        ("Long vowel matra",                 "రాముడు"),
        # Conjuncts
        ("Single conjunct",                  "స్కారం"),
        ("Triple conjunct",                  "స్త్రీ"),
        ("Conjunct + matra",                 "కృష్ణ"),
        # Pollu (trailing virama)
        ("Pollu at end of word",             "పూసెన్"),
        ("Pollu before space",               "పదిన్ కాలం"),
        ("Pollu at end of input",            "ఆదిన్"),
        # Independent vowels
        ("Independent vowels",               "అఆఇఈ"),
        ("Vowel + anusvara",                 "అందం"),
        # Anusvara / visarga
        ("Anusvara mid-word",                "సందడి"),
        ("Visarga",                          "దుఃఖము"),
        # Space and newlines
        ("Two words with space",             "తెలుగు భాష"),
        ("Two lines",                        "ఆదిన్\nమదిన్"),
        # Complex
        ("Mixed sentence",                   "శ్రీరాముడు జయం"),
        ("Multiple pollus",                  "పూసెన్ నాటిన్"),
        ("Anusvara after conjunct",          "సంస్కృతం"),
        ("Gana test phrase",                 "అమల"),
        ("Deergham",                         "రాముడు"),
        ("Before dvithvam",                  "అమ్మ"),
        ("Before samyuktham",                "సత్యము"),
        ("Contains au + sunna",              "గౌరవం"),
        ("Contains ai",                      "సైనికుడు"),
        ("Sunna mid",                        "సందడి"),
        ("Visarga + before conjunct",        "దుఃఖము"),
        ("Ends with halant",                 "పూసెన్"),
        ("Vocalic R vowel",                  "కృషి"),
    ]

    passed = 0
    failed = 0

    for i, (desc, text) in enumerate(test_cases, 1):
        result, trace = assembler.process_with_trace(text)
        gt = split_aksharalu(text)
        match = result == gt
        _print_trace(desc, text, trace, result, gt, i)
        if match:
            passed += 1
        else:
            failed += 1

    print()
    print("=" * 80)
    print(f"SUMMARY: {passed} passed, {failed} failed out of {passed + failed} tests")
    print("=" * 80)
    return failed == 0


if __name__ == "__main__":
    ok = run_tests()
    raise SystemExit(0 if ok else 1)
