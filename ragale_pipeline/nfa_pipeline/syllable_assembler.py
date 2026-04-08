# -*- coding: utf-8 -*-
"""
Syllable Assembler FST for Kannada.

Reads a stream of Unicode characters and emits complete syllables (aksharalu),
spaces, and newlines — adapted from the Telugu SyllableAssembler in
nfa_for_dwipada/syllable_assembler.py.

States:
    IDLE               - No syllable buffer active
    CONSONANT_CLUSTER  - Building a consonant-based syllable
    PENDING_VIRAMA     - Buffer ends with [C + ್], waiting to see if conjunct or pollu
    VOWEL              - Building an independent-vowel syllable

The state machine is structurally identical to the Telugu version because
Kannada and Telugu share the same Brahmic script mechanics (consonant-vowel-
virama model).  Only the Unicode codepoint sets differ.
"""

# ---------------------------------------------------------------------------
# Codepoint categories — Kannada Unicode
# ---------------------------------------------------------------------------

KANNADA_CONSONANTS = {
    "ಕ", "ಖ", "ಗ", "ಘ", "ಙ", "ಚ", "ಛ", "ಜ", "ಝ", "ಞ",
    "ಟ", "ಠ", "ಡ", "ಢ", "ಣ", "ತ", "ಥ", "ದ", "ಧ", "ನ",
    "ಪ", "ಫ", "ಬ", "ಭ", "ಮ", "ಯ", "ರ", "ಲ", "ವ", "ಶ",
    "ಷ", "ಸ", "ಹ", "ಳ",
}
INDEPENDENT_VOWELS = {
    "ಅ", "ಆ", "ಇ", "ಈ", "ಉ", "ಊ", "ಋ",
    "ಎ", "ಏ", "ಐ", "ಒ", "ಓ", "ಔ",
}
MATRAS = {"ಾ", "ಿ", "ೀ", "ು", "ೂ", "ೃ", "ೆ", "ೇ", "ೈ", "ೊ", "ೋ", "ೌ"}
VIRAMA = "್"
DIACRITICS = {"ಂ", "ಃ"}   # anusvara, visarga
SKIP_CHARS = {"\u200c", "\u200b"}  # ZWNJ, ZWSP


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
    if ch in KANNADA_CONSONANTS:  return CAT_CONSONANT
    if ch in INDEPENDENT_VOWELS:  return CAT_INDEP_VOWEL
    if ch in MATRAS:              return CAT_MATRA
    if ch == VIRAMA:              return CAT_VIRAMA
    if ch in DIACRITICS:          return CAT_DIACRITIC
    if ch == " ":                 return CAT_SPACE
    if ch == "\n":                return CAT_NEWLINE
    if ch in SKIP_CHARS:          return CAT_SKIP
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
    FST-based Kannada syllable assembler.

    Usage:
        asm = SyllableAssembler()
        syllables = asm.process("ಕನ್ನಡ")
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
        s = self._buffer_str()
        if s:
            self.output.append(s)
            self.prev_syllable = s
        self.buffer = []

    def _emit_pollu_merge(self):
        pollu = self._buffer_str()
        if self.prev_syllable is not None:
            if self.output and self.output[-1] == self.prev_syllable:
                merged = self.prev_syllable + pollu
                self.output[-1] = merged
                self.prev_syllable = merged
            else:
                self.output.append(pollu)
                self.prev_syllable = pollu
        else:
            self.output.append(pollu)
            self.prev_syllable = pollu
        self.buffer = []

    def _emit_boundary(self, ch: str):
        self.output.append(ch)
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
            if self.prev_syllable is not None and self.output and self.output[-1] == self.prev_syllable:
                merged = self.prev_syllable + ch
                self.output[-1] = merged
                self.prev_syllable = merged
            else:
                self.output.append(ch)
                self.prev_syllable = ch
        elif cat in (CAT_SKIP, CAT_OTHER, CAT_MATRA, CAT_VIRAMA):
            pass

    def _from_consonant_cluster(self, ch: str, cat: str):
        if cat == CAT_VIRAMA:
            self.buffer.append(ch)
            self.state = STATE_PENDING_VIRAMA
        elif cat in (CAT_MATRA, CAT_DIACRITIC):
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
            pass
        else:
            self._emit_syllable()
            self.state = STATE_IDLE

    def _from_pending_virama(self, ch: str, cat: str):
        if cat == CAT_CONSONANT:
            self.buffer.append(ch)
            self.state = STATE_CONSONANT_CLUSTER
        elif cat in (CAT_MATRA, CAT_DIACRITIC):
            self.buffer.append(ch)
            self._emit_syllable()
            self.state = STATE_IDLE
        elif cat in (CAT_SPACE, CAT_NEWLINE):
            self._emit_pollu_merge()
            self._emit_boundary(ch)
            self.state = STATE_IDLE
        elif cat == CAT_INDEP_VOWEL:
            self._emit_pollu_merge()
            self.buffer = [ch]
            self.state = STATE_VOWEL
        elif cat == CAT_SKIP:
            pass
        else:
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
            pass
        else:
            self._emit_syllable()
            self.state = STATE_IDLE

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def feed(self, ch: str):
        cat = classify(ch)
        if self.state == STATE_IDLE:
            self._from_idle(ch, cat)
        elif self.state == STATE_CONSONANT_CLUSTER:
            self._from_consonant_cluster(ch, cat)
        elif self.state == STATE_PENDING_VIRAMA:
            self._from_pending_virama(ch, cat)
        elif self.state == STATE_VOWEL:
            self._from_vowel(ch, cat)

    def snapshot(self):
        return (self.state, list(self.buffer), self.prev_syllable)

    def restore(self, snap):
        self.state, self.buffer, self.prev_syllable = snap[0], list(snap[1]), snap[2]
        self.output = []

    def flush(self):
        if self.state == STATE_CONSONANT_CLUSTER:
            self._emit_syllable()
        elif self.state == STATE_PENDING_VIRAMA:
            self._emit_pollu_merge()
        elif self.state == STATE_VOWEL:
            self._emit_syllable()
        self.state = STATE_IDLE

    def process(self, text: str) -> list[str]:
        self._reset()
        for ch in text:
            self.feed(ch)
        self.flush()
        return self.output

    def process_with_trace(self, text: str) -> tuple[list[str], list[dict]]:
        self._reset()
        trace: list[dict] = []

        chars = list(text) + [None]
        for ch in chars:
            state_before = self.state
            buffer_before = self._buffer_str()
            output_len_before = len(self.output)
            last_out_before = self.output[-1] if self.output else None

            if ch is None:
                self.flush()
                cat = "END"
                cp = "—"
            else:
                self.feed(ch)
                cat = classify(ch)
                cp = f"U+{ord(ch):04X}"

            new_entries = list(self.output[output_len_before:])

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
# Tests — verified against split_aksharalu() from kannada_ragale_analyser.py
# ---------------------------------------------------------------------------

def run_tests():
    assembler = SyllableAssembler()

    test_cases = [
        ("Simple consonants",       "ನಮಕ",                    ["ನ", "ಮ", "ಕ"]),
        ("Consonant + matra",       "ಕರಿಯಾ",                  ["ಕ", "ರಿ", "ಯಾ"]),
        ("Conjunct",                "ಕೃಷ್ಣ",                  ["ಕೃ", "ಷ್ಣ"]),
        ("Independent vowels",      "ಅಆಇ",                    ["ಅ", "ಆ", "ಇ"]),
        ("Anusvara",                "ಸಂತಸ",                   ["ಸಂ", "ತ", "ಸ"]),
        ("Full ragale word",        "ಮುಗಿಲಾ",                 ["ಮು", "ಗಿ", "ಲಾ"]),
        ("Two words with space",    "ಕರಿಯಾ ಕುರುಳೂ",           ["ಕ", "ರಿ", "ಯಾ", " ", "ಕು", "ರು", "ಳೂ"]),
        ("Newline boundary",        "ಜಲದಾ\nನಿಲದೇ",           ["ಜ", "ಲ", "ದಾ", "\n", "ನಿ", "ಲ", "ದೇ"]),
        ("Long vowels only",        "ಆಈಊ",                    ["ಆ", "ಈ", "ಊ"]),
        ("Visarga",                 "ದುಃಖ",                   ["ದುಃ", "ಖ"]),
        ("Ragale line",             "ಜಲದಾ ಮಣಿಯೂ ಮುದದೀ ನಲಿಯೇ",
         ["ಜ", "ಲ", "ದಾ", " ", "ಮ", "ಣಿ", "ಯೂ", " ", "ಮು", "ದ", "ದೀ", " ", "ನ", "ಲಿ", "ಯೇ"]),
    ]

    passed = 0
    failed = 0

    for i, (desc, text, expected) in enumerate(test_cases, 1):
        result = assembler.process(text)
        match = result == expected
        status = "PASS" if match else "FAIL"
        print(f"  Test {i:2d}: {desc:<30s}  [{status}]  {result}")
        if not match:
            print(f"           Expected: {expected}")
            failed += 1
        else:
            passed += 1

    print()
    print(f"SUMMARY: {passed} passed, {failed} failed out of {passed + failed} tests")
    return failed == 0


if __name__ == "__main__":
    ok = run_tests()
    raise SystemExit(0 if ok else 1)
