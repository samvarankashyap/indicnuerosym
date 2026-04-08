# -*- coding: utf-8 -*-
"""
Guru/Laghu Classifier FST for Kannada.

Reads a stream of syllables (output from SyllableAssembler) and classifies
each syllable as Guru (U) or Laghu (I).

Six classification rules:

    Rule 1 (Deergham)  — syllable contains a long matra {ಾ ೀ ೂ ೇ ೋ ೌ}
                         OR is a standalone long independent vowel {ಆ ಈ ಊ ಏ ಓ}  →  U
    Rule 2 (Pluta)     — syllable contains ಐ / ಔ (independent) or
                         ೈ / ೌ (dependent)                                      →  U
    Rule 3 (Diacritic) — syllable contains anusvara (ಂ) or visarga (ಃ)         →  U
    Rule 4 (Pollu)     — syllable ends with halant / virama (್)                →  U
    Rule 5 (Conjunct)  — *next* syllable in the same word contains a C+್+C
                         pattern (conjunct or doubled consonant)                →  U
                         BLOCKED by a word boundary (SPACE or NEWLINE).
    (default)          — none of the above                                      →  I

Rules 1–4 are intrinsic: decided the moment the syllable arrives.
Rule 5 requires a 1-syllable lookahead.

States:
    EMPTY      — nothing buffered; ready for next syllable
    PENDING_I  — one intrinsically Laghu syllable in buffer,
                 waiting to see whether next syllable is a conjunct

Adapted from nfa_for_dwipada/guru_laghu_classifier.py (Telugu version).
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from syllable_assembler import (
    KANNADA_CONSONANTS,
    VIRAMA,
    SyllableAssembler,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LONG_MATRAS: frozenset[str] = frozenset({"ಾ", "ೀ", "ೂ", "ೇ", "ೋ", "ೌ"})

INDEPENDENT_LONG_VOWELS: frozenset[str] = frozenset({"ಆ", "ಈ", "ಊ", "ಏ", "ಓ"})

BOUNDARIES: frozenset[str] = frozenset({" ", "\n"})

STATE_EMPTY     = "EMPTY"
STATE_PENDING_I = "PENDING_I"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def intrinsic_label(syl: str) -> str:
    """
    Classify a syllable as 'U' (Guru) or 'I' (Laghu) using intrinsic rules only
    (Rules 1–4).  Rule 5 — syllable before conjunct — is handled by the FST.
    """
    if any(c in LONG_MATRAS for c in syl):          return "U"   # Rule 1a — long matra
    if syl in INDEPENDENT_LONG_VOWELS:               return "U"   # Rule 1b — long indep vowel
    if "ಐ" in syl or "ಔ" in syl:                    return "U"   # Rule 2  — pluta (indep)
    if "ೈ" in syl or "ೌ" in syl:                    return "U"   # Rule 2  — pluta (dep)
    if "ಂ" in syl:                                   return "U"   # Rule 3  — anusvara
    if "ಃ" in syl:                                   return "U"   # Rule 3  — visarga
    if syl.endswith("್"):                            return "U"   # Rule 4  — pollu (trailing halant)
    return "I"


def is_conjunct_trigger(syl: str) -> bool:
    """
    Return True if *syl* contains a C+್+C pattern — a conjunct (C1≠C2) or
    doubled consonant (C1==C2).  This is the trigger for Rule 5 on the
    preceding syllable.
    """
    for i in range(len(syl) - 2):
        if (syl[i]     in KANNADA_CONSONANTS
                and syl[i + 1] == VIRAMA
                and syl[i + 2] in KANNADA_CONSONANTS):
            return True
    return False


def _rule_note(syl: str, promoted_by_rule5: bool = False) -> str:
    if promoted_by_rule5:                                          return "Rule 5 (conjunct follows)"
    if any(c in LONG_MATRAS for c in syl):                        return "Rule 1 (long matra)"
    if syl in INDEPENDENT_LONG_VOWELS:                            return "Rule 1 (long indep vowel)"
    if "ಐ" in syl or "ಔ" in syl or "ೈ" in syl or "ೌ" in syl:    return "Rule 2 (pluta)"
    if "ಂ" in syl:                                                 return "Rule 3 (anusvara)"
    if "ಃ" in syl:                                                 return "Rule 3 (visarga)"
    if syl.endswith("್"):                                          return "Rule 4 (pollu)"
    return "—"


# ---------------------------------------------------------------------------
# FST
# ---------------------------------------------------------------------------

class GuruLaghuClassifier:
    """
    FST-based Kannada Guru/Laghu classifier.

    Receives output from SyllableAssembler (syllables + boundaries) and emits
    ``(syllable_text, label)`` pairs where label is ``'U'`` (Guru) or ``'I'``
    (Laghu).

    Usage::

        asm = SyllableAssembler()
        clf = GuruLaghuClassifier()
        labels = clf.process(asm.process("ಕನ್ನಡ"))
        # → [('ಕ', 'U'), ('ನ್ನ', 'I'), ('ಡ', 'I')]
    """

    def __init__(self):
        self._reset()

    def _reset(self):
        self.state:       str       = STATE_EMPTY
        self.buffer_syl:  str | None = None
        self.output: list[tuple[str, str]] = []

    def snapshot(self):
        return (self.state, self.buffer_syl)

    def restore(self, snap):
        self.state, self.buffer_syl = snap
        self.output = []

    def _emit(self, syl: str, label: str):
        self.output.append((syl, label))

    def _flush_buffer(self, rule5_applies: bool) -> list[tuple[str, str]]:
        if self.state == STATE_PENDING_I and self.buffer_syl is not None:
            label = "U" if rule5_applies else "I"
            self._emit(self.buffer_syl, label)
            released = [(self.buffer_syl, label)]
            self.buffer_syl = None
            self.state      = STATE_EMPTY
            return released
        return []

    def _on_syllable(self, syl: str) -> list[tuple[str, str]]:
        is_ct = is_conjunct_trigger(syl)
        emitted = self._flush_buffer(rule5_applies=is_ct)

        label = intrinsic_label(syl)

        if label == "U":
            self._emit(syl, "U")
            emitted.append((syl, "U"))
            self.state = STATE_EMPTY
        else:
            self.buffer_syl = syl
            self.state      = STATE_PENDING_I

        return emitted

    def _on_boundary(self) -> list[tuple[str, str]]:
        return self._flush_buffer(rule5_applies=False)

    def flush(self) -> list[tuple[str, str]]:
        return self._flush_buffer(rule5_applies=False)

    def process(
        self, syllables_with_boundaries: list[str]
    ) -> list[tuple[str, str]]:
        self._reset()
        for item in syllables_with_boundaries:
            if item in BOUNDARIES:
                self._on_boundary()
            else:
                self._on_syllable(item)
        self.flush()
        return self.output

    def process_text(self, text: str) -> list[tuple[str, str]]:
        return self.process(SyllableAssembler().process(text))

    def process_with_trace(
        self, syllables_with_boundaries: list[str]
    ) -> tuple[list[tuple[str, str]], list[dict]]:
        self._reset()
        trace: list[dict] = []

        items = list(syllables_with_boundaries) + [None]

        for item in items:
            state_before  = self.state
            buffer_before = self.buffer_syl or ""

            if item is None:
                released    = self.flush()
                item_type   = "flush"
                intrinsic_v = None
                is_ct       = False
            elif item in BOUNDARIES:
                released    = self._on_boundary()
                item_type   = "boundary"
                intrinsic_v = None
                is_ct       = False
            else:
                is_ct       = is_conjunct_trigger(item)
                intrinsic_v = intrinsic_label(item)
                released    = self._on_syllable(item)
                item_type   = "syllable"

            emitted_annotated: list[tuple[str, str, str]] = []
            for syl, lbl in released:
                if lbl == "U":
                    promoted = (
                        syl == buffer_before
                        and buffer_before != ""
                        and intrinsic_label(syl) == "I"
                    )
                    note = _rule_note(syl, promoted_by_rule5=promoted)
                else:
                    note = "Laghu"
                emitted_annotated.append((syl, lbl, note))

            trace.append({
                "item":          item if item is not None else "∎",
                "item_type":     item_type,
                "intrinsic":     intrinsic_v,
                "is_conjunct":   is_ct,
                "state_before":  state_before,
                "buffer_before": buffer_before,
                "state_after":   self.state,
                "buffer_after":  self.buffer_syl or "",
                "emitted":       emitted_annotated,
            })

        return self.output, trace


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def run_tests():
    asm = SyllableAssembler()
    clf = GuruLaghuClassifier()

    test_cases = [
        ("All Laghu",              "ಕರಿ",              "I I"),
        ("Long matra (ಾ)",        "ಕಾಲ",              "U I"),
        ("Long indep vowel (ಆ)",  "ಆಗ",               "U I"),
        ("Diphthong ಐ",           "ಐದು",              "U I"),
        ("Anusvara (ಂ)",          "ಸಂತಸ",             "U I I"),
        ("Visarga (ಃ)",           "ದುಃಖ",              "U I"),
        ("Before conjunct",        "ಕೃಷ್ಣ",            "U I"),
        ("Before doubled",         "ಕನ್ನಡ",            "U I I"),
        ("Space blocks Rule 5",   "ಕನ ಕೃಷಿ",          "I I I I"),
        ("Ragale word: ಕರಿಯಾ",    "ಕರಿಯಾ",            "I I U"),
        ("Full ragale line",       "ಜಲದಾ ಮಣಿಯೂ ಮುದದೀ ನಲಿಯೇ",
                                   "I I U I I U I I U I I U"),
        ("Mixed III+IIU",          "ಚೆಲುವ ಮುಡಿಯ ಕುಸುಮ ಸೆಳೆಯೂ",
                                   "I I I I I I I I I I I U"),
    ]

    passed = 0
    failed = 0

    for i, (desc, text, expected_str) in enumerate(test_cases, 1):
        expected = expected_str.split()
        syllables = asm.process(text)
        result = clf.process(syllables)
        our_labels = [lbl for _, lbl in result]
        match = our_labels == expected
        status = "PASS" if match else "FAIL"
        syl_label = " ".join(f"{s}={l}" for s, l in result)
        print(f"  Test {i:2d}: {desc:<30s}  [{status}]  {syl_label}")
        if not match:
            print(f"           Expected: {' '.join(expected)}")
            print(f"           Got:      {' '.join(our_labels)}")
            failed += 1
        else:
            passed += 1

    print()
    print(f"SUMMARY: {passed} passed, {failed} failed out of {passed + failed} tests")
    return failed == 0


if __name__ == "__main__":
    ok = run_tests()
    raise SystemExit(0 if ok else 1)
