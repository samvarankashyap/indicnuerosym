# -*- coding: utf-8 -*-
"""
Ādi Prāsa NFA for Kannada Utsaha Ragale.
==========================================

Validates the Ādi Prāsa rhyme constraint of a two-line Ragale couplet.

The rule:
    The **2nd aksharam** (syllable) of every line in a couplet must share the
    same **base consonant**.

"Base consonant" means the first Kannada consonant character of the syllable,
ignoring vowel marks (matras), conjunct extensions, anusvara, and visarga.

Optionally, two consonant pairs can be treated as equivalent:
    ಲ ↔ ಳ   (laterals)
    ಶ ↔ ಷ ↔ ಸ  (sibilants)

This is Stage 4 of the NFA pipeline:

    Raw text
       |
       v
    [SyllableAssembler FST]   -- Stage 1: chars -> syllables
       |
       v
    [GuruLaghuClassifier FST] -- Stage 2: syllables -> (syl, U/I)
       |
       v
    [GanaNFA]                 -- Stage 3: U/I -> gana partition
    [PrasaNFA]                -- Stage 4: syllable stream -> prasa check (THIS)

NFA State Machine:

    LINE1_SYL0 --[syl]--> LINE1_SYL1 --[syl, store consonant]--> LINE1_REST
                                                                       |
                                                                  [newline]
                                                                       v
    LINE2_SYL0 --[syl]--> LINE2_SYL1 --[syl, check match]--> ACCEPT / REJECT

Adapted from nfa_for_dwipada/prasa_nfa.py (Telugu version).
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from syllable_assembler import KANNADA_CONSONANTS, SyllableAssembler
from guru_laghu_classifier import GuruLaghuClassifier


###############################################################################
# 2) PRASA CONSTANTS
###############################################################################

PRASA_EQUIVALENTS = [
    frozenset({"ಲ", "ಳ"}),         # laterals
    frozenset({"ಶ", "ಷ", "ಸ"}),    # sibilants
]

CONSONANT_TO_CLASS: dict[str, frozenset[str]] = {}
for _group in PRASA_EQUIVALENTS:
    for _c in _group:
        CONSONANT_TO_CLASS[_c] = _group

CLASS_NAMES: dict[frozenset[str], str] = {
    frozenset({"ಲ", "ಳ"}):      "lateral",
    frozenset({"ಶ", "ಷ", "ಸ"}): "sibilant",
}

# NFA state constants
LINE1_SYL0 = "LINE1_SYL0"
LINE1_SYL1 = "LINE1_SYL1"
LINE1_REST = "LINE1_REST"
LINE2_SYL0 = "LINE2_SYL0"
LINE2_SYL1 = "LINE2_SYL1"
ACCEPT     = "ACCEPT"
REJECT     = "REJECT"

SKIP_ITEMS = frozenset({" "})


###############################################################################
# 3) PURE HELPER FUNCTIONS
###############################################################################


def get_base_consonant(aksharam: str) -> str | None:
    """Extract the base (first) consonant from a Kannada syllable."""
    if not aksharam:
        return None
    first_char = aksharam[0]
    if first_char in KANNADA_CONSONANTS:
        return first_char
    return None


def get_consonant_class(consonant: str) -> str:
    group = CONSONANT_TO_CLASS.get(consonant)
    if group is not None:
        return CLASS_NAMES[group]
    return consonant


def are_prasa_equivalent(c1: str | None, c2: str | None, strict: bool = False) -> bool:
    """Check whether two consonants satisfy prasa.

    Args:
        c1: Base consonant from line 1's 2nd aksharam (or None).
        c2: Base consonant from line 2's 2nd aksharam (or None).
        strict: If True, only exact match counts. If False, equivalence groups apply.
    """
    if c1 is None or c2 is None:
        return False
    if c1 == c2:
        return True
    if strict:
        return False
    group = CONSONANT_TO_CLASS.get(c1)
    return group is not None and c2 in group


def classify_match(c1: str | None, c2: str | None) -> tuple[bool, str, frozenset | None]:
    if c1 is None or c2 is None:
        return (False, "no_consonant", None)
    if c1 == c2:
        return (True, "exact", None)
    group = CONSONANT_TO_CLASS.get(c1)
    if group is not None and c2 in group:
        return (True, "equivalent", group)
    return (False, "mismatch", None)


###############################################################################
# 4) PrasaNFA CLASS
###############################################################################


class PrasaNFA:
    """
    NFA-based Ādi Prāsa validator for Kannada Ragale couplets.

    Three interfaces:
    1. ``process(poem)`` — full pipeline from raw Kannada text.
    2. ``process_syllables(line1, line2)`` — from pre-split syllable lists.
    3. ``feed(item)`` / ``flush()`` — streaming, one item at a time.
    """

    def __init__(self, strict: bool = False):
        self.strict = strict
        self._reset()

    def _reset(self):
        self.state = LINE1_SYL0
        self.line1_aksharalu: list[str] = []
        self.line2_aksharalu: list[str] = []
        self.prasa_consonant: str | None = None
        self.prasa_aksharam1: str | None = None
        self.prasa_aksharam2: str | None = None
        self.consonant2: str | None = None
        self.rejection_reason: str | None = None

    def feed(self, item: str):
        if item in SKIP_ITEMS:
            return
        if item == "\n":
            self._on_newline()
            return
        if self.state in (ACCEPT, REJECT):
            if self.state == ACCEPT:
                self.line2_aksharalu.append(item)
            return

        if self.state == LINE1_SYL0:
            self.line1_aksharalu.append(item)
            self.state = LINE1_SYL1
        elif self.state == LINE1_SYL1:
            self.line1_aksharalu.append(item)
            self.prasa_aksharam1 = item
            self.prasa_consonant = get_base_consonant(item)
            self.state = LINE1_REST
        elif self.state == LINE1_REST:
            self.line1_aksharalu.append(item)
        elif self.state == LINE2_SYL0:
            self.line2_aksharalu.append(item)
            self.state = LINE2_SYL1
        elif self.state == LINE2_SYL1:
            self.line2_aksharalu.append(item)
            self.prasa_aksharam2 = item
            self.consonant2 = get_base_consonant(item)
            if are_prasa_equivalent(self.prasa_consonant, self.consonant2, self.strict):
                self.state = ACCEPT
            else:
                self.state = REJECT
                self.rejection_reason = (
                    f"Consonant mismatch: line1={self.prasa_consonant} "
                    f"vs line2={self.consonant2}"
                )

    def _on_newline(self):
        if self.state in (LINE1_SYL0, LINE1_SYL1, LINE1_REST):
            self.state = LINE2_SYL0

    def flush(self):
        if self.state not in (ACCEPT, REJECT):
            self.state = REJECT
            self.rejection_reason = "Incomplete: not enough syllables or lines"

    def _result(self) -> dict:
        is_match, match_type, equiv_group = classify_match(
            self.prasa_consonant, self.consonant2
        )
        return {
            "is_valid": self.state == ACCEPT,
            "match_type": match_type,
            "line1_aksharam2": self.prasa_aksharam1,
            "line2_aksharam2": self.prasa_aksharam2,
            "line1_consonant": self.prasa_consonant,
            "line2_consonant": self.consonant2,
            "equivalence_group": sorted(equiv_group) if equiv_group else None,
            "rejection_reason": self.rejection_reason,
        }

    def process(self, poem: str) -> dict:
        self._reset()
        asm = SyllableAssembler()
        syllables = asm.process(poem)
        for item in syllables:
            self.feed(item)
        self.flush()
        return self._result()

    def process_syllables(self, line1: list[str], line2: list[str]) -> dict:
        self._reset()
        for syl in line1:
            self.feed(syl)
        self.feed("\n")
        for syl in line2:
            self.feed(syl)
        self.flush()
        return self._result()

    def process_with_trace(self, poem: str) -> tuple[dict, list[dict]]:
        self._reset()
        asm = SyllableAssembler()
        syllables = asm.process(poem)
        trace = []

        items = list(syllables) + [None]
        for item in items:
            state_before = self.state
            if item is None:
                self.flush()
            else:
                self.feed(item)
            trace.append({
                "item": item if item is not None else "∎",
                "state_before": state_before,
                "state_after": self.state,
                "prasa_consonant": self.prasa_consonant,
            })

        return self._result(), trace

    def process_with_trace_raw(self, syllables_with_boundaries: list[str]) -> tuple[dict, list[dict]]:
        self._reset()
        trace = []
        items = list(syllables_with_boundaries) + [None]
        for item in items:
            state_before = self.state
            if item is None:
                self.flush()
            else:
                self.feed(item)
            trace.append({
                "item": item if item is not None else "∎",
                "state_before": state_before,
                "state_after": self.state,
                "prasa_consonant": self.prasa_consonant,
            })
        return self._result(), trace


###############################################################################
# 5) TESTS
###############################################################################

def run_tests():
    test_cases = [
        ("Matching prasa: ಲ",
         "ಜಲದಾ ಮಣಿಯೂ ಮುದದೀ ನಲಿಯೇ\nನಿಲದೇ ಒಡೆದೂ ಮರೆಯಾಗುವುದೂ",
         True, "ಲ", "ಲ"),

        ("Matching prasa: ರ",
         "ನುರೆಯಾ ನಡುವೇ ಮಿನುಗೂ ಮಣಿಯೂ\nಕರಗೀ ಕೊನೆಗೂ ಸಿಡಿದೂ ಅಳಿದೂ",
         True, "ರ", "ರ"),

        ("Matching prasa: ಡ",
         "ಒಡಲೀ ಉಸಿರೂ ತುಳುಕೀ ಕುಣಿತಾ\nಒಡೆದೂ ಜಲದೀ ಬೆರೆತೂ ಅಳಿದೂ",
         True, "ಡ", "ಡ"),

        ("Mismatched prasa",
         "ಜಲದಾ ಮಣಿಯೂ ಮುದದೀ ನಲಿಯೇ\nಕರಗೀ ಕೊನೆಗೂ ಸಿಡಿದೂ ಅಳಿದೂ",
         False, "ಲ", "ರ"),

        ("Matching prasa: ಮ (test_poems3 poem 1)",
         "ಕಮಲ ಮುಖದ ಬದಿಯ ಕುರುಳು ನಲಿದೂ\nವಿಮಲ ಮತಿಯ ಒಲವ ಸುರುಳಿ ಬೆಳೆದೂ",
         True, "ಮ", "ಮ"),
    ]

    nfa = PrasaNFA()
    passed = 0
    failed = 0

    for i, (desc, poem, exp_valid, exp_c1, exp_c2) in enumerate(test_cases, 1):
        result = nfa.process(poem)
        match = (result["is_valid"] == exp_valid
                 and result["line1_consonant"] == exp_c1
                 and result["line2_consonant"] == exp_c2)
        status = "PASS" if match else "FAIL"
        print(f"  Test {i:2d}: {desc:<40s}  [{status}]  "
              f"valid={result['is_valid']} c1={result['line1_consonant']} c2={result['line2_consonant']}")
        if not match:
            print(f"           Expected: valid={exp_valid} c1={exp_c1} c2={exp_c2}")
            failed += 1
        else:
            passed += 1

    print()
    print(f"SUMMARY: {passed} passed, {failed} failed out of {passed + failed} tests")
    return failed == 0


if __name__ == "__main__":
    ok = run_tests()
    raise SystemExit(0 if ok else 1)
