# -*- coding: utf-8 -*-
"""
Composite Pipeline State for Logit Masking.
============================================

Wraps all pipeline stages (SyllableAssembler + GuruLaghuClassifier + GanaNFA +
PrasaNFA) into a single incremental state object that supports fast
snapshot/clone for computing token validity masks.

Pipeline chain (inlined in feed_char):

    Unicode char
       → SyllableAssembler logic (may emit syllable or boundary)
       → GuruLaghuClassifier logic (may emit U/I marker)
       → GanaNFA _advance() (updates branches)
       → PrasaNFA (tracks 2nd syllable consonant)

Usage:

    state = CompositeState()
    for ch in "ಕನ್ನಡ":
        state.feed_char(ch)
    state.flush()
    print(state.syllable_count, state.is_alive())

    # For masking: snapshot, clone, simulate token
    snap = state.snapshot()
    clone = CompositeState.from_snapshot(snap)
    clone.feed_token_text("ಕಾ")
    if clone.is_alive():
        print("token is valid")

Adapted from nfa_for_dwipada/composite_state.py (Telugu version).
No yati — only gana + prasa.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from syllable_assembler import (
    KANNADA_CONSONANTS, INDEPENDENT_VOWELS, MATRAS, VIRAMA, DIACRITICS,
    SKIP_CHARS, classify,
    CAT_CONSONANT, CAT_INDEP_VOWEL, CAT_MATRA, CAT_VIRAMA, CAT_DIACRITIC,
    CAT_SPACE, CAT_NEWLINE, CAT_SKIP, CAT_OTHER,
    STATE_IDLE, STATE_CONSONANT_CLUSTER, STATE_PENDING_VIRAMA, STATE_VOWEL,
)
from guru_laghu_classifier import (
    intrinsic_label, is_conjunct_trigger,
    STATE_EMPTY, STATE_PENDING_I,
    BOUNDARIES,
)
from gana_nfa import (
    _advance, _spawn_slot, SLOT_ACCEPT,
    RAGALE_GANAS,
)
from prasa_nfa import get_base_consonant, are_prasa_equivalent


# Ragale: always 12 syllables per line (4 ganas × 3 syllables)
VALID_LINE_LENGTHS = {12}
MAX_LINE_LENGTH = 12


def _min_to_accept(slot, gana_name, sub_pos):
    """Minimum syllables needed from this branch state to reach ACCEPT."""
    if slot == SLOT_ACCEPT:
        return 0
    remaining_in_current = 3 - sub_pos  # all ragale ganas are 3 syllables
    remaining_slots = 3 - slot
    return remaining_in_current + remaining_slots * 3


def _max_to_accept(slot, gana_name, sub_pos):
    """Maximum syllables from this branch state to reach ACCEPT.
    For ragale this equals _min_to_accept since all ganas are 3 syllables.
    """
    return _min_to_accept(slot, gana_name, sub_pos)


class CompositeState:
    """Combined incremental pipeline state for Kannada Ragale.

    Inlines all three stages (assembler, classifier, NFA) and tracks
    prasa consonant matching for efficient logit masking.
    """

    def __init__(self):
        # -- SyllableAssembler state --
        self.asm_state = STATE_IDLE
        self.asm_buffer: list[str] = []
        self.asm_prev_syllable: str | None = None

        # -- GuruLaghuClassifier state --
        self.clf_state = STATE_EMPTY
        self.clf_buffer_syl: str | None = None

        # -- GanaNFA state --
        self.nfa_branches: set = _spawn_slot(0, ())
        self.syllable_count: int = 0
        self.lines_complete: int = 0

        # -- Line-level tracking --
        self.line_syllable_index: int = 0

        # -- Prasa state --
        self.line1_prasa_consonant: str | None = None
        self.line2_prasa_consonant: str | None = None
        self.prasa_state: str = "LINE1"  # LINE1, LINE2, DECIDED

    # ------------------------------------------------------------------
    # Syllable assembler (inlined)
    # ------------------------------------------------------------------

    def _asm_buffer_str(self) -> str:
        return "".join(self.asm_buffer)

    def _on_syllable_emitted(self, syl: str):
        """Route an emitted syllable through classifier and NFA."""
        self.line_syllable_index += 1

        # -- Track prasa at position 2 of each line --
        if self.line_syllable_index == 2:
            cons = get_base_consonant(syl)
            if self.prasa_state == "LINE1":
                self.line1_prasa_consonant = cons
            elif self.prasa_state == "LINE2":
                self.line2_prasa_consonant = cons
                self.prasa_state = "DECIDED"

        # -- Classifier: resolve buffer + classify new syllable --
        is_ct = is_conjunct_trigger(syl)

        if self.clf_state == STATE_PENDING_I and self.clf_buffer_syl is not None:
            label = "U" if is_ct else "I"
            self._on_marker(label)
            self.clf_buffer_syl = None
            self.clf_state = STATE_EMPTY

        label = intrinsic_label(syl)
        if label == "U":
            self._on_marker("U")
            self.clf_state = STATE_EMPTY
        else:
            self.clf_buffer_syl = syl
            self.clf_state = STATE_PENDING_I

        self.asm_prev_syllable = syl

    def _on_boundary(self):
        """Handle word boundary (space/newline) — flush classifier buffer."""
        if self.clf_state == STATE_PENDING_I and self.clf_buffer_syl is not None:
            self._on_marker("I")
            self.clf_buffer_syl = None
            self.clf_state = STATE_EMPTY

    def _on_newline(self):
        """Handle line break."""
        self._on_boundary()

        # Flush NFA line
        accepting = [b for b in self.nfa_branches if b[0] == SLOT_ACCEPT]
        if accepting:
            self.lines_complete += 1

        # Start new line
        self.nfa_branches = _spawn_slot(0, ())
        self.syllable_count = 0
        self.line_syllable_index = 0
        self.asm_prev_syllable = None

        if self.prasa_state == "LINE1":
            self.prasa_state = "LINE2"

    def _on_marker(self, marker: str):
        """Feed a U/I marker into the GanaNFA."""
        self.syllable_count += 1
        self.nfa_branches = _advance(self.nfa_branches, marker)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def feed_char(self, ch: str):
        """Feed a single Unicode character through the full pipeline."""
        cat = classify(ch)

        if self.asm_state == STATE_IDLE:
            if cat == CAT_CONSONANT:
                self.asm_buffer = [ch]
                self.asm_state = STATE_CONSONANT_CLUSTER
            elif cat == CAT_INDEP_VOWEL:
                self.asm_buffer = [ch]
                self.asm_state = STATE_VOWEL
            elif cat == CAT_SPACE:
                self._on_boundary()
            elif cat == CAT_NEWLINE:
                self._on_newline()
            elif cat == CAT_DIACRITIC:
                if self.asm_prev_syllable is not None:
                    self.asm_prev_syllable += ch
            # SKIP, OTHER, MATRA, VIRAMA: ignored

        elif self.asm_state == STATE_CONSONANT_CLUSTER:
            if cat == CAT_VIRAMA:
                self.asm_buffer.append(ch)
                self.asm_state = STATE_PENDING_VIRAMA
            elif cat in (CAT_MATRA, CAT_DIACRITIC):
                self.asm_buffer.append(ch)
                syl = self._asm_buffer_str()
                self.asm_buffer = []
                self.asm_state = STATE_IDLE
                self._on_syllable_emitted(syl)
            elif cat == CAT_CONSONANT:
                syl = self._asm_buffer_str()
                self.asm_buffer = [ch]
                self._on_syllable_emitted(syl)
            elif cat == CAT_INDEP_VOWEL:
                syl = self._asm_buffer_str()
                self.asm_buffer = [ch]
                self.asm_state = STATE_VOWEL
                self._on_syllable_emitted(syl)
            elif cat == CAT_SPACE:
                syl = self._asm_buffer_str()
                self.asm_buffer = []
                self.asm_state = STATE_IDLE
                self._on_syllable_emitted(syl)
                self._on_boundary()
            elif cat == CAT_NEWLINE:
                syl = self._asm_buffer_str()
                self.asm_buffer = []
                self.asm_state = STATE_IDLE
                self._on_syllable_emitted(syl)
                self._on_newline()
            elif cat == CAT_SKIP:
                pass
            else:
                syl = self._asm_buffer_str()
                self.asm_buffer = []
                self.asm_state = STATE_IDLE
                self._on_syllable_emitted(syl)

        elif self.asm_state == STATE_PENDING_VIRAMA:
            if cat == CAT_CONSONANT:
                self.asm_buffer.append(ch)
                self.asm_state = STATE_CONSONANT_CLUSTER
            elif cat in (CAT_MATRA, CAT_DIACRITIC):
                self.asm_buffer.append(ch)
                syl = self._asm_buffer_str()
                self.asm_buffer = []
                self.asm_state = STATE_IDLE
                self._on_syllable_emitted(syl)
            elif cat in (CAT_SPACE, CAT_NEWLINE):
                # Pollu merge
                pollu = self._asm_buffer_str()
                if self.asm_prev_syllable is not None:
                    self.asm_prev_syllable += pollu
                self.asm_buffer = []
                self.asm_state = STATE_IDLE
                if cat == CAT_SPACE:
                    self._on_boundary()
                else:
                    self._on_newline()
            elif cat == CAT_INDEP_VOWEL:
                pollu = self._asm_buffer_str()
                if self.asm_prev_syllable is not None:
                    self.asm_prev_syllable += pollu
                self.asm_buffer = [ch]
                self.asm_state = STATE_VOWEL
            elif cat == CAT_SKIP:
                pass
            else:
                pollu = self._asm_buffer_str()
                if self.asm_prev_syllable is not None:
                    self.asm_prev_syllable += pollu
                self.asm_buffer = []
                self.asm_state = STATE_IDLE

        elif self.asm_state == STATE_VOWEL:
            if cat == CAT_DIACRITIC:
                self.asm_buffer.append(ch)
                syl = self._asm_buffer_str()
                self.asm_buffer = []
                self.asm_state = STATE_IDLE
                self._on_syllable_emitted(syl)
            elif cat == CAT_CONSONANT:
                syl = self._asm_buffer_str()
                self.asm_buffer = [ch]
                self.asm_state = STATE_CONSONANT_CLUSTER
                self._on_syllable_emitted(syl)
            elif cat == CAT_INDEP_VOWEL:
                syl = self._asm_buffer_str()
                self.asm_buffer = [ch]
                self._on_syllable_emitted(syl)
            elif cat == CAT_SPACE:
                syl = self._asm_buffer_str()
                self.asm_buffer = []
                self.asm_state = STATE_IDLE
                self._on_syllable_emitted(syl)
                self._on_boundary()
            elif cat == CAT_NEWLINE:
                syl = self._asm_buffer_str()
                self.asm_buffer = []
                self.asm_state = STATE_IDLE
                self._on_syllable_emitted(syl)
                self._on_newline()
            elif cat == CAT_SKIP:
                pass
            else:
                syl = self._asm_buffer_str()
                self.asm_buffer = []
                self.asm_state = STATE_IDLE
                self._on_syllable_emitted(syl)

    def feed_token_text(self, token_text: str):
        for ch in token_text:
            self.feed_char(ch)

    def flush(self):
        # Flush assembler
        if self.asm_state == STATE_CONSONANT_CLUSTER:
            syl = self._asm_buffer_str()
            self.asm_buffer = []
            self.asm_state = STATE_IDLE
            self._on_syllable_emitted(syl)
        elif self.asm_state == STATE_PENDING_VIRAMA:
            pollu = self._asm_buffer_str()
            if self.asm_prev_syllable is not None:
                self.asm_prev_syllable += pollu
            self.asm_buffer = []
            self.asm_state = STATE_IDLE
        elif self.asm_state == STATE_VOWEL:
            syl = self._asm_buffer_str()
            self.asm_buffer = []
            self.asm_state = STATE_IDLE
            self._on_syllable_emitted(syl)

        # Flush classifier
        self._on_boundary()

    def is_alive(self) -> bool:
        """Return True if any NFA branch can still reach ACCEPT."""
        if not self.nfa_branches:
            return False

        remaining = MAX_LINE_LENGTH - self.syllable_count
        for b in self.nfa_branches:
            if b[0] == SLOT_ACCEPT:
                return True
            mn = _min_to_accept(b[0], b[1], b[2])
            mx = _max_to_accept(b[0], b[1], b[2])
            if mn <= remaining <= mx:
                return True
        return False

    def has_accept(self) -> bool:
        """Return True if any branch is in ACCEPT state."""
        return any(b[0] == SLOT_ACCEPT for b in self.nfa_branches)

    def prasa_alive(self) -> bool:
        """Check if prasa constraint is still satisfiable."""
        if self.prasa_state == "DECIDED":
            return are_prasa_equivalent(
                self.line1_prasa_consonant, self.line2_prasa_consonant
            )
        return True  # not yet decided

    # ------------------------------------------------------------------
    # Snapshot / Clone
    # ------------------------------------------------------------------

    def snapshot(self):
        return (
            self.asm_state,
            list(self.asm_buffer),
            self.asm_prev_syllable,
            self.clf_state,
            self.clf_buffer_syl,
            frozenset(self.nfa_branches),
            self.syllable_count,
            self.lines_complete,
            self.line_syllable_index,
            self.line1_prasa_consonant,
            self.line2_prasa_consonant,
            self.prasa_state,
        )

    @classmethod
    def from_snapshot(cls, snap):
        obj = cls.__new__(cls)
        (
            obj.asm_state,
            buf,
            obj.asm_prev_syllable,
            obj.clf_state,
            obj.clf_buffer_syl,
            branches_frozen,
            obj.syllable_count,
            obj.lines_complete,
            obj.line_syllable_index,
            obj.line1_prasa_consonant,
            obj.line2_prasa_consonant,
            obj.prasa_state,
        ) = snap
        obj.asm_buffer = list(buf)
        obj.nfa_branches = set(branches_frozen)
        return obj


###############################################################################
# MASK BUILDING & TOKEN EXTRACTION
###############################################################################


def build_gana_mask(snapshot, kannada_token_ids, kannada_token_texts):
    """
    Build the set of valid token IDs given the current composite state.

    For each Kannada token, clones the state, simulates feeding
    the token's characters, and checks if the NFA is still alive.

    Args:
        snapshot: tuple from CompositeState.snapshot()
        kannada_token_ids: list of int, the Kannada token IDs
        kannada_token_texts: list of str, pre-decoded text for each token

    Returns:
        set of valid token IDs
    """
    valid = set()
    for tid, text in zip(kannada_token_ids, kannada_token_texts):
        clone = CompositeState.from_snapshot(snapshot)
        clone.feed_token_text(text)
        clone.flush()
        if clone.is_alive():
            valid.add(tid)
    return valid


def get_kannada_token_set(tokenizer):
    """
    Extract the set of Kannada-capable tokens from a tokenizer.

    Returns:
        (kannada_token_ids, kannada_token_texts): parallel lists
    """
    ids = []
    texts = []
    for token_id in range(tokenizer.vocab_size):
        text = tokenizer.decode([token_id])
        if _is_kannada_token(text):
            ids.append(token_id)
            texts.append(text)
    return ids, texts


def _is_kannada_token(text):
    """Check if a token text contains only Kannada characters and whitespace."""
    if not text:
        return False
    for ch in text:
        cp = ord(ch)
        if (0x0C80 <= cp <= 0x0CFF   # Kannada Unicode block
                or ch in " \n:"):      # space, newline, colon
            continue
        return False
    return True


###############################################################################
# TESTS
###############################################################################

def run_tests():
    test_cases = [
        ("Feed full valid line",
         "ಜಲದಾ ಮಣಿಯೂ ಮುದದೀ ನಲಿಯೇ",
         12, True),

        ("Feed 6 syllables",
         "ಜಲದಾ ಮಣಿಯೂ",
         6, True),

        ("All laghu — no guru ending possible from slot 3",
         "ಕರಿಯ ಕರಿಯ ಕರಿಯ ಕರಿಯ",
         12, False),
    ]

    passed = 0
    failed = 0

    for i, (desc, text, exp_count, exp_alive) in enumerate(test_cases, 1):
        state = CompositeState()
        for ch in text:
            state.feed_char(ch)
        state.flush()

        count_match = state.syllable_count == exp_count
        alive_match = state.is_alive() == exp_alive or state.has_accept() == exp_alive
        match = count_match and (exp_alive == state.has_accept() if exp_count == 12 else alive_match)

        status = "PASS" if match else "FAIL"
        print(f"  Test {i:2d}: {desc:<45s}  [{status}]  "
              f"syls={state.syllable_count} alive={state.is_alive()} accept={state.has_accept()}")
        if match:
            passed += 1
        else:
            print(f"           Expected: count={exp_count} alive/accept={exp_alive}")
            failed += 1

    # Snapshot/clone test
    state = CompositeState()
    for ch in "ಜಲದಾ ":
        state.feed_char(ch)
    snap = state.snapshot()
    clone = CompositeState.from_snapshot(snap)
    for ch in "ಮಣಿಯೂ ಮುದದೀ ನಲಿಯೇ":
        clone.feed_char(ch)
    clone.flush()
    snap_pass = clone.has_accept()
    status = "PASS" if snap_pass else "FAIL"
    print(f"  Test  4: Snapshot/clone continues correctly          [{status}]  accept={snap_pass}")
    if snap_pass:
        passed += 1
    else:
        failed += 1

    print()
    print(f"SUMMARY: {passed} passed, {failed} failed out of {passed + failed} tests")
    return failed == 0


if __name__ == "__main__":
    ok = run_tests()
    raise SystemExit(0 if ok else 1)
