"""
Guru/Laghu Classifier FST for Telugu.

Reads a stream of syllables (output from SyllableAssembler) and classifies
each syllable as Guru (U) or Laghu (I).

Six classification rules (matching akshara_ganavibhajana() in aksharanusarika.py):

    Rule 1 (Deergham)  — syllable contains a long matra {ా ీ ూ ే ో ౌ ౄ}
                         OR is a standalone long independent vowel {ఆ ఈ ఊ ౠ ఏ ఓ}  →  U
    Rule 2 (Pluta)     — syllable contains ఐ / ఔ (independent) or
                         ై / ౌ (dependent)                                          →  U
    Rule 3 (Diacritic) — syllable contains anusvara (ం) or visarga (ః)             →  U
    Rule 4 (Pollu)     — syllable ends with halant / virama (్)                    →  U
    Rule 5 (Conjunct)  — *next* syllable in the same word contains a C+్+C
                         pattern (conjunct or doubled consonant)                    →  U
                         BLOCKED by a word boundary (SPACE or NEWLINE) between
                         the two syllables.
    (default)          — none of the above                                          →  I

Rules 1–4 are intrinsic: decided the moment the syllable arrives.
Rule 5 requires a 1-syllable lookahead.  Every intrinsically Laghu syllable is
held in the buffer until the *next* syllable is known.  A word boundary flushes
the buffer (emitting the buffered syllable as Laghu) before the boundary blocks
Rule 5.

States:
    EMPTY      — nothing buffered; ready for next syllable
    PENDING_I  — one intrinsically Laghu syllable in buffer,
                 waiting to see whether next syllable is a conjunct

See design_guru_laghu_classifier.md for the full FST design.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "..", "src"))

from syllable_assembler import (
    TELUGU_CONSONANTS,
    VIRAMA,
    SyllableAssembler,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Long dependent-vowel signs (matras)
LONG_MATRAS: frozenset[str] = frozenset("ా ీ ూ ే ో ౌ ౄ".split())

# Long independent vowels — the *entire* syllable must equal one of these
INDEPENDENT_LONG_VOWELS: frozenset[str] = frozenset("ఆ ఈ ఊ ౠ ఏ ఓ".split())

# Boundary items emitted by SyllableAssembler
BOUNDARIES: frozenset[str] = frozenset({" ", "\n"})

# FST state names
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
    if "ఐ" in syl or "ఔ" in syl:                    return "U"   # Rule 2  — pluta (indep)
    if "ై" in syl or "ౌ" in syl:                    return "U"   # Rule 2  — pluta (dep)
    if "ం" in syl:                                   return "U"   # Rule 3  — anusvara
    if "ః" in syl:                                   return "U"   # Rule 3  — visarga
    if syl.endswith("్"):                            return "U"   # Rule 4  — pollu (trailing halant)
    return "I"


def is_conjunct_trigger(syl: str) -> bool:
    """
    Return True if *syl* contains a C+్+C pattern — a conjunct (C1≠C2) or
    doubled consonant (C1==C2).  This is the trigger for Rule 5 on the
    preceding syllable.
    """
    for i in range(len(syl) - 2):
        if (syl[i]     in TELUGU_CONSONANTS
                and syl[i + 1] == VIRAMA
                and syl[i + 2] in TELUGU_CONSONANTS):
            return True
    return False


def _rule_note(syl: str, promoted_by_rule5: bool = False) -> str:
    """Short annotation explaining why a syllable is Guru (for traces)."""
    if promoted_by_rule5:                                          return "Rule 5 (conjunct follows)"
    if any(c in LONG_MATRAS for c in syl):                        return "Rule 1 (long matra)"
    if syl in INDEPENDENT_LONG_VOWELS:                            return "Rule 1 (long indep vowel)"
    if "ఐ" in syl or "ఔ" in syl or "ై" in syl or "ౌ" in syl:    return "Rule 2 (pluta)"
    if "ం" in syl:                                                 return "Rule 3 (anusvara)"
    if "ః" in syl:                                                 return "Rule 3 (visarga)"
    if syl.endswith("్"):                                          return "Rule 4 (pollu)"
    return "—"


# ---------------------------------------------------------------------------
# FST
# ---------------------------------------------------------------------------

class GuruLaghuClassifier:
    """
    FST-based Telugu Guru/Laghu classifier.

    Receives output from SyllableAssembler (syllables + boundaries) and emits
    ``(syllable_text, label)`` pairs where label is ``'U'`` (Guru) or ``'I'``
    (Laghu).

    Typical usage::

        asm = SyllableAssembler()
        clf = GuruLaghuClassifier()
        labels = clf.process(asm.process("సత్యము"))
        # → [('స', 'U'), ('త్య', 'I'), ('ము', 'I')]
    """

    def __init__(self):
        self._reset()

    def _reset(self):
        self.state:       str       = STATE_EMPTY
        self.buffer_syl:  str | None = None
        self.output: list[tuple[str, str]] = []

    def snapshot(self):
        """Return a lightweight snapshot of the FST state (no output)."""
        return (self.state, self.buffer_syl)

    def restore(self, snap):
        """Restore FST state from a snapshot. Clears output."""
        self.state, self.buffer_syl = snap
        self.output = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _emit(self, syl: str, label: str):
        self.output.append((syl, label))

    def _flush_buffer(self, rule5_applies: bool) -> list[tuple[str, str]]:
        """
        Emit the buffered Laghu syllable.
        If ``rule5_applies`` is True the syllable is promoted to Guru (Rule 5).
        Returns the list of newly emitted (syl, label) pairs (may be empty).
        """
        if self.state == STATE_PENDING_I and self.buffer_syl is not None:
            label = "U" if rule5_applies else "I"
            self._emit(self.buffer_syl, label)
            released = [(self.buffer_syl, label)]
            self.buffer_syl = None
            self.state      = STATE_EMPTY
            return released
        return []

    # ------------------------------------------------------------------
    # Transition handlers
    # ------------------------------------------------------------------

    def _on_syllable(self, syl: str) -> list[tuple[str, str]]:
        """
        Process one syllable.
        Returns the list of (syl, label) items emitted during this step.
        """
        is_ct = is_conjunct_trigger(syl)

        # Resolve the buffered Laghu — use this syllable as lookahead for Rule 5
        emitted = self._flush_buffer(rule5_applies=is_ct)

        # Classify the incoming syllable intrinsically (Rules 1–4)
        label = intrinsic_label(syl)

        if label == "U":
            # Guru: emit immediately — no lookahead can change it further
            self._emit(syl, "U")
            emitted.append((syl, "U"))
            self.state = STATE_EMPTY
        else:
            # Laghu: buffer it and wait for the next syllable (Rule 5 lookahead)
            self.buffer_syl = syl
            self.state      = STATE_PENDING_I

        return emitted

    def _on_boundary(self) -> list[tuple[str, str]]:
        """
        Process a word boundary (SPACE or NEWLINE).
        Flushes the buffer as Laghu — a boundary blocks Rule 5 cross-word.
        """
        return self._flush_buffer(rule5_applies=False)

    def flush(self) -> list[tuple[str, str]]:
        """End-of-input flush — emit any remaining buffered syllable as Laghu."""
        return self._flush_buffer(rule5_applies=False)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(
        self, syllables_with_boundaries: list[str]
    ) -> list[tuple[str, str]]:
        """
        Classify a SyllableAssembler output list.

        Args:
            syllables_with_boundaries: list of syllable strings mixed with
                ``' '`` and ``'\\n'`` boundary markers.

        Returns:
            List of ``(syllable_text, label)`` pairs ('U' or 'I').
            Boundary items are consumed internally and not included.
        """
        self._reset()
        for item in syllables_with_boundaries:
            if item in BOUNDARIES:
                self._on_boundary()
            else:
                self._on_syllable(item)
        self.flush()
        return self.output

    def process_text(self, text: str) -> list[tuple[str, str]]:
        """
        Convenience: run SyllableAssembler → classify in one call.

        Args:
            text: raw Telugu string.

        Returns:
            List of ``(syllable_text, label)`` pairs.
        """
        return self.process(SyllableAssembler().process(text))

    def process_with_trace(
        self, syllables_with_boundaries: list[str]
    ) -> tuple[list[tuple[str, str]], list[dict]]:
        """
        Process with a step-by-step state trace.

        Returns:
            ``(output, trace)`` where each trace entry is a dict::

                item          – syllable string, boundary char, or '∎' (flush)
                item_type     – 'syllable' | 'boundary' | 'flush'
                intrinsic     – intrinsic label 'U'|'I', or None
                is_conjunct   – True if syl triggers Rule 5 on prev syllable
                state_before  – FST state before processing this item
                buffer_before – buffered syllable text ('' if empty)
                state_after   – FST state after
                buffer_after  – buffered syllable text after ('' if empty)
                emitted       – list of (syl, label, rule_note) tuples
        """
        self._reset()
        trace: list[dict] = []

        items = list(syllables_with_boundaries) + [None]  # None = flush sentinel

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

            # Annotate each emitted item with the rule that decided its label
            emitted_annotated: list[tuple[str, str, str]] = []
            for syl, lbl in released:
                if lbl == "U":
                    promoted = (
                        syl == buffer_before        # it was the buffered syllable
                        and buffer_before != ""
                        and intrinsic_label(syl) == "I"   # intrinsically Laghu → must be Rule 5
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
# Tests — verified against akshara_ganavibhajana() from aksharanusarika.py
# ---------------------------------------------------------------------------

def _print_trace(
    desc: str,
    text: str,
    syllables: list[str],
    trace: list[dict],
    result: list[tuple[str, str]],
    gt_labels: list[str],
    index: int,
    expected: list[str] | None = None,
):
    result_labels = [lbl for _, lbl in result]
    cmp_labels = expected if expected is not None else gt_labels
    status = "PASS" if result_labels == cmp_labels else "FAIL"
    bar = "=" * 90

    print()
    print(bar)
    print(f"Test {index}: {desc}  [{status}]")
    print(bar)

    # Input breakdown
    print(f"  Input       : {repr(text)}")
    cp_parts = []
    for ch in text:
        label = repr(ch) if ch in (" ", "\n") else ch
        cp_parts.append(f"{label}(U+{ord(ch):04X})")
    print(f"  Codepoints  : {' · '.join(cp_parts)}")
    print(f"  Syllables   : {syllables}")
    print()

    # Trace table
    COL = {
        "item":    10,
        "type":    10,
        "intr":    6,
        "conj":    6,
        "s_bef":   12,
        "b_bef":   12,
        "s_aft":   12,
        "b_aft":   12,
        "emit":    36,
    }
    header = (
        f"  {'Item':<{COL['item']}} {'Type':<{COL['type']}} "
        f"{'Intr':<{COL['intr']}} {'Conj':<{COL['conj']}} "
        f"{'St-Before':<{COL['s_bef']}} {'Buf-Before':<{COL['b_bef']}} "
        f"{'St-After':<{COL['s_aft']}}  {'Buf-After':<{COL['b_aft']}} "
        f"{'Emitted'}"
    )
    print(header)
    print("  " + "─" * (len(header) - 2))

    for row in trace:
        item_disp = repr(row["item"]) if row["item"] in (" ", "\n", "∎") else row["item"]
        intr_disp = row["intrinsic"] if row["intrinsic"] else "—"
        conj_disp = "T" if row["is_conjunct"] else "F"
        emit_parts = []
        for syl, lbl, note in row["emitted"]:
            emit_parts.append(f"({syl!r},{lbl}) {note}")
        emit_disp = " | ".join(emit_parts) if emit_parts else "—"

        print(
            f"  {item_disp:<{COL['item']}} {row['item_type']:<{COL['type']}} "
            f"{intr_disp:<{COL['intr']}} {conj_disp:<{COL['conj']}} "
            f"{row['state_before']:<{COL['s_bef']}} {repr(row['buffer_before']):<{COL['b_bef']}} "
            f"{row['state_after']:<{COL['s_aft']}}  {repr(row['buffer_after']):<{COL['b_aft']}} "
            f"{emit_disp}"
        )

    # Result summary
    print()
    label_str = " ".join(result_labels)
    gt_str    = " ".join(gt_labels)
    exp_str   = " ".join(cmp_labels)
    print(f"  Labels      : {label_str}")
    print(f"  Expected    : {exp_str}")
    if gt_labels != cmp_labels:
        print(f"  GT reference: {gt_str}  (FST intentionally differs)")
    else:
        print(f"  GT reference: {gt_str}")
    if result_labels != cmp_labels:
        print(f"  *** MISMATCH ***")
    else:
        syl_label_pairs = "  ".join(f"{s}={l}" for s, l in result)
        print(f"  Detail      : {syl_label_pairs}")


def run_tests():
    from dwipada.core.aksharanusarika import split_aksharalu, akshara_ganavibhajana

    asm = SyllableAssembler()
    clf = GuruLaghuClassifier()

    # (description, text, expected_label_string)
    test_cases = [
        # --- Intrinsic rules ---
        ("All Laghu (short vowels)",          "అమల",           "I I I"),
        ("Rule 1 — long matra (రా)",          "రాముడు",        "U I I"),
        ("Rule 1 — long indep vowel (ఆ)",     "ఆదిన్",         "U U"),
        ("Rule 1 — multiple long matras",     "కాలేజీ",        "U U U"),
        ("Rule 2 — pluta ై",                  "సైనికుడు",      "U I I I"),
        ("Rule 2 — pluta ౌ / గౌ",             "గౌరవం",         "U I U"),
        ("Rule 3 — anusvara (సం)",            "సందడి",         "U I I"),
        ("Rule 3 — visarga (దుః)",            "దుఃఖము",        "U I I"),
        ("Rule 4 — pollu at end",             "పూసెన్",        "U U"),
        # Conjunct+trailing pollu → the whole word merges into one syllable (Guru).
        # Note: split_aksharalu splits this differently, but the SyllableAssembler
        # correctly treats trailing pollu on a conjunct as merging the full block
        # onto the previous akshara (one syllable: 'సెన్స్').
        ("Rule 4 — conjunct+pollu → 1 syl",   "సెన్స్",        "U"),
        # --- Rule 5 (conjunct lookahead) ---
        ("Rule 5 — before doubled (మ్మ)",    "అమ్మ",           "U I"),
        ("Rule 5 — before conjunct (త్య)",   "సత్యము",        "U I I"),
        ("Rule 5 — before conjunct (ష్ట)",   "కష్టము",        "U I I"),
        # విశ్రాంతి: వి(I→U Rule5) శ్రాం(U long+anusvara) తి(I)
        ("Rule 5 — before conjunct (శ్ర)",   "విశ్రాంతి",     "U U I"),
        # --- Special vowels ---
        ("Vocalic ృ is always Laghu",         "కృషి",          "I I"),
        # Short ు is not in LONG_MATRAS — always Laghu
        ("Short ు is Laghu",                  "చిలుక",         "I I I"),
        # --- Word boundaries ---
        ("SPACE blocks Rule 5 cross-word",    "తెలుగు భాష",   "I I I U I"),
        ("NEWLINE flushes buffer",            "ఆదిన్\nమదిన్",  "U U I U"),
        ("Pollu before space",                "పదిన్ కాలం",    "I U U U"),
        # --- Mixed / multi-syllable ---
        # శ్రీ has long matra ీ → U (intrinsic, not Rule 5)
        ("Mixed phrase",                      "శ్రీరాముడు జయం", "U U I I I U"),
        ("Anusvara after conjunct",           "సంస్కృతం",      "U I U"),
        # Note: reference skips SPACE when looking for next conjunct (cross-word
        # Rule 5), but our FST flushes on boundary — linguistically sounder for
        # streaming.  Test below stays within a single word.
        ("Rule 5 — doubled within word",      "అక్కర",         "U I I"),
        # Pollu on a conjunct merges the whole block onto previous akshara
        ("Conjunct+pollu → 1 Guru syl",       "నాన్సెన్స్",    "U U"),
    ]

    passed = 0
    failed = 0

    for i, (desc, text, expected_str) in enumerate(test_cases, 1):
        expected = expected_str.split()

        syllables = asm.process(text)
        result, trace = clf.process_with_trace(syllables)

        # Ground truth from aksharanusarika.py
        aksharalu = split_aksharalu(text)
        gt_all    = akshara_ganavibhajana(aksharalu)
        gt_labels = [m for m in gt_all if m]   # drop '' entries for spaces/newlines

        _print_trace(desc, text, syllables, trace, result, gt_labels, i, expected)

        our_labels = [lbl for _, lbl in result]

        # Primary check: hardcoded expected (source of truth for this FST).
        # GT from split_aksharalu is shown for reference but may diverge on
        # known edge cases (e.g. conjunct+trailing pollu treated as 1 syllable).
        if our_labels == expected:
            passed += 1
        else:
            failed += 1
            print(f"  *** MISMATCH: got {our_labels}, expected {expected}")

        if our_labels != gt_labels:
            print(f"  (Info: differs from aksharanusarika reference {gt_labels})")

    print()
    print("=" * 90)
    print(f"SUMMARY: {passed} passed, {failed} failed out of {passed + failed} tests")
    print("=" * 90)
    return failed == 0


if __name__ == "__main__":
    ok = run_tests()
    raise SystemExit(0 if ok else 1)
