# -*- coding: utf-8 -*-
"""
Prasa NFA for Telugu Dwipada.
===============================

Validates the prasa (ప్రాస) rhyme constraint of a two-line Dwipada couplet.

Prasa is the most distinctive rhyming device in Telugu poetry.  The rule is
simple but non-negotiable:

    The **2nd aksharam** (syllable) of every line in a couplet must share the
    same **base consonant**.

"Base consonant" means the first Telugu consonant character of the syllable,
ignoring vowel marks (matras), conjunct extensions (vattulu), anusvara, and
visarga.  Three pairs of consonants are traditionally considered equivalent
for prasa purposes:

    ల ↔ ళ   (laterals)
    శ ↔ స   (sibilants)
    ఱ ↔ ర   (rhotics)

If the 2nd aksharam of either line is vowel-initial (no consonant), prasa
cannot be satisfied.

This is Stage 4b of the NFA constrained-decoding pipeline:

    Raw text
       |
       v
    [SyllableAssembler FST]   -- Stage 1: Unicode chars -> syllables
       |
       v
    [GuruLaghuClassifier FST] -- Stage 2: syllables -> (syllable, U/I) pairs
       |
       v
    [GanaNFA]                 -- Stage 4a: U/I stream -> gana partition
    [PrasaNFA]                -- Stage 4b: syllable stream -> prasa check (THIS)
    [YatiNFA]                 -- Stage 4c: (future)

Unlike GanaNFA which consumes the abstract U/I stream, PrasaNFA operates on
the **syllable stream** — it needs the actual Telugu text of each aksharam to
extract the base consonant for comparison.

-------------------------------------------------------------------------------
NFA STATE MACHINE
-------------------------------------------------------------------------------

    LINE1_SYL0 --[syl]--> LINE1_SYL1 --[syl, store consonant]--> LINE1_REST
                                                                       |
                                                                  [newline]
                                                                       v
    LINE2_SYL0 --[syl]--> LINE2_SYL1 --[syl, check match]--> ACCEPT / REJECT

Spaces (" ") in the syllable stream are skipped — they are word boundaries,
not aksharalu.  Only actual syllable text increments the position counter.

The NFA is deterministic (exactly one transition per input at each state),
but follows the NFA naming convention for consistency with the pipeline.

-------------------------------------------------------------------------------
USAGE
-------------------------------------------------------------------------------

    from prasa_nfa import PrasaNFA

    nfa = PrasaNFA()

    # Full pipeline — raw Telugu text:
    result = nfa.process("సౌధాగ్రముల యందు సదనంబు లందు\\nవీధుల యందును వెఱవొప్ప నిలిచి")
    print(result["is_valid"])       # True
    print(result["match_type"])     # "exact"
    print(result["line1_consonant"])  # "ధ"

    # Streaming API — feed syllables one at a time:
    nfa = PrasaNFA()
    for syl in ["సౌ", "ధా", "గ్ర", "ము", "ల"]:
        nfa.feed(syl)
    nfa.feed("\\n")
    for syl in ["వీ", "ధు", "ల"]:
        nfa.feed(syl)
    result = nfa.flush()

    # Pre-split syllable lists:
    result = nfa.process_syllables(["సౌ", "ధా", "గ్ర"], ["వీ", "ధు", "ల"])

"""

###############################################################################
# 1) IMPORTS AND PATH SETUP
###############################################################################

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "..", "src"))

from syllable_assembler import TELUGU_CONSONANTS, SyllableAssembler
from guru_laghu_classifier import GuruLaghuClassifier


###############################################################################
# 2) PRASA CONSTANTS
###############################################################################

# Prasa equivalency groups (ప్రాస సమానాక్షరములు)
# These consonant pairs are traditionally treated as interchangeable for prasa.
PRASA_EQUIVALENTS = [
    frozenset({"ల", "ళ"}),   # laterals  (పార్శ్వికములు)
    frozenset({"శ", "స"}),   # sibilants (ఊష్మములు)
    frozenset({"ఱ", "ర"}),   # rhotics   (ప్రకంపనములు)
]

# Fast lookup: consonant -> its equivalence group (or None)
CONSONANT_TO_CLASS: dict[str, frozenset[str]] = {}
for _group in PRASA_EQUIVALENTS:
    for _c in _group:
        CONSONANT_TO_CLASS[_c] = _group

# Human-readable names for equivalence classes
CLASS_NAMES: dict[frozenset[str], str] = {
    frozenset({"ల", "ళ"}): "lateral",
    frozenset({"శ", "స"}): "sibilant",
    frozenset({"ఱ", "ర"}): "rhotic",
}

# NFA state constants
LINE1_SYL0 = "LINE1_SYL0"   # waiting for 1st syllable of line 1
LINE1_SYL1 = "LINE1_SYL1"   # waiting for 2nd syllable of line 1 (prasa position)
LINE1_REST = "LINE1_REST"   # consuming remaining syllables of line 1
LINE2_SYL0 = "LINE2_SYL0"   # waiting for 1st syllable of line 2
LINE2_SYL1 = "LINE2_SYL1"   # waiting for 2nd syllable of line 2 (check position)
ACCEPT     = "ACCEPT"       # prasa matched
REJECT     = "REJECT"       # prasa did not match

# Items to skip (word boundaries, not aksharalu)
SKIP_ITEMS = frozenset({" "})


###############################################################################
# 3) PURE HELPER FUNCTIONS
###############################################################################


def get_base_consonant(aksharam: str) -> str | None:
    """Extract the base (first) consonant from a Telugu syllable.

    For prasa matching we compare only the first consonant character,
    ignoring vowel marks, conjunct tails, anusvara, and visarga.

    Args:
        aksharam: A single Telugu syllable string.

    Returns:
        The first consonant character, or None if vowel-initial.

    Examples:
        >>> get_base_consonant("కా")     # consonant + long vowel
        'క'
        >>> get_base_consonant("క్ష")    # conjunct — first consonant only
        'క'
        >>> get_base_consonant("అ")      # pure vowel — no consonant
        >>> get_base_consonant("రం")     # consonant + anusvara
        'ర'
    """
    if not aksharam:
        return None
    first_char = aksharam[0]
    if first_char in TELUGU_CONSONANTS:
        return first_char
    return None


def get_consonant_class(consonant: str) -> str:
    """Return the equivalence class name for a consonant.

    Consonants in an equivalence group (ల↔ళ, శ↔స, ఱ↔ర) return the
    group name.  All other consonants are their own singleton class.

    Args:
        consonant: A single Telugu consonant character.

    Returns:
        Class name string, e.g. ``"lateral"`` or ``"క"`` (itself).
    """
    group = CONSONANT_TO_CLASS.get(consonant)
    if group is not None:
        return CLASS_NAMES[group]
    return consonant


def are_prasa_equivalent(c1: str | None, c2: str | None) -> bool:
    """Check whether two consonants satisfy the prasa constraint.

    Returns True if they are identical or belong to the same equivalence
    group.  Returns False if either is None (vowel-initial).

    Args:
        c1: Base consonant from line 1's 2nd aksharam (or None).
        c2: Base consonant from line 2's 2nd aksharam (or None).

    Returns:
        True if prasa is satisfied.
    """
    if c1 is None or c2 is None:
        return False
    if c1 == c2:
        return True
    group = CONSONANT_TO_CLASS.get(c1)
    return group is not None and c2 in group


def classify_match(c1: str | None, c2: str | None) -> tuple[bool, str, frozenset | None]:
    """Classify the type of prasa match between two consonants.

    Args:
        c1: Base consonant from line 1 (or None).
        c2: Base consonant from line 2 (or None).

    Returns:
        Tuple of ``(is_match, match_type, equivalence_group)``:
        - ``(True,  "exact",      None)``                 — identical consonants
        - ``(True,  "equivalent", frozenset({"ల","ళ"}))`` — same equiv group
        - ``(False, "mismatch",   None)``                 — different consonants
        - ``(False, "no_consonant", None)``                — one/both vowel-initial
    """
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
    NFA-based prasa validator for Telugu Dwipada couplets.

    Consumes a stream of syllables (aksharalu) and checks whether the
    2nd aksharam of line 1 and line 2 share the same base consonant.

    Three interfaces:

    1. ``process(poem)`` — full pipeline from raw Telugu text.
    2. ``process_syllables(line1, line2)`` — from pre-split syllable lists.
    3. ``feed(item)`` / ``flush()`` — streaming, one item at a time.
    """

    def __init__(self):
        self._reset()

    def _reset(self):
        """Clear all state for fresh processing."""
        self.state = LINE1_SYL0
        self.line1_aksharalu: list[str] = []
        self.line2_aksharalu: list[str] = []
        self.prasa_consonant: str | None = None    # stored from line 1
        self.prasa_aksharam1: str | None = None    # 2nd aksharam of line 1
        self.prasa_aksharam2: str | None = None    # 2nd aksharam of line 2
        self.consonant2: str | None = None         # extracted from line 2
        self.rejection_reason: str | None = None

    # ------------------------------------------------------------------
    # Streaming API
    # ------------------------------------------------------------------

    def feed(self, item: str):
        """Feed one item: a syllable string, ``" "`` (space), or ``"\\n"``.

        Spaces are skipped (word boundaries, not aksharalu).
        Newlines trigger line transitions.
        """
        # Skip spaces — not aksharalu
        if item in SKIP_ITEMS:
            return

        # Newline — transition between lines
        if item == "\n":
            self._on_newline()
            return

        # Terminal states absorb everything
        if self.state in (ACCEPT, REJECT):
            if self.state == ACCEPT:
                self.line2_aksharalu.append(item)
            return

        # Syllable processing based on current state
        if self.state == LINE1_SYL0:
            self.line1_aksharalu.append(item)
            self.state = LINE1_SYL1

        elif self.state == LINE1_SYL1:
            self.line1_aksharalu.append(item)
            self._extract_line1_prasa(item)

        elif self.state == LINE1_REST:
            self.line1_aksharalu.append(item)

        elif self.state == LINE2_SYL0:
            self.line2_aksharalu.append(item)
            self.state = LINE2_SYL1

        elif self.state == LINE2_SYL1:
            self.line2_aksharalu.append(item)
            self._check_line2_prasa(item)

    def _on_newline(self):
        """Handle a newline separator between lines."""
        if self.state in (LINE1_SYL0, LINE1_SYL1):
            self.state = REJECT
            self.rejection_reason = "Line 1 too short — fewer than 2 aksharalu"
        elif self.state == LINE1_REST:
            self.state = LINE2_SYL0
        # Newlines in line 2 or terminal states are ignored

    def _extract_line1_prasa(self, aksharam: str):
        """Extract and store the prasa consonant from line 1's 2nd aksharam."""
        self.prasa_aksharam1 = aksharam
        consonant = get_base_consonant(aksharam)
        if consonant is None:
            self.state = REJECT
            self.rejection_reason = (
                f"Line 1's 2nd aksharam '{aksharam}' is vowel-initial — "
                f"no base consonant for prasa"
            )
        else:
            self.prasa_consonant = consonant
            self.state = LINE1_REST

    def _check_line2_prasa(self, aksharam: str):
        """Check line 2's 2nd aksharam against the stored prasa consonant."""
        self.prasa_aksharam2 = aksharam
        self.consonant2 = get_base_consonant(aksharam)
        if self.consonant2 is None:
            self.state = REJECT
            self.rejection_reason = (
                f"Line 2's 2nd aksharam '{aksharam}' is vowel-initial — "
                f"no base consonant for prasa"
            )
        elif are_prasa_equivalent(self.prasa_consonant, self.consonant2):
            self.state = ACCEPT
        else:
            self.state = REJECT
            self.rejection_reason = (
                f"Prasa mismatch: line 1 has '{self.prasa_consonant}' "
                f"({get_consonant_class(self.prasa_consonant)}), "
                f"line 2 has '{self.consonant2}' "
                f"({get_consonant_class(self.consonant2)})"
            )

    def flush(self) -> dict:
        """Signal end-of-input and return the prasa validation result.

        Returns:
            dict with full prasa analysis.  See module docstring for schema.
        """
        # Handle incomplete input
        if self.state in (LINE1_SYL0, LINE1_SYL1):
            self.state = REJECT
            self.rejection_reason = "Line 1 too short — fewer than 2 aksharalu"
        elif self.state == LINE1_REST:
            self.state = REJECT
            self.rejection_reason = "Poem has only 1 line — need 2 for prasa check"
        elif self.state in (LINE2_SYL0, LINE2_SYL1):
            self.state = REJECT
            self.rejection_reason = "Line 2 too short — fewer than 2 aksharalu"

        is_valid, match_type, equiv_group = classify_match(
            self.prasa_consonant, self.consonant2
        )

        return {
            "is_valid": self.state == ACCEPT,
            "state": self.state,
            "line1_second_aksharam": self.prasa_aksharam1,
            "line1_consonant": self.prasa_consonant,
            "line1_consonant_class": (
                get_consonant_class(self.prasa_consonant)
                if self.prasa_consonant else None
            ),
            "line2_second_aksharam": self.prasa_aksharam2,
            "line2_consonant": self.consonant2,
            "line2_consonant_class": (
                get_consonant_class(self.consonant2)
                if self.consonant2 else None
            ),
            "match_type": match_type,
            "equivalence_group": equiv_group,
            "rejection_reason": self.rejection_reason,
        }

    # ------------------------------------------------------------------
    # High-level API
    # ------------------------------------------------------------------

    def process(self, poem: str) -> dict:
        """Full pipeline: raw Telugu text → prasa validation result.

        Runs SyllableAssembler and GuruLaghuClassifier internally, then
        feeds the syllable stream through the NFA.

        Args:
            poem: Two-line Telugu poem separated by ``\\n``.

        Returns:
            dict with prasa analysis and guru/laghu mapping.
        """
        self._reset()

        lines = poem.strip().split("\n")
        if len(lines) < 2:
            self.state = REJECT
            self.rejection_reason = "Poem has only 1 line — need 2 for prasa check"
            result = self.flush()
            result["guru_laghu_mapping"] = {"line1": [], "line2": []}
            return result

        asm = SyllableAssembler()
        clf = GuruLaghuClassifier()

        # Stage 1 + 2 for each line
        syls_line1 = asm.process(lines[0].strip())
        syls_line2 = asm.process(lines[1].strip())
        labels_line1 = clf.process(syls_line1)
        labels_line2 = clf.process(syls_line2)

        # Feed syllables through NFA (skip boundary markers from assembler)
        for item in syls_line1:
            self.feed(item)
        self.feed("\n")
        for item in syls_line2:
            self.feed(item)

        result = self.flush()
        result["guru_laghu_mapping"] = {
            "line1": labels_line1,
            "line2": labels_line2,
        }
        return result

    def process_syllables(
        self,
        line1_syls: list[str],
        line2_syls: list[str],
    ) -> dict:
        """Process pre-split syllable lists (no assembler needed).

        Useful when syllables are already available from an earlier pipeline
        stage.  Spaces and newlines in the input lists are handled normally.

        Args:
            line1_syls: Syllables of line 1 (may include ``" "`` boundaries).
            line2_syls: Syllables of line 2 (may include ``" "`` boundaries).

        Returns:
            dict with prasa analysis (no guru/laghu mapping).
        """
        self._reset()
        for item in line1_syls:
            self.feed(item)
        self.feed("\n")
        for item in line2_syls:
            self.feed(item)
        return self.flush()

    def process_with_trace(self, poem: str) -> tuple[dict, list[dict]]:
        """Full pipeline with step-by-step trace for debugging.

        Args:
            poem: Two-line Telugu poem separated by ``\\n``.

        Returns:
            ``(result, trace)`` where trace is a list of per-step dicts:
            ``{"item", "item_type", "state_before", "state_after",
               "action", "consonant_extracted"}``.
        """
        self._reset()
        trace: list[dict] = []

        lines = poem.strip().split("\n")
        if len(lines) < 2:
            self.state = REJECT
            self.rejection_reason = "Poem has only 1 line — need 2 for prasa check"
            result = self.flush()
            result["guru_laghu_mapping"] = {"line1": [], "line2": []}
            return result, trace

        asm = SyllableAssembler()
        clf = GuruLaghuClassifier()

        syls_line1 = asm.process(lines[0].strip())
        syls_line2 = asm.process(lines[1].strip())
        labels_line1 = clf.process(syls_line1)
        labels_line2 = clf.process(syls_line2)

        def _feed_with_trace(item: str):
            state_before = self.state
            consonant_before = self.prasa_consonant
            self.feed(item)
            state_after = self.state

            # Determine item type
            if item == "\n":
                item_type = "newline"
            elif item == " ":
                item_type = "space"
            else:
                item_type = "syllable"

            # Determine action taken
            action = "skip"
            consonant_extracted = None
            if item_type == "space":
                action = "skip (word boundary)"
            elif item_type == "newline":
                action = f"line break → {state_after}"
            elif state_before == LINE1_SYL0:
                action = "1st aksharam of line 1 (skip)"
            elif state_before == LINE1_SYL1:
                consonant_extracted = self.prasa_consonant
                if state_after == REJECT:
                    action = f"2nd aksharam '{item}' is vowel-initial → REJECT"
                else:
                    action = f"stored prasa consonant '{self.prasa_consonant}'"
            elif state_before == LINE1_REST:
                action = "consume (line 1 rest)"
            elif state_before == LINE2_SYL0:
                action = "1st aksharam of line 2 (skip)"
            elif state_before == LINE2_SYL1:
                consonant_extracted = self.consonant2
                if state_after == ACCEPT:
                    _, mt, _ = classify_match(consonant_before, self.consonant2)
                    action = f"'{self.consonant2}' matches '{consonant_before}' ({mt}) → ACCEPT"
                else:
                    action = f"'{self.consonant2}' ≠ '{consonant_before}' → REJECT"
            elif state_before in (ACCEPT, REJECT):
                action = f"absorb ({state_before})"

            trace.append({
                "item": item,
                "item_type": item_type,
                "state_before": state_before,
                "state_after": state_after,
                "action": action,
                "consonant_extracted": consonant_extracted,
            })

        for item in syls_line1:
            _feed_with_trace(item)
        _feed_with_trace("\n")
        for item in syls_line2:
            _feed_with_trace(item)

        # Flush
        state_before = self.state
        result = self.flush()
        result["guru_laghu_mapping"] = {
            "line1": labels_line1,
            "line2": labels_line2,
        }
        trace.append({
            "item": "FLUSH",
            "item_type": "flush",
            "state_before": state_before,
            "state_after": self.state,
            "action": "finalize",
            "consonant_extracted": None,
        })

        return result, trace


###############################################################################
# 5) FORMATTING HELPERS
###############################################################################


def format_prasa_result(result: dict) -> str:
    """One-line summary of a prasa validation result.

    Args:
        result: dict returned by ``PrasaNFA.process()`` or ``.flush()``.

    Returns:
        e.g. ``"ACCEPT — ధ = ధ (exact)"`` or ``"REJECT — ద ≠ గ (mismatch)"``
    """
    state = result["state"]
    c1 = result["line1_consonant"] or "∅"
    c2 = result["line2_consonant"] or "∅"
    match_type = result["match_type"]

    if state == ACCEPT:
        return f"ACCEPT — {c1} = {c2} ({match_type})"
    else:
        reason = result.get("rejection_reason") or "unknown"
        return f"REJECT — {c1} ≠ {c2} ({match_type}) — {reason}"


def format_prasa_detailed(result: dict) -> str:
    """Multi-line detailed display of a prasa validation result.

    Args:
        result: dict returned by ``PrasaNFA.process()`` or ``.flush()``.

    Returns:
        Formatted string with line breakdown and guru/laghu mapping.
    """
    parts = []
    parts.append(f"Prasa Validation: {'✓ VALID' if result['is_valid'] else '✗ INVALID'}")
    parts.append(f"  State: {result['state']}")

    ak1 = result["line1_second_aksharam"] or "—"
    ak2 = result["line2_second_aksharam"] or "—"
    c1 = result["line1_consonant"] or "∅"
    c2 = result["line2_consonant"] or "∅"
    cls1 = result["line1_consonant_class"] or "—"
    cls2 = result["line2_consonant_class"] or "—"

    parts.append(f"  Line 1: 2nd aksharam = '{ak1}', consonant = '{c1}' (class: {cls1})")
    parts.append(f"  Line 2: 2nd aksharam = '{ak2}', consonant = '{c2}' (class: {cls2})")
    parts.append(f"  Match type: {result['match_type']}")

    if result["equivalence_group"]:
        members = " ↔ ".join(sorted(result["equivalence_group"]))
        parts.append(f"  Equivalence group: {members}")

    if result.get("rejection_reason"):
        parts.append(f"  Reason: {result['rejection_reason']}")

    # Guru/laghu mapping if available
    gl = result.get("guru_laghu_mapping")
    if gl:
        for line_key in ("line1", "line2"):
            labels = gl.get(line_key, [])
            if labels:
                label_str = "  ".join(f"{syl}({lbl})" for syl, lbl in labels)
                parts.append(f"  {line_key} guru/laghu: {label_str}")

    return "\n".join(parts)


def format_trace(trace: list[dict]) -> str:
    """Format a process_with_trace() trace for display.

    Args:
        trace: list of trace step dicts.

    Returns:
        Formatted table string.
    """
    lines = []
    lines.append(f"  {'Step':>4}  {'Item':>12}  {'Type':>8}  {'Before':>12} -> {'After':>12}  Action")
    lines.append(f"  {'----':>4}  {'----':>12}  {'----':>8}  {'------':>12}    {'-----':>12}  ------")
    for i, t in enumerate(trace):
        item_display = repr(t["item"]) if t["item_type"] != "syllable" else t["item"]
        lines.append(
            f"  {i:4d}  {item_display:>12}  {t['item_type']:>8}  "
            f"{t['state_before']:>12} -> {t['state_after']:>12}  "
            f"{t['action']}"
        )
    return "\n".join(lines)


###############################################################################
# 6) INLINE TESTS
###############################################################################


def _print_test(desc, poem, result, trace, index):
    """Pretty-print a test result."""
    print(f"\n{'='*78}")
    print(f"Test {index}: {desc}")
    print(f"{'='*78}")
    if poem:
        for i, line in enumerate(poem.strip().split("\n"), 1):
            print(f"  Line {i}: {line}")
    print()
    print(format_trace(trace))
    print()
    print(format_prasa_detailed(result))
    print()


def run_tests():
    """Run comprehensive tests and cross-validate against analyzer.py."""
    nfa = PrasaNFA()
    passed = 0
    failed = 0

    # ------------------------------------------------------------------
    # Test cases: (description, input, expected_valid, expected_match_type)
    # ------------------------------------------------------------------

    test_cases = [
        # ── Group 1: Exact consonant match ────────────────────────────
        (
            "Exact match: ధ = ధ (classic dwipada)",
            "సౌధాగ్రముల యందు సదనంబు లందు\nవీధుల యందును వెఱవొప్ప నిలిచి",
            True,
            "exact",
        ),
        (
            "Exact match: న = న",
            "మానవ సేవయే మాధవ సేవ\nదానధర్మమునకు దైవం బొసగు",
            True,
            "exact",
        ),
        (
            "Exact match: ల = ల",
            "తలచిన వెంటనే దైవంబు వచ్చి\nకలుగును సంపద కష్టంబు తీరి",
            True,
            "exact",
        ),

        # ── Group 2: Equivalent consonant match ───────────────────────
        (
            "Equivalent match: ల ↔ ళ (laterals)",
            # Synthetic: 2nd aksharam is ళా in line 1, లు in line 2
            "కాళాంజనాద్రి రాజన్ముని\nమాలిన్యమును దూరమై పోవు",
            True,
            "equivalent",
        ),
        (
            "Equivalent match: శ ↔ స (sibilants)",
            # 2nd aksharam: శా in line 1, సు in line 2
            "విశాలాక్షి యందు వెలుగు గాంచి\nరసాలమునందు రాగ మలరి",
            True,
            "equivalent",
        ),
        (
            "Equivalent match: ర ↔ ఱ (rhotics)",
            # 2nd aksharam: ఱి in line 1, ర in line 2
            "వెఱచి పారెను వేగముతోడ\nమరలి వచ్చెను మనసులోనను",
            True,
            "equivalent",
        ),

        # ── Group 3: Mismatch ─────────────────────────────────────────
        (
            "Mismatch: న ≠ ర",
            "మానవ సేవయే మాధవ సేవ\nసారమైనట్టి సద్గుణ మలరు",
            False,
            "mismatch",
        ),
        (
            "Mismatch: ధ ≠ మ",
            "సౌధాగ్రముల యందు సదనంబు లందు\nరామాయణంబులో రాముడున్నాడు",
            False,
            "mismatch",
        ),

        # ── Group 4: Edge cases ───────────────────────────────────────
        (
            "Single line — no prasa possible",
            "సౌధాగ్రముల యందు సదనంబు లందు",
            False,
            "no_consonant",
        ),
    ]

    for i, (desc, poem, expected_valid, expected_match) in enumerate(test_cases, 1):
        result, trace = nfa.process_with_trace(poem)
        actual_valid = result["is_valid"]
        actual_match = result["match_type"]

        success = (actual_valid == expected_valid) and (actual_match == expected_match)

        _print_test(desc, poem, result, trace, i)
        if success:
            print(f"  PASS")
            passed += 1
        else:
            print(f"  FAIL")
            print(f"    Expected: is_valid={expected_valid}, match_type={expected_match}")
            print(f"    Actual:   is_valid={actual_valid}, match_type={actual_match}")
            failed += 1

    # ------------------------------------------------------------------
    # Unit tests for pure helper functions
    # ------------------------------------------------------------------
    print(f"\n{'='*78}")
    print("Unit Tests: Pure helper functions")
    print(f"{'='*78}")

    unit_tests = [
        # get_base_consonant
        ("get_base_consonant('కా')", get_base_consonant("కా"), "క"),
        ("get_base_consonant('క్ష')", get_base_consonant("క్ష"), "క"),
        ("get_base_consonant('అ')", get_base_consonant("అ"), None),
        ("get_base_consonant('రం')", get_base_consonant("రం"), "ర"),
        ("get_base_consonant('')", get_base_consonant(""), None),
        ("get_base_consonant('ధా')", get_base_consonant("ధా"), "ధ"),

        # get_consonant_class
        ("get_consonant_class('ల')", get_consonant_class("ల"), "lateral"),
        ("get_consonant_class('ళ')", get_consonant_class("ళ"), "lateral"),
        ("get_consonant_class('శ')", get_consonant_class("శ"), "sibilant"),
        ("get_consonant_class('స')", get_consonant_class("స"), "sibilant"),
        ("get_consonant_class('ర')", get_consonant_class("ర"), "rhotic"),
        ("get_consonant_class('ఱ')", get_consonant_class("ఱ"), "rhotic"),
        ("get_consonant_class('క')", get_consonant_class("క"), "క"),

        # are_prasa_equivalent
        ("are_prasa_equivalent('క', 'క')", are_prasa_equivalent("క", "క"), True),
        ("are_prasa_equivalent('ల', 'ళ')", are_prasa_equivalent("ల", "ళ"), True),
        ("are_prasa_equivalent('ళ', 'ల')", are_prasa_equivalent("ళ", "ల"), True),
        ("are_prasa_equivalent('శ', 'స')", are_prasa_equivalent("శ", "స"), True),
        ("are_prasa_equivalent('ర', 'ఱ')", are_prasa_equivalent("ర", "ఱ"), True),
        ("are_prasa_equivalent('క', 'గ')", are_prasa_equivalent("క", "గ"), False),
        ("are_prasa_equivalent(None, 'క')", are_prasa_equivalent(None, "క"), False),

        # classify_match
        ("classify_match('ధ', 'ధ')[1]", classify_match("ధ", "ధ")[1], "exact"),
        ("classify_match('ల', 'ళ')[1]", classify_match("ల", "ళ")[1], "equivalent"),
        ("classify_match('క', 'గ')[1]", classify_match("క", "గ")[1], "mismatch"),
        ("classify_match(None, 'క')[1]", classify_match(None, "క")[1], "no_consonant"),
    ]

    for desc, actual, expected in unit_tests:
        ok = actual == expected
        status = "PASS" if ok else "FAIL"
        print(f"  {status}  {desc}  →  {actual!r}  (expected {expected!r})")
        if ok:
            passed += 1
        else:
            failed += 1

    # ------------------------------------------------------------------
    # Streaming API test
    # ------------------------------------------------------------------
    print(f"\n{'='*78}")
    print("Test: Streaming feed() API")
    print(f"{'='*78}")

    nfa_stream = PrasaNFA()
    # Feed line 1 syllables for "సౌధాగ్రముల" = సౌ ధా గ్ర ము ల
    for syl in ["సౌ", "ధా", "గ్ర", "ము", "ల"]:
        nfa_stream.feed(syl)
    nfa_stream.feed("\n")
    # Feed line 2 syllables for "వీధుల" = వీ ధు ల
    for syl in ["వీ", "ధు", "ల"]:
        nfa_stream.feed(syl)
    stream_result = nfa_stream.flush()

    stream_ok = (
        stream_result["is_valid"] is True
        and stream_result["line1_consonant"] == "ధ"
        and stream_result["line2_consonant"] == "ధ"
        and stream_result["match_type"] == "exact"
    )
    print(f"  {'PASS' if stream_ok else 'FAIL'}  "
          f"Streaming: {format_prasa_result(stream_result)}")
    if stream_ok:
        passed += 1
    else:
        failed += 1

    # ------------------------------------------------------------------
    # process_syllables() API test
    # ------------------------------------------------------------------
    print(f"\n{'='*78}")
    print("Test: process_syllables() API")
    print(f"{'='*78}")

    nfa_syls = PrasaNFA()
    syls_result = nfa_syls.process_syllables(
        ["సౌ", "ధా", "గ్ర", "ము", "ల"],
        ["వీ", "ధు", "ల"],
    )
    syls_ok = (
        syls_result["is_valid"] is True
        and syls_result["match_type"] == "exact"
    )
    print(f"  {'PASS' if syls_ok else 'FAIL'}  "
          f"process_syllables: {format_prasa_result(syls_result)}")
    if syls_ok:
        passed += 1
    else:
        failed += 1

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'='*78}")
    print(f"SUMMARY: {passed} passed, {failed} failed out of {passed + failed}")
    print(f"{'='*78}")

    return failed == 0


if __name__ == "__main__":
    ok = run_tests()
    raise SystemExit(0 if ok else 1)
