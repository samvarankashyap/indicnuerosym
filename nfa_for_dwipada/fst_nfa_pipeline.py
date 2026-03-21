# -*- coding: utf-8 -*-
"""
FST+NFA Pipeline for Telugu Dwipada Validation.
=================================================

End-to-end pipeline that takes raw Telugu poem text and validates it against
all three Dwipada metrical constraints: gana pattern, prasa rhyme, and yati
alliteration.

The pipeline chains 3 FSTs and 3 NFAs as described in architecture.md:

    Raw Telugu Poem (2 lines)
        │
        ▼
    ┌─────────── Layer 1: FST Pipeline ───────────┐
    │  Stage 1: SyllableAssembler                  │
    │           text → syllables + boundaries      │
    │  Stage 2: GuruLaghuClassifier                │
    │           syllables → (syllable, U/I) tuples │
    └──────────────────────────────────────────────┘
        │
        ▼
    ┌─────────── Position Tracker ─────────────────┐
    │  Extracts positions from gana partition:      │
    │  • Prasa: 2nd syllable of each line           │
    │  • Yati:  1st syllable of gana 1 and gana 3   │
    └──────────────────────────────────────────────┘
        │               │               │
        ▼               ▼               ▼
    ┌─────────┐  ┌────────────┐  ┌────────────┐
    │ Gana NFA│  │ Prasa NFA  │  │ Yati NFA   │
    │ (every  │  │ (2nd syl   │  │ (gana1 1st │
    │  U/I)   │  │  per line) │  │  vs gana3) │
    └─────────┘  └────────────┘  └────────────┘
        │               │               │
        ▼               ▼               ▼
    ┌──────────── Combined Verdict ────────────────┐
    │  is_valid = gana AND prasa [AND yati]         │
    └──────────────────────────────────────────────┘

-------------------------------------------------------------------------------
USAGE
-------------------------------------------------------------------------------

    from fst_nfa_pipeline import DwipadaPipeline

    pipeline = DwipadaPipeline()
    result = pipeline.process("భువనత్రయాధారభూతమయుండు\\nపవనుండు లేకున్న బడు శరీరములు")

    print(result["is_valid_dwipada"])        # True or False
    print(result["validation_summary"])      # per-rule breakdown

    # With full trace from every FST and NFA stage:
    result, traces = pipeline.process_with_trace(poem)

"""

import os
import sys

# -- Setup import paths for sibling modules -------------------------------- #
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from syllable_assembler import SyllableAssembler
from guru_laghu_classifier import GuruLaghuClassifier
from gana_nfa import GanaNFA, format_partition_str
from prasa_nfa import PrasaNFA
from yati_nfa import YatiNFA, format_yati_result_str


###############################################################################
# 1) POSITION TRACKER HELPERS
###############################################################################
# These pure functions extract the syllable positions needed by Yati and Prasa
# NFAs from the gana partition output.


def _get_gana3_start_index(partition):
    """Compute the syllable index where gana 3 begins.

    Gana 3 is the third Indra gana (index 2). Its start position is
    the sum of syllable lengths of gana 0 and gana 1.

    Args:
        partition: A gana partition list (from GanaNFA), or None.

    Returns:
        Integer syllable index, or None if partition is invalid.
    """
    if not partition or len(partition) < 3:
        return None
    return len(partition[0]["pattern"]) + len(partition[1]["pattern"])


def _extract_yati_pair(partition, labeled_syllables):
    """Extract the (gana1_first, gana3_first) aksharam pair for yati checking.

    Args:
        partition:          Gana partition list from GanaNFA.
        labeled_syllables:  List of (syllable, label) tuples from GuruLaghuClassifier.

    Returns:
        Tuple (gana1_first_aksharam, gana3_first_aksharam), or ("", "") if
        the partition is invalid or syllables are insufficient.
    """
    if not partition or not labeled_syllables:
        return ("", "")

    g3_start = _get_gana3_start_index(partition)
    if g3_start is None:
        return ("", "")

    gana1_first = labeled_syllables[0][0] if len(labeled_syllables) > 0 else ""
    gana3_first = labeled_syllables[g3_start][0] if g3_start < len(labeled_syllables) else ""

    return (gana1_first, gana3_first)


###############################################################################
# 2) DwipadaPipeline CLASS
###############################################################################


class DwipadaPipeline:
    """Full FST+NFA validation pipeline for Telugu Dwipada poems.

    Chains together:
        SyllableAssembler → GuruLaghuClassifier → GanaNFA / PrasaNFA / YatiNFA

    and produces a combined validation result with optional full traces.

    Args:
        strict_yati: If True, yati match is required for is_valid_dwipada.
                     If False (default), yati is informational only.
    """

    def __init__(self, strict_yati=False):
        self.strict_yati = strict_yati

    def process(self, poem):
        """Run a poem through the full FST+NFA pipeline.

        Args:
            poem: A Telugu Dwipada poem string (2 lines separated by newline).

        Returns:
            Result dict with keys:
                - is_valid_dwipada: bool (final verdict)
                - poem: the input string
                - syllables: {line1: [...], line2: [...]}
                - guru_laghu: {line1: [(syl, label), ...], line2: [...]}
                - gana: {line1: partition|None, line2: ..., line1_valid, line2_valid}
                - prasa: dict (PrasaNFA result)
                - yati: {line1: dict, line2: dict}
                - validation_summary: {gana_valid, prasa_valid, yati_line1_match, ...}
        """
        # -- Split poem into 2 lines --
        lines = [l.strip() for l in poem.strip().split('\n') if l.strip()]
        if len(lines) < 2:
            return self._empty_result(poem, error="Poem must have 2 lines")

        line1_text, line2_text = lines[0], lines[1]

        # -- FST Stage 1: Syllable Assembly --
        assembler = SyllableAssembler()
        line1_syls = assembler.process(line1_text)
        assembler_2 = SyllableAssembler()
        line2_syls = assembler_2.process(line2_text)

        # -- FST Stage 2: Guru/Laghu Classification --
        classifier = GuruLaghuClassifier()
        line1_labeled = classifier.process(line1_syls)
        classifier_2 = GuruLaghuClassifier()
        line2_labeled = classifier_2.process(line2_syls)

        # -- NFA 1: Gana Partition --
        line1_markers = [label for _, label in line1_labeled]
        line2_markers = [label for _, label in line2_labeled]
        markers_combined = line1_markers + ["\n"] + line2_markers

        gana_nfa = GanaNFA()
        gana_partitions = gana_nfa.process(markers_combined)

        line1_partition = gana_partitions[0] if len(gana_partitions) > 0 else None
        line2_partition = gana_partitions[1] if len(gana_partitions) > 1 else None

        # -- NFA 2: Prasa --
        prasa_nfa = PrasaNFA()
        prasa_result = prasa_nfa.process(poem)

        # -- Position Tracker: extract yati pairs --
        yati_pair_1 = _extract_yati_pair(line1_partition, line1_labeled)
        yati_pair_2 = _extract_yati_pair(line2_partition, line2_labeled)

        # -- NFA 3: Yati --
        yati_nfa = YatiNFA()
        yati_results = yati_nfa.process([yati_pair_1, yati_pair_2])
        yati_line1 = yati_results[0] if len(yati_results) > 0 else {"match": False, "match_type": "no_match"}
        yati_line2 = yati_results[1] if len(yati_results) > 1 else {"match": False, "match_type": "no_match"}

        # -- Combine verdict --
        gana_valid = line1_partition is not None and line2_partition is not None
        prasa_valid = prasa_result.get("is_valid", False)
        yati_l1 = yati_line1.get("match", False)
        yati_l2 = yati_line2.get("match", False)

        is_valid = gana_valid and prasa_valid
        if self.strict_yati:
            is_valid = is_valid and yati_l1 and yati_l2

        return {
            "is_valid_dwipada": is_valid,
            "poem": poem,

            "syllables": {
                "line1": line1_syls,
                "line2": line2_syls,
            },

            "guru_laghu": {
                "line1": line1_labeled,
                "line2": line2_labeled,
            },

            "gana": {
                "line1": line1_partition,
                "line2": line2_partition,
                "line1_valid": line1_partition is not None,
                "line2_valid": line2_partition is not None,
            },

            "prasa": prasa_result,

            "yati": {
                "line1": yati_line1,
                "line2": yati_line2,
            },

            "validation_summary": {
                "gana_valid": gana_valid,
                "prasa_valid": prasa_valid,
                "yati_line1_match": yati_l1,
                "yati_line2_match": yati_l2,
            },
        }

    def process_with_trace(self, poem):
        """Run poem through the pipeline with full traces from every stage.

        Args:
            poem: A Telugu Dwipada poem string (2 lines separated by newline).

        Returns:
            (result, traces) where:
                - result: same dict as process()
                - traces: dict with per-stage trace data:
                    - syllable_assembler: {line1: [...], line2: [...]}
                    - guru_laghu: {line1: [...], line2: [...]}
                    - gana_nfa: [...]
                    - prasa_nfa: [...]
                    - yati_nfa: [...]
        """
        # -- Split poem --
        lines = [l.strip() for l in poem.strip().split('\n') if l.strip()]
        if len(lines) < 2:
            return self._empty_result(poem, error="Poem must have 2 lines"), {}

        line1_text, line2_text = lines[0], lines[1]

        # -- FST Stage 1: Syllable Assembly (with trace) --
        asm1 = SyllableAssembler()
        line1_syls, asm_trace1 = asm1.process_with_trace(line1_text)
        asm2 = SyllableAssembler()
        line2_syls, asm_trace2 = asm2.process_with_trace(line2_text)

        # -- FST Stage 2: Guru/Laghu Classification (with trace) --
        clf1 = GuruLaghuClassifier()
        line1_labeled, gl_trace1 = clf1.process_with_trace(line1_syls)
        clf2 = GuruLaghuClassifier()
        line2_labeled, gl_trace2 = clf2.process_with_trace(line2_syls)

        # -- NFA 1: Gana Partition (with trace) --
        line1_markers = [label for _, label in line1_labeled]
        line2_markers = [label for _, label in line2_labeled]
        markers_combined = line1_markers + ["\n"] + line2_markers

        gana_nfa = GanaNFA()
        gana_partitions, gana_trace = gana_nfa.process_with_trace(markers_combined)

        line1_partition = gana_partitions[0] if len(gana_partitions) > 0 else None
        line2_partition = gana_partitions[1] if len(gana_partitions) > 1 else None

        # -- NFA 2: Prasa (with trace) --
        prasa_nfa = PrasaNFA()
        prasa_result, prasa_trace = prasa_nfa.process_with_trace(poem)

        # -- Position Tracker: extract yati pairs --
        yati_pair_1 = _extract_yati_pair(line1_partition, line1_labeled)
        yati_pair_2 = _extract_yati_pair(line2_partition, line2_labeled)

        # -- NFA 3: Yati (with trace) --
        yati_nfa = YatiNFA()
        yati_results, yati_trace = yati_nfa.process_with_trace([yati_pair_1, yati_pair_2])
        yati_line1 = yati_results[0] if len(yati_results) > 0 else {"match": False, "match_type": "no_match"}
        yati_line2 = yati_results[1] if len(yati_results) > 1 else {"match": False, "match_type": "no_match"}

        # -- Combine verdict --
        gana_valid = line1_partition is not None and line2_partition is not None
        prasa_valid = prasa_result.get("is_valid", False)
        yati_l1 = yati_line1.get("match", False)
        yati_l2 = yati_line2.get("match", False)

        is_valid = gana_valid and prasa_valid
        if self.strict_yati:
            is_valid = is_valid and yati_l1 and yati_l2

        result = {
            "is_valid_dwipada": is_valid,
            "poem": poem,

            "syllables": {
                "line1": line1_syls,
                "line2": line2_syls,
            },

            "guru_laghu": {
                "line1": line1_labeled,
                "line2": line2_labeled,
            },

            "gana": {
                "line1": line1_partition,
                "line2": line2_partition,
                "line1_valid": line1_partition is not None,
                "line2_valid": line2_partition is not None,
            },

            "prasa": prasa_result,

            "yati": {
                "line1": yati_line1,
                "line2": yati_line2,
            },

            "validation_summary": {
                "gana_valid": gana_valid,
                "prasa_valid": prasa_valid,
                "yati_line1_match": yati_l1,
                "yati_line2_match": yati_l2,
            },
        }

        traces = {
            "syllable_assembler": {"line1": asm_trace1, "line2": asm_trace2},
            "guru_laghu": {"line1": gl_trace1, "line2": gl_trace2},
            "gana_nfa": gana_trace,
            "prasa_nfa": prasa_trace,
            "yati_nfa": yati_trace,
        }

        return result, traces

    def _empty_result(self, poem, error=""):
        """Return an empty/invalid result for malformed input."""
        return {
            "is_valid_dwipada": False,
            "poem": poem,
            "error": error,
            "syllables": {"line1": [], "line2": []},
            "guru_laghu": {"line1": [], "line2": []},
            "gana": {"line1": None, "line2": None, "line1_valid": False, "line2_valid": False},
            "prasa": {"is_valid": False},
            "yati": {"line1": {"match": False}, "line2": {"match": False}},
            "validation_summary": {
                "gana_valid": False, "prasa_valid": False,
                "yati_line1_match": False, "yati_line2_match": False,
            },
        }


###############################################################################
# 3) FORMATTING HELPERS
###############################################################################


def format_pipeline_result(result):
    """Format a pipeline result as a human-readable report.

    Shows each stage's output and the final verdict.

    Args:
        result: Dict from DwipadaPipeline.process().

    Returns:
        Multi-line report string.
    """
    lines = []
    poem = result.get("poem", "")
    lines.append("=" * 70)
    lines.append("DWIPADA FST+NFA PIPELINE REPORT")
    lines.append("=" * 70)

    # -- Poem --
    poem_lines = [l.strip() for l in poem.strip().split('\n') if l.strip()]
    for i, pl in enumerate(poem_lines[:2]):
        lines.append(f"  Line {i+1}: {pl}")
    lines.append("")

    # -- FST Stage 1: Syllables --
    lines.append("--- FST Stage 1: Syllable Assembly ---")
    for key in ["line1", "line2"]:
        syls = result.get("syllables", {}).get(key, [])
        # Filter out boundary markers for display
        display_syls = [s for s in syls if s not in (" ", "\n")]
        lines.append(f"  {key}: {' | '.join(display_syls)}")
    lines.append("")

    # -- FST Stage 2: Guru/Laghu --
    lines.append("--- FST Stage 2: Guru/Laghu Classification ---")
    for key in ["line1", "line2"]:
        labeled = result.get("guru_laghu", {}).get(key, [])
        display = " ".join(f"{syl}({label})" for syl, label in labeled)
        lines.append(f"  {key}: {display}")
    lines.append("")

    # -- NFA 1: Gana --
    lines.append("--- NFA 1: Gana Partition ---")
    gana = result.get("gana", {})
    for key in ["line1", "line2"]:
        partition = gana.get(key)
        valid = gana.get(f"{key}_valid", False)
        if partition:
            gana_str = format_partition_str(partition)
            lines.append(f"  {key}: {gana_str}  [VALID]")
        else:
            lines.append(f"  {key}: [INVALID — no valid gana partition found]")
    lines.append("")

    # -- NFA 2: Prasa --
    lines.append("--- NFA 2: Prasa Rhyme ---")
    prasa = result.get("prasa", {})
    prasa_valid = prasa.get("is_valid", False)
    match_type = prasa.get("match_type", "?")
    c1 = prasa.get("line1_consonant", "?")
    c2 = prasa.get("line2_consonant", "?")
    syl1 = prasa.get("line1_aksharam2", "?")
    syl2 = prasa.get("line2_aksharam2", "?")
    status = "VALID" if prasa_valid else "INVALID"
    lines.append(f"  2nd syllable: '{syl1}' (L1) ↔ '{syl2}' (L2)")
    lines.append(f"  Base consonant: '{c1}' ↔ '{c2}'  →  {match_type}  [{status}]")
    lines.append("")

    # -- NFA 3: Yati --
    lines.append("--- NFA 3: Yati Alliteration ---")
    yati = result.get("yati", {})
    for key in ["line1", "line2"]:
        yr = yati.get(key, {})
        lines.append(f"  {key}: {format_yati_result_str(yr)}")
    lines.append("")

    # -- Final Verdict --
    summary = result.get("validation_summary", {})
    is_valid = result.get("is_valid_dwipada", False)
    lines.append("=" * 70)
    verdict = "VALID DWIPADA" if is_valid else "INVALID DWIPADA"
    lines.append(f"VERDICT: {verdict}")
    lines.append(f"  Gana:  {'PASS' if summary.get('gana_valid') else 'FAIL'}")
    lines.append(f"  Prasa: {'PASS' if summary.get('prasa_valid') else 'FAIL'}")
    lines.append(f"  Yati:  L1={'PASS' if summary.get('yati_line1_match') else 'FAIL'}"
                 f"  L2={'PASS' if summary.get('yati_line2_match') else 'FAIL'}")
    lines.append("=" * 70)

    return "\n".join(lines)


def format_pipeline_trace(traces):
    """Format traces from all stages as a human-readable trace report.

    Args:
        traces: Dict from DwipadaPipeline.process_with_trace().

    Returns:
        Multi-line trace string (can be long — use for debugging).
    """
    lines = []

    # -- Syllable Assembler trace --
    lines.append("=== TRACE: Syllable Assembler ===")
    for key in ["line1", "line2"]:
        trace = traces.get("syllable_assembler", {}).get(key, [])
        lines.append(f"  --- {key} ({len(trace)} steps) ---")
        for t in trace[:30]:  # limit output
            char = t.get("char", "?")
            cat = t.get("category", "?")
            state_b = t.get("state_before", "?")
            state_a = t.get("state_after", "?")
            emitted = t.get("emitted", [])
            emit_str = f" → emitted: {emitted}" if emitted else ""
            lines.append(f"    '{char}' ({cat}) | {state_b} → {state_a}{emit_str}")
        if len(trace) > 30:
            lines.append(f"    ... ({len(trace) - 30} more steps)")
    lines.append("")

    # -- Guru/Laghu trace --
    lines.append("=== TRACE: Guru/Laghu Classifier ===")
    for key in ["line1", "line2"]:
        trace = traces.get("guru_laghu", {}).get(key, [])
        lines.append(f"  --- {key} ({len(trace)} steps) ---")
        for t in trace[:30]:
            item = t.get("item", "?")
            item_type = t.get("item_type", "?")
            state_b = t.get("state_before", "?")
            state_a = t.get("state_after", "?")
            emitted = t.get("emitted", [])
            emit_str = f" → emitted: {emitted}" if emitted else ""
            lines.append(f"    '{item}' ({item_type}) | {state_b} → {state_a}{emit_str}")
        if len(trace) > 30:
            lines.append(f"    ... ({len(trace) - 30} more steps)")
    lines.append("")

    # -- Gana NFA trace --
    lines.append("=== TRACE: Gana NFA ===")
    gana_trace = traces.get("gana_nfa", [])
    lines.append(f"  ({len(gana_trace)} steps)")
    for t in gana_trace[:40]:
        symbol = t.get("symbol", "?")
        branches_b = t.get("branches_before", "?")
        branches_a = t.get("branches_after", "?")
        lines.append(f"    symbol='{symbol}' | branches: {branches_b} → {branches_a}")
    if len(gana_trace) > 40:
        lines.append(f"    ... ({len(gana_trace) - 40} more steps)")
    lines.append("")

    # -- Prasa NFA trace --
    lines.append("=== TRACE: Prasa NFA ===")
    prasa_trace = traces.get("prasa_nfa", [])
    lines.append(f"  ({len(prasa_trace)} steps)")
    for t in prasa_trace[:20]:
        item = t.get("item", "?")
        state_b = t.get("state_before", "?")
        state_a = t.get("state_after", "?")
        details = t.get("details", "")
        lines.append(f"    '{item}' | {state_b} → {state_a}  {details}")
    if len(prasa_trace) > 20:
        lines.append(f"    ... ({len(prasa_trace) - 20} more steps)")
    lines.append("")

    # -- Yati NFA trace --
    lines.append("=== TRACE: Yati NFA ===")
    yati_trace = traces.get("yati_nfa", [])
    lines.append(f"  ({len(yati_trace)} steps)")
    for t in yati_trace:
        event = t.get("event", "?")
        phase_b = t.get("phase_before", "?")
        phase_a = t.get("phase_after", "?")
        details = t.get("details", "")
        lines.append(f"    {event} | {phase_b} → {phase_a} | {details}")
    lines.append("")

    return "\n".join(lines)


###############################################################################
# 4) INLINE TESTS
###############################################################################


def run_tests():
    """Run pipeline tests with real Dwipada poems.

    Returns:
        True if all tests pass, False otherwise.
    """
    pipeline = DwipadaPipeline()
    all_passed = True
    total = 0
    passed_count = 0

    test_cases = [
        # (name, poem, expected_valid, expected_gana, expected_prasa)
        (
            "Valid dwipada (Ranganatha Ramayanam)",
            "భువనత్రయాధారభూతమయుండు \nపవనుండు లేకున్న బడు శరీరములు",
            True,    # is_valid_dwipada
            True,    # gana_valid
            True,    # prasa_valid
        ),
        (
            "Valid dwipada (Ranganatha Ramayanam #2)",
            "సౌధాగ్రముల యందు సదనంబు లందు\nవీధుల యందును వెఱవొప్ప నిలిచి",
            True,
            True,
            True,
        ),
        (
            "Single line (should fail — need 2 lines)",
            "భువనత్రయాధారభూతమయుండు",
            False,
            False,
            False,
        ),
    ]

    print("=" * 70)
    print("FST+NFA PIPELINE TEST SUITE")
    print("=" * 70)
    print()

    for name, poem, exp_valid, exp_gana, exp_prasa in test_cases:
        total += 1
        result = pipeline.process(poem)

        actual_valid = result["is_valid_dwipada"]
        actual_gana = result["validation_summary"]["gana_valid"]
        actual_prasa = result["validation_summary"]["prasa_valid"]

        passed = (
            actual_valid == exp_valid
            and actual_gana == exp_gana
            and actual_prasa == exp_prasa
        )

        if passed:
            passed_count += 1
        else:
            all_passed = False

        icon = "✓" if passed else "✗"
        status = "PASS" if passed else "FAIL"
        print(f"  {icon} [{status}] {name}")
        print(f"       valid={actual_valid} (exp={exp_valid}), "
              f"gana={actual_gana} (exp={exp_gana}), "
              f"prasa={actual_prasa} (exp={exp_prasa})")
        if not passed:
            print(f"       >>> MISMATCH")
            print(format_pipeline_result(result))
        print()

    # -- Full report test: print a detailed report for the first valid poem --
    print("-" * 70)
    print("Full pipeline report for poem #1:")
    print("-" * 70)
    result = pipeline.process(test_cases[0][1])
    print(format_pipeline_result(result))
    print()

    # -- Trace test: verify traces are populated --
    total += 1
    print("-" * 70)
    print("Trace test: verify all trace sections are populated")
    print("-" * 70)
    result, traces = pipeline.process_with_trace(test_cases[0][1])

    trace_sections = ["syllable_assembler", "guru_laghu", "gana_nfa", "prasa_nfa", "yati_nfa"]
    trace_passed = True
    for section in trace_sections:
        trace_data = traces.get(section)
        has_data = False
        if isinstance(trace_data, dict):
            has_data = any(len(v) > 0 for v in trace_data.values() if isinstance(v, list))
        elif isinstance(trace_data, list):
            has_data = len(trace_data) > 0

        icon = "✓" if has_data else "✗"
        print(f"  {icon} {section}: {'has data' if has_data else 'EMPTY'}")
        if not has_data:
            trace_passed = False

    if trace_passed:
        passed_count += 1
    else:
        all_passed = False
    print()

    # Print abbreviated trace
    print("-" * 70)
    print("Abbreviated trace output:")
    print("-" * 70)
    print(format_pipeline_trace(traces))

    # -- Summary --
    print("=" * 70)
    print(f"RESULTS: {passed_count}/{total} tests passed")
    if all_passed:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    print("=" * 70)

    return all_passed


if __name__ == "__main__":
    ok = run_tests()
    raise SystemExit(0 if ok else 1)
