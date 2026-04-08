# -*- coding: utf-8 -*-
"""
FST+NFA Pipeline for Kannada Utsaha Ragale Validation.
========================================================

End-to-end pipeline that takes raw Kannada poem text and validates it against
all Ragale metrical constraints: gana pattern, Ādi Prāsa, and guru ending.

Pipeline architecture:

    Raw Kannada Poem (2 lines)
        │
        ▼
    ┌─────────── Layer 1: FST Pipeline ───────────┐
    │  Stage 1: SyllableAssembler                  │
    │           text → syllables + boundaries      │
    │  Stage 2: GuruLaghuClassifier                │
    │           syllables → (syllable, U/I) tuples │
    └──────────────────────────────────────────────┘
        │               │
        ▼               ▼
    ┌─────────┐  ┌────────────┐
    │ Gana NFA│  │ Prasa NFA  │
    │ (every  │  │ (2nd syl   │
    │  U/I)   │  │  per line) │
    └─────────┘  └────────────┘
        │               │
        ▼               ▼
    ┌──────────── Combined Verdict ────────────────┐
    │  is_valid = gana AND prasa AND guru_ending   │
    └──────────────────────────────────────────────┘

Usage:

    from ragale_pipeline import RagalePipeline

    pipeline = RagalePipeline()
    result = pipeline.process("ಜಲದಾ ಮಣಿಯೂ ಮುದದೀ ನಲಿಯೇ\\nನಿಲದೇ ಒಡೆದೂ ಮರೆಯಾಗುವುದೂ")

    print(result["is_valid_ragale"])
    print(result["validation_summary"])

Adapted from nfa_for_dwipada/fst_nfa_pipeline.py (Telugu Dwipada version).
"""

import os
import sys
import json

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from syllable_assembler import SyllableAssembler
from guru_laghu_classifier import GuruLaghuClassifier
from gana_nfa import GanaNFA, format_partition_str
from prasa_nfa import PrasaNFA


###############################################################################
# 1) RagalePipeline CLASS
###############################################################################


class RagalePipeline:
    """Full FST+NFA validation pipeline for Kannada Utsaha Ragale poems.

    Chains together:
        SyllableAssembler → GuruLaghuClassifier → GanaNFA / PrasaNFA

    and produces a combined validation result.

    Args:
        strict_prasa: If True (default), prasa equivalences are disabled
                      (exact consonant match only).
    """

    def __init__(self, strict_prasa=False):
        self.strict_prasa = strict_prasa

    def process(self, poem):
        """Run a poem through the full FST+NFA pipeline.

        Args:
            poem: A Kannada Ragale poem string (2 lines separated by newline).

        Returns:
            Result dict with keys:
                - is_valid_ragale: bool (final verdict)
                - poem: the input string
                - syllables: {line1: [...], line2: [...]}
                - guru_laghu: {line1: [(syl, label), ...], line2: [...]}
                - gana: {line1: partition|None, line2: ..., line1_valid, line2_valid}
                - prasa: dict (PrasaNFA result)
                - guru_ending: {line1: bool, line2: bool}
                - validation_summary: per-rule breakdown
        """
        lines = [l.strip() for l in poem.strip().split('\n') if l.strip()]
        if len(lines) < 2:
            return self._empty_result(poem, error="Poem must have 2 lines")

        line1_text, line2_text = lines[0], lines[1]

        # -- FST Stage 1: Syllable Assembly --
        asm1 = SyllableAssembler()
        line1_syls = asm1.process(line1_text)
        asm2 = SyllableAssembler()
        line2_syls = asm2.process(line2_text)

        # -- FST Stage 2: Guru/Laghu Classification --
        clf1 = GuruLaghuClassifier()
        line1_labeled = clf1.process(line1_syls)
        clf2 = GuruLaghuClassifier()
        line2_labeled = clf2.process(line2_syls)

        # -- NFA 1: Gana Partition --
        line1_markers = [label for _, label in line1_labeled]
        line2_markers = [label for _, label in line2_labeled]
        markers_combined = line1_markers + ["\n"] + line2_markers

        gana_nfa = GanaNFA()
        gana_partitions = gana_nfa.process(markers_combined)

        line1_partition = gana_partitions[0] if len(gana_partitions) > 0 else None
        line2_partition = gana_partitions[1] if len(gana_partitions) > 1 else None

        # -- NFA 2: Prasa --
        prasa_nfa = PrasaNFA(strict=self.strict_prasa)
        prasa_result = prasa_nfa.process(poem)

        # -- Guru ending check --
        guru_end_l1 = line1_markers[-1] == "U" if line1_markers else False
        guru_end_l2 = line2_markers[-1] == "U" if line2_markers else False

        # -- Combine verdict --
        gana_valid = line1_partition is not None and line2_partition is not None
        prasa_valid = prasa_result.get("is_valid", False)
        guru_valid = guru_end_l1 and guru_end_l2

        is_valid = gana_valid and prasa_valid and guru_valid

        return {
            "is_valid_ragale": is_valid,
            "poem": poem,

            "syllables": {
                "line1": [s for s in line1_syls if s not in (" ", "\n")],
                "line2": [s for s in line2_syls if s not in (" ", "\n")],
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
                "line1_display": format_partition_str(line1_partition),
                "line2_display": format_partition_str(line2_partition),
            },

            "prasa": prasa_result,

            "guru_ending": {
                "line1": guru_end_l1,
                "line2": guru_end_l2,
            },

            "validation_summary": {
                "gana_valid": gana_valid,
                "prasa_valid": prasa_valid,
                "guru_ending_valid": guru_valid,
                "syllable_count_line1": len(line1_markers),
                "syllable_count_line2": len(line2_markers),
            },
        }

    def process_with_trace(self, poem):
        """Run poem through the pipeline with full traces from every stage."""
        lines = [l.strip() for l in poem.strip().split('\n') if l.strip()]
        if len(lines) < 2:
            return self._empty_result(poem, error="Poem must have 2 lines"), {}

        line1_text, line2_text = lines[0], lines[1]

        asm1 = SyllableAssembler()
        line1_syls, asm_trace1 = asm1.process_with_trace(line1_text)
        asm2 = SyllableAssembler()
        line2_syls, asm_trace2 = asm2.process_with_trace(line2_text)

        clf1 = GuruLaghuClassifier()
        line1_labeled, gl_trace1 = clf1.process_with_trace(line1_syls)
        clf2 = GuruLaghuClassifier()
        line2_labeled, gl_trace2 = clf2.process_with_trace(line2_syls)

        line1_markers = [label for _, label in line1_labeled]
        line2_markers = [label for _, label in line2_labeled]
        markers_combined = line1_markers + ["\n"] + line2_markers

        gana_nfa = GanaNFA()
        gana_partitions, gana_trace = gana_nfa.process_with_trace(markers_combined)

        line1_partition = gana_partitions[0] if len(gana_partitions) > 0 else None
        line2_partition = gana_partitions[1] if len(gana_partitions) > 1 else None

        prasa_nfa = PrasaNFA(strict=self.strict_prasa)
        prasa_result, prasa_trace = prasa_nfa.process_with_trace(poem)

        guru_end_l1 = line1_markers[-1] == "U" if line1_markers else False
        guru_end_l2 = line2_markers[-1] == "U" if line2_markers else False

        gana_valid = line1_partition is not None and line2_partition is not None
        prasa_valid = prasa_result.get("is_valid", False)
        guru_valid = guru_end_l1 and guru_end_l2

        is_valid = gana_valid and prasa_valid and guru_valid

        result = {
            "is_valid_ragale": is_valid,
            "poem": poem,
            "syllables": {
                "line1": [s for s in line1_syls if s not in (" ", "\n")],
                "line2": [s for s in line2_syls if s not in (" ", "\n")],
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
                "line1_display": format_partition_str(line1_partition),
                "line2_display": format_partition_str(line2_partition),
            },
            "prasa": prasa_result,
            "guru_ending": {"line1": guru_end_l1, "line2": guru_end_l2},
            "validation_summary": {
                "gana_valid": gana_valid,
                "prasa_valid": prasa_valid,
                "guru_ending_valid": guru_valid,
                "syllable_count_line1": len(line1_markers),
                "syllable_count_line2": len(line2_markers),
            },
        }

        traces = {
            "syllable_assembler": {"line1": asm_trace1, "line2": asm_trace2},
            "guru_laghu": {"line1": gl_trace1, "line2": gl_trace2},
            "gana_nfa": gana_trace,
            "prasa_nfa": prasa_trace,
        }

        return result, traces

    def _empty_result(self, poem, error=""):
        return {
            "is_valid_ragale": False,
            "poem": poem,
            "error": error,
            "syllables": {"line1": [], "line2": []},
            "guru_laghu": {"line1": [], "line2": []},
            "gana": {"line1": None, "line2": None,
                     "line1_valid": False, "line2_valid": False,
                     "line1_display": "N/A", "line2_display": "N/A"},
            "prasa": {"is_valid": False},
            "guru_ending": {"line1": False, "line2": False},
            "validation_summary": {
                "gana_valid": False, "prasa_valid": False,
                "guru_ending_valid": False,
                "syllable_count_line1": 0, "syllable_count_line2": 0,
            },
        }


###############################################################################
# 2) FORMATTING HELPERS
###############################################################################

def format_pipeline_result(result: dict) -> str:
    """Format a pipeline result dict as a human-readable report."""
    lines = []
    lines.append("=" * 60)

    if "error" in result:
        lines.append(f"ERROR: {result['error']}")
        lines.append(f"Poem: {result['poem']}")
        return "\n".join(lines)

    verdict = "VALID" if result["is_valid_ragale"] else "INVALID"
    lines.append(f"Ragale Validation: {verdict}")
    lines.append("=" * 60)

    for label, lk in [("Line 1", "line1"), ("Line 2", "line2")]:
        syls = result["syllables"][lk]
        gl = result["guru_laghu"][lk]
        markers = " ".join(l for _, l in gl)
        syl_str = " | ".join(syls)
        lines.append(f"\n  {label}: {' '.join(s for s, _ in gl)}")
        lines.append(f"  Syllables ({len(syls)}): {syl_str}")
        lines.append(f"  Markers: {markers}")
        lines.append(f"  Gana: {result['gana'][lk + '_display']}")
        lines.append(f"  Guru ending: {'✓' if result['guru_ending'][lk] else '✗'}")

    p = result["prasa"]
    lines.append(f"\n  Ādi Prāsa: {p.get('line1_consonant')} vs {p.get('line2_consonant')} "
                 f"→ {'✓' if p['is_valid'] else '✗'} ({p.get('match_type', 'N/A')})")

    vs = result["validation_summary"]
    lines.append(f"\n  Summary: gana={vs['gana_valid']} prasa={vs['prasa_valid']} "
                 f"guru={vs['guru_ending_valid']}")
    lines.append(f"  Syllable counts: L1={vs['syllable_count_line1']} L2={vs['syllable_count_line2']}")
    lines.append("")
    return "\n".join(lines)


###############################################################################
# 3) BATCH PROCESSING FOR JSON FILES
###############################################################################

def process_json_file(filepath: str, strict_prasa: bool = False) -> list[dict]:
    """Process a JSON file of poems and return all results."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        poems = [data]
    elif isinstance(data, list):
        poems = data
    else:
        print("Error: JSON must be an object or array of objects.")
        return []

    pipeline = RagalePipeline(strict_prasa=strict_prasa)
    results = []

    for poem_obj in poems:
        poem_text = poem_obj.get("poem_kannada", "")
        result = pipeline.process(poem_text)
        result["theme"] = poem_obj.get("theme", "")
        results.append(result)

    return results


###############################################################################
# 4) TESTS
###############################################################################

def run_tests():
    pipeline = RagalePipeline()

    test_cases = [
        ("Valid — all IIU",
         "ಜಲದಾ ಮಣಿಯೂ ಮುದದೀ ನಲಿಯೇ\nನಿಲದೇ ಒಡೆದೂ ಮರೆಯಾಗುವುದೂ",
         True, True, True, True),

        ("Valid — III+IIU mix",
         "ಚೆಲುವ ಮುಡಿಯ ಕುಸುಮ ಸೆಳೆಯೂ\nನಲುವ ಕುರುಳು ನಯನ ಕುಸಿಯೂ",
         True, True, True, True),

        ("Invalid — single line",
         "ಜಲದಾ ಮಣಿಯೂ ಮುದದೀ ನಲಿಯೇ",
         False, False, False, False),

        ("Valid — matching prasa ಡ",
         "ಒಡಲೀ ಉಸಿರೂ ತುಳುಕೀ ಕುಣಿತಾ\nಒಡೆದೂ ಜಲದೀ ಬೆರೆತೂ ಅಳಿದೂ",
         True, True, True, True),
    ]

    passed = 0
    failed = 0

    for i, (desc, poem, exp_valid, exp_gana, exp_prasa, exp_guru) in enumerate(test_cases, 1):
        result = pipeline.process(poem)
        vs = result["validation_summary"]

        match = (result["is_valid_ragale"] == exp_valid
                 and vs["gana_valid"] == exp_gana
                 and vs["prasa_valid"] == exp_prasa
                 and vs["guru_ending_valid"] == exp_guru)

        status = "PASS" if match else "FAIL"
        print(f"  Test {i:2d}: {desc:<35s}  [{status}]  "
              f"valid={result['is_valid_ragale']} "
              f"gana={vs['gana_valid']} prasa={vs['prasa_valid']} guru={vs['guru_ending_valid']}")

        if not match:
            print(f"           Expected: valid={exp_valid} gana={exp_gana} prasa={exp_prasa} guru={exp_guru}")
            failed += 1
        else:
            passed += 1

    print()
    print(f"SUMMARY: {passed} passed, {failed} failed out of {passed + failed} tests")
    return failed == 0


if __name__ == "__main__":
    import sys as _sys

    if len(_sys.argv) > 1:
        # Process JSON file
        filepath = _sys.argv[1]
        results = process_json_file(filepath)
        valid_count = sum(1 for r in results if r["is_valid_ragale"])

        for r in results:
            print(format_pipeline_result(r))

        print("=" * 60)
        print(f"SUMMARY: {valid_count}/{len(results)} poems valid")
        print("=" * 60)

        # Write results
        output_path = filepath.rsplit(".", 1)[0] + "_nfa_results.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        print(f"Results written to: {output_path}")
    else:
        ok = run_tests()
        raise SystemExit(0 if ok else 1)
