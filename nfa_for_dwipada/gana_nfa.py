# -*- coding: utf-8 -*-
"""
Gana Pattern NFA for Telugu Dwipada.
=====================================

Reads a stream of Guru (U) / Laghu (I) markers — the output of GanaMarker —
and partitions each line into exactly 3 Indra ganas + 1 Surya gana,
identifying each gana by name.

This is Stage 3 of the NFA constrained-decoding pipeline:

    Raw text
       |
       v
    [SyllableAssembler FST]   -- Stage 1: Unicode chars -> syllables
       |
       v
    [GanaMarker FST]          -- Stage 2: syllables -> U/I markers
       |
       v
    [Gana NFA]                -- Stage 3: U/I stream -> gana partition  (THIS FILE)

-------------------------------------------------------------------------------
FORMAL LANGUAGE
-------------------------------------------------------------------------------

Each Dwipada line must match:

    L_line = (IIII | IIIU | IIUI | UII | UIU | UUI)^3 · (III | UI)
              |__________ Indra ganas ___________|     |_ Surya _|

Indra Ganas (ఇంద్ర గణములు) — 6 types, 3–4 syllables each:

    Nala  (నల)       I I I I
    Naga  (నగ)       I I I U
    Sala  (సల)       I I U I
    Bha   (భ)        U I I
    Ra    (ర)        U I U
    Ta    (త)        U U I

Surya Ganas (సూర్య గణములు) — 2 types, 2–3 syllables each:

    Na      (న)        I I I
    Ha/Gala (హ/గల)     U I

-------------------------------------------------------------------------------
HOW THE NFA WORKS
-------------------------------------------------------------------------------

The NFA non-deterministically guesses which gana pattern is being read.
After the first syllable:
  - On I → could be Nala, Naga, or Sala (need more symbols)
  - On U → could be Bha, Ra, or Ta

State = set of active branches. Each branch is a tuple:
    (slot, gana_name, sub_pos, matched_ganas)

Dead state detection prunes impossible branches after each symbol.
When a gana completes, branches for the next slot are spawned.
A line is valid when any branch reaches ACCEPT (all 4 slots filled).

-------------------------------------------------------------------------------
USAGE
-------------------------------------------------------------------------------

    from gana_nfa import GanaNFA

    nfa = GanaNFA()
    result = nfa.process(["I","I","I","U","I","I","I","I","U","I","U","U","I"])
    # => [[ {type, name, pattern, symbols}, ... ]]

    # Two-line dwipada (with newline separator):
    markers = ["I","I","I","U","I","I","I","I","U","I","U","U","I",
               "\\n",
               "I","I","I","U","I","I","I","I","U","I","U","U","I"]
    result = nfa.process(markers)

    # Format for display:
    for line_partition in result:
        print(format_partition_str(line_partition))

"""

###############################################################################
# 1) GANA PATTERN CONSTANTS
###############################################################################

INDRA_GANAS = {
    "Nala":  ("I", "I", "I", "I"),
    "Naga":  ("I", "I", "I", "U"),
    "Sala":  ("I", "I", "U", "I"),
    "Bha":   ("U", "I", "I"),
    "Ra":    ("U", "I", "U"),
    "Ta":    ("U", "U", "I"),
}

SURYA_GANAS = {
    "Na":      ("I", "I", "I"),
    "Ha_Gala": ("U", "I"),
}

# Telugu display names
GANA_DISPLAY_NAMES = {
    "Nala":    "Nala (నల)",
    "Naga":    "Naga (నగ)",
    "Sala":    "Sala (సల)",
    "Bha":     "Bha (భ)",
    "Ra":      "Ra (ర)",
    "Ta":      "Ta (త)",
    "Na":      "Na (న)",
    "Ha_Gala": "Ha/Gala (హ/గల)",
}

SLOT_ACCEPT = "ACCEPT"

###############################################################################
# 2) PURE HELPER FUNCTIONS
###############################################################################


def _spawn_slot(slot, matched_so_far):
    """Spawn all candidate branches for a given slot.

    Args:
        slot: 0-2 for Indra, 3 for Surya.
        matched_so_far: tuple of gana names already completed.

    Returns:
        set of Branch tuples.
    """
    pool = INDRA_GANAS if slot <= 2 else SURYA_GANAS
    return {(slot, name, 0, matched_so_far) for name in pool}


def _get_pattern(slot, gana_name):
    """Look up the pattern tuple for a gana in the appropriate pool."""
    if slot <= 2:
        return INDRA_GANAS[gana_name]
    return SURYA_GANAS[gana_name]


def _advance(branches, symbol):
    """Advance all branches by one symbol, pruning dead branches.

    Args:
        branches: set of (slot, gana_name, sub_pos, matched_ganas) tuples.
        symbol: "U" or "I".

    Returns:
        set of surviving branches after consuming the symbol.
    """
    new_branches = set()
    for branch in branches:
        slot, gana_name, sub_pos, matched = branch

        # Accept branches don't consume more symbols
        if slot == SLOT_ACCEPT:
            continue

        pattern = _get_pattern(slot, gana_name)
        expected = pattern[sub_pos]

        if symbol != expected:
            # Dead — prune
            continue

        if sub_pos + 1 == len(pattern):
            # Gana completed — advance to next slot
            new_matched = matched + (gana_name,)
            if slot < 3:
                new_branches |= _spawn_slot(slot + 1, new_matched)
            else:
                # Surya (slot 3) completed — accept
                new_branches.add((SLOT_ACCEPT, None, None, new_matched))
        else:
            # Partial match — advance sub_pos
            new_branches.add((slot, gana_name, sub_pos + 1, matched))

    return new_branches


def _summarize_branches(branches):
    """Human-readable summary of active branches for trace output."""
    summaries = []
    for b in sorted(branches, key=str):
        if b[0] == SLOT_ACCEPT:
            summaries.append(f"ACCEPT({','.join(b[3])})")
        else:
            slot, name, sub_pos, matched = b
            pattern = _get_pattern(slot, name)
            progress = "".join(pattern[:sub_pos + 1])
            remaining = "_" * (len(pattern) - sub_pos - 1)
            summaries.append(f"S{slot}:{name}[{progress}{remaining}]")
    return summaries


###############################################################################
# 3) GanaNFA CLASS
###############################################################################


class GanaNFA:
    """
    NFA-based gana partitioner for Telugu Dwipada lines.

    Consumes U/I markers and partitions each line into
    3 Indra ganas + 1 Surya gana.
    """

    def __init__(self):
        self._reset()

    def _reset(self):
        """Clear all state for fresh processing."""
        self.branches = set()
        self.line_symbols = []
        self.output = []
        self._start_new_line()

    def _start_new_line(self):
        """Initialize branches for a new line."""
        self.branches = _spawn_slot(0, ())
        self.line_symbols = []

    def feed(self, symbol):
        """Feed a single symbol: 'U', 'I', or '\\n'."""
        if symbol == "\n":
            self._flush_line()
            self._start_new_line()
        elif symbol in ("U", "I"):
            self.line_symbols.append(symbol)
            self.branches = _advance(self.branches, symbol)

    def flush(self):
        """Signal end of input. Flush final line if any symbols consumed."""
        if self.line_symbols:
            self._flush_line()
        self.branches = set()

    def _flush_line(self):
        """Check for accepting branches and emit result."""
        accepting = [b for b in self.branches if b[0] == SLOT_ACCEPT]
        if accepting:
            matched = accepting[0][3]  # tuple of gana names
            self.output.append(self._format_partition(matched))
        else:
            self.output.append(None)  # invalid line

    def _format_partition(self, matched_ganas):
        """Build structured output for a valid partition.

        Returns:
            list of dicts, one per gana:
            [{"type": "INDRA"/"SURYA", "name": str, "pattern": tuple, "symbols": tuple}, ...]
        """
        result = []
        pos = 0
        for i, name in enumerate(matched_ganas):
            pool = INDRA_GANAS if i < 3 else SURYA_GANAS
            pattern = pool[name]
            gana_type = "INDRA" if i < 3 else "SURYA"
            gana_symbols = tuple(self.line_symbols[pos:pos + len(pattern)])
            result.append({
                "type": gana_type,
                "name": name,
                "display_name": GANA_DISPLAY_NAMES[name],
                "pattern": pattern,
                "symbols": gana_symbols,
            })
            pos += len(pattern)
        return result

    def process(self, markers):
        """Process a complete marker list and return partitions.

        Args:
            markers: list of "U", "I", and optionally "\\n" strings.

        Returns:
            list of partitions (one per line). Each partition is a list of
            gana dicts, or None if the line is invalid.
        """
        self._reset()
        for m in markers:
            self.feed(m)
        self.flush()
        return self.output

    def process_with_trace(self, markers):
        """Process with step-by-step trace for debugging.

        Returns:
            (output, trace) where trace is a list of per-step dicts.
        """
        self._reset()
        trace = []
        for m in markers:
            branches_before = len(self.branches)
            output_before = len(self.output)
            self.feed(m)
            trace.append({
                "symbol": m,
                "branches_before": branches_before,
                "branches_after": len(self.branches),
                "active_branches": _summarize_branches(self.branches),
                "emitted": self.output[output_before:],
            })
        # flush
        branches_before = len(self.branches)
        output_before = len(self.output)
        self.flush()
        trace.append({
            "symbol": "FLUSH",
            "branches_before": branches_before,
            "branches_after": 0,
            "active_branches": [],
            "emitted": self.output[output_before:],
        })
        return self.output, trace


###############################################################################
# 4) FORMATTING HELPERS
###############################################################################


def format_partition_str(partition):
    """Format a partition for display.

    Args:
        partition: list of gana dicts from GanaNFA, or None.

    Returns:
        str like "INDRA - I I I U\\tINDRA I I I I\\tINDRA U I U\\tSURYA U I"
    """
    if partition is None:
        return "INVALID (no valid gana partition found)"
    parts = []
    for g in partition:
        type_label = g["type"]
        pattern_str = " ".join(g["symbols"])
        parts.append(f"{type_label} - {pattern_str}")
    return "\t".join(parts)


def format_partition_detailed(partition):
    """Format a partition with gana names for detailed display.

    Args:
        partition: list of gana dicts from GanaNFA, or None.

    Returns:
        str with type, name, and pattern per gana.
    """
    if partition is None:
        return "INVALID (no valid gana partition found)"
    parts = []
    for g in partition:
        type_label = g["type"]
        display = g["display_name"]
        pattern_str = " ".join(g["symbols"])
        parts.append(f"{type_label} {display}: {pattern_str}")
    return "  |  ".join(parts)


###############################################################################
# 5) INLINE TESTS
###############################################################################


def _print_trace(desc, markers, result, trace, index):
    """Pretty-print a test result with trace."""
    marker_str = " ".join(m if m != "\n" else "\\n" for m in markers)
    print(f"\n{'='*70}")
    print(f"Test {index}: {desc}")
    print(f"{'='*70}")
    print(f"Input: {marker_str}")
    print()

    # Trace table
    print(f"  {'Step':>4}  {'Symbol':>6}  {'Branches':>10} -> {'Branches':>10}  Active")
    print(f"  {'----':>4}  {'------':>6}  {'--------':>10}    {'--------':>10}  ------")
    for i, t in enumerate(trace):
        sym = t["symbol"]
        active = ", ".join(t["active_branches"][:5])
        if len(t["active_branches"]) > 5:
            active += f" ... (+{len(t['active_branches'])-5} more)"
        print(f"  {i:4d}  {sym:>6}  {t['branches_before']:10d} -> {t['branches_after']:10d}  {active}")

    print()
    if result:
        for line_idx, partition in enumerate(result):
            print(f"  Line {line_idx + 1}: {format_partition_str(partition)}")
            if partition:
                print(f"           {format_partition_detailed(partition)}")
    print()


def run_tests():
    """Run comprehensive test cases."""
    nfa = GanaNFA()
    passed = 0
    failed = 0

    test_cases = [
        # (description, marker_list, expected_gana_names_per_line)
        (
            "Naga + Nala + Ra + Ha/Gala (provided test case)",
            "I I I U I I I I U I U U I".split(),
            [("Naga", "Nala", "Ra", "Ha_Gala")],
        ),
        (
            "Two-line dwipada (same pattern)",
            "I I I U I I I I U I U U I".split() + ["\n"] + "I I I U I I I I U I U U I".split(),
            [("Naga", "Nala", "Ra", "Ha_Gala"), ("Naga", "Nala", "Ra", "Ha_Gala")],
        ),
        (
            "Ta + Ra + Ta + Ha/Gala",
            "U U I U I U U U I U I".split(),
            [("Ta", "Ra", "Ta", "Ha_Gala")],
        ),
        (
            "Bha + Bha + Bha + Na",
            "U I I U I I U I I I I I".split(),
            [("Bha", "Bha", "Bha", "Na")],
        ),
        (
            "Sala + Naga + Bha + Ha/Gala",
            "I I U I I I I U U I I U I".split(),
            [("Sala", "Naga", "Bha", "Ha_Gala")],
        ),
        (
            "Nala + Nala + Nala + Na",
            "I I I I I I I I I I I I I I I".split(),
            # Ambiguous: could be multiple partitions. Accept any valid one.
            None,  # special handling
        ),
        (
            "Invalid line (too short: 5 symbols)",
            "I I U U I".split(),
            [None],  # no valid partition
        ),
        (
            "Invalid line (wrong count: 10 symbols)",
            "I I I U I I I I U I".split(),
            [None],  # no valid partition
        ),
        (
            "Ta + Ta + Ta + Na (all heavy Indra)",
            "U U I U U I U U I I I I".split(),
            [("Ta", "Ta", "Ta", "Na")],
        ),
        (
            "Naga + Sala + Ra + Na",
            "I I I U I I U I U I U I I I".split(),
            [("Naga", "Sala", "Ra", "Na")],
        ),
    ]

    for i, (desc, markers, expected_names) in enumerate(test_cases, 1):
        result, trace = nfa.process_with_trace(markers)

        if expected_names is None:
            # Ambiguous case — just check it's valid (not None)
            success = all(p is not None for p in result)
            if success:
                _print_trace(desc, markers, result, trace, i)
                actual_names = tuple(g["name"] for g in result[0])
                print(f"  PASS (ambiguous, got: {actual_names})")
            else:
                _print_trace(desc, markers, result, trace, i)
                print(f"  FAIL (expected valid partition, got None)")
        else:
            success = True
            for line_idx, expected in enumerate(expected_names):
                if line_idx >= len(result):
                    success = False
                    break
                partition = result[line_idx]
                if expected is None:
                    if partition is not None:
                        success = False
                else:
                    if partition is None:
                        success = False
                    else:
                        actual = tuple(g["name"] for g in partition)
                        if actual != expected:
                            success = False

            _print_trace(desc, markers, result, trace, i)
            if success:
                print(f"  PASS")
            else:
                actual_summary = []
                for p in result:
                    if p is None:
                        actual_summary.append("None")
                    else:
                        actual_summary.append(str(tuple(g["name"] for g in p)))
                print(f"  FAIL")
                print(f"    Expected: {expected_names}")
                print(f"    Actual:   {actual_summary}")

        if success:
            passed += 1
        else:
            failed += 1

    print(f"\n{'='*70}")
    print(f"SUMMARY: {passed} passed, {failed} failed out of {passed + failed}")
    print(f"{'='*70}")

    # Final demo: the exact test case from the user
    print(f"\n{'='*70}")
    print("DEMO: User-provided test case")
    print(f"{'='*70}")
    line1 = "I I I U I I I I U I U U I".split()
    line2 = "I I I U I I I I U I U U I".split()
    markers = line1 + ["\n"] + line2
    result = nfa.process(markers)
    for partition in result:
        print(format_partition_str(partition))

    return failed == 0


if __name__ == "__main__":
    ok = run_tests()
    raise SystemExit(0 if ok else 1)
