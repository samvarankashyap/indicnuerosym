# -*- coding: utf-8 -*-
"""
Gana Pattern NFA for Kannada Utsaha Ragale.
=============================================

Reads a stream of Guru (U) / Laghu (I) markers — the output of
GuruLaghuClassifier — and partitions each line into exactly 4 ganas,
identifying each gana by pattern name.

This is Stage 3 of the NFA constrained-decoding pipeline:

    Raw text
       |
       v
    [SyllableAssembler FST]   -- Stage 1: Unicode chars -> syllables
       |
       v
    [GuruLaghuClassifier FST] -- Stage 2: syllables -> U/I markers
       |
       v
    [Gana NFA]                -- Stage 3: U/I stream -> gana partition (THIS)

-------------------------------------------------------------------------------
FORMAL LANGUAGE
-------------------------------------------------------------------------------

Each Utsaha Ragale line must match:

    L_line = (III | IIU)^3 · IIU
              |_ ganas 1-3 _|  |_ gana 4 (must end Guru) _|

Valid gana patterns (each 3 syllables):

    III  (ಲಘು-ಲಘು-ಲಘು)    I I I    3 matras
    IIU  (ಲಘು-ಲಘು-ಗುರು)   I I U    4 matras

Forbidden pattern (never in the pool):

    IU   (ಲಘು-ಗುರು)        — breaks rhythmic flow

Each line = 4 ganas × 3 syllables = 12 syllables exactly.
Gana 4 is constrained to IIU only (to enforce guru ending).

-------------------------------------------------------------------------------
HOW THE NFA WORKS
-------------------------------------------------------------------------------

The NFA non-deterministically guesses which gana pattern is being read.
After the first two symbols:
  - I I → could be III (next: I) or IIU (next: U)

State = set of active branches.  Each branch is a tuple:
    (slot, gana_name, sub_pos, matched_ganas)

Dead state detection prunes impossible branches after each symbol.
When a gana completes, branches for the next slot are spawned.
A line is valid when any branch reaches ACCEPT (all 4 slots filled).

-------------------------------------------------------------------------------
USAGE
-------------------------------------------------------------------------------

    from gana_nfa import GanaNFA

    nfa = GanaNFA()
    result = nfa.process("I I U I I U I I U I I U".split())
    # => [[{type, name, pattern, symbols}, ...]]

    # Two-line ragale (with newline separator):
    markers = "I I U I I U I I U I I U".split() + ["\\n"] + \
              "I I U I I U I I U I I U".split()
    result = nfa.process(markers)

"""

###############################################################################
# 1) GANA PATTERN CONSTANTS
###############################################################################

RAGALE_GANAS = {
    "III": ("I", "I", "I"),
    "IIU": ("I", "I", "U"),
}

GANA_DISPLAY_NAMES = {
    "III": "III (ಲಘು-ಲಘು-ಲಘು)",
    "IIU": "IIU (ಲಘು-ಲಘು-ಗುರು)",
}

SLOT_ACCEPT = "ACCEPT"

###############################################################################
# 2) PURE HELPER FUNCTIONS
###############################################################################


def _spawn_slot(slot, matched_so_far):
    """Spawn all candidate branches for a given slot.

    Args:
        slot: 0-2 for general ganas, 3 for the final gana (IIU only).
        matched_so_far: tuple of gana names already completed.

    Returns:
        set of Branch tuples.
    """
    if slot == 3:
        # Gana 4: only IIU allowed (enforces guru ending)
        return {(3, "IIU", 0, matched_so_far)}
    return {(slot, name, 0, matched_so_far) for name in RAGALE_GANAS}


def _get_pattern(gana_name):
    """Look up the pattern tuple for a gana."""
    return RAGALE_GANAS[gana_name]


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

        if slot == SLOT_ACCEPT:
            continue

        pattern = _get_pattern(gana_name)
        expected = pattern[sub_pos]

        if symbol != expected:
            continue

        if sub_pos + 1 == len(pattern):
            new_matched = matched + (gana_name,)
            if slot < 3:
                new_branches |= _spawn_slot(slot + 1, new_matched)
            else:
                new_branches.add((SLOT_ACCEPT, None, None, new_matched))
        else:
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
            pattern = _get_pattern(name)
            progress = "".join(pattern[:sub_pos + 1])
            remaining = "_" * (len(pattern) - sub_pos - 1)
            summaries.append(f"S{slot}:{name}[{progress}{remaining}]")
    return summaries


###############################################################################
# 3) GanaNFA CLASS
###############################################################################


class GanaNFA:
    """
    NFA-based gana partitioner for Kannada Utsaha Ragale lines.

    Consumes U/I markers and partitions each line into 4 ganas
    (ganas 1-3: III or IIU, gana 4: IIU only).
    """

    def __init__(self):
        self._reset()

    def _reset(self):
        self.branches = set()
        self.line_symbols = []
        self.output = []
        self._start_new_line()

    def _start_new_line(self):
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
        if self.line_symbols:
            self._flush_line()
        self.branches = set()

    def _flush_line(self):
        accepting = [b for b in self.branches if b[0] == SLOT_ACCEPT]
        if accepting:
            matched = accepting[0][3]
            self.output.append(self._format_partition(matched))
        else:
            self.output.append(None)

    def _format_partition(self, matched_ganas):
        partition = []
        for i, name in enumerate(matched_ganas):
            pattern = _get_pattern(name)
            partition.append({
                "slot": i,
                "name": name,
                "display_name": GANA_DISPLAY_NAMES.get(name, name),
                "pattern": pattern,
                "symbols": list(pattern),
            })
        return partition

    def process(self, markers):
        self._reset()
        for m in markers:
            self.feed(m)
        self.flush()
        return self.output

    def process_with_trace(self, markers):
        self._reset()
        trace = []

        items = list(markers) + [None]

        for item in items:
            branches_before = len(self.branches)
            summaries_before = _summarize_branches(self.branches)

            if item is None:
                self.flush()
            else:
                self.feed(item)

            branches_after = len(self.branches)
            summaries_after = _summarize_branches(self.branches)

            trace.append({
                "symbol": item if item is not None else "∎",
                "branches_before": branches_before,
                "branches_after": branches_after,
                "active_before": summaries_before,
                "active_after": summaries_after,
            })

        return self.output, trace


def format_partition_str(partition):
    """Format a gana partition for display."""
    if partition is None:
        return "INVALID (no valid partition found)"
    parts = []
    for g in partition:
        parts.append(f"{g['name']}({''.join(g['pattern'])})")
    return " + ".join(parts)


###############################################################################
# 4) TESTS
###############################################################################

def run_tests():
    nfa = GanaNFA()

    test_cases = [
        ("All IIU (typical ragale)",
         "I I U I I U I I U I I U".split(),
         ("IIU", "IIU", "IIU", "IIU")),

        ("Mixed III+IIU",
         "I I I I I U I I I I I U".split(),
         ("III", "IIU", "III", "IIU")),

        ("III III III IIU (guru ending)",
         "I I I I I I I I I I I U".split(),
         ("III", "III", "III", "IIU")),

        ("All III — FAIL (no guru ending)",
         "I I I I I I I I I I I I".split(),
         None),

        ("Too short — 10 syllables — FAIL",
         "I I U I I U I I U I".split(),
         None),

        ("IU in stream — FAIL (forbidden, kills all branches)",
         "I U I I I U I I U I I U".split(),
         None),

        ("Two-line valid ragale",
         "I I U I I U I I U I I U".split() + ["\n"] +
         "I I I I I I I I I I I U".split(),
         [("IIU", "IIU", "IIU", "IIU"), ("III", "III", "III", "IIU")]),
    ]

    passed = 0
    failed = 0

    for i, (desc, markers, expected) in enumerate(test_cases, 1):
        result = nfa.process(markers)

        if isinstance(expected, tuple):
            # Single line — check first partition
            p = result[0]
            if p is None:
                match = expected is None
            else:
                got = tuple(g["name"] for g in p)
                match = got == expected
        elif expected is None:
            match = result[0] is None
        elif isinstance(expected, list):
            # Multi-line
            match = True
            for j, exp in enumerate(expected):
                p = result[j] if j < len(result) else None
                if p is None:
                    match = exp is None
                else:
                    got = tuple(g["name"] for g in p)
                    if got != exp:
                        match = False
        else:
            match = False

        status = "PASS" if match else "FAIL"
        display = format_partition_str(result[0]) if result else "NO OUTPUT"
        print(f"  Test {i:2d}: {desc:<45s}  [{status}]  {display}")
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
