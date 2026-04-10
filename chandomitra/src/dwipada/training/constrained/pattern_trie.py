"""Prefix trie for valid Dwipada line patterns.

A Dwipada line consists of 4 ganas: 3 Indra + 1 Surya.
This module pre-computes all 432 valid I/U (Laghu/Guru) patterns
and stores them in a trie for O(n) prefix matching during generation.

Indra ganas (6 patterns, 3-4 syllables each):
    IIII (Nala), IIIU (Naga), IIUI (Sala), UII (Bha), UIU (Ra), UUI (Ta)

Surya ganas (2 patterns, 2-3 syllables each):
    III (Na), UI (Ha/Gala)

Total valid line patterns: 6 x 6 x 6 x 2 = 432
Pattern lengths range from 11 (UII+UII+UII+UI) to 15 (IIII+IIII+IIII+III).
"""

from __future__ import annotations

from typing import Optional

# Gana patterns as I/U strings (keys from analyzer.py INDRA_GANAS / SURYA_GANAS)
INDRA_PATTERNS = ["IIII", "IIIU", "IIUI", "UII", "UIU", "UUI"]
SURYA_PATTERNS = ["III", "UI"]


class TrieNode:
    """A single node in the pattern trie."""

    __slots__ = ("children", "is_terminal", "gana_boundaries_list")

    def __init__(self):
        self.children: dict[str, TrieNode] = {}  # 'I' or 'U' -> child
        self.is_terminal: bool = False
        # At terminal nodes: list of boundary tuples for all patterns ending here.
        # Each tuple = (g1_end, g2_end, g3_end) — cumulative syllable positions.
        # e.g. for Nala+Bha+Ra+Na: (4, 7, 10) with total length 13.
        self.gana_boundaries_list: list[tuple[int, int, int]] = []


class DwipadaPatternTrie:
    """Trie of all valid I/U patterns for a single Dwipada line.

    Supports:
        - Checking if a marker string is a valid prefix of any pattern
        - Checking if a marker string is a complete valid pattern
        - Querying which next markers (I, U, or both) are valid continuations
        - Retrieving gana boundary positions for complete patterns
    """

    def __init__(self):
        self.root = TrieNode()
        self._pattern_count = 0
        self._build()

    def _build(self) -> None:
        """Enumerate all 432 valid line patterns and insert into trie."""
        for i1 in INDRA_PATTERNS:
            for i2 in INDRA_PATTERNS:
                for i3 in INDRA_PATTERNS:
                    for s in SURYA_PATTERNS:
                        pattern = i1 + i2 + i3 + s
                        g1_end = len(i1)
                        g2_end = g1_end + len(i2)
                        g3_end = g2_end + len(i3)
                        boundaries = (g1_end, g2_end, g3_end)
                        self._insert(pattern, boundaries)

    def _insert(self, pattern: str, boundaries: tuple[int, int, int]) -> None:
        """Insert a single I/U pattern with its gana boundaries."""
        node = self.root
        for char in pattern:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_terminal = True
        node.gana_boundaries_list.append(boundaries)
        self._pattern_count += 1

    def _traverse(self, prefix: str) -> Optional[TrieNode]:
        """Walk the trie along prefix, returning the final node or None."""
        node = self.root
        for char in prefix:
            if char not in node.children:
                return None
            node = node.children[char]
        return node

    def is_valid_prefix(self, prefix: str) -> bool:
        """Return True if prefix matches the start of any valid pattern.

        An empty prefix is always valid (matches everything).
        """
        if not prefix:
            return True
        return self._traverse(prefix) is not None

    def is_complete_pattern(self, pattern: str) -> bool:
        """Return True if pattern exactly matches a valid full-line pattern."""
        node = self._traverse(pattern)
        return node is not None and node.is_terminal

    def get_valid_next_markers(self, prefix: str) -> set[str]:
        """Return which markers ('I', 'U', or both) can follow the given prefix.

        Returns empty set if prefix is invalid (no continuations possible).
        If prefix is a complete pattern, returns empty set (line is done).
        """
        node = self._traverse(prefix)
        if node is None:
            return set()
        return set(node.children.keys())

    def get_gana_boundaries(self, pattern: str) -> list[tuple[int, int, int]]:
        """Return all possible gana boundary tuples for a complete pattern.

        Each tuple is (g1_end, g2_end, g3_end) — cumulative syllable positions.
        For example, (4, 7, 10) means gana1 is syllables 0-3, gana2 is 4-6,
        gana3 is 7-9, and surya is 10 onwards.

        Returns empty list if pattern is not a complete valid pattern.
        """
        node = self._traverse(pattern)
        if node is None or not node.is_terminal:
            return []
        return node.gana_boundaries_list

    def has_continuation(self, prefix: str) -> bool:
        """Return True if the prefix can be extended (is not at a dead end).

        Unlike is_valid_prefix, this also checks that the node has children
        (i.e., the pattern is not yet complete OR there are longer patterns).
        """
        node = self._traverse(prefix)
        return node is not None and len(node.children) > 0

    def is_prefix_or_complete(self, prefix: str) -> tuple[bool, bool]:
        """Check both prefix validity and completeness in one traversal.

        Returns:
            (is_valid, is_complete): Whether the prefix exists in the trie,
            and whether it is a terminal (complete pattern).
        """
        node = self._traverse(prefix)
        if node is None:
            return False, False
        return True, node.is_terminal

    @property
    def pattern_count(self) -> int:
        """Total number of valid patterns stored."""
        return self._pattern_count
