"""Generation state machine for Dwipada constrained decoding.

Tracks the metrical state of a poem being generated autoregressively:
which line we're on, the prasa consonant from line 1, and whether
each line / the full poem is complete.
"""

from __future__ import annotations

from typing import Optional

from dwipada.core.analyzer import get_base_consonant, get_first_letter, ignorable_chars, split_aksharalu
from dwipada.training.constrained.pattern_trie import DwipadaPatternTrie
from dwipada.training.constrained.syllable_utils import get_full_markers, get_pure_aksharalu


class DwipadaGenerationState:
    """Tracks metrical state during autoregressive Dwipada generation.

    A Dwipada poem has exactly 2 lines separated by a newline.
    This state machine tracks:
        - Which line we're currently generating (1 or 2)
        - Line 1's text and prasa consonant (for line 2 constraint)
        - Whether each line / the whole poem is complete
    """

    def __init__(self, pattern_trie: DwipadaPatternTrie):
        self.pattern_trie = pattern_trie

        # Line tracking
        self.current_line: int = 1
        self.line1_text: str = ""
        self.line2_text: str = ""

        # Prasa: 2nd syllable's base consonant from line 1
        self.line1_prasa_consonant: Optional[str] = None

        # Yati: first letter of current line (for matching with gana 3's first letter)
        self.line1_first_letter: Optional[str] = None
        self.line2_first_letter: Optional[str] = None

        # Completion flags
        self.line1_complete: bool = False
        self.poem_complete: bool = False

    def update(self, full_generated_text: str) -> None:
        """Update state from the full generated text so far.

        Called at each generation step. Detects line boundaries (newlines)
        and extracts prasa/yati information.
        """
        # Split on first newline to get line 1 and line 2
        parts = full_generated_text.split("\n", 1)
        self.line1_text = parts[0]

        if len(parts) > 1:
            # We've crossed the newline — now generating line 2
            self.line2_text = parts[1]
            if not self.line1_complete:
                self._freeze_line1()
            self.current_line = 2
        else:
            self.line2_text = ""
            self.current_line = 1

    def _freeze_line1(self) -> None:
        """Extract prasa and yati info from completed line 1."""
        self.line1_complete = True
        pure = get_pure_aksharalu(self.line1_text)

        # Prasa: base consonant of 2nd syllable
        if len(pure) >= 2:
            self.line1_prasa_consonant = get_base_consonant(pure[1])

        # Yati: first letter of line
        if pure:
            self.line1_first_letter = get_first_letter(pure[0])

    def get_current_line_text(self) -> str:
        """Return the text of the line currently being generated."""
        if self.current_line == 1:
            return self.line1_text
        return self.line2_text

    def is_current_line_complete(self) -> bool:
        """Check if the current line's markers form a complete valid pattern."""
        line_text = self.get_current_line_text()
        if not line_text.strip():
            return False
        markers = get_full_markers(line_text)
        return self.pattern_trie.is_complete_pattern(markers)

    def mark_poem_complete(self) -> None:
        """Mark the poem as fully generated."""
        self.poem_complete = True

    def get_line2_first_letter(self) -> Optional[str]:
        """Get first letter of line 2 (for yati checking)."""
        if self.line2_first_letter is not None:
            return self.line2_first_letter
        pure = get_pure_aksharalu(self.line2_text)
        if pure:
            self.line2_first_letter = get_first_letter(pure[0])
        return self.line2_first_letter
