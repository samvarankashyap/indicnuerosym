"""Constrained decoding for Telugu Dwipada poetry generation.

Adapts the Chandomitra paper's constrained decoding approach (Algorithm 1)
for Dwipada meter: at each generation step, filter top-k tokens against
valid gana patterns using a prefix trie, enforcing metrical constraints
(gana structure, prasa, yati) without modifying the model.
"""

from dwipada.training.constrained.pattern_trie import DwipadaPatternTrie
from dwipada.training.constrained.syllable_utils import get_markers_with_stability
from dwipada.training.constrained.generation_state import DwipadaGenerationState
from dwipada.training.constrained.logits_processor import DwipadaConstrainedLogitsProcessor

__all__ = [
    "DwipadaPatternTrie",
    "DwipadaConstrainedLogitsProcessor",
    "DwipadaGenerationState",
    "get_markers_with_stability",
]
