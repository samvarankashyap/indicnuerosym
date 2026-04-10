"""Constrained LogitsProcessor for Dwipada poetry generation.

Implements Algorithm 1 from the Chandomitra paper, adapted for Telugu
Dwipada meter. At each generation step, filters top-k tokens against
valid gana patterns using a prefix trie, enforcing metrical constraints
(gana structure, prasa, yati) without modifying the model.

Usage with HuggingFace model.generate():
    processor = DwipadaConstrainedLogitsProcessor(tokenizer)
    processor.prompt_length = input_ids.shape[1]
    outputs = model.generate(
        input_ids,
        logits_processor=LogitsProcessorList([processor]),
    )
"""

from __future__ import annotations

import logging
import re
from typing import Optional

import torch
from transformers import LogitsProcessor

from dwipada.core.analyzer import (
    are_prasa_equivalent,
    check_yati_maitri_simple,
    get_base_consonant,
    get_first_letter,
)
from dwipada.training.constrained.generation_state import DwipadaGenerationState
from dwipada.training.constrained.pattern_trie import DwipadaPatternTrie
from dwipada.training.constrained.syllable_utils import (
    get_markers_with_stability,
    get_pure_aksharalu,
)

logger = logging.getLogger(__name__)

# Telugu Unicode range: \u0C00-\u0C7F
_TELUGU_RE = re.compile(r"[\u0C00-\u0C7F]")

# Min/max pure syllables for a valid dwipada line
_MIN_LINE_SYLLABLES = 11  # UII+UII+UII+UI
_MAX_LINE_SYLLABLES = 15  # IIII+IIII+IIII+III


def _has_telugu(text: str) -> bool:
    """Check if text contains any Telugu character."""
    return bool(_TELUGU_RE.search(text))


class DwipadaConstrainedLogitsProcessor(LogitsProcessor):
    """HuggingFace LogitsProcessor that enforces Dwipada metrical constraints.

    Constraint layers (applied in order):
    1. HARD: Only Telugu tokens and spaces are allowed (masks everything else)
    2. SOFT: Gana pattern must be a valid prefix in the trie
    3. SOFT: Prasa rhyme constraint on line 2
    4. SOFT: Yati alliteration constraint

    If no token passes all layers, we relax from layer 4 upward until
    at least one token is valid. Layer 1 (Telugu-only) is never relaxed.
    """

    def __init__(
        self,
        tokenizer,
        initial_k: int = 25,
        max_k: int = 200,
        enable_prasa: bool = True,
        enable_yati: bool = True,
        hard_constraints: bool = False,
    ):
        self.tokenizer = tokenizer
        self.initial_k = initial_k
        self.max_k = max_k
        self.enable_prasa = enable_prasa
        self.enable_yati = enable_yati
        self.hard_constraints = hard_constraints

        self.pattern_trie = DwipadaPatternTrie()
        self.state = DwipadaGenerationState(self.pattern_trie)

        # Must be set before generation starts
        self.prompt_length: int = 0

        # Pre-compute special token IDs
        newline_ids = tokenizer.encode("\n", add_special_tokens=False)
        self._newline_token_id: Optional[int] = newline_ids[-1] if newline_ids else None
        self._eos_token_id: int = tokenizer.eos_token_id

        # Pre-compute Telugu token mask (True = token contains Telugu chars or is space)
        self._telugu_mask = self._build_telugu_mask()

    def _build_telugu_mask(self) -> torch.BoolTensor:
        """Build a boolean mask: True for tokens containing Telugu chars or single space."""
        vocab_size = self.tokenizer.vocab_size
        mask = torch.zeros(vocab_size, dtype=torch.bool)
        for tid in range(vocab_size):
            token_text = self.tokenizer.decode([tid])
            if token_text == " " or _has_telugu(token_text):
                mask[tid] = True
        # Also allow newline and EOS (handled separately)
        if self._newline_token_id is not None and self._newline_token_id < vocab_size:
            mask[self._newline_token_id] = True
        if self._eos_token_id < vocab_size:
            mask[self._eos_token_id] = True
        return mask

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """Filter logits to enforce Dwipada metrical constraints."""
        if self.state.poem_complete:
            return self._force_eos(scores)

        # Decode generated text so far (excluding prompt)
        generated_ids = input_ids[0, self.prompt_length :]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Update state
        self.state.update(generated_text)

        # Check if current line is metrically complete
        current_line_text = self.state.get_current_line_text()
        pure_count = len(get_pure_aksharalu(current_line_text))

        if pure_count >= _MIN_LINE_SYLLABLES and self.state.is_current_line_complete():
            if self.state.current_line == 1:
                return self._force_newline(scores)
            else:
                self.state.mark_poem_complete()
                return self._force_eos(scores)

        # HARD GATE: Mask all non-Telugu tokens unconditionally
        telugu_mask = self._telugu_mask.to(scores.device)
        # Handle case where scores has more columns than vocab (padding)
        if scores.shape[1] > telugu_mask.shape[0]:
            extended = torch.zeros(scores.shape[1], dtype=torch.bool, device=scores.device)
            extended[: telugu_mask.shape[0]] = telugu_mask
            telugu_mask = extended
        scores[0, ~telugu_mask[: scores.shape[1]]] = float("-inf")

        # Mask newline/EOS (they're allowed through Telugu gate but controlled separately)
        if self._newline_token_id is not None:
            scores[0, self._newline_token_id] = float("-inf")
        scores[0, self._eos_token_id] = float("-inf")

        # SOFT GATES: Apply gana/prasa/yati constraints on top-k
        return self._filter_tokens(scores, current_line_text)

    def _filter_tokens(
        self, scores: torch.FloatTensor, current_line_text: str
    ) -> torch.FloatTensor:
        """Apply metrical constraints to top-k tokens.

        Progressively relaxes constraints if no token passes:
        1. Try with all constraints (gana + prasa + yati)
        2. Try with gana + prasa only
        3. Try with gana only
        4. Allow any Telugu token (skip gana constraint)

        In hard_constraints mode, relaxation is disabled — instead we scan
        the entire Telugu vocabulary to find any token satisfying all constraints.
        """
        # Save original scores before masking (needed for hard mode full scan)
        original_scores = scores.clone() if self.hard_constraints else None

        k = self.initial_k

        while k <= self.max_k:
            top_k = torch.topk(scores[0], min(k, (scores[0] > float("-inf")).sum().item()))
            top_k_indices = top_k.indices
            any_valid = False

            for token_id in top_k_indices:
                tid = token_id.item()
                token_text = self.tokenizer.decode([tid])
                tentative_text = current_line_text + token_text

                if not self._check_all_constraints(tentative_text):
                    scores[0, tid] = float("-inf")
                else:
                    any_valid = True

            if any_valid:
                return scores

            # Expand k
            k = min(k * 2, self.max_k + 1)

        if self.hard_constraints:
            # Hard mode: restore scores and scan ALL Telugu tokens (no relaxation)
            logger.debug(
                "Hard constraints: scanning full vocabulary for line: %r",
                current_line_text[:50],
            )
            scores = original_scores
            valid_mask = scores[0] > float("-inf")
            valid_indices = valid_mask.nonzero(as_tuple=True)[0]
            any_valid = False
            for tid in valid_indices:
                token_text = self.tokenizer.decode([tid.item()])
                tentative = current_line_text + token_text
                if not self._check_all_constraints(tentative):
                    scores[0, tid.item()] = float("-inf")
                else:
                    any_valid = True
            if any_valid:
                return scores
            # Truly nothing works — log warning and allow any Telugu token
            logger.warning(
                "Hard constraints: no token in entire vocabulary satisfies "
                "all constraints. Falling back to Telugu-only. Line: %r",
                current_line_text[:60],
            )
            return original_scores

        # RELAXATION: No token passes all constraints at max k.
        # Restore scores and try with fewer constraints.
        logger.debug(
            "Relaxing constraints for line text: %r", current_line_text[:50]
        )
        return self._relaxed_filter(scores, current_line_text)

    def _relaxed_filter(
        self, scores: torch.FloatTensor, current_line_text: str
    ) -> torch.FloatTensor:
        """Try progressively weaker constraint sets.

        Order: gana+prasa → gana only → Telugu only (no gana).
        Telugu-only gate is always enforced (already applied in __call__).
        """
        # Collect all non-masked token IDs
        valid_mask = scores[0] > float("-inf")
        valid_indices = valid_mask.nonzero(as_tuple=True)[0]

        # Try gana-only (no prasa/yati)
        for tid in valid_indices:
            token_text = self.tokenizer.decode([tid.item()])
            tentative = current_line_text + token_text
            if self._check_gana_constraint(tentative):
                # At least one token passes gana — mask others that don't
                for tid2 in valid_indices:
                    t2 = self.tokenizer.decode([tid2.item()])
                    tent2 = current_line_text + t2
                    if not self._check_gana_constraint(tent2):
                        scores[0, tid2.item()] = float("-inf")
                return scores

        # Nothing passes gana either — allow any Telugu token (already filtered)
        logger.warning(
            "No token satisfies gana constraint. Allowing any Telugu token. "
            "Line: %r",
            current_line_text[:60],
        )
        return scores

    def _check_all_constraints(self, tentative_text: str) -> bool:
        """Check gana + prasa + yati constraints."""
        if not self._check_gana_constraint(tentative_text):
            return False

        pure = get_pure_aksharalu(tentative_text)
        if len(pure) > _MAX_LINE_SYLLABLES:
            return False

        if self.enable_prasa and self.state.current_line == 2:
            if not self._check_prasa_constraint(tentative_text):
                return False

        if self.enable_yati:
            if not self._check_yati_constraint(tentative_text):
                return False

        return True

    def _check_gana_constraint(self, tentative_text: str) -> bool:
        """Check if marker sequence is a valid prefix of any line pattern."""
        stable_prefix, could_be_laghu, could_be_guru = get_markers_with_stability(
            tentative_text
        )

        if stable_prefix and not self.pattern_trie.is_valid_prefix(stable_prefix):
            return False

        if not stable_prefix:
            return True  # Too few syllables

        valid = False
        if could_be_laghu:
            is_valid, _ = self.pattern_trie.is_prefix_or_complete(stable_prefix + "I")
            valid |= is_valid
        if could_be_guru:
            is_valid, _ = self.pattern_trie.is_prefix_or_complete(stable_prefix + "U")
            valid |= is_valid

        return valid

    def _check_prasa_constraint(self, tentative_text: str) -> bool:
        """Check 2nd syllable consonant rhyme on line 2."""
        if self.state.line1_prasa_consonant is None:
            return True

        pure = get_pure_aksharalu(tentative_text)
        if len(pure) < 3:
            return True

        consonant = get_base_consonant(pure[1])
        if consonant is None:
            return True

        return are_prasa_equivalent(consonant, self.state.line1_prasa_consonant)

    def _check_yati_constraint(self, tentative_text: str) -> bool:
        """Check gana 1 ↔ gana 3 first letter alliteration."""
        pure = get_pure_aksharalu(tentative_text)
        if len(pure) < 7:
            return True

        first_letter = get_first_letter(pure[0])
        if first_letter is None:
            return True

        stable_prefix, could_l, could_u = get_markers_with_stability(tentative_text)
        if not stable_prefix:
            return True

        candidates = []
        if could_l:
            candidates.append(stable_prefix + "I")
        if could_u:
            candidates.append(stable_prefix + "U")

        for candidate in candidates:
            if self._any_valid_yati_partition(candidate, pure, first_letter):
                return True

        return False

    def _any_valid_yati_partition(
        self,
        marker_prefix: str,
        pure_aksharalu: list[str],
        first_letter: str,
    ) -> bool:
        """Check if any partition allows yati-compatible gana 3 start."""
        node_valid, _ = self.pattern_trie.is_prefix_or_complete(marker_prefix)
        if not node_valid:
            return False

        prefix_len = len(marker_prefix)
        for g3_start in range(6, min(9, prefix_len + 1)):
            if g3_start >= len(pure_aksharalu):
                continue

            g3_first_letter = get_first_letter(pure_aksharalu[g3_start])
            if g3_first_letter is None:
                continue

            is_match, _ = check_yati_maitri_simple(first_letter, g3_first_letter)
            if is_match:
                g1_g2_prefix = marker_prefix[:g3_start]
                if self._is_valid_g1_g2_split(g1_g2_prefix):
                    return True

        return False

    @staticmethod
    def _is_valid_g1_g2_split(g1_g2_markers: str) -> bool:
        """Check if marker string splits into 2 valid Indra ganas."""
        from dwipada.training.constrained.pattern_trie import INDRA_PATTERNS

        total = len(g1_g2_markers)
        for g1_len in (3, 4):
            if g1_len > total:
                continue
            g1 = g1_g2_markers[:g1_len]
            g2 = g1_g2_markers[g1_len:]
            if g1 in INDRA_PATTERNS and g2 in INDRA_PATTERNS:
                return True
        return False

    def _force_newline(self, scores: torch.FloatTensor) -> torch.FloatTensor:
        """Force newline token (end of line 1)."""
        scores[0, :] = float("-inf")
        if self._newline_token_id is not None:
            scores[0, self._newline_token_id] = 0.0
        return scores

    def _force_eos(self, scores: torch.FloatTensor) -> torch.FloatTensor:
        """Force EOS token (end of poem)."""
        scores[0, :] = float("-inf")
        scores[0, self._eos_token_id] = 0.0
        return scores
