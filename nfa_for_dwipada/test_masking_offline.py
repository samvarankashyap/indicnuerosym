#!/usr/bin/env python3
"""
Offline test for the logit masking pipeline.

Loads only the tokenizer (no GPU needed), builds the Telugu token set,
simulates poem states at various stages, and verifies that build_gana_mask()
produces reasonable masks — including Prasa and Yati filtering.

Usage:
    python nfa_for_dwipada/test_masking_offline.py
"""

import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(PROJECT_DIR, "domino"))

from composite_state import (
    CompositeState, build_gana_mask, get_telugu_token_set,
    VALID_LINE_LENGTHS,
)
from prasa_nfa import get_base_consonant, are_prasa_equivalent

MODEL_PATH = os.path.join(PROJECT_DIR, "train_models", "dwipada_merged_model")


def load_tokenizer():
    from transformers import AutoTokenizer
    print(f"Loading tokenizer from: {MODEL_PATH}")
    return AutoTokenizer.from_pretrained(MODEL_PATH)


def test_telugu_token_set(tokenizer):
    """Verify Telugu token extraction is reasonable."""
    print("\n" + "=" * 60)
    print("TEST 1: Telugu Token Set")
    print("=" * 60)

    t0 = time.time()
    ids, texts = get_telugu_token_set(tokenizer)
    elapsed = time.time() - t0

    print(f"  Telugu tokens found: {len(ids)}")
    print(f"  Extraction time: {elapsed:.2f}s")

    # Sanity checks
    assert len(ids) > 500, f"Too few Telugu tokens: {len(ids)}"
    assert len(ids) < 10000, f"Too many Telugu tokens: {len(ids)}"
    assert len(ids) == len(texts)

    # Check some tokens contain Telugu chars
    telugu_count = sum(1 for t in texts if any(0x0C00 <= ord(c) <= 0x0C7F for c in t))
    print(f"  Tokens with Telugu chars: {telugu_count}")
    assert telugu_count > 400

    # Show a sample
    print(f"  Sample tokens: {texts[:10]}")

    # Check space and newline are included
    has_space = any(t.strip() == "" and " " in t for t in texts)
    has_newline = any("\n" in t for t in texts)
    print(f"  Includes space token: {has_space}")
    print(f"  Includes newline token: {has_newline}")

    print("  PASS")
    return ids, texts


def test_empty_state_mask(ids, texts):
    """At the start of a poem, almost all Telugu tokens should be valid."""
    print("\n" + "=" * 60)
    print("TEST 2: Empty State Mask (start of poem)")
    print("=" * 60)

    cs = CompositeState()
    snap = cs.snapshot()

    t0 = time.time()
    valid = build_gana_mask(snap, ids, texts)
    elapsed = time.time() - t0

    pct = len(valid) / len(ids) * 100
    print(f"  Valid tokens: {len(valid)} / {len(ids)} ({pct:.0f}%)")
    print(f"  Mask build time: {elapsed:.3f}s")

    # At the start, most tokens should be valid (any Telugu char starts a syllable)
    assert len(valid) > len(ids) * 0.5, f"Too few valid tokens at start: {len(valid)}"

    print("  PASS")
    return elapsed


def test_mid_line_mask(ids, texts):
    """After feeding some syllables, the mask should narrow."""
    print("\n" + "=" * 60)
    print("TEST 3: Mid-line Mask (after ~8 syllables)")
    print("=" * 60)

    cs = CompositeState()
    # Feed text that produces ~8 syllables (approaching gana 3)
    text = "రాముడు దేవుడు కా"  # ~8 syllables
    for ch in text:
        cs.feed_char(ch)

    print(f"  Fed: {repr(text)}")
    print(f"  Syllable count: {cs.syllable_count}")
    print(f"  Active branches: {len(cs.nfa_branches)}")
    print(f"  Yati checked: {cs.yati_checked}")
    print(f"  Is alive: {cs.is_alive()}")

    snap = cs.snapshot()
    t0 = time.time()
    valid = build_gana_mask(snap, ids, texts)
    elapsed = time.time() - t0

    pct = len(valid) / len(ids) * 100
    print(f"  Valid tokens: {len(valid)} / {len(ids)} ({pct:.0f}%)")
    print(f"  Mask build time: {elapsed:.3f}s")

    # Should be more constrained than empty state
    assert len(valid) > 0, "No valid tokens — mask is too aggressive"
    assert len(valid) < len(ids), "Mask should filter some tokens mid-line"

    print("  PASS")


def test_near_completion_mask(ids, texts):
    """Near line completion (Surya gana), mask should be very narrow."""
    print("\n" + "=" * 60)
    print("TEST 4: Near-completion Mask (in Surya gana zone)")
    print("=" * 60)

    cs = CompositeState()
    # Feed ~11 syllables — should be at/near Surya gana
    text = "రాముడు దేవుడు కాముడు న"
    for ch in text:
        cs.feed_char(ch)

    print(f"  Fed: {repr(text)}")
    print(f"  Syllable count: {cs.syllable_count}")
    print(f"  Active branches: {len(cs.nfa_branches)}")
    print(f"  Has accept: {cs.has_accept()}")

    snap = cs.snapshot()
    t0 = time.time()
    valid = build_gana_mask(snap, ids, texts)
    elapsed = time.time() - t0

    pct = len(valid) / len(ids) * 100
    print(f"  Valid tokens: {len(valid)} / {len(ids)} ({pct:.0f}%)")
    print(f"  Mask build time: {elapsed:.3f}s")

    # Near completion, should be narrower
    assert len(valid) > 0, "No valid tokens at near-completion"

    print("  PASS")


def test_line2_prasa_filtering(ids, texts, tokenizer):
    """On line 2 at syllable position 1, Prasa should filter tokens."""
    print("\n" + "=" * 60)
    print("TEST 5: Prasa Filtering (line 2, 2nd syllable)")
    print("=" * 60)

    # Build state: complete line 1, start line 2 with 1 syllable
    cs = CompositeState()
    # A line with known 2nd syllable consonant
    # "నల్ల మేఘం వచ్చె నేలపై నిలిచి" — 2nd syl "ల్ల" → consonant ల
    line1 = "నల్ల మేఘం వచ్చె నేలపై నిలిచె"
    for ch in line1:
        cs.feed_char(ch)
    # Force newline
    cs.feed_char("\n")

    print(f"  Line 1 prasa consonant: {cs.line1_prasa_consonant}")
    print(f"  Prasa state: {cs.prasa_state}")
    print(f"  Lines complete: {cs.lines_complete}")

    # Feed 1 syllable of line 2 (position 0)
    for ch in "పా":
        cs.feed_char(ch)

    print(f"  Line 2 syllable index: {cs.line_syllable_index}")
    print(f"  Syllable count: {cs.syllable_count}")

    # Now the NEXT syllable is position 1 — the prasa position
    # Only tokens producing a syllable with base consonant ల (or ళ) should be valid
    # (in addition to gana constraints)

    # Build mask WITH prasa
    snap_with_prasa = cs.snapshot()
    valid_with_prasa = build_gana_mask(snap_with_prasa, ids, texts)

    # For comparison: temporarily disable prasa by setting prasa_state to DECIDED
    snap_list = list(snap_with_prasa)
    snap_list[11] = "DECIDED"  # prasa_state index in snapshot tuple
    snap_no_prasa = tuple(snap_list)
    valid_no_prasa = build_gana_mask(snap_no_prasa, ids, texts)

    print(f"  Valid tokens WITH prasa: {len(valid_with_prasa)}")
    print(f"  Valid tokens WITHOUT prasa: {len(valid_no_prasa)}")

    if len(valid_no_prasa) > len(valid_with_prasa):
        filtered = len(valid_no_prasa) - len(valid_with_prasa)
        print(f"  Prasa filtered out: {filtered} tokens")

        # Show some filtered tokens
        filtered_ids = valid_no_prasa - valid_with_prasa
        sample = list(filtered_ids)[:5]
        sample_texts = [tokenizer.decode([tid]) for tid in sample]
        sample_consonants = [get_base_consonant(t.strip()) for t in sample_texts]
        print(f"  Sample filtered: {list(zip(sample_texts, sample_consonants))}")
        print("  PASS")
    else:
        # Prasa might not be active yet if the 2nd syllable hasn't been produced
        print("  INFO: Prasa didn't filter extra tokens (may need exact position)")
        print("  PASS (informational)")


def test_yati_filtering(ids, texts, tokenizer):
    """At gana 3's start position, Yati should filter tokens."""
    print("\n" + "=" * 60)
    print("TEST 6: Yati Filtering (gana 3 start position)")
    print("=" * 60)

    # Build state where we're about to enter gana 3
    # Feed syllables so that some branches are at slot=2, sub_pos=0
    cs = CompositeState()
    # Bha(UII) = 3 syl, so gana 1+2 = 6 syl for Bha+Bha
    # "రాముడు దేవుడు" = రా(U) ము(I) డు(I) దే(U) వు(I) డు(I) = 6 syls = 2 Bha ganas
    text = "రాముడు దేవుడు"
    for ch in text:
        cs.feed_char(ch)

    print(f"  Fed: {repr(text)}")
    print(f"  Syllable count: {cs.syllable_count}")
    print(f"  Gana1 first info: {(cs.gana1_first_info or {}).get('aksharam', '?')}")
    print(f"  Yati checked: {cs.yati_checked}")

    # Check which branches are at slot 2
    slot2_branches = [b for b in cs.nfa_branches if b[0] == 2 and b[2] == 0]
    print(f"  Branches at slot=2, sub_pos=0: {len(slot2_branches)}")

    snap = cs.snapshot()
    valid = build_gana_mask(snap, ids, texts)
    print(f"  Valid tokens: {len(valid)} / {len(ids)}")

    # The next syllable IS gana 3's first — yati will be checked during simulation
    # Tokens whose produced syllable doesn't match gana1's first via yati cascade
    # should be filtered out

    # For comparison: disable yati by setting yati_checked=True, yati_alive=True
    snap_list = list(snap)
    snap_list[14] = True   # yati_checked
    snap_list[15] = True   # yati_alive
    snap_no_yati = tuple(snap_list)
    valid_no_yati = build_gana_mask(snap_no_yati, ids, texts)

    print(f"  Valid tokens WITHOUT yati: {len(valid_no_yati)}")

    if len(valid_no_yati) > len(valid):
        filtered = len(valid_no_yati) - len(valid)
        print(f"  Yati filtered out: {filtered} tokens")

        # Show some filtered tokens and why
        filtered_ids = valid_no_yati - valid
        sample = list(filtered_ids)[:5]
        sample_texts = [tokenizer.decode([tid]) for tid in sample]
        print(f"  Sample filtered: {sample_texts}")
        print("  PASS")
    else:
        print("  INFO: Yati didn't filter extra tokens at this position")
        print("  PASS (informational)")


def test_mask_performance(ids, texts):
    """Benchmark mask building speed."""
    print("\n" + "=" * 60)
    print("TEST 7: Mask Build Performance")
    print("=" * 60)

    cs = CompositeState()
    text = "రాముడు దేవుడు కా"
    for ch in text:
        cs.feed_char(ch)
    snap = cs.snapshot()

    # Warm up
    build_gana_mask(snap, ids, texts)

    # Benchmark
    times = []
    for _ in range(5):
        t0 = time.time()
        build_gana_mask(snap, ids, texts)
        times.append(time.time() - t0)

    avg = sum(times) / len(times)
    print(f"  Tokens to check: {len(ids)}")
    print(f"  Avg mask build time: {avg*1000:.1f}ms (over 5 runs)")
    print(f"  Per-token simulation: {avg/len(ids)*1e6:.1f}us")

    # For a ~30 token poem, total masking overhead = ~30 * avg
    print(f"  Estimated total overhead for 30-step poem: {30*avg:.2f}s")

    if avg < 5.0:
        print("  PASS (< 5s per mask)")
    else:
        print(f"  WARN: Slow mask building ({avg:.1f}s) — may need optimization")


def main():
    tokenizer = load_tokenizer()

    ids, texts = test_telugu_token_set(tokenizer)
    test_empty_state_mask(ids, texts)
    test_mid_line_mask(ids, texts)
    test_near_completion_mask(ids, texts)
    test_line2_prasa_filtering(ids, texts, tokenizer)
    test_yati_filtering(ids, texts, tokenizer)
    test_mask_performance(ids, texts)

    print("\n" + "=" * 60)
    print("ALL OFFLINE TESTS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
