"""
Data Feasibility Audit
========================
Checks which profiles are actually supportable given the populated fields
in your corpus. Run this BEFORE building any training datasets.

Usage:
    python data_audit.py --input dwipada_augmented_dataset.json

Output:
    - Field coverage report (% records with each field populated)
    - Per-profile feasibility verdict with exact counts
    - Recommended profiles to keep vs drop
"""

import json
import re
import argparse
from pathlib import Path
from collections import defaultdict


# ──────────────────────────────────────────────
# LOADER
# ──────────────────────────────────────────────

def load_data(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        first = f.read(1); f.seek(0)
        if first == "[":
            return json.load(f)
        return [json.loads(l) for l in f if l.strip()]


# ──────────────────────────────────────────────
# FIELD CHECKERS
# ──────────────────────────────────────────────

def has_poem(rec):
    p = rec.get("poem", "")
    return bool(p and len(p.strip()) > 5)

def has_two_lines(rec):
    lines = [l.strip() for l in rec.get("poem","").split("\n") if l.strip()]
    return len(lines) >= 2

def has_english_meaning(rec):
    return bool(rec.get("english_meaning","").strip())

def has_telugu_meaning(rec):
    return bool(rec.get("telugu_meaning","").strip())

def has_wtw_rich(rec):
    return len(rec.get("word_to_word_meaning", {})) >= 3

def has_wtw_any(rec):
    return len(rec.get("word_to_word_meaning", {})) >= 1

def has_chandassu(rec):
    ca = rec.get("chandassu_analysis", {})
    return bool(ca.get("line_1") and ca.get("line_2"))

def has_prasa(rec):
    try:
        pr = rec["chandassu_analysis"]["line_1"]["prasa_check"]
        m  = re.search(r"'([\u0C00-\u0C7F])'", pr)
        return bool(m)
    except Exception:
        return False

def has_yati(rec):
    try:
        yt = rec["chandassu_analysis"]["line_1"]["yati_check"]
        letters = re.findall(r"'([\u0C00-\u0C7F])'", yt)
        return len(letters) >= 2
    except Exception:
        return False

def has_gana_breakdown(rec):
    try:
        bd = rec["chandassu_analysis"]["line_1"]["breakdown"]
        return bool(bd and len(bd) > 10)
    except Exception:
        return False

def has_tags_telugu(rec):
    tags = rec.get("tags_telugu", [])
    return len(tags) >= 2

def has_tags_english(rec):
    tags = rec.get("tags_english", [])
    return len(tags) >= 2

def has_source(rec):
    return bool(rec.get("source","").strip())

def line1_len_ok(rec):
    """Line 1 long enough to mask half of it for RT6."""
    line1 = rec.get("poem","").split("\n")[0].strip() if rec.get("poem") else ""
    return len(line1) >= 8

def tel_meaning_descriptive(rec):
    """Telugu meaning long enough to be useful as a riddle hint (GN7)."""
    return len(rec.get("telugu_meaning","").strip()) >= 20


# ──────────────────────────────────────────────
# PROFILE FEASIBILITY DEFINITIONS
# ──────────────────────────────────────────────
# Each profile lists the checks that ALL must pass
# for that record to be usable for that profile.

PROFILES = {
    # ── ANALYSIS ──────────────────────────────
    "AN1 Syllabify full poem":          [has_poem, has_chandassu, has_gana_breakdown],
    "AN2 Identify Gana names":          [has_poem, has_chandassu, has_gana_breakdown],
    "AN3 Meter detection":              [has_poem, has_chandassu],
    "AN4 WTW gloss (sandhi breaks)":    [has_poem, has_wtw_rich],
    "AN5 WTW + Telugu bhavam":          [has_poem, has_wtw_rich, has_telugu_meaning],
    "AN6 Single line syllabify":        [has_two_lines, has_gana_breakdown],

    # ── RETRIEVAL ─────────────────────────────
    "RT1 First line → full poem":       [has_two_lines],
    "RT2 Last line → full poem":        [has_two_lines],
    "RT3 Fragment → full poem":         [has_two_lines],
    "RT4 English meaning → poem":       [has_poem, has_english_meaning],
    "RT5 Telugu meaning → poem":        [has_poem, has_telugu_meaning],
    "RT6 Masked line → complete":       [has_two_lines, line1_len_ok],

    # ── GENERATION ────────────────────────────
    "GN1 Theme (English short) → poem": [has_poem, has_english_meaning],
    "GN2 Full English meaning → poem":  [has_poem, has_english_meaning],
    "GN3 Telugu meaning → poem":        [has_poem, has_telugu_meaning],
    "GN4 Line1 → complete Line2":       [has_two_lines],
    "GN5 Theme + Prasa → poem":         [has_poem, has_english_meaning, has_prasa],
    "GN6 Theme + Yati → poem":          [has_poem, has_english_meaning, has_yati],
    "GN7 Riddle poem (RPFW)":           [has_poem, has_telugu_meaning, tel_meaning_descriptive],
    "GN8 Samasya (last line given)":    [has_two_lines, has_telugu_meaning],

    # ── SUPPORT (overhead check) ───────────────
    "SP1 Prasa error detect+fix":       [has_poem, has_chandassu, has_prasa],
    "SP2 Yati error detect+fix":        [has_poem, has_chandassu, has_yati],
    "SP3 Gana error detect+fix":        [has_poem, has_chandassu, has_gana_breakdown],
    "SP4 Vocabulary suggestion":        [has_poem, has_chandassu, has_wtw_rich],
    "SP5 Descriptive feedback":         [has_poem, has_chandassu, has_wtw_any, has_telugu_meaning],
    "SP6 Vocabulary alternatives":      [has_poem, has_chandassu, has_wtw_rich],
}

# What each profile actually GENERATES (so we know data quality matters)
PROFILE_NOTES = {
    "GN5 Theme + Prasa → poem":
        "Needs prasa_check field with a Telugu aksara in quotes e.g. '2nd Letter \\'వ\\' matches'",
    "GN6 Theme + Yati → poem":
        "Needs yati_check field with TWO Telugu aksaras in quotes",
    "GN7 Riddle poem (RPFW)":
        "Telugu meaning used as the 'answer' hint — needs to be descriptive (>=20 chars)",
    "SP4 Vocabulary suggestion":
        "Requires masking a word from WTW — overhead: need to pick which word to mask",
    "SP6 Vocabulary alternatives":
        "Requires generating 3 metrically valid alternatives — needs external validation",
}


# ──────────────────────────────────────────────
# AUDIT
# ──────────────────────────────────────────────

def run_audit(records: list[dict]):
    total = len(records)
    print(f"\n{'═'*62}")
    print(f"  CORPUS: {total:,} records")
    print(f"{'═'*62}")

    # ── Field coverage ─────────────────────────
    field_checks = {
        "poem (non-empty)":           has_poem,
        "two lines":                  has_two_lines,
        "english_meaning":            has_english_meaning,
        "telugu_meaning":             has_telugu_meaning,
        "word_to_word_meaning (≥1)":  has_wtw_any,
        "word_to_word_meaning (≥3)":  has_wtw_rich,
        "chandassu_analysis":         has_chandassu,
        "chandassu: prasa letter":    has_prasa,
        "chandassu: yati letters":    has_yati,
        "chandassu: gana breakdown":  has_gana_breakdown,
        "tags_telugu (≥2 tokens)":    has_tags_telugu,
        "tags_english (≥2 tokens)":   has_tags_english,
        "source field":               has_source,
        "line1 len ≥ 8 chars":        line1_len_ok,
        "telugu_meaning ≥ 20 chars":  tel_meaning_descriptive,
    }

    print(f"\n── FIELD COVERAGE {'─'*43}")
    print(f"  {'FIELD':<38} {'COUNT':>6}  {'%':>6}  VERDICT")
    print(f"  {'─'*58}")
    for fname, fn in field_checks.items():
        count = sum(1 for r in records if fn(r))
        pct   = count / total * 100
        verdict = "✓ GOOD" if pct >= 80 else ("⚠ PARTIAL" if pct >= 40 else "✗ SPARSE")
        print(f"  {fname:<38} {count:>6}  {pct:>5.1f}%  {verdict}")

    # ── Profile feasibility ────────────────────
    print(f"\n── PROFILE FEASIBILITY {'─'*38}")
    print(f"  {'PROFILE':<35} {'USABLE':>6}  {'%':>6}  VERDICT")
    print(f"  {'─'*58}")

    results = {}
    for pname, checks in PROFILES.items():
        usable = sum(1 for r in records if all(fn(r) for fn in checks))
        pct    = usable / total * 100
        if pct >= 80:
            verdict = "✓ KEEP"
        elif pct >= 50:
            verdict = "⚠ KEEP (partial)"
        elif pct >= 20:
            verdict = "△ RISKY"
        else:
            verdict = "✗ DROP"
        results[pname] = (usable, pct, verdict)
        print(f"  {pname:<35} {usable:>6}  {pct:>5.1f}%  {verdict}")
        if pname in PROFILE_NOTES:
            print(f"    ↳ Note: {PROFILE_NOTES[pname]}")

    # ── Summary recommendations ────────────────
    print(f"\n── RECOMMENDATIONS {'─'*42}")
    keep   = [p for p, (_, pct, v) in results.items() if pct >= 80]
    partial= [p for p, (_, pct, v) in results.items() if 50 <= pct < 80]
    risky  = [p for p, (_, pct, v) in results.items() if 20 <= pct < 50]
    drop   = [p for p, (_, pct, v) in results.items() if pct < 20]

    print(f"\n  ✓ KEEP ({len(keep)}):")
    for p in keep:   print(f"    {p}")
    print(f"\n  ⚠ KEEP with caveat ({len(partial)}):")
    for p in partial: print(f"    {p}")
    print(f"\n  △ RISKY — investigate ({len(risky)}):")
    for p in risky:  print(f"    {p}")
    print(f"\n  ✗ DROP ({len(drop)}):")
    for p in drop:   print(f"    {p}")

    # ── Support overhead estimate ──────────────
    support_profiles = [p for p in PROFILES if p.startswith("SP")]
    print(f"\n── SUPPORT OVERHEAD ASSESSMENT {'─'*29}")
    print(f"  Support has {len(support_profiles)} profiles.")
    print(f"  Key concerns:")
    print(f"  1. Error profiles (SP1-SP3) require SYNTHETIC corruption of the input.")
    print(f"     The corruption logic is deterministic (rule-based) but the corrected")
    print(f"     output must still be the original poem — this works with existing data.")
    print(f"  2. SP4 (vocab suggestion) requires MASKING a word and asking for a")
    print(f"     replacement. You'd need to pick which word to mask — adds pipeline step.")
    print(f"  3. SP5 (descriptive feedback) output must be GENERATED, not extracted.")
    print(f"     There is no 'feedback' field in your data — needs synthetic generation.")
    print(f"  4. SP6 (vocab alternatives) requires generating 3 alternatives metrically —")
    print(f"     no ground truth exists. Fully synthetic output needed.")
    print(f"\n  VERDICT: SP1-SP3 are feasible with existing data (corrupt+correct pattern).")
    print(f"           SP4-SP6 require synthetic output generation — significant overhead.")
    print(f"           Recommend: include SP1-SP3 only if Support model is pursued.")
    print(f"{'═'*62}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data feasibility audit for METRICALARGS profiles")
    parser.add_argument("--input", required=True,
                        help="Path to dataset (JSON array or JSONL)")
    args = parser.parse_args()

    print(f"Loading {args.input}...")
    records = load_data(args.input)
    run_audit(records)
