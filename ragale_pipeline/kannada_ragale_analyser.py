# -*- coding: utf-8 -*-
"""
Kannada Ragale Analyser
=======================
Analyzes Utsaha Ragale poems (2-line Kannada couplets) against metrical rules.

Usage:
    python kannada_ragale_analyser.py poems.json

Rules checked:
    1. 12 matras per line (4 ganas × 3 matras)
    2. Each gana must be III or UI (IU is forbidden)
    3. Both lines must end on Guru
    4. Ādi Prāsa: 2nd syllable consonant must match between lines

Syllable division logic adapted from Telugu Dwipada Analyzer.
"""

import json
import sys
from typing import List, Dict, Optional, Tuple

###############################################################################
# KANNADA CHARACTER CONSTANTS
###############################################################################

kannada_consonants = {
    "ಕ", "ಖ", "ಗ", "ಘ", "ಙ",
    "ಚ", "ಛ", "ಜ", "ಝ", "ಞ",
    "ಟ", "ಠ", "ಡ", "ಢ", "ಣ",
    "ತ", "ಥ", "ದ", "ಧ", "ನ",
    "ಪ", "ಫ", "ಬ", "ಭ", "ಮ",
    "ಯ", "ರ", "ಲ", "ವ", "ಶ",
    "ಷ", "ಸ", "ಹ", "ಳ",
}

independent_vowels = {
    "ಅ", "ಆ", "ಇ", "ಈ", "ಉ", "ಊ", "ಋ",
    "ಎ", "ಏ", "ಐ", "ಒ", "ಓ", "ಔ",
}

independent_long_vowels = {"ಆ", "ಈ", "ಊ", "ಏ", "ಓ"}

dependent_vowels = {"ಾ", "ಿ", "ೀ", "ು", "ೂ", "ೃ", "ೆ", "ೇ", "ೈ", "ೊ", "ೋ", "ೌ"}

long_vowel_matras = {"ಾ", "ೀ", "ೂ", "ೇ", "ೋ", "ೌ"}

halant = "್"

diacritics = {"ಂ", "ಃ"}

ignorable_chars = {' ', '\n', '\u200C', '\u200B', ',', '-', '.', '!', '?', ';', ':', '|', '।', '॥'}


###############################################################################
# SYLLABLE DIVISION (adapted from Telugu analyzer split_aksharalu)
###############################################################################

def split_aksharalu(word: str) -> List[str]:
    """
    Split Kannada word into aksharalu (syllables).

    Two-pass algorithm:
    Pass 1: Coarse split at consonant/vowel boundaries, keeping conjuncts together.
    Pass 2: Merge trailing pollu hallu (consonant+halant only) with previous syllable.
    """
    # Pass 1: Coarse split
    coarse_split = []
    i, n = 0, len(word)

    while i < n:
        if word[i] in ignorable_chars:
            coarse_split.append(word[i])
            i += 1
            continue

        current = []
        if word[i] in kannada_consonants:
            current.append(word[i])
            i += 1
            # Collect conjunct consonants: C್C್C...
            while i < n and word[i] == halant:
                current.append(word[i])  # halant
                i += 1
                if i < n and word[i] in kannada_consonants:
                    current.append(word[i])  # next consonant
                    i += 1
                else:
                    break
            # Attach dependent vowels and diacritics
            while i < n and (word[i] in dependent_vowels or word[i] in diacritics):
                current.append(word[i])
                i += 1
        elif word[i] in independent_vowels:
            current.append(word[i])
            i += 1
            # Independent vowel can have diacritics (ಅಂ ಅಃ)
            if i < n and word[i] in diacritics:
                current.append(word[i])
                i += 1
        else:
            # Unknown character — skip
            i += 1
            continue

        if current:
            coarse_split.append("".join(current))

    if not coarse_split:
        return []

    # Pass 2: Merge pollu hallu (consonant + halant only) with previous syllable
    final = []
    for chunk in coarse_split:
        is_pollu = (len(chunk) == 2 and
                    chunk[0] in kannada_consonants and
                    chunk[1] == halant)
        if is_pollu and final and final[-1] not in ignorable_chars:
            final[-1] += chunk
        else:
            final.append(chunk)

    return [ak for ak in final if ak and ak not in ignorable_chars]


###############################################################################
# GURU / LAGHU CLASSIFICATION (adapted from Telugu analyzer)
###############################################################################

def _categorize(aksharam: str) -> set:
    """Return linguistic tags for a single syllable."""
    tags = set()

    if aksharam[0] in independent_vowels:
        tags.add("vowel")
    if any(c in kannada_consonants for c in aksharam):
        tags.add("consonant")
    if any(m in aksharam for m in long_vowel_matras) or aksharam[0] in independent_long_vowels:
        tags.add("long")
    if "ಃ" in aksharam:
        tags.add("visarga")
    if "ಂ" in aksharam:
        tags.add("anusvara")

    # Check conjunct (C್C different) or double (C್C same)
    for idx in range(len(aksharam) - 2):
        if (aksharam[idx] in kannada_consonants and
                aksharam[idx + 1] == halant and
                idx + 2 < len(aksharam) and
                aksharam[idx + 2] in kannada_consonants):
            if aksharam[idx] == aksharam[idx + 2]:
                tags.add("double")
            else:
                tags.add("conjunct")

    return tags


def classify_guru_laghu(aksharalu: List[str]) -> List[str]:
    """
    Mark each syllable as Guru (U) or Laghu (I).

    Pass 1: Own properties — long vowel, diphthong, anusvara, visarga, halant ending.
    Pass 2: Sandhi — if next syllable (same word) starts with conjunct/double → current becomes Guru.
    """
    if not aksharalu:
        return []

    markers = ["I"] * len(aksharalu)

    # Pass 1: intrinsic properties
    for i, ak in enumerate(aksharalu):
        tags = _categorize(ak)

        is_guru = False
        if "long" in tags:
            is_guru = True
        if "ಐ" in ak or "ಔ" in ak or "ೈ" in ak or "ೌ" in ak:
            is_guru = True
        if "anusvara" in tags or "visarga" in tags:
            is_guru = True
        if ak.endswith(halant):
            is_guru = True

        if is_guru:
            markers[i] = "U"

    # Pass 2: sandhi rule — syllable before conjunct/double becomes Guru
    for i in range(len(aksharalu) - 1):
        next_tags = _categorize(aksharalu[i + 1])
        if "conjunct" in next_tags or "double" in next_tags:
            markers[i] = "U"

    return markers


###############################################################################
# GANA PARTITION (best-fit search over all III/UI combos)
###############################################################################

VALID_GANA_PATTERNS = {"III", "IIU", "UI"}  # IU is forbidden

def find_gana_partition(markers: List[str]) -> Dict:
    """
    Find best partition of syllables into exactly 4 ganas.

    Each gana is one of:
        III  (3 syllables, 3 matras) — Primary, high frequency
        IIU  (3 syllables, 4 matras) — Standard, commonly used
        UI   (2 syllables, 3 matras) — Standard variation

    IU (laghu-guru) is forbidden — breaks rhythmic flow.

    Tries all 2^4 = 16 size combinations (each gana is 2 or 3 syllables).
    Picks the partition with the most valid ganas.
    """
    n = len(markers)
    best = None
    best_valid_count = -1

    # Try all 16 combos of gana sizes [2 or 3] × 4
    for combo in range(16):
        sizes = []
        for g in range(4):
            sizes.append(3 if (combo >> g) & 1 else 2)

        if sum(sizes) != n:
            continue

        ganas = []
        pos = 0
        valid_count = 0
        for size in sizes:
            pattern = "".join(markers[pos:pos + size])
            pos += size

            is_valid = pattern in VALID_GANA_PATTERNS
            if is_valid:
                valid_count += 1

            ganas.append({
                "pattern": pattern,
                "size": size,
                "valid": is_valid,
                "forbidden": pattern == "IU",
            })

        if valid_count > best_valid_count:
            best_valid_count = valid_count
            best = {
                "ganas": ganas,
                "valid_count": valid_count,
                "total_ganas": 4,
                "fully_valid": valid_count == 4,
            }

    # No partition found — syllable count doesn't fit any combo (not 8-12)
    if best is None:
        ganas = []
        pos = 0
        for idx in range(4):
            if pos + 3 <= n:
                pattern = "".join(markers[pos:pos + 3])
                pos += 3
            elif pos < n:
                pattern = "".join(markers[pos:])
                pos = n
            else:
                pattern = ""
            ganas.append({
                "pattern": pattern,
                "size": len(pattern),
                "valid": pattern in VALID_GANA_PATTERNS,
                "forbidden": pattern == "IU",
            })

        valid_count = sum(1 for g in ganas if g["valid"])
        best = {
            "ganas": ganas,
            "valid_count": valid_count,
            "total_ganas": len(ganas),
            "fully_valid": False,
        }

    return best


###############################################################################
# BASE CONSONANT EXTRACTION
###############################################################################

def get_base_consonant(syllable: str) -> Optional[str]:
    """Extract the first/base consonant from a syllable."""
    for ch in syllable:
        if ch in kannada_consonants:
            return ch
    return None


###############################################################################
# LINE ANALYSIS
###############################################################################

def analyze_line(line: str) -> Dict:
    """Analyze a single line of Ragale poetry."""
    words = line.strip().split()
    all_syllables = []
    for word in words:
        all_syllables.extend(split_aksharalu(word))

    markers = classify_guru_laghu(all_syllables)

    # Compute matras
    matra_count = sum(1 if m == "I" else 2 for m in markers)

    # Find best gana partition
    partition = find_gana_partition(markers)

    # Check guru ending
    ends_guru = markers[-1] == "U" if markers else False

    # Check for any forbidden IU ganas
    has_forbidden = any(g.get("forbidden", False) for g in partition["ganas"])

    # Strictly 12 syllables per line
    syllable_valid = len(all_syllables) == 12

    return {
        "text": line.strip(),
        "syllables": all_syllables,
        "markers": markers,
        "marker_string": "".join(markers),
        "syllable_count": len(all_syllables),
        "matra_count": matra_count,
        "syllable_valid": syllable_valid,
        "partition": partition,
        "ends_guru": ends_guru,
        "has_forbidden_iu": has_forbidden,
    }


###############################################################################
# ĀDI PRĀSA CHECK
###############################################################################

def check_adi_prasa(line1_syllables: List[str], line2_syllables: List[str]) -> Dict:
    """Check if the 2nd syllable's base consonant matches between lines."""
    if len(line1_syllables) < 2 or len(line2_syllables) < 2:
        return {
            "match": False,
            "error": "Not enough syllables in one or both lines",
            "line1_2nd": None,
            "line2_2nd": None,
            "line1_consonant": None,
            "line2_consonant": None,
        }

    syl1 = line1_syllables[1]
    syl2 = line2_syllables[1]
    con1 = get_base_consonant(syl1)
    con2 = get_base_consonant(syl2)

    return {
        "match": con1 is not None and con2 is not None and con1 == con2,
        "line1_2nd": syl1,
        "line2_2nd": syl2,
        "line1_consonant": con1,
        "line2_consonant": con2,
    }


###############################################################################
# SCORING
###############################################################################

def calculate_score(line1: Dict, line2: Dict, prasa: Dict) -> Dict:
    """
    Calculate overall accuracy score.
    Weights: Matra count 20%, Gana validity 30%, Guru ending 15%, Ādi Prāsa 35%
    """
    # Syllable count score (per line, averaged) — must be exactly 12
    def syllable_score(line):
        diff = abs(line["syllable_count"] - 12)
        return max(0, 100 - diff * 20)

    syl_avg = (syllable_score(line1) + syllable_score(line2)) / 2

    # Gana validity score (per line, averaged)
    def gana_score(line):
        p = line["partition"]
        if p["total_ganas"] == 0:
            return 0
        return (p["valid_count"] / 4) * 100

    gana_avg = (gana_score(line1) + gana_score(line2)) / 2

    # Guru ending score
    guru_count = int(line1["ends_guru"]) + int(line2["ends_guru"])
    guru_score = guru_count * 50  # 0, 50, or 100

    # Ādi Prāsa score
    prasa_score = 100 if prasa["match"] else 0

    # Weighted overall
    overall = (
        syl_avg * 0.20 +
        gana_avg * 0.30 +
        guru_score * 0.15 +
        prasa_score * 0.35
    )

    return {
        "overall": round(overall, 1),
        "breakdown": {
            "syllable_line1": syllable_score(line1),
            "syllable_line2": syllable_score(line2),
            "syllable_average": round(syl_avg, 1),
            "gana_line1": gana_score(line1),
            "gana_line2": gana_score(line2),
            "gana_average": round(gana_avg, 1),
            "guru_ending": guru_score,
            "adi_prasa": prasa_score,
        },
        "weights": {
            "syllable_count": 0.20,
            "gana": 0.30,
            "guru_ending": 0.15,
            "adi_prasa": 0.35,
        },
    }


###############################################################################
# POEM ANALYSIS
###############################################################################

def analyze_poem(poem: Dict) -> Dict:
    """Analyze a single Ragale poem JSON object."""
    poem_text = poem.get("poem_kannada", "")
    lines = [l.strip() for l in poem_text.split("\n") if l.strip()]

    if len(lines) < 2:
        return {
            "error": f"Expected 2 lines, got {len(lines)}",
            "poem": poem_text,
            "theme": poem.get("theme", ""),
        }

    line1_analysis = analyze_line(lines[0])
    line2_analysis = analyze_line(lines[1])

    prasa = check_adi_prasa(line1_analysis["syllables"], line2_analysis["syllables"])

    score = calculate_score(line1_analysis, line2_analysis, prasa)

    return {
        "theme": poem.get("theme", ""),
        "line1": line1_analysis,
        "line2": line2_analysis,
        "adi_prasa": prasa,
        "score": score,
    }


###############################################################################
# REPORT FORMATTING
###############################################################################

def format_report(analysis: Dict) -> str:
    """Format a human-readable analysis report."""
    if "error" in analysis:
        return f"ERROR: {analysis['error']}\n  Poem: {analysis.get('poem', '')}\n"

    lines = []
    lines.append(f"{'='*60}")
    lines.append(f"Theme: {analysis['theme']}")
    lines.append(f"Overall Score: {analysis['score']['overall']}%")
    lines.append(f"{'='*60}")

    for label, la in [("Line 1", analysis["line1"]), ("Line 2", analysis["line2"])]:
        lines.append(f"\n  {label}: {la['text']}")
        lines.append(f"  Syllables ({la['syllable_count']}): {' | '.join(la['syllables'])}")
        lines.append(f"  Markers: {' '.join(la['markers'])}")
        lines.append(f"  Syllables: {la['syllable_count']}/12 {'✓' if la['syllable_valid'] else '✗'}  (Matras: {la['matra_count']})")

        gana_strs = []
        for i, g in enumerate(la["partition"]["ganas"], 1):
            status = "✓" if g["valid"] else ("⛔ IU!" if g["forbidden"] else "✗")
            gana_strs.append(f"G{i}:{g['pattern']}({status})")
        lines.append(f"  Ganas: {' '.join(gana_strs)} — {la['partition']['valid_count']}/4 valid")
        lines.append(f"  Ends Guru: {'✓' if la['ends_guru'] else '✗'}")
        if la["has_forbidden_iu"]:
            lines.append(f"  ⚠ WARNING: Forbidden IU gana detected!")

    p = analysis["adi_prasa"]
    lines.append(f"\n  Ādi Prāsa: {p['line1_consonant']} vs {p['line2_consonant']} → {'✓ Match' if p['match'] else '✗ Mismatch'}")
    lines.append(f"    (Line1 2nd: {p['line1_2nd']}, Line2 2nd: {p['line2_2nd']})")

    bd = analysis["score"]["breakdown"]
    lines.append(f"\n  Score Breakdown:")
    lines.append(f"    Syllable count: {bd['syllable_average']}% (L1:{bd['syllable_line1']}% L2:{bd['syllable_line2']}%) × 20%")
    lines.append(f"    Gana pattern: {bd['gana_average']}% (L1:{bd['gana_line1']}% L2:{bd['gana_line2']}%) × 30%")
    lines.append(f"    Guru ending:  {bd['guru_ending']}% × 15%")
    lines.append(f"    Ādi Prāsa:    {bd['adi_prasa']}% × 35%")
    lines.append("")

    return "\n".join(lines)


###############################################################################
# MAIN
###############################################################################

def main():
    if len(sys.argv) < 2:
        print("Usage: python kannada_ragale_analyser.py <poems.json>")
        print("  JSON can be a single poem object or an array of poems.")
        sys.exit(1)

    filepath = sys.argv[1]

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        poems = [data]
    elif isinstance(data, list):
        poems = data
    else:
        print("Error: JSON must be an object or array of objects.")
        sys.exit(1)

    all_results = []
    total_score = 0

    for i, poem in enumerate(poems, 1):
        analysis = analyze_poem(poem)
        all_results.append(analysis)
        print(format_report(analysis))
        if "score" in analysis:
            total_score += analysis["score"]["overall"]

    if len(poems) > 1:
        avg = total_score / len(poems)
        print(f"\n{'='*60}")
        print(f"SUMMARY: {len(poems)} poems analyzed, Average Score: {avg:.1f}%")
        print(f"{'='*60}")

    # Write results JSON
    output_path = filepath.rsplit(".", 1)[0] + "_analysis.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\nDetailed results written to: {output_path}")


if __name__ == "__main__":
    main()
