# -*- coding: utf-8 -*-
"""
Dwipada Analyzer v2.0
=====================
Telugu Dwipada Chandassu (prosody) analysis tool with percentage scoring.

QUICK REFERENCE
---------------
Main Functions:
    analyze_dwipada(poem)       → Full analysis with match_score (0-100%)
    format_analysis_report()    → Human-readable report with diagnostics
    analyze_single_line(line)   → Analyze one line of poetry
    analyze_pada(line)          → Low-level line analysis (dict output)

Key Concepts:
    - Dwipada (ద్విపద): 2-line couplet, each line has 3 Indra + 1 Surya gana
    - Gana (గణము): Prosodic foot with Guru (U) / Laghu (I) pattern
    - Prasa (ప్రాస): 2nd syllable consonant must match between lines
    - Yati (యతి): 1st letter of gana 1 must match 1st letter of gana 3

Scoring (0-100%):
    - Gana: 40% weight (25% per valid gana × 4 ganas)
    - Prasa: 35% weight (100% if match, 0% if mismatch)
    - Yati: 25% weight (100% exact, 70% same varga, 0% different)

Example:
    >>> poem = \"\"\"సౌధాగ్రముల యందు సదనంబు లందు
    ... వీధుల యందును వెఱవొప్ప నిలిచి\"\"\"
    >>> analysis = analyze_dwipada(poem)
    >>> analysis["is_valid_dwipada"]
    True
    >>> analysis["match_score"]["overall"]
    100.0

Based on Aksharanusarika v0.0.7a logic.
"""

import re
from typing import List, Tuple, Dict, Optional, Set

###############################################################################
# TELUGU PROSODY GLOSSARY (ఛందస్సు పదకోశం)
###############################################################################
#
# BASIC TERMS:
# - Aksharam (అక్షరము): Syllable - the fundamental unit of Telugu writing
# - Gana (గణము): Prosodic foot - group of syllables with specific pattern
# - Pada (పాదము): Line/verse - one line of poetry
# - Chandassu (ఛందస్సు): Meter/prosody - the rhythmic system
#
# SYLLABLE WEIGHT:
# - Guru (గురువు, U): Heavy/long syllable - takes more time to pronounce
# - Laghu (లఘువు, I): Light/short syllable - quick to pronounce
#
# GANA TYPES IN DWIPADA:
# - Indra Ganas (ఇంద్ర గణములు): 3-4 syllable patterns
#     * Nala (నల): IIII - 4 light syllables
#     * Naga (నగ): IIIU - 3 light + 1 heavy
#     * Sala (సల): IIUI - 2 light + 1 heavy + 1 light
#     * Bha (భ): UII - 1 heavy + 2 light
#     * Ra (ర): UIU - heavy + light + heavy
#     * Ta (త): UUI - 2 heavy + 1 light
# - Surya Ganas (సూర్య గణములు): 2-3 syllable patterns
#     * Na (న): III - 3 light syllables
#     * Ha/Gala (హ/గల): UI - 1 heavy + 1 light
#
# DWIPADA RULES:
# - Prasa (ప్రాస): Rhyme rule - 2nd syllable's consonant must match
#   between lines (e.g., "ధా" and "ధు" both have consonant "ధ")
# - Yati (యతి): Alliteration rule - 1st letter of gana 1 must match
#   1st letter of gana 3 in each line
# - Yati Maitri (యతి మైత్రి): Phonetic groups where different letters
#   can substitute for each other (e.g., క and గ are in same group)
#
# SPECIAL TERMS:
# - Pollu Hallu (పొల్లు హల్లు): Consonant + halant that can't stand alone
#   Example: In "సత్య", the "త్" is pollu hallu, merges with "య" → "త్య"
# - Varga (వర్గము): Consonant class by articulation point (velar, dental, etc.)
# - Halant (హలంతు, ్): Vowel-killer mark that removes inherent 'అ' sound
# - Anusvara (అనుస్వారం, ం): Nasal sound marker
# - Visarga (విసర్గ, ః): Breath sound marker
#
###############################################################################

###############################################################################
# LINGUISTIC DATA AND CONSTANTS
###############################################################################

dependent_to_independent = {
    "ా": "ఆ", "ి": "ఇ", "ీ": "ఈ", "ు": "ఉ", "ూ": "ఊ", "ృ": "ఋ",
    "ౄ": "ౠ", "ె": "ఎ", "ే": "ఏ", "ై": "ఐ", "ొ": "ఒ", "ో": "ఓ", "ౌ": "ఔ"
}
halant = "్"
telugu_consonants = {
    "క", "ఖ", "గ", "ఘ", "ఙ", "చ", "ఛ", "జ", "ఝ", "ఞ",
    "ట", "ఠ", "డ", "ఢ", "ణ", "త", "థ", "ద", "ధ", "న",
    "ప", "ఫ", "బ", "భ", "మ", "య", "ర", "ల", "వ", "శ",
    "ష", "స", "హ", "ళ", "ఱ"
}
long_vowels = {"ా", "ీ", "ూ", "ే", "ో", "ౌ", "ౄ"}
independent_vowels = {
    "అ", "ఆ", "ఇ", "ఈ", "ఉ", "ఊ", "ఋ", "ౠ",
    "ఎ", "ఏ", "ఐ", "ఒ", "ఓ", "ఔ"
}
independent_long_vowels = {"ఆ", "ఈ", "ఊ", "ౠ", "ఏ", "ఓ"}
diacritics = {"ం", "ః"}
dependent_vowels = set(dependent_to_independent.keys())
ignorable_chars = {' ', '\n', 'ఁ', '​'}  # space, newline, arasunna, zero-width space

# Yati Maitri Groups (Vargas)
# These groups define which letters can substitute for each other in Yati (యతి) matching
# Letters in the same group are phonetically related and can satisfy Yati requirements
YATI_MAITRI_GROUPS = [
    {"అ", "ఆ", "ఐ", "ఔ", "హ", "య", "అం", "అః"},
    {"ఇ", "ఈ", "ఎ", "ఏ", "ఋ"},
    {"ఉ", "ఊ", "ఒ", "ఓ"},
    {"క", "ఖ", "గ", "ఘ", "క్ష"},
    {"చ", "ఛ", "జ", "ఝ", "శ", "ష", "స"},
    {"ట", "ఠ", "డ", "ఢ"},
    {"త", "థ", "ద", "ధ"},
    {"ప", "ఫ", "బ", "భ", "వ"},
    {"ర", "ల", "ఱ", "ళ"},
    {"న", "ణ"},
    {"మ", "పు", "ఫు", "బు", "భు", "ము"},
]

# Svara Yati Groups (స్వర యతి) — Vowel family harmony
# Vowels in the same group can satisfy Yati regardless of consonants.
# Uses independent vowel forms; dependent vowels are mapped via dependent_to_independent.
SVARA_YATI_GROUPS = [
    {"అ", "ఆ", "ఐ", "ఔ"},
    {"ఇ", "ఈ", "ఎ", "ఏ", "ఋ", "ౠ"},
    {"ఉ", "ఊ", "ఒ", "ఓ"},
]

# Bindu Yati (బిందు యతి) — Varga-to-Nasal mapping
# When a syllable has anusvara (ం), it can match the nasal of its consonant's varga.
VARGA_NASALS = {
    "క": "ఙ", "ఖ": "ఙ", "గ": "ఙ", "ఘ": "ఙ",
    "చ": "ఞ", "ఛ": "ఞ", "జ": "ఞ", "ఝ": "ఞ",
    "ట": "ణ", "ఠ": "ణ", "డ": "ణ", "ఢ": "ణ",
    "త": "న", "థ": "న", "ద": "న", "ధ": "న",
    "ప": "మ", "ఫ": "మ", "బ": "మ", "భ": "మ",
}
NASAL_TO_VARGA = {
    "ఙ": {"క", "ఖ", "గ", "ఘ"},
    "ఞ": {"చ", "ఛ", "జ", "ఝ"},
    "ణ": {"ట", "ఠ", "డ", "ఢ"},
    "న": {"త", "థ", "ద", "ధ"},
    "మ": {"ప", "ఫ", "బ", "భ"},
}

# Prasa Equivalency Groups (ప్రాస సమానాక్షరములు)
# These consonant pairs are traditionally treated as equivalent for Prasa matching.
# The equivalency applies regardless of gudinthams (vowel marks) or vattulu (conjuncts)
# because get_base_consonant() extracts only the first consonant before comparison.
PRASA_EQUIVALENTS = [
    {"ల", "ళ"},
    {"శ", "స"},
    {"ఱ", "ర"},
]

# =============================================================================
# CONSONANT CLASSIFICATION (వర్ణమాల విభజన)
# =============================================================================
# Telugu consonants are grouped by place of articulation (ఉచ్చారణ స్థానం)
# Each varga shares similar mouth position when pronounced
#
# This classification is used for:
# 1. Prasa mismatch diagnostics - explaining why consonants don't match
# 2. Yati analysis - providing varga information for letter matching
# 3. Educational purposes - showing phonetic relationships

CONSONANT_VARGAS = {
    # Velar (కంఠ్యము) - produced at the soft palate (back of mouth)
    "క-వర్గము (Velar)": ["క", "ఖ", "గ", "ఘ", "ఙ"],

    # Palatal (తాలవ్యము) - produced at the hard palate
    # Includes sibilants (శ, ష, స) which share palatal articulation
    "చ-వర్గము (Palatal)": ["చ", "ఛ", "జ", "ఝ", "ఞ", "శ", "ష", "స"],

    # Retroflex (మూర్ధన్యము) - tongue curled back touching roof of mouth
    "ట-వర్గము (Retroflex)": ["ట", "ఠ", "డ", "ఢ", "ణ"],

    # Dental (దంత్యము) - tongue touches upper teeth
    "త-వర్గము (Dental)": ["త", "థ", "ద", "ధ", "న"],

    # Labial (ఓష్ఠ్యము) - produced with lips
    "ప-వర్గము (Labial)": ["ప", "ఫ", "బ", "భ", "మ"],

    # Semi-vowels and approximants (అంతస్థములు)
    "య-వర్గము (Approximant)": ["య", "ర", "ల", "వ", "ళ", "ఱ"],

    # Aspirate (ఊష్మము)
    "హ-వర్గము (Aspirate)": ["హ"],
}

# =============================================================================
# SCORING CONSTANTS
# =============================================================================
# These constants define the scoring rules for Dwipada analysis.
# Using named constants improves readability and makes it easy to adjust scoring.

EXPECTED_GANAS_PER_LINE = 4           # 3 Indra + 1 Surya gana per line
SCORE_PER_VALID_GANA = 25.0           # Each valid gana contributes 25% (100% / 4)

# Yati quality scores - measures how well the alliteration rule is satisfied
YATI_EXACT_MATCH_SCORE = 100.0        # Same letter (e.g., స ↔ స)
YATI_VARGA_MATCH_SCORE = 100.0        # Same phonetic group (e.g., క ↔ గ) — treated as equivalent
YATI_NO_MATCH_SCORE = 0.0             # Different groups (e.g., క ↔ చ)

# Prasa scores - binary match for rhyme rule
PRASA_MATCH_SCORE = 100.0             # Consonants match
PRASA_NO_MATCH_SCORE = 0.0            # Consonants don't match


def get_consonant_varga(consonant: str) -> Optional[str]:
    """
    Get the varga (consonant class) for a Telugu consonant.

    Telugu consonants are classified into vargas based on their place of
    articulation (ఉచ్చారణ స్థానం). This function returns which varga
    a given consonant belongs to.

    Args:
        consonant: A single Telugu consonant character (హల్లు)

    Returns:
        Varga name (e.g., "క-వర్గము (Velar)") or None if not a consonant

    Example:
        >>> get_consonant_varga("క")
        "క-వర్గము (Velar)"
        >>> get_consonant_varga("ధ")
        "త-వర్గము (Dental)"
        >>> get_consonant_varga("అ")
        None  # అ is a vowel, not a consonant
    """
    if not consonant:
        return None

    for varga_name, consonants in CONSONANT_VARGAS.items():
        if consonant in consonants:
            return varga_name

    return None


def get_letter_info(letter: str) -> Dict:
    """
    Get complete classification information for a Telugu letter.

    This function provides comprehensive details about any Telugu letter,
    including whether it's a vowel or consonant, its varga classification,
    and which Yati Maitri groups it belongs to.

    Args:
        letter: A single Telugu letter (vowel or consonant)

    Returns:
        Dictionary with the following keys:
        - letter: The input letter
        - type: "vowel" (అచ్చు), "consonant" (హల్లు), or "unknown"
        - varga: Consonant varga name (only for consonants)
        - yati_groups: List of Yati Maitri group indices this letter belongs to
        - yati_group_members: List of all members in the letter's Yati groups

    Example:
        >>> get_letter_info("క")
        {
            "letter": "క",
            "type": "consonant",
            "varga": "క-వర్గము (Velar)",
            "yati_groups": [3],
            "yati_group_members": ["క", "ఖ", "గ", "ఘ", "క్ష"]
        }
    """
    result = {
        "letter": letter,
        "type": "unknown",
        "varga": None,
        "yati_groups": [],
        "yati_group_members": [],
    }

    if not letter:
        return result

    # Determine if vowel or consonant
    if letter in independent_vowels or letter in dependent_vowels:
        result["type"] = "vowel"
    elif letter in telugu_consonants:
        result["type"] = "consonant"
        result["varga"] = get_consonant_varga(letter)

    # Find Yati Maitri groups this letter belongs to
    for idx, group in enumerate(YATI_MAITRI_GROUPS):
        if letter in group:
            result["yati_groups"].append(idx)
            result["yati_group_members"].extend(list(group))

    # Remove duplicates from group members while preserving order
    seen = set()
    unique_members = []
    for member in result["yati_group_members"]:
        if member not in seen:
            seen.add(member)
            unique_members.append(member)
    result["yati_group_members"] = unique_members

    return result


# =============================================================================
# SCORING HELPER FUNCTIONS
# =============================================================================
# These functions calculate percentage scores for different aspects of
# Dwipada poetry analysis. Scores range from 0-100%.
#
# Scoring Philosophy:
# - Gana matching: 25% per valid gana (4 ganas per line = 100%)
# - Prasa: Binary - 100% if match, 0% if mismatch
# - Yati: 100% for exact letter match, 70% for same varga, 0% for different

# Weights for overall score calculation
SCORE_WEIGHTS = {
    "gana": 0.40,   # 40% weight - gana sequence is fundamental structure
    "prasa": 0.35,  # 35% weight - prasa (rhyme) is essential for dwipada
    "yati": 0.25,   # 25% weight - yati adds phonetic beauty
}


def calculate_gana_score(partition_result: Optional[Dict]) -> Dict:
    """
    Calculate the percentage score for gana matching.

    A valid Dwipada line has 4 ganas: 3 Indra ganas + 1 Surya gana.
    Each valid gana contributes 25% to the score (4 × 25% = 100%).

    Args:
        partition_result: The result from find_dwipada_gana_partition()
                         Contains "ganas" list with each gana's validity

    Returns:
        Dictionary with:
        - score: Float 0-100 representing percentage match
        - ganas_matched: Number of valid ganas found (0-4)
        - ganas_total: Expected number of ganas (4)
        - details: List of per-gana validity info

    Example:
        >>> partition = find_dwipada_gana_partition(gana_markers, aksharalu)
        >>> calculate_gana_score(partition)
        {"score": 100.0, "ganas_matched": 4, "ganas_total": 4, "details": [...]}
    """
    result = {
        "score": 0.0,
        "ganas_matched": 0,
        "ganas_total": 4,
        "details": [],
    }

    if not partition_result or "ganas" not in partition_result:
        return result

    ganas = partition_result["ganas"]
    valid_count = 0

    for i, gana in enumerate(ganas, 1):
        is_valid = gana.get("name") is not None
        if is_valid:
            valid_count += 1

        result["details"].append({
            "position": i,
            "type": gana.get("type", "Unknown"),
            "pattern": gana.get("pattern", ""),
            "name": gana.get("name"),
            "valid": is_valid,
            "aksharalu": gana.get("aksharalu", []),
        })

    result["ganas_matched"] = valid_count
    # Each valid gana contributes 25% to the score (100% / 4 ganas)
    result["score"] = valid_count * SCORE_PER_VALID_GANA

    return result


def calculate_prasa_score(prasa_result: Optional[Dict]) -> Dict:
    """
    Calculate the percentage score for prasa (rhyme) matching.

    Prasa is binary: either the 2nd syllable consonants match (100%) or
    they don't (0%). This function also provides diagnostic information
    about why a mismatch occurred.

    Args:
        prasa_result: Dictionary containing prasa analysis results
                     with "match", "line1_consonant", "line2_consonant" keys

    Returns:
        Dictionary with:
        - score: Float 0 or 100
        - match: Boolean indicating if prasa matches
        - mismatch_details: Diagnostic info if mismatch (None if match)

    Example:
        >>> prasa = {"match": False, "line1_consonant": "ధ", "line2_consonant": "మ"}
        >>> calculate_prasa_score(prasa)
        {"score": 0.0, "match": False, "mismatch_details": {...}}
    """
    result = {
        "score": 0.0,
        "match": False,
        "mismatch_details": None,
    }

    if not prasa_result:
        return result

    is_match = prasa_result.get("match", False)
    result["match"] = is_match
    result["score"] = PRASA_MATCH_SCORE if is_match else PRASA_NO_MATCH_SCORE

    # Generate mismatch diagnostics if not matching
    if not is_match:
        cons1 = prasa_result.get("line1_consonant")
        cons2 = prasa_result.get("line2_consonant")
        varga1 = get_consonant_varga(cons1) if cons1 else None
        varga2 = get_consonant_varga(cons2) if cons2 else None

        result["mismatch_details"] = {
            "line1_full_breakdown": {
                "aksharam": prasa_result.get("line1_second_aksharam"),
                "consonant": cons1,
                "consonant_varga": varga1,
            },
            "line2_full_breakdown": {
                "aksharam": prasa_result.get("line2_second_aksharam"),
                "consonant": cons2,
                "consonant_varga": varga2,
            },
            "why_mismatch": _generate_prasa_mismatch_explanation(cons1, cons2, varga1, varga2),
            "suggestion": _generate_prasa_suggestion(cons1),
        }

    return result


def _generate_prasa_mismatch_explanation(cons1: str, cons2: str,
                                         varga1: Optional[str],
                                         varga2: Optional[str]) -> str:
    """
    Generate a human-readable explanation for why prasa doesn't match.

    This helper function creates educational diagnostic messages explaining
    why two consonants don't satisfy the prasa requirement.

    Args:
        cons1: First consonant
        cons2: Second consonant
        varga1: Varga of first consonant
        varga2: Varga of second consonant

    Returns:
        Explanation string in Telugu/English mixed format
    """
    if not cons1 or not cons2:
        return "One or both lines don't have a valid consonant in 2nd position"

    if varga1 and varga2:
        if varga1 == varga2:
            return f"Consonants '{cons1}' and '{cons2}' are from same varga ({varga1}) but prasa requires exact match"
        else:
            return f"Consonants '{cons1}' ({varga1}) and '{cons2}' ({varga2}) are from different vargas"

    return f"Consonants '{cons1}' and '{cons2}' do not match - prasa requires identical consonants"


def _generate_prasa_suggestion(consonant: str) -> str:
    """
    Generate a suggestion for fixing prasa mismatch.

    Provides examples of syllables that would create valid prasa
    with the given consonant.

    Args:
        consonant: The consonant from line 1's 2nd syllable

    Returns:
        Suggestion string with example valid syllables
    """
    if not consonant:
        return "Unable to generate suggestion - no valid consonant found"

    # Common vowel combinations for examples
    vowels = ["", "ా", "ి", "ీ", "ు", "ూ", "ె", "ే", "ో"]
    examples = [consonant + v for v in vowels[:5]]

    return f"Line 2 needs 2nd syllable with '{consonant}' consonant (e.g., {', '.join(examples)}...)"


def calculate_yati_score(yati_result: Optional[Dict]) -> Dict:
    """
    Calculate the percentage score for yati (alliteration) matching.

    Yati scoring:
    - 100%: Exact letter match OR same Yati Maitri varga (both treated as full match)
    - 0%: Different vargas (no phonetic relationship)

    Args:
        yati_result: Dictionary containing yati analysis with
                    "match", "first_gana_letter", "third_gana_letter" keys

    Returns:
        Dictionary with:
        - score: Float 0, 70, or 100
        - quality: "exact", "varga_match", or "no_match"
        - mismatch_details: Diagnostic info (always provided for transparency)

    Example:
        >>> yati = {"match": True, "first_gana_letter": "స", "third_gana_letter": "స"}
        >>> calculate_yati_score(yati)
        {"score": 100.0, "quality": "exact", "mismatch_details": {...}}
    """
    result = {
        "score": 0.0,
        "quality": "no_match",
        "mismatch_details": None,
    }

    if not yati_result:
        return result

    letter1 = yati_result.get("first_gana_letter")
    letter2 = yati_result.get("third_gana_letter")
    is_match = yati_result.get("match", False)
    group_idx = yati_result.get("group_index")

    # Get detailed letter information
    info1 = get_letter_info(letter1) if letter1 else None
    info2 = get_letter_info(letter2) if letter2 else None

    # Determine quality of match
    if is_match:
        if letter1 == letter2:
            result["score"] = YATI_EXACT_MATCH_SCORE
            result["quality"] = "exact"
        else:
            # Same varga but different letter
            result["score"] = YATI_VARGA_MATCH_SCORE
            result["quality"] = "varga_match"
    else:
        result["score"] = YATI_NO_MATCH_SCORE
        result["quality"] = "no_match"

    # Always provide letter details for educational purposes
    result["mismatch_details"] = {
        "letter1_info": info1,
        "letter2_info": info2,
        "why_result": _generate_yati_explanation(letter1, letter2, is_match, info1, info2),
        "suggestion": _generate_yati_suggestion(letter1, info1) if not is_match else None,
    }

    return result


def _generate_yati_explanation(letter1: str, letter2: str, is_match: bool,
                               info1: Optional[Dict], info2: Optional[Dict]) -> str:
    """
    Generate a human-readable explanation for yati match result.

    Args:
        letter1: First letter (1st gana start)
        letter2: Second letter (3rd gana start)
        is_match: Whether yati is satisfied
        info1: Letter info for first letter
        info2: Letter info for second letter

    Returns:
        Explanation string describing why yati matches or doesn't
    """
    if not letter1 or not letter2:
        return "Unable to determine yati - missing letter information"

    if letter1 == letter2:
        return f"Exact match: both positions have '{letter1}' → MATCH (100%)"

    if is_match:
        # Same varga match
        groups1 = info1.get("yati_group_members", []) if info1 else []
        return f"'{letter1}' and '{letter2}' belong to same Yati Maitri group {groups1} → MATCH (70%)"

    # No match - explain why
    varga1 = info1.get("varga") if info1 else None
    varga2 = info2.get("varga") if info2 else None

    if varga1 and varga2:
        return f"'{letter1}' is in {varga1}, '{letter2}' is in {varga2} → NO MATCH"

    return f"'{letter1}' and '{letter2}' are not in the same Yati Maitri group → NO MATCH"


def _generate_yati_suggestion(letter: str, info: Optional[Dict]) -> str:
    """
    Generate a suggestion for fixing yati mismatch.

    Args:
        letter: The letter from 1st gana position
        info: Letter info dict

    Returns:
        Suggestion string with valid alternatives
    """
    if not letter or not info:
        return "Unable to generate suggestion"

    group_members = info.get("yati_group_members", [])
    if group_members:
        return f"1st syllable of 3rd gana should start with: {', '.join(group_members)}"

    return f"1st syllable of 3rd gana should start with '{letter}' or related letters"


def calculate_overall_score(gana_score1: Dict, gana_score2: Dict,
                           prasa_score: Dict,
                           yati_score1: Dict, yati_score2: Dict) -> Dict:
    """
    Calculate the weighted overall match score for a Dwipada couplet.

    This function combines individual scores from gana, prasa, and yati
    analysis into a single percentage score using configurable weights.

    Weights (defined in SCORE_WEIGHTS):
    - Gana: 40% (average of both lines)
    - Prasa: 35%
    - Yati: 25% (average of both lines)

    Args:
        gana_score1: Gana score dict for line 1
        gana_score2: Gana score dict for line 2
        prasa_score: Prasa score dict for the couplet
        yati_score1: Yati score dict for line 1
        yati_score2: Yati score dict for line 2

    Returns:
        Dictionary with:
        - overall: Float 0-100 representing weighted average
        - breakdown: Individual component scores
        - weights: The weights used for calculation

    Example:
        >>> calculate_overall_score(gana1, gana2, prasa, yati1, yati2)
        {
            "overall": 85.0,
            "breakdown": {
                "gana_line1": 100.0, "gana_line2": 75.0,
                "prasa": 100.0, "yati_line1": 70.0, "yati_line2": 100.0
            },
            "weights": {"gana": 0.40, "prasa": 0.35, "yati": 0.25}
        }
    """
    # Extract individual scores
    gana1 = gana_score1.get("score", 0.0)
    gana2 = gana_score2.get("score", 0.0)
    prasa = prasa_score.get("score", 0.0)
    yati1 = yati_score1.get("score", 0.0)
    yati2 = yati_score2.get("score", 0.0)

    # Calculate averages for multi-line components
    avg_gana = (gana1 + gana2) / 2
    avg_yati = (yati1 + yati2) / 2

    # Calculate weighted overall score
    overall = (
        avg_gana * SCORE_WEIGHTS["gana"] +
        prasa * SCORE_WEIGHTS["prasa"] +
        avg_yati * SCORE_WEIGHTS["yati"]
    )

    return {
        "overall": round(overall, 1),
        "breakdown": {
            "gana_line1": gana1,
            "gana_line2": gana2,
            "gana_average": round(avg_gana, 1),
            "prasa": prasa,
            "yati_line1": yati1,
            "yati_line2": yati2,
            "yati_average": round(avg_yati, 1),
        },
        "weights": SCORE_WEIGHTS.copy(),
    }


# Indra Gana patterns (3 or 4 syllables)
INDRA_GANAS = {
    "IIII": "Nala (నల)",
    "IIIU": "Naga (నగ)",
    "IIUI": "Sala (సల)",
    "UII": "Bha (భ)",
    "UIU": "Ra (ర)",
    "UUI": "Ta (త)",
}

# Surya Gana patterns (2 or 3 syllables)
SURYA_GANAS = {
    "III": "Na (న)",
    "UI": "Ha/Gala (హ/గల)",
}


###############################################################################
# CORE AKSHARAM SPLITTING FUNCTIONS
###############################################################################

def categorize_aksharam(aksharam: str) -> List[str]:
    """
    Categorize an aksharam with linguistic tags.

    This function analyzes a syllable and returns all applicable
    linguistic categories. These tags are used for Guru/Laghu marking
    and other prosody analysis.

    TAGS RETURNED (Telugu linguistic terms):
        అచ్చు (Vowel): Starts with independent vowel or is a diacritic
        హల్లు (Consonant): Contains any consonant
        దీర్ఘ (Long): Has a long vowel (ా ీ ూ ే ో ౌ)
        విసర్గ అక్షరం (Visarga): Contains visarga (ః)
        అనుస్వారం (Anusvara): Contains anusvara (ం)
        సంయుక్తాక్షరం (Conjunct): Has different consonant cluster (C్C)
        ద్విత్వాక్షరం (Double): Has same consonant doubled (C్C)

    Args:
        aksharam: A single Telugu syllable

    Returns:
        List of applicable tags (sorted alphabetically)

    Example:
        >>> categorize_aksharam("క")
        ['హల్లు']
        >>> categorize_aksharam("కా")
        ['దీర్ఘ', 'హల్లు']
        >>> categorize_aksharam("సం")
        ['అనుస్వారం', 'హల్లు']
        >>> categorize_aksharam("క్ష")
        ['సంయుక్తాక్షరం', 'హల్లు']
        >>> categorize_aksharam("అమ్మ")
        ['ద్విత్వాక్షరం', 'హల్లు']  # మ్మ is doubled మ
    """
    categories = set()

    # Check for vowel (అచ్చు)
    if aksharam[0] in independent_vowels:
        categories.add("అచ్చు")
    elif aksharam in diacritics:
        categories.add("అచ్చు")

    # Check for consonant (హల్లు)
    if any(c in telugu_consonants for c in aksharam):
        categories.add("హల్లు")

    # Check for long vowel (దీర్ఘ)
    if any(dv in aksharam for dv in long_vowels) or aksharam in independent_long_vowels:
        categories.add("దీర్ఘ")

    # Check for visarga (ః) and anusvara (ం)
    if "ః" in aksharam:
        categories.add("విసర్గ అక్షరం")
    if "ం" in aksharam:
        categories.add("అనుస్వారం")

    # Check for conjunct (C్C different) or double (C్C same) consonants
    found_conjunct, found_double = False, False
    for i in range(len(aksharam) - 2):
        if (aksharam[i] in telugu_consonants and
            aksharam[i+1] == halant and
            aksharam[i+2] in telugu_consonants):
            if aksharam[i] == aksharam[i+2]:
                found_double = True    # Same consonant doubled (e.g., మ్మ)
            else:
                found_conjunct = True  # Different consonants (e.g., క్ష)

    if found_conjunct:
        categories.add("సంయుక్తాక్షరం")
    if found_double:
        categories.add("ద్విత్వాక్షరం")

    return sorted(list(categories))


def split_aksharalu(word: str) -> List[str]:
    """
    Split Telugu word into aksharalu (syllables).

    ALGORITHM (Two-Pass):
    Pass 1 - Coarse Split: Break at vowels and consonant boundaries
        - Consonant clusters (C్C) are kept together via halant
        - Dependent vowels attach to preceding consonant
        - Independent vowels form their own syllable

    Pass 2 - Pollu Hallu Merge: Attach trailing consonant+halant to previous syllable
        - "Pollu hallu" (పొల్లు హల్లు) = consonant + halant that can't stand alone
        - These get merged back into the previous syllable

    Telugu Syllable Rules:
        - An aksharam starts with a consonant or vowel
        - Conjunct consonants (సంయుక్తాక్షరం) like త్య belong to one syllable
        - Anusvara (ం) and visarga (ః) attach to the syllable they follow

    Args:
        word: Telugu word or text to split

    Returns:
        List of aksharalu (syllables)

    Example:
        >>> split_aksharalu("తెలుగు")
        ['తె', 'లు', 'గు']
        >>> split_aksharalu("సత్యము")
        ['స', 'త్య', 'ము']  # త్య is kept as one syllable (conjunct)
        >>> split_aksharalu("గౌరవం")
        ['గౌ', 'ర', 'వం']  # ం stays with వ
    """
    # ─────────────────────────────────────────────────────────────────────────
    # PASS 1: Coarse split - identify syllable boundaries
    # ─────────────────────────────────────────────────────────────────────────
    coarse_split = []
    i, n = 0, len(word)

    while i < n:
        # Skip ignorable characters (spaces, arasunna, etc.)
        if word[i] in ignorable_chars:
            coarse_split.append(word[i])
            i += 1
            continue

        current = []
        if word[i] in telugu_consonants:
            # Start with consonant - collect entire consonant cluster
            current.append(word[i])
            i += 1
            # Handle conjunct consonants: C్C్C... (halant chains)
            while i < n and word[i] == halant:
                current.append(word[i])  # Add halant
                i += 1
                if i < n and word[i] in telugu_consonants:
                    current.append(word[i])  # Add next consonant in cluster
                    i += 1
                else:
                    break  # Halant at end (pollu hallu case)
            # Attach dependent vowels and diacritics (ా ి ం ః etc.)
            while i < n and (word[i] in dependent_vowels or word[i] in diacritics):
                current.append(word[i])
                i += 1
        else:
            # Start with vowel (independent vowel like అ ఆ ఇ)
            char = word[i]
            current.append(char)
            i += 1
            # Independent vowel can have diacritics (అం అః)
            if char in independent_vowels and i < n and word[i] in diacritics:
                current.append(word[i])
                i += 1
        coarse_split.append("".join(current))

    if not coarse_split:
        return []

    # ─────────────────────────────────────────────────────────────────────────
    # PASS 2: Merge pollu hallu (trailing consonant+halant) with previous
    # ─────────────────────────────────────────────────────────────────────────
    # Pollu hallu = consonant followed by halant only (no vowel)
    # Example: In "విద్య", after pass 1 we might have ["వి", "ద్", "య"]
    #          Pass 2 merges "ద్" into previous → ["వి", "ద్య"] ... wait, that's not right
    #          Actually this handles edge cases like standalone హల్లు at word boundaries
    final_aksharalu = []
    for chunk in coarse_split:
        # Check if this is a "pollu hallu" - consonant + halant only (2 chars)
        is_pollu_hallu = len(chunk) == 2 and chunk[0] in telugu_consonants and chunk[1] == halant
        if is_pollu_hallu and final_aksharalu and final_aksharalu[-1] not in ignorable_chars:
            # Merge with previous syllable
            final_aksharalu[-1] += chunk
        else:
            final_aksharalu.append(chunk)

    return [ak for ak in final_aksharalu if ak]


def akshara_ganavibhajana(aksharalu_list: List[str]) -> List[str]:
    """
    Mark each syllable as Guru (U/heavy) or Laghu (I/light).

    This is the core prosody analysis that determines syllable weight,
    essential for identifying gana patterns in Telugu poetry.

    GURU RULES (గురువు - heavy syllable):
    A syllable is marked Guru (U) if ANY of these apply:
        Rule 1: Has long vowel (దీర్ఘ) - ా ీ ూ ే ో
        Rule 2: Has diphthong - ై (ai) or ౌ (au)
        Rule 3: Has anusvara (ం) or visarga (ః)
        Rule 4: Ends with halant (్) - incomplete syllable
        Rule 5: NEXT syllable (within same word) starts with conjunct (C్C)
                or double consonant → makes CURRENT syllable Guru (sandhi effect)
                NOTE: This rule does NOT cross word boundaries (spaces).
                Compound words (samasam) written as one word are handled naturally.

    LAGHU RULES (లఘువు - light syllable):
    A syllable is Laghu (I) if NONE of the Guru rules apply.
    Default: short vowel (అ ఇ ఉ ఎ ఒ) without special markers.

    TWO-PASS ALGORITHM:
        Pass 1: Mark based on syllable's own properties (rules 1-4)
        Pass 2: Look ahead - if next syllable is conjunct/double,
                mark current as Guru (rule 5)

    Args:
        aksharalu_list: List of syllables from split_aksharalu()

    Returns:
        List of markers: "U" for Guru, "I" for Laghu, "" for ignorable

    Example:
        >>> akshara_ganavibhajana(['తె', 'లు', 'గు'])
        ['I', 'I', 'I']  # All short vowels → Laghu
        >>> akshara_ganavibhajana(['సం', 'ప', 'ద'])
        ['U', 'I', 'I']  # సం has anusvara → Guru
        >>> akshara_ganavibhajana(['స', 'త్య', 'ము'])
        ['U', 'I', 'I']  # స is before conjunct త్య → becomes Guru
    """
    if not aksharalu_list:
        return []

    ganam_markers = [None] * len(aksharalu_list)

    # ─────────────────────────────────────────────────────────────────────────
    # PASS 1: Mark Guru based on syllable's own properties (Rules 1-4)
    # ─────────────────────────────────────────────────────────────────────────
    for i, aksharam in enumerate(aksharalu_list):
        if aksharam in ignorable_chars:
            ganam_markers[i] = ""
            continue

        ganam_markers[i] = "I"  # Default: Laghu (light)
        tags = set(categorize_aksharam(aksharam))

        is_guru = False
        # Rule 1: Long vowel (దీర్ఘ స్వరం)
        if 'దీర్ఘ' in tags:
            is_guru = True
        # Rule 2: Diphthongs (సంధ్యక్షరం) - ఐ, ఔ
        if 'ఐ' in aksharam or 'ఔ' in aksharam or 'ై' in aksharam or 'ౌ' in aksharam:
            is_guru = True
        # Rule 3: Anusvara or Visarga
        if 'అనుస్వారం' in tags or 'విసర్గ అక్షరం' in tags:
            is_guru = True
        # Rule 4: Ends with halant (incomplete syllable)
        if aksharam.endswith(halant):
            is_guru = True

        if is_guru:
            ganam_markers[i] = "U"

    # ─────────────────────────────────────────────────────────────────────────
    # PASS 2: Sandhi rule - syllable before conjunct/double becomes Guru
    # ─────────────────────────────────────────────────────────────────────────
    # This implements Rule 5: If the NEXT syllable starts with a consonant
    # cluster (conjunct or double), the CURRENT syllable becomes heavy.
    # Linguistic basis: The first consonant of the cluster "closes" the
    # previous syllable, making it a closed syllable (always Guru).
    for i in range(len(aksharalu_list)):
        if ganam_markers[i] == "":
            continue

        # Find next non-ignorable syllable — stop at word boundaries (spaces)
        next_syllable_index = -1
        for j in range(i + 1, len(aksharalu_list)):
            if aksharalu_list[j] == ' ':
                break  # Word boundary — conjunct rule does not cross words
            if aksharalu_list[j] not in ignorable_chars:
                next_syllable_index = j
                break

        if next_syllable_index != -1:
            next_aksharam_tags = set(categorize_aksharam(aksharalu_list[next_syllable_index]))
            # Check if next syllable starts with conjunct or double consonant
            if 'సంయుక్తాక్షరం' in next_aksharam_tags or 'ద్విత్వాక్షరం' in next_aksharam_tags:
                ganam_markers[i] = "U"  # Make current syllable Guru

    return ganam_markers


###############################################################################
# DWIPADA SPECIFIC FUNCTIONS
###############################################################################

def get_base_consonant(aksharam: str) -> Optional[str]:
    """
    Extract the base consonant from an aksharam (syllable).

    For Prasa (ప్రాస) matching in Dwipada poetry, we need to compare
    the consonants of the 2nd syllables of both lines. This function
    extracts the first (base) consonant, ignoring vowel modifiers
    and any conjunct extensions.

    Telugu Syllable Structure:
        [Consonant(s)] + [Vowel/Modifier]
        Example: "క్ష" = "క" + "్" + "ష" → base consonant is "క"
        Example: "కా" = "క" + "ా" → base consonant is "క"

    Args:
        aksharam: A Telugu syllable (single aksharam)

    Returns:
        The first consonant character if present, None if vowel-initial

    Examples:
        >>> get_base_consonant("కా")    # Consonant + long vowel
        "క"
        >>> get_base_consonant("క్ష")   # Conjunct consonant
        "క"
        >>> get_base_consonant("అ")     # Pure vowel
        None
        >>> get_base_consonant("రం")    # Consonant + anusvara
        "ర"
    """
    if not aksharam:
        return None
    first_char = aksharam[0]
    if first_char in telugu_consonants:
        return first_char
    return None


def get_first_letter(aksharam: str) -> Optional[str]:
    """Get the first letter of an aksharam for Yati matching."""
    if not aksharam:
        return None
    return aksharam[0]


def get_independent_vowel(aksharam: str) -> Optional[str]:
    """
    Extract the vowel of an aksharam as its independent vowel form.

    Used for Svara Yati matching — compares vowel families regardless of consonants.
    Dependent vowel signs (ా, ి, ు, etc.) are mapped to their independent forms (ఆ, ఇ, ఉ, etc.).
    If no explicit vowel marker, returns "అ" (implicit inherent vowel).

    Examples:
        >>> get_independent_vowel("కా")   # dependent ా → ఆ
        "ఆ"
        >>> get_independent_vowel("అ")    # independent vowel
        "అ"
        >>> get_independent_vowel("క")    # no vowel marker → implicit అ
        "అ"
    """
    if not aksharam:
        return None
    for dv in dependent_vowels:
        if dv in aksharam:
            return dependent_to_independent[dv]
    if aksharam[0] in independent_vowels:
        return aksharam[0]
    return "అ"


def get_all_consonants(aksharam: str) -> List[str]:
    """
    Extract all consonants from an aksharam (for Samyukta Yati).

    For conjunct aksharalu like "ప్ర", returns all consonant components.
    For simple aksharalu like "క", returns just that consonant.

    Examples:
        >>> get_all_consonants("ప్ర")
        ["ప", "ర"]
        >>> get_all_consonants("క్షా")
        ["క", "ష"]
        >>> get_all_consonants("అ")
        []
    """
    consonants = []
    for ch in aksharam:
        if ch in telugu_consonants:
            consonants.append(ch)
    return consonants


def check_svara_yati(aksharam1: str, aksharam2: str) -> bool:
    """
    Check Svara Yati (స్వర యతి) — vowel family match between two aksharalu.

    Svara Yati is satisfied when the vowels of both aksharalu belong to the same
    vowel family, regardless of the consonants. E.g., "కా" (vowel ఆ) matches
    "అన" (vowel అ) because ఆ and అ are in the same family.
    """
    v1 = get_independent_vowel(aksharam1)
    v2 = get_independent_vowel(aksharam2)
    if not v1 or not v2:
        return False
    if v1 == v2:
        return True
    for group in SVARA_YATI_GROUPS:
        if v1 in group and v2 in group:
            return True
    return False


def check_samyukta_yati(aksharam1: str, aksharam2: str) -> bool:
    """
    Check Samyukta Yati (సంయుక్త యతి) — conjunct consonant harmony.

    When either aksharam is a conjunct (e.g., "ప్ర"), any consonant within the
    conjunct cluster can satisfy yati. The match uses YATI_MAITRI_GROUPS for
    varga equivalence. E.g., "ప్ర" can match with "ర"-varga or "ప"-varga.
    """
    consonants1 = get_all_consonants(aksharam1)
    consonants2 = get_all_consonants(aksharam2)
    if not consonants1 or not consonants2:
        return False
    for c1 in consonants1:
        for c2 in consonants2:
            if c1 == c2:
                return True
            for group in YATI_MAITRI_GROUPS:
                if c1 in group and c2 in group:
                    return True
    return False


def check_bindu_yati(aksharam1: str, aksharam2: str) -> bool:
    """
    Check Bindu Yati (బిందు యతి) — anusvara syllable matches its varga nasal.

    When a syllable contains anusvara (ం), it can form a valid yati match with
    the nasal consonant of its varga. E.g., "కం" can match "ఙ" (K-varga nasal),
    "పం" can match "మ" (P-varga nasal).
    """
    anusvara = "ం"
    for a_with, a_other in [(aksharam1, aksharam2), (aksharam2, aksharam1)]:
        if anusvara not in a_with:
            continue
        base = get_base_consonant(a_with)
        if not base or base not in VARGA_NASALS:
            continue
        other_consonant = get_base_consonant(a_other)
        if not other_consonant:
            # Other starts with a vowel — check if it's the nasal as first letter
            if a_other and a_other[0] == VARGA_NASALS[base]:
                return True
            continue
        varga_nasal = VARGA_NASALS[base]
        if other_consonant == varga_nasal:
            return True
        # Reverse: if other is a nasal, check if base is in its varga
        if other_consonant in NASAL_TO_VARGA:
            if base in NASAL_TO_VARGA[other_consonant]:
                return True
    return False


def check_yati_maitri(letter1: str, letter2: str) -> Tuple[bool, Optional[int], Dict]:
    """
    Check if two letters belong to the same Yati Maitri group.

    Yati (యతి) is the rule of phonetic harmony in Telugu poetry.
    The 1st letter of the 1st gana must match (or be phonetically related to)
    the 1st letter of the 3rd gana in a Dwipada line.

    Match quality levels:
    - Exact match: Same letter (100% quality)
    - Varga match: Same Yati Maitri group (70% quality)
    - No match: Different groups (0% quality)

    Args:
        letter1: First letter (from 1st gana start)
        letter2: Second letter (from 3rd gana start)

    Returns:
        Tuple of (is_match, group_index, details_dict) where:
        - is_match: Boolean indicating if yati is satisfied
        - group_index: Index of matching group (-1 for exact, None for no match)
        - details_dict: Contains quality_score, match_type, and letter info

    Example:
        >>> match, idx, details = check_yati_maitri("స", "స")
        >>> match, details["quality_score"], details["match_type"]
        (True, 100.0, "exact")

        >>> match, idx, details = check_yati_maitri("క", "గ")
        >>> match, details["quality_score"], details["match_type"]
        (True, 70.0, "varga_match")
    """
    details = {
        "letter1": letter1,
        "letter2": letter2,
        "quality_score": YATI_NO_MATCH_SCORE,
        "match_type": "no_match",
        "letter1_info": None,
        "letter2_info": None,
        "matching_group_members": None,
    }

    if not letter1 or not letter2:
        return False, None, details

    # Get detailed info for both letters
    details["letter1_info"] = get_letter_info(letter1)
    details["letter2_info"] = get_letter_info(letter2)

    # Check for exact match first (highest quality)
    if letter1 == letter2:
        details["quality_score"] = YATI_EXACT_MATCH_SCORE
        details["match_type"] = "exact"
        return True, -1, details

    # Check for Yati Maitri group match (medium quality)
    for idx, group in enumerate(YATI_MAITRI_GROUPS):
        if letter1 in group and letter2 in group:
            details["quality_score"] = YATI_VARGA_MATCH_SCORE
            details["match_type"] = "varga_match"
            details["matching_group_members"] = list(group)
            return True, idx, details

    # No match
    details["quality_score"] = YATI_NO_MATCH_SCORE
    details["match_type"] = "no_match"
    return False, None, details


def check_yati_maitri_simple(letter1: str, letter2: str) -> Tuple[bool, Optional[int]]:
    """
    Simple version of check_yati_maitri for backward compatibility.

    Returns only (is_match, group_index) without detailed diagnostics.
    Use check_yati_maitri() for full analysis with quality scoring.

    Args:
        letter1: First letter
        letter2: Second letter

    Returns:
        Tuple of (is_match, group_index)
    """
    is_match, group_idx, _ = check_yati_maitri(letter1, letter2)
    return is_match, group_idx


def are_prasa_equivalent(c1: str, c2: str) -> bool:
    """
    Check if two consonants are equivalent for Prasa matching.

    Returns True if consonants are identical or belong to the same
    PRASA_EQUIVALENTS group (e.g., ల↔ళ, శ↔స, ఱ↔ర).
    """
    if c1 == c2:
        return True
    for group in PRASA_EQUIVALENTS:
        if c1 in group and c2 in group:
            return True
    return False


def check_prasa(line1: str, line2: str) -> Tuple[bool, Dict]:
    """
    Check Prasa (rhyme) between two lines.

    Prasa rule: The 2nd aksharam's base consonant must match between the two lines.
    This is the fundamental rhyming requirement for Dwipada poetry.

    This enhanced version provides detailed mismatch diagnostics including:
    - Full aksharam breakdown (consonant, vowel, varga)
    - Explanation of why mismatch occurred
    - Suggestions for fixing the mismatch

    Args:
        line1: First line of dwipada
        line2: Second line of dwipada

    Returns:
        Tuple of (is_match, details_dict) where details_dict contains:
        - line1_second_aksharam: The 2nd syllable from line 1
        - line1_consonant: Base consonant of line 1's 2nd syllable
        - line2_second_aksharam: The 2nd syllable from line 2
        - line2_consonant: Base consonant of line 2's 2nd syllable
        - match: Boolean indicating if prasa is satisfied
        - mismatch_details: Diagnostic info when match is False (None if match)

    Example:
        >>> is_match, details = check_prasa("సౌధాగ్రముల యందు", "వీధుల యందును")
        >>> is_match
        True  # Both have 'ధ' as 2nd consonant
    """
    # Split lines into aksharalu
    aksharalu1 = split_aksharalu(line1)
    aksharalu2 = split_aksharalu(line2)

    # Filter out spaces/ignorable chars
    pure1 = [ak for ak in aksharalu1 if ak not in ignorable_chars]
    pure2 = [ak for ak in aksharalu2 if ak not in ignorable_chars]

    # Check if we have at least 2 aksharalu
    if len(pure1) < 2 or len(pure2) < 2:
        return False, {"error": "Lines too short - need at least 2 aksharalu each"}

    # Get 2nd aksharam from each
    second_ak1 = pure1[1]
    second_ak2 = pure2[1]

    # Extract base consonant
    consonant1 = get_base_consonant(second_ak1)
    consonant2 = get_base_consonant(second_ak2)

    # Compare consonants (exact match or prasa equivalency like ల↔ళ, శ↔స, ఱ↔ర)
    is_match = are_prasa_equivalent(consonant1, consonant2) if consonant1 and consonant2 else False

    # Build result dictionary
    result = {
        "line1_second_aksharam": second_ak1,
        "line1_consonant": consonant1,
        "line2_second_aksharam": second_ak2,
        "line2_consonant": consonant2,
        "match": is_match,
        "mismatch_details": None,
    }

    # Add detailed diagnostics if not matching
    if not is_match:
        varga1 = get_consonant_varga(consonant1) if consonant1 else None
        varga2 = get_consonant_varga(consonant2) if consonant2 else None

        # Extract vowel from aksharam for full breakdown
        vowel1 = _extract_vowel_from_aksharam(second_ak1)
        vowel2 = _extract_vowel_from_aksharam(second_ak2)

        result["mismatch_details"] = {
            "line1_full_breakdown": {
                "aksharam": second_ak1,
                "consonant": consonant1,
                "vowel": vowel1,
                "consonant_varga": varga1,
            },
            "line2_full_breakdown": {
                "aksharam": second_ak2,
                "consonant": consonant2,
                "vowel": vowel2,
                "consonant_varga": varga2,
            },
            "why_mismatch": _generate_prasa_mismatch_explanation(consonant1, consonant2, varga1, varga2),
            "suggestion": _generate_prasa_suggestion(consonant1),
        }

    return is_match, result


def _extract_vowel_from_aksharam(aksharam: str) -> str:
    """
    Extract the vowel component from a Telugu aksharam.

    An aksharam typically consists of:
    - Consonant(s) + dependent vowel (e.g., కా = క + ా)
    - Or standalone vowel (e.g., అ, ఆ)

    If no explicit vowel marker, returns "అ (implicit)" since Telugu
    consonants without vowel markers have inherent 'అ' sound.

    Args:
        aksharam: A Telugu syllable

    Returns:
        The vowel part as a string (e.g., "ా", "ి", or "అ (implicit)")
    """
    if not aksharam:
        return ""

    # Check for dependent vowels
    for dv in dependent_vowels:
        if dv in aksharam:
            return dv

    # Check if it's an independent vowel
    if aksharam[0] in independent_vowels:
        return aksharam[0]

    # No explicit vowel - inherent 'అ' sound
    return "అ (implicit)"


def check_prasa_aksharalu(aksharam1: str, aksharam2: str) -> Tuple[bool, Dict]:
    """
    Check if two aksharalu have matching Prasa consonant.

    Useful for finding rhyming words or checking individual syllable pairs.

    Args:
        aksharam1: First aksharam/syllable
        aksharam2: Second aksharam/syllable

    Returns:
        Tuple of (is_match, details_dict)
    """
    consonant1 = get_base_consonant(aksharam1)
    consonant2 = get_base_consonant(aksharam2)

    is_match = are_prasa_equivalent(consonant1, consonant2) if consonant1 and consonant2 else False

    return is_match, {
        "aksharam1": aksharam1,
        "consonant1": consonant1,
        "aksharam2": aksharam2,
        "consonant2": consonant2,
        "match": is_match
    }


def identify_gana(pattern: str) -> Tuple[Optional[str], str]:
    """
    Identify the Gana type from a pattern.

    Returns:
        Tuple of (gana_name, gana_type) where gana_type is 'Indra' or 'Surya' or 'Unknown'
    """
    if pattern in INDRA_GANAS:
        return INDRA_GANAS[pattern], "Indra"
    if pattern in SURYA_GANAS:
        return SURYA_GANAS[pattern], "Surya"
    return None, "Unknown"


def find_dwipada_gana_partition(gana_markers: List[str], aksharalu: List[str]) -> Optional[Dict]:
    """
    Try to find a valid Dwipada Gana partition (3 Indra + 1 Surya).

    This function tries all 16 possible combinations of gana lengths:
    - Indra ganas: 3 or 4 syllables (2 choices × 3 ganas = 8 combinations)
    - Surya gana: 2 or 3 syllables (2 choices)
    - Total: 2 × 2 × 2 × 2 = 16 combinations

    The function returns the best partition found, with detailed information
    about how many ganas matched and which ones failed.

    Args:
        gana_markers: List of "U" (Guru) and "I" (Laghu) markers
        aksharalu: List of syllables corresponding to the markers

    Returns:
        Dict with partition details including:
        - ganas: List of 4 gana dicts with name, pattern, aksharalu, type, valid
        - total_syllables: Number of syllables in the line
        - ganas_matched: How many of 4 ganas are valid (0-4)
        - all_partitions_tried: Total partition combinations attempted
        - valid_partitions_found: How many fully valid partitions exist
        - is_fully_valid: True if all 4 ganas match expected types

        Returns None only if line has fewer than 4 syllables.
    """
    # ═══════════════════════════════════════════════════════════════════════════
    # STEP 1: Prepare data - filter empty markers and ignorable characters
    # ═══════════════════════════════════════════════════════════════════════════
    # Remove empty markers (from ignorable chars like spaces)
    pure_ganas = [g for g in gana_markers if g]
    # Remove ignorable characters from aksharalu list
    pure_aksharalu = [ak for ak in aksharalu if ak not in ignorable_chars]

    # Minimum syllables check: theoretical min is 3+3+3+2=11, but >= 4 for safety
    if len(pure_ganas) < 4:
        return None

    # Convert to pattern string for slicing: "UIUIIU..." format
    pattern_str = "".join(pure_ganas)
    valid_partitions = []      # Partitions where all 4 ganas are valid
    all_partitions = []        # All attempted partitions (for finding best partial match)
    partitions_tried = 0

    # ═══════════════════════════════════════════════════════════════════════════
    # STEP 2: Try all 16 gana length combinations
    # ═══════════════════════════════════════════════════════════════════════════
    # Indra ganas: 3 or 4 syllables (Nala, Naga, Sala, Bha, Ra, Ta)
    # Surya ganas: 2 or 3 syllables (Na, Ha/Gala)
    # Combinations: 2 × 2 × 2 × 2 = 16 possibilities
    for i1_len in [3, 4]:           # Indra Gana 1: 3 or 4 syllables
        for i2_len in [3, 4]:       # Indra Gana 2: 3 or 4 syllables
            for i3_len in [3, 4]:   # Indra Gana 3: 3 or 4 syllables
                for s_len in [2, 3]:  # Surya Gana: 2 or 3 syllables
                    # ───────────────────────────────────────────────────────────
                    # CHECK: Does this combination fit the line length?
                    # Example: 3+4+4+2 = 13 syllables
                    # ───────────────────────────────────────────────────────────
                    total_len = i1_len + i2_len + i3_len + s_len

                    if total_len != len(pure_ganas):
                        continue  # Skip: doesn't match line length

                    partitions_tried += 1

                    # ───────────────────────────────────────────────────────────
                    # EXTRACT: Slice patterns for each gana position
                    # ───────────────────────────────────────────────────────────
                    pos = 0
                    i1_pattern = pattern_str[pos:pos + i1_len]
                    pos += i1_len
                    i2_pattern = pattern_str[pos:pos + i2_len]
                    pos += i2_len
                    i3_pattern = pattern_str[pos:pos + i3_len]
                    pos += i3_len
                    s_pattern = pattern_str[pos:pos + s_len]

                    # ───────────────────────────────────────────────────────────
                    # IDENTIFY: Look up gana names and types
                    # Returns (name, type) where type is "Indra", "Surya", or None
                    # ───────────────────────────────────────────────────────────
                    i1_name, i1_type = identify_gana(i1_pattern)
                    i2_name, i2_type = identify_gana(i2_pattern)
                    i3_name, i3_type = identify_gana(i3_pattern)
                    s_name, s_type = identify_gana(s_pattern)

                    # ───────────────────────────────────────────────────────────
                    # MAP: Get corresponding syllables for each gana
                    # ───────────────────────────────────────────────────────────
                    pos = 0
                    i1_aksharalu = pure_aksharalu[pos:pos + i1_len]
                    pos += i1_len
                    i2_aksharalu = pure_aksharalu[pos:pos + i2_len]
                    pos += i2_len
                    i3_aksharalu = pure_aksharalu[pos:pos + i3_len]
                    pos += i3_len
                    s_aksharalu = pure_aksharalu[pos:pos + s_len]

                    # ───────────────────────────────────────────────────────────
                    # VALIDATE: Check if each gana matches expected type
                    # Positions 1-3 must be Indra, position 4 must be Surya
                    # ───────────────────────────────────────────────────────────
                    g1_valid = i1_type == "Indra"
                    g2_valid = i2_type == "Indra"
                    g3_valid = i3_type == "Indra"
                    g4_valid = s_type == "Surya"

                    valid_count = sum([g1_valid, g2_valid, g3_valid, g4_valid])
                    is_fully_valid = valid_count == EXPECTED_GANAS_PER_LINE

                    # Build gana detail with validity info
                    partition_data = {
                        "ganas": [
                            {
                                "position": 1,
                                "name": i1_name,
                                "pattern": i1_pattern,
                                "aksharalu": i1_aksharalu,
                                "type": "Indra",
                                "expected_type": "Indra",
                                "valid": g1_valid,
                                "error": None if g1_valid else f"Pattern '{i1_pattern}' is not a valid Indra gana"
                            },
                            {
                                "position": 2,
                                "name": i2_name,
                                "pattern": i2_pattern,
                                "aksharalu": i2_aksharalu,
                                "type": "Indra",
                                "expected_type": "Indra",
                                "valid": g2_valid,
                                "error": None if g2_valid else f"Pattern '{i2_pattern}' is not a valid Indra gana"
                            },
                            {
                                "position": 3,
                                "name": i3_name,
                                "pattern": i3_pattern,
                                "aksharalu": i3_aksharalu,
                                "type": "Indra",
                                "expected_type": "Indra",
                                "valid": g3_valid,
                                "error": None if g3_valid else f"Pattern '{i3_pattern}' is not a valid Indra gana"
                            },
                            {
                                "position": 4,
                                "name": s_name,
                                "pattern": s_pattern,
                                "aksharalu": s_aksharalu,
                                "type": "Surya",
                                "expected_type": "Surya",
                                "valid": g4_valid,
                                "error": None if g4_valid else f"Pattern '{s_pattern}' is not a valid Surya gana"
                            },
                        ],
                        "total_syllables": len(pure_ganas),
                        "ganas_matched": valid_count,
                        "is_fully_valid": is_fully_valid,
                        "partition_lengths": [i1_len, i2_len, i3_len, s_len],
                    }

                    all_partitions.append(partition_data)

                    if is_fully_valid:
                        valid_partitions.append(partition_data)

    # ═══════════════════════════════════════════════════════════════════════════
    # STEP 3: Return best partition (prefer fully valid, then most matches)
    # ═══════════════════════════════════════════════════════════════════════════

    # No valid length combinations matched the line
    if partitions_tried == 0:
        return None

    # Selection priority:
    # 1. First fully valid partition (all 4 ganas match expected types)
    # 2. Partition with highest ganas_matched count (best partial match)
    if valid_partitions:
        best = valid_partitions[0]
    else:
        # Find partition with most valid ganas
        best = max(all_partitions, key=lambda p: p["ganas_matched"])

    # Add metadata about all attempts
    best["all_partitions_tried"] = partitions_tried
    best["valid_partitions_found"] = len(valid_partitions)

    return best


def analyze_pada(line: str) -> Dict:
    """
    Analyze a single pada (line) of a Dwipada.

    Returns:
        Dict with analysis results
    """
    line = line.strip()
    aksharalu = split_aksharalu(line)
    pure_aksharalu = [ak for ak in aksharalu if ak not in ignorable_chars]
    gana_markers = akshara_ganavibhajana(aksharalu)
    pure_ganas = [g for g in gana_markers if g]
    partition = find_dwipada_gana_partition(gana_markers, aksharalu)

    first_aksharam = pure_aksharalu[0] if len(pure_aksharalu) > 0 else None
    second_aksharam = pure_aksharalu[1] if len(pure_aksharalu) > 1 else None

    # Get first letter/aksharam of 3rd Gana for Yati check
    third_gana_first_letter = None
    third_gana_first_aksharam = None
    if partition and len(partition["ganas"]) >= 3:
        third_gana_aksharalu = partition["ganas"][2]["aksharalu"]
        if third_gana_aksharalu:
            third_gana_first_letter = get_first_letter(third_gana_aksharalu[0])
            third_gana_first_aksharam = third_gana_aksharalu[0]

    return {
        "line": line,
        "aksharalu": pure_aksharalu,
        "gana_markers": pure_ganas,
        "gana_string": "".join(pure_ganas),
        "partition": partition,
        "first_aksharam": first_aksharam,
        "second_aksharam": second_aksharam,
        "first_letter": get_first_letter(first_aksharam) if first_aksharam else None,
        "second_consonant": get_base_consonant(second_aksharam) if second_aksharam else None,
        "third_gana_first_letter": third_gana_first_letter,
        "third_gana_first_aksharam": third_gana_first_aksharam,
        "is_valid_gana_sequence": partition is not None
    }


def analyze_dwipada(poem: str) -> Dict:
    """
    Analyze a complete Dwipada (2 lines separated by newline).

    This is the main analysis function that provides comprehensive feedback on
    a Dwipada couplet including:
    - Per-line gana partition analysis
    - Prasa (rhyme) verification with mismatch diagnostics
    - Yati (alliteration) verification with quality scoring
    - Overall percentage match score

    Args:
        poem: A string containing two lines separated by newline character

    Returns:
        Dict with complete analysis including:
        - pada1, pada2: Per-line analysis results
        - prasa: Prasa check with mismatch details if applicable
        - yati_line1, yati_line2: Yati check with quality scores
        - is_valid_dwipada: Boolean - True if all rules satisfied
        - match_score: Percentage scores (overall and breakdown)
        - validation_summary: Quick boolean summary of all checks

    Example:
        >>> analysis = analyze_dwipada("సౌధాగ్రముల యందు సదనంబు లందు\\nవీధుల యందును వెఱవొప్ప నిలిచి")
        >>> analysis["is_valid_dwipada"]
        True
        >>> analysis["match_score"]["overall"]
        100.0
    """
    lines = [l.strip() for l in poem.strip().split('\n') if l.strip()]
    if len(lines) < 2:
        raise ValueError("Dwipada must have 2 lines separated by newline")
    line1, line2 = lines[0], lines[1]

    # Analyze each pada (line)
    pada1 = analyze_pada(line1)
    pada2 = analyze_pada(line2)

    # Use enhanced check_prasa with full mismatch diagnostics
    prasa_match, prasa_details = check_prasa(line1, line2)

    # Check Yati for each line with enhanced diagnostics
    yati_line1 = None
    yati_line2 = None
    yati_details1 = None
    yati_details2 = None

    if pada1["first_letter"] and pada1["third_gana_first_letter"]:
        match, group_idx, yati_details1 = check_yati_maitri(
            pada1["first_letter"],
            pada1["third_gana_first_letter"]
        )
        match_type = yati_details1.get("match_type", "no_match")

        # If Vyanjana Yati failed, try Svara Yati and Samyukta Yati
        aksharam1 = pada1["first_aksharam"]
        aksharam3 = pada1["third_gana_first_aksharam"]
        if not match and aksharam1 and aksharam3:
            if check_svara_yati(aksharam1, aksharam3):
                match = True
                match_type = "svara_yati"
            elif check_samyukta_yati(aksharam1, aksharam3):
                match = True
                match_type = "samyukta_yati"
            elif check_bindu_yati(aksharam1, aksharam3):
                match = True
                match_type = "bindu_yati"

        yati_line1 = {
            "first_gana_letter": pada1["first_letter"],
            "third_gana_letter": pada1["third_gana_first_letter"],
            "match": match,
            "group_index": group_idx,
            "quality_score": YATI_EXACT_MATCH_SCORE if match else YATI_NO_MATCH_SCORE,
            "match_type": match_type,
            "mismatch_details": yati_details1,
        }

    if pada2["first_letter"] and pada2["third_gana_first_letter"]:
        match, group_idx, yati_details2 = check_yati_maitri(
            pada2["first_letter"],
            pada2["third_gana_first_letter"]
        )
        match_type = yati_details2.get("match_type", "no_match")

        # If Vyanjana Yati failed, try Svara Yati and Samyukta Yati
        aksharam1 = pada2["first_aksharam"]
        aksharam3 = pada2["third_gana_first_aksharam"]
        if not match and aksharam1 and aksharam3:
            if check_svara_yati(aksharam1, aksharam3):
                match = True
                match_type = "svara_yati"
            elif check_samyukta_yati(aksharam1, aksharam3):
                match = True
                match_type = "samyukta_yati"
            elif check_bindu_yati(aksharam1, aksharam3):
                match = True
                match_type = "bindu_yati"

        yati_line2 = {
            "first_gana_letter": pada2["first_letter"],
            "third_gana_letter": pada2["third_gana_first_letter"],
            "match": match,
            "group_index": group_idx,
            "quality_score": YATI_EXACT_MATCH_SCORE if match else YATI_NO_MATCH_SCORE,
            "match_type": match_type,
            "mismatch_details": yati_details2,
        }

    # Calculate scores for each component
    gana_score1 = calculate_gana_score(pada1.get("partition"))
    gana_score2 = calculate_gana_score(pada2.get("partition"))
    prasa_score_result = calculate_prasa_score(prasa_details)
    yati_score1 = calculate_yati_score(yati_line1)
    yati_score2 = calculate_yati_score(yati_line2)

    # Calculate overall weighted score
    match_score = calculate_overall_score(
        gana_score1, gana_score2,
        prasa_score_result,
        yati_score1, yati_score2
    )

    # Add component scores to match_score breakdown
    match_score["component_scores"] = {
        "gana_line1": gana_score1,
        "gana_line2": gana_score2,
        "prasa": prasa_score_result,
        "yati_line1": yati_score1,
        "yati_line2": yati_score2,
    }

    # Determine overall validity (binary pass/fail for strict validation)
    is_valid = (
        pada1["is_valid_gana_sequence"] and
        pada2["is_valid_gana_sequence"] and
        prasa_match and
        (yati_line1 is None or yati_line1["match"]) and
        (yati_line2 is None or yati_line2["match"])
    )

    return {
        "pada1": pada1,
        "pada2": pada2,
        "prasa": prasa_details,
        "yati_line1": yati_line1,
        "yati_line2": yati_line2,
        "is_valid_dwipada": is_valid,
        "match_score": match_score,
        "validation_summary": {
            "gana_sequence_line1": pada1["is_valid_gana_sequence"],
            "gana_sequence_line2": pada2["is_valid_gana_sequence"],
            "prasa_match": prasa_match,
            "yati_line1_match": yati_line1["match"] if yati_line1 else None,
            "yati_line2_match": yati_line2["match"] if yati_line2 else None,
        }
    }


def format_analysis_report(analysis: Dict) -> str:
    """
    Format the analysis as a human-readable report.

    This function creates a comprehensive report showing:
    - Per-line analysis with gana breakdown
    - Prasa (rhyme) analysis with mismatch diagnostics
    - Yati (alliteration) analysis with quality scores
    - Overall match percentage and validation summary

    Args:
        analysis: Dict returned by analyze_dwipada()

    Returns:
        Formatted string report suitable for display
    """
    lines = []
    lines.append("=" * 70)
    lines.append("DWIPADA CHANDASSU ANALYSIS REPORT")
    lines.append("=" * 70)

    # Match Score Summary (NEW - show percentage at top)
    if "match_score" in analysis:
        score = analysis["match_score"]
        overall = score.get("overall", 0)
        lines.append(f"\n📊 OVERALL MATCH SCORE: {overall:.1f}%")
        lines.append("-" * 35)

    # Line 1 Analysis
    lines.append("\n--- LINE 1 (పాదము 1) ---")
    pada1 = analysis["pada1"]
    lines.append(f"Text: {pada1['line']}")
    lines.append(f"Aksharalu: {' | '.join(pada1['aksharalu'])}")
    lines.append(f"Gana Markers: {' '.join(pada1['gana_markers'])}")

    if pada1["partition"]:
        partition = pada1["partition"]
        ganas_matched = partition.get("ganas_matched", 4)
        lines.append(f"\nGana Breakdown ({ganas_matched}/4 valid):")
        for gana in partition["ganas"]:
            gana_type_label = "ఇంద్ర గణము" if gana["type"] == "Indra" else "సూర్య గణము"
            valid_mark = "✓" if gana.get("valid", True) else "✗"
            name_str = gana['name'] if gana['name'] else "[Invalid]"
            lines.append(f"  {valid_mark} Gana {gana.get('position', '?')}: {''.join(gana['aksharalu'])} = {gana['pattern']} = {name_str} ({gana_type_label})")
            # Show error message if gana is invalid
            if not gana.get("valid", True) and gana.get("error"):
                lines.append(f"      ↳ {gana['error']}")
    else:
        lines.append("\n[WARNING] Could not find valid 3 Indra + 1 Surya Gana partition")

    # Line 2 Analysis
    lines.append("\n--- LINE 2 (పాదము 2) ---")
    pada2 = analysis["pada2"]
    lines.append(f"Text: {pada2['line']}")
    lines.append(f"Aksharalu: {' | '.join(pada2['aksharalu'])}")
    lines.append(f"Gana Markers: {' '.join(pada2['gana_markers'])}")

    if pada2["partition"]:
        partition = pada2["partition"]
        ganas_matched = partition.get("ganas_matched", 4)
        lines.append(f"\nGana Breakdown ({ganas_matched}/4 valid):")
        for gana in partition["ganas"]:
            gana_type_label = "ఇంద్ర గణము" if gana["type"] == "Indra" else "సూర్య గణము"
            valid_mark = "✓" if gana.get("valid", True) else "✗"
            name_str = gana['name'] if gana['name'] else "[Invalid]"
            lines.append(f"  {valid_mark} Gana {gana.get('position', '?')}: {''.join(gana['aksharalu'])} = {gana['pattern']} = {name_str} ({gana_type_label})")
            if not gana.get("valid", True) and gana.get("error"):
                lines.append(f"      ↳ {gana['error']}")
    else:
        lines.append("\n[WARNING] Could not find valid 3 Indra + 1 Surya Gana partition")

    # Prasa Analysis with enhanced diagnostics
    lines.append("\n--- PRASA (ప్రాస) ANALYSIS ---")
    if analysis["prasa"]:
        prasa = analysis["prasa"]
        status = "✓ MATCH" if prasa["match"] else "✗ NO MATCH"
        lines.append(f"Line 1 - 2nd Aksharam: '{prasa['line1_second_aksharam']}' (Consonant: {prasa['line1_consonant']})")
        lines.append(f"Line 2 - 2nd Aksharam: '{prasa['line2_second_aksharam']}' (Consonant: {prasa['line2_consonant']})")
        lines.append(f"Prasa Status: {status}")

        # Show mismatch diagnostics if prasa doesn't match
        if not prasa["match"] and prasa.get("mismatch_details"):
            details = prasa["mismatch_details"]
            lines.append("\n  📋 Mismatch Details:")
            if details.get("line1_full_breakdown"):
                bd1 = details["line1_full_breakdown"]
                lines.append(f"    Line 1: '{bd1.get('aksharam')}' → consonant '{bd1.get('consonant')}' ({bd1.get('consonant_varga', 'unknown')})")
            if details.get("line2_full_breakdown"):
                bd2 = details["line2_full_breakdown"]
                lines.append(f"    Line 2: '{bd2.get('aksharam')}' → consonant '{bd2.get('consonant')}' ({bd2.get('consonant_varga', 'unknown')})")
            if details.get("why_mismatch"):
                lines.append(f"    Why: {details['why_mismatch']}")
            if details.get("suggestion"):
                lines.append(f"    💡 Suggestion: {details['suggestion']}")
    else:
        lines.append("Could not determine Prasa")

    # Yati Analysis with enhanced diagnostics
    lines.append("\n--- YATI (యతి) ANALYSIS ---")

    if analysis["yati_line1"]:
        yati1 = analysis["yati_line1"]
        match_type = yati1.get("match_type", "unknown")
        quality = yati1.get("quality_score", 0)
        status = f"✓ MATCH ({match_type}, {quality:.0f}%)" if yati1["match"] else "✗ NO MATCH"
        lines.append(f"Line 1: '{yati1['first_gana_letter']}' (1st gana) ↔ '{yati1['third_gana_letter']}' (3rd gana) - {status}")

        # Show details for mismatches or varga matches
        if not yati1["match"] or match_type == "varga_match":
            mismatch = yati1.get("mismatch_details", {})
            if mismatch:
                info1 = mismatch.get("letter1_info", {})
                info2 = mismatch.get("letter2_info", {})
                if info1 and info2:
                    lines.append(f"    '{yati1['first_gana_letter']}' groups: {info1.get('yati_group_members', [])}")
                if mismatch.get("matching_group_members"):
                    lines.append(f"    Matching group: {mismatch['matching_group_members']}")
    else:
        lines.append("Line 1: Could not determine Yati")

    if analysis["yati_line2"]:
        yati2 = analysis["yati_line2"]
        match_type = yati2.get("match_type", "unknown")
        quality = yati2.get("quality_score", 0)
        status = f"✓ MATCH ({match_type}, {quality:.0f}%)" if yati2["match"] else "✗ NO MATCH"
        lines.append(f"Line 2: '{yati2['first_gana_letter']}' (1st gana) ↔ '{yati2['third_gana_letter']}' (3rd gana) - {status}")

        if not yati2["match"] or match_type == "varga_match":
            mismatch = yati2.get("mismatch_details", {})
            if mismatch:
                info1 = mismatch.get("letter1_info", {})
                if info1:
                    lines.append(f"    '{yati2['first_gana_letter']}' groups: {info1.get('yati_group_members', [])}")
                if mismatch.get("matching_group_members"):
                    lines.append(f"    Matching group: {mismatch['matching_group_members']}")
    else:
        lines.append("Line 2: Could not determine Yati")

    # Score Breakdown (NEW)
    if "match_score" in analysis:
        lines.append("\n--- SCORE BREAKDOWN ---")
        score = analysis["match_score"]
        breakdown = score.get("breakdown", {})
        weights = score.get("weights", {})

        lines.append(f"  Gana (weight {weights.get('gana', 0.4)*100:.0f}%):")
        lines.append(f"    Line 1: {breakdown.get('gana_line1', 0):.1f}%")
        lines.append(f"    Line 2: {breakdown.get('gana_line2', 0):.1f}%")
        lines.append(f"    Average: {breakdown.get('gana_average', 0):.1f}%")

        lines.append(f"  Prasa (weight {weights.get('prasa', 0.35)*100:.0f}%): {breakdown.get('prasa', 0):.1f}%")

        lines.append(f"  Yati (weight {weights.get('yati', 0.25)*100:.0f}%):")
        lines.append(f"    Line 1: {breakdown.get('yati_line1', 0):.1f}%")
        lines.append(f"    Line 2: {breakdown.get('yati_line2', 0):.1f}%")
        lines.append(f"    Average: {breakdown.get('yati_average', 0):.1f}%")

    # Summary
    lines.append("\n" + "=" * 70)
    lines.append("VALIDATION SUMMARY")
    lines.append("=" * 70)
    summary = analysis["validation_summary"]
    lines.append(f"Gana Sequence Line 1: {'✓ Valid' if summary['gana_sequence_line1'] else '✗ Invalid'}")
    lines.append(f"Gana Sequence Line 2: {'✓ Valid' if summary['gana_sequence_line2'] else '✗ Invalid'}")
    lines.append(f"Prasa Match: {'✓ Yes' if summary['prasa_match'] else '✗ No'}")
    lines.append(f"Yati Line 1: {'✓ Match' if summary['yati_line1_match'] else '✗ No Match' if summary['yati_line1_match'] is False else 'N/A'}")
    lines.append(f"Yati Line 2: {'✓ Match' if summary['yati_line2_match'] else '✗ No Match' if summary['yati_line2_match'] is False else 'N/A'}")
    lines.append("")

    # Final verdict with percentage
    if "match_score" in analysis:
        overall = analysis["match_score"].get("overall", 0)
        lines.append(f"OVERALL: {'✓ VALID DWIPADA' if analysis['is_valid_dwipada'] else '✗ INVALID DWIPADA'} ({overall:.1f}% match)")
    else:
        lines.append(f"OVERALL: {'✓ VALID DWIPADA' if analysis['is_valid_dwipada'] else '✗ INVALID DWIPADA'}")
    lines.append("=" * 70)

    return "\n".join(lines)


def analyze_single_line(line: str) -> str:
    """
    Analyze a single line and return a formatted report.
    Useful for quick analysis of individual padas.
    """
    pada = analyze_pada(line)
    lines = []
    lines.append("=" * 60)
    lines.append("SINGLE LINE ANALYSIS")
    lines.append("=" * 60)
    lines.append(f"Text: {pada['line']}")
    lines.append(f"Aksharalu: {' | '.join(pada['aksharalu'])}")
    lines.append(f"Gana Markers: {' '.join(pada['gana_markers'])}")
    lines.append(f"Gana String: {pada['gana_string']}")

    if pada["partition"]:
        lines.append("\nGana Breakdown (3 Indra + 1 Surya):")
        for i, gana in enumerate(pada["partition"]["ganas"], 1):
            gana_type_label = "ఇంద్ర" if gana["type"] == "Indra" else "సూర్య"
            aksharalu_str = "".join(gana['aksharalu'])
            lines.append(f"  {i}. {aksharalu_str} = {gana['pattern']} = {gana['name']} ({gana_type_label})")
        lines.append(f"\n✓ Valid Dwipada line structure")
    else:
        lines.append(f"\n✗ Could not find valid 3 Indra + 1 Surya partition")

    lines.append("=" * 60)
    return "\n".join(lines)


###############################################################################
# MAIN - COMPREHENSIVE TEST SUITE
###############################################################################

