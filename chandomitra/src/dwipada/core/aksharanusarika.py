# -*- coding: utf-8 -*-
"""
Aksharanusarika v0.0.7a
-----------------------
A Python script for advanced Telugu linguistic and prosody (Gana) analysis.

This version is a direct port of the features from the v0.0.7a JavaScript version,
implementing new features such as:
- Gana Combination Analysis (GanaAnalyzer)
- Jaccard Similarity for Gana Sequences
- Jaccard Similarity for Linguistic Features
- Logic parity with JS version for aksharam splitting and categorization.
"""

import re
import hashlib
import datetime
import json
from itertools import combinations

try:
    import pytz
    HAS_PYTZ = True
except ImportError:
    HAS_PYTZ = False

###############################################################################
# 1) LINGUISTIC DATA AND CONSTANTS (v0.0.7a)
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
ignorable_chars = {' ', '\n', 'ఁ', '​'} # Includes space, newline, arasunna, zero-width space

PLUTAMULU     = {"ఐ", "ఔ"}
SARALAMULU    = {"గ", "జ", "డ", "ద", "బ"}
PARUSHAMULU   = {"క", "చ", "ట", "త", "ప"}
STHIRAMULU    = {
    "ఖ", "ఘ", "ఙ", "ఛ", "ఝ", "ఞ", "ఠ", "ఢ", "ణ", "థ", "ధ", "న",
    "ఫ", "భ", "మ", "య", "ర", "ఱ", "ల", "ళ", "వ", "శ", "ష", "స", "హ"
}
KA_VARGAMU    = {"క", "ఖ", "గ", "ఘ", "ఙ"}
CHA_VARGAMU   = {"చ", "ౘ", "ఛ", "జ", "ౙ", "ఝ", "ఞ"}
TA_VARGAMU    = {"ట", "ఠ", "డ", "ఢ", "ణ"}
THA_VARGAMU   = {"త", "థ", "ద", "ధ", "న"}
PA_VARGAMU    = {"ప", "ఫ", "బ", "భ", "మ"}
SPARSHA_MULU  = KA_VARGAMU.union(CHA_VARGAMU).union(TA_VARGAMU).union(THA_VARGAMU).union(PA_VARGAMU)
OOSHMA_MULU        = {"శ", "స", "ష", "హ"}
ANTASTA_MULU       = {"య", "ర", "ఱ", "ల", "ళ", "వ"}
KANTHYAMULU        = {"అ", "ఆ", "క", "ఖ", "గ", "ఘ", "ఙ", "హ"}
TAALAVYAMULU       = {"ఇ", "ఈ", "చ", "ఛ", "జ", "ఝ", "య", "శ"}
MOORDHANYAMULU     = {"ఋ", "ౠ", "ట", "ఠ", "డ", "ఢ", "ణ", "ష", "ఱ", "ర"}
DANTYAMULU         = {"ఌ", "ౡ", "త", "థ", "ద", "ధ", "ౘ", "ౙ", "ల", "స"}
OOSHTYAMULU        = {"ఉ", "ఊ", "ప", "ఫ", "బ", "భ", "మ"}
ANUNAASIKA_MULU    = {"ఙ", "ఞ", "ణ", "న", "మ"}
KANTHATAALAVYA_MULU= {"ఎ", "ఏ", "ఐ"}
KANTHOSH_TYAMULU   = {"ఒ", "ఓ", "ఔ"}
DANTOSH_TYAMULU   = {"వ"}

# NEW: Gana Definitions (from v0.0.7a JS)
GANA_DEFINITIONS = {
    "Ekaakshara Ganas (1-Syllable)": {"U": "Guru", "I": "Laghu"},
    "Rendakshara Ganas (2-Syllable)": {"II": "Lalamu", "IU": "Lagamu (Va)", "UI": "Galamu (Ha)", "UU": "Gagamu"},
    "Moodakshara Ganas (3-Syllable)": {"IUU": "Ya", "UUU": "Ma", "UUI": "Ta", "UIU": "Ra", "IUI": "Ja", "UII": "Bha", "III": "Na", "IIU": "Sa"},
    "Surya Ganas": {"III": "Na", "UI": "Ha"}, # Note: Overlaps are intentional as per definitions
    "Indra Ganas": {"IIIU": "Naga", "IIUI": "Sala", "IIII": "Nala", "UII": "Bha", "UIU": "Ra", "UUI": "Ta"},
    "Chandra Ganas": {"UIII": "Bhala", "UIIU": "Bhagaru", "UUII": "Tala", "UUIU": "Taga", "UUUI": "Malagha", "IIIII": "Nalala", "IIIUU": "Nagaga", "IIIIU": "Nava", "IIUUI": "Saha", "IIUIU": "Sava", "IIUUU": "Sagaga", "IIIUI": "Naha", "UIUU": "Raguru", "IIII": "Nala"}
}


###############################################################################
# 2) CORE LOGIC FUNCTIONS (v0.0.7a)
###############################################################################

def simple_hash(s):
    """
    NEW: Creates a simple, non-cryptographic hash for a unique ID.
    Matches the v0.0.7a JS simpleHash function.
    """
    hash_val = 0
    for char in s:
        chr_val = ord(char)
        hash_val = ((hash_val << 5) - hash_val) + chr_val
        hash_val &= 0xFFFFFFFF # Ensure 32-bit integer
    # Convert to unsigned 32-bit and then to hex
    return 'id-' + hex(hash_val & 0xFFFFFFFF)[2:]

def add_letter_categories(ch, categories):
    """
    Adds linguistic categories for a given character.
    (Logic unchanged, confirmed identical to JS)
    """
    if ch in PLUTAMULU: categories.add("ప్లుతములు")
    if ch in SARALAMULU: categories.add("సరళములు")
    if ch in PARUSHAMULU: categories.add("పరుషములు")
    if ch in STHIRAMULU: categories.add("స్థిరములు")
    if ch in KA_VARGAMU: categories.add("క వర్గము")
    if ch in CHA_VARGAMU: categories.add("చ వర్గము")
    if ch in TA_VARGAMU: categories.add("ట వర్గము")
    if ch in THA_VARGAMU: categories.add("త వర్గము")
    if ch in PA_VARGAMU: categories.add("ప వర్గము")
    if ch in SPARSHA_MULU: categories.add("స్పర్శములు")
    if ch in OOSHMA_MULU: categories.add("ఊష్మాలు")
    if ch in ANTASTA_MULU: categories.add("అంతస్తములు")
    if ch in KANTHYAMULU: categories.add("కంఠ్యములు")
    if ch in TAALAVYAMULU: categories.add("తాలవ్యములు")
    if ch in MOORDHANYAMULU: categories.add("మూర్ధన్యములు")
    if ch in DANTYAMULU: categories.add("దంత్యములు")
    if ch in OOSHTYAMULU: categories.add("ఓష్ఠ్యములు")
    if ch in ANUNAASIKA_MULU: categories.add("అనునాసికములు")
    if ch in KANTHATAALAVYA_MULU: categories.add("కంఠతాలవ్యములు")
    if ch in KANTHOSH_TYAMULU: categories.add("కంఠోష్ఠ్యములు")
    if ch in DANTOSH_TYAMULU: categories.add("దంత్యోష్ఠ్యములు")

def categorize_aksharam(aksharam):
    """
    REFACTORED: Logic updated to match v0.0.7a JS logic exactly.
    """
    categories = set()

    if aksharam[0] in independent_vowels: categories.add("అచ్చు")
    elif aksharam in diacritics: categories.add("అచ్చు")

    if any(c in telugu_consonants for c in aksharam): categories.add("హల్లు")

    if any(dv in aksharam for dv in long_vowels) or aksharam in independent_long_vowels:
        categories.add("దీర్ఘ")

    if "ః" in aksharam: categories.add("విసర్గ అక్షరం")
    if "ం" in aksharam: categories.add("అనుస్వారం")

    found_conjunct, found_double = False, False
    for i in range(len(aksharam) - 2):
        if (aksharam[i] in telugu_consonants and
            aksharam[i+1] == halant and
            aksharam[i+2] in telugu_consonants):
            if aksharam[i] == aksharam[i+2]: found_double = True
            else: found_conjunct = True

    if found_conjunct: categories.add("సంయుక్తాక్షరం")
    if found_double: categories.add("ద్విత్వాక్షరం")

    if (("హల్లు" in categories or "అచ్చు" in categories) and
        ("దీర్ఘ" not in categories) and
        ("అనుస్వారం" not in categories) and
        ("విసర్గ అక్షరం" not in categories)):
        categories.add("హ్రస్వాక్షరం")

    for ch in aksharam:
        add_letter_categories(ch, categories)

    found_dependent_vowel = False
    for dv_sign, dv_vowel in dependent_to_independent.items():
        if dv_sign in aksharam:
            found_dependent_vowel = True
            add_letter_categories(dv_vowel, categories)

    if any(c in telugu_consonants for c in aksharam) and not found_dependent_vowel and not aksharam.endswith(halant):
         add_letter_categories("అ", categories)

    return sorted(list(categories))

def split_aksharalu(word):
    """
    REFACTORED: Logic updated to match v0.0.7a JS logic (two-pass coarse split + pollu merge).
    """
    coarse_split = []
    i, n = 0, len(word)

    while i < n:
        if word[i] in ignorable_chars:
            coarse_split.append(word[i])
            i += 1
            continue

        current = []
        if word[i] in telugu_consonants:
            current.append(word[i])
            i += 1
            while i < n and word[i] == halant:
                current.append(word[i])
                i += 1
                if i < n and word[i] in telugu_consonants:
                    current.append(word[i])
                    i += 1
                else: break
            while i < n and (word[i] in dependent_vowels or word[i] in diacritics):
                current.append(word[i])
                i += 1
        else:
            char = word[i]
            current.append(char)
            i += 1
            if char in independent_vowels and i < n and word[i] in diacritics:
                current.append(word[i])
                i += 1
        coarse_split.append("".join(current))

    if not coarse_split:
        return []

    # Second pass: merge pollu hallu (e.g., "న్") with the previous aksharam
    final_aksharalu = []
    for chunk in coarse_split:
        is_pollu_hallu = len(chunk) == 2 and chunk[0] in telugu_consonants and chunk[1] == halant
        if is_pollu_hallu and final_aksharalu and final_aksharalu[-1] not in ignorable_chars:
            final_aksharalu[-1] += chunk
        else:
            final_aksharalu.append(chunk)

    return [ak for ak in final_aksharalu if ak]

def akshara_ganavibhajana(aksharalu_list):
    """
    Performs prosody (Gana) analysis.
    (Logic confirmed identical to v0.0.7a JS two-pass system)
    """
    if not aksharalu_list:
        return []

    ganam_markers = [None] * len(aksharalu_list)

    # First Pass: Identify Gurus based on their own properties
    for i, aksharam in enumerate(aksharalu_list):
        if aksharam in ignorable_chars:
            ganam_markers[i] = ""
            continue

        ganam_markers[i] = "I" # Default to Laghu
        tags = set(categorize_aksharam(aksharam))

        is_guru = False
        if 'దీర్ఘ' in tags: is_guru = True
        if 'ఐ' in aksharam or 'ఔ' in aksharam or 'ై' in aksharam or 'ౌ' in aksharam: is_guru = True
        if 'అనుస్వారం' in tags or 'విసర్గ అక్షరం' in tags: is_guru = True
        if aksharam.endswith(halant): is_guru = True
        if is_guru: ganam_markers[i] = "U"

    # Second pass: Handle the contextual rule (syllable before conjunct/double)
    for i in range(len(aksharalu_list)):
        if ganam_markers[i] == "": continue

        # Find the next non-ignorable syllable
        next_syllable_index = -1
        for j in range(i + 1, len(aksharalu_list)):
            if aksharalu_list[j] not in ignorable_chars:
                next_syllable_index = j
                break

        if next_syllable_index != -1:
            next_aksharam_tags = set(categorize_aksharam(aksharalu_list[next_syllable_index]))
            if 'సంయుక్తాక్షరం' in next_aksharam_tags or 'ద్విత్వాక్షరం' in next_aksharam_tags:
                ganam_markers[i] = "U"

    return ganam_markers

class GanaAnalyzer:
    """
    NEW: Port of the GanaAnalyzer class from v0.0.7a JS.
    Finds all possible sequential Gana combinations in a prosody string.
    """
    def __init__(self, definitions):
        self.definitions = definitions
        self.flat_ganas = self._flatten_definitions(definitions)
        self.memo = {}

    def _flatten_definitions(self, definitions):
        flat_map = {}
        for category, ganas in definitions.items():
            for pattern, name in ganas.items():
                flat_map[pattern] = name
        return flat_map

    def find_sequential_combinations(self, syllables):
        self.memo = {}  # Reset memo for each new analysis
        return self._find_combinations_recursive_memoized(tuple(syllables))

    def _find_combinations_recursive_memoized(self, remaining_syllables):
        key = remaining_syllables
        if key in self.memo:
            return self.memo[key]
        if not remaining_syllables:
            return [[]]

        all_possible_partitions = []
        for i in range(1, len(remaining_syllables) + 1):
            prefix = "".join(remaining_syllables[:i])
            if prefix in self.flat_ganas:
                gana_info = {"name": self.flat_ganas[prefix], "pattern": prefix}
                suffix = remaining_syllables[i:]
                suffix_combinations = self._find_combinations_recursive_memoized(suffix)
                for combo in suffix_combinations:
                    all_possible_partitions.append([gana_info] + combo)

        self.memo[key] = all_possible_partitions
        return all_possible_partitions

def map_syllables_to_partition(partition, syllables):
    """
    NEW: Helper function to map a Gana partition back to the original aksharalu.
    """
    mapped_partition = []
    syllable_index = 0
    for gana in partition:
        pattern_len = len(gana["pattern"])
        syllable_slice = syllables[syllable_index : syllable_index + pattern_len]
        syllable_text = "".join(syllable_slice)

        mapped_partition.append({
            "syllable_text": syllable_text,
            "name": gana["name"],
            "pattern": gana["pattern"]
        })
        syllable_index += pattern_len
    return mapped_partition

def calculate_gana_jaccard(markers1, markers2):
    """
    NEW: Calculates Jaccard similarity/distance based on Gana bigrams.
    """
    def get_bigrams(markers):
        bigrams = set()
        pure_markers = [m for m in markers if m]
        if len(pure_markers) < 2:
            return bigrams
        for i in range(len(pure_markers) - 1):
            bigrams.add(pure_markers[i] + pure_markers[i+1])
        return bigrams

    bigrams1 = get_bigrams(markers1)
    bigrams2 = get_bigrams(markers2)

    intersection = bigrams1.intersection(bigrams2)
    union = bigrams1.union(bigrams2)

    similarity = len(intersection) / len(union) if len(union) > 0 else 1.0
    distance = 1.0 - similarity

    return {"similarity": similarity, "distance": distance}

def find_longest_common_substring(seq1, seq2):
    """
    Finds the longest common substring between two sequences.
    (Logic unchanged, confirmed identical to JS)
    """
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    max_length = 0
    end_index = 0

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                if dp[i][j] > max_length:
                    max_length = dp[i][j]
                    end_index = i
            else:
                dp[i][j] = 0

    if max_length == 0:
        return []

    return seq1[end_index - max_length : end_index]

###############################################################################
# 3) JSON OUTPUT HELPER FUNCTIONS (v0.0.7a+)
###############################################################################

def calculate_linguistic_statistics(analysis):
    """
    Calculates comprehensive linguistic statistics from analysis result.
    """
    category_counts = analysis["categoryCounts"]
    aksharalu_list = analysis["aksharaluList"]

    # Filter out ignorable characters
    pure_aksharalu = [ak for ak in aksharalu_list if ak not in ignorable_chars]

    vowel_count = category_counts.get("అచ్చు", 0)
    consonant_count = category_counts.get("హల్లు", 0)
    long_vowel_count = category_counts.get("దీర్ఘ", 0)
    short_vowel_count = category_counts.get("హ్రస్వాక్షరం", 0)
    conjunct_count = category_counts.get("సంయుక్తాక్షరం", 0)
    doublet_count = category_counts.get("ద్విత్వాక్షరం", 0)
    anusvaram_count = category_counts.get("అనుస్వారం", 0)
    visarga_count = category_counts.get("విసర్గ అక్షరం", 0)

    total_aksharas = len(pure_aksharalu)
    unique_aksharas = len(analysis["aksharalu"])

    return {
        "totalAksharas": total_aksharas,
        "uniqueAksharas": unique_aksharas,
        "vowelCount": vowel_count,
        "consonantCount": consonant_count,
        "vowelToConsonantRatio": round(vowel_count / consonant_count, 3) if consonant_count > 0 else 0,
        "longVowelCount": long_vowel_count,
        "shortVowelCount": short_vowel_count,
        "conjunctCount": conjunct_count,
        "doubletCount": doublet_count,
        "anusvaaramCount": anusvaram_count,
        "visargaCount": visarga_count,
        "complexityScore": round((conjunct_count + doublet_count) / total_aksharas * 100, 2) if total_aksharas > 0 else 0
    }

def calculate_vargam_distribution(analysis):
    """
    Calculates distribution of consonant groups (vargams).
    """
    category_counts = analysis["categoryCounts"]

    return {
        "క వర్గము": category_counts.get("క వర్గము", 0),
        "చ వర్గము": category_counts.get("చ వర్గము", 0),
        "ట వర్గము": category_counts.get("ట వర్గము", 0),
        "త వర్గము": category_counts.get("త వర్గము", 0),
        "ప వర్గము": category_counts.get("ప వర్గము", 0),
        "స్పర్శములు": category_counts.get("స్పర్శములు", 0),
        "ఊష్మాలు": category_counts.get("ఊష్మాలు", 0),
        "అంతస్తములు": category_counts.get("అంతస్తములు", 0)
    }

def calculate_articulation_distribution(analysis):
    """
    Calculates distribution of place-of-articulation categories.
    """
    category_counts = analysis["categoryCounts"]

    return {
        "కంఠ్యములు": category_counts.get("కంఠ్యములు", 0),
        "తాలవ్యములు": category_counts.get("తాలవ్యములు", 0),
        "మూర్ధన్యములు": category_counts.get("మూర్ధన్యములు", 0),
        "దంత్యములు": category_counts.get("దంత్యములు", 0),
        "ఓష్ఠ్యములు": category_counts.get("ఓష్ఠ్యములు", 0),
        "అనునాసికములు": category_counts.get("అనునాసికములు", 0),
        "కంఠతాలవ్యములు": category_counts.get("కంఠతాలవ్యములు", 0),
        "కంఠోష్ఠ్యములు": category_counts.get("కంఠోష్ఠ్యములు", 0),
        "దంత్యోష్ఠ్యములు": category_counts.get("దంత్యోష్ఠ్యములు", 0)
    }

def calculate_prosody_statistics(gana_markers, gana_combinations):
    """
    Calculates comprehensive prosody statistics.
    """
    pure_ganas = [m for m in gana_markers if m]

    if not pure_ganas:
        return {
            "totalSyllables": 0,
            "guruCount": 0,
            "laghuCount": 0,
            "guruToLaghuRatio": 0,
            "guruPercentage": 0,
            "laghuPercentage": 0,
            "mostCommonGana": None,
            "ganaVariety": 0
        }

    guru_count = pure_ganas.count("U")
    laghu_count = pure_ganas.count("I")
    total_syllables = len(pure_ganas)

    # Find most common gana pattern from combinations
    most_common_gana = None
    gana_variety = 0

    if gana_combinations and len(gana_combinations) > 0:
        gana_names = {}
        for combo in gana_combinations:
            for gana in combo:
                name = gana["name"]
                gana_names[name] = gana_names.get(name, 0) + 1

        if gana_names:
            most_common_gana = max(gana_names.items(), key=lambda x: x[1])[0]
            gana_variety = len(gana_names)

    return {
        "totalSyllables": total_syllables,
        "guruCount": guru_count,
        "laghuCount": laghu_count,
        "guruToLaghuRatio": round(guru_count / laghu_count, 3) if laghu_count > 0 else 0,
        "guruPercentage": round(guru_count / total_syllables * 100, 2),
        "laghuPercentage": round(laghu_count / total_syllables * 100, 2),
        "mostCommonGana": most_common_gana,
        "ganaVariety": gana_variety
    }

def generate_comprehensive_json(text, output_file=None, skip_gana_combinations=False):
    """
    Generates comprehensive JSON analysis for Telugu text input.

    Args:
        text (str): Telugu text (letter, word, sentence, or paragraph)
        output_file (str, optional): Path to save JSON file. If None, returns JSON string.
        skip_gana_combinations (bool, optional): If True, skips Gana combination analysis to reduce processing time.
    Returns:
        dict or str: Complete analysis as dictionary or JSON string
    """
    import time
    start_time = time.time()
    # Sanitize input
    sanitized = re.sub(r'[^\u0C00-\u0C7F\s\u0C01\u200B\n]+', '', text)

    # Generate unique hash
    text_hash = simple_hash(sanitized)

    # Basic input statistics
    char_count = len(sanitized)
    word_count = len([w for w in sanitized.split() if w.strip()])
    sentence_count = len([s for s in re.split(r'[।॥\.\?\!]', sanitized) if s.strip()])
    paragraph_count = len([p for p in sanitized.split('\n') if p.strip()])

    # Detect invalid characters
    removed_chars = set(text) - set(sanitized) - {'\r'}

    # Perform core analysis
    analysis = analyze_telugu_word(sanitized)
    gana_markers = akshara_ganavibhajana(analysis["aksharaluList"])

    # Add positional information to aksharalu
    aksharalu_with_positions = []
    position = 0
    aksharam_positions = {}

    for aksharam in analysis["aksharaluList"]:
        if aksharam not in ignorable_chars:
            if aksharam not in aksharam_positions:
                aksharam_positions[aksharam] = []
            aksharam_positions[aksharam].append(position)
            position += 1

    for ak_data in analysis["aksharalu"]:
        aksharam = ak_data["aksharam"]
        aksharalu_with_positions.append({
            "aksharam": aksharam,
            "categories": ak_data["tags"],
            "count": ak_data["count"],
            "positions": aksharam_positions.get(aksharam, [])
        })

    # Gana marker details with aksharalu
    gana_marker_details = []
    pure_aksharalu = [ak for ak in analysis["aksharaluList"] if ak not in ignorable_chars]

    for idx, aksharam in enumerate(pure_aksharalu):
        if idx < len(gana_markers):
            marker = gana_markers[idx] if gana_markers[idx] else ""
            if marker:
                gana_marker_details.append({
                    "aksharam": aksharam,
                    "marker": marker,
                    "position": idx
                })

    # Gana combinations analysis
    gana_combinations_list = []
    pure_ganas = [m for m in gana_markers if m]
    combinations_limited = False

    if not skip_gana_combinations:
        gana_analyzer = GanaAnalyzer(GANA_DEFINITIONS)
        gana_combinations_list = []
        combinations_limited = False

        if pure_ganas:
            combinations = gana_analyzer.find_sequential_combinations(pure_ganas)

            # Limit output to prevent huge JSON files
            MAX_COMBINATIONS = 50
            if len(combinations) > MAX_COMBINATIONS:
                combinations = combinations[:MAX_COMBINATIONS]
                combinations_limited = True

            for combo in combinations:
                mapped = map_syllables_to_partition(combo, pure_aksharalu)
                gana_combinations_list.append(mapped)

    # Calculate statistics
    linguistic_stats = calculate_linguistic_statistics(analysis)
    vargam_dist = calculate_vargam_distribution(analysis)
    articulation_dist = calculate_articulation_distribution(analysis)
    prosody_stats = calculate_prosody_statistics(gana_markers, gana_combinations_list)

    # Generate summary
    dominant_categories = sorted(analysis["categoryCounts"].items(), key=lambda x: x[1], reverse=True)[:3]
    dominant_cat_names = [cat[0] for cat in dominant_categories]

    linguistic_profile = f"Text with {linguistic_stats['totalAksharas']} aksharas, {linguistic_stats['vowelCount']} vowels, {linguistic_stats['consonantCount']} consonants"
    if linguistic_stats['conjunctCount'] > 0:
        linguistic_profile += f", {linguistic_stats['conjunctCount']} conjuncts"

    prosodic_profile = f"Prosodic pattern: {prosody_stats['guruPercentage']:.1f}% Guru, {prosody_stats['laghuPercentage']:.1f}% Laghu"
    if prosody_stats['mostCommonGana']:
        prosodic_profile += f", dominant Gana: {prosody_stats['mostCommonGana']}"

    # Get timestamp
    if HAS_PYTZ:
        timestamp = datetime.datetime.now(pytz.timezone('Asia/Kolkata')).isoformat()
    else:
        timestamp = datetime.datetime.now().isoformat()

    # Build comprehensive JSON structure
    result = {
        "metadata": {
            "schemaVersion": "1.0.0",
            "analysisTimestamp": timestamp,
            "analyzerVersion": "0.0.7a+",
            "inputHash": text_hash,
            "processingTimeMs": round((time.time() - start_time) * 1000, 2)
        },

        "input": {
            "rawText": text,
            "sanitizedText": sanitized,
            "characterCount": char_count,
            "wordCount": word_count,
            "sentenceCount": sentence_count,
            "paragraphCount": paragraph_count,
            "language": "Telugu",
            "scriptValidation": {
                "isValid": len(removed_chars) == 0,
                "invalidCharacters": sorted(list(removed_chars)),
                "warnings": ["Non-Telugu characters removed"] if removed_chars else []
            }
        },

        "linguistic": {
            "aksharalu": aksharalu_with_positions,
            "aksharaluList": analysis["aksharaluList"],
            "categoryCounts": analysis["categoryCounts"],
            "statistics": linguistic_stats,
            "vargamDistribution": vargam_dist,
            "articulationDistribution": articulation_dist
        },

        "prosody": {
            "ganaSequence": pure_ganas,
            "ganaMarkers": gana_marker_details,
            "ganaCombinations": {
                "count": len(gana_combinations_list),
                "combinations": gana_combinations_list,
                "limitedOutput": combinations_limited,
                "maxCombinationsShown": 50 if combinations_limited else len(gana_combinations_list)
            },
            "statistics": prosody_stats
        },

        "summary": {
            "linguisticProfile": linguistic_profile,
            "prosodicProfile": prosodic_profile,
            "dominantCategories": dominant_cat_names
        }
    }

    # Output handling
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        return f"JSON analysis saved to {output_file}"
    else:
        return result

###############################################################################
# 4) ANALYSIS WRAPPER FUNCTIONS (v0.0.7a)
###############################################################################

def analyze_telugu_word(word):
    """
    REFACTORED: Returns a structured dict matching the v0.0.7a JS version.
    """
    sanitized = re.sub(r'[^\u0C00-\u0C7F\s\u0C01\u200B]+', '', word)
    aksharalu_list = split_aksharalu(sanitized)
    analysis = {}
    category_counts = {}
    all_tags = set()

    processed_aksharalu = []

    for aksharam in aksharalu_list:
        if aksharam in ignorable_chars:
            continue

        tags = categorize_aksharam(aksharam)
        key = aksharam

        if key not in analysis:
            analysis[key] = {"tags": tags, "count": 0}
        analysis[key]["count"] += 1

    for key, info in analysis.items():
        processed_aksharalu.append({"aksharam": key, "tags": info["tags"], "count": info["count"]})
        for cat in info["tags"]:
            category_counts[cat] = category_counts.get(cat, 0) + info["count"]
            all_tags.add(cat)

    return {
        "word": sanitized,
        "uniqueId": simple_hash(sanitized),
        "aksharalu": processed_aksharalu,
        "aksharaluList": aksharalu_list,
        "categoryCounts": category_counts,
        "tags": all_tags
    }

def compare_telugu_words(word1, word2):
    """
    REFACTORED: Performs all comparisons from v0.0.7a JS and returns
    the new, comprehensive data structure.
    """
    analysis1 = analyze_telugu_word(word1)
    analysis2 = analyze_telugu_word(word2)

    tags1 = analysis1["tags"]
    tags2 = analysis2["tags"]

    common_tags = tags1.intersection(tags2)
    all_tags_union = tags1.union(tags2)
    unique_to_word1 = tags1.difference(tags2)
    unique_to_word2 = tags2.difference(tags1)

    jaccard_similarity = len(common_tags) / len(all_tags_union) if len(all_tags_union) > 0 else 0.0
    jaccard_distance = 1.0 - jaccard_similarity

    gana_markers1 = akshara_ganavibhajana(analysis1["aksharaluList"])
    gana_markers2 = akshara_ganavibhajana(analysis2["aksharaluList"])
    gana_jaccard = calculate_gana_jaccard(gana_markers1, gana_markers2)

    pure_gana1 = [m for m in gana_markers1 if m]
    pure_gana2 = [m for m in gana_markers2 if m]
    lcs_result = find_longest_common_substring(pure_gana1, pure_gana2)

    # Add full gana_markers list to analysis objects
    analysis1["ganaMarkers"] = gana_markers1
    analysis2["ganaMarkers"] = gana_markers2

    return {
        "word1Analysis": analysis1,
        "word2Analysis": analysis2,
        "comparison": {
            "commonTags": sorted(list(common_tags)),
            "uniqueToWord1": sorted(list(unique_to_word1)),
            "uniqueToWord2": sorted(list(unique_to_word2)),
            "jaccardSimilarity": jaccard_similarity,
            "jaccardDistance": jaccard_distance,
            "ganaJaccard": gana_jaccard,
            "lcs": lcs_result
        }
    }

###############################################################################
# 5) EXAMPLE TESTING (v0.0.7a+)
###############################################################################
if __name__ == "__main__":
    if HAS_PYTZ:
        timestamp = datetime.datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%Y-%m-%d %H:%M:%S %Z')
    else:
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    print(f"Aksharanusarika v0.0.7a (Python Port)")
    print(f"Analysis performed at: {timestamp}")
    print(f"Location: Hyderabad, Telangana, India\n")
    print("="*60 + "\n")

    # --- Test Case 1: Comprehensive Single Phrase Prosody Analysis ---
    print("--- 1. COMPREHENSIVE PROSODY (GANA) ANALYSIS ---")
    test_phrases = [
        ("అమల", "I I I", "All Laghus"),
        ("రాముడు", "U I I", "Deergham (long vowel)"),
        ("అమ్మ", "U I", "Before dvithvaksharam (doubled)"),
        ("సత్యము", "U I I", "Before samyukthaksharam (conjunct)"),
        ("గౌరవం", "U I U", "Contains 'ఔ' and Sunna"),
        ("సైనికుడు", "U I I I", "Contains 'ఐ'"),
        ("సందడి", "U I I", "Contains Sunna (anusvaram)"),
        ("దుఃఖము", "U I I", "Contains Visarga and before conjunct"),
        ("పూసెన్", "U U", "Ends with Pollu (halant)"),
        ("కృషి", "I I", "'ఋ' vowel is always Laghu"),
    ]

    all_passed_prosody = True
    for phrase, expected, reason in test_phrases:
        analysis = analyze_telugu_word(phrase)
        ganas = akshara_ganavibhajana(analysis["aksharaluList"])
        pure_ganas = [g for g in ganas if g]

        expected_clean = expected.replace(' ', '')
        calculated_clean = "".join(pure_ganas)

        status = "PASS" if calculated_clean == expected_clean else "FAIL"
        if status == "FAIL":
            all_passed_prosody = False
            print(f"  [{status}] \"{phrase}\" ({reason})")
            print(f"     Expected: {expected}, Got: {' '.join(pure_ganas)}")
        else:
            print(f"  [{status}] \"{phrase}\" = {' '.join(pure_ganas)} ({reason})")

    print(f"\nOverall Prosody Test Status: {'ALL PASSED' if all_passed_prosody else 'SOME FAILED'}")
    print("\n" + "="*60)
