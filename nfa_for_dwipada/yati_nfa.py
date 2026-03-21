# -*- coding: utf-8 -*-
"""
Yati NFA for Telugu Dwipada.
============================

Validates the Yati (యతి) alliteration constraint in Telugu Dwipada poetry.
The 1st syllable of **gana 1** must phonetically match the 1st syllable of
**gana 3** within each line, under Yati Maitri group equivalence.

This is one of three parallel NFAs in the constrained-decoding pipeline:

    Raw text
       |
       v
    [SyllableAssembler FST]   -- Stage 1: Unicode chars -> syllables
       |
       v
    [GanaMarker FST]          -- Stage 2: syllables -> U/I markers
       |
       v
    [Position Tracker]        -- routes signals to 3 parallel NFAs
       |            |              |
       v            v              v
    [Gana NFA]   [Prasa NFA]   [Yati NFA]   <-- THIS FILE

-------------------------------------------------------------------------------
THE YATI RULE
-------------------------------------------------------------------------------

In each line of a Dwipada, the first letter of gana 1's first syllable must
match the first letter of gana 3's first syllable. "Match" is defined by a
cascade of four yati types, tried in priority order:

    1. Exact match        — same letter (e.g., క ↔ క)
    2. Vyanjana Yati      — same Yati Maitri group (e.g., క ↔ గ, both Velars)
    3. Svara Yati         — same vowel family (e.g., కా ↔ అ, both అ-family)
    4. Samyukta Yati      — any consonant in a conjunct matches (e.g., ప్ర ↔ ర)
    5. Bindu Yati         — anusvara maps to varga nasal (e.g., కం ↔ ఙ)

If any check passes, yati is satisfied (alive). Otherwise it fails (dead).

-------------------------------------------------------------------------------
THREE-PHASE OPERATION (per line)
-------------------------------------------------------------------------------

    Phase       Position              Action
    ─────       ────────              ──────
    RECORDING   Gana 1, 1st syllable  Store maitri info of the aksharam
    PASS        Gana 2                No signal received (Position Tracker skips)
    CHECKING    Gana 3, 1st syllable  Verify against stored info

-------------------------------------------------------------------------------
USAGE
-------------------------------------------------------------------------------

    from yati_nfa import YatiNFA, format_yati_result_str

    nfa = YatiNFA()
    results = nfa.process([
        ("కా", "గు"),   # line 1: gana1 first syllable, gana3 first syllable
        ("సా", "సి"),   # line 2
    ])
    # results[0]["match"] == True   (క and గ in same maitri group: Velars)
    # results[1]["match"] == True   (స and స are exact match)

    # With trace for debugging:
    results, trace = nfa.process_with_trace([("క", "గ")])

"""

###############################################################################
# 1) YATI CONSTANTS
###############################################################################

# -- Telugu character sets -------------------------------------------------- #

TELUGU_CONSONANTS = {
    "క", "ఖ", "గ", "ఘ", "ఙ",
    "చ", "ఛ", "జ", "ఝ", "ఞ",
    "ట", "ఠ", "డ", "ఢ", "ణ",
    "త", "థ", "ద", "ధ", "న",
    "ప", "ఫ", "బ", "భ", "మ",
    "య", "ర", "ల", "వ",
    "శ", "ష", "స", "హ",
    "ళ", "ఱ",
}

INDEPENDENT_VOWELS = {
    "అ", "ఆ", "ఇ", "ఈ", "ఉ", "ఊ", "ఋ", "ౠ",
    "ఎ", "ఏ", "ఐ", "ఒ", "ఓ", "ఔ",
}

# Dependent vowel marks (matras) → independent vowel forms
DEPENDENT_TO_INDEPENDENT = {
    "ా": "ఆ", "ి": "ఇ", "ీ": "ఈ", "ు": "ఉ", "ూ": "ఊ",
    "ృ": "ఋ", "ౄ": "ౠ",
    "ె": "ఎ", "ే": "ఏ", "ై": "ఐ",
    "ొ": "ఒ", "ో": "ఓ", "ౌ": "ఔ",
}

DEPENDENT_VOWELS = set(DEPENDENT_TO_INDEPENDENT.keys())

HALANT = "్"       # Virama — joins consonants into conjuncts
ANUSVARA = "ం"     # Nasalization mark
VISARGA = "ః"      # Aspiration mark

# -- Yati Maitri Groups (యతి మైత్రి వర్గములు) ------------------------------ #
# 11 phonetic equivalence classes. Letters in the same group satisfy Yati.
# Each group is annotated with its phonetic basis.

YATI_MAITRI_GROUPS = [
    {"అ", "ఆ", "ఐ", "ఔ", "హ", "య", "అం", "అః"},  # Group 0: Open vowels + Glides
    {"ఇ", "ఈ", "ఎ", "ఏ", "ఋ"},                      # Group 1: Front vowels
    {"ఉ", "ఊ", "ఒ", "ఓ"},                             # Group 2: Back vowels
    {"క", "ఖ", "గ", "ఘ", "క్ష"},                     # Group 3: Velars (కంఠ్యము)
    {"చ", "ఛ", "జ", "ఝ", "శ", "ష", "స"},             # Group 4: Palatals + Sibilants (తాలవ్యము)
    {"ట", "ఠ", "డ", "ఢ"},                              # Group 5: Retroflexes (మూర్ధన్యము)
    {"త", "థ", "ద", "ధ"},                              # Group 6: Dentals (దంత్యము)
    {"ప", "ఫ", "బ", "భ", "వ"},                        # Group 7: Labials (ఓష్ఠ్యము)
    {"ర", "ల", "ఱ", "ళ"},                              # Group 8: Liquids
    {"న", "ణ"},                                         # Group 9: Nasals
    {"మ", "పు", "ఫు", "బు", "భు", "ము"},              # Group 10: Labial nasals
]

# Human-readable names for each maitri group (used in diagnostics)
MAITRI_GROUP_NAMES = [
    "Open vowels + Glides",
    "Front vowels",
    "Back vowels",
    "Velars (కంఠ్యము)",
    "Palatals + Sibilants (తాలవ్యము)",
    "Retroflexes (మూర్ధన్యము)",
    "Dentals (దంత్యము)",
    "Labials (ఓష్ఠ్యము)",
    "Liquids",
    "Nasals",
    "Labial nasals",
]

# -- Svara Yati Groups (స్వర యతి) ------------------------------------------ #
# Vowel family harmony — vowels in the same group satisfy Yati
# regardless of consonants.

SVARA_YATI_GROUPS = [
    {"అ", "ఆ", "ఐ", "ఔ"},             # Group 0: అ-family (open)
    {"ఇ", "ఈ", "ఎ", "ఏ", "ఋ", "ౠ"},  # Group 1: ఇ-family (front)
    {"ఉ", "ఊ", "ఒ", "ఓ"},             # Group 2: ఉ-family (back)
]

SVARA_GROUP_NAMES = [
    "అ-family (open)",
    "ఇ-family (front)",
    "ఉ-family (back)",
]

# -- Bindu Yati (బిందు యతి) ------------------------------------------------ #
# When a syllable has anusvara (ం), it maps to the nasal of its consonant's
# varga. E.g., "కం" → velar nasal "ఙ".

VARGA_NASALS = {
    "క": "ఙ", "ఖ": "ఙ", "గ": "ఙ", "ఘ": "ఙ",   # Velars → ఙ
    "చ": "ఞ", "ఛ": "ఞ", "జ": "ఞ", "ఝ": "ఞ",   # Palatals → ఞ
    "ట": "ణ", "ఠ": "ణ", "డ": "ణ", "ఢ": "ణ",   # Retroflexes → ణ
    "త": "న", "థ": "న", "ద": "న", "ధ": "న",   # Dentals → న
    "ప": "మ", "ఫ": "మ", "బ": "మ", "భ": "మ",   # Labials → మ
}

NASAL_TO_VARGA = {
    "ఙ": {"క", "ఖ", "గ", "ఘ"},   # Velar nasal → velar consonants
    "ఞ": {"చ", "ఛ", "జ", "ఝ"},   # Palatal nasal → palatal consonants
    "ణ": {"ట", "ఠ", "డ", "ఢ"},   # Retroflex nasal → retroflex consonants
    "న": {"త", "థ", "ద", "ధ"},   # Dental nasal → dental consonants
    "మ": {"ప", "ఫ", "బ", "భ"},   # Labial nasal → labial consonants
}

# -- Scoring ---------------------------------------------------------------- #
# Binary: any valid yati type = full match, no match = zero.
# The match_type field records which check succeeded for diagnostics.

MATCH_SCORE = 100.0
NO_MATCH_SCORE = 0.0

# -- NFA Phase constants --------------------------------------------------- #

PHASE_IDLE = "IDLE"           # Waiting for gana 1's first syllable
PHASE_RECORDED = "RECORDED"   # Gana 1 info stored, waiting for gana 3
PHASE_ACCEPTED = "ACCEPTED"   # Yati check passed
PHASE_REJECTED = "REJECTED"   # Yati check failed

# -- Precomputed lookup dicts ----------------------------------------------- #
# Built at module load time for O(1) lookups instead of iterating over groups.

# letter → list of maitri group indices it belongs to
LETTER_TO_MAITRI_GROUP = {}
for _idx, _group in enumerate(YATI_MAITRI_GROUPS):
    for _letter in _group:
        LETTER_TO_MAITRI_GROUP.setdefault(_letter, []).append(_idx)

# independent vowel → svara group index
VOWEL_TO_SVARA_GROUP = {}
for _idx, _group in enumerate(SVARA_YATI_GROUPS):
    for _vowel in _group:
        VOWEL_TO_SVARA_GROUP[_vowel] = _idx


###############################################################################
# 2) PURE HELPER FUNCTIONS
###############################################################################


def _get_first_letter(aksharam):
    """Extract the first character of an aksharam.

    Args:
        aksharam: A Telugu syllable string (e.g., "కా", "ప్ర", "అ").

    Returns:
        The first character, or None if empty.
    """
    if not aksharam:
        return None
    return aksharam[0]


def _get_base_consonant(aksharam):
    """Extract the first consonant from an aksharam, skipping any leading vowel.

    Used for Bindu Yati — identifies the consonant whose varga determines
    the nasal mapping for anusvara.

    Args:
        aksharam: A Telugu syllable string.

    Returns:
        The first consonant character, or None if the aksharam has no consonant.

    Examples:
        >>> _get_base_consonant("కా")
        'క'
        >>> _get_base_consonant("అ")
        None
    """
    if not aksharam:
        return None
    for ch in aksharam:
        if ch in TELUGU_CONSONANTS:
            return ch
    return None


def _get_independent_vowel(aksharam):
    """Extract the vowel of an aksharam as its independent vowel form.

    Dependent vowel signs (ా, ి, ు, etc.) are mapped to their independent
    forms (ఆ, ఇ, ఉ, etc.). If no explicit vowel marker is present, returns
    "అ" (the implicit inherent vowel of Telugu consonants).

    Used for Svara Yati (స్వర యతి) — compares vowel families regardless
    of consonants.

    Args:
        aksharam: A Telugu syllable string.

    Returns:
        The independent vowel form, or None if empty.

    Examples:
        >>> _get_independent_vowel("కా")   # dependent ా → ఆ
        'ఆ'
        >>> _get_independent_vowel("అ")    # already independent
        'అ'
        >>> _get_independent_vowel("క")    # no vowel marker → implicit అ
        'అ'
    """
    if not aksharam:
        return None
    # Check for dependent vowel marks (matras)
    for dv in DEPENDENT_VOWELS:
        if dv in aksharam:
            return DEPENDENT_TO_INDEPENDENT[dv]
    # Check if the first character is an independent vowel
    if aksharam[0] in INDEPENDENT_VOWELS:
        return aksharam[0]
    # No explicit vowel → inherent "అ" (short a)
    return "అ"


def _get_all_consonants(aksharam):
    """Extract all consonants from an aksharam.

    For conjunct aksharalu like "ప్ర", returns all consonant components.
    Used for Samyukta Yati (సంయుక్త యతి) — any consonant in the cluster
    can satisfy yati.

    Args:
        aksharam: A Telugu syllable string.

    Returns:
        List of consonant characters found in the aksharam.

    Examples:
        >>> _get_all_consonants("ప్ర")
        ['ప', 'ర']
        >>> _get_all_consonants("క్షా")
        ['క', 'ష']
        >>> _get_all_consonants("అ")
        []
    """
    return [ch for ch in aksharam if ch in TELUGU_CONSONANTS]


def _analyze_aksharam(aksharam):
    """Fully analyze an aksharam for all yati matching purposes.

    Extracts every piece of information needed by the four yati checks,
    so each check function can work from this pre-computed dict without
    re-parsing the aksharam.

    Args:
        aksharam: A Telugu syllable string.

    Returns:
        Dict with keys:
            - aksharam:          the original string
            - first_letter:      first character (for exact + vyanjana yati)
            - maitri_groups:     list of maitri group indices for first_letter
            - independent_vowel: vowel in independent form (for svara yati)
            - svara_group:       svara group index, or None
            - all_consonants:    list of all consonants (for samyukta yati)
            - has_anusvara:      bool (for bindu yati)
            - base_consonant:    first consonant (for bindu yati varga lookup)
    """
    first_letter = _get_first_letter(aksharam)
    vowel = _get_independent_vowel(aksharam)

    return {
        "aksharam": aksharam or "",
        "first_letter": first_letter,
        "maitri_groups": LETTER_TO_MAITRI_GROUP.get(first_letter, []),
        "independent_vowel": vowel,
        "svara_group": VOWEL_TO_SVARA_GROUP.get(vowel),
        "all_consonants": _get_all_consonants(aksharam or ""),
        "has_anusvara": ANUSVARA in (aksharam or ""),
        "base_consonant": _get_base_consonant(aksharam),
    }


def _has_explicit_vowel(aksharam):
    """Check if an aksharam has an explicit vowel (not just the inherent అ).

    Returns True if the aksharam starts with an independent vowel or
    contains a dependent vowel mark (matra). Returns False for bare
    consonants that carry only the implicit inherent vowel.

    Used by svara yati to avoid matching bare consonant pairs.

    Examples:
        >>> _has_explicit_vowel("కా")   # has matra ా
        True
        >>> _has_explicit_vowel("అ")    # independent vowel
        True
        >>> _has_explicit_vowel("క")    # bare consonant, only inherent అ
        False
    """
    if not aksharam:
        return False
    # Starts with an independent vowel
    if aksharam[0] in INDEPENDENT_VOWELS:
        return True
    # Contains a dependent vowel mark (matra)
    for ch in aksharam:
        if ch in DEPENDENT_VOWELS:
            return True
    return False


# -- Individual yati check functions --------------------------------------- #
# Each returns (matched: bool, extra_info: dict or None).
# They are pure functions operating on pre-analyzed info dicts.


def _check_exact(info1, info2):
    """Check exact letter match.

    The simplest yati form: both aksharalu start with the same letter.

    Returns:
        (True, None) if exact match, (False, None) otherwise.
    """
    if info1["first_letter"] and info1["first_letter"] == info2["first_letter"]:
        return True, None
    return False, None


def _check_vyanjana_maitri(info1, info2):
    """Check Vyanjana Yati (వ్యంజన యతి) — Yati Maitri group match.

    Two letters are vyanjana-yati-compatible if they belong to the same
    Yati Maitri group (e.g., క and గ are both Velars, group 3).

    Returns:
        (True, {"group_index": int}) if matched, (False, None) otherwise.
    """
    groups1 = set(info1["maitri_groups"])
    groups2 = set(info2["maitri_groups"])
    common = groups1 & groups2
    if common:
        group_index = min(common)  # deterministic: pick lowest group
        return True, {"group_index": group_index}
    return False, None


def _check_svara_yati(info1, info2):
    """Check Svara Yati (స్వర యతి) — vowel family harmony.

    Satisfied when both aksharalu have vowels from the same family,
    regardless of their consonants. E.g., "కా" (vowel ఆ) matches
    "అ" (vowel అ) because ఆ and అ are in the అ-family.

    Note: bare consonants carry the inherent vowel "అ", so two bare
    consonants like "క" and "న" will match via the అ-family. This
    matches the analyzer's behavior in check_svara_yati().

    Returns:
        (True, {"svara_group": int}) if matched, (False, None) otherwise.
    """
    sg1 = info1["svara_group"]
    sg2 = info2["svara_group"]
    if sg1 is not None and sg2 is not None and sg1 == sg2:
        return True, {"svara_group": sg1}
    return False, None


def _check_samyukta_yati(info1, info2):
    """Check Samyukta Yati (సంయుక్త యతి) — conjunct consonant harmony.

    When either aksharam is a conjunct (e.g., "ప్ర"), any consonant within
    the conjunct cluster can satisfy yati via maitri groups. E.g., "ప్ర"
    can match with ర-group or ప-group letters.

    Returns:
        (True, {"matched_pair": (c1, c2), "group_index": int}) if matched,
        (False, None) otherwise.
    """
    consonants1 = info1["all_consonants"]
    consonants2 = info2["all_consonants"]
    if not consonants1 or not consonants2:
        return False, None

    for c1 in consonants1:
        for c2 in consonants2:
            # Exact consonant match within conjuncts
            if c1 == c2:
                return True, {"matched_pair": (c1, c2), "group_index": None}
            # Maitri group match between consonants
            groups1 = set(LETTER_TO_MAITRI_GROUP.get(c1, []))
            groups2 = set(LETTER_TO_MAITRI_GROUP.get(c2, []))
            common = groups1 & groups2
            if common:
                return True, {"matched_pair": (c1, c2), "group_index": min(common)}
    return False, None


def _check_bindu_yati(info1, info2):
    """Check Bindu Yati (బిందు యతి) — anusvara to varga nasal mapping.

    When a syllable contains anusvara (ం), it can form a valid yati match
    with the nasal consonant of its varga. E.g., "కం" (velar + anusvara)
    can match "ఙ" (velar nasal). The check is symmetric.

    Returns:
        (True, {"nasal": str}) if matched, (False, None) otherwise.
    """
    # Try both directions: info1 has anusvara → check info2, and vice versa
    for a_info, b_info in [(info1, info2), (info2, info1)]:
        if not a_info["has_anusvara"]:
            continue
        base = a_info["base_consonant"]
        if not base or base not in VARGA_NASALS:
            continue

        varga_nasal = VARGA_NASALS[base]
        other_consonant = b_info["base_consonant"]

        # Direct nasal match
        if other_consonant == varga_nasal:
            return True, {"nasal": varga_nasal}

        # Other is the first letter and equals the nasal
        if b_info["first_letter"] == varga_nasal:
            return True, {"nasal": varga_nasal}

        # Reverse: if other is a nasal, check if base is in its varga
        if other_consonant and other_consonant in NASAL_TO_VARGA:
            if base in NASAL_TO_VARGA[other_consonant]:
                return True, {"nasal": other_consonant}

    return False, None


def _cascade_yati_check(info1, info2):
    """Run all yati checks in priority order and return the first match.

    Cascade order:
        1. Exact match        (same letter)
        2. Vyanjana Maitri    (same maitri group)
        3. Svara Yati         (same vowel family)
        4. Samyukta Yati      (conjunct consonant via maitri)
        5. Bindu Yati         (anusvara → varga nasal)

    If any check passes, yati is alive. Otherwise, dead.

    Args:
        info1: Pre-analyzed dict for gana 1's first aksharam.
        info2: Pre-analyzed dict for gana 3's first aksharam.

    Returns:
        Result dict with match, match_type, quality_score, and diagnostics.
    """
    result = {
        "match": False,
        "match_type": "no_match",
        "gana1_aksharam": info1["aksharam"],
        "gana3_aksharam": info2["aksharam"],
        "gana1_first_letter": info1["first_letter"],
        "gana3_first_letter": info2["first_letter"],
        "maitri_group_index": None,
        "quality_score": NO_MATCH_SCORE,
    }

    # Guard: if either aksharam is empty, no match possible
    if not info1["first_letter"] or not info2["first_letter"]:
        return result

    # 1. Exact match
    matched, _ = _check_exact(info1, info2)
    if matched:
        result["match"] = True
        result["match_type"] = "exact"
        result["quality_score"] = MATCH_SCORE
        return result

    # 2. Vyanjana Yati — same maitri group
    matched, extra = _check_vyanjana_maitri(info1, info2)
    if matched:
        result["match"] = True
        result["match_type"] = "vyanjana_maitri"
        result["maitri_group_index"] = extra["group_index"]
        result["quality_score"] = MATCH_SCORE
        return result

    # 3. Svara Yati — same vowel family
    matched, extra = _check_svara_yati(info1, info2)
    if matched:
        result["match"] = True
        result["match_type"] = "svara_yati"
        result["quality_score"] = MATCH_SCORE
        if extra:
            result["maitri_group_index"] = extra["svara_group"]
        return result

    # 4. Samyukta Yati — conjunct consonant via maitri
    matched, extra = _check_samyukta_yati(info1, info2)
    if matched:
        result["match"] = True
        result["match_type"] = "samyukta_yati"
        result["quality_score"] = MATCH_SCORE
        if extra:
            result["maitri_group_index"] = extra["group_index"]
        return result

    # 5. Bindu Yati — anusvara → varga nasal
    matched, extra = _check_bindu_yati(info1, info2)
    if matched:
        result["match"] = True
        result["match_type"] = "bindu_yati"
        result["quality_score"] = MATCH_SCORE
        return result

    # No match
    return result


###############################################################################
# 3) YatiNFA CLASS
###############################################################################


class YatiNFA:
    """
    NFA-based yati validator for Telugu Dwipada lines.

    Validates that the 1st syllable of gana 1 matches the 1st syllable
    of gana 3 under Yati Maitri group equivalence, with cascading
    fallbacks through 4 yati types.

    State machine (per line):

        IDLE ──(RECORD)──► RECORDED ──(CHECK)──► ACCEPTED / REJECTED
          ▲                                              │
          └──────────────────(NEWLINE)────────────────────┘
    """

    def __init__(self):
        self._reset()

    def _reset(self):
        """Clear all state for fresh processing."""
        self.phase = PHASE_IDLE
        self.recorded_info = None     # _analyze_aksharam() result for gana 1
        self.line_results = []        # list of result dicts, one per line

    def _start_new_line(self):
        """Reset phase for a new line, keeping accumulated results."""
        self.phase = PHASE_IDLE
        self.recorded_info = None

    def feed(self, event):
        """Feed a single event to the NFA.

        Event types:
            ("RECORD", aksharam) — store gana 1's first syllable info
            ("CHECK", aksharam)  — verify gana 3's first syllable matches
            ("NEWLINE",)         — line boundary, emit result and reset

        Args:
            event: A tuple whose first element is the event type string.
        """
        event_type = event[0]

        if event_type == "RECORD":
            aksharam = event[1]
            self.recorded_info = _analyze_aksharam(aksharam)
            self.phase = PHASE_RECORDED

        elif event_type == "CHECK":
            aksharam = event[1]
            if self.recorded_info is None:
                # No RECORD received — treat as rejected
                self.phase = PHASE_REJECTED
                self.line_results.append({
                    "match": False,
                    "match_type": "no_match",
                    "gana1_aksharam": "",
                    "gana3_aksharam": aksharam or "",
                    "gana1_first_letter": None,
                    "gana3_first_letter": _get_first_letter(aksharam),
                    "maitri_group_index": None,
                    "quality_score": NO_MATCH_SCORE,
                })
            else:
                check_info = _analyze_aksharam(aksharam)
                result = _cascade_yati_check(self.recorded_info, check_info)
                self.phase = PHASE_ACCEPTED if result["match"] else PHASE_REJECTED
                self.line_results.append(result)

        elif event_type == "NEWLINE":
            # If we haven't emitted a result for this line yet
            # (e.g., RECORD received but no CHECK), emit incomplete
            if self.phase == PHASE_RECORDED:
                self.line_results.append({
                    "match": False,
                    "match_type": "no_match",
                    "gana1_aksharam": self.recorded_info["aksharam"],
                    "gana3_aksharam": "",
                    "gana1_first_letter": self.recorded_info["first_letter"],
                    "gana3_first_letter": None,
                    "maitri_group_index": None,
                    "quality_score": NO_MATCH_SCORE,
                })
            self._start_new_line()

    def flush(self):
        """Signal end of input. Handle any pending state."""
        if self.phase == PHASE_RECORDED:
            # RECORD without CHECK — incomplete line
            self.line_results.append({
                "match": False,
                "match_type": "no_match",
                "gana1_aksharam": self.recorded_info["aksharam"],
                "gana3_aksharam": "",
                "gana1_first_letter": self.recorded_info["first_letter"],
                "gana3_first_letter": None,
                "maitri_group_index": None,
                "quality_score": NO_MATCH_SCORE,
            })
        self.phase = PHASE_IDLE
        self.recorded_info = None

    def process(self, pairs):
        """Process a list of (gana1_first, gana3_first) aksharam pairs.

        Each pair represents one line of the Dwipada. The NFA records
        gana 1's first syllable, then checks gana 3's first syllable.

        Args:
            pairs: List of (aksharam1, aksharam3) tuples, one per line.

        Returns:
            List of result dicts (one per line).

        Example:
            >>> nfa = YatiNFA()
            >>> results = nfa.process([("క", "గ"), ("స", "స")])
            >>> results[0]["match"]
            True
        """
        self._reset()
        for gana1_first, gana3_first in pairs:
            self.feed(("RECORD", gana1_first))
            self.feed(("CHECK", gana3_first))
            self.feed(("NEWLINE",))
        return self.line_results

    def process_with_trace(self, pairs):
        """Process with step-by-step trace for debugging.

        Args:
            pairs: List of (aksharam1, aksharam3) tuples, one per line.

        Returns:
            (results, trace) where trace is a list of per-step dicts
            with keys: step, event, phase_before, phase_after, details.
        """
        self._reset()
        trace = []
        step = 0

        for line_idx, (gana1_first, gana3_first) in enumerate(pairs):
            # RECORD event
            phase_before = self.phase
            self.feed(("RECORD", gana1_first))
            trace.append({
                "step": step,
                "line": line_idx,
                "event": f"RECORD '{gana1_first}'",
                "phase_before": phase_before,
                "phase_after": self.phase,
                "details": f"Stored info for '{gana1_first}', "
                           f"first_letter='{_get_first_letter(gana1_first)}', "
                           f"maitri_groups={LETTER_TO_MAITRI_GROUP.get(_get_first_letter(gana1_first), [])}",
            })
            step += 1

            # CHECK event
            phase_before = self.phase
            results_before = len(self.line_results)
            self.feed(("CHECK", gana3_first))
            result = self.line_results[-1] if len(self.line_results) > results_before else None
            trace.append({
                "step": step,
                "line": line_idx,
                "event": f"CHECK '{gana3_first}'",
                "phase_before": phase_before,
                "phase_after": self.phase,
                "details": f"match={result['match']}, type={result['match_type']}" if result else "no result",
            })
            step += 1

            # NEWLINE event
            phase_before = self.phase
            self.feed(("NEWLINE",))
            trace.append({
                "step": step,
                "line": line_idx,
                "event": "NEWLINE",
                "phase_before": phase_before,
                "phase_after": self.phase,
                "details": "Reset for next line",
            })
            step += 1

        return self.line_results, trace


###############################################################################
# 4) FORMATTING HELPERS
###############################################################################


def format_yati_result_str(result):
    """Format a yati result as a one-line summary string.

    Args:
        result: A result dict from YatiNFA.process().

    Returns:
        Human-readable string like:
            "MATCH (vyanjana_maitri) క ↔ గ [group 3: Velars]"
            "NO MATCH ట ↔ ప"

    Example:
        >>> r = {"match": True, "match_type": "exact", "gana1_first_letter": "స",
        ...      "gana3_first_letter": "స", "maitri_group_index": None, "quality_score": 100.0}
        >>> format_yati_result_str(r)
        "MATCH (exact) స ↔ స"
    """
    letter1 = result.get("gana1_first_letter", "?")
    letter2 = result.get("gana3_first_letter", "?")
    match_type = result.get("match_type", "no_match")

    if result.get("match"):
        group_idx = result.get("maitri_group_index")
        if match_type == "exact":
            return f"MATCH (exact) {letter1} ↔ {letter2}"
        elif group_idx is not None and match_type == "vyanjana_maitri":
            group_name = MAITRI_GROUP_NAMES[group_idx] if group_idx < len(MAITRI_GROUP_NAMES) else "?"
            return f"MATCH (vyanjana_maitri) {letter1} ↔ {letter2} [group {group_idx}: {group_name}]"
        elif match_type == "svara_yati" and group_idx is not None:
            group_name = SVARA_GROUP_NAMES[group_idx] if group_idx < len(SVARA_GROUP_NAMES) else "?"
            return f"MATCH (svara_yati) {letter1} ↔ {letter2} [svara group {group_idx}: {group_name}]"
        else:
            return f"MATCH ({match_type}) {letter1} ↔ {letter2}"
    else:
        return f"NO MATCH {letter1} ↔ {letter2}"


def format_yati_result_detailed(result):
    """Format a yati result with full diagnostic details.

    Shows the cascade of checks performed and which one succeeded (if any),
    along with group membership information.

    Args:
        result: A result dict from YatiNFA.process().

    Returns:
        Multi-line string with detailed diagnostics.
    """
    lines = []
    letter1 = result.get("gana1_first_letter", "?")
    letter2 = result.get("gana3_first_letter", "?")
    aksharam1 = result.get("gana1_aksharam", "?")
    aksharam3 = result.get("gana3_aksharam", "?")
    match_type = result.get("match_type", "no_match")
    score = result.get("quality_score", 0.0)

    lines.append(f"Yati Analysis: '{aksharam1}' (gana 1) ↔ '{aksharam3}' (gana 3)")
    lines.append(f"  First letters: '{letter1}' ↔ '{letter2}'")
    lines.append(f"  Result: {'MATCH' if result.get('match') else 'NO MATCH'} ({match_type})")
    lines.append(f"  Score: {score:.0f}%")

    # Show maitri group membership for both letters
    groups1 = LETTER_TO_MAITRI_GROUP.get(letter1, [])
    groups2 = LETTER_TO_MAITRI_GROUP.get(letter2, [])
    if groups1:
        group_names = [f"{i}: {MAITRI_GROUP_NAMES[i]}" for i in groups1]
        lines.append(f"  '{letter1}' maitri groups: [{', '.join(group_names)}]")
    if groups2:
        group_names = [f"{i}: {MAITRI_GROUP_NAMES[i]}" for i in groups2]
        lines.append(f"  '{letter2}' maitri groups: [{', '.join(group_names)}]")

    # Show matched group details
    group_idx = result.get("maitri_group_index")
    if group_idx is not None:
        if match_type == "vyanjana_maitri":
            members = sorted(YATI_MAITRI_GROUPS[group_idx])
            lines.append(f"  Matched group {group_idx} ({MAITRI_GROUP_NAMES[group_idx]}): {{{', '.join(members)}}}")
        elif match_type == "svara_yati":
            members = sorted(SVARA_YATI_GROUPS[group_idx])
            lines.append(f"  Matched svara group {group_idx} ({SVARA_GROUP_NAMES[group_idx]}): {{{', '.join(members)}}}")

    return "\n".join(lines)


###############################################################################
# 5) INLINE TESTS
###############################################################################


def _print_test_result(test_name, aksharam1, aksharam3, expected_match,
                       expected_type, result, passed):
    """Pretty-print a single test case result.

    Args:
        test_name:      Short description of the test.
        aksharam1:      Gana 1's first aksharam.
        aksharam3:      Gana 3's first aksharam.
        expected_match: Expected match boolean.
        expected_type:  Expected match_type string.
        result:         Actual result dict from YatiNFA.
        passed:         Whether the test passed.
    """
    status = "PASS" if passed else "FAIL"
    icon = "  ✓" if passed else "  ✗"
    print(f"{icon} [{status}] {test_name}")
    print(f"       Input: '{aksharam1}' ↔ '{aksharam3}'")
    print(f"       Expected: match={expected_match}, type={expected_type}")
    print(f"       Got:      match={result['match']}, type={result['match_type']}")
    if not passed:
        print(f"       >>> {format_yati_result_detailed(result)}")
    print()


def run_tests():
    """Run comprehensive test suite for the Yati NFA.

    Tests cover all 4 yati types, edge cases, and multi-line input.

    Returns:
        True if all tests pass, False otherwise.
    """
    nfa = YatiNFA()
    all_passed = True
    total = 0
    passed_count = 0

    # Define test cases: (name, aksharam1, aksharam3, expected_match, expected_type)
    test_cases = [
        # --- Exact match tests ---
        ("Exact: same consonant క↔క",
         "క", "క", True, "exact"),

        ("Exact: same vowel అ↔అ",
         "అ", "అ", True, "exact"),

        ("Exact: same consonant+vowel సా↔సా",
         "సా", "సా", True, "exact"),

        # --- Vyanjana Maitri tests (one per group) ---
        ("Vyanjana Maitri group 0: అ↔హ (open vowels + glides)",
         "అ", "హ", True, "vyanjana_maitri"),

        ("Vyanjana Maitri group 1: ఇ↔ఏ (front vowels)",
         "ఇ", "ఏ", True, "vyanjana_maitri"),

        ("Vyanjana Maitri group 2: ఉ↔ఓ (back vowels)",
         "ఉ", "ఓ", True, "vyanjana_maitri"),

        ("Vyanjana Maitri group 3: క↔గ (velars)",
         "క", "గ", True, "vyanjana_maitri"),

        ("Vyanjana Maitri group 4: చ↔స (palatals + sibilants)",
         "చ", "స", True, "vyanjana_maitri"),

        ("Vyanjana Maitri group 5: ట↔డ (retroflexes)",
         "ట", "డ", True, "vyanjana_maitri"),

        ("Vyanjana Maitri group 6: త↔ధ (dentals)",
         "త", "ధ", True, "vyanjana_maitri"),

        ("Vyanjana Maitri group 7: ప↔భ (labials)",
         "ప", "భ", True, "vyanjana_maitri"),

        ("Vyanjana Maitri group 8: ర↔ల (liquids)",
         "ర", "ల", True, "vyanjana_maitri"),

        ("Vyanjana Maitri group 9: న↔ణ (nasals)",
         "న", "ణ", True, "vyanjana_maitri"),

        # --- Svara Yati tests ---
        ("Svara Yati: కా↔అ (అ-family vowel harmony)",
         "కా", "అ", True, "svara_yati"),

        ("Svara Yati: కి↔ఎ (ఇ-family vowel harmony)",
         "కి", "ఎ", True, "svara_yati"),

        ("Svara Yati: కు↔ఒ (ఉ-family vowel harmony)",
         "కు", "ఒ", True, "svara_yati"),

        # --- Samyukta Yati tests ---
        # Note: ప్ర and ర are bare consonants → both have inherent అ →
        # svara yati matches first in the cascade. This matches analyzer behavior.
        ("Svara before Samyukta: ప్ర↔ర (bare consonants → svara fires first)",
         "ప్ర", "ర", True, "svara_yati"),

        # To isolate samyukta yati, use aksharalu with vowels in different
        # svara groups so svara yati won't match:
        # క్షి (vowel ఇ, front) ↔ షు (vowel ఉ, back) — different svara groups.
        # క్ష contains ష, which matches షు's first letter ష → samyukta.
        ("Samyukta Yati: క్షి↔షు (conjunct క్ష has ష, different svara groups)",
         "క్షి", "షు", True, "samyukta_yati"),

        # --- Bindu Yati tests ---
        # కం and ఙ are bare consonants → both have inherent అ →
        # svara yati matches first. This matches analyzer cascade order.
        ("Svara before Bindu: కం↔ఙ (bare consonants → svara fires first)",
         "కం", "ఙ", True, "svara_yati"),

        # To isolate bindu yati, use aksharalu with different svara groups:
        # కిం (vowel ఇ, front) ↔ ఙు (vowel ఉ, back) — different svara groups.
        # కిం has anusvara + base క (velar) → velar nasal ఙ → matches ఙు.
        ("Bindu Yati: కిం↔ఙు (anusvara velar→nasal, different svara groups)",
         "కిం", "ఙు", True, "bindu_yati"),

        # --- No match tests ---
        # Bare consonants in different maitri groups still match via
        # svara yati (both carry inherent అ vowel → అ-family).
        # To get a true no-match, use aksharalu with vowels in different
        # svara groups AND consonants in different maitri groups.
        ("No match: కి↔పు (front vs back vowels, different vargas)",
         "కి", "పు", False, "no_match"),

        ("No match: టూ↔దీ (back vs front vowels, different vargas)",
         "టూ", "దీ", False, "no_match"),

        ("No match: గే↔డొ (front vs back vowels, different vargas)",
         "గే", "డొ", False, "no_match"),

        # --- Edge cases ---
        ("Edge: empty aksharam",
         "", "క", False, "no_match"),

        ("Edge: both empty",
         "", "", False, "no_match"),
    ]

    print("=" * 70)
    print("YATI NFA TEST SUITE")
    print("=" * 70)
    print()

    # Run single-pair tests
    for name, a1, a3, exp_match, exp_type in test_cases:
        total += 1
        results = nfa.process([(a1, a3)])
        result = results[0]
        passed = (result["match"] == exp_match and result["match_type"] == exp_type)
        if passed:
            passed_count += 1
        else:
            all_passed = False
        _print_test_result(name, a1, a3, exp_match, exp_type, result, passed)

    # --- Multi-line test ---
    total += 1
    print("-" * 70)
    print("Multi-line test: 2 lines, first matches (vyanjana), second exact")
    print("-" * 70)
    results = nfa.process([("క", "గ"), ("స", "స")])
    multi_passed = (
        results[0]["match"] is True
        and results[0]["match_type"] == "vyanjana_maitri"
        and results[1]["match"] is True
        and results[1]["match_type"] == "exact"
    )
    if multi_passed:
        passed_count += 1
        print(f"  ✓ [PASS] Line 1: {format_yati_result_str(results[0])}")
        print(f"  ✓ [PASS] Line 2: {format_yati_result_str(results[1])}")
    else:
        all_passed = False
        print(f"  ✗ [FAIL] Line 1: {format_yati_result_str(results[0])}")
        print(f"  ✗ [FAIL] Line 2: {format_yati_result_str(results[1])}")
    print()

    # --- Multi-line test: mixed pass/fail ---
    total += 1
    print("-" * 70)
    print("Multi-line test: line 1 matches, line 2 fails")
    print("-" * 70)
    # Use vowels in different svara groups to ensure a true no-match
    results = nfa.process([("క", "గ"), ("కి", "పు")])
    mixed_passed = (
        results[0]["match"] is True
        and results[1]["match"] is False
    )
    if mixed_passed:
        passed_count += 1
        print(f"  ✓ [PASS] Line 1: {format_yati_result_str(results[0])}")
        print(f"  ✓ [PASS] Line 2: {format_yati_result_str(results[1])}")
    else:
        all_passed = False
        print(f"  ✗ [FAIL] Line 1: {format_yati_result_str(results[0])}")
        print(f"  ✗ [FAIL] Line 2: {format_yati_result_str(results[1])}")
    print()

    # --- Trace test ---
    total += 1
    print("-" * 70)
    print("Trace test: process_with_trace output")
    print("-" * 70)
    results, trace = nfa.process_with_trace([("క", "గ")])
    trace_passed = (
        len(trace) == 3  # RECORD, CHECK, NEWLINE
        and results[0]["match"] is True
    )
    if trace_passed:
        passed_count += 1
        print(f"  ✓ [PASS] Trace has {len(trace)} steps, result matches")
        for t in trace:
            print(f"       Step {t['step']}: {t['event']} | {t['phase_before']} → {t['phase_after']} | {t['details']}")
    else:
        all_passed = False
        print(f"  ✗ [FAIL] Trace has {len(trace)} steps (expected 3), match={results[0]['match']}")
    print()

    # --- Summary ---
    print("=" * 70)
    print(f"RESULTS: {passed_count}/{total} tests passed")
    if all_passed:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    print("=" * 70)

    return all_passed


if __name__ == "__main__":
    ok = run_tests()
    raise SystemExit(0 if ok else 1)
