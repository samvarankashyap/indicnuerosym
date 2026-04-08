#!/usr/bin/env python3
"""Tests for Kannada-Telugu transliteration converter.

Coverage:
- 100+ Telugu -> Kannada test cases (words, phrases, verses)
- 100+ Kannada -> Telugu test cases (words, phrases, verses)
- 100+ Round-trip tests (T->K->T and K->T->K)
- Mapping completeness and correctness checks
"""

import sys
from pathlib import Path

import pytest

# Ensure the converter module is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from kannada_telugu_converter_paper_based import EnhancedKannadaTeluguConverter


converter = EnhancedKannadaTeluguConverter()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def t2k(text: str) -> str:
    return converter.telugu_to_kannada(text)


def k2t(text: str) -> str:
    return converter.kannada_to_telugu(text)


# ===========================================================================
#  SECTION 1: TELUGU -> KANNADA  (100+ test cases)
# ===========================================================================


# ---------------------------------------------------------------------------
# 1a. Independent vowels (16 cases)
# ---------------------------------------------------------------------------

VOWEL_PAIRS_T2K = [
    ('అ', 'ಅ'),   ('ఆ', 'ಆ'),   ('ఇ', 'ಇ'),   ('ఈ', 'ಈ'),
    ('ఉ', 'ಉ'),   ('ఊ', 'ಊ'),   ('ఋ', 'ಋ'),   ('ౠ', 'ೠ'),
    ('ఎ', 'ಎ'),   ('ఏ', 'ಏ'),   ('ఐ', 'ಐ'),
    ('ఒ', 'ಒ'),   ('ఓ', 'ಓ'),   ('ఔ', 'ಔ'),
    ('ఌ', 'ಌ'),   ('ౡ', 'ೡ'),
]


@pytest.mark.parametrize("telugu,kannada", VOWEL_PAIRS_T2K,
                         ids=[f"vowel_{chr(ord(t))}" for t, _ in VOWEL_PAIRS_T2K])
def test_vowel_t2k(telugu, kannada):
    assert t2k(telugu) == kannada


# ---------------------------------------------------------------------------
# 1b. Vowel signs / matras (13 cases)
# ---------------------------------------------------------------------------

VOWEL_SIGN_PAIRS_T2K = [
    ('ా', 'ಾ'),   ('ి', 'ಿ'),   ('ీ', 'ೀ'),   ('ు', 'ು'),
    ('ూ', 'ೂ'),   ('ృ', 'ೃ'),   ('ౄ', 'ೄ'),
    ('ె', 'ೆ'),   ('ే', 'ೇ'),   ('ై', 'ೈ'),
    ('ొ', 'ೊ'),   ('ో', 'ೋ'),   ('ౌ', 'ೌ'),
]


@pytest.mark.parametrize("telugu,kannada", VOWEL_SIGN_PAIRS_T2K,
                         ids=[f"sign_U+{ord(t):04X}" for t, _ in VOWEL_SIGN_PAIRS_T2K])
def test_vowel_sign_t2k(telugu, kannada):
    result = t2k(telugu)
    assert result == kannada, (
        f"U+{ord(telugu):04X} -> got U+{ord(result):04X}, expected U+{ord(kannada):04X}"
    )


# ---------------------------------------------------------------------------
# 1c. Consonants (34 cases)
# ---------------------------------------------------------------------------

CONSONANT_PAIRS_T2K = [
    # High resemblance (99%)
    ('గ', 'ಗ'), ('ఠ', 'ಠ'), ('డ', 'ಡ'), ('ద', 'ದ'), ('న', 'ನ'),
    ('ర', 'ರ'), ('స', 'ಸ'), ('ళ', 'ಳ'), ('ఢ', 'ಢ'), ('థ', 'ಥ'),
    ('ధ', 'ಧ'), ('భ', 'ಭ'), ('జ', 'ಜ'), ('ణ', 'ಣ'), ('బ', 'ಬ'),
    ('ల', 'ಲ'), ('ఱ', 'ಱ'),
    # Medium resemblance
    ('ఖ', 'ಖ'), ('ఙ', 'ಙ'), ('ప', 'ಪ'), ('ఝ', 'ಝ'), ('య', 'ಯ'),
    # Low resemblance
    ('చ', 'ಚ'), ('శ', 'ಶ'), ('త', 'ತ'), ('ష', 'ಷ'), ('హ', 'ಹ'),
    # Other
    ('క', 'ಕ'), ('ఛ', 'ಛ'), ('ట', 'ಟ'), ('ఫ', 'ಫ'), ('మ', 'ಮ'),
    ('వ', 'ವ'), ('ఞ', 'ಞ'), ('ఘ', 'ಘ'),
]


@pytest.mark.parametrize("telugu,kannada", CONSONANT_PAIRS_T2K,
                         ids=[f"cons_{t}" for t, _ in CONSONANT_PAIRS_T2K])
def test_consonant_t2k(telugu, kannada):
    assert t2k(telugu) == kannada


# ---------------------------------------------------------------------------
# 1d. Special symbols (4 cases)
# ---------------------------------------------------------------------------

SPECIAL_PAIRS = [
    ('ం', 'ಂ'),   ('ః', 'ಃ'),   ('్', '್'),   ('ఁ', 'ಁ'),
]


@pytest.mark.parametrize("telugu,kannada", SPECIAL_PAIRS)
def test_special_t2k(telugu, kannada):
    assert t2k(telugu) == kannada


# ---------------------------------------------------------------------------
# 1e. Numbers (10 cases)
# ---------------------------------------------------------------------------

NUMBER_PAIRS = [
    ('౦', '೦'), ('౧', '೧'), ('౨', '೨'), ('౩', '೩'), ('౪', '೪'),
    ('౫', '೫'), ('౬', '೬'), ('౭', '೭'), ('౮', '೮'), ('౯', '೯'),
]


@pytest.mark.parametrize("telugu,kannada", NUMBER_PAIRS)
def test_number_t2k(telugu, kannada):
    assert t2k(telugu) == kannada


# ---------------------------------------------------------------------------
# 1f. Words — Telugu to Kannada (50+ cases)
# ---------------------------------------------------------------------------

WORD_PAIRS_T2K = [
    # Common words
    ('తెలుగు', 'ತೆಲುಗು'),
    ('కన్నడ', 'ಕನ್ನಡ'),
    ('నమస్కారం', 'ನಮಸ್ಕಾರಂ'),
    ('భారతదేశం', 'ಭಾರತದೇಶಂ'),
    ('ద్విపద', 'ದ್ವಿಪದ'),
    ('అమ్మ', 'ಅಮ್ಮ'),
    ('పద్యం', 'ಪದ್ಯಂ'),
    ('హరిహర', 'ಹರಿಹರ'),
    ('బ్రహ్మ', 'ಬ್ರಹ್ಮ'),
    ('గురు', 'ಗುರು'),
    ('విద్యా', 'ವಿದ್ಯಾ'),
    ('పుస్తకం', 'ಪುಸ್ತಕಂ'),
    # Names / proper nouns
    ('రామ', 'ರಾಮ'),
    ('కృష్ణ', 'ಕೃಷ್ಣ'),
    ('శివ', 'ಶಿವ'),
    ('విష్ణు', 'ವಿಷ್ಣು'),
    ('లక్ష్మి', 'ಲಕ್ಷ್ಮಿ'),
    ('సరస్వతి', 'ಸರಸ್ವತಿ'),
    ('గణేశ', 'ಗಣೇಶ'),
    ('హనుమాన్', 'ಹನುಮಾನ್'),
    ('శ్రీనివాసుడు', 'ಶ್ರೀನಿವಾಸುಡು'),
    ('వెంకటేశ్వర', 'ವೆಂಕಟೇಶ್ವರ'),
    ('రామాయణము', 'ರಾಮಾಯಣಮು'),
    ('మహాభారతము', 'ಮಹಾಭಾರತಮು'),
    ('భాగవతము', 'ಭಾಗವತಮು'),
    # Literary terms
    ('కావ్యం', 'ಕಾವ್ಯಂ'),
    ('పురాణము', 'ಪುರಾಣಮು'),
    ('ఆశ్వాసము', 'ಆಶ್ವಾಸಮು'),
    ('అధ్యాయము', 'ಅಧ್ಯಾಯಮು'),
    ('శ్లోకం', 'ಶ್ಲೋಕಂ'),
    ('పదం', 'ಪದಂ'),
    ('వచనం', 'ವಚನಂ'),
    # Geography / places
    ('హైదరాబాద్', 'ಹೈದರಾಬಾದ್'),
    ('బెంగళూరు', 'ಬೆಂಗಳೂರು'),
    ('తిరుపతి', 'ತಿರುಪತಿ'),
    ('విజయవాడ', 'ವಿಜಯವಾಡ'),
    # Words with specific vowel signs (regression)
    ('వెంకట', 'ವೆಂಕಟ'),
    ('నెల', 'ನೆಲ'),
    ('కుమారుడు', 'ಕುಮಾರುಡು'),
    ('సుందరం', 'ಸುಂದರಂ'),
    ('పుస్తకము', 'ಪುಸ್ತಕಮು'),
    # Conjunct-heavy words
    ('సంస్కృతం', 'ಸಂಸ್ಕೃತಂ'),
    ('ప్రపంచం', 'ಪ್ರಪಂಚಂ'),
    ('స్వాతంత్ర్యం', 'ಸ್ವಾತಂತ್ರ್ಯಂ'),
    ('విజ్ఞానం', 'ವಿಜ್ಞಾನಂ'),
    ('క్షేత్రం', 'ಕ್ಷೇತ್ರಂ'),
    ('జ్ఞానం', 'ಜ್ಞಾನಂ'),
    # Words with anusvara / visarga
    ('దుఃఖం', 'ದುಃಖಂ'),
    ('నమః', 'ನಮಃ'),
    ('ఆనందం', 'ಆನಂದಂ'),
    ('శాంతి', 'ಶಾಂತಿ'),
    # Everyday words
    ('నీరు', 'ನೀರು'),
    ('మంచి', 'ಮಂಚಿ'),
    ('చెట్టు', 'ಚೆಟ್ಟು'),
    ('పువ్వు', 'ಪುವ್ವು'),
    ('ఇల్లు', 'ಇಲ್ಲು'),
]


@pytest.mark.parametrize("telugu,kannada", WORD_PAIRS_T2K,
                         ids=[t for t, _ in WORD_PAIRS_T2K])
def test_word_t2k(telugu, kannada):
    result = t2k(telugu)
    assert result == kannada, f"'{telugu}' -> got '{result}', expected '{kannada}'"


# ---------------------------------------------------------------------------
# 1g. Phrases and sentences — Telugu to Kannada (10 cases)
# ---------------------------------------------------------------------------

PHRASE_PAIRS_T2K = [
    ('జయ హింద్', 'ಜಯ ಹಿಂದ್'),
    ('ఓం నమః శివాయ', 'ಓಂ ನಮಃ ಶಿವಾಯ'),
    ('శ్రీ రామ జయ రామ', 'ಶ್ರೀ ರಾಮ ಜಯ ರಾಮ'),
    ('సత్యమేవ జయతే', 'ಸತ್ಯಮೇವ ಜಯತೇ'),
    ('వందే మాతరం', 'ವಂದೇ ಮಾತರಂ'),
    ('భారత మాత కీ జై', 'ಭಾರತ ಮಾತ ಕೀ ಜೈ'),
    ('ధర్మో రక్షతి రక్షితః', 'ಧರ್ಮೋ ರಕ್ಷತಿ ರಕ್ಷಿತಃ'),
    ('సర్వే జనాః సుఖినో భవంతు', 'ಸರ್ವೇ ಜನಾಃ ಸುಖಿನೋ ಭವಂತು'),
    ('అహింసా పరమో ధర్మః', 'ಅಹಿಂಸಾ ಪರಮೋ ಧರ್ಮಃ'),
    ('జ్ఞానం పరమం బలం', 'ಜ್ಞಾನಂ ಪರಮಂ ಬಲಂ'),
]


@pytest.mark.parametrize("telugu,kannada", PHRASE_PAIRS_T2K,
                         ids=[f"phrase_{i}" for i in range(len(PHRASE_PAIRS_T2K))])
def test_phrase_t2k(telugu, kannada):
    assert t2k(telugu) == kannada


# ===========================================================================
#  SECTION 2: KANNADA -> TELUGU  (100+ test cases)
# ===========================================================================


# ---------------------------------------------------------------------------
# 2a. Independent vowels (16 cases)
# ---------------------------------------------------------------------------

VOWEL_PAIRS_K2T = [(k, t) for t, k in VOWEL_PAIRS_T2K]


@pytest.mark.parametrize("kannada,telugu", VOWEL_PAIRS_K2T,
                         ids=[f"vowel_{chr(ord(k))}" for k, _ in VOWEL_PAIRS_K2T])
def test_vowel_k2t(kannada, telugu):
    assert k2t(kannada) == telugu


# ---------------------------------------------------------------------------
# 2b. Vowel signs (13 cases)
# ---------------------------------------------------------------------------

VOWEL_SIGN_PAIRS_K2T = [(k, t) for t, k in VOWEL_SIGN_PAIRS_T2K]


@pytest.mark.parametrize("kannada,telugu", VOWEL_SIGN_PAIRS_K2T,
                         ids=[f"sign_U+{ord(k):04X}" for k, _ in VOWEL_SIGN_PAIRS_K2T])
def test_vowel_sign_k2t(kannada, telugu):
    result = k2t(kannada)
    assert result == telugu, (
        f"U+{ord(kannada):04X} -> got U+{ord(result):04X}, expected U+{ord(telugu):04X}"
    )


# ---------------------------------------------------------------------------
# 2c. Consonants (34 cases)
# ---------------------------------------------------------------------------

CONSONANT_PAIRS_K2T = [(k, t) for t, k in CONSONANT_PAIRS_T2K]


@pytest.mark.parametrize("kannada,telugu", CONSONANT_PAIRS_K2T,
                         ids=[f"cons_{k}" for k, _ in CONSONANT_PAIRS_K2T])
def test_consonant_k2t(kannada, telugu):
    assert k2t(kannada) == telugu


# ---------------------------------------------------------------------------
# 2d. Special symbols (4 cases)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("telugu,kannada", SPECIAL_PAIRS)
def test_special_k2t(telugu, kannada):
    assert k2t(kannada) == telugu


# ---------------------------------------------------------------------------
# 2e. Numbers (10 cases)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("telugu,kannada", NUMBER_PAIRS)
def test_number_k2t(telugu, kannada):
    assert k2t(kannada) == telugu


# ---------------------------------------------------------------------------
# 2f. Words — Kannada to Telugu (50+ cases)
# ---------------------------------------------------------------------------

WORD_PAIRS_K2T = [
    # Common words
    ('ಕನ್ನಡ', 'కన్నడ'),
    ('ತೆಲುಗು', 'తెలుగు'),
    ('ನಮಸ್ಕಾರ', 'నమస్కార'),
    ('ಭಾರತ', 'భారత'),
    ('ರಾಮ', 'రామ'),
    ('ಶ್ರೀ', 'శ్రీ'),
    ('ದ್ವಿಪದ', 'ద్విపద'),
    ('ಅಮ್ಮ', 'అమ్మ'),
    ('ಗುರು', 'గురు'),
    ('ವಿದ್ಯಾ', 'విద్యా'),
    # Names
    ('ಕೃಷ್ಣ', 'కృష్ణ'),
    ('ಶಿವ', 'శివ'),
    ('ವಿಷ್ಣು', 'విష్ణు'),
    ('ಲಕ್ಷ್ಮಿ', 'లక్ష్మి'),
    ('ಸರಸ್ವತಿ', 'సరస్వతి'),
    ('ಗಣೇಶ', 'గణేశ'),
    ('ಹನುಮಾನ್', 'హనుమాన్'),
    ('ಶ್ರೀನಿವಾಸ', 'శ్రీనివాస'),
    ('ವೆಂಕಟೇಶ್ವರ', 'వెంకటేశ్వర'),
    ('ಮಹಾಭಾರತ', 'మహాభారత'),
    # Literary terms
    ('ಕಾವ್ಯ', 'కావ్య'),
    ('ಪುರಾಣ', 'పురాణ'),
    ('ಶ್ಲೋಕ', 'శ్లోక'),
    ('ಪದ', 'పద'),
    ('ವಚನ', 'వచన'),
    # Places
    ('ಬೆಂಗಳೂರು', 'బెంగళూరు'),
    ('ಮೈಸೂರು', 'మైసూరు'),
    ('ಹೈದರಾಬಾದ್', 'హైదరాబాద్'),
    ('ತಿರುಪತಿ', 'తిరుపతి'),
    ('ಮಂಗಳೂರು', 'మంగళూరు'),
    # Everyday words
    ('ನೀರು', 'నీరు'),
    ('ಮಂಚಿ', 'మంచి'),
    ('ಚೆಟ್ಟು', 'చెట్టు'),
    ('ಪುವ್ವು', 'పువ్వు'),
    ('ಇಲ್ಲು', 'ఇల్లు'),
    ('ಮನೆ', 'మనె'),
    ('ಹಣ', 'హణ'),
    ('ಬೆಳಕು', 'బెళకు'),
    ('ಹಸಿರು', 'హసిరు'),
    ('ಕಾಡು', 'కాడు'),
    # Conjunct-heavy
    ('ಸಂಸ್ಕೃತ', 'సంస్కృత'),
    ('ಪ್ರಪಂಚ', 'ప్రపంచ'),
    ('ವಿಜ್ಞಾನ', 'విజ్ఞాన'),
    ('ಜ್ಞಾನ', 'జ్ఞాన'),
    ('ಕ್ಷೇತ್ರ', 'క్షేత్ర'),
    # Words with specific signs
    ('ಸುಂದರ', 'సుందర'),
    ('ಕುಮಾರ', 'కుమార'),
    ('ಪುಸ್ತಕ', 'పుస్తక'),
    ('ವೆಂಕಟ', 'వెంకట'),
    ('ನೆಲ', 'నెల'),
    # Anusvara / visarga
    ('ದುಃಖ', 'దుఃఖ'),
    ('ನಮಃ', 'నమః'),
    ('ಆನಂದ', 'ఆనంద'),
    ('ಶಾಂತಿ', 'శాంతి'),
]


@pytest.mark.parametrize("kannada,telugu", WORD_PAIRS_K2T,
                         ids=[k for k, _ in WORD_PAIRS_K2T])
def test_word_k2t(kannada, telugu):
    result = k2t(kannada)
    assert result == telugu, f"'{kannada}' -> got '{result}', expected '{telugu}'"


# ---------------------------------------------------------------------------
# 2g. Phrases — Kannada to Telugu (10 cases)
# ---------------------------------------------------------------------------

PHRASE_PAIRS_K2T = [(k, t) for t, k in PHRASE_PAIRS_T2K]


@pytest.mark.parametrize("kannada,telugu", PHRASE_PAIRS_K2T,
                         ids=[f"phrase_{i}" for i in range(len(PHRASE_PAIRS_K2T))])
def test_phrase_k2t(kannada, telugu):
    assert k2t(kannada) == telugu


# ===========================================================================
#  SECTION 3: ROUND-TRIP TESTS  (100+ cases)
# ===========================================================================


# ---------------------------------------------------------------------------
# 3a. Round-trip: Telugu -> Kannada -> Telugu (single chars — 79 cases)
# ---------------------------------------------------------------------------

ALL_TELUGU_KEYS = list(converter.telugu_to_kannada_map.keys())


@pytest.mark.parametrize("telugu", ALL_TELUGU_KEYS,
                         ids=[f"rt_char_U+{ord(c):04X}" if len(c) == 1 else f"rt_seq_{c}"
                              for c in ALL_TELUGU_KEYS])
def test_roundtrip_char_t2k2t(telugu):
    """Every mapped Telugu character must survive T->K->T round-trip."""
    kannada = t2k(telugu)
    back = k2t(kannada)
    assert back == telugu, (
        f"Round-trip failed: '{telugu}' -> '{kannada}' -> '{back}'"
    )


# ---------------------------------------------------------------------------
# 3b. Round-trip: Kannada -> Telugu -> Kannada (single chars — 79 cases)
# ---------------------------------------------------------------------------

ALL_KANNADA_KEYS = list(converter.kannada_to_telugu_map.keys())


@pytest.mark.parametrize("kannada", ALL_KANNADA_KEYS,
                         ids=[f"rt_char_U+{ord(c):04X}" if len(c) == 1 else f"rt_seq_{c}"
                              for c in ALL_KANNADA_KEYS])
def test_roundtrip_char_k2t2k(kannada):
    """Every mapped Kannada character must survive K->T->K round-trip."""
    telugu = k2t(kannada)
    back = t2k(telugu)
    assert back == kannada, (
        f"Round-trip failed: '{kannada}' -> '{telugu}' -> '{back}'"
    )


# ---------------------------------------------------------------------------
# 3c. Round-trip: words T->K->T (50+ cases)
# ---------------------------------------------------------------------------

ROUNDTRIP_WORDS_TELUGU = [t for t, _ in WORD_PAIRS_T2K]


@pytest.mark.parametrize("word", ROUNDTRIP_WORDS_TELUGU,
                         ids=ROUNDTRIP_WORDS_TELUGU)
def test_roundtrip_word_t2k2t(word):
    kannada = t2k(word)
    back = k2t(kannada)
    assert back == word, f"'{word}' -> '{kannada}' -> '{back}'"


# ---------------------------------------------------------------------------
# 3d. Round-trip: phrases T->K->T (10 cases)
# ---------------------------------------------------------------------------

ROUNDTRIP_PHRASES = [t for t, _ in PHRASE_PAIRS_T2K]


@pytest.mark.parametrize("phrase", ROUNDTRIP_PHRASES,
                         ids=[f"rt_phrase_{i}" for i in range(len(ROUNDTRIP_PHRASES))])
def test_roundtrip_phrase_t2k2t(phrase):
    kannada = t2k(phrase)
    back = k2t(kannada)
    assert back == phrase


# ---------------------------------------------------------------------------
# 3e. Round-trip: Kannada words K->T->K (50+ cases)
# ---------------------------------------------------------------------------

ROUNDTRIP_WORDS_KANNADA = [k for k, _ in WORD_PAIRS_K2T]


@pytest.mark.parametrize("word", ROUNDTRIP_WORDS_KANNADA,
                         ids=ROUNDTRIP_WORDS_KANNADA)
def test_roundtrip_word_k2t2k(word):
    telugu = k2t(word)
    back = t2k(telugu)
    assert back == word, f"'{word}' -> '{telugu}' -> '{back}'"


# ===========================================================================
#  SECTION 4: MAPPING COMPLETENESS & CORRECTNESS
# ===========================================================================


def test_no_wrong_script_keys():
    """All Telugu->Kannada map keys must be in Telugu Unicode range (U+0C00-0C7F)."""
    for key in converter.telugu_to_kannada_map:
        for ch in key:
            code = ord(ch)
            if code > 127:
                assert 0x0C00 <= code <= 0x0C7F, (
                    f"Key char U+{code:04X} ({ch}) is not Telugu — "
                    f"might be Kannada (0C80-0CFF) or Tamil (0B80-0BFF)"
                )


def test_no_wrong_script_values():
    """All Telugu->Kannada map values must be in Kannada Unicode range (U+0C80-0CFF)."""
    for key, val in converter.telugu_to_kannada_map.items():
        for ch in val:
            code = ord(ch)
            if code > 127:
                assert 0x0C80 <= code <= 0x0CFF, (
                    f"Value char U+{code:04X} ({ch}) for key '{key}' is not Kannada"
                )


def test_reverse_map_no_wrong_script_keys():
    """All Kannada->Telugu map keys must be in Kannada range."""
    for key in converter.kannada_to_telugu_map:
        for ch in key:
            code = ord(ch)
            if code > 127:
                assert 0x0C80 <= code <= 0x0CFF, (
                    f"Reverse map key U+{code:04X} ({ch}) is not Kannada"
                )


def test_reverse_map_no_wrong_script_values():
    """All Kannada->Telugu map values must be in Telugu range."""
    for key, val in converter.kannada_to_telugu_map.items():
        for ch in val:
            code = ord(ch)
            if code > 127:
                assert 0x0C00 <= code <= 0x0C7F, (
                    f"Reverse map value U+{code:04X} ({ch}) for key '{key}' is not Telugu"
                )


def test_mapping_count():
    """Verify we have mappings for all expected character categories."""
    t2k_map = converter.telugu_to_kannada_map
    # 34 consonants + 16 vowels + 13 vowel signs + 4 special + 10 numbers + 1 conjunct = 78+
    assert len(t2k_map) >= 78, f"Expected >= 78 mappings, got {len(t2k_map)}"


def test_maps_are_inverses():
    """t2k and k2t maps must be exact inverses of each other."""
    for tel, kan in converter.telugu_to_kannada_map.items():
        assert kan in converter.kannada_to_telugu_map, (
            f"Kannada '{kan}' (from Telugu '{tel}') not in reverse map"
        )
        assert converter.kannada_to_telugu_map[kan] == tel, (
            f"Reverse map mismatch: k2t['{kan}'] = '{converter.kannada_to_telugu_map[kan]}', expected '{tel}'"
        )


def test_no_duplicate_values():
    """No two Telugu keys should map to the same Kannada value."""
    seen = {}
    for tel, kan in converter.telugu_to_kannada_map.items():
        assert kan not in seen, (
            f"Duplicate: both '{seen[kan]}' and '{tel}' map to '{kan}'"
        )
        seen[kan] = tel


def test_all_telugu_vowel_signs_mapped():
    """All standard Telugu vowel signs (U+0C3E..0C4C) plus halant (U+0C4D) must be mapped."""
    expected = [0x0C3E, 0x0C3F, 0x0C40, 0x0C41, 0x0C42, 0x0C43, 0x0C44,
                0x0C46, 0x0C47, 0x0C48, 0x0C4A, 0x0C4B, 0x0C4C, 0x0C4D]
    mapped = {ord(k) for k in converter.telugu_to_kannada_map if len(k) == 1}
    for code in expected:
        assert code in mapped, (
            f"Telugu vowel sign U+{code:04X} ({chr(code)}) is not in the mapping"
        )


# ===========================================================================
#  SECTION 5: EDGE CASES
# ===========================================================================


def test_empty_string():
    assert t2k('') == ''
    assert k2t('') == ''


def test_english_passthrough():
    text = "Hello World 123 !@#"
    assert t2k(text) == text
    assert k2t(text) == text


def test_mixed_telugu_english():
    text = 'రామ (Rama)'
    result = t2k(text)
    assert '(Rama)' in result
    assert 'ರಾಮ' in result


def test_mixed_kannada_english():
    text = 'ರಾಮ (Rama)'
    result = k2t(text)
    assert '(Rama)' in result
    assert 'రామ' in result


def test_spaces_preserved():
    assert t2k('రామ కృష్ణ') == 'ರಾಮ ಕೃಷ್ಣ'
    assert k2t('ರಾಮ ಕೃಷ್ಣ') == 'రామ కృష్ణ'


def test_newlines_preserved():
    telugu = 'రామ\nకృష్ణ'
    result = t2k(telugu)
    assert '\n' in result


def test_punctuation_preserved():
    for ch in '.,!?;:-()[]{}':
        assert t2k(ch) == ch
        assert k2t(ch) == ch


def test_verse_no_telugu_leakage():
    """After T->K conversion, no Telugu codepoints should remain."""
    telugu = 'శ్రీరమాపరిణయము తరిగొండ వెంగమాంబ'
    result = t2k(telugu)
    for ch in result:
        assert not (0x0C00 <= ord(ch) <= 0x0C7F), (
            f"Telugu U+{ord(ch):04X} ({ch}) leaked: {result}"
        )


def test_verse_no_kannada_leakage():
    """After K->T conversion, no Kannada codepoints should remain."""
    kannada = 'ಶ್ರೀರಮಾಪರಿಣಯಮು ತರಿಗೊಂಡ ವೆಂಗಮಾಂಬ'
    result = k2t(kannada)
    for ch in result:
        assert not (0x0C80 <= ord(ch) <= 0x0CFF), (
            f"Kannada U+{ord(ch):04X} ({ch}) leaked: {result}"
        )


def test_number_string():
    assert t2k('౧౨౩') == '೧೨೩'
    assert k2t('೧೨೩') == '౧౨౩'


def test_conjunct_ksha():
    assert t2k('క్ష') == 'ಕ್ಷ'
    assert k2t('ಕ್ಷ') == 'క్ష'


CONJUNCT_PAIRS = [
    ('స్త', 'ಸ್ತ'), ('న్న', 'ನ್ನ'), ('మ్మ', 'ಮ್ಮ'), ('ద్ధ', 'ದ್ಧ'),
    ('శ్ర', 'ಶ್ರ'), ('క్క', 'ಕ್ಕ'), ('ల్ల', 'ಲ್ಲ'), ('ప్ప', 'ಪ್ಪ'),
    ('త్త', 'ತ್ತ'), ('ద్ద', 'ದ್ದ'), ('బ్బ', 'ಬ್ಬ'), ('గ్గ', 'ಗ್ಗ'),
]


@pytest.mark.parametrize("telugu,kannada", CONJUNCT_PAIRS,
                         ids=[t for t, _ in CONJUNCT_PAIRS])
def test_conjunct_t2k(telugu, kannada):
    assert t2k(telugu) == kannada


@pytest.mark.parametrize("telugu,kannada", CONJUNCT_PAIRS,
                         ids=[k for _, k in CONJUNCT_PAIRS])
def test_conjunct_k2t(telugu, kannada):
    assert k2t(kannada) == telugu


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
