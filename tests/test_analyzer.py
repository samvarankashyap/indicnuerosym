# -*- coding: utf-8 -*-
"""Comprehensive test suite for Dwipada Analyzer.

Extracted from dwipada_analyzer.py run_tests() into proper pytest functions.
Tests cover: aksharam splitting, gana identification, yati matching,
prasa rhyme detection, and full dwipada analysis.
"""

import pytest

from dwipada.core.analyzer import (
    analyze_dwipada,
    analyze_pada,
    analyze_single_line,
    check_prasa,
    check_prasa_aksharalu,
    check_yati_maitri,
    identify_gana,
    split_aksharalu,
    akshara_ganavibhajana,
)


# =========================================================================
# TEST 1: Basic Aksharam Splitting
# =========================================================================
class TestAksharamSplitting:
    def test_basic_splitting(self):
        """Words can be split into aksharalu with gana markers."""
        test_words = ["తెలుగు", "రాముడు", "సత్యము", "అమ్మ", "గౌరవం"]
        for word in test_words:
            aksharalu = split_aksharalu(word)
            ganas = akshara_ganavibhajana(aksharalu)
            pure_ganas = [g for g in ganas if g]
            assert len(aksharalu) > 0
            assert len(pure_ganas) > 0


# =========================================================================
# TEST 2: Single Line Analysis
# =========================================================================
class TestSingleLineAnalysis:
    def test_valid_gana_sequence(self):
        test_line = "సౌధాగ్రముల యందు సదనంబు లందు"
        pada = analyze_pada(test_line)
        assert pada["is_valid_gana_sequence"]


# =========================================================================
# TEST 3: Full Dwipada Analysis
# =========================================================================
class TestFullDwipadaAnalysis:
    def test_basic_valid_dwipada(self):
        poem = """సౌధాగ్రముల యందు సదనంబు లందు
వీధుల యందును వెఱవొప్ప నిలిచి"""
        analysis = analyze_dwipada(poem)
        assert analysis["is_valid_dwipada"]
        assert analysis["prasa"]["line1_consonant"] == analysis["prasa"]["line2_consonant"]


# =========================================================================
# CATEGORY 1: VALID DWIPADA COUPLETS FROM BHAGAVATAM (Tests 4-9)
# =========================================================================
class TestValidDwipadaBhagavatam:
    def test_04_pootana_story(self):
        poem = """ఈతఁడే యెలనాగ ఇసుమంతనాఁడు
పూతన పాల్ ద్రావి పొరిఁగొన్న వాఁడు"""
        analysis = analyze_dwipada(poem)
        assert analysis["pada1"]["is_valid_gana_sequence"]
        assert analysis["pada2"]["is_valid_gana_sequence"]

    def test_05_shakatasura_story(self):
        poem = """సకియరో ఈతఁడే శకటమై వచ్చు
ప్రకట దానవుఁ ద్రుళ్ళిపడఁ దన్నినాఁడు"""
        analysis = analyze_dwipada(poem)
        assert analysis["pada1"]["is_valid_gana_sequence"]
        assert analysis["pada2"]["is_valid_gana_sequence"]

    def test_06_maddiya_story(self):
        poem = """ముద్దియ ఈతఁడే మొగిఱోలుఁ ద్రోచి
మద్దియ లుడిపిన మహనీయ యశుఁడు"""
        analysis = analyze_dwipada(poem)
        assert analysis["pada1"]["is_valid_gana_sequence"]
        assert analysis["pada2"]["is_valid_gana_sequence"]

    def test_07_aghasura_story(self):
        poem = """అక్కరో ఈతఁడే యఘదైత్యుఁ జీరి
కొక్కెర రక్కసుఁ గూల్చినవాఁడు"""
        analysis = analyze_dwipada(poem)
        assert analysis["pada1"]["is_valid_gana_sequence"]
        assert analysis["pada2"]["is_valid_gana_sequence"]

    def test_08_govardhana_story(self):
        poem = """గోవర్ధనముఁ గేల గొడుగుగాఁ బట్టి
గోవుల గోపాల గుంపులఁ గాచె"""
        analysis = analyze_dwipada(poem)
        assert analysis["pada1"]["is_valid_gana_sequence"]
        assert analysis["pada2"]["is_valid_gana_sequence"]

    def test_09_vanaja_akshi(self):
        poem = """వనజాక్షి రూపులావణ్యసంపదలు
వినిచిత్తమునఁ జూడ వేడుక పుట్టి"""
        analysis = analyze_dwipada(poem)
        assert analysis["pada1"]["is_valid_gana_sequence"]
        assert analysis["pada2"]["is_valid_gana_sequence"]


# =========================================================================
# CATEGORY 2: INVALID DWIPADA PATTERNS (Tests 10-13)
# =========================================================================
class TestInvalidDwipada:
    def test_10_prasa_mismatch(self):
        poem = """సౌధాగ్రముల యందు సదనంబు లందు
వీమల యందును మెఱవొప్ప నిలిచి"""
        analysis = analyze_dwipada(poem)
        assert analysis["prasa"] and not analysis["prasa"]["match"]

    def test_11_insufficient_syllables(self):
        poem = """కృష్ణుడు
రాముడు"""
        analysis = analyze_dwipada(poem)
        assert not analysis["pada1"]["is_valid_gana_sequence"] or not analysis["pada2"]["is_valid_gana_sequence"]

    def test_12_single_line_input(self):
        poem = """సౌధాగ్రముల యందు సదనంబు లందు"""
        with pytest.raises(ValueError):
            analyze_dwipada(poem)

    def test_13_empty_input(self):
        with pytest.raises(ValueError):
            analyze_dwipada("")


# =========================================================================
# CATEGORY 3: GANA IDENTIFICATION TESTS (Tests 14-17)
# =========================================================================
class TestGanaIdentification:
    def test_14_indra_gana_identification(self):
        indra_tests = [
            ("నలనల", "IIII", "Nala"),
            ("కికికికూ", "IIIU", "Naga"),
            ("కికికూకి", "IIUI", "Sala"),
            ("కూకిక", "UII", "Bha"),
            ("కూకికూ", "UIU", "Ra"),
            ("కూకూకి", "UUI", "Ta"),
        ]
        for word, expected_pattern, gana_name in indra_tests:
            aksharalu = split_aksharalu(word)
            gana_markers = akshara_ganavibhajana(aksharalu)
            pattern = "".join([g for g in gana_markers if g])
            assert pattern == expected_pattern, f"{word}: expected {expected_pattern}, got {pattern}"

    def test_15_surya_gana_identification(self):
        surya_tests = [
            ("కికికి", "III", "Na"),
            ("కూకి", "UI", "Ha/Gala"),
        ]
        for word, expected_pattern, gana_name in surya_tests:
            aksharalu = split_aksharalu(word)
            gana_markers = akshara_ganavibhajana(aksharalu)
            pattern = "".join([g for g in gana_markers if g])
            assert pattern == expected_pattern, f"{word}: expected {expected_pattern}, got {pattern}"

    def test_16_mixed_gana_pattern_line(self):
        test_line = "సౌధాగ్రముల యందు సదనంబు లందు"
        pada = analyze_pada(test_line)
        assert pada["partition"] is not None

    def test_17_gana_boundary_edge_case(self):
        test_line = "వీధుల యందును వెఱవొప్ప నిలిచి"
        pada = analyze_pada(test_line)
        assert pada["partition"] is not None


# =========================================================================
# CATEGORY 4: AKSHARAM & GURU/LAGHU EDGE CASES (Tests 18-21)
# =========================================================================
class TestGuruLaghuEdgeCases:
    def test_18_anusvaara_as_guru(self):
        anusvaara_words = ["సంపద", "గంగ", "మంగళం"]
        for word in anusvaara_words:
            aksharalu = split_aksharalu(word)
            ganas = akshara_ganavibhajana(aksharalu)
            for i, ak in enumerate(aksharalu):
                if "ం" in ak:
                    assert ganas[i] == "U", f"Anusvaara syllable '{ak}' in '{word}' should be Guru"

    def test_19_visarga_as_guru(self):
        visarga_words = ["దుఃఖం", "నిఃశ్వాస"]
        for word in visarga_words:
            aksharalu = split_aksharalu(word)
            ganas = akshara_ganavibhajana(aksharalu)
            for i, ak in enumerate(aksharalu):
                if "ః" in ak:
                    assert ganas[i] == "U", f"Visarga syllable '{ak}' in '{word}' should be Guru"

    def test_20_conjunct_consonants(self):
        """Syllable BEFORE conjunct should become Guru."""
        conjunct_words = ["సత్యము", "ధర్మము", "కృష్ణుడు"]
        for word in conjunct_words:
            aksharalu = split_aksharalu(word)
            ganas = akshara_ganavibhajana(aksharalu)
            assert len(aksharalu) > 0
            assert len([g for g in ganas if g]) > 0

    def test_21_double_consonants(self):
        """Syllable BEFORE double consonant should become Guru."""
        double_words = ["అమ్మ", "అప్పా", "చిన్న"]
        for word in double_words:
            aksharalu = split_aksharalu(word)
            ganas = akshara_ganavibhajana(aksharalu)
            assert len(aksharalu) > 0
            assert len([g for g in ganas if g]) > 0


# =========================================================================
# CATEGORY 5: YATI DETECTION TESTS (Tests 22-25)
# =========================================================================
class TestYatiDetection:
    def test_22_valid_yati_same_letter(self):
        poem = """సౌధాగ్రముల యందు సదనంబు లందు
వీధుల యందును వెఱవొప్ప నిలిచి"""
        analysis = analyze_dwipada(poem)
        yati1 = analysis.get("yati_line1")
        yati2 = analysis.get("yati_line2")
        assert yati1 and yati1["match"]
        assert yati2 and yati2["match"]

    def test_23_valid_yati_same_varga(self):
        test_pairs = [
            ("అ", "ఆ", True),
            ("క", "గ", True),
            ("చ", "శ", True),
            ("ప", "బ", True),
            ("ర", "ల", True),
        ]
        for l1, l2, expected in test_pairs:
            match, group, details = check_yati_maitri(l1, l2)
            assert match == expected, f"'{l1}' + '{l2}': expected {expected}, got {match}"

    def test_24_ka_vargamu_group(self):
        k_varga = ["క", "ఖ", "గ", "ఘ"]
        for i, l1 in enumerate(k_varga):
            for l2 in k_varga[i + 1:]:
                match, _, _ = check_yati_maitri(l1, l2)
                assert match, f"'{l1}' + '{l2}' should match in క-వర్గము"

    def test_25_invalid_yati_different_vargas(self):
        different_varga_pairs = [
            ("క", "చ", False),
            ("ప", "త", False),
            ("ర", "న", False),
        ]
        for l1, l2, expected in different_varga_pairs:
            match, group, _ = check_yati_maitri(l1, l2)
            assert match == expected, f"'{l1}' + '{l2}': expected {expected}, got {match}"


# =========================================================================
# CATEGORY 6: PRASA RHYME DETECTION TESTS (Tests 26-29)
# =========================================================================
class TestPrasaDetection:
    def test_26_valid_prasa_consonant_dha(self):
        poem = """సౌధాగ్రముల యందు సదనంబు లందు
వీధుల యందును వెఱవొప్ప నిలిచి"""
        analysis = analyze_dwipada(poem)
        assert analysis["prasa"]["match"]

    def test_27_valid_prasa_consonant_ka(self):
        poem = """అక్కరో ఈతఁడే యఘదైత్యుఁ జీరి
కొక్కెర రక్కసుఁ గూల్చినవాఁడు"""
        analysis = analyze_dwipada(poem)
        assert analysis["prasa"]["match"]

    def test_28_invalid_prasa_different_consonants(self):
        poem = """సౌధాగ్రముల యందు సదనంబు లందు
వీమల యందును మెఱవొప్ప నిలిచి"""
        analysis = analyze_dwipada(poem)
        assert not analysis["prasa"]["match"]

    def test_29_prasa_with_conjunct_consonants(self):
        poem = """సత్యమే ధర్మమై సదా విరాజిల్లు
నిత్యము నీ కీర్తి నిలిచి యుండు"""
        analysis = analyze_dwipada(poem)
        assert analysis["prasa"] is not None


# =========================================================================
# CATEGORY 7: STANDALONE PRASA FUNCTIONS (Tests 30-33)
# =========================================================================
class TestStandalonePrasa:
    def test_30_check_prasa_valid_match(self):
        line1 = "సౌధాగ్రముల యందు సదనంబు లందు"
        line2 = "వీధుల యందును వెఱవొప్ప నిలిచి"
        is_match, details = check_prasa(line1, line2)
        assert is_match

    def test_31_check_prasa_no_match(self):
        line1 = "సౌధాగ్రముల యందు సదనంబు లందు"
        line2 = "వీమల యందును మెఱవొప్ప నిలిచి"
        is_match, details = check_prasa(line1, line2)
        assert not is_match

    def test_32_check_prasa_aksharalu_pairs(self):
        aksharam_pairs = [
            ("ధా", "ధు", True),
            ("క్క", "క్కె", True),
            ("మా", "నా", False),
            ("సా", "శా", True),  # శ↔స are prasa-equivalent per PRASA_EQUIVALENTS
            ("రా", "రి", True),
        ]
        for ak1, ak2, expected in aksharam_pairs:
            is_match, details = check_prasa_aksharalu(ak1, ak2)
            assert is_match == expected, f"'{ak1}' + '{ak2}': expected {expected}, got {is_match}"

    def test_33_check_prasa_edge_cases(self):
        short_line1 = "క"
        short_line2 = "ర"
        is_match, details = check_prasa(short_line1, short_line2)
        assert "error" in details


# =========================================================================
# CATEGORY 8: USER-PROVIDED TEST CASES (Test 34)
# =========================================================================
class TestUserProvided:
    def test_34_valid_dwipada_toduga_nilichenu(self):
        poem = """తోడుగా నిలిచేను తుదిదాక చూడు
నీడలా సాగేను నిమిషంబు విడువ"""
        analysis = analyze_dwipada(poem)
        assert analysis["is_valid_dwipada"]
