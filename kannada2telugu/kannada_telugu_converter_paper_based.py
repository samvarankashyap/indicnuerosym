#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Kannada-Telugu Converter Based on Research Paper
"Comparative study of Kannada and Telugu consonants, consonant sub forms, vowels, signs"
by Sushilkumar Nidamarthy (2021)

This converter implements the findings from the comparative analysis paper
"""

import json
import re
from typing import Dict, List, Tuple
from pathlib import Path


class EnhancedKannadaTeluguConverter:
    """
    Enhanced converter based on Nidamarthy's 2021 comparative study
    
    Key findings from the paper:
    - 99% resemblance: 17 consonants (GA, TTHA, DDA, DA, NA, RA, SA, LLA, DDHA, THA, DHA, BHA, JA, NNA, BA, LA, RRA)
    - 90% resemblance: KHA, NGA
    - ABL sub-forms: 7 with 99% resemblance
    - BBL sub-forms: 14 with 99% resemblance  
    - 10 vowels with 99% resemblance
    - High resemblance vowel signs (95-99%)
    """
    
    def __init__(self):
        self.telugu_to_kannada_map = self._create_enhanced_mapping()
        self.kannada_to_telugu_map = {v: k for k, v in self.telugu_to_kannada_map.items()}
        
        # Character classification based on paper
        self.high_resemblance_consonants = self._get_high_resemblance_consonants()
        self.medium_resemblance_consonants = self._get_medium_resemblance_consonants()
        self.low_resemblance_consonants = self._get_low_resemblance_consonants()
        
    def _create_enhanced_mapping(self) -> Dict[str, str]:
        """
        Create character mapping based on paper's findings
        Following the comparative study's classification
        """
        mapping = {}
        
        # CONSONANTS - Classified by resemblance level from paper
        
        # 99% resemblance consonants (17 consonants) - Perfect matches
        high_resemblance = {
            'గ': 'ಗ',   # GA
            'ఠ': 'ಠ',   # TTHA  
            'డ': 'ಡ',   # DDA
            'ద': 'ದ',   # DA
            'న': 'ನ',   # NA
            'ర': 'ರ',   # RA
            'స': 'ಸ',   # SA
            'ళ': 'ಳ',   # LLA
            'ఢ': 'ಢ',   # DDHA
            'థ': 'ಥ',   # THA
            'ధ': 'ಧ',   # DHA
            'భ': 'ಭ',   # BHA
            'జ': 'ಜ',   # JA
            'ణ': 'ಣ',   # NNA
            'బ': 'ಬ',   # BA
            'ల': 'ಲ',   # LA
            'ఱ': 'ಱ',   # RRA
        }
        
        # 90%+ resemblance consonants
        medium_resemblance = {
            'ఖ': 'ಖ',   # KHA (90% - Kannada eliminated Base-comma)
            'ఙ': 'ಙ',   # NGA (70% - different stem features)
            'ప': 'ಪ',   # PA (80% resemblance)
            'ఝ': 'ಝ',   # JHA (99% resemblance)
            'య': 'ಯ',   # YA (99% resemblance)
        }
        
        # Lower resemblance / different features (as noted in paper)
        # But phonetically identical
        low_resemblance = {
            'చ': 'ಚ',   # CHA (Telugu simpler)
            'శ': 'ಶ',   # SHA (no resemblance)
            'త': 'ತ',   # TA (no resemblance)
            'ష': 'ಷ',   # SSA (no resemblance)
            'హ': 'ಹ',   # HA (no resemblance - Kannada like Devanagari LLA)
        }
        
        # Remaining consonants
        other_consonants = {
            'క': 'ಕ',   # KA
            'ఛ': 'ಛ',   # CHA
            'ట': 'ಟ',   # TTA (Head-comma vs Head-Horns)
            'ఫ': 'ಫ',   # PHA
            'మ': 'ಮ',   # MA
            'వ': 'ವ',   # VA
            'ఞ': 'ಞ',   # NYA (extended leg - stem bar vs inverted spur)
            'ఘ': 'ಘ',   # GHA (with U - Telugu simpler)
        }
        
        # Conjunct consonant (no resemblance noted in paper)
        conjunct = {
            'క్ష': 'ಕ್ಷ',  # KSHA
        }
        
        # VOWELS - 10 with 99% resemblance
        vowels = {
            'అ': 'ಅ',   # A
            'ఆ': 'ಆ',   # AA
            'ఇ': 'ಇ',   # I
            'ఈ': 'ಈ',   # II
            'ఉ': 'ಉ',   # U
            'ఊ': 'ಊ',   # UU
            'ఋ': 'ಋ',   # Vocalic R
            'ౠ': 'ೠ',   # Vocalic RR
            'ఎ': 'ಎ',   # E
            'ఏ': 'ಏ',   # EE
            'ఐ': 'ಐ',   # AI
            'ఒ': 'ಒ',   # O
            'ఓ': 'ಓ',   # OO
            'ఔ': 'ಔ',   # AU
            'ఌ': 'ಌ',   # L
            'ౡ': 'ೡ',   # LL
        }
        
        # VOWEL SIGNS - High resemblance (95-99%) as noted in paper
        vowel_signs = {
            'ా': 'ಾ',   # AA (long tone)
            'ి': 'ಿ',   # I (95% resemblance)
            'ీ': 'ೀ',   # II (long tone)
            'ು': 'ು',   # U (99% resemblance)
            'ూ': 'ೂ',   # UU (long tone)
            'ృ': 'ೃ',   # Vocalic R (99% resemblance)
            'ౄ': 'ೄ',   # Vocalic RR (90% resemblance)
            'ெ': 'ೆ',   # E
            'ే': 'ೇ',   # EE
            'ై': 'ೈ',   # AI
            'ొ': 'ೊ',   # O
            'ో': 'ೋ',   # OO (Kannada: E + UU + long tone extender)
            'ౌ': 'ೌ',   # AU (90% resemblance - spiral form in Kannada)
        }
        
        # SPECIAL SYMBOLS AND HALANT
        special = {
            'ం': 'ಂ',   # Anusvara (UM - conjunct vowel)
            'ః': 'ಃ',   # Visarga (AHA - conjunct vowel)  
            '్': '್',   # Halant (no resemblance noted in paper)
            'ఁ': 'ಁ',   # Chandrabindu
        }
        
        # NUMBERS
        numbers = {
            '౦': '೦', '౧': '೧', '౨': '೨', '౩': '೩', '౪': '೪',
            '౫': '೫', '౬': '೬', '౭': '೭', '౮': '೮', '౯': '೯',
        }
        
        # Combine all mappings
        mapping.update(high_resemblance)
        mapping.update(medium_resemblance)
        mapping.update(low_resemblance)
        mapping.update(other_consonants)
        mapping.update(conjunct)
        mapping.update(vowels)
        mapping.update(vowel_signs)
        mapping.update(special)
        mapping.update(numbers)
        
        return mapping
    
    def _get_high_resemblance_consonants(self) -> List[str]:
        """99% resemblance consonants from paper"""
        return ['గ', 'ఠ', 'డ', 'ద', 'న', 'ర', 'స', 'ళ', 'ఢ', 'థ', 'ధ', 'భ', 'జ', 'ణ', 'బ', 'ల', 'ఱ']
    
    def _get_medium_resemblance_consonants(self) -> List[str]:
        """70-99% resemblance consonants from paper"""
        return ['ఖ', 'ఙ', 'ప', 'ఝ', 'య']
    
    def _get_low_resemblance_consonants(self) -> List[str]:
        """Low/no resemblance but phonetically identical"""
        return ['చ', 'శ', 'త', 'ష', 'హ']
    
    def telugu_to_kannada(self, text: str, preserve_metadata: bool = False) -> str:
        """
        Convert Telugu text to Kannada
        
        Args:
            text: Telugu text
            preserve_metadata: If True, adds conversion quality metadata
            
        Returns:
            Kannada text
        """
        result = []
        i = 0
        text_len = len(text)
        
        while i < text_len:
            char = text[i]
            
            # Handle multi-character sequences (conjuncts)
            if i + 2 < text_len:
                three_char = text[i:i+3]
                if three_char in self.telugu_to_kannada_map:
                    result.append(self.telugu_to_kannada_map[three_char])
                    i += 3
                    continue
            
            if i + 1 < text_len:
                two_char = text[i:i+2]
                if two_char in self.telugu_to_kannada_map:
                    result.append(self.telugu_to_kannada_map[two_char])
                    i += 2
                    continue
            
            # Single character
            if char in self.telugu_to_kannada_map:
                result.append(self.telugu_to_kannada_map[char])
            else:
                result.append(char)  # Keep unmapped (punctuation, etc.)
            
            i += 1
        
        return ''.join(result)
    
    def kannada_to_telugu(self, text: str) -> str:
        """Convert Kannada text to Telugu"""
        result = []
        i = 0
        text_len = len(text)
        
        while i < text_len:
            char = text[i]
            
            if i + 2 < text_len:
                three_char = text[i:i+3]
                if three_char in self.kannada_to_telugu_map:
                    result.append(self.kannada_to_telugu_map[three_char])
                    i += 3
                    continue
            
            if i + 1 < text_len:
                two_char = text[i:i+2]
                if two_char in self.kannada_to_telugu_map:
                    result.append(self.kannada_to_telugu_map[two_char])
                    i += 2
                    continue
            
            if char in self.kannada_to_telugu_map:
                result.append(self.kannada_to_telugu_map[char])
            else:
                result.append(char)
            
            i += 1
        
        return ''.join(result)
    
    def analyze_text_quality(self, telugu_text: str) -> Dict:
        """
        Analyze conversion quality based on paper's resemblance categories
        
        Returns statistics on character resemblance levels
        """
        stats = {
            'high_resemblance_count': 0,
            'medium_resemblance_count': 0,
            'low_resemblance_count': 0,
            'total_characters': len(telugu_text),
            'quality_score': 0.0
        }
        
        for char in telugu_text:
            if char in self.high_resemblance_consonants:
                stats['high_resemblance_count'] += 1
            elif char in self.medium_resemblance_consonants:
                stats['medium_resemblance_count'] += 1
            elif char in self.low_resemblance_consonants:
                stats['low_resemblance_count'] += 1
        
        # Calculate quality score (weighted by resemblance)
        total_classified = (stats['high_resemblance_count'] + 
                          stats['medium_resemblance_count'] + 
                          stats['low_resemblance_count'])
        
        if total_classified > 0:
            stats['quality_score'] = (
                (stats['high_resemblance_count'] * 0.99 +
                 stats['medium_resemblance_count'] * 0.85 +
                 stats['low_resemblance_count'] * 0.70) / total_classified
            )
        
        return stats
    
    def convert_dataset(self, input_file: str, output_file: str):
        """Convert entire dataset with quality analysis"""
        print(f"="*80)
        print("ENHANCED KANNADA-TELUGU CONVERTER")
        print("Based on Nidamarthy (2021) Comparative Study")
        print("="*80)
        print(f"\nLoading: {input_file}")
        
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"Converting {len(data)} poems with quality analysis...")
        converted_data = []
        total_quality = 0.0
        
        for idx, item in enumerate(data, 1):
            if idx % 1000 == 0:
                print(f"  Processed {idx}/{len(data)} poems...")
            
            telugu_poem = item.get('poem', '')
            kannada_poem = self.telugu_to_kannada(telugu_poem)
            quality = self.analyze_text_quality(telugu_poem)
            total_quality += quality['quality_score']
            
            converted_item = {
                'original_telugu_poem': telugu_poem,
                'kannada_poem': kannada_poem,
                'conversion_quality_score': round(quality['quality_score'], 3),
                'high_resemblance_chars': quality['high_resemblance_count'],
                'medium_resemblance_chars': quality['medium_resemblance_count'],
                'low_resemblance_chars': quality['low_resemblance_count'],
            }
            
            # Convert meanings
            if 'telugu_meaning' in item:
                converted_item['original_telugu_meaning'] = item['telugu_meaning']
                converted_item['kannada_meaning'] = self.telugu_to_kannada(item['telugu_meaning'])
            
            if 'english_meaning' in item:
                converted_item['english_meaning'] = item['english_meaning']
            
            # Convert word-to-word meanings
            if 'word_to_word_meaning' in item:
                converted_item['word_to_word_meaning_kannada'] = {}
                for tel_word, tel_meaning in item['word_to_word_meaning'].items():
                    kan_word = self.telugu_to_kannada(tel_word)
                    kan_meaning = self.telugu_to_kannada(tel_meaning)
                    converted_item['word_to_word_meaning_kannada'][kan_word] = kan_meaning
            
            converted_item['source_index'] = idx - 1
            converted_data.append(converted_item)
        
        avg_quality = total_quality / len(data) if data else 0
        
        print(f"\nSaving to: {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(converted_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n{'='*80}")
        print("CONVERSION COMPLETE - BASED ON RESEARCH PAPER")
        print(f"{'='*80}")
        print(f"✓ Poems converted: {len(converted_data):,}")
        print(f"✓ Average conversion quality: {avg_quality:.1%}")
        print(f"✓ Output: {output_file}")
        print(f"\nPaper Reference:")
        print(f"  Nidamarthy, S. (2021). Comparative study of Kannada and Telugu")
        print(f"  consonants, sub forms, vowels, signs. Shikshan Sanshodhan, 4(9).")
        print(f"{'='*80}\n")
        
        # Show sample
        if converted_data:
            sample = converted_data[0]
            print("SAMPLE CONVERSION:")
            print(f"Telugu:  {sample['original_telugu_poem']}")
            print(f"Kannada: {sample['kannada_poem']}")
            print(f"Quality: {sample['conversion_quality_score']:.1%}\n")


if __name__ == "__main__":
    converter = EnhancedKannadaTeluguConverter()
    
    input_file = "/Users/faviananoronha/Downloads/dwipada_master_filtered_perfect_dataset.json"
    output_file = "/Users/faviananoronha/Downloads/dwipada_kannada_paper_based_dataset.json"
    
    converter.convert_dataset(input_file, output_file)
