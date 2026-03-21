# Kannada-Telugu Converter

Bidirectional character-level Kannada-Telugu transliteration based on phonological equivalence research (Nidamarthy, 2021).

## Files

| File | Description |
|---|---|
| `kannada_telugu_converter_paper_based.py` | Character-level converter implementing phonological mapping tables |

## Usage

```python
from kannada_telugu_converter_paper_based import kannada_to_telugu, telugu_to_kannada

telugu = kannada_to_telugu("ಕನ್ನಡ ಪದ್ಯ")
kannada = telugu_to_kannada("తెలుగు పద్యం")
```
