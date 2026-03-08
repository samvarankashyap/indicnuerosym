"""
Dwipada IFT Pipeline — Unified Tool
=====================================
Merges all preprocessing, format conversion, profile assignment, and
fine-tuning into a single modular script with subcommands.

Subcommands:
    pipeline  — POS tagging + V1 profile assignment + curriculum split
    convert   — Convert training_ready.jsonl to alpaca/sharegpt/trl formats
    reassign  — Re-assign V2 profiles (G1-G12 + A1-A10) without re-doing POS
    finetune  — QLoRA fine-tune Gemma 3 4B on ift_alpaca.jsonl

Usage:
    python dwipada_ift.py pipeline  --input data.json --output training_ready.jsonl
    python dwipada_ift.py convert   --input training_ready.jsonl --outdir ./ift_data
    python dwipada_ift.py reassign  --input training_ready.jsonl --outdir ./v2
    python dwipada_ift.py finetune  --dataset ift_alpaca.jsonl --hf-token $HF_TOKEN --hf-repo user/repo

Requirements:
    Core     : pip install tqdm
    Pipeline : pip install spacy stanza
               python -m spacy download en_core_web_sm
               python -c "import stanza; stanza.download('te')"
    Finetune : pip install unsloth transformers trl datasets huggingface_hub
"""

import json
import random
import re
import os
import argparse
from pathlib import Path
from tqdm import tqdm


# ──────────────────────────────────────────────
# SECTION 0 ▸ SHARED UTILITIES
# ──────────────────────────────────────────────

def load_jsonl(path: str) -> list[dict]:
    """
    Transparently loads either:
      - A JSON array file  (.json)  → list parsed directly
      - A JSONL file       (.jsonl) → one record per line
    """
    with open(path, encoding="utf-8") as f:
        first_char = f.read(1)
        f.seek(0)
        if first_char == "[":
            data = json.load(f)
            if not isinstance(data, list):
                raise ValueError(f"{path} is a JSON object, expected a list/array.")
            return data
        else:
            records = []
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
            return records


def save_jsonl(records: list[dict], path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"  Saved {len(records)} records → {path}")


def _gana_breakdown(rec: dict) -> str:
    """Serialise chandassu_analysis into a readable string."""
    ca = rec.get("chandassu_analysis", {})
    parts = []
    for line_key, line_val in ca.items():
        bd = line_val.get("breakdown", "")
        yt = line_val.get("yati_check", "")
        pr = line_val.get("prasa_check", "")
        parts.append(f"{line_key}: {bd} | Yati: {yt} | Prasa: {pr}")
    return "\n".join(parts) if parts else "Analysis not available."


def _word_breakdown(rec: dict) -> str:
    """Serialise word_to_word_meaning with + sandhi notation."""
    wtw = rec.get("word_to_word_meaning", {})
    if not wtw:
        return "Word breakdown not available."
    return "\n".join(f"{k}: {v}" for k, v in wtw.items())


def _is_valid_dwipada(rec: dict) -> bool:
    """Heuristic validity check based on presence of chandassu_analysis."""
    ca = rec.get("chandassu_analysis", {})
    return bool(ca.get("line_1") and ca.get("line_2"))


def _extract_prasa(rec):
    try:
        pr = rec["chandassu_analysis"]["line_1"]["prasa_check"]
        m  = re.search(r"'([\u0C00-\u0C7F])'", pr)
        return m.group(1) if m else "వ"
    except Exception:
        return "వ"


def _extract_yati(rec):
    try:
        yt = rec["chandassu_analysis"]["line_1"]["yati_check"]
        letters = re.findall(r"'([\u0C00-\u0C7F])'", yt)
        return " and ".join(letters) if letters else "భ and భ"
    except Exception:
        return "భ and భ"


def _pos_tokens(rec, pos_types):
    """
    Try tags_telugu first. If empty, fall back to WTW keys
    (which are Telugu words with sandhi breaks — good enough as keyword hints).
    """
    tags = rec.get("tags_telugu", [])
    tokens = [t["token"] for t in tags if t.get("pos") in pos_types]
    if not tokens:
        wtw_keys = list(rec.get("word_to_word_meaning", {}).keys())[:3]
        tokens = [k.split("+")[0].strip() for k in wtw_keys if k.strip()]
    return tokens


def _line1(rec):
    lines = rec.get("poem", "").split("\n")
    return lines[0].strip() if lines else ""


# ── Output combination helpers ────────────────
def out_poem_wtw(rec):
    return (f"ద్విపద:\n{rec.get('poem','')}\n\n"
            f"ప్రతిపదార్థం:\n{_word_breakdown(rec)}")

def out_poem_wtw_chandassu(rec):
    return (f"ద్విపద:\n{rec.get('poem','')}\n\n"
            f"ప్రతిపదార్థం:\n{_word_breakdown(rec)}\n\n"
            f"ఛందస్సు విశ్లేషణ:\n{_gana_breakdown(rec)}")

def out_poem_chandassu(rec):
    return (f"ద్విపద:\n{rec.get('poem','')}\n\n"
            f"ఛందస్సు విశ్లేషణ:\n{_gana_breakdown(rec)}")

def out_poem_only(rec):
    return f"ద్విపద:\n{rec.get('poem','')}"


# ──────────────────────────────────────────────
# SECTION 1 ▸ POS TAGGING
# ──────────────────────────────────────────────

def load_spacy_english():
    """Load spaCy English model — fast & reliable for English POS."""
    try:
        import spacy
        return spacy.load("en_core_web_sm")
    except OSError:
        raise RuntimeError(
            "spaCy English model not found.\n"
            "Run: python -m spacy download en_core_web_sm"
        )

def load_stanza_telugu():
    """Load Stanza Telugu model — best available for Telugu POS tagging."""
    try:
        import stanza
        return stanza.Pipeline(
            lang='te',
            processors='tokenize,pos',
            verbose=False
        )
    except Exception:
        raise RuntimeError(
            "Stanza Telugu model not found.\n"
            "Run: python -c \"import stanza; stanza.download('te')\""
        )

def tag_english(text: str, nlp_en) -> list[dict]:
    """Returns list of {token, pos} dicts for content words only."""
    KEEP_POS = {"NOUN", "VERB", "ADJ", "ADV", "PROPN", "NUM"}
    doc = nlp_en(text)
    return [
        {"token": token.text, "pos": token.pos_}
        for token in doc
        if token.pos_ in KEEP_POS and not token.is_stop
    ]

def tag_telugu(text: str, nlp_te) -> list[dict]:
    """Returns list of {token, pos, xpos} dicts."""
    KEEP_UPOS = {"NOUN", "VERB", "ADJ", "ADV", "PROPN", "NUM"}
    doc = nlp_te(text)
    results = []
    for sentence in doc.sentences:
        for word in sentence.words:
            if word.upos in KEEP_UPOS:
                results.append({
                    "token": word.text,
                    "pos":   word.upos,
                    "xpos":  word.xpos
                })
    return results

def add_pos_tags(records: list[dict], nlp_en, nlp_te) -> list[dict]:
    """Enriches every record with tags_english and tags_telugu."""
    print("\n[Step 1] Adding POS tags...")
    for rec in tqdm(records, desc="POS Tagging"):
        eng_text = rec.get("english_meaning", "")
        tel_text = rec.get("telugu_meaning", "")
        rec["tags_english"] = tag_english(eng_text, nlp_en) if eng_text else []
        rec["tags_telugu"]  = tag_telugu(tel_text, nlp_te)  if tel_text else []
    return records


# ──────────────────────────────────────────────
# SECTION 2 ▸ V1 PROFILES (10 profiles)
# ──────────────────────────────────────────────

def build_profile_1(rec):  # Educational / Student
    out  = f"ప్రతిపదార్థం (Word-to-Word Meaning):\n{_word_breakdown(rec)}\n\n"
    out += f"తెలుగు భావం:\n{rec.get('telugu_meaning','')}\n\n"
    out += f"English Meaning:\n{rec.get('english_meaning','')}"
    return out

def build_profile_2(rec):  # Scholarly / Technical
    out  = f"Chandassu Analysis:\n{_gana_breakdown(rec)}\n\n"
    out += f"Poem (with Gana markers):\n{rec.get('poem','')}"
    return out

def build_profile_3(rec):  # Creative Writer
    out  = f"Poem:\n{rec.get('poem','')}\n\n"
    out += f"English Meaning:\n{rec.get('english_meaning','')}"
    return out

def build_profile_4(rec):  # Linguistic / Deconstruction
    out  = f"Sandhi Deconstruction:\n{_word_breakdown(rec)}\n\n"
    out += f"తెలుగు భావం:\n{rec.get('telugu_meaning','')}"
    return out

def build_profile_5(rec):  # Constraint-Bound (prove Prasa)
    out  = f"Poem:\n{rec.get('poem','')}\n\n"
    out += f"Chandassu Verification:\n{_gana_breakdown(rec)}"
    return out

def build_profile_6(rec):  # Comparative / Meaning Only
    out  = f"తెలుగు భావం: {rec.get('telugu_meaning','')}\n"
    out += f"English Meaning: {rec.get('english_meaning','')}"
    return out

def build_profile_7(rec):  # Debugger / Error-Fixing
    poem_lines = rec.get("poem", "").split("\n")
    corrupted  = poem_lines[0] + " [Prasa Error Introduced]" if poem_lines else ""
    corrected  = rec.get("poem", "")
    ca         = _gana_breakdown(rec)
    out  = f"Input (Potentially Invalid):\n{corrupted}\n\n"
    out += f"Corrected Poem:\n{corrected}\n\n"
    out += f"Analysis of Error:\n{ca}"
    return out

def build_profile_8(rec):  # Modern Context
    out  = f"English Meaning:\n{rec.get('english_meaning','')}\n\n"
    out += f"Poem:\n{rec.get('poem','')}\n\n"
    out += f"Chandassu Analysis:\n{_gana_breakdown(rec)}"
    return out

def build_profile_9(rec):  # Minimalist
    payload = {
        "prathipadartham":  rec.get("word_to_word_meaning", {}),
        "bhavam_te":        rec.get("telugu_meaning", ""),
        "bhavam_en":        rec.get("english_meaning", ""),
        "is_valid_dwipada": _is_valid_dwipada(rec)
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)

def build_profile_10(rec):  # Multi-Gana Variety
    out  = f"Detailed Gana Scansion:\n{_gana_breakdown(rec)}\n\n"
    out += f"Poem:\n{rec.get('poem','')}"
    return out


PROFILES_V1 = {
    1:  {
        "weight": 0.14,
        "system_tag": "[Educational]",
        "instruction": (
            "Assume role of a Telugu and Sanskrit scholar. "
            "Explain this Dwipada poem to a student. Break down the words "
            "(Prathipadartham) and give a simple summary in both Telugu and English."
        ),
        "builder": build_profile_1
    },
    2:  {
        "weight": 0.12,
        "system_tag": "[Scholarly]",
        "instruction": (
            "Perform a rigorous Chandassu analysis on the provided Dwipada. "
            "Identify every Gana (Indra/Surya), verify Yati Maitri, and confirm "
            "Prasa alignment. Present results in a structured format."
        ),
        "builder": build_profile_2
    },
    3:  {
        "weight": 0.10,
        "system_tag": "[Creative]",
        "instruction": (
            "I need an appreciation of this Dwipada poem. "
            "Show the poem and provide its English meaning so I can share it "
            "with a wider audience. Do not include scansion details."
        ),
        "builder": build_profile_3
    },
    4:  {
        "weight": 0.10,
        "system_tag": "[Linguistic]",
        "instruction": (
            "Deconstruct this Dwipada morphologically. Show Sandhi breaks using "
            "the '+' sign and provide a word-to-word translation to English "
            "showing grammatical relationships."
        ),
        "builder": build_profile_4
    },
    5:  {
        "weight": 0.10,
        "system_tag": "[Constraint]",
        "instruction": (
            "Verify this Dwipada poem for Prasa correctness. Show the poem and "
            "provide a full Gana breakdown proving it follows the 3 Indra + 1 Surya rule."
        ),
        "builder": build_profile_5
    },
    6:  {
        "weight": 0.13,
        "system_tag": "[Comparative]",
        "instruction": (
            "Given this Dwipada, provide only the Telugu Bhavam in a single line "
            "and the English meaning. Do not include the poem or scansion."
        ),
        "builder": build_profile_6
    },
    7:  {
        "weight": 0.08,
        "system_tag": "[Debugger]",
        "instruction": (
            "Analyze the following line for Dwipada validity. Check the Yati position "
            "and Prasa. If invalid, explain the error and provide a corrected version."
        ),
        "builder": build_profile_7
    },
    8:  {
        "weight": 0.08,
        "system_tag": "[Modern]",
        "instruction": (
            "Appreciate this Dwipada in a modern context. Show the English meaning "
            "first, then the poem, then the Chandassu analysis to show how the "
            "vocabulary fits the classical meter."
        ),
        "builder": build_profile_8
    },
    9:  {
        "weight": 0.13,
        "system_tag": "[Minimalist]",
        "instruction": (
            "Assume role of a Telugu and Sanskrit scholar and give me bhavam and "
            "prathipadartham of the following dwipada poem. If there are combined "
            "words please break them with + in prathipadartham. Further bhavam "
            "should be in single line in telugu and English. "
            "Just give only bhavam and prathipadartham of the given input. "
            "No additional data.\nPoem:"
        ),
        "builder": build_profile_9
    },
    10: {
        "weight": 0.02,
        "system_tag": "[GanaVariety]",
        "instruction": (
            "Provide a detailed rhythmic scansion of this Dwipada, identifying "
            "each Gana type used. I want to see the diverse rhythmic structure "
            "followed by the poem text."
        ),
        "builder": build_profile_10
    },
}

assert abs(sum(p["weight"] for p in PROFILES_V1.values()) - 1.0) < 0.01, \
    "V1 profile weights must sum to 1.0"

WORD_BREAKDOWN_PROFILES = {1, 4, 7}
THIN_WTW_SAFE_PROFILES  = {3, 6, 9}
WTW_RICHNESS_THRESHOLD  = 3


def get_eligible_profiles(rec: dict) -> tuple[list[int], list[float]]:
    """Returns (profile_ids, weights) filtered by the record's data richness."""
    wtw_len = len(rec.get("word_to_word_meaning", {}))
    is_thin = wtw_len < WTW_RICHNESS_THRESHOLD

    if is_thin:
        eligible_ids = sorted(THIN_WTW_SAFE_PROFILES)
    else:
        eligible_ids = list(PROFILES_V1.keys())

    raw_weights  = [PROFILES_V1[pid]["weight"] for pid in eligible_ids]
    total        = sum(raw_weights)
    norm_weights = [w / total for w in raw_weights]

    return eligible_ids, norm_weights


def assign_prompt_profile(rec: dict, rng: random.Random,
                          wtw_threshold: int = WTW_RICHNESS_THRESHOLD) -> dict:
    """Assigns a V1 profile to a record based on data richness."""
    wtw_len  = len(rec.get("word_to_word_meaning", {}))
    is_thin  = wtw_len < wtw_threshold

    eligible_ids, norm_weights = get_eligible_profiles(rec)
    pid     = rng.choices(eligible_ids, weights=norm_weights, k=1)[0]
    profile = PROFILES_V1[pid]

    source_tag  = "[Human_Style]" if not rec.get("is_synthetic_data", True) else "[Synthetic]"
    instruction = f"{source_tag} {profile['system_tag']}\n{profile['instruction']}"
    poem_block  = rec.get("poem", "")
    full_input  = f"{instruction}\n\n{poem_block}"
    output      = profile["builder"](rec)

    return {
        **rec,
        "profile_id":  pid,
        "input":       full_input,
        "output":      output,
        "source_tag":  source_tag,
        "thin_wtw":    is_thin,
        "wtw_len":     wtw_len,
    }


# ──────────────────────────────────────────────
# SECTION 3 ▸ V2 PROFILES (G1-G12 + A1-A10)
# ──────────────────────────────────────────────

def audit_tags(records: list[dict]) -> dict:
    """Check how many records have tags_telugu populated."""
    total      = len(records)
    has_tags   = sum(1 for r in records if r.get("tags_telugu"))
    empty_tags = total - has_tags
    return {
        "total":      total,
        "has_tags":   has_tags,
        "empty_tags": empty_tags,
        "pct_tagged": has_tags / total * 100 if total else 0,
    }


# ── Generation profile builders (G1-G12) ─────

def g1(rec):
    topic = rec.get("english_meaning", "")[:60].rstrip(",. ")
    return (
        f"Write a Dwipada poem in Telugu about the following theme.\n"
        f"Ensure it follows the classical Dwipada meter: 3 Indra Ganas + 1 Surya Gana per line.\n"
        f"Provide the poem followed by word-by-word meaning (ప్రతిపదార్థం).\n\n"
        f"Theme: {topic}"
    ), out_poem_wtw(rec)

def g2(rec):
    topic = rec.get("english_meaning", "")[:60].rstrip(",. ")
    return (
        f"Compose a Dwipada poem in Telugu on the theme: {topic}\n"
        f"Constraint: The Prasa letter (2nd letter of each line) must be '{_extract_prasa(rec)}'.\n"
        f"Provide the poem, ప్రతిపదార్థం, and Chandassu analysis proving the Prasa rule is met."
    ), out_poem_wtw_chandassu(rec)

def g3(rec):
    topic = rec.get("english_meaning", "")[:60].rstrip(",. ")
    return (
        f"Write a Telugu Dwipada poem about: {topic}\n"
        f"The Yati (caesura) letters must be: {_extract_yati(rec)}.\n"
        f"Provide the poem and a full Chandassu breakdown confirming the Yati placement."
    ), out_poem_chandassu(rec)

def g4(rec):
    topic = rec.get("english_meaning", "")[:60].rstrip(",. ")
    return (
        f"Compose a classical Telugu Dwipada on the theme: {topic}\n"
        f"Constraints:\n"
        f"  - Prasa letter: '{_extract_prasa(rec)}'\n"
        f"  - Yati letters: {_extract_yati(rec)}\n"
        f"  - Meter: 3 Indra Ganas + 1 Surya Gana per line\n"
        f"Output the poem, ప్రతిపదార్థం, and Chandassu proof."
    ), out_poem_wtw_chandassu(rec)

def g5(rec):
    return (
        f"The following is the meaning of a Telugu Dwipada poem in English.\n"
        f"Generate the Telugu Dwipada poem that expresses this meaning.\n"
        f"Follow the Dwipada meter strictly. Provide the poem and ప్రతిపదార్థం.\n\n"
        f"English Meaning: {rec.get('english_meaning','')}"
    ), out_poem_wtw(rec)

def g6(rec):
    return (
        f"క్రింది తెలుగు భావానికి అనుగుణంగా ఒక ద్విపద పద్యం రచించండి.\n"
        f"ద్విపద ఛందస్సు నియమాలు పాటించండి: ప్రతి పాదంలో 3 ఇంద్ర గణాలు + 1 సూర్య గణం.\n"
        f"పద్యం మరియు ప్రతిపదార్థం ఇవ్వండి.\n\n"
        f"తెలుగు భావం: {rec.get('telugu_meaning','')}"
    ), out_poem_wtw(rec)

def g7(rec):
    lines = rec.get("poem", "").split("\n")
    line2 = lines[1].strip() if len(lines) > 1 else ""
    return (
        f"The following is the first line of a Telugu Dwipada poem.\n"
        f"Complete the poem by writing the second line.\n"
        f"The second line must maintain the same Prasa (2nd letter match) and meter.\n\n"
        f"First line: {_line1(rec)}"
    ), f"రెండవ పాదం:\n{line2}"

def g8(rec):
    topic = rec.get("english_meaning", "")[:60].rstrip(",. ")
    nouns = _pos_tokens(rec, ["NOUN", "PROPN"])[:4]
    kw    = ", ".join(nouns) if nouns else "వాయువు, శరీరం"
    return (
        f"Write a Telugu Dwipada poem about: {topic}\n"
        f"The poem must incorporate these Telugu nouns/concepts: {kw}\n"
        f"Follow the Dwipada meter. Provide the poem and ప్రతిపదార్థం."
    ), out_poem_wtw(rec)

def g9(rec):
    topic = rec.get("english_meaning", "")[:60].rstrip(",. ")
    quals = _pos_tokens(rec, ["ADJ", "ADV"])[:3]
    kw    = ", ".join(quals) if quals else "మహిమాన్వితమైన, శాశ్వతమైన"
    return (
        f"Compose a Telugu Dwipada on the theme: {topic}\n"
        f"Use these descriptive qualities in the poem: {kw}\n"
        f"Strict Dwipada meter required. Provide poem and ప్రతిపదార్థం."
    ), out_poem_wtw(rec)

def g10(rec):
    return (
        f"Using the following English meaning as your guide, compose a Telugu Dwipada.\n"
        f"Hard constraint: Prasa letter must be '{_extract_prasa(rec)}'.\n"
        f"Output the poem and Chandassu analysis proving the constraint is satisfied.\n\n"
        f"English Meaning: {rec.get('english_meaning','')}"
    ), out_poem_chandassu(rec)

def g11(rec):
    topic = rec.get("english_meaning", "")[:60].rstrip(",. ")
    nouns = _pos_tokens(rec, ["NOUN", "PROPN"])[:3]
    adjs  = _pos_tokens(rec, ["ADJ"])[:2]
    kw    = ", ".join(nouns + adjs) if (nouns or adjs) else "ప్రాణం, లోకం"
    return (
        f"Compose a classical Telugu Dwipada with all of the following constraints:\n"
        f"  Theme    : {topic}\n"
        f"  Keywords : {kw}\n"
        f"  Prasa    : '{_extract_prasa(rec)}'\n"
        f"  Yati     : {_extract_yati(rec)}\n"
        f"  Meter    : 3 Indra Ganas + 1 Surya Gana per line\n\n"
        f"Provide the complete poem, ప్రతిపదార్థం, and Chandassu proof."
    ), out_poem_wtw_chandassu(rec)

def g12(rec):
    nouns = _pos_tokens(rec, ["NOUN", "PROPN"])[:3]
    hint  = f"\nKeyword hints: {', '.join(nouns)}" if nouns else ""
    return (
        f"క్రింది భావానికి తగిన ద్విపద పద్యం రచించండి.{hint}\n"
        f"కేవలం పద్యం మాత్రమే ఇవ్వండి.\n\n"
        f"భావం: {rec.get('telugu_meaning','')}"
    ), out_poem_only(rec)


# ── Analysis profile builders (A1-A10) ────────

def a1(rec):
    return (f"ప్రతిపదార్థం:\n{_word_breakdown(rec)}\n\n"
            f"తెలుగు భావం:\n{rec.get('telugu_meaning','')}\n\n"
            f"English Meaning:\n{rec.get('english_meaning','')}")

def a2(rec):
    return f"Chandassu Analysis:\n{_gana_breakdown(rec)}\n\nPoem:\n{rec.get('poem','')}"

def a3(rec):
    return f"Poem:\n{rec.get('poem','')}\n\nEnglish Meaning:\n{rec.get('english_meaning','')}"

def a4(rec):
    return (f"Sandhi Deconstruction:\n{_word_breakdown(rec)}\n\n"
            f"తెలుగు భావం:\n{rec.get('telugu_meaning','')}")

def a5(rec):
    return f"Poem:\n{rec.get('poem','')}\n\nChandassu Verification:\n{_gana_breakdown(rec)}"

def a6(rec):
    return (f"తెలుగు భావం: {rec.get('telugu_meaning','')}\n"
            f"English Meaning: {rec.get('english_meaning','')}")

def a7(rec):
    lines     = rec.get("poem", "").split("\n")
    corrupted = lines[0] + " [Prasa Error Introduced]" if lines else ""
    return (f"Input (Potentially Invalid):\n{corrupted}\n\n"
            f"Corrected Poem:\n{rec.get('poem','')}\n\n"
            f"Analysis of Error:\n{_gana_breakdown(rec)}")

def a8(rec):
    return (f"English Meaning:\n{rec.get('english_meaning','')}\n\n"
            f"Poem:\n{rec.get('poem','')}\n\n"
            f"Chandassu Analysis:\n{_gana_breakdown(rec)}")

def a9(rec):
    return json.dumps({
        "prathipadartham":  rec.get("word_to_word_meaning", {}),
        "bhavam_te":        rec.get("telugu_meaning", ""),
        "bhavam_en":        rec.get("english_meaning", ""),
        "is_valid_dwipada": _is_valid_dwipada(rec),
    }, ensure_ascii=False, indent=2)

def a10(rec):
    return f"Detailed Gana Scansion:\n{_gana_breakdown(rec)}\n\nPoem:\n{rec.get('poem','')}"


PROFILES_V2 = {
    # ── Generation ────────────────────────────
    "G1":  {"w": 0.0583, "type": "gen", "tag": "[Generate]",               "fn": g1,  "needs_wtw": False},
    "G2":  {"w": 0.0583, "type": "gen", "tag": "[Generate-Prasa]",          "fn": g2,  "needs_wtw": False},
    "G3":  {"w": 0.0583, "type": "gen", "tag": "[Generate-Yati]",           "fn": g3,  "needs_wtw": False},
    "G4":  {"w": 0.0583, "type": "gen", "tag": "[Generate-Prasa-Yati]",     "fn": g4,  "needs_wtw": False},
    "G5":  {"w": 0.0583, "type": "gen", "tag": "[Generate-from-English]",   "fn": g5,  "needs_wtw": False},
    "G6":  {"w": 0.0583, "type": "gen", "tag": "[Generate-from-Telugu]",    "fn": g6,  "needs_wtw": False},
    "G7":  {"w": 0.0583, "type": "gen", "tag": "[Complete-Line2]",          "fn": g7,  "needs_wtw": False},
    "G8":  {"w": 0.0583, "type": "gen", "tag": "[Generate-Noun-Constraint]","fn": g8,  "needs_wtw": False},
    "G9":  {"w": 0.0583, "type": "gen", "tag": "[Generate-Adj-Constraint]", "fn": g9,  "needs_wtw": False},
    "G10": {"w": 0.0583, "type": "gen", "tag": "[Generate-English-Prasa]",  "fn": g10, "needs_wtw": False},
    "G11": {"w": 0.0583, "type": "gen", "tag": "[Generate-Full-Constraints]","fn": g11,"needs_wtw": False},
    "G12": {"w": 0.0583, "type": "gen", "tag": "[Generate-Telugu-Keywords]","fn": g12, "needs_wtw": False},
    # ── Analysis ──────────────────────────────
    "A1":  {"w": 0.030,  "type": "ana", "tag": "[Educational]",
            "instruction": "Assume role of a Telugu and Sanskrit scholar. Explain this Dwipada poem to a student. Break down the words (Prathipadartham) and give a simple summary in both Telugu and English.",
            "fn": a1, "needs_wtw": True},
    "A2":  {"w": 0.030,  "type": "ana", "tag": "[Scholarly]",
            "instruction": "Perform a rigorous Chandassu analysis on the provided Dwipada. Identify every Gana (Indra/Surya), verify Yati Maitri, and confirm Prasa alignment.",
            "fn": a2, "needs_wtw": False},
    "A3":  {"w": 0.030,  "type": "ana", "tag": "[Creative]",
            "instruction": "Appreciate this Dwipada poem. Show the poem and provide its English meaning. Do not include scansion.",
            "fn": a3, "needs_wtw": False},
    "A4":  {"w": 0.030,  "type": "ana", "tag": "[Linguistic]",
            "instruction": "Deconstruct this Dwipada morphologically. Show Sandhi breaks using '+' and provide a word-to-word translation to Telugu.",
            "fn": a4, "needs_wtw": True},
    "A5":  {"w": 0.030,  "type": "ana", "tag": "[Constraint]",
            "instruction": "Verify this Dwipada for Prasa correctness. Show the poem and provide a full Gana breakdown proving the 3 Indra + 1 Surya rule.",
            "fn": a5, "needs_wtw": False},
    "A6":  {"w": 0.030,  "type": "ana", "tag": "[Comparative]",
            "instruction": "Give only the Telugu Bhavam in a single line and the English meaning. Do not include the poem or scansion.",
            "fn": a6, "needs_wtw": False},
    "A7":  {"w": 0.030,  "type": "ana", "tag": "[Debugger]",
            "instruction": "Analyze the following line for Dwipada validity. Check Yati and Prasa. If invalid, explain and provide a corrected version.",
            "fn": a7, "needs_wtw": False},
    "A8":  {"w": 0.030,  "type": "ana", "tag": "[Modern]",
            "instruction": "Appreciate this Dwipada in a modern context. Show English meaning first, then the poem, then Chandassu analysis.",
            "fn": a8, "needs_wtw": False},
    "A9":  {"w": 0.030,  "type": "ana", "tag": "[Minimalist]",
            "instruction": "Assume role of a Telugu and Sanskrit scholar and give me bhavam and prathipadartham of the following dwipada poem. If there are combined words please break them with + in prathipadartham. Further bhavam should be in single line in telugu and English. Just give only bhavam and prathipadartham of the given input. No additional data.\nPoem:",
            "fn": a9, "needs_wtw": True},
    "A10": {"w": 0.030,  "type": "ana", "tag": "[GanaVariety]",
            "instruction": "Provide a detailed rhythmic scansion of this Dwipada identifying each Gana type. Show the diverse rhythmic structure followed by the poem.",
            "fn": a10, "needs_wtw": False},
}

# Precomputed V2 weight tables
V2_WTW_THRESHOLD  = 3
V2_THIN_SAFE_IDS  = {pid for pid, p in PROFILES_V2.items() if not p["needs_wtw"]}
V2_ALL_IDS        = list(PROFILES_V2.keys())
V2_ALL_WEIGHTS    = [PROFILES_V2[pid]["w"] for pid in V2_ALL_IDS]
V2_THIN_IDS       = sorted(V2_THIN_SAFE_IDS)
V2_THIN_WEIGHTS_RAW = [PROFILES_V2[pid]["w"] for pid in V2_THIN_IDS]
V2_THIN_TOTAL     = sum(V2_THIN_WEIGHTS_RAW)
V2_THIN_WEIGHTS   = [w / V2_THIN_TOTAL for w in V2_THIN_WEIGHTS_RAW]


def assign_v2_profile(rec: dict, rng: random.Random) -> dict:
    """Assigns a V2 profile (G1-G12 or A1-A10) to a record."""
    wtw_len = len(rec.get("word_to_word_meaning", {}))
    is_thin = wtw_len < V2_WTW_THRESHOLD

    if is_thin:
        pid = rng.choices(V2_THIN_IDS, weights=V2_THIN_WEIGHTS, k=1)[0]
    else:
        pid = rng.choices(V2_ALL_IDS, weights=V2_ALL_WEIGHTS, k=1)[0]

    p       = PROFILES_V2[pid]
    src_tag = rec.get("source_tag", "[Human_Style]")

    if p["type"] == "gen":
        instruction, output = p["fn"](rec)
        full_input = f"{src_tag} {p['tag']}\n{instruction}"
    else:
        output     = p["fn"](rec)
        poem       = rec.get("poem", "")
        full_input = f"{src_tag} {p['tag']}\n{p['instruction']}\n\n{poem}"

    return {
        **rec,
        "profile_id":   pid,
        "profile_type": p["type"],
        "input":        full_input,
        "output":       output,
        "thin_wtw":     is_thin,
        "wtw_len":      wtw_len,
    }


# ──────────────────────────────────────────────
# SECTION 4 ▸ CURRICULUM SPLIT
# ──────────────────────────────────────────────

def split_curriculum(records: list[dict]) -> tuple[list, list]:
    """
    Stage 1 (Structural Tutor)  → synthetic only
    Stage 2 (Artistic Master)   → human (3x upsampled) + 10% synthetic rehearsal
    """
    human     = [r for r in records if not r.get("is_synthetic_data", True)]
    synthetic = [r for r in records if r.get("is_synthetic_data", True)]

    print(f"\n[Curriculum Split]")
    print(f"  Human poems    : {len(human)}")
    print(f"  Synthetic poems: {len(synthetic)}")

    stage1 = synthetic
    stage2 = human * 3 + synthetic[:max(1, len(synthetic) // 10)]
    random.shuffle(stage2)

    print(f"  Stage-1 size   : {len(stage1)}")
    print(f"  Stage-2 size   : {len(stage2)}")
    return stage1, stage2


# ──────────────────────────────────────────────
# SECTION 5 ▸ IFT FORMAT CONVERTERS
# ──────────────────────────────────────────────

def to_alpaca(rec: dict) -> dict:
    """Alpaca format — used by Unsloth, Axolotl (alpaca template), TRL alpaca."""
    return {
        "instruction": rec.get("input", ""),
        "input":       "",
        "output":      rec.get("output", ""),
        "_profile_id": rec.get("profile_id"),
        "_source_tag": rec.get("source_tag"),
        "_thin_wtw":   rec.get("thin_wtw"),
        "_wtw_len":    rec.get("wtw_len"),
        "_source":     rec.get("source", ""),
    }


def to_sharegpt(rec: dict) -> dict:
    """ShareGPT / conversation format — for Axolotl (sharegpt), LLaMA-Factory."""
    full_input = rec.get("input", "")
    output     = rec.get("output", "")

    lines = full_input.split("\n", 1)
    if len(lines) == 2 and lines[0].startswith("["):
        system_line = lines[0].strip()
        human_turn  = lines[1].strip()
    else:
        system_line = ""
        human_turn  = full_input.strip()

    conversations = []
    if system_line:
        conversations.append({"from": "system", "value": system_line})
    conversations.append({"from": "human", "value": human_turn})
    conversations.append({"from": "gpt",   "value": output})

    return {
        "conversations": conversations,
        "_profile_id": rec.get("profile_id"),
        "_source_tag": rec.get("source_tag"),
        "_thin_wtw":   rec.get("thin_wtw"),
        "_wtw_len":    rec.get("wtw_len"),
        "_source":     rec.get("source", ""),
    }


def to_trl(rec: dict) -> dict:
    """HuggingFace TRL SFTTrainer native format with OpenAI-style messages."""
    full_input = rec.get("input", "")
    output     = rec.get("output", "")

    lines = full_input.split("\n", 1)
    if len(lines) == 2 and lines[0].startswith("["):
        system_content = lines[0].strip()
        user_content   = lines[1].strip()
    else:
        system_content = "You are a Telugu and Sanskrit scholar specializing in Dwipada poetry."
        user_content   = full_input.strip()

    return {
        "messages": [
            {"role": "system",    "content": system_content},
            {"role": "user",      "content": user_content},
            {"role": "assistant", "content": output},
        ],
        "_profile_id": rec.get("profile_id"),
        "_source_tag": rec.get("source_tag"),
        "_thin_wtw":   rec.get("thin_wtw"),
        "_wtw_len":    rec.get("wtw_len"),
        "_source":     rec.get("source", ""),
    }


def print_stats(records: list[dict], fmt_name: str):
    profile_counts = {}
    thin_count     = 0
    human_count    = 0

    for r in records:
        pid = r.get("_profile_id")
        profile_counts[pid] = profile_counts.get(pid, 0) + 1
        if r.get("_thin_wtw"):
            thin_count += 1
        if r.get("_source_tag") == "[Human_Style]":
            human_count += 1

    total = len(records)
    print(f"\n── {fmt_name} Distribution ({total} records) ──────────────")
    for pid in sorted(profile_counts):
        count = profile_counts[pid]
        bar   = "█" * int(count / total * 40)
        print(f"  Profile {pid:>2}: {count:>5} ({count/total*100:4.1f}%)  {bar}")
    print(f"\n  Human records  : {human_count:>6} ({human_count/total*100:.1f}%)")
    print(f"  Thin WTW       : {thin_count:>6} ({thin_count/total*100:.1f}%)")
    print("─" * 55)


# ──────────────────────────────────────────────
# SECTION 6 ▸ FINE-TUNING
# ──────────────────────────────────────────────

ALPACA_PROMPT = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{}

### Response:
{}"""


def run_finetune(dataset_path, hf_token, hf_repo,
                 max_seq_len=1024, lora_rank=32, epochs=2,
                 push_to_hub=True):
    """
    Full QLoRA fine-tuning pipeline for Gemma 3 4B on Dwipada data.

    Args:
        dataset_path: Path to ift_alpaca.jsonl
        hf_token:     HuggingFace token with WRITE scope
        hf_repo:      HuggingFace repo name (e.g. user/dwipada-gemma3-4b)
        max_seq_len:  Max sequence length (1024 for T4, 2048 for A100)
        lora_rank:    LoRA rank (32 conservative for T4)
        epochs:       Number of training epochs
        push_to_hub:  Whether to push adapter to HuggingFace Hub
    """
    import subprocess
    import torch
    from datasets import Dataset
    from unsloth import FastLanguageModel, is_bfloat16_supported
    from trl import SFTTrainer
    from transformers import TrainingArguments
    from huggingface_hub import login, create_repo

    # ── GPU Check ──────────────────────────────
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        print(result.stdout)
    except FileNotFoundError:
        print("nvidia-smi not found — ensure a CUDA GPU is available.")

    # ── HuggingFace Login ──────────────────────
    login(token=hf_token, add_to_git_credential=False)
    if push_to_hub:
        create_repo(hf_repo, repo_type='model', exist_ok=True, private=True)
    print(f"Logged in. Adapter will be pushed to: https://huggingface.co/{hf_repo}")

    # ── Load Model ─────────────────────────────
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name     = "google/gemma-3-4b-it",
        max_seq_length = max_seq_len,
        dtype          = None,
        load_in_4bit   = True,
    )
    print(f"Model loaded. Dtype: {model.dtype}")
    print(f"VRAM used: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # ── Attach LoRA Adapters ───────────────────
    model = FastLanguageModel.get_peft_model(
        model,
        r                           = lora_rank,
        lora_alpha                  = lora_rank * 2,
        lora_dropout                = 0.05,
        target_modules              = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias                        = "none",
        use_gradient_checkpointing  = "unsloth",
        random_state                = 42,
        use_rslora                  = False,
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable/1e6:.1f}M / {total/1e6:.0f}M "
          f"({100*trainable/total:.2f}%)")

    # ── Load & Order Dataset ───────────────────
    raw = load_jsonl(dataset_path)
    synthetic = [r for r in raw if r.get('_source_tag') == '[Synthetic]']
    human     = [r for r in raw if r.get('_source_tag') == '[Human_Style]']
    ordered   = synthetic + human

    print(f"Total records : {len(ordered)}")
    print(f"Synthetic     : {len(synthetic)} (first in training order)")
    print(f"Human         : {len(human)} (second in training order)")

    dataset = Dataset.from_list([
        {"instruction": r["instruction"], "output": r["output"]}
        for r in ordered
    ])

    # ── Apply Alpaca Prompt Template ───────────
    eos_token = tokenizer.eos_token

    def format_prompts(examples):
        texts = []
        for instr, out in zip(examples["instruction"], examples["output"]):
            text = ALPACA_PROMPT.format(instr, out) + eos_token
            texts.append(text)
        return {"text": texts}

    dataset = dataset.map(format_prompts, batched=True)
    print(f"Dataset ready: {dataset}")

    # ── VRAM Check ─────────────────────────────
    gpu_stats  = torch.cuda.get_device_properties(0)
    used_vram  = torch.cuda.memory_allocated() / 1e9
    total_vram = gpu_stats.total_memory / 1e9
    print(f"GPU           : {gpu_stats.name}")
    print(f"VRAM used     : {used_vram:.2f} GB")
    print(f"VRAM total    : {total_vram:.2f} GB")
    print(f"VRAM free     : {total_vram - used_vram:.2f} GB")

    # ── Configure Trainer ──────────────────────
    training_args_kwargs = dict(
        per_device_train_batch_size  = 1,
        gradient_accumulation_steps  = 8,
        warmup_steps                 = 100,
        num_train_epochs             = epochs,
        learning_rate                = 2e-4,
        fp16                         = not is_bfloat16_supported(),
        bf16                         = is_bfloat16_supported(),
        logging_steps                = 50,
        optim                        = 'adamw_8bit',
        weight_decay                 = 0.01,
        lr_scheduler_type            = 'cosine',
        seed                         = 42,
        output_dir                   = './dwipada_checkpoints',
        save_strategy                = 'steps',
        save_steps                   = 500,
        save_total_limit             = 2,
        report_to                    = 'none',
    )

    if push_to_hub:
        training_args_kwargs.update(
            push_to_hub   = True,
            hub_model_id  = hf_repo,
            hub_token     = hf_token,
            hub_strategy  = 'checkpoint',
        )

    trainer = SFTTrainer(
        model              = model,
        tokenizer          = tokenizer,
        train_dataset      = dataset,
        dataset_text_field = 'text',
        max_seq_length     = max_seq_len,
        dataset_num_proc   = 2,
        packing            = True,
        args               = TrainingArguments(**training_args_kwargs),
    )

    print('Trainer configured.')
    print(f'Effective batch size : {1 * 8}')
    print(f'Max sequence length  : {max_seq_len}')
    print(f'Epochs               : {epochs}')

    # ── Train ──────────────────────────────────
    trainer_stats = trainer.train()
    print(f"\nTraining complete.")
    print(f"Total steps     : {trainer_stats.global_step}")
    print(f"Training loss   : {trainer_stats.training_loss:.4f}")
    print(f"Runtime         : {trainer_stats.metrics['train_runtime']/3600:.2f} hours")

    # ── Quick Inference Test ───────────────────
    FastLanguageModel.for_inference(model)

    test_poem = "భువనత్రయాధారభూతమయుండు \nపవనుండు లేకున్న బడు శరీరములు"
    test_prompt = ALPACA_PROMPT.format(
        f"[Human_Style] [Minimalist]\n"
        f"Assume role of a Telugu and Sanskrit scholar and give me bhavam and "
        f"prathipadartham of the following dwipada poem. If there are combined "
        f"words please break them with + in prathipadartham. Further bhavam "
        f"should be in single line in telugu and English. Just give only bhavam "
        f"and prathipadartham of the given input. No additional data.\n"
        f"Poem:\n\n{test_poem}",
        ""
    )

    inputs = tokenizer(test_prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(
        **inputs,
        max_new_tokens  = 512,
        temperature     = 0.7,
        top_p           = 0.9,
        do_sample       = True,
        pad_token_id    = tokenizer.eos_token_id,
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response_only = response[len(test_prompt):].strip()
    print("── Model Response ───────────────────────────")
    print(response_only)

    # ── Save & Push ────────────────────────────
    save_dir = './dwipada_lora_adapter'
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f'Adapter saved locally to {save_dir}')

    if push_to_hub:
        print(f'Pushing to https://huggingface.co/{hf_repo} ...')
        model.push_to_hub(hf_repo, token=hf_token,
                          commit_message='Add Dwipada QLoRA adapter — final')
        tokenizer.push_to_hub(hf_repo, token=hf_token,
                              commit_message='Add tokenizer')
        print(f'\nAdapter pushed to: https://huggingface.co/{hf_repo}')

    print('\nLocal adapter files:')
    for f in sorted(os.listdir(save_dir)):
        size = os.path.getsize(f'{save_dir}/{f}') / 1e6
        print(f'  {f:<45} {size:.1f} MB')

    print("\nDone.")


# ──────────────────────────────────────────────
# SECTION 7 ▸ SUBCOMMAND RUNNERS
# ──────────────────────────────────────────────

def cmd_pipeline(args):
    """Run the full pipeline: POS tagging → V1 profile assignment → curriculum split."""
    rng = random.Random(42)

    print(f"\n[Loading] {args.input}")
    records = load_jsonl(args.input)
    print(f"  Loaded {len(records)} records.")

    # Step 1: POS Tagging
    nlp_en = load_spacy_english()
    nlp_te = load_stanza_telugu()
    records = add_pos_tags(records, nlp_en, nlp_te)

    save_jsonl(records, args.pos_output)
    print(f"\n[Checkpoint] POS-tagged data saved to {args.pos_output}")

    # Step 2: Assign Prompt Profiles
    print("\n[Step 2] Assigning prompt profiles...")
    transformed = [assign_prompt_profile(rec, rng) for rec in tqdm(records, desc="Prompting")]

    # Step 3: Curriculum split
    stage1, stage2 = split_curriculum(transformed)

    # Save outputs
    print("\n[Saving outputs...]")
    stem   = Path(args.output).stem
    parent = Path(args.output).parent

    save_jsonl(transformed, args.output)
    save_jsonl(stage1, str(parent / f"{stem}_stage1_synthetic.jsonl"))
    save_jsonl(stage2, str(parent / f"{stem}_stage2_human.jsonl"))

    # Summary stats
    profile_counts = {}
    for r in transformed:
        pid = r.get("profile_id")
        profile_counts[pid] = profile_counts.get(pid, 0) + 1

    print("\n── Profile Distribution ──────────────────")
    for pid in sorted(profile_counts):
        count = profile_counts[pid]
        pct   = count / len(transformed) * 100
        print(f"  Profile {pid:>2}: {count:>5} samples ({pct:.1f}%)")
    print("──────────────────────────────────────────")

    thin_records = [r for r in transformed if r.get("thin_wtw")]
    thin_profile_counts = {}
    for r in thin_records:
        pid = r.get("profile_id")
        thin_profile_counts[pid] = thin_profile_counts.get(pid, 0) + 1

    print(f"\n── Thin WTW Records (< {WTW_RICHNESS_THRESHOLD} entries) ──────")
    print(f"  Total thin records : {len(thin_records)} "
          f"({len(thin_records)/len(transformed)*100:.1f}% of dataset)")
    print(f"  Restricted to profiles: {sorted(THIN_WTW_SAFE_PROFILES)}")
    print(f"  Distribution across safe profiles:")
    for pid in sorted(thin_profile_counts):
        print(f"    Profile {pid}: {thin_profile_counts[pid]} samples")
    print("──────────────────────────────────────────")
    print("\nPipeline complete.")


def cmd_convert(args):
    """Convert training_ready.jsonl to alpaca/sharegpt/trl formats."""
    print(f"\n[Loading] {args.input}")
    records = load_jsonl(args.input)
    print(f"  Loaded {len(records)} records.")

    alpaca_records   = [to_alpaca(r)   for r in records]
    sharegpt_records = [to_sharegpt(r) for r in records]
    trl_records      = [to_trl(r)      for r in records]

    out = Path(args.outdir)
    print("\n[Saving IFT files...]")
    save_jsonl(alpaca_records,   str(out / "ift_alpaca.jsonl"))
    save_jsonl(sharegpt_records, str(out / "ift_sharegpt.jsonl"))
    save_jsonl(trl_records,      str(out / "ift_trl.jsonl"))

    print_stats(alpaca_records, "All Formats")

    print("\n── Framework → File mapping ───────────────────────────")
    print("  Unsloth                 →  ift_alpaca.jsonl")
    print("  Axolotl (alpaca)        →  ift_alpaca.jsonl")
    print("  Axolotl (sharegpt)      →  ift_sharegpt.jsonl")
    print("  LLaMA-Factory           →  ift_sharegpt.jsonl")
    print("  HuggingFace TRL         →  ift_trl.jsonl")
    print("─" * 55)
    print("\nDone.")


def cmd_reassign(args):
    """Re-assign V2 profiles (G1-G12 + A1-A10) without re-doing POS tagging."""
    rng = random.Random(42)
    out = Path(args.outdir)

    print(f"\n[Loading] {args.input}")
    records = load_jsonl(args.input)
    print(f"  Loaded {len(records)} records.")

    # Audit tags_telugu
    audit = audit_tags(records)
    print(f"\n[Audit] tags_telugu:")
    print(f"  Populated : {audit['has_tags']:>6} ({audit['pct_tagged']:.1f}%)")
    print(f"  Empty     : {audit['empty_tags']:>6}")
    if audit["pct_tagged"] < 50:
        print("  Warning: Less than 50% have tags_telugu — G8/G9/G11 will use WTW-key fallback.")
    else:
        print("  tags_telugu looks good — keyword profiles will use POS tokens.")

    # Re-assign profiles
    print("\n[Re-assigning profiles...]")
    transformed = [assign_v2_profile(r, rng) for r in tqdm(records)]

    # Curriculum split
    stage1, stage2 = split_curriculum(transformed)

    # Save
    stem = Path(args.input).stem
    print("\n[Saving...]")
    save_jsonl(transformed, str(out / f"{stem}_v2.jsonl"))
    save_jsonl(stage1,      str(out / f"{stem}_v2_stage1.jsonl"))
    save_jsonl(stage2,      str(out / f"{stem}_v2_stage2.jsonl"))

    # Stats
    total = len(transformed)
    gen_c = sum(1 for r in transformed if r["profile_type"] == "gen")
    ana_c = sum(1 for r in transformed if r["profile_type"] == "ana")
    thin_c = sum(1 for r in transformed if r["thin_wtw"])

    profile_counts = {}
    for r in transformed:
        pid = r["profile_id"]
        profile_counts[pid] = profile_counts.get(pid, 0) + 1

    print(f"\n── Profile Distribution ({'─'*44})")
    print(f"  {'TYPE':<6} {'PROFILE':<8} {'COUNT':>6}  {'%':>5}  {'BAR'}")
    print(f"  {'─'*56}")
    for pid in sorted(profile_counts, key=lambda x: (x[0], int(x[1:]))):
        c   = profile_counts[pid]
        pct = c / total * 100
        bar = "█" * int(pct * 1.5)
        print(f"  {PROFILES_V2[pid]['type'].upper():<6} {pid:<8} {c:>6}  {pct:>4.1f}%  {bar}")
    print(f"  {'─'*56}")
    print(f"  {'GEN':<6} {'ALL':<8} {gen_c:>6}  {gen_c/total*100:>4.1f}%")
    print(f"  {'ANA':<6} {'ALL':<8} {ana_c:>6}  {ana_c/total*100:>4.1f}%")
    print(f"\n  Thin WTW records : {thin_c} ({thin_c/total*100:.1f}%)")
    print(f"  Stage-1 (synth)  : {len(stage1)}")
    print(f"  Stage-2 (human)  : {len(stage2)}")
    print("\nDone.")


def cmd_finetune(args):
    """QLoRA fine-tune Gemma 3 4B on ift_alpaca.jsonl."""
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    hf_repo  = args.hf_repo  or os.environ.get("HF_REPO")

    if not hf_token:
        raise SystemExit("Error: --hf-token is required (or set HF_TOKEN env var)")
    if not hf_repo:
        raise SystemExit("Error: --hf-repo is required (or set HF_REPO env var)")

    run_finetune(
        dataset_path = args.dataset,
        hf_token     = hf_token,
        hf_repo      = hf_repo,
        max_seq_len  = args.max_seq_len,
        lora_rank    = args.lora_rank,
        epochs       = args.epochs,
        push_to_hub  = not args.no_push,
    )


# ──────────────────────────────────────────────
# SECTION 8 ▸ CLI
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Dwipada IFT Pipeline — unified preprocessing, conversion, and fine-tuning tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python dwipada_ift.py pipeline  --input data.json --output training_ready.jsonl
  python dwipada_ift.py convert   --input training_ready.jsonl --outdir ./ift_data
  python dwipada_ift.py reassign  --input training_ready.jsonl --outdir ./v2
  python dwipada_ift.py finetune  --dataset ift_alpaca.jsonl --hf-token $HF_TOKEN --hf-repo user/repo
"""
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # ── pipeline ───────────────────────────────
    p_pipeline = subparsers.add_parser(
        "pipeline",
        help="POS tagging + V1 profile assignment + curriculum split",
        description="Full pipeline: POS tag → assign 10 prompt profiles → curriculum split"
    )
    p_pipeline.add_argument(
        "--input", required=True,
        help="Path to input dataset — JSON array or JSONL"
    )
    p_pipeline.add_argument(
        "--output", required=True,
        help="Path for final training-ready JSONL"
    )
    p_pipeline.add_argument(
        "--pos-output", default="poems_with_pos.jsonl",
        help="Checkpoint path for POS-tagged data (default: poems_with_pos.jsonl)"
    )
    p_pipeline.set_defaults(func=cmd_pipeline)

    # ── convert ────────────────────────────────
    p_convert = subparsers.add_parser(
        "convert",
        help="Convert training_ready.jsonl to alpaca/sharegpt/trl formats",
        description="Converts training_ready.jsonl into three IFT format files"
    )
    p_convert.add_argument(
        "--input", required=True,
        help="Path to training_ready.jsonl"
    )
    p_convert.add_argument(
        "--outdir", default=".",
        help="Output directory for IFT files (default: current directory)"
    )
    p_convert.set_defaults(func=cmd_convert)

    # ── reassign ───────────────────────────────
    p_reassign = subparsers.add_parser(
        "reassign",
        help="Re-assign V2 profiles (G1-G12 + A1-A10) without POS re-tagging",
        description="Re-assigns 22 profiles (12 generation + 10 analysis) to training_ready.jsonl"
    )
    p_reassign.add_argument(
        "--input", required=True,
        help="Path to training_ready.jsonl"
    )
    p_reassign.add_argument(
        "--outdir", default=".",
        help="Output directory (default: current directory)"
    )
    p_reassign.set_defaults(func=cmd_reassign)

    # ── finetune ───────────────────────────────
    p_finetune = subparsers.add_parser(
        "finetune",
        help="QLoRA fine-tune Gemma 3 4B on ift_alpaca.jsonl",
        description="Fine-tunes Gemma 3 4B with QLoRA using Unsloth"
    )
    p_finetune.add_argument(
        "--dataset", required=True,
        help="Path to ift_alpaca.jsonl"
    )
    p_finetune.add_argument(
        "--hf-token", default=None,
        help="HuggingFace token (default: $HF_TOKEN env var)"
    )
    p_finetune.add_argument(
        "--hf-repo", default=None,
        help="HuggingFace repo name (default: $HF_REPO env var)"
    )
    p_finetune.add_argument(
        "--max-seq-len", type=int, default=1024,
        help="Max sequence length (default: 1024, use 2048 for A100)"
    )
    p_finetune.add_argument(
        "--lora-rank", type=int, default=32,
        help="LoRA rank (default: 32)"
    )
    p_finetune.add_argument(
        "--epochs", type=int, default=2,
        help="Number of training epochs (default: 2)"
    )
    p_finetune.add_argument(
        "--no-push", action="store_true",
        help="Skip pushing adapter to HuggingFace Hub"
    )
    p_finetune.set_defaults(func=cmd_finetune)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        raise SystemExit(1)

    args.func(args)


if __name__ == "__main__":
    main()