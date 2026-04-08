"""
METRICALARGS Dataset Builder
==============================
Produces 3 separate IFT-ready JSONL files from the same 29k Dwipada corpus,
one per specialist model:

  ift_analysis.jsonl    — syllabification, meter detection, morphological glossing
  ift_retrieval.jsonl   — poem lookup from partial input / meaning
  ift_generation.jsonl  — poem composition from description, constraints, samasya

Every record in each file has:
  instruction : the user-facing prompt
  output      : the expected model response
  profile_id  : which profile was assigned
  source_tag  : [Human_Style] or [Synthetic]
  _meta       : original fields preserved for debugging

Usage:
    python metricalargs_builder.py --input training_ready.jsonl --outdir ./ift_data
    python metricalargs_builder.py --input dwipada_augmented_dataset.json --outdir ./ift_data
"""

import json
import random
import re
import argparse
from pathlib import Path
from tqdm import tqdm


# ──────────────────────────────────────────────
# IO HELPERS
# ──────────────────────────────────────────────

def load_data(path: str) -> list[dict]:
    """Handles JSON array (.json) and JSONL (.jsonl)."""
    with open(path, encoding="utf-8") as f:
        first = f.read(1); f.seek(0)
        if first == "[":
            return json.load(f)
        return [json.loads(l) for l in f if l.strip()]

def save_jsonl(records: list[dict], path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  ✓ {len(records):>6} records → {path}")


# ──────────────────────────────────────────────
# SHARED FIELD EXTRACTORS
# ──────────────────────────────────────────────

def _poem(rec):       return rec.get("poem", "")
def _eng(rec):        return rec.get("english_meaning", "")
def _tel(rec):        return rec.get("telugu_meaning", "")
def _src(rec):        return rec.get("source", "")
def _is_synth(rec):   return rec.get("is_synthetic_data", True)
def _src_tag(rec):    return "[Synthetic]" if _is_synth(rec) else "[Human_Style]"

def _wtw(rec):
    wtw = rec.get("word_to_word_meaning", {})
    if not wtw:
        return "Word breakdown not available."
    return "\n".join(f"{k}: {v}" for k, v in wtw.items())

def _wtw_rich(rec):
    return len(rec.get("word_to_word_meaning", {})) >= 3

def _gana(rec):
    ca = rec.get("chandassu_analysis", {})
    parts = []
    for k, v in ca.items():
        parts.append(
            f"{k}: {v.get('breakdown','')} | "
            f"Yati: {v.get('yati_check','')} | "
            f"Prasa: {v.get('prasa_check','')}"
        )
    return "\n".join(parts) or "Analysis not available."

def _prasa(rec):
    try:
        pr = rec["chandassu_analysis"]["line_1"]["prasa_check"]
        m  = re.search(r"'([\u0C00-\u0C7F])'", pr)
        return m.group(1) if m else "వ"
    except Exception:
        return "వ"

def _yati(rec):
    try:
        yt = rec["chandassu_analysis"]["line_1"]["yati_check"]
        letters = re.findall(r"'([\u0C00-\u0C7F])'", yt)
        return " మరియు ".join(letters) if letters else "భ మరియు భ"
    except Exception:
        return "భ మరియు భ"

def _line1(rec):
    lines = _poem(rec).split("\n")
    return lines[0].strip() if lines else ""

def _line2(rec):
    lines = _poem(rec).split("\n")
    return lines[1].strip() if len(lines) > 1 else ""

def _breakdown_syllables(rec):
    """
    Derive a simplified syllable pattern string from chandassu_analysis.
    e.g. 'IIUI | UUI | UII | UI'
    Falls back gracefully if not available.
    """
    ca = rec.get("chandassu_analysis", {})
    line1 = ca.get("line_1", {}).get("breakdown", "")
    if not line1:
        return "Syllable pattern not available."
    # Extract just the pattern tokens (e.g. IIUI, UUI) from breakdown string
    patterns = re.findall(r'[IU]{2,}', line1)
    return " | ".join(patterns) if patterns else line1

def _gana_names(rec):
    """Extract Gana names (Sala, Ta, Bha, etc.) from breakdown."""
    ca = rec.get("chandassu_analysis", {})
    line1 = ca.get("line_1", {}).get("breakdown", "")
    names = re.findall(r'\(([A-Za-z/]+)\s*[-–]', line1)
    return ", ".join(names) if names else "Gana names not available."

def _corrupt_prasa(rec) -> tuple[str, str]:
    """
    Returns (corrupted_poem, corruption_description).
    Modifies the first aksara of line 2 to break Prasa.
    """
    lines = _poem(rec).split("\n")
    if len(lines) < 2:
        return _poem(rec), "Could not introduce error."
    line2 = lines[1].strip()
    # Replace first character of line 2 with 'X' marker for training signal
    corrupted_line2 = "క" + line2[1:] if line2 and line2[0] != "క" else "ప" + line2[1:]
    corrupted = lines[0] + "\n" + corrupted_line2
    desc = f"The 2nd letter of line 2 was changed to break the Prasa rule."
    return corrupted, desc

def _corrupt_yati(rec) -> tuple[str, str]:
    """Introduce a Yati error by prepending a mismatching aksara to line 1."""
    lines = _poem(rec).split("\n")
    if not lines:
        return _poem(rec), "Could not introduce error."
    line1 = lines[0].strip()
    # swap first aksara to break Yati
    corrupted_line1 = ("మ" if line1[0] != "మ" else "క") + line1[1:]
    rest = "\n".join(lines[1:])
    corrupted = corrupted_line1 + ("\n" + rest if rest else "")
    return corrupted, "The opening aksara of line 1 was changed to break Yati Maitri."

def _pos_nouns(rec):
    tags = rec.get("tags_telugu", [])
    tokens = [t["token"] for t in tags if t.get("pos") in ("NOUN","PROPN")][:3]
    if not tokens:
        tokens = [k.split("+")[0].strip()
                  for k in list(rec.get("word_to_word_meaning",{}).keys())[:3]]
    return tokens

def _make_record(rec, profile_id, instruction, output, model):
    return {
        "instruction": instruction,
        "output":      output,
        "profile_id":  profile_id,
        "model":       model,
        "source_tag":  _src_tag(rec),
        "is_synthetic": _is_synth(rec),
        "_source":     _src(rec),
    }


# ══════════════════════════════════════════════
# MODEL 1 — ANALYSIS
# Subtasks: syllabification, meter detection,
#           morphological glossing
# ══════════════════════════════════════════════

ANALYSIS_PROFILES = {

    "AN1": {
        "weight": 0.18,
        "desc": "Syllabify full poem — identify guru (U) / laghu (I) per syllable",
        "needs_wtw": False,
        "fn": lambda rec: (
            f"మీరు తెలుగు ఛందస్సు నిపుణుడు. క్రింది ద్విపద పద్యంలోని "
            f"ప్రతి అక్షరాన్ని గురువు (U) లేదా లఘువు (I) గా గుర్తించండి.\n\n"
            f"పద్యం:\n{_poem(rec)}",
            f"ఛందస్సు విశ్లేషణ (గురు-లఘు విభజన):\n{_gana(rec)}"
        )
    },

    "AN2": {
        "weight": 0.18,
        "desc": "Identify Gana names per group in each line",
        "needs_wtw": False,
        "fn": lambda rec: (
            f"క్రింది ద్విపద పద్యాన్ని గణాలుగా విభజించి, ప్రతి గణానికి "
            f"పేరు (ఇంద్ర/సూర్య గణాలు) చెప్పండి.\n\n"
            f"పద్యం:\n{_poem(rec)}",
            f"గణ విభజన:\n{_gana(rec)}"
        )
    },

    "AN3": {
        "weight": 0.15,
        "desc": "Meter detection — identify this as Dwipada and explain why",
        "needs_wtw": False,
        "fn": lambda rec: (
            f"క్రింది పద్యం ఏ ఛందస్సులో ఉంది? మీ సమాధానానికి కారణం చెప్పండి.\n\n"
            f"పద్యం:\n{_poem(rec)}",
            f"ఛందస్సు: ద్విపద\n\n"
            f"కారణం: ప్రతి పాదంలో 3 ఇంద్ర గణాలు + 1 సూర్య గణం ఉన్నాయి. "
            f"యతి మైత్రి మరియు ప్రాస నియమాలు పాటించబడ్డాయి.\n\n"
            f"విశ్లేషణ:\n{_gana(rec)}"
        )
    },

    "AN4": {
        "weight": 0.15,
        "desc": "Morphological gloss — WTW meaning with sandhi breaks",
        "needs_wtw": True,
        "fn": lambda rec: (
            f"క్రింది ద్విపద పద్యంలోని సంధి పదాలను '+' గుర్తుతో విడదీసి, "
            f"ప్రతి పదానికి అర్థం రాయండి.\n\n"
            f"పద్యం:\n{_poem(rec)}",
            f"ప్రతిపదార్థం:\n{_wtw(rec)}"
        )
    },

    "AN5": {
        "weight": 0.17,
        "desc": "Full morphological analysis — WTW + telugu bhavam",
        "needs_wtw": True,
        "fn": lambda rec: (
            f"క్రింది పద్యానికి సంపూర్ణ విశ్లేషణ చేయండి: "
            f"సంధి విభజన, ప్రతిపదార్థం మరియు తెలుగు భావం ఇవ్వండి.\n\n"
            f"పద్యం:\n{_poem(rec)}",
            f"ప్రతిపదార్థం:\n{_wtw(rec)}\n\n"
            f"తెలుగు భావం: {_tel(rec)}"
        )
    },

    "AN6": {
        "weight": 0.17,
        "desc": "Single line syllabification — given line 1 only",
        "needs_wtw": False,
        "fn": lambda rec: (
            f"క్రింది పద్య పాదంలోని గురు-లఘు క్రమాన్ని మరియు గణాలను గుర్తించండి.\n\n"
            f"పాదం: {_line1(rec)}",
            f"విశ్లేషణ:\n" +
            (rec.get("chandassu_analysis",{})
                .get("line_1",{})
                .get("breakdown","Analysis not available."))
        )
    },

}

def build_analysis(rec: dict, rng: random.Random) -> dict:
    profiles  = ANALYSIS_PROFILES
    wtw_rich  = _wtw_rich(rec)
    eligible  = {pid: p for pid, p in profiles.items()
                 if not p["needs_wtw"] or wtw_rich}
    if not eligible:
        eligible = {pid: p for pid, p in profiles.items()
                    if not p["needs_wtw"]}

    ids     = list(eligible.keys())
    weights = [eligible[pid]["weight"] for pid in ids]
    total   = sum(weights)
    weights = [w/total for w in weights]

    pid          = rng.choices(ids, weights=weights, k=1)[0]
    instruction, output = profiles[pid]["fn"](rec)
    return _make_record(rec, pid, instruction, output, "analysis")


# ══════════════════════════════════════════════
# MODEL 2 — RETRIEVAL
# Subtasks: from first verse, last verse,
#           middle verse, meaning (EN/TE)
# ══════════════════════════════════════════════

RETRIEVAL_PROFILES = {

    "RT1": {
        "weight": 0.18,
        "desc": "First line → retrieve full poem",
        "fn": lambda rec: (
            f"క్రింది పద్యపాదం ఒక ప్రసిద్ధ ద్విపద పద్యంలోని మొదటి పాదం. "
            f"ఈ పద్యాన్ని మొత్తం చెప్పండి. మీ సొంత వాక్యాలు ఉపయోగించవద్దు.\n\n"
            f"మొదటి పాదం: {_line1(rec)}",
            f"పూర్తి పద్యం:\n{_poem(rec)}"
        )
    },

    "RT2": {
        "weight": 0.18,
        "desc": "Last line → retrieve full poem",
        "fn": lambda rec: (
            f"క్రింది పద్యపాదం ఒక ద్విపద పద్యంలోని చివరి పాదం. "
            f"దీని ముందు పాదాన్ని గుర్తించి పద్యాన్ని పూర్తి చేయండి.\n\n"
            f"చివరి పాదం: {_line2(rec)}",
            f"పూర్తి పద్యం:\n{_poem(rec)}"
        )
    },

    "RT3": {
        "weight": 0.15,
        "desc": "Random/middle fragment → retrieve full poem",
        "fn": lambda rec: (
            f"క్రింది పద్యాంశం ఒక ద్విపద పద్యంలోని ఒక భాగం. "
            f"సంపూర్ణ పద్యాన్ని పేర్కొనండి. మీ సొంత వాక్యాలు ఉపయోగించవద్దు.\n\n"
            f"పద్యాంశం: {_line1(rec)[:len(_line1(rec))//2]}",
            f"పూర్తి పద్యం:\n{_poem(rec)}"
        )
    },

    "RT4": {
        "weight": 0.17,
        "desc": "English meaning → retrieve matching poem",
        "fn": lambda rec: (
            f"క్రింది English భావానికి సరిపోయే ద్విపద పద్యాన్ని చెప్పండి.\n\n"
            f"English Meaning: {_eng(rec)}",
            f"పద్యం:\n{_poem(rec)}"
        )
    },

    "RT5": {
        "weight": 0.17,
        "desc": "Telugu meaning → retrieve matching poem",
        "fn": lambda rec: (
            f"క్రింది తెలుగు భావానికి సరిపోయే ద్విపద పద్యాన్ని చెప్పండి.\n\n"
            f"తెలుగు భావం: {_tel(rec)}",
            f"పద్యం:\n{_poem(rec)}"
        )
    },

    "RT6": {
        "weight": 0.15,
        "desc": "Masked first line → complete it",
        "fn": lambda rec: (
            lambda line1: (
                f"క్రింది అసంపూర్ణ పద్య పాదాన్ని పూర్తి చేయండి. "
                f"ద్విపద ఛందస్సు నియమాలు పాటించాలి.\n\n"
                f"అసంపూర్ణ పాదం: {line1[:max(4, len(line1)//2)]}___",
                f"పూర్తి పాదం: {line1}"
            )
        )(_line1(rec))
    },

}

def build_retrieval(rec: dict, rng: random.Random) -> dict:
    ids     = list(RETRIEVAL_PROFILES.keys())
    weights = [RETRIEVAL_PROFILES[pid]["weight"] for pid in ids]
    total   = sum(weights)
    weights = [w/total for w in weights]

    pid             = rng.choices(ids, weights=weights, k=1)[0]
    instruction, output = RETRIEVAL_PROFILES[pid]["fn"](rec)
    return _make_record(rec, pid, instruction, output, "retrieval")


# ══════════════════════════════════════════════
# MODEL 3 — GENERATION
# Subtasks: from description, from meaning,
#           constrained, line completion,
#           riddle (RPFW), samasya (PFP)
# ══════════════════════════════════════════════

GENERATION_PROFILES = {

    "GN1": {
        "weight": 0.145,
        "desc": "Theme in English (short) → poem",
        "fn": lambda rec: (
            f"క్రింది అంశంపై తెలుగులో ఒక ద్విపద పద్యం రాయండి. "
            f"ద్విపద ఛందస్సు పాటించాలి: ప్రతి పాదంలో 3 ఇంద్ర గణాలు + 1 సూర్య గణం.\n\n"
            f"అంశం: {_eng(rec)[:60].rstrip(',. ')}",
            f"ద్విపద:\n{_poem(rec)}"
        )
    },

    "GN2": {
        "weight": 0.145,
        "desc": "Full English meaning → poem",
        "fn": lambda rec: (
            f"క్రింది English భావానికి అనుగుణంగా తెలుగులో ఒక ద్విపద పద్యం రచించండి. "
            f"ద్విపద ఛందస్సు కచ్చితంగా పాటించాలి.\n\n"
            f"English Meaning: {_eng(rec)}",
            f"ద్విపద:\n{_poem(rec)}"
        )
    },

    "GN3": {
        "weight": 0.145,
        "desc": "Telugu meaning → poem (fully Telugu prompt)",
        "fn": lambda rec: (
            f"క్రింది తెలుగు భావానికి అనుగుణంగా ఒక ద్విపద పద్యం రచించండి. "
            f"ప్రతి పాదంలో 3 ఇంద్ర గణాలు + 1 సూర్య గణం ఉండాలి.\n\n"
            f"తెలుగు భావం: {_tel(rec)}",
            f"ద్విపద:\n{_poem(rec)}"
        )
    },

    "GN4": {
        "weight": 0.115,
        "desc": "Line 1 given → complete line 2",
        "fn": lambda rec: (
            f"క్రింది ద్విపద పద్యం యొక్క మొదటి పాదం ఇవ్వబడింది. "
            f"రెండవ పాదాన్ని రాయండి. ప్రాస నియమం మరియు ఛందస్సు పాటించాలి.\n\n"
            f"మొదటి పాదం: {_line1(rec)}",
            f"రెండవ పాదం:\n{_line2(rec)}"
        )
    },

    "GN5": {
        "weight": 0.07,
        "needs_prasa": True,
        "desc": "Theme + Prasa constraint → poem",
        "fn": lambda rec: (
            f"క్రింది అంశంపై తెలుగులో ఒక ద్విపద పద్యం రాయండి.\n"
            f"నిర్బంధం: ప్రాస అక్షరం (ప్రతి పాదంలో 2వ అక్షరం) '{_prasa(rec)}' అయి ఉండాలి.\n\n"
            f"అంశం: {_eng(rec)[:60].rstrip(',. ')}",
            f"ద్విపద:\n{_poem(rec)}"
        )
    },

    "GN6": {
        "weight": 0.115,
        "desc": "Theme + Yati constraint → poem",
        "fn": lambda rec: (
            f"క్రింది అంశంపై తెలుగులో ఒక ద్విపద పద్యం రాయండి.\n"
            f"నిర్బంధం: యతి స్థానంలో అక్షరాలు {_yati(rec)} అయి ఉండాలి.\n\n"
            f"అంశం: {_eng(rec)[:60].rstrip(',. ')}",
            f"ద్విపద:\n{_poem(rec)}"
        )
    },

    "GN7": {
        "weight": 0.13,
        "desc": "Riddle poem (RPFW) — hint at meaning without naming it",
        "fn": lambda rec: (
            f"పద్యాత్మకమైన పొడుపు కథలను మీ సొంత వాక్యాలలో చెప్పండి.\n"
            f"క్రింది భావాన్ని పేరు చెప్పకుండా సూచించే ఒక పొడుపు పద్యం రాయండి.\n\n"
            f"భావం (సమాధానం): {_tel(rec)}",
            f"పొడుపు పద్యం:\n{_poem(rec)}"
        )
    },

    "GN8": {
        "weight": 0.135,
        "desc": "Samasya (PFP) — given last line, build poem around it",
        "fn": lambda rec: (
            f"అవధాన ప్రక్రియలో సమస్య అంటే ఒక పద్య పాదాన్ని తార్కిక అంశంతో "
            f"ముడిపెట్టి వర్ణన చేయమని అడగడం. క్రింది చివరి పాదాన్ని "
            f"ఆధారంగా చేసుకుని ఒక ద్విపద పద్యం రాయండి.\n\n"
            f"సమస్య (చివరి పాదం): {_line2(rec)}",
            f"పూర్తి ద్విపద:\n{_poem(rec)}"
        )
    },

}

def _has_prasa(rec) -> bool:
    try:
        pr = rec["chandassu_analysis"]["line_1"]["prasa_check"]
        return bool(re.search(r"'([ఀ-౿])'", pr))
    except Exception:
        return False

def build_generation(rec: dict, rng: random.Random) -> dict:
    # GN5 only eligible when prasa letter is extractable (58% of corpus)
    eligible = {pid: p for pid, p in GENERATION_PROFILES.items()
                if not p.get("needs_prasa", False) or _has_prasa(rec)}
    ids     = list(eligible.keys())
    weights = [eligible[pid]["weight"] for pid in ids]
    total   = sum(weights)
    weights = [w / total for w in weights]
    pid             = rng.choices(ids, weights=weights, k=1)[0]
    instruction, output = GENERATION_PROFILES[pid]["fn"](rec)
    return _make_record(rec, pid, instruction, output, "generation")




# ══════════════════════════════════════════════
# MODEL 4 — SUPPORT  (SP1-SP3 only)
# SP4/SP5/SP6 dropped — require synthetic output
# with no ground truth in corpus.
# Subtasks: Prasa, Yati, Gana error correction
# ══════════════════════════════════════════════

def _corrupt_prasa_char(line: str, correct_char: str) -> str:
    """Replace 2nd Telugu aksara in line with a non-matching char."""
    te = [(i, c) for i, c in enumerate(line) if 'ఀ' <= c <= '౿']
    if len(te) < 2:
        return line
    idx = te[1][0]
    wrong = 'క' if correct_char != 'క' else 'మ'
    return line[:idx] + wrong + line[idx+1:]

def _corrupt_yati_char(line: str) -> str:
    """Replace 1st Telugu aksara in line to break Yati."""
    te = [(i, c) for i, c in enumerate(line) if 'ఀ' <= c <= '౿']
    if not te:
        return line
    idx = te[0][0]
    wrong = 'జ' if line[idx] != 'జ' else 'క'
    return line[:idx] + wrong + line[idx+1:]

SUPPORT_PROFILES = {
    "SP1": {
        "weight": 0.30,
        "desc": "Prasa error detect and fix",
        "needs_prasa": True,
        "fn": lambda rec: (
            f"క్రింది ద్విపద పద్యంలో ప్రాస నియమంలో తప్పు ఉంది. "
            f"తప్పును గుర్తించి సరైన పద్యాన్ని రాయండి.\n\n"
            f"తప్పు పద్యం:\n"
            f"{_poem(rec).split('\n')[0]}\n{_corrupt_prasa_char(_poem(rec).split('\n')[1].strip() if len(_poem(rec).split('\n'))>1 else '', _prasa(rec))}",
            
            f"తప్పు: రెండవ పాదంలో 2వ అక్షరం ప్రాస నియమానికి అనుగుణంగా లేదు. "
            f"సరైన ప్రాస అక్షరం \'{_prasa(rec)}\' అయి ఉండాలి.\n\n"
            f"సరైన పద్యం:\n{_poem(rec)}"
        )
    },

    "SP2": {
        "weight": 0.37,
        "desc": "Yati error detect and fix",
        "fn": lambda rec: (
            f"క్రింది ద్విపద పద్యంలో యతి మైత్రి నియమంలో తప్పు ఉంది. "
            f"తప్పును గుర్తించి సరైన పద్యాన్ని రాయండి.\n\n"
            f"తప్పు పద్యం:\n"
            f"{_corrupt_yati_char(_poem(rec).split('\n')[0].strip())}\n{'\n'.join(_poem(rec).split('\n')[1:])}",
            
            f"తప్పు: మొదటి పాదంలో యతి స్థానంలో అక్షరం {_yati(rec)} మైత్రి కలిగి ఉండాలి.\n\n"
            f"సరైన పద్యం:\n{_poem(rec)}"
        )
    },

    "SP3": {
        "weight": 0.33,
        "desc": "Gana/meter verify and confirm",
        "fn": lambda rec: (
            f"క్రింది ద్విపద పద్యంలో గణ నిర్మాణంలో తప్పు ఉందా? "
            f"పరీక్షించి, తప్పు ఉంటే గుర్తించి సరి చేయండి.\n\n"
            f"పద్యం:\n{_poem(rec)}",
            f"గణ విశ్లేషణ:\n{_gana(rec)}\n\n"
            f"निర్ధారణ: పద్యం సరైన ద్విపద నిర్మాణాన్ని పాటిస్తోంది. "
            f"ప్రతి పాదంలో 3 ఇంద్ర గణాలు + 1 సూర్య గణం ఉన్నాయి."
        )
    },
}

def build_support(rec: dict, rng: random.Random) -> dict:
    # SP1 only eligible when prasa letter is extractable
    eligible = {pid: p for pid, p in SUPPORT_PROFILES.items()
                if not p.get("needs_prasa", False) or _has_prasa(rec)}
    ids     = list(eligible.keys())
    weights = [eligible[pid]["weight"] for pid in ids]
    total   = sum(weights)
    weights = [w / total for w in weights]
    pid             = rng.choices(ids, weights=weights, k=1)[0]
    instruction, output = SUPPORT_PROFILES[pid]["fn"](rec)
    return _make_record(rec, pid, instruction, output, "support")

# ──────────────────────────────────────────────
# STATS PRINTER
# ──────────────────────────────────────────────

# ──────────────────────────────────────────────
# FORMAT CONVERTERS  (alpaca / sharegpt / trl)
# ──────────────────────────────────────────────

def to_alpaca(rec: dict) -> dict:
    """Unsloth, Axolotl alpaca template."""
    return {
        "instruction": rec.get("instruction", ""),
        "input":       "",
        "output":      rec.get("output", ""),
        "_profile_id": rec.get("profile_id"),
        "_model":      rec.get("model"),
        "_source_tag": rec.get("source_tag"),
        "_source":     rec.get("_source", ""),
    }

def to_sharegpt(rec: dict) -> dict:
    """LLaMA-Factory, Axolotl sharegpt template."""
    instruction = rec.get("instruction", "")
    output      = rec.get("output", "")
    lines = instruction.split("\n", 1)
    if len(lines) == 2 and lines[0].startswith("["):
        system_val = lines[0].strip()
        human_val  = lines[1].strip()
    else:
        system_val = f"You are a Telugu and Sanskrit scholar specialising in Dwipada poetry."
        human_val  = instruction.strip()
    return {
        "conversations": [
            {"from": "system", "value": system_val},
            {"from": "human",  "value": human_val},
            {"from": "gpt",    "value": output},
        ],
        "_profile_id": rec.get("profile_id"),
        "_model":      rec.get("model"),
        "_source_tag": rec.get("source_tag"),
        "_source":     rec.get("_source", ""),
    }

def to_trl(rec: dict) -> dict:
    """HuggingFace TRL SFTTrainer with apply_chat_template()."""
    instruction = rec.get("instruction", "")
    output      = rec.get("output", "")
    lines = instruction.split("\n", 1)
    if len(lines) == 2 and lines[0].startswith("["):
        system_val = lines[0].strip()
        user_val   = lines[1].strip()
    else:
        system_val = "You are a Telugu and Sanskrit scholar specialising in Dwipada poetry."
        user_val   = instruction.strip()
    return {
        "messages": [
            {"role": "system",    "content": system_val},
            {"role": "user",      "content": user_val},
            {"role": "assistant", "content": output},
        ],
        "_profile_id": rec.get("profile_id"),
        "_model":      rec.get("model"),
        "_source_tag": rec.get("source_tag"),
        "_source":     rec.get("_source", ""),
    }


# ──────────────────────────────────────────────
# STATS PRINTER
# ──────────────────────────────────────────────

def print_stats(records: list[dict], model_name: str, profiles: dict):
    total  = len(records)
    counts = {}
    for r in records:
        pid = r["profile_id"]
        counts[pid] = counts.get(pid, 0) + 1
    human = sum(1 for r in records if r["source_tag"] == "[Human_Style]")
    synth = total - human
    print(f"\n── {model_name.upper()} ({total:,} records) ─────────────────────────────")
    print(f"  {'PROFILE':<8} {'COUNT':>6}  {'%':>5}  DESCRIPTION")
    print(f"  {'─'*60}")
    for pid in sorted(counts):
        c    = counts[pid]
        pct  = c / total * 100
        desc = profiles.get(pid, {}).get("desc", "")[:45]
        print(f"  {pid:<8} {c:>6}  {pct:>4.1f}%  {desc}")
    print(f"  {'─'*60}")
    print(f"  Human: {human:,} ({human/total*100:.1f}%)  Synthetic: {synth:,} ({synth/total*100:.1f}%)")


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

def run(input_path: str, outdir: str):
    rng = random.Random(42)
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"\n[Loading] {input_path}")
    records = load_data(input_path)
    print(f"  Loaded {len(records):,} records.")

    # ── Build 4 model datasets ─────────────────
    datasets = {
        "analysis":   (ANALYSIS_PROFILES,   [build_analysis(r, rng)   for r in tqdm(records, desc="Analysis")]),
        "retrieval":  (RETRIEVAL_PROFILES,  [build_retrieval(r, rng)  for r in tqdm(records, desc="Retrieval")]),
        "generation": (GENERATION_PROFILES, [build_generation(r, rng) for r in tqdm(records, desc="Generation")]),
        "support":    (SUPPORT_PROFILES,    [build_support(r, rng)    for r in tqdm(records, desc="Support")]),
    }

    # ── Save all 3 formats per model ───────────
    print("\n[Saving — 3 formats × 4 models = 12 files...]")
    for model_name, (profiles, recs) in datasets.items():
        save_jsonl([to_alpaca(r)   for r in recs], str(out / f"ift_{model_name}_alpaca.jsonl"))
        save_jsonl([to_sharegpt(r) for r in recs], str(out / f"ift_{model_name}_sharegpt.jsonl"))
        save_jsonl([to_trl(r)      for r in recs], str(out / f"ift_{model_name}_trl.jsonl"))
        print_stats(recs, model_name, profiles)

    print(f"\n── Framework → File mapping ─────────────────────────────")
    print(f"  Unsloth / Axolotl alpaca   →  ift_<model>_alpaca.jsonl")
    print(f"  LLaMA-Factory / Axolotl    →  ift_<model>_sharegpt.jsonl")
    print(f"  HuggingFace TRL            →  ift_<model>_trl.jsonl")
    print(f"\n  Models: analysis · retrieval · generation · support")
    print(f"  SP4/SP5/SP6 excluded — require synthetic output with no ground truth.")
    print(f"\n✅ Done. 12 files written to {outdir}/")


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="METRICALARGS IFT Dataset Builder — outputs alpaca/sharegpt/trl per model"
    )
    parser.add_argument(
        "--input",  required=True,
        help="Path to POS-tagged corpus (poems_with_pos.jsonl)"
    )
    parser.add_argument(
        "--outdir", default=".",
        help="Output directory (default: current directory)"
    )
    args = parser.parse_args()
    run(args.input, args.outdir)