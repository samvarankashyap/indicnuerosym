"""
Profile Re-assignment Script
==============================
Works directly on training_ready.jsonl — skips POS tagging entirely.

If tags_telugu is missing or empty, G8/G9/G11 (keyword-constraint profiles)
fall back gracefully to default keywords extracted from word_to_word_meaning keys.

Outputs:
  - training_ready_v2.jsonl          (all records, new profiles assigned)
  - training_ready_v2_stage1.jsonl   (synthetic only — curriculum stage 1)
  - training_ready_v2_stage2.jsonl   (3x human + 10% synthetic rehearsal)

Usage:
    python reassign_profiles.py --input training_ready.jsonl
    python reassign_profiles.py --input training_ready.jsonl --outdir ./v2
"""

import json
import random
import re
import argparse
from pathlib import Path
from tqdm import tqdm


# ──────────────────────────────────────────────
# HELPERS
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

def audit_tags(records: list[dict]) -> dict:
    """Check how many records have tags_telugu populated."""
    total       = len(records)
    has_tags    = sum(1 for r in records if r.get("tags_telugu"))
    empty_tags  = total - has_tags
    return {
        "total":      total,
        "has_tags":   has_tags,
        "empty_tags": empty_tags,
        "pct_tagged": has_tags / total * 100 if total else 0,
    }


# ──────────────────────────────────────────────
# OUTPUT BUILDERS  (shared across profiles)
# ──────────────────────────────────────────────

def _gana_breakdown(rec):
    ca = rec.get("chandassu_analysis", {})
    parts = []
    for k, v in ca.items():
        parts.append(
            f"{k}: {v.get('breakdown','')} | "
            f"Yati: {v.get('yati_check','')} | "
            f"Prasa: {v.get('prasa_check','')}"
        )
    return "\n".join(parts) or "Analysis not available."

def _word_breakdown(rec):
    wtw = rec.get("word_to_word_meaning", {})
    if not wtw:
        return "Word breakdown not available."
    return "\n".join(f"{k}: {v}" for k, v in wtw.items())

def _is_valid(rec):
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
        # Fallback: use first 3 WTW keys, strip sandhi markers
        wtw_keys = list(rec.get("word_to_word_meaning", {}).keys())[:3]
        tokens = [k.split("+")[0].strip() for k in wtw_keys if k.strip()]
    return tokens

def _line1(rec):
    lines = rec.get("poem", "").split("\n")
    return lines[0].strip() if lines else ""

# ── output combinations ────────────────────────
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
# GENERATION PROFILE BUILDERS  (G1–G12)
# ──────────────────────────────────────────────

def g1(rec):   # theme only → poem + WTW
    topic = rec.get("english_meaning", "")[:60].rstrip(",. ")
    return (
        f"Write a Dwipada poem in Telugu about the following theme.\n"
        f"Ensure it follows the classical Dwipada meter: 3 Indra Ganas + 1 Surya Gana per line.\n"
        f"Provide the poem followed by word-by-word meaning (ప్రతిపదార్థం).\n\n"
        f"Theme: {topic}"
    ), out_poem_wtw(rec)

def g2(rec):   # theme + Prasa → poem + WTW + proof
    topic = rec.get("english_meaning", "")[:60].rstrip(",. ")
    return (
        f"Compose a Dwipada poem in Telugu on the theme: {topic}\n"
        f"Constraint: The Prasa letter (2nd letter of each line) must be '{_extract_prasa(rec)}'.\n"
        f"Provide the poem, ప్రతిపదార్థం, and Chandassu analysis proving the Prasa rule is met."
    ), out_poem_wtw_chandassu(rec)

def g3(rec):   # theme + Yati → poem + proof
    topic = rec.get("english_meaning", "")[:60].rstrip(",. ")
    return (
        f"Write a Telugu Dwipada poem about: {topic}\n"
        f"The Yati (caesura) letters must be: {_extract_yati(rec)}.\n"
        f"Provide the poem and a full Chandassu breakdown confirming the Yati placement."
    ), out_poem_chandassu(rec)

def g4(rec):   # theme + Prasa + Yati → full
    topic = rec.get("english_meaning", "")[:60].rstrip(",. ")
    return (
        f"Compose a classical Telugu Dwipada on the theme: {topic}\n"
        f"Constraints:\n"
        f"  - Prasa letter: '{_extract_prasa(rec)}'\n"
        f"  - Yati letters: {_extract_yati(rec)}\n"
        f"  - Meter: 3 Indra Ganas + 1 Surya Gana per line\n"
        f"Output the poem, ప్రతిపదార్థం, and Chandassu proof."
    ), out_poem_wtw_chandassu(rec)

def g5(rec):   # English meaning → poem + WTW
    return (
        f"The following is the meaning of a Telugu Dwipada poem in English.\n"
        f"Generate the Telugu Dwipada poem that expresses this meaning.\n"
        f"Follow the Dwipada meter strictly. Provide the poem and ప్రతిపదార్థం.\n\n"
        f"English Meaning: {rec.get('english_meaning','')}"
    ), out_poem_wtw(rec)

def g6(rec):   # Telugu meaning → poem + WTW
    return (
        f"క్రింది తెలుగు భావానికి అనుగుణంగా ఒక ద్విపద పద్యం రచించండి.\n"
        f"ద్విపద ఛందస్సు నియమాలు పాటించండి: ప్రతి పాదంలో 3 ఇంద్ర గణాలు + 1 సూర్య గణం.\n"
        f"పద్యం మరియు ప్రతిపదార్థం ఇవ్వండి.\n\n"
        f"తెలుగు భావం: {rec.get('telugu_meaning','')}"
    ), out_poem_wtw(rec)

def g7(rec):   # line 1 → complete line 2
    lines = rec.get("poem", "").split("\n")
    line2 = lines[1].strip() if len(lines) > 1 else ""
    return (
        f"The following is the first line of a Telugu Dwipada poem.\n"
        f"Complete the poem by writing the second line.\n"
        f"The second line must maintain the same Prasa (2nd letter match) and meter.\n\n"
        f"First line: {_line1(rec)}"
    ), f"రెండవ పాదం:\n{line2}"

def g8(rec):   # theme + noun keywords → poem + WTW
    topic = rec.get("english_meaning", "")[:60].rstrip(",. ")
    nouns = _pos_tokens(rec, ["NOUN", "PROPN"])[:4]
    kw    = ", ".join(nouns) if nouns else "వాయువు, శరీరం"
    return (
        f"Write a Telugu Dwipada poem about: {topic}\n"
        f"The poem must incorporate these Telugu nouns/concepts: {kw}\n"
        f"Follow the Dwipada meter. Provide the poem and ప్రతిపదార్థం."
    ), out_poem_wtw(rec)

def g9(rec):   # theme + adj/adv constraints → poem + WTW
    topic = rec.get("english_meaning", "")[:60].rstrip(",. ")
    quals = _pos_tokens(rec, ["ADJ", "ADV"])[:3]
    kw    = ", ".join(quals) if quals else "మహిమాన్వితమైన, శాశ్వతమైన"
    return (
        f"Compose a Telugu Dwipada on the theme: {topic}\n"
        f"Use these descriptive qualities in the poem: {kw}\n"
        f"Strict Dwipada meter required. Provide poem and ప్రతిపదార్థం."
    ), out_poem_wtw(rec)

def g10(rec):  # English meaning + Prasa → poem + proof
    return (
        f"Using the following English meaning as your guide, compose a Telugu Dwipada.\n"
        f"Hard constraint: Prasa letter must be '{_extract_prasa(rec)}'.\n"
        f"Output the poem and Chandassu analysis proving the constraint is satisfied.\n\n"
        f"English Meaning: {rec.get('english_meaning','')}"
    ), out_poem_chandassu(rec)

def g11(rec):  # full constraints → full output
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

def g12(rec):  # Telugu meaning + keyword hints → poem only
    nouns = _pos_tokens(rec, ["NOUN", "PROPN"])[:3]
    hint  = f"\nKeyword hints: {', '.join(nouns)}" if nouns else ""
    return (
        f"క్రింది భావానికి తగిన ద్విపద పద్యం రచించండి.{hint}\n"
        f"కేవలం పద్యం మాత్రమే ఇవ్వండి.\n\n"
        f"భావం: {rec.get('telugu_meaning','')}"
    ), out_poem_only(rec)


# ──────────────────────────────────────────────
# ANALYSIS PROFILE BUILDERS  (A1–A10)
# ──────────────────────────────────────────────

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
    lines    = rec.get("poem", "").split("\n")
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
        "is_valid_dwipada": _is_valid(rec),
    }, ensure_ascii=False, indent=2)

def a10(rec):
    return f"Detailed Gana Scansion:\n{_gana_breakdown(rec)}\n\nPoem:\n{rec.get('poem','')}"


# ──────────────────────────────────────────────
# PROFILE REGISTRY
# ──────────────────────────────────────────────
# Generation G1–G12 : 70%  (0.0583 each)
# Analysis   A1–A10 : 30%  (0.030  each)

PROFILES = {
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

WTW_THRESHOLD   = 3
THIN_SAFE_IDS   = {pid for pid, p in PROFILES.items() if not p["needs_wtw"]}
ALL_IDS         = list(PROFILES.keys())
ALL_WEIGHTS     = [PROFILES[pid]["w"] for pid in ALL_IDS]
THIN_IDS        = sorted(THIN_SAFE_IDS)
THIN_WEIGHTS_RAW = [PROFILES[pid]["w"] for pid in THIN_IDS]
THIN_TOTAL      = sum(THIN_WEIGHTS_RAW)
THIN_WEIGHTS    = [w / THIN_TOTAL for w in THIN_WEIGHTS_RAW]


def assign(rec: dict, rng: random.Random) -> dict:
    wtw_len = len(rec.get("word_to_word_meaning", {}))
    is_thin = wtw_len < WTW_THRESHOLD

    if is_thin:
        pid = rng.choices(THIN_IDS, weights=THIN_WEIGHTS, k=1)[0]
    else:
        pid = rng.choices(ALL_IDS, weights=ALL_WEIGHTS, k=1)[0]

    p          = PROFILES[pid]
    src_tag    = rec.get("source_tag", "[Human_Style]")   # preserve existing tag

    if p["type"] == "gen":
        instruction, output = p["fn"](rec)
        full_input = f"{src_tag} {p['tag']}\n{instruction}"
    else:
        output    = p["fn"](rec)
        poem      = rec.get("poem", "")
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
# CURRICULUM SPLIT
# ──────────────────────────────────────────────

def curriculum_split(records):
    synthetic = [r for r in records if r.get("is_synthetic_data", True)]
    human     = [r for r in records if not r.get("is_synthetic_data", True)]
    stage2    = human * 3 + synthetic[:max(1, len(synthetic) // 10)]
    random.shuffle(stage2)
    return synthetic, stage2


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

def run(input_path: str, outdir: str):
    rng = random.Random(42)
    out = Path(outdir)

    # ── Load ──────────────────────────────────
    print(f"\n[Loading] {input_path}")
    records = load_data(input_path)
    print(f"  Loaded {len(records)} records.")

    # ── Audit tags_telugu ─────────────────────
    audit = audit_tags(records)
    print(f"\n[Audit] tags_telugu:")
    print(f"  Populated : {audit['has_tags']:>6} ({audit['pct_tagged']:.1f}%)")
    print(f"  Empty     : {audit['empty_tags']:>6}")
    if audit["pct_tagged"] < 50:
        print("  ⚠  Less than 50% have tags_telugu — G8/G9/G11 will use WTW-key fallback.")
    else:
        print("  ✓ tags_telugu looks good — keyword profiles will use POS tokens.")

    # ── Re-assign profiles ────────────────────
    print("\n[Re-assigning profiles...]")
    transformed = [assign(r, rng) for r in tqdm(records)]

    # ── Curriculum split ──────────────────────
    stage1, stage2 = curriculum_split(transformed)

    # ── Save ──────────────────────────────────
    stem = Path(input_path).stem
    print("\n[Saving...]")
    save_jsonl(transformed, str(out / f"{stem}_v2.jsonl"))
    save_jsonl(stage1,      str(out / f"{stem}_v2_stage1.jsonl"))
    save_jsonl(stage2,      str(out / f"{stem}_v2_stage2.jsonl"))

    # ── Stats ─────────────────────────────────
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
        print(f"  {PROFILES[pid]['type'].upper():<6} {pid:<8} {c:>6}  {pct:>4.1f}%  {bar}")
    print(f"  {'─'*56}")
    print(f"  {'GEN':<6} {'ALL':<8} {gen_c:>6}  {gen_c/total*100:>4.1f}%")
    print(f"  {'ANA':<6} {'ALL':<8} {ana_c:>6}  {ana_c/total*100:>4.1f}%")
    print(f"\n  Thin WTW records : {thin_c} ({thin_c/total*100:.1f}%)")
    print(f"  Stage-1 (synth)  : {len(stage1)}")
    print(f"  Stage-2 (human)  : {len(stage2)}")
    print("\n✅ Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Re-assign generation + analysis profiles")
    parser.add_argument("--input",  required=True, help="Path to training_ready.jsonl")
    parser.add_argument("--outdir", default=".",   help="Output directory (default: .)")
    args = parser.parse_args()
    run(args.input, args.outdir)