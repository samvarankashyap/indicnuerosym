"""
Dwipada Training Data Pipeline
================================
Step 1: POS Tagging  — adds tags_english, tags_telugu to every record
Step 2: Prompt Transformation — generates 10 prompt-profile variants per poem
Step 3: Curriculum Split — separates human vs synthetic for Stage 1/2 training

Requirements:
    pip install spacy stanza tqdm
    python -m spacy download en_core_web_sm
    python -c "import stanza; stanza.download('te')"

Usage:
    python dwipada_pipeline.py \
        --input  dwipada_augmented_dataset.json \
        --output training_ready.jsonl \
        --pos_output poems_with_pos.jsonl
"""

import json
import random
import argparse
from pathlib import Path
from tqdm import tqdm

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
        # processors='tokenize,pos' — skip NER/dependency for speed
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
    """
    Returns list of {token, pos} dicts for content words only.
    Filters out PUNCT, SPACE, DET — keeps NOUN, VERB, ADJ, ADV etc.
    """
    KEEP_POS = {"NOUN", "VERB", "ADJ", "ADV", "PROPN", "NUM"}
    doc = nlp_en(text)
    return [
        {"token": token.text, "pos": token.pos_}
        for token in doc
        if token.pos_ in KEEP_POS and not token.is_stop
    ]

def tag_telugu(text: str, nlp_te) -> list[dict]:
    """
    Returns list of {token, pos, xpos} dicts.
    xpos = language-specific tag (e.g. NN, VM, JJ in Paninian scheme).
    """
    KEEP_UPOS = {"NOUN", "VERB", "ADJ", "ADV", "PROPN", "NUM"}
    doc = nlp_te(text)
    results = []
    for sentence in doc.sentences:
        for word in sentence.words:
            if word.upos in KEEP_UPOS:
                results.append({
                    "token": word.text,
                    "pos":   word.upos,       # Universal POS
                    "xpos":  word.xpos        # Paninian POS tag
                })
    return results

def add_pos_tags(records: list[dict], nlp_en, nlp_te) -> list[dict]:
    """
    Enriches every record with tags_english and tags_telugu.
    Works from english_meaning and telugu_meaning fields.
    """
    print("\n[Step 1] Adding POS tags...")
    for rec in tqdm(records, desc="POS Tagging"):
        eng_text = rec.get("english_meaning", "")
        tel_text = rec.get("telugu_meaning", "")

        rec["tags_english"] = tag_english(eng_text, nlp_en) if eng_text else []
        rec["tags_telugu"]  = tag_telugu(tel_text, nlp_te)  if tel_text else []
    return records


# ──────────────────────────────────────────────
# SECTION 2 ▸ THE 10 PROMPT PROFILES
# ──────────────────────────────────────────────
# Each profile defines:
#   - system_tag : persona hint injected into the instruction
#   - instruction: the user-facing prompt template
#   - build_output: function(record) -> str  (what the model should respond)
# ──────────────────────────────────────────────

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

# ── Profile builders ──────────────────────────

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
    # For debugger: we intentionally corrupt Prasa in Line 1 and ask model to fix.
    # The *correct* output is the real poem + explanation.
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
    # Strict JSON only — no prose
    payload = {
        "prathipadartham":  rec.get("word_to_word_meaning", {}),
        "bhavam_te":        rec.get("telugu_meaning", ""),
        "bhavam_en":        rec.get("english_meaning", ""),
        "is_valid_dwipada": _is_valid_dwipada(rec)
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)

def build_profile_10(rec):  # Multi-Gana Variety (detailed scansion)
    out  = f"Detailed Gana Scansion:\n{_gana_breakdown(rec)}\n\n"
    out += f"Poem:\n{rec.get('poem','')}"
    return out


PROFILES = {
    1:  {
        "weight": 0.14,   # ~40% educational bucket shared across 1,6,9
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
        "weight": 0.02,   # 10% edge-case
        "system_tag": "[GanaVariety]",
        "instruction": (
            "Provide a detailed rhythmic scansion of this Dwipada, identifying "
            "each Gana type used. I want to see the diverse rhythmic structure "
            "followed by the poem text."
        ),
        "builder": build_profile_10
    },
}

# Verify weights sum to ~1.0
assert abs(sum(p["weight"] for p in PROFILES.values()) - 1.0) < 0.01, \
    "Profile weights must sum to 1.0"

# Profiles that rely on word_to_word_meaning being rich
WORD_BREAKDOWN_PROFILES = {1, 4, 7}

# Profiles that are safe for thin/incomplete word_to_word_meaning
THIN_WTW_SAFE_PROFILES  = {3, 6, 9}

# Threshold: records with fewer than this many WTW entries are "thin"
WTW_RICHNESS_THRESHOLD  = 3


def get_eligible_profiles(rec: dict) -> tuple[list[int], list[float]]:
    """
    Returns (profile_ids, weights) filtered by the record's data richness.

    Rule:
      - If word_to_word_meaning has < WTW_RICHNESS_THRESHOLD entries,
        restrict to THIN_WTW_SAFE_PROFILES (3, 6, 9) — no word-breakdown outputs.
      - Otherwise all 10 profiles are eligible with their original weights.

    Weights are renormalized after filtering so rng.choices() stays valid.
    """
    wtw_len = len(rec.get("word_to_word_meaning", {}))
    is_thin = wtw_len < WTW_RICHNESS_THRESHOLD

    if is_thin:
        eligible_ids = sorted(THIN_WTW_SAFE_PROFILES)
    else:
        eligible_ids = list(PROFILES.keys())

    raw_weights = [PROFILES[pid]["weight"] for pid in eligible_ids]
    total       = sum(raw_weights)
    norm_weights = [w / total for w in raw_weights]   # renormalize to sum=1

    return eligible_ids, norm_weights


def assign_prompt_profile(rec: dict, rng: random.Random,
                          wtw_threshold: int = WTW_RICHNESS_THRESHOLD) -> dict:
    """
    Assigns a profile to a record based on data richness.

    - Thin records (word_to_word_meaning < wtw_threshold entries) are
      restricted to profiles 3, 6, 9 (poem/meaning-only — no word breakdown).
    - Rich records are assigned from all 10 profiles using original weights.
    - A 'thin_wtw' flag is stored on the record for downstream inspection.
    """
    wtw_len  = len(rec.get("word_to_word_meaning", {}))
    is_thin  = wtw_len < wtw_threshold

    eligible_ids, norm_weights = get_eligible_profiles(rec)
    pid     = rng.choices(eligible_ids, weights=norm_weights, k=1)[0]
    profile = PROFILES[pid]

    source_tag  = "[Human_Style]" if not rec.get("is_synthetic_data", True) else "[Synthetic]"

    instruction = f"{source_tag} {profile['system_tag']}\n{profile['instruction']}"
    poem_block  = rec.get("poem", "")

    # Full input that will be seen at inference:
    full_input  = f"{instruction}\n\n{poem_block}"
    output      = profile["builder"](rec)

    return {
        **rec,
        "profile_id":  pid,
        "input":       full_input,
        "output":      output,
        "source_tag":  source_tag,
        "thin_wtw":    is_thin,    # True if word_to_word_meaning was below threshold
        "wtw_len":     wtw_len,    # Actual entry count — useful for auditing
    }


# ──────────────────────────────────────────────
# SECTION 3 ▸ CURRICULUM SPLIT
# ──────────────────────────────────────────────

def split_curriculum(records: list[dict]) -> tuple[list, list]:
    """
    Stage 1 (Structural Tutor)  → synthetic only
    Stage 2 (Artistic Master)   → human (3x upsampled) + 10% synthetic rehearsal
    """
    human     = [r for r in records if not r.get("is_synthetic_data", True)]
    synthetic = [r for r in records if r.get("is_synthetic_data", True)]

    print(f"\n[Step 3] Curriculum Split:")
    print(f"  Human poems    : {len(human)}")
    print(f"  Synthetic poems: {len(synthetic)}")

    stage1 = synthetic                           # raw synthetic — teach the math
    stage2 = human * 3 + synthetic[:len(synthetic)//10]   # 3x human + 10% rehearsal

    random.shuffle(stage2)
    print(f"  Stage-1 size   : {len(stage1)}")
    print(f"  Stage-2 size   : {len(stage2)}")
    return stage1, stage2


# ──────────────────────────────────────────────
# SECTION 4 ▸ MAIN PIPELINE
# ──────────────────────────────────────────────

def load_jsonl(path: str) -> list[dict]:
    """
    Transparently loads either:
      - A JSON array file  (.json)  → list parsed directly
      - A JSONL file       (.jsonl) → one record per line
    No hardcoded record counts; works for any dataset size.
    """
    with open(path, encoding="utf-8") as f:
        first_char = f.read(1)
        f.seek(0)
        if first_char == "[":
            # JSON array format (e.g. dwipada_augmented_dataset.json)
            data = json.load(f)
            if not isinstance(data, list):
                raise ValueError(f"{path} is a JSON object, expected a list/array.")
            return data
        else:
            # JSONL format — one record per line
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

def run_pipeline(input_path: str, output_path: str, pos_output_path: str):
    rng = random.Random(42)

    # Load
    print(f"\n[Loading] {input_path}")
    records = load_jsonl(input_path)
    print(f"  Loaded {len(records)} records.")

    # Step 1: POS Tagging
    nlp_en = load_spacy_english()
    nlp_te = load_stanza_telugu()
    records = add_pos_tags(records, nlp_en, nlp_te)

    # Save POS-enriched version (useful checkpoint)
    save_jsonl(records, pos_output_path)
    print(f"\n[Checkpoint] POS-tagged data saved to {pos_output_path}")

    # Step 2: Assign Prompt Profiles
    print("\n[Step 2] Assigning prompt profiles...")
    transformed = [assign_prompt_profile(rec, rng) for rec in tqdm(records, desc="Prompting")]

    # Step 3: Curriculum split
    stage1, stage2 = split_curriculum(transformed)

    # Save outputs
    print("\n[Saving outputs...]")
    stem   = Path(output_path).stem
    parent = Path(output_path).parent

    save_jsonl(transformed, output_path)
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

    # Thin-record audit
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
    print("\n✅ Pipeline complete.")


# ──────────────────────────────────────────────
# SECTION 5 ▸ CLI
# ──────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dwipada Training Data Pipeline")
    parser.add_argument(
        "--input",  required=True,
        help="Path to input dataset — JSON array or JSONL (e.g. dwipada_augmented_dataset.json)"
    )
    parser.add_argument(
        "--output", required=True,
        help="Path for final training-ready JSONL (e.g. training_ready.jsonl)"
    )
    parser.add_argument(
        "--pos_output", default="poems_with_pos.jsonl",
        help="Checkpoint path for POS-tagged data (default: poems_with_pos.jsonl)"
    )
    args = parser.parse_args()
    run_pipeline(args.input, args.output, args.pos_output)