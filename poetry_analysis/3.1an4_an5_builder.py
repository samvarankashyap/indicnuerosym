"""
AN4 / AN5 Bilingual Dataset Builder
=====================================
Produces a single IFT-ready JSONL (TRL format) from the Dwipada corpus
covering ONLY two profiles:

  AN4 — Morphological gloss: sandhi breaks + word-by-word meaning
  AN5 — Full analysis: sandhi breaks + WTW meaning + Telugu bhavam

Each record is prompted in EITHER English or Telugu (chosen at random,
50/50 per profile), giving the model exposure to both instruction languages.

Multiple prompt templates per (profile × language) pair add surface
diversity so the model doesn't overfit to a single phrasing.

Output format: TRL (HuggingFace SFTTrainer / apply_chat_template)
  messages: [system, user, assistant]

Eligibility gates:
  AN4 — word_to_word_meaning with ≥ 3 entries (has_wtw_rich)
  AN5 — word_to_word_meaning with ≥ 3 entries AND telugu_meaning present

Usage:
    python an4_an5_builder.py --input poems_with_pos.jsonl
    python an4_an5_builder.py --input dwipada_augmented_dataset.json --output my_dataset.jsonl --seed 99
"""

import json
import random
import argparse
from pathlib import Path
from tqdm import tqdm


# ──────────────────────────────────────────────
# TOP-LEVEL CONSTANTS
# ──────────────────────────────────────────────

SYSTEM_PROMPT_EN = (
    "You are a Telugu and Sanskrit scholar specialising in classical Dwipada poetry. "
    "When asked to analyse a verse, decompose sandhi compounds using '+' notation "
    "and provide precise word-by-word glosses. Your output must be factual, "
    "structured, and grounded strictly in the given verse."
)

SYSTEM_PROMPT_TE = (
    "మీరు తెలుగు మరియు సంస్కృత పండితులు, ద్విపద కవిత్వంలో నిపుణులు. "
    "పద్యాన్ని విశ్లేషించమని అడిగినప్పుడు, సంధి పదాలను '+' గుర్తుతో విడదీయండి "
    "మరియు ప్రతి పదానికి ఖచ్చితమైన అర్థం ఇవ్వండి. మీ సమాధానం నిర్ణీతంగా, "
    "క్రమంగా మరియు కేవలం ఇచ్చిన పద్యం ఆధారంగా ఉండాలి."
)

# Language assignment (equal split)
LANGUAGES = ["en", "te"]

# Random seed (overridable via CLI)
DEFAULT_SEED = 42


# ──────────────────────────────────────────────
# IO HELPERS
# ──────────────────────────────────────────────

def load_data(path: str) -> list[dict]:
    """Accepts JSON array or JSONL."""
    with open(path, encoding="utf-8") as f:
        first = f.read(1)
        f.seek(0)
        if first == "[":
            return json.load(f)
        return [json.loads(line) for line in f if line.strip()]

def save_jsonl(records: list[dict], path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  ✓ {len(records):>6} records → {path}")


# ──────────────────────────────────────────────
# FIELD EXTRACTORS
# ──────────────────────────────────────────────

def _poem(rec):     return rec.get("poem", "").strip()
def _tel(rec):      return rec.get("telugu_meaning", "").strip()
def _src(rec):      return rec.get("source", "")
def _is_synth(rec): return rec.get("is_synthetic_data", True)
def _src_tag(rec):  return "[Synthetic]" if _is_synth(rec) else "[Human_Style]"

def _wtw(rec) -> str:
    """Format word_to_word_meaning dict as 'word: meaning' lines."""
    wtw = rec.get("word_to_word_meaning", {})
    if not wtw:
        return "Word breakdown not available."
    return "\n".join(f"{k}: {v}" for k, v in wtw.items())

def _has_wtw_rich(rec) -> bool:
    return len(rec.get("word_to_word_meaning", {})) >= 3

def _has_telugu_meaning(rec) -> bool:
    return bool(rec.get("telugu_meaning", "").strip())


# ──────────────────────────────────────────────
# PROMPT TEMPLATE POOLS
# ──────────────────────────────────────────────
# Each entry is a callable: rec → (user_prompt_str, output_str)
# Multiple templates per (profile, language) pair add surface diversity.

# ── AN4 — English prompts ──────────────────────────────────────────────────
AN4_EN = [
    lambda rec: (
        f"Break the sandhi compounds in the following Dwipada verse using '+' notation "
        f"and provide a word-by-word gloss for each resulting word.\n\n"
        f"Verse:\n{_poem(rec)}",
        f"Word-by-word gloss (pratipada-artha):\n{_wtw(rec)}"
    ),
    lambda rec: (
        f"Perform morphological analysis on the Dwipada verse below. "
        f"Decompose each sandhi compound with '+' and give the meaning of every constituent word.\n\n"
        f"Verse:\n{_poem(rec)}",
        f"Morphological gloss:\n{_wtw(rec)}"
    ),
    lambda rec: (
        f"For the classical Telugu verse given below, split all sandhi junctions "
        f"using '+' notation, then list the word-by-word (pratipada) meaning.\n\n"
        f"Verse:\n{_poem(rec)}",
        f"Pratipada-artha (word-by-word meaning):\n{_wtw(rec)}"
    ),
    lambda rec: (
        f"Analyse the following Dwipada verse: identify sandhi compounds, "
        f"separate them with '+', and provide the individual gloss for each word.\n\n"
        f"Verse:\n{_poem(rec)}",
        f"Sandhi analysis and word glosses:\n{_wtw(rec)}"
    ),
]

# ── AN4 — Telugu prompts ───────────────────────────────────────────────────
AN4_TE = [
    lambda rec: (
        f"క్రింది ద్విపద పద్యంలోని సంధి పదాలను '+' గుర్తుతో విడదీసి, "
        f"ప్రతి పదానికి అర్థం రాయండి.\n\n"
        f"పద్యం:\n{_poem(rec)}",
        f"ప్రతిపదార్థం:\n{_wtw(rec)}"
    ),
    lambda rec: (
        f"క్రింది పద్యంలో సంధి విభజన చేసి '+' గుర్తు వాడి, "
        f"విడిపోయిన ప్రతి పదానికి అర్థం వివరించండి.\n\n"
        f"పద్యం:\n{_poem(rec)}",
        f"పద విశ్లేషణ:\n{_wtw(rec)}"
    ),
    lambda rec: (
        f"ఈ ద్విపద పద్యంలోని సమాసాలను మరియు సంధులను '+' తో విడగొట్టి, "
        f"ప్రతిపదార్థం అందించండి.\n\n"
        f"పద్యం:\n{_poem(rec)}",
        f"ప్రతిపదార్థం:\n{_wtw(rec)}"
    ),
    lambda rec: (
        f"క్రింది తెలుగు ద్విపద పద్యాన్ని పదవిభజన చేయండి: "
        f"సంధి పదాలను '+' గుర్తుతో వేరు చేసి, ప్రతి పదం యొక్క అర్థాన్ని చెప్పండి.\n\n"
        f"పద్యం:\n{_poem(rec)}",
        f"పద విభజన మరియు అర్థాలు:\n{_wtw(rec)}"
    ),
]

# ── AN5 — English prompts ──────────────────────────────────────────────────
AN5_EN = [
    lambda rec: (
        f"Provide a complete linguistic analysis of the Dwipada verse below:\n"
        f"1. Decompose sandhi compounds using '+' notation\n"
        f"2. Give the word-by-word (pratipada) gloss\n"
        f"3. Provide the overall meaning in Telugu (telugu bhavam)\n\n"
        f"Verse:\n{_poem(rec)}",
        f"Word-by-word gloss (pratipada-artha):\n{_wtw(rec)}\n\n"
        f"Telugu meaning (bhavam): {_tel(rec)}"
    ),
    lambda rec: (
        f"Analyse the following classical Telugu verse in three steps: "
        f"(a) break sandhi compounds with '+', "
        f"(b) list the individual word meanings, "
        f"(c) give the full meaning in Telugu.\n\n"
        f"Verse:\n{_poem(rec)}",
        f"Pratipada-artha:\n{_wtw(rec)}\n\n"
        f"Telegu bhavam: {_tel(rec)}"
    ),
    lambda rec: (
        f"For the Dwipada verse below, perform morphological decomposition "
        f"(sandhi splits with '+'), word-level glossing, and provide the "
        f"overall Telugu interpretation.\n\n"
        f"Verse:\n{_poem(rec)}",
        f"Morphological gloss:\n{_wtw(rec)}\n\n"
        f"Meaning in Telugu: {_tel(rec)}"
    ),
    lambda rec: (
        f"Do a full textual analysis of this Dwipada verse:\n"
        f"— Split all sandhi using '+'\n"
        f"— Gloss each word separately\n"
        f"— State the overall meaning in Telugu\n\n"
        f"Verse:\n{_poem(rec)}",
        f"Word-by-word meaning:\n{_wtw(rec)}\n\n"
        f"Telugu bhavam: {_tel(rec)}"
    ),
]

# ── AN5 — Telugu prompts ───────────────────────────────────────────────────
AN5_TE = [
    lambda rec: (
        f"క్రింది పద్యానికి సంపూర్ణ విశ్లేషణ చేయండి: "
        f"సంధి విభజన, ప్రతిపదార్థం మరియు తెలుగు భావం ఇవ్వండి.\n\n"
        f"పద్యం:\n{_poem(rec)}",
        f"ప్రతిపదార్థం:\n{_wtw(rec)}\n\n"
        f"తెలుగు భావం: {_tel(rec)}"
    ),
    lambda rec: (
        f"క్రింది ద్విపద పద్యాన్ని మూడు దశల్లో విశ్లేషించండి:\n"
        f"౧. సంధి విభజన ('+' గుర్తు వాడి)\n"
        f"౨. ప్రతిపదార్థం\n"
        f"౩. తెలుగు భావం\n\n"
        f"పద్యం:\n{_poem(rec)}",
        f"ప్రతిపదార్థం:\n{_wtw(rec)}\n\n"
        f"తెలుగు భావం: {_tel(rec)}"
    ),
    lambda rec: (
        f"ఈ తెలుగు పద్యానికి సమగ్ర వ్యాఖ్యానం ఇవ్వండి — "
        f"సంధి విభజన, పద విశ్లేషణ మరియు మొత్తం భావం తెలుగులో వివరించండి.\n\n"
        f"పద్యం:\n{_poem(rec)}",
        f"పద విశ్లేషణ:\n{_wtw(rec)}\n\n"
        f"తెలుగు భావం: {_tel(rec)}"
    ),
    lambda rec: (
        f"క్రింది ద్విపద పద్యంలోని సంధులను '+' తో విడగొట్టి, "
        f"ప్రతిపదార్థం చెప్పి, తెలుగు భావం అందించండి.\n\n"
        f"పద్యం:\n{_poem(rec)}",
        f"ప్రతిపదార్థం:\n{_wtw(rec)}\n\n"
        f"తెలుగు భావం: {_tel(rec)}"
    ),
]

# Map: profile_id → {lang → [template_fns], eligibility_fn}
PROFILES = {
    "AN4": {
        "templates": {"en": AN4_EN, "te": AN4_TE},
        "eligible":  _has_wtw_rich,
        "desc":      "Sandhi decomposition + word-by-word gloss",
    },
    "AN5": {
        "templates": {"en": AN5_EN, "te": AN5_TE},
        "eligible":  lambda rec: _has_wtw_rich(rec) and _has_telugu_meaning(rec),
        "desc":      "Sandhi + WTW gloss + Telugu bhavam",
    },
}


# ──────────────────────────────────────────────
# RECORD BUILDER
# ──────────────────────────────────────────────

def build_trl_record(
    rec: dict,
    profile_id: str,
    lang: str,
    instruction: str,
    output: str,
) -> dict:
    """Returns a TRL-format record (messages list + metadata)."""
    system = SYSTEM_PROMPT_EN if lang == "en" else SYSTEM_PROMPT_TE
    return {
        "messages": [
            {"role": "system",    "content": system},
            {"role": "user",      "content": instruction},
            {"role": "assistant", "content": output},
        ],
        "_profile_id":  f"{profile_id}_{lang}",
        "_model":       "analysis",
        "_source_tag":  _src_tag(rec),
        "_is_synthetic": _is_synth(rec),
        "_source":      _src(rec),
    }


# ──────────────────────────────────────────────
# MAIN BUILD LOOP
# ──────────────────────────────────────────────

def build_dataset(records: list[dict], rng: random.Random) -> list[dict]:
    """
    For each record, attempt AN5 first (stricter gate).
    If ineligible for AN5, fall back to AN4.
    If ineligible for both, skip.

    Language is chosen randomly (50/50) per record.
    Template is chosen randomly from the pool for that (profile, lang).
    """
    dataset = []
    skipped = 0

    for rec in tqdm(records, desc="Building AN4/AN5 dataset"):
        # Determine which profile to assign
        an5_ok = PROFILES["AN5"]["eligible"](rec)
        an4_ok = PROFILES["AN4"]["eligible"](rec)

        if an5_ok:
            # Eligible for both — randomly assign with slight AN5 preference (55/45)
            profile_id = rng.choices(["AN5", "AN4"], weights=[0.55, 0.45], k=1)[0]
        elif an4_ok:
            profile_id = "AN4"
        else:
            skipped += 1
            continue

        lang = rng.choice(LANGUAGES)
        templates = PROFILES[profile_id]["templates"][lang]
        fn = rng.choice(templates)

        instruction, output = fn(rec)
        dataset.append(build_trl_record(rec, profile_id, lang, instruction, output))

    return dataset, skipped


# ──────────────────────────────────────────────
# STATS PRINTER
# ──────────────────────────────────────────────

def print_stats(records: list[dict], skipped: int, total_input: int):
    total = len(records)
    print(f"\n{'═'*62}")
    print(f"  AN4/AN5 BILINGUAL DATASET — {total:,} records")
    print(f"  Skipped (ineligible): {skipped:,} / {total_input:,}")
    print(f"{'═'*62}")

    # Count per (profile, lang)
    counts: dict[str, int] = {}
    for r in records:
        pid = r["_profile_id"]
        counts[pid] = counts.get(pid, 0) + 1

    print(f"\n  {'PROFILE+LANG':<18} {'COUNT':>6}  {'%':>5}  DESCRIPTION")
    print(f"  {'─'*55}")
    for key in sorted(counts):
        c   = counts[key]
        pct = c / total * 100
        profile = key.split("_")[0]
        lang    = key.split("_")[1].upper()
        desc    = PROFILES[profile]["desc"]
        print(f"  {key:<18} {c:>6}  {pct:>4.1f}%  [{lang}] {desc}")

    # Language split totals
    en_total = sum(c for k, c in counts.items() if k.endswith("_en"))
    te_total = sum(c for k, c in counts.items() if k.endswith("_te"))
    print(f"\n  Language split — EN: {en_total:,} ({en_total/total*100:.1f}%)  "
          f"TE: {te_total:,} ({te_total/total*100:.1f}%)")

    human = sum(1 for r in records if r["_source_tag"] == "[Human_Style]")
    synth = total - human
    print(f"  Source split   — Human: {human:,} ({human/total*100:.1f}%)  "
          f"Synthetic: {synth:,} ({synth/total*100:.1f}%)")
    print(f"{'═'*62}\n")


# ──────────────────────────────────────────────
# ENTRYPOINT
# ──────────────────────────────────────────────

def run(input_path: str, output_path: str, seed: int):
    rng = random.Random(seed)

    print(f"\n[Loading] {input_path}")
    records = load_data(input_path)
    total_input = len(records)
    print(f"  Loaded {total_input:,} records.")

    print(f"\n[Building] Profiles: AN4, AN5 | Languages: EN, TE | Seed: {seed}")
    dataset, skipped = build_dataset(records, rng)

    print(f"\n[Saving]")
    save_jsonl(dataset, output_path)
    print_stats(dataset, skipped, total_input)
    print(f"✅ Done → {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AN4/AN5 bilingual IFT dataset builder (TRL format)"
    )
    parser.add_argument(
        "--input",  required=True,
        help="POS-tagged corpus (poems_with_pos.jsonl or raw JSON array)"
    )
    parser.add_argument(
        "--output", default="ift_an4_an5_trl.jsonl",
        help="Output JSONL path (default: ift_an4_an5_trl.jsonl)"
    )
    parser.add_argument(
        "--seed",   type=int, default=DEFAULT_SEED,
        help=f"Random seed (default: {DEFAULT_SEED})"
    )
    args = parser.parse_args()
    run(args.input, args.output, args.seed)