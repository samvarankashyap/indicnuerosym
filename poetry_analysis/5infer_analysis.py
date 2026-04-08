"""
METRICALARGS — Base vs Fine-tuned Inference + Comparison
==========================================================
Runs BOTH the base model and the fine-tuned model on the same test
samples and saves results in formats ready for LLM-as-a-judge evaluation.

Output files (saved locally in ./results/):
  finetuned/infer_results_raw.jsonl      — fine-tuned model full records
  finetuned/infer_results_judge.jsonl    — fine-tuned model judge-ready
  finetuned/infer_results_summary.txt    — fine-tuned model human-readable

  base/infer_results_raw.jsonl           — base model full records
  base/infer_results_judge.jsonl         — base model judge-ready
  base/infer_results_summary.txt         — base model human-readable

  comparison/infer_comparison.jsonl      — both outputs side-by-side per sample
  comparison/infer_comparison.txt        — human-readable side-by-side

Usage:
    modal run infer_analysis.py                     # both models, test set
    modal run infer_analysis.py --mode seen         # seen training samples
    modal run infer_analysis.py --mode new          # hardcoded new verses
    modal run infer_analysis.py --model finetuned   # fine-tuned only
    modal run infer_analysis.py --model base        # base only
"""

import json
import os
import re
import modal
from pathlib import Path
from modal import App, Image as ModalImage, Secret

app = App("metricalargs-inference")

huggingface_secret = Secret.from_name("huggingface")

# =============================================================================
# CONFIGURATION
# =============================================================================

FINETUNED_MODEL = "maheshemani/dwipada-gemma4-E4B-analysis"
BASE_MODEL      = "google/gemma-4-E4B-it"

TEST_JSONL      = "ift_data/ift_an4_an5_test.jsonl"
RESULTS_DIR     = "results"
MAX_NEW_TOKENS  = 512
TEMPERATURE     = 0.1

INFERENCE_IMAGE = (
    ModalImage.debian_slim(python_version="3.11")
    .apt_install("git", "build-essential")
    .run_commands(
        "pip install --upgrade unsloth unsloth_zoo bitsandbytes accelerate "
        "sentencepiece protobuf "
        "--extra-index-url https://download.pytorch.org/whl/cu126",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

HOURS = 60 * 60

# =============================================================================
# SYSTEM PROMPTS
# =============================================================================

SYSTEM_EN = (
    "You are a Telugu and Sanskrit scholar specialising in classical Dwipada poetry. "
    "When asked to analyse a verse, decompose sandhi compounds using '+' notation "
    "and provide precise word-by-word glosses. Your output must be factual, "
    "structured, and grounded strictly in the given verse."
)
SYSTEM_TE = (
    "మీరు తెలుగు మరియు సంస్కృత పండితులు, ద్విపద కవిత్వంలో నిపుణులు. "
    "పద్యాన్ని విశ్లేషించమని అడిగినప్పుడు, సంధి పదాలను '+' గుర్తుతో విడదీయండి "
    "మరియు ప్రతి పదానికి ఖచ్చితమైన అర్థం ఇవ్వండి."
)

# =============================================================================
# HARDCODED SAMPLES (for --mode seen / --mode new)
# =============================================================================

SEEN_SAMPLES = [
    {
        "label":     "AN4_en — seen",
        "profile":   "AN4",
        "lang":      "en",
        "seen_as":   "training",
        "verse":     "భువనత్రయాధారభూతమయుండు \nపవనుండు లేకున్న బడు శరీరములు",
        "prompt": (
            "Break the sandhi compounds in the following Dwipada verse using '+' "
            "notation and provide a word-by-word gloss for each resulting word.\n\n"
            "Verse:\nభువనత్రయాధారభూతమయుండు \nపవనుండు లేకున్న బడు శరీరములు"
        ),
        "reference": (
            "Pratipada-artha (word-by-word meaning):\n"
            "భువనత్రయ + ఆధార + భూతమయుండు: మూడు లోకములకు ఆధారమైన పంచభూత స్వరూపుడు\n"
            "పవనుండు: వాయువు (ప్రాణవాయువు)\n"
            "లేకున్న: లేకపోయినట్లయితే\n"
            "పడు: పడిపోవును\n"
            "శరీరములు: శరీరాలు"
        ),
    },
    {
        "label":     "AN5_te — seen",
        "profile":   "AN5",
        "lang":      "te",
        "seen_as":   "training",
        "verse":     "యమ్మునిగొనిపోయి యర్ఘ్యపాద్యముల \nనెమ్మితో నిచ్చిన నృపుజూచి యతడు",
        "prompt": (
            "క్రింది పద్యానికి సంపూర్ణ విశ్లేషణ చేయండి: సంధి విభజన, "
            "ప్రతిపదార్థం మరియు తెలుగు భావం ఇవ్వండి.\n\n"
            "పద్యం:\nయమ్మునిగొనిపోయి యర్ఘ్యపాద్యముల \nనెమ్మితో నిచ్చిన నృపుజూచి యతడు"
        ),
        "reference": (
            "ప్రతిపదార్థం:\n"
            "ఆ + మునిన్: ఆ మునిని\n"
            "కొనిపోయి: తీసుకువెళ్ళి\n"
            "అర్ఘ్య + పాద్యములన్: అర్ఘ్యమును పాద్యమును\n"
            "నెమ్మితోన్: ప్రేమతో\n"
            "ఇచ్చిన: సమర్పించిన\n"
            "నృపున్ + చూచి: రాజును చూసి\n"
            "అతడు: ఆ ముని\n\n"
            "తెలుగు భావం: ఆ మునిని లోపలికి తీసుకువెళ్ళి ప్రేమతో "
            "అర్ఘ్యపాద్యములను సమర్పించిన దశరథ మహారాజును చూసి విశ్వామిత్రుడు ఇట్లన్నాడు."
        ),
    },
]

NEW_SAMPLES = [
    {
        "label":     "AN4_te — new verse",
        "profile":   "AN4",
        "lang":      "te",
        "seen_as":   "unseen",
        "verse":     "శ్రీరామచంద్రుని శ్రీపాదసేవలు \nభారమై తనువున బ్రహ్మానందమాయె",
        "prompt": (
            "క్రింది ద్విపద పద్యంలోని సంధి పదాలను '+' గుర్తుతో విడదీసి, "
            "ప్రతి పదానికి అర్థం రాయండి.\n\n"
            "పద్యం:\nశ్రీరామచంద్రుని శ్రీపాదసేవలు \nభారమై తనువున బ్రహ్మానందమాయె"
        ),
        "reference": None,
    },
    {
        "label":     "AN5_en — new verse",
        "profile":   "AN5",
        "lang":      "en",
        "seen_as":   "unseen",
        "verse":     "హరిభక్తిపరులకు నఖిలసౌఖ్యంబుల \nపరమేశుడిచ్చును బ్రహ్మపదంబు నిచ్చు",
        "prompt": (
            "Provide a complete linguistic analysis of the Dwipada verse below:\n"
            "1. Decompose sandhi compounds using '+' notation\n"
            "2. Give the word-by-word (pratipada) gloss\n"
            "3. Provide the overall meaning in Telugu (telugu bhavam)\n\n"
            "Verse:\nహరిభక్తిపరులకు నఖిలసౌఖ్యంబుల \nపరమేశుడిచ్చును బ్రహ్మపదంబు నిచ్చు"
        ),
        "reference": None,
    },
    {
        "label":     "AN4_en — new verse",
        "profile":   "AN4",
        "lang":      "en",
        "seen_as":   "unseen",
        "verse":     "సీతాపతిస్మరణ సేయుచు నుండిన \nభూతాపహారమగు బుద్ధిశుద్ధియగు",
        "prompt": (
            "For the classical Telugu verse given below, split all sandhi junctions "
            "using '+' notation, then list the word-by-word (pratipada) meaning.\n\n"
            "Verse:\nసీతాపతిస్మరణ సేయుచు నుండిన \nభూతాపహారమగు బుద్ధిశుద్ధియగు"
        ),
        "reference": None,
    },
]

# =============================================================================
# JUDGE RUBRICS
# =============================================================================

AN4_RUBRIC = """You are an expert judge evaluating a Telugu NLP model on a morphological
glossing task (AN4: sandhi decomposition + word-by-word meaning).

Score each criterion 1-5:

1. SANDHI_ACCURACY: Are sandhi compounds correctly split using '+' notation?
   5=all correct, 3=mostly correct with minor errors, 1=mostly wrong or missing

2. GLOSS_ACCURACY: Are the word-by-word Telugu meanings correct?
   5=all accurate, 3=mostly accurate, 1=mostly wrong

3. COVERAGE: Are all significant words in the verse glossed?
   5=complete, 3=most words covered, 1=many missing

4. FORMAT: Is the output structured and readable (no hallucinated tables)?
   5=clean structured output, 3=mostly structured, 1=unstructured or hallucinated

Respond ONLY with valid JSON, no preamble:
{
  "SANDHI_ACCURACY": <1-5>,
  "GLOSS_ACCURACY": <1-5>,
  "COVERAGE": <1-5>,
  "FORMAT": <1-5>,
  "TOTAL": <sum>,
  "REASONING": "<one sentence>"
}"""

AN5_RUBRIC = """You are an expert judge evaluating a Telugu NLP model on a full
morphological analysis task (AN5: sandhi decomposition + WTW gloss + Telugu bhavam).

Score each criterion 1-5:

1. SANDHI_ACCURACY: Are sandhi compounds correctly split using '+' notation?
   5=all correct, 3=mostly correct with minor errors, 1=mostly wrong or missing

2. GLOSS_ACCURACY: Are the word-by-word Telugu meanings correct?
   5=all accurate, 3=mostly accurate, 1=mostly wrong

3. BHAVAM_QUALITY: Is the Telugu bhavam (overall meaning) accurate and fluent?
   5=accurate and natural Telugu, 3=mostly accurate, 1=wrong or missing

4. FORMAT: Are both pratipada and bhavam sections present and structured?
   5=both sections clean, 3=partial, 1=missing sections or hallucinated

Respond ONLY with valid JSON, no preamble:
{
  "SANDHI_ACCURACY": <1-5>,
  "GLOSS_ACCURACY": <1-5>,
  "BHAVAM_QUALITY": <1-5>,
  "FORMAT": <1-5>,
  "TOTAL": <sum>,
  "REASONING": "<one sentence>"
}"""

# =============================================================================
# INFERENCE FUNCTION — accepts model_name as a parameter
# =============================================================================

@app.function(
    image=INFERENCE_IMAGE,
    gpu="a10g",
    secrets=[huggingface_secret],
    timeout=1 * HOURS,
)
def run_inference(samples: list[dict], model_name: str) -> list[dict]:
    import os, torch
    from unsloth import FastLanguageModel

    def to_multimodal(messages: list[dict]) -> list[dict]:
        return [
            {
                "role": msg["role"],
                "content": [{"type": "text", "text": msg["content"]}]
            }
            for msg in messages
        ]

    print(f"\n[Loading] {model_name}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name     = model_name,
        max_seq_length = 1024,
        load_in_4bit   = True,
        token          = os.environ["HF_TOKEN"],
    )
    FastLanguageModel.for_inference(model)
    print(f"[Loaded] Ready. Running {len(samples)} samples...\n")

    results = []
    for i, sample in enumerate(samples):
        system = SYSTEM_EN if sample["lang"] == "en" else SYSTEM_TE

        messages = to_multimodal([
            {"role": "system", "content": system},
            {"role": "user",   "content": sample["prompt"]},
        ])

        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to("cuda")

        with torch.no_grad():
            outputs = model.generate(
                input_ids      = inputs,
                max_new_tokens = MAX_NEW_TOKENS,
                temperature    = TEMPERATURE,
                do_sample      = True,
            )

        prediction = tokenizer.decode(
            outputs[0][inputs.shape[1]:],
            skip_special_tokens=True,
        )

        print(f"  [{i+1}/{len(samples)}] {sample['label']} — {len(prediction)} chars")

        results.append({
            "id":         i,
            "label":      sample["label"],
            "profile":    sample["profile"],
            "lang":       sample["lang"],
            "seen_as":    sample.get("seen_as", "unknown"),
            "verse":      sample["verse"],
            "prompt":     sample["prompt"],
            "model":      model_name,
            "prediction": prediction,
            "reference":  sample.get("reference"),
        })

    return results


# =============================================================================
# SAVE FUNCTIONS (all run locally)
# =============================================================================

def save_raw(results: list[dict], path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  ✓ Raw            → {path}")


def save_judge_ready(results: list[dict], path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in results:
            rubric = AN4_RUBRIC if r["profile"] == "AN4" else AN5_RUBRIC
            judge_prompt = (
                f"{rubric}\n\n"
                f"---VERSE---\n{r['verse']}\n\n"
                f"---MODEL OUTPUT---\n{r['prediction']}\n\n"
            )
            if r["reference"]:
                judge_prompt += f"---REFERENCE---\n{r['reference']}\n\n"
            judge_prompt += "---YOUR EVALUATION (JSON only)---"

            record = {
                "id":           r["id"],
                "label":        r["label"],
                "profile":      r["profile"],
                "lang":         r["lang"],
                "seen_as":      r["seen_as"],
                "model":        r["model"],
                "verse":        r["verse"],
                "model_output": r["prediction"],
                "reference":    r["reference"],
                "rubric":       rubric,
                "judge_prompt": judge_prompt,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"  ✓ Judge-ready    → {path}")


def save_summary(results: list[dict], path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    model_name = results[0]["model"] if results else "unknown"
    lines = [
        "METRICALARGS — Inference Results",
        f"Model  : {model_name}",
        f"Total  : {len(results)} samples",
        "=" * 70,
    ]
    for r in results:
        lines += [
            f"\n[{r['id']+1}] {r['label']}  |  seen_as={r['seen_as']}",
            "-" * 70,
            f"VERSE:\n{r['verse']}",
            f"\nMODEL OUTPUT:\n{r['prediction']}",
        ]
        if r["reference"]:
            lines.append(f"\nREFERENCE:\n{r['reference']}")
        lines.append("=" * 70)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  ✓ Summary        → {path}")


def save_comparison(
    finetuned_results: list[dict],
    base_results: list[dict],
    jsonl_path: str,
    txt_path: str,
):
    """
    Merge both result sets by id into a single side-by-side comparison.
    Each record has both base_output and finetuned_output for the same verse.
    The judge_prompt includes BOTH outputs so the judge can compare directly.
    """
    Path(jsonl_path).parent.mkdir(parents=True, exist_ok=True)

    # Index base results by id for easy lookup
    base_by_id = {r["id"]: r for r in base_results}

    # ── JSONL comparison ─────────────────────────────────────────────────────
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for ft in finetuned_results:
            base = base_by_id.get(ft["id"], {})
            rubric = AN4_RUBRIC if ft["profile"] == "AN4" else AN5_RUBRIC

            judge_prompt = (
                f"{rubric}\n\n"
                f"---VERSE---\n{ft['verse']}\n\n"
                f"---BASE MODEL OUTPUT---\n{base.get('prediction', 'N/A')}\n\n"
                f"---FINETUNED MODEL OUTPUT---\n{ft['prediction']}\n\n"
            )
            if ft["reference"]:
                judge_prompt += f"---REFERENCE---\n{ft['reference']}\n\n"
            judge_prompt += (
                "Evaluate EACH model separately using the rubric above.\n"
                "Respond ONLY with valid JSON:\n"
                "{\n"
                "  \"base\": {\"SANDHI_ACCURACY\": <1-5>, \"GLOSS_ACCURACY\": <1-5>, "
                "\"COVERAGE_OR_BHAVAM\": <1-5>, \"FORMAT\": <1-5>, \"TOTAL\": <sum>, "
                "\"REASONING\": \"<one sentence>\"},\n"
                "  \"finetuned\": {\"SANDHI_ACCURACY\": <1-5>, \"GLOSS_ACCURACY\": <1-5>, "
                "\"COVERAGE_OR_BHAVAM\": <1-5>, \"FORMAT\": <1-5>, \"TOTAL\": <sum>, "
                "\"REASONING\": \"<one sentence>\"}\n"
                "}"
            )

            record = {
                "id":                ft["id"],
                "label":             ft["label"],
                "profile":           ft["profile"],
                "lang":              ft["lang"],
                "seen_as":           ft["seen_as"],
                "verse":             ft["verse"],
                "reference":         ft["reference"],
                "base_model":        BASE_MODEL,
                "finetuned_model":   FINETUNED_MODEL,
                "base_output":       base.get("prediction", "N/A"),
                "finetuned_output":  ft["prediction"],
                "rubric":            rubric,
                "judge_prompt":      judge_prompt,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"  ✓ Comparison     → {jsonl_path}")

    # ── TXT comparison ───────────────────────────────────────────────────────
    lines = [
        "METRICALARGS — Base vs Fine-tuned Comparison",
        f"Base      : {BASE_MODEL}",
        f"Finetuned : {FINETUNED_MODEL}",
        f"Samples   : {len(finetuned_results)}",
        "=" * 70,
    ]
    for ft in finetuned_results:
        base = base_by_id.get(ft["id"], {})
        lines += [
            f"\n[{ft['id']+1}] {ft['label']}  |  seen_as={ft['seen_as']}",
            "-" * 70,
            f"VERSE:\n{ft['verse']}",
            f"\n── BASE MODEL ──────────────────────────────",
            base.get("prediction", "N/A"),
            f"\n── FINE-TUNED MODEL ────────────────────────",
            ft["prediction"],
        ]
        if ft["reference"]:
            lines.append(f"\n── REFERENCE ───────────────────────────────")
            lines.append(ft["reference"])
        lines.append("=" * 70)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  ✓ Comparison txt → {txt_path}")


# =============================================================================
# SAMPLE LOADER
# =============================================================================

def load_samples(mode: str) -> list[dict]:
    if mode == "seen":
        return SEEN_SAMPLES
    elif mode == "new":
        return NEW_SAMPLES
    elif mode == "all":
        return SEEN_SAMPLES + NEW_SAMPLES
    else:  # "test"
        samples = []
        with open(TEST_JSONL, encoding="utf-8") as f:
            for line in f:
                rec      = json.loads(line)
                user_msg = next(m for m in rec["messages"] if m["role"] == "user")
                profile  = rec["_profile_id"].split("_")[0]
                lang     = rec["_profile_id"].split("_")[1]
                verse_match = re.search(
                    r'(?:పద్యం|Verse)\s*:\s*\n(.+?)$',
                    user_msg["content"], re.DOTALL
                )
                verse = verse_match.group(1).strip() if verse_match else ""
                samples.append({
                    "label":     rec["_profile_id"],
                    "profile":   profile,
                    "lang":      lang,
                    "seen_as":   rec.get("_seen_as", "unseen"),
                    "verse":     verse,
                    "prompt":    user_msg["content"],
                    "reference": None,
                })
        return samples


# =============================================================================
# LOCAL ENTRYPOINT
# =============================================================================

@app.local_entrypoint()
def main(mode: str = "test", model: str = "both"):

    samples = load_samples(mode)
    print(f"\nMode: {mode} | Model: {model} | Samples: {len(samples)}")

    if model in ("both", "finetuned"):
        print(f"\n[Fine-tuned] Running inference...")
        ft_results = run_inference.remote(samples, FINETUNED_MODEL)
        out = f"{RESULTS_DIR}/finetuned"
        print(f"\n[Saving fine-tuned results to ./{out}/]")
        save_raw(ft_results,         f"{out}/infer_results_raw.jsonl")
        save_judge_ready(ft_results, f"{out}/infer_results_judge.jsonl")
        save_summary(ft_results,     f"{out}/infer_results_summary.txt")

    if model in ("both", "base"):
        print(f"\n[Base] Running inference...")
        base_results = run_inference.remote(samples, BASE_MODEL)
        out = f"{RESULTS_DIR}/base"
        print(f"\n[Saving base results to ./{out}/]")
        save_raw(base_results,         f"{out}/infer_results_raw.jsonl")
        save_judge_ready(base_results, f"{out}/infer_results_judge.jsonl")
        save_summary(base_results,     f"{out}/infer_results_summary.txt")

    # ── Comparison — load finetuned from disk if not in memory ───────────────
    if model in ("both", "base"):
        ft_path = f"{RESULTS_DIR}/finetuned/infer_results_raw.jsonl"
        if model == "base" and Path(ft_path).exists():
            print(f"\n[Loading existing fine-tuned results from {ft_path}]")
            with open(ft_path, encoding="utf-8") as f:
                ft_results = [json.loads(line) for line in f if line.strip()]
        
        if "ft_results" in dir() or model == "both":
            out = f"{RESULTS_DIR}/comparison"
            print(f"\n[Saving comparison to ./{out}/]")
            save_comparison(
                ft_results,
                base_results,
                jsonl_path = f"{out}/infer_comparison.jsonl",
                txt_path   = f"{out}/infer_comparison.txt",
            )
        else:
            print(f"\n[Skipping comparison — no fine-tuned results found at {ft_path}]")

    print(f"\n✅ Done.")