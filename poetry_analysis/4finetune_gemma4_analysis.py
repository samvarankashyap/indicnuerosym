"""
METRICALARGS — Gemma 4 Analysis Fine-tuning (Modal)
====================================================
Trains a Gemma 4 E4B model on combined analysis data:
  • ift_analysis_trl.jsonl   (AN1–AN6: syllabification, meter, gana, WTW gloss)
  • ift_an4_an5_trl.jsonl    (AN4/AN5 bilingual: sandhi + WTW + bhavam)

Both files are merged inside the training function before training begins.
The merged dataset is shuffled with a fixed seed for reproducibility.

Upload data to the volume first:
    modal volume put metricalargs-vol ./ift_analysis_trl.jsonl   /data/ift_analysis_trl.jsonl
    modal volume put metricalargs-vol ./ift_an4_an5_trl.jsonl    /data/ift_an4_an5_trl.jsonl

Run training:
    modal run finetune_gemma4_analysis.py

Push to Hub after training:
    modal run finetune_gemma4_analysis.py::main   (calls push_to_hub)
"""

import modal
from modal import App, Image as ModalImage, Volume, Secret

# =============================================================================
# § A  MODAL APP / VOLUME / SECRETS
# =============================================================================

app = App("metricalargs-analysis-gemma4")

exp_volume = Volume.from_name("metricalargs-vol", create_if_missing=True)
VOLUME_CONFIG = {"/vol": exp_volume}

huggingface_secret = Secret.from_name("huggingface")
wandb_secret       = Secret.from_name("wandb")

# =============================================================================
# § B  CONFIGURATION CONSTANTS
# =============================================================================

HOURS = 60 * 60

# Model — Gemma 4 E4B (4-bit BnB quantised Unsloth variant).
# "E" = effective parameters; E4B is text-only capable and A10G-friendly.
BASE_MODEL_NAME = "google/gemma-4-E4B-it"

# Data files on the Modal volume (merged at runtime)
REMOTE_DATA_ANALYSIS  = "/vol/data/ift_analysis_trl.jsonl"
REMOTE_DATA_AN4_AN5   = "/vol/data/ift_an4_an5_trl.jsonl"

# Output paths
CHECKPOINT_DIR        = "/vol/checkpoints/analysis"
FINAL_OUTPUT_DIR      = "/vol/output/analysis/final_merged"

# Experiment tracking
WANDB_PROJECT         = "metricalargs"
WANDB_RUN_NAME        = "gemma4-analysis"

# Hub destination
HF_REPO               = "maheshemani/dwipada-gemma4-E4B-analysis"

# =============================================================================
# § C  IMAGE BUILD
# =============================================================================

FINETUNING_GPU_IMAGE = (
    ModalImage.debian_slim(python_version="3.11")
    .apt_install("git", "build-essential")
    .run_commands(
        # Single-pass install: let pip resolve the full Gemma 4 stack.
        # unsloth_zoo carries the Gemma 4 patches; latest unsloth surfaces them.
        "pip install --upgrade unsloth unsloth_zoo bitsandbytes accelerate peft trl "
        "triton xformers wandb datasets sentencepiece protobuf "
        "--extra-index-url https://download.pytorch.org/whl/cu126",
        "pip check",
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "HF_HOME": "/vol/.cache",
    })
)

# =============================================================================
# § D  FINE-TUNING FUNCTION
# =============================================================================

@app.function(
    image=FINETUNING_GPU_IMAGE,
    gpu="a10g",
    volumes=VOLUME_CONFIG,
    secrets=[huggingface_secret, wandb_secret],
    timeout=8 * HOURS,
)
def fine_tune_analysis(
    model_path:        str   = BASE_MODEL_NAME,
    data_analysis:     str   = REMOTE_DATA_ANALYSIS,
    data_an4_an5:      str   = REMOTE_DATA_AN4_AN5,
    checkpoint_dir:    str   = CHECKPOINT_DIR,
    output_dir:        str   = FINAL_OUTPUT_DIR,
    lora_r:            int   = 16,
    lora_alpha:        int   = 16,
    num_train_epochs:  int   = 3,
    learning_rate:     float = 2e-4,
    max_seq_length:    int   = 1024,
    seed:              int   = 42,
):
    import os, json, random, torch
    from unsloth import FastLanguageModel, get_chat_template
    from trl import SFTTrainer, SFTConfig
    from transformers import TrainerCallback
    from datasets import Dataset

    # ── Print run config ─────────────────────────────────────────────────────
    print("=" * 60)
    print("METRICALARGS — Gemma 4 Analysis Fine-tuning")
    print("=" * 60)
    print(f"  Model          : {model_path}")
    print(f"  Data (analysis): {data_analysis}")
    print(f"  Data (AN4/AN5) : {data_an4_an5}")
    print(f"  Checkpoint dir : {checkpoint_dir}")
    print(f"  Output dir     : {output_dir}")
    print(f"  LoRA r / alpha : {lora_r} / {lora_alpha}")
    print(f"  Epochs         : {num_train_epochs}")
    print(f"  Learning rate  : {learning_rate}")
    print(f"  Max seq length : {max_seq_length}")
    print(f"  Seed           : {seed}")
    print("=" * 60)

    os.environ["WANDB_PROJECT"] = WANDB_PROJECT

    # ── 1. Load and merge datasets ────────────────────────────────────────────
    def load_jsonl(path: str) -> list[dict]:
        records = []
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        return records

    print(f"\n[Data] Loading {data_analysis}...")
    records_main  = load_jsonl(data_analysis)
    print(f"       {len(records_main):,} records from analysis dataset.")

    print(f"[Data] Loading {data_an4_an5}...")
    records_extra = load_jsonl(data_an4_an5)
    print(f"       {len(records_extra):,} records from AN4/AN5 dataset.")

    all_records = records_main + records_extra
    rng = random.Random(seed)
    rng.shuffle(all_records)
    print(f"[Data] Merged & shuffled → {len(all_records):,} total records.")

    # Sanity-check: confirm messages format
    sample = all_records[0]
    assert "messages" in sample, "Records must have a 'messages' key (TRL format)."

    dataset = Dataset.from_list(all_records)

    # ── 2. Load model & tokenizer ─────────────────────────────────────────────
    print(f"\n[Model] Loading {model_path}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name     = model_path,
        max_seq_length = max_seq_length,
        load_in_4bit   = True,
    )

    # ── 3. LoRA adapters ──────────────────────────────────────────────────────
    model = FastLanguageModel.get_peft_model(
        model,
        r                        = lora_r,
        lora_alpha               = lora_alpha,
        target_modules           = ["q_proj", "k_proj", "v_proj", "o_proj",
                                    "gate_proj", "up_proj", "down_proj"],
        lora_dropout             = 0.05,
        bias                     = "none",
        use_gradient_checkpointing = "unsloth",   # Gemma 4 recommended setting
        random_state             = seed,
        max_seq_length           = max_seq_length,
    )

    # ── 4. Chat template (Gemma 4 uses standard chat format) ─────────────────
    tokenizer = get_chat_template(tokenizer, chat_template="gemma-4")

    # ── 5. Formatting function ────────────────────────────────────────────────
    def formatting_func(examples):
        """Apply Gemma-4 chat template. Handles single-row probe and batched calls."""
        messages = examples["messages"]
        if isinstance(messages[0], dict):
            messages = [messages]
        texts = []
        for msgs in messages:
            text = tokenizer.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=False,
            )
            texts.append(text)
        return texts

    # ── 6. Volume commit callback ─────────────────────────────────────────────
    class VolumeCommitCallback(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            exp_volume.commit()
            print(f"  [vol] Committed checkpoint at step {state.global_step}")

    # ── 7. Trainer ────────────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model           = model,
        tokenizer       = tokenizer,
        train_dataset   = dataset,
        formatting_func = formatting_func,
        max_seq_length  = max_seq_length,
        callbacks       = [VolumeCommitCallback()],
        args            = SFTConfig(
            per_device_train_batch_size  = 4,
            gradient_accumulation_steps  = 4,
            warmup_steps                 = 50,
            num_train_epochs             = num_train_epochs,
            learning_rate                = learning_rate,
            fp16                         = not torch.cuda.is_bf16_supported(),
            bf16                         = torch.cuda.is_bf16_supported(),
            logging_steps                = 10,
            save_strategy                = "steps",
            save_steps                   = 75,
            output_dir                   = checkpoint_dir,
            optim                        = "adamw_8bit",
            packing                      = True,
            report_to                    = "none",
            run_name                     = WANDB_RUN_NAME,
            seed                         = seed,
            resume_from_checkpoint       = "/vol/checkpoints/analysis/checkpoint-6225",
        ),
    )

    # ── 8. Train ──────────────────────────────────────────────────────────────
    print("\n[Training] Starting...")
    FastLanguageModel.for_training(model)
    trainer.train()

    # ── 9. Save merged 16-bit model ───────────────────────────────────────────
    print(f"\n[Save] Merging and saving to {output_dir}...")
    model.save_pretrained_merged(output_dir, tokenizer, save_method="merged_16bit")
    exp_volume.commit()
    print(f"[Done] Merged model saved to {output_dir}")


# =============================================================================
# § E  HUB PUSH
# =============================================================================

@app.function(
    image=FINETUNING_GPU_IMAGE,
    volumes=VOLUME_CONFIG,
    secrets=[huggingface_secret],
    timeout=1 * HOURS,
)
def push_to_hub(
    hf_repo:   str = HF_REPO,
    model_dir: str = FINAL_OUTPUT_DIR,
):
    from huggingface_hub import HfApi
    import os

    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(repo_id=hf_repo, repo_type="model", private=True, exist_ok=True)
    api.upload_folder(folder_path=model_dir, repo_id=hf_repo, repo_type="model")
    print(f"[Hub] Pushed to {hf_repo}")


# =============================================================================
# § F  EVALUATION TEST CASES (Analysis profiles)
# =============================================================================

SYSTEM_PROMPT = "You are a Telugu and Sanskrit scholar specialising in Dwipada poetry."

ANALYSIS_TEST_CASES = {
    "AN4_te": {
        "desc": "Sandhi decomposition + WTW gloss (Telugu prompt)",
        "user": (
            "క్రింది ద్విపద పద్యంలోని సంధి పదాలను '+' గుర్తుతో విడదీసి, "
            "ప్రతి పదానికి అర్థం రాయండి.\n\n"
            "పద్యం:\nచతురబ్ధిజలములజలమున ముంచి \nయతులసత్త్వక్రీడ నడరి కారించి"
        ),
        "expected": (
            "ప్రతిపదార్థం:\n"
            "చతుః + అబ్ధి + జలముల: నాలుగు సముద్రాల యొక్క నీటిలో\n"
            "ముంచి: మునకలు వేయించి\n"
            "అతుల: సాటిలేని\n"
            "సత్త్వ: బలముతో కూడిన\n"
            "క్రీడన్: విలాసముతో (ఆటతో)\n"
            "అడరి: అతిశయించి\n"
            "కారించి: వేధించి"
        ),
    },
    "AN4_en": {
        "desc": "Sandhi decomposition + WTW gloss (English prompt)",
        "user": (
            "Break the sandhi compounds in the following Dwipada verse using '+' notation "
            "and provide a word-by-word gloss for each resulting word.\n\n"
            "Verse:\nచతురబ్ధిజలములజలమున ముంచి \nయతులసత్త్వక్రీడ నడరి కారించి"
        ),
        "expected": (
            "Word-by-word gloss (pratipada-artha):\n"
            "చతుః + అబ్ధి + జలముల: in the waters of the four oceans\n"
            "ముంచి: having immersed\n"
            "అతుల: incomparable\n"
            "సత్త్వ: with strength\n"
            "క్రీడన్: in play / sport\n"
            "అడరి: having intensified\n"
            "కారించి: having caused / afflicted"
        ),
    },
    "AN5_te": {
        "desc": "Full analysis — WTW + Telugu bhavam (Telugu prompt)",
        "user": (
            "క్రింది పద్యానికి సంపూర్ణ విశ్లేషణ చేయండి: సంధి విభజన, "
            "ప్రతిపదార్థం మరియు తెలుగు భావం ఇవ్వండి.\n\n"
            "పద్యం:\nయమ్మునిగొనిపోయి యర్ఘ్యపాద్యముల \nనెమ్మితో నిచ్చిన నృపుజూచి యతడు"
        ),
        "expected": (
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
    "AN5_en": {
        "desc": "Full analysis — WTW + Telugu bhavam (English prompt)",
        "user": (
            "Provide a complete linguistic analysis of the Dwipada verse below:\n"
            "1. Decompose sandhi compounds using '+' notation\n"
            "2. Give the word-by-word (pratipada) gloss\n"
            "3. Provide the overall meaning in Telugu (telugu bhavam)\n\n"
            "Verse:\nయమ్మునిగొనిపోయి యర్ఘ్యపాద్యముల \nనెమ్మితో నిచ్చిన నృపుజూచి యతడు"
        ),
        "expected": (
            "Word-by-word gloss (pratipada-artha):\n"
            "ఆ + మునిన్: that sage (Viswamitra)\n"
            "కొనిపోయి: having brought inside\n"
            "అర్ఘ్య + పాద్యములన్: the arghya and padya offerings\n"
            "నెమ్మితోన్: with affection\n"
            "ఇచ్చిన: having offered\n"
            "నృపున్ + చూచి: seeing the king (Dasharatha)\n"
            "అతడు: he (the sage)\n\n"
            "Telugu bhavam: Having brought that sage inside and affectionately "
            "offered arghya and padya, Viswamitra, upon seeing king Dasharatha, spoke thus."
        ),
    },
}


@app.function(
    image=FINETUNING_GPU_IMAGE,
    gpu="a10g",
    volumes=VOLUME_CONFIG,
    timeout=1 * HOURS,
)
def evaluate(model_dir: str = FINAL_OUTPUT_DIR):
    from unsloth import FastLanguageModel
    import torch

    print(f"\n[Eval] Loading model from {model_dir}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_dir, max_seq_length=1024, load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    results = {}
    for pid, case in ANALYSIS_TEST_CASES.items():
        print(f"\n{'='*60}\nProfile: {pid} — {case['desc']}\n{'='*60}")

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": case["user"]},
        ]
        inputs = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt",
        ).to("cuda")

        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs, max_new_tokens=512, temperature=0.1, do_sample=True,
            )

        prediction = tokenizer.decode(
            outputs[0][inputs.shape[1]:], skip_special_tokens=True,
        )

        print(f"\n--- EXPECTED ---\n{case['expected']}")
        print(f"\n--- PREDICTED ---\n{prediction}")

        exp_lines  = set(case["expected"].strip().splitlines())
        pred_lines = set(prediction.strip().splitlines())
        score = len(exp_lines & pred_lines) / max(len(exp_lines), 1)
        print(f"\nLine overlap: {score:.1%}")
        results[pid] = {"prediction": prediction, "score": score}

    print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
    for pid, r in results.items():
        print(f"  {pid}: {r['score']:.1%}")
    return results



@app.function(
    image=FINETUNING_GPU_IMAGE,
    gpu="a10g",
    volumes=VOLUME_CONFIG,
    timeout=1 * HOURS,
)
def merge_checkpoint(
    checkpoint_path: str = "/vol/checkpoints/analysis/checkpoint-6225",
    output_dir: str = "/vol/output/analysis/final_merged",
):
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name     = checkpoint_path,
        max_seq_length = 1024,
        load_in_4bit   = True,
    )
    model.save_pretrained_merged(output_dir, tokenizer, save_method="merged_16bit")
    exp_volume.commit()
    print(f"Merged to {output_dir}")


# =============================================================================
# § G  LOCAL ENTRYPOINT
# =============================================================================

@app.local_entrypoint()
def main():
    # Uncomment the step you want to run:
    # fine_tune_analysis.remote()
    # push_to_hub.remote()
    # evaluate.remote()
    # merge_checkpoint.remote()
    push_to_hub.remote()
