"""
METRICALARGS — Gemma 4 Retrieval Fine-tuning (Modal)
=====================================================
Trains a Gemma 4 E4B model on retrieval data:
  • ift_retrieval_trl.jsonl  (RT1–RT6: first/last line, fragment, meaning → poem)

Upload data to the volume first:
    modal volume put metricalargs-vol ./ift_retrieval_trl.jsonl /data/ift_retrieval_trl.jsonl

Run training:
    modal run finetune_gemma4_retrieval.py

Push to Hub after training:
    modal run finetune_gemma4_retrieval.py::main   (calls push_to_hub)
"""

import modal
from modal import App, Image as ModalImage, Volume, Secret

# =============================================================================
# § A  MODAL APP / VOLUME / SECRETS
# =============================================================================

app = App("metricalargs-retrieval-gemma4")

exp_volume = Volume.from_name("metricalargs-vol", create_if_missing=True)
VOLUME_CONFIG = {"/vol": exp_volume}

huggingface_secret = Secret.from_name("huggingface")
wandb_secret       = Secret.from_name("wandb")

# =============================================================================
# § B  CONFIGURATION CONSTANTS
# =============================================================================

HOURS = 60 * 60

BASE_MODEL_NAME = "google/gemma-4-E4B-it"

REMOTE_DATA      = "/vol/data/ift_retrieval_trl.jsonl"

CHECKPOINT_DIR   = "/vol/checkpoints/retrieval"
FINAL_OUTPUT_DIR = "/vol/output/retrieval/final_merged"

WANDB_PROJECT    = "metricalargs"
WANDB_RUN_NAME   = "gemma4-retrieval"

HF_REPO          = "maheshemani/dwipada-gemma4-E4B-retrieval"

# =============================================================================
# § C  IMAGE BUILD
# =============================================================================

FINETUNING_GPU_IMAGE = (
    ModalImage.debian_slim(python_version="3.11")
    .apt_install("git", "build-essential")
    .run_commands(
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
    timeout=6 * HOURS,
)
def fine_tune_retrieval(
    model_path:       str   = BASE_MODEL_NAME,
    data_path:        str   = REMOTE_DATA,
    checkpoint_dir:   str   = CHECKPOINT_DIR,
    output_dir:       str   = FINAL_OUTPUT_DIR,
    lora_r:           int   = 16,
    lora_alpha:       int   = 16,
    num_train_epochs: int   = 3,
    learning_rate:    float = 2e-4,
    max_seq_length:   int   = 1024,
    seed:             int   = 42,
):
    import os, json, torch
    from unsloth import FastLanguageModel, get_chat_template
    from trl import SFTTrainer, SFTConfig
    from transformers import TrainerCallback
    from datasets import Dataset

    # ── Print run config ─────────────────────────────────────────────────────
    print("=" * 60)
    print("METRICALARGS — Gemma 4 Retrieval Fine-tuning")
    print("=" * 60)
    print(f"  Model          : {model_path}")
    print(f"  Data           : {data_path}")
    print(f"  Checkpoint dir : {checkpoint_dir}")
    print(f"  Output dir     : {output_dir}")
    print(f"  LoRA r / alpha : {lora_r} / {lora_alpha}")
    print(f"  Epochs         : {num_train_epochs}")
    print(f"  Learning rate  : {learning_rate}")
    print(f"  Max seq length : {max_seq_length}")
    print(f"  Seed           : {seed}")
    print("=" * 60)

    os.environ["WANDB_PROJECT"] = WANDB_PROJECT

    # ── 1. Load dataset ───────────────────────────────────────────────────────
    print(f"\n[Data] Loading {data_path}...")
    records = []
    with open(data_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    print(f"[Data] {len(records):,} records loaded.")

    assert "messages" in records[0], "Records must have a 'messages' key (TRL format)."
    dataset = Dataset.from_list(records)

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
        r                          = lora_r,
        lora_alpha                 = lora_alpha,
        target_modules             = ["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"],
        lora_dropout               = 0.05,
        bias                       = "none",
        use_gradient_checkpointing = "unsloth",
        random_state               = seed,
        max_seq_length             = max_seq_length,
    )

    # ── 4. Chat template ──────────────────────────────────────────────────────
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
                msgs, tokenize=False, add_generation_prompt=False,
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
        ),
    )

    # ── 8. Train ──────────────────────────────────────────────────────────────
    print("\n[Training] Starting...")
    FastLanguageModel.for_training(model)
    trainer.train()

    # ── 9. Save ───────────────────────────────────────────────────────────────
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
# § F  EVALUATION TEST CASES (Retrieval profiles)
# =============================================================================

SYSTEM_PROMPT = "You are a Telugu and Sanskrit scholar specialising in Dwipada poetry."

RETRIEVAL_TEST_CASES = {
    "RT1": {
        "desc": "First line → retrieve full poem",
        "user": (
            "క్రింది పద్యపాదం ఒక ప్రసిద్ధ ద్విపద పద్యంలోని మొదటి పాదం. "
            "ఈ పద్యాన్ని మొత్తం చెప్పండి. మీ సొంత వాక్యాలు ఉపయోగించవద్దు.\n\n"
            "మొదటి పాదం: భువనత్రయాధారభూతమయుండు"
        ),
        "expected": (
            "పూర్తి పద్యం:\n"
            "భువనత్రయాధారభూతమయుండు \n"
            "పవనుండు లేకున్న బడు శరీరములు"
        ),
    },
    "RT2": {
        "desc": "Last line → retrieve full poem",
        "user": (
            "క్రింది పద్యపాదం ఒక ద్విపద పద్యంలోని చివరి పాదం. "
            "దీని ముందు పాదాన్ని గుర్తించి పద్యాన్ని పూర్తి చేయండి.\n\n"
            "చివరి పాదం: పవనుండు లేకున్న బడు శరీరములు"
        ),
        "expected": (
            "పూర్తి పద్యం:\n"
            "భువనత్రయాధారభూతమయుండు \n"
            "పవనుండు లేకున్న బడు శరీరములు"
        ),
    },
    "RT4": {
        "desc": "English meaning → retrieve matching poem",
        "user": (
            "క్రింది English భావానికి సరిపోయే ద్విపద పద్యాన్ని చెప్పండి.\n\n"
            "English Meaning: Without Pavana (the wind / life-breath) who is the "
            "foundation of the three worlds, bodies perish."
        ),
        "expected": (
            "పద్యం:\n"
            "భువనత్రయాధారభూతమయుండు \n"
            "పవనుండు లేకున్న బడు శరీరములు"
        ),
    },
    "RT5": {
        "desc": "Telugu meaning → retrieve matching poem",
        "user": (
            "క్రింది తెలుగు భావానికి సరిపోయే ద్విపద పద్యాన్ని చెప్పండి.\n\n"
            "తెలుగు భావం: మూడు లోకాలకు ఆధారమైన వాయువు లేకుండా శరీరాలు నశించిపోతాయి."
        ),
        "expected": (
            "పద్యం:\n"
            "భువనత్రయాధారభూతమయుండు \n"
            "పవనుండు లేకున్న బడు శరీరములు"
        ),
    },
    "RT6": {
        "desc": "Masked line → complete the verse",
        "user": (
            "క్రింది పద్యంలో మొదటి పాదంలో కొంత భాగం తొలగించబడింది. "
            "పూర్తి పాదాన్ని పూర్చి పద్యం పూర్తి చేయండి.\n\n"
            "పాక్షిక పద్యం:\nభువనత్రయా_______\nపవనుండు లేకున్న బడు శరీరములు"
        ),
        "expected": (
            "పూర్తి పద్యం:\n"
            "భువనత్రయాధారభూతమయుండు \n"
            "పవనుండు లేకున్న బడు శరీరములు"
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
    for pid, case in RETRIEVAL_TEST_CASES.items():
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
                input_ids=inputs, max_new_tokens=256, temperature=0.1, do_sample=True,
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


# =============================================================================
# § G  LOCAL ENTRYPOINT
# =============================================================================

@app.local_entrypoint()
def main():
    fine_tune_retrieval.remote()
    # push_to_hub.remote()
    # evaluate.remote()
