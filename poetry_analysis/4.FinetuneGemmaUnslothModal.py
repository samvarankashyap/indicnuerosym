import os
import modal
from modal import App, Image as ModalImage, Volume, Secret

# =============================================================================
# § A  MODAL APP DEFINITION, VOLUME AND SECRET SETUP
# =============================================================================

app = App("metricalargs-analysis-finetune")

# Persistent volume for checkpoints and data
exp_volume = Volume.from_name("metricalargs-vol", create_if_missing=True)
VOLUME_CONFIG = {
    "/vol": exp_volume,
}

# Ensure you have created this secret: modal secret create huggingface HF_TOKEN=...
huggingface_secret = Secret.from_name("huggingface")
wandb_secret = Secret.from_name("wandb")

# =============================================================================
# § B  CONFIGURATION CONSTANTS
# =============================================================================

HOURS = 60 * 60
BASE_MODEL_NAME = "unsloth/gemma-3-1b-it-bnb-4bit"  #
WANDB_PROJECT_DEFAULT = "metricalargs"
REMOTE_DATA = "/vol/data/ift_analysis_trl.jsonl"    #
OUTPUT_DIR_DEFAULT = "/vol/output"

# =============================================================================
# § C  IMAGE BUILD (Synchronized Stack)
# =============================================================================

FINETUNING_GPU_IMAGE = (
    ModalImage.debian_slim(python_version="3.11")
    .apt_install("git", "build-essential")
    .run_commands(
        # 1. Install the full stack in ONE resolver pass.
        #    Unsloth's transitive deps (transformers 5.x → torchao 0.17+)
        #    require torch ≥ 2.7, so we let pip pick the right torch from
        #    the cu126 index rather than pre-pinning 2.6.
        "pip install unsloth unsloth_zoo bitsandbytes accelerate peft trl "
        "triton xformers wandb datasets sentencepiece protobuf "
        "--extra-index-url https://download.pytorch.org/whl/cu126",

        # 2. Audit
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
    gpu="a10g", # A10G is cost-effective for 1B models
    volumes=VOLUME_CONFIG,
    secrets=[huggingface_secret, wandb_secret],
    timeout=5 * HOURS,
)
def fine_tune_metricalargs(
    model_path: str = BASE_MODEL_NAME,
    data_path: str = REMOTE_DATA,
    output_dir: str = OUTPUT_DIR_DEFAULT,
    lora_r: int = 16,
    lora_alpha: int = 32,
    num_train_epochs: int = 3,
    learning_rate: float = 2e-4,
    max_seq_length: int = 1024,
):
    from unsloth import FastLanguageModel, get_chat_template
    from trl import SFTTrainer, SFTConfig
    from transformers import TrainerCallback
    from datasets import Dataset
    import torch
    import json
    import os
    os.environ["WANDB_PROJECT"] = WANDB_PROJECT_DEFAULT 
    # 1. Load Model & Tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_path,
        max_seq_length = max_seq_length,
        load_in_4bit = True,
    )

    # 2. Add LoRA Adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r = lora_r,
        lora_alpha = lora_alpha,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", 
                          "gate_proj", "up_proj", "down_proj"],
        lora_dropout = 0.05,
        bias = "none",
        random_state = 42,
    )

    # 3. Apply Gemma 3 Chat Template
    tokenizer = get_chat_template(tokenizer, chat_template="gemma-3")

    # 4. Load Local Dataset (Telugu UTF-8 Guard)
    print(f"Loading data from {data_path}...")
    raw_records = []
    with open(data_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.strip():
                raw_records.append(json.loads(line))
    
    dataset = Dataset.from_list(raw_records)
    print(f"Loaded {len(dataset)} records.")

    # 5. Volume Persistence Callback
    class VolumeCommitCallback(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            exp_volume.commit()
            print(f" [vol] Committed checkpoint at step {state.global_step}")

    # 6. Trainer Setup
    def formatting_func(examples):
            """Apply Gemma-3 chat template — handles both single-row probe and batched calls."""
            messages = examples["messages"]

            # Single-row probe: messages is a list of dicts like [{"role":..., "content":...}, ...]
            # Batch call: messages is a list of those lists
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

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        formatting_func=formatting_func,      # replaces dataset_text_field
        max_seq_length=max_seq_length,
        callbacks=[VolumeCommitCallback()],
        args=SFTConfig(
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            warmup_steps=50,                  # warmup_ratio is deprecated in TRL 0.24
            num_train_epochs=num_train_epochs,
            learning_rate=learning_rate,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=10,
            save_strategy="steps",
            save_steps=75,
            output_dir="/vol/checkpoints",
            optim="adamw_8bit",
            packing=True,
            report_to="none",
            run_name="metricalargs-analysis",
        ),
    )

    # 7. Execute Training
    FastLanguageModel.for_training(model)
    trainer.train()

    # 8. Save & Merge
    final_dir = os.path.join(output_dir, "final_merged")
    model.save_pretrained_merged(final_dir, tokenizer, save_method="merged_16bit")
    
    # Final Volume Commit
    exp_volume.commit()
    print(f"Training complete. Merged model at {final_dir}")


@app.function(
    image=FINETUNING_GPU_IMAGE,
    volumes=VOLUME_CONFIG,
    secrets=[huggingface_secret],
    timeout=1 * HOURS,
)
def push_to_hub(
    hf_repo: str = "maheshemani/dwipada-gemma3-1b-analysis",
    model_dir: str = "/vol/output/final_merged",
):
    from huggingface_hub import HfApi
    import os

    api = HfApi(token=os.environ["HF_TOKEN"])
    
    # Create the repo first (sets privacy), then upload
    api.create_repo(
        repo_id=hf_repo,
        repo_type="model",
        private=True,
        exist_ok=True,   # won't error if repo already exists
    )
    api.upload_folder(
        folder_path=model_dir,
        repo_id=hf_repo,
        repo_type="model",
    )
    print(f"Pushed to {hf_repo}")

SYSTEM = "You are a Telugu and Sanskrit scholar specialising in Dwipada poetry."

TEST_CASES = {
    "AN1": {
        "desc": "Full poem syllabification",
        "user": (
            "మీరు తెలుగు ఛందస్సు నిపుణుడు. క్రింది ద్విపద పద్యంలోని "
            "ప్రతి అక్షరాన్ని గురువు (U) లేదా లఘువు (I) గా గుర్తించండి.\n\n"
            "పద్యం:\nభువనత్రయాధారభూతమయుండు \nపవనుండు లేకున్న బడు శరీరములు"
        ),
        "expected": (
            "ఛందస్సు విశ్లేషణ (గురు-లఘు విభజన):\n"
            "line_1: భువనత్ర (Sala - IIUI - Indra) | యాధార (Ta - UUI - Indra) | "
            "భూతమ (Bha - UII - Indra) | యుండు (Ha/Gala - UI - Surya) | "
            "Yati: Matches: 'భ' and 'భ'. | Prasa: 2nd Letter 'వ' matches 'వ'.\n"
            "line_2: పవనుండు (Sala - IIUI - Indra) | లేకున్న (Ta - UUI - Indra) | "
            "బడుశరీ (Naga - IIIU - Indra) | రములు (Na - III - Surya) | "
            "Yati: Matches: 'ప' and 'బ'. | Prasa: 2nd Letter 'వ' matches 'వ'."
        ),
    },
    "AN2": {
        "desc": "Gana identification",
        "user": (
            "క్రింది ద్విపద పద్యాన్ని గణాలుగా విభజించి, ప్రతి గణానికి "
            "పేరు (ఇంద్ర/సూర్య గణాలు) చెప్పండి.\n\n"
            "పద్యం:\nయీక్షింప నదిగాక యిరువదినాల్గు \nయక్షోహిణులు దాను నట దండువచ్చి"
        ),
        "expected": (
            "గణ విభజన:\n"
            "line_1: యీక్షింప (Ta - UUI - Indra) | నదిగాక (Sala - IIUI - Indra) | "
            "యిరువది (Nala - IIII - Indra) | నాల్గు (Ha/Gala - UI - Surya) | "
            "Yati: Matches: 'య' and 'య'. | Prasa: 2nd Letter 'క్షిం' matches 'క్షో'.\n"
            "line_2: యక్షోహి (Ta - UUI - Indra) | ణులుదాను (Sala - IIUI - Indra) | "
            "నటదండు (Sala - IIUI - Indra) | వచ్చి (Ha/Gala - UI - Surya) | "
            "Yati: Matches: 'య' and 'న'. | Prasa: 2nd Letter 'క్షో' matches 'క్షిం'."
        ),
    },
    "AN3": {
        "desc": "Meter detection",
        "user": (
            "క్రింది పద్యం ఏ ఛందస్సులో ఉంది? మీ సమాధానానికి కారణం చెప్పండి.\n\n"
            "పద్యం:\nమా రాముబాణనిర్మథితమాంసముల \nకీ రాదె నీ నాక మేల యిచ్చెదవు"
        ),
        "expected": (
            "ఛందస్సు: ద్విపద\n\n"
            "కారణం: ప్రతి పాదంలో 3 ఇంద్ర గణాలు + 1 సూర్య గణం ఉన్నాయి. "
            "యతి మైత్రి మరియు ప్రాస నియమాలు పాటించబడ్డాయి.\n\n"
            "విశ్లేషణ:\n"
            "line_1: మారాము (Ta - UUI - Indra) | బాణని (Ra - UIU - Indra) | "
            "ర్మథితమాం (Naga - IIIU - Indra) | సముల (Na - III - Surya) | "
            "Yati: Matches: 'మ' and 'ర'. | Prasa: 2nd Letter 'రా' matches 'రా'.\n"
            "line_2: కీరాదె (Ta - UUI - Indra) | నీనాక (Ta - UUI - Indra) | "
            "మేలయి (Ra - UIU - Indra) | చ్చెదవు (Na - III - Surya) | "
            "Yati: Matches: 'క' and 'మ'. | Prasa: 2nd Letter 'రా' matches 'రా'."
        ),
    },
    "AN4": {
        "desc": "Morphological gloss with sandhi breaks",
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
            "అడరి: అతిశయించి/విజృంభించి\n"
            "కారించి: వేధించి"
        ),
    },
    "AN5": {
        "desc": "Full analysis — WTW + Telugu bhavam",
        "user": (
            "క్రింది పద్యానికి సంపూర్ణ విశ్లేషణ చేయండి: సంధి విభజన, "
            "ప్రతిపదార్థం మరియు తెలుగు భావం ఇవ్వండి.\n\n"
            "పద్యం:\nయమ్మునిగొనిపోయి యర్ఘ్యపాద్యముల \nనెమ్మితో నిచ్చిన నృపుజూచి యతడు"
        ),
        "expected": (
            "ప్రతిపదార్థం:\n"
            "ఆ + మునిన్ (యమ్ముని): ఆ మునిని (విశ్వామిత్రుని)\n"
            "కొనిపోయి: (లోపలికి) తీసుకువెళ్ళి\n"
            "అర్ఘ్య + పాద్యములన్: అర్ఘ్యమును, పాద్యమును\n"
            "నెమ్మితోన్: ప్రేమతో/సంతోషముతో\n"
            "ఇచ్చిన: సమర్పించిన\n"
            "నృపున్ + చూచి (నృపుజూచి): రాజును (దశరథుని) చూసి\n"
            "అతడు: ఆ ముని (విశ్వామిత్రుడు)\n\n"
            "తెలుగు భావం: తెలుగు: ఆ మునిని లోపలికి తీసుకువెళ్ళి ప్రేమతో "
            "అర్ఘ్యపాద్యములను సమర్పించిన దశరథ మహారాజును చూసి ఆ విశ్వామిత్ర మహర్షి ఇట్లన్నాడు."
        ),
    },
    "AN6": {
        "desc": "Single line syllabification",
        "user": (
            "క్రింది పద్య పాదంలోని గురు-లఘు క్రమాన్ని మరియు గణాలను గుర్తించండి.\n\n"
            "పాదం: చరితంబు ధైర్యంబు శౌర్యంబు నతడు"
        ),
        "expected": (
            "విశ్లేషణ:\n"
            "చరితంబు (Sala - IIUI - Indra) | ధైర్యంబు (Ta - UUI - Indra) | "
            "శౌర్యంబు (Ta - UUI - Indra) | నతడు (Na - III - Surya)"
        ),
    },
}

@app.function(
    image=FINETUNING_GPU_IMAGE,
    gpu="a10g",
    volumes=VOLUME_CONFIG,
    timeout=1 * HOURS,
)
def evaluate_all_profiles(
    model_dir: str = "/vol/output/final_merged",
):
    from unsloth import FastLanguageModel
    import torch

    # Load model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_dir,
        max_seq_length=1024,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    SYSTEM = "You are a Telugu and Sanskrit scholar specialising in Dwipada poetry."

    # paste TEST_CASES dict here (or import it)

    results = {}
    for profile, case in TEST_CASES.items():
        print(f"\n{'='*60}")
        print(f"Profile: {profile} — {case['desc']}")
        print(f"{'='*60}")

        messages = [
            {"role": "system",    "content": SYSTEM},
            {"role": "user",      "content": case["user"]},
        ]
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to("cuda")

        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs,
                max_new_tokens=512,
                temperature=0.1,
                do_sample=True,
            )

        prediction = tokenizer.decode(
            outputs[0][inputs.shape[1]:],
            skip_special_tokens=True,
        )

        print(f"\n--- EXPECTED ---\n{case['expected']}")
        print(f"\n--- PREDICTED ---\n{prediction}")

        # Simple exact-match check on key structural elements
        expected_lines = set(case["expected"].strip().splitlines())
        predicted_lines = set(prediction.strip().splitlines())
        overlap = len(expected_lines & predicted_lines)
        score = overlap / max(len(expected_lines), 1)
        print(f"\nLine overlap score: {score:.1%}")

        results[profile] = {
            "prediction": prediction,
            "score": score,
        }

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for profile, r in results.items():
        print(f"{profile}: {r['score']:.1%}")

    return results

# =============================================================================
# § E  LOCAL ENTRYPOINT
# =============================================================================

@app.local_entrypoint()
def main():
    # 1. Upload your local data to the volume first
    # modal volume put metricalargs-vol ./ift_analysis_trl.jsonl /data/ift_analysis_trl.jsonl
    # fine_tune_metricalargs.remote()
    # push_to_hub.remote()
    evaluate_all_profiles.remote()