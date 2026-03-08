"""
Dwipada IFT Format Converter
==============================
Reads training_ready.jsonl (output of dwipada_pipeline.py) and writes
three IFT-ready files, one per framework format:

  ┌─────────────────────┬──────────────────────────────────────────┐
  │ File                │ Framework                                │
  ├─────────────────────┼──────────────────────────────────────────┤
  │ ift_alpaca.jsonl    │ Unsloth, Axolotl (alpaca), TRL alapaca  │
  │ ift_sharegpt.jsonl  │ Axolotl (sharegpt), LLaMA-Factory       │
  │ ift_trl.jsonl       │ HuggingFace TRL SFTTrainer (native)     │
  └─────────────────────┴──────────────────────────────────────────┘

Each format also preserves: profile_id, source_tag, thin_wtw, wtw_len
so you can filter/debug by profile after training.

Usage:
    python dwipada_ift_converter.py --input training_ready.jsonl
    python dwipada_ift_converter.py --input training_ready.jsonl --outdir ./ift_data
"""

import json
import argparse
from pathlib import Path


# ──────────────────────────────────────────────
# LOADERS
# ──────────────────────────────────────────────

def load_jsonl(path: str) -> list[dict]:
    """Handles both JSON array (.json) and JSONL (.jsonl) formats."""
    with open(path, encoding="utf-8") as f:
        first_char = f.read(1)
        f.seek(0)
        if first_char == "[":
            data = json.load(f)
            if not isinstance(data, list):
                raise ValueError(f"{path} must be a JSON array.")
            return data
        else:
            return [json.loads(line) for line in f if line.strip()]


def save_jsonl(records: list[dict], path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"  ✓ Saved {len(records):>6} records → {path}")


# ──────────────────────────────────────────────
# FORMAT CONVERTERS
# ──────────────────────────────────────────────

def to_alpaca(rec: dict) -> dict:
    """
    Alpaca format — used by Unsloth, Axolotl (alpaca template), TRL alpaca.

    Fields:
      instruction : the user prompt  (maps to rec["input"])
      input       : optional context — empty string if not used
      output      : the expected model response

    The 'instruction' here is the full prompt including the source/profile
    tags and the poem. Alpaca templates typically combine instruction+input
    at training time; since our instruction already contains the poem we
    leave 'input' empty.

    Reference template:
      ### Instruction: {instruction}
      ### Input: {input}          ← omitted when empty by most trainers
      ### Response: {output}
    """
    return {
        "instruction": rec.get("input", ""),
        "input":       "",                       # poem is already in instruction
        "output":      rec.get("output", ""),
        # ── metadata (stripped by trainer, useful for debugging) ──
        "_profile_id": rec.get("profile_id"),
        "_source_tag": rec.get("source_tag"),
        "_thin_wtw":   rec.get("thin_wtw"),
        "_wtw_len":    rec.get("wtw_len"),
        "_source":     rec.get("source", ""),
    }


def to_sharegpt(rec: dict) -> dict:
    """
    ShareGPT / conversation format — used by Axolotl (sharegpt template),
    LLaMA-Factory, and OpenAI fine-tune API.

    Fields:
      conversations: list of turn dicts
        {"from": "system",  "value": "..."}   ← optional system prompt
        {"from": "human",   "value": "..."}   ← user turn
        {"from": "gpt",     "value": "..."}   ← assistant response

    We extract the system persona (the [Human_Style]/[Scholarly] tag line)
    as a system turn, and the rest of the instruction + poem as the human turn.
    This gives the model a cleaner signal: system = who you are,
    human = what the user asked, gpt = what you respond.
    """
    full_input = rec.get("input", "")
    output     = rec.get("output", "")

    # Split the source/profile tag line from the rest of the instruction
    # Format: "[Human_Style] [Educational]\n<actual instruction>\n\n<poem>"
    lines = full_input.split("\n", 1)
    if len(lines) == 2 and lines[0].startswith("["):
        system_line  = lines[0].strip()   # e.g. "[Human_Style] [Educational]"
        human_turn   = lines[1].strip()   # instruction + poem
    else:
        system_line  = ""
        human_turn   = full_input.strip()

    conversations = []
    if system_line:
        conversations.append({"from": "system", "value": system_line})
    conversations.append({"from": "human", "value": human_turn})
    conversations.append({"from": "gpt",   "value": output})

    return {
        "conversations": conversations,
        # ── metadata ──
        "_profile_id": rec.get("profile_id"),
        "_source_tag": rec.get("source_tag"),
        "_thin_wtw":   rec.get("thin_wtw"),
        "_wtw_len":    rec.get("wtw_len"),
        "_source":     rec.get("source", ""),
    }


def to_trl(rec: dict) -> dict:
    """
    HuggingFace TRL SFTTrainer native format.

    SFTTrainer accepts a 'text' field containing the fully formatted string,
    OR a 'messages' field in OpenAI chat format. We use 'messages' because
    it works with both SFTTrainer's apply_chat_template() and the newer
    trl.DataCollatorForSeq2Seq approach.

    Fields:
      messages: list of OpenAI-style message dicts
        {"role": "system",    "content": "..."}
        {"role": "user",      "content": "..."}
        {"role": "assistant", "content": "..."}

    SFTTrainer will call tokenizer.apply_chat_template(messages) internally,
    so this works for any model that has a chat template (Gemma 3 does).
    """
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
        # ── metadata ──
        "_profile_id": rec.get("profile_id"),
        "_source_tag": rec.get("source_tag"),
        "_thin_wtw":   rec.get("thin_wtw"),
        "_wtw_len":    rec.get("wtw_len"),
        "_source":     rec.get("source", ""),
    }


# ──────────────────────────────────────────────
# STATS PRINTER
# ──────────────────────────────────────────────

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
# MAIN
# ──────────────────────────────────────────────

def run(input_path: str, outdir: str):
    print(f"\n[Loading] {input_path}")
    records = load_jsonl(input_path)
    print(f"  Loaded {len(records)} records.")

    alpaca_records   = [to_alpaca(r)   for r in records]
    sharegpt_records = [to_sharegpt(r) for r in records]
    trl_records      = [to_trl(r)      for r in records]

    out = Path(outdir)
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
    print("\n✅ Done.")


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dwipada IFT Format Converter")
    parser.add_argument(
        "--input",  required=True,
        help="Path to training_ready.jsonl from dwipada_pipeline.py"
    )
    parser.add_argument(
        "--outdir", default=".",
        help="Output directory for IFT files (default: current directory)"
    )
    args = parser.parse_args()
    run(args.input, args.outdir)
