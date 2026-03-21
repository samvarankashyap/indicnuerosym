#!/usr/bin/env python3
"""
Dwipada Generation Benchmark

Generates Telugu dwipada poems across base and fine-tuned models,
analyzes each for metrical correctness, and produces a comparison report.

Usage:
    python inference_scripts/benchmark.py                        # run all models
    python inference_scripts/benchmark.py --models "gemma-3-1b-it (base)"  # single model
    python inference_scripts/benchmark.py --seed 123             # custom seed
"""

import argparse
import gc
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dwipada.core import analyze_dwipada, format_analysis_report
from dwipada.core.constants import DWIPADA_RULES_BLOCK

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"

MODEL_CONFIGS = [
    {
        "name": "gemma-3-1b-it (base)",
        "base_model": "google/gemma-3-1b-it",
        "adapter_path": None,
    },
    {
        "name": "gemma-3-4b-it (base)",
        "base_model": "google/gemma-3-4b-it",
        "adapter_path": None,
    },
    {
        "name": "gemma3-1b-lora-best",
        "base_model": "google/gemma-3-1b-it",
        "adapter_path": str(CHECKPOINTS_DIR / "gemma3-1b-dwipada-lora" / "best_checkpoint_epoch3.918"),
    },
]

PROMPTS = [
    "ద్విపదలో ఒక పద్యం వ్రాయండి.",
    "ద్విపదలో ఒక పద్యం వ్రాయండి.",
    "ద్విపదలో ఒక పద్యం వ్రాయండి.",
    "ద్విపదలో ఒక పద్యం వ్రాయండి.",
    "ద్విపదలో ఒక పద్యం వ్రాయండి.",
]

GENERATION_PARAMS = {
    "max_new_tokens": 256,
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 50,
    "repetition_penalty": 1.1,
    "do_sample": True,
}

# Telugu Unicode range
TELUGU_RE = re.compile(r"[\u0C00-\u0C7F]")


# ─────────────────────────────────────────────────────────────────────────────
# MODEL MANAGEMENT
# ─────────────────────────────────────────────────────────────────────────────

def load_base_model(base_model_id):
    """Load a base model and tokenizer."""
    print(f"  Loading base model: {base_model_id}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    )
    model.eval()
    return model, tokenizer


def attach_adapter(base_model, adapter_path, adapter_name):
    """Wrap base model with a LoRA adapter (or load additional adapter)."""
    from peft import PeftModel

    print(f"  Loading LoRA adapter: {adapter_path}")
    if isinstance(base_model, PeftModel):
        base_model.load_adapter(adapter_path, adapter_name=adapter_name)
        base_model.set_adapter(adapter_name)
    else:
        base_model = PeftModel.from_pretrained(
            base_model, adapter_path, adapter_name=adapter_name,
        )
    base_model.eval()
    return base_model


def detach_adapters(model):
    """Remove all adapters, returning the unwrapped base model."""
    from peft import PeftModel

    if isinstance(model, PeftModel):
        model = model.unload()
    return model


def unload_model(model, tokenizer):
    """Free GPU memory."""
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ─────────────────────────────────────────────────────────────────────────────
# GENERATION & ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def generate_poem(model, tokenizer, prompt):
    """Generate a poem using the Gemma chat template."""
    prompt_text = (
        f"{DWIPADA_RULES_BLOCK}\n\n{prompt}\n\n"
        "Output only the two-line poem. No explanations, titles, or extra text."
    )
    formatted = (
        f"<start_of_turn>user\n{prompt_text}<end_of_turn>\n"
        f"<start_of_turn>model\n"
    )

    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, **GENERATION_PARAMS)

    # Decode only newly generated tokens
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(generated, skip_special_tokens=False)

    # Strip at end-of-turn
    if "<end_of_turn>" in text:
        text = text[: text.index("<end_of_turn>")]

    return text.strip()


def extract_poem_lines(raw_output):
    """Extract the first 2 Telugu lines from raw model output.

    Returns the 2-line poem string, or None if not enough Telugu lines found.
    """
    telugu_lines = []
    for line in raw_output.split("\n"):
        line = line.strip()
        if line and TELUGU_RE.search(line):
            telugu_lines.append(line)
        if len(telugu_lines) == 2:
            break

    if len(telugu_lines) < 2:
        return None
    return "\n".join(telugu_lines)


def analyze_poem(poem_text):
    """Run the dwipada meter analyzer. Returns a result dict."""
    if poem_text is None:
        return {
            "overall_score": 0.0,
            "is_valid": False,
            "gana_score": 0.0,
            "prasa_score": 0.0,
            "yati_score": 0.0,
            "error": "generation_failure",
        }

    try:
        analysis = analyze_dwipada(poem_text)
        breakdown = analysis["match_score"].get("breakdown", {})

        # Extract component scores
        gana_avg = breakdown.get("gana_average", 0.0)
        prasa = breakdown.get("prasa", 0.0)
        yati_avg = breakdown.get("yati_average", 0.0)

        return {
            "overall_score": analysis["match_score"]["overall"],
            "is_valid": analysis["is_valid_dwipada"],
            "gana_score": gana_avg,
            "prasa_score": prasa,
            "yati_score": yati_avg,
            "report": format_analysis_report(analysis),
            "error": None,
        }
    except Exception as e:
        return {
            "overall_score": 0.0,
            "is_valid": False,
            "gana_score": 0.0,
            "prasa_score": 0.0,
            "yati_score": 0.0,
            "error": str(e),
        }


# ─────────────────────────────────────────────────────────────────────────────
# BENCHMARK RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def _error_results(model_name, prompts, error_msg):
    """Generate error placeholder results for all prompts."""
    return [
        {
            "model": model_name,
            "prompt": prompt,
            "raw_output": None,
            "extracted_poem": None,
            "analysis": {
                "overall_score": 0.0,
                "is_valid": False,
                "gana_score": 0.0,
                "prasa_score": 0.0,
                "yati_score": 0.0,
                "error": f"model_load_failure: {error_msg}",
            },
        }
        for prompt in prompts
    ]


def _run_prompts(model, tokenizer, model_name, prompts, seed):
    """Run all prompts against a loaded model and return results."""
    results = []
    for i, prompt in enumerate(prompts, 1):
        print(f"  [{i}/{len(prompts)}] {prompt[:50]}...")

        run_seed = seed + i - 1
        torch.manual_seed(run_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(run_seed)

        try:
            raw_output = generate_poem(model, tokenizer, prompt)
        except torch.cuda.OutOfMemoryError:
            print(f"    OOM — skipping")
            raw_output = None
        except Exception as e:
            print(f"    Generation error: {e}")
            raw_output = None

        extracted = extract_poem_lines(raw_output) if raw_output else None
        analysis = analyze_poem(extracted)

        score_str = f"{analysis['overall_score']:.1f}%"
        valid_str = "VALID" if analysis["is_valid"] else "invalid"
        print(f"    Score: {score_str} ({valid_str})")

        results.append({
            "model": model_name,
            "prompt": prompt,
            "raw_output": raw_output,
            "extracted_poem": extracted,
            "analysis": analysis,
        })
    return results


def run_benchmark(model_configs, prompts, seed=42):
    """Run the full benchmark, loading each base model once and swapping adapters."""
    results = []

    # Group configs by base model, preserving order
    from collections import OrderedDict

    groups = OrderedDict()
    for cfg in model_configs:
        groups.setdefault(cfg["base_model"], []).append(cfg)

    for base_model_id, cfgs in groups.items():
        print(f"\n{'#'*60}")
        print(f"Base model: {base_model_id} ({len(cfgs)} config(s))")
        print(f"{'#'*60}")

        try:
            base_model, tokenizer = load_base_model(base_model_id)
        except Exception as e:
            print(f"  ERROR loading base model: {e}")
            for cfg in cfgs:
                results.extend(_error_results(cfg["name"], prompts, e))
            continue

        model = base_model
        for cfg in cfgs:
            model_name = cfg["name"]
            print(f"\n{'='*60}")
            print(f"Model: {model_name}")
            print(f"{'='*60}")

            if cfg["adapter_path"]:
                try:
                    model = attach_adapter(model, cfg["adapter_path"], model_name)
                except Exception as e:
                    print(f"  ERROR loading adapter: {e}")
                    results.extend(_error_results(model_name, prompts, e))
                    continue

            results.extend(_run_prompts(model, tokenizer, model_name, prompts, seed))

            # After running, remove adapter to restore base for the next config
            if cfg["adapter_path"]:
                model = detach_adapters(model)

        unload_model(model, tokenizer)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# REPORT GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def generate_report(results, output_dir):
    """Generate Markdown report and JSON data file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Compute per-model summaries ──
    model_stats = {}
    for r in results:
        name = r["model"]
        if name not in model_stats:
            model_stats[name] = {
                "scores": [],
                "valid_count": 0,
                "total": 0,
                "gen_failures": 0,
                "gana_scores": [],
                "prasa_scores": [],
                "yati_scores": [],
            }
        s = model_stats[name]
        s["total"] += 1
        a = r["analysis"]
        s["scores"].append(a["overall_score"])
        s["gana_scores"].append(a["gana_score"])
        s["prasa_scores"].append(a["prasa_score"])
        s["yati_scores"].append(a["yati_score"])
        if a["is_valid"]:
            s["valid_count"] += 1
        if a.get("error") == "generation_failure":
            s["gen_failures"] += 1

    # ── Markdown report ──
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# Dwipada Generation Benchmark Report",
        f"Generated: {timestamp}",
        "",
        f"Models: {len(model_stats)} | Prompts per model: {len(PROMPTS)} | Total generations: {len(results)}",
        "",
        "---",
        "",
        "## Summary",
        "",
        "| Model | Avg Score | % Valid | Avg Gana | Avg Prasa | Avg Yati | Gen Failures |",
        "|-------|-----------|---------|----------|-----------|----------|--------------|",
    ]

    # Preserve model order from results
    seen = []
    for r in results:
        if r["model"] not in seen:
            seen.append(r["model"])

    for name in seen:
        s = model_stats[name]
        avg = lambda lst: sum(lst) / len(lst) if lst else 0.0
        valid_pct = (s["valid_count"] / s["total"] * 100) if s["total"] else 0.0
        lines.append(
            f"| {name} "
            f"| {avg(s['scores']):.1f}% "
            f"| {valid_pct:.0f}% ({s['valid_count']}/{s['total']}) "
            f"| {avg(s['gana_scores']):.1f}% "
            f"| {avg(s['prasa_scores']):.1f}% "
            f"| {avg(s['yati_scores']):.1f}% "
            f"| {s['gen_failures']} |"
        )

    lines += ["", "---", ""]

    # ── Detailed results per model ──
    lines.append("## Detailed Results")
    lines.append("")

    current_model = None
    for i, r in enumerate(results):
        if r["model"] != current_model:
            current_model = r["model"]
            lines.append(f"### {current_model}")
            lines.append("")

        prompt_num = (i % len(PROMPTS)) + 1
        a = r["analysis"]

        lines.append(f"**Prompt {prompt_num}:** {r['prompt']}")
        lines.append("")

        if r["raw_output"]:
            lines.append("**Full response:**")
            lines.append("```")
            lines.append(r["raw_output"])
            lines.append("```")
            if r["extracted_poem"]:
                lines.append("")
                lines.append("**Extracted poem:**")
                lines.append("```")
                lines.append(r["extracted_poem"])
                lines.append("```")
            else:
                lines.append("*(Could not extract valid 2-line Telugu poem)*")
        else:
            lines.append("*(No output generated)*")

        lines.append("")

        status = "VALID" if a["is_valid"] else "invalid"
        error_note = f" | Error: {a['error']}" if a.get("error") else ""
        lines.append(
            f"Score: **{a['overall_score']:.1f}%** ({status}) | "
            f"Gana: {a['gana_score']:.1f}% | "
            f"Prasa: {a['prasa_score']:.1f}% | "
            f"Yati: {a['yati_score']:.1f}%{error_note}"
        )
        lines.append("")
        lines.append("---")
        lines.append("")

    # Write markdown
    report_path = output_dir / "benchmark_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport saved to: {report_path}")

    # ── JSON data ──
    json_results = []
    for r in results:
        entry = {
            "model": r["model"],
            "prompt": r["prompt"],
            "raw_output": r["raw_output"],
            "extracted_poem": r["extracted_poem"],
            "overall_score": r["analysis"]["overall_score"],
            "is_valid": r["analysis"]["is_valid"],
            "gana_score": r["analysis"]["gana_score"],
            "prasa_score": r["analysis"]["prasa_score"],
            "yati_score": r["analysis"]["yati_score"],
            "error": r["analysis"].get("error"),
        }
        json_results.append(entry)

    json_path = output_dir / "benchmark_results.json"
    json_path.write_text(
        json.dumps(json_results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"JSON data saved to: {json_path}")

    return report_path, json_path


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark dwipada poem generation across models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated model names to run (default: all). "
        'Example: --models "gemma-3-1b-it (base),gemma3-trl"',
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "inference_scripts"),
        help="Directory for report output (default: inference_scripts/)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Filter models if specified
    configs = MODEL_CONFIGS
    if args.models:
        selected = [m.strip() for m in args.models.split(",")]
        configs = [c for c in MODEL_CONFIGS if c["name"] in selected]
        if not configs:
            print(f"Error: No matching models found. Available: {[c['name'] for c in MODEL_CONFIGS]}")
            sys.exit(1)

    print(f"Dwipada Generation Benchmark")
    print(f"Models: {[c['name'] for c in configs]}")
    print(f"Prompts: {len(PROMPTS)}")
    print(f"Seed: {args.seed}")

    start = time.time()
    results = run_benchmark(configs, PROMPTS, seed=args.seed)
    elapsed = time.time() - start

    print(f"\nBenchmark completed in {elapsed:.0f}s")

    generate_report(results, args.output_dir)


if __name__ == "__main__":
    main()
