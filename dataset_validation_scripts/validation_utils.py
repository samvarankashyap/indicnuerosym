#!/usr/bin/env python3
"""
Shared utilities for the dataset validation pipeline.

Provides helpers for:
- Path resolution (independent of dwipada package)
- Loading datasets (JSON and JSONL)
- Sentence embedding model loading (LaBSE, mSBERT-mpnet, L3Cube-IndicSBERT)
- Batch encoding for semantic similarity
- Gemma 3 tokenizer loading for lexical diversity analysis
- Report formatting and output
"""

import json
from pathlib import Path

import numpy as np


# ── Path Resolution ─────────────────────────────────────────────────────────

def _find_project_root() -> Path:
    """Walk up from this file to find the project root (contains pyproject.toml)."""
    current = Path(__file__).resolve().parent
    for parent in [current] + list(current.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("Could not find project root (no pyproject.toml found)")


PROJECT_ROOT = _find_project_root()
DATASETS_DIR = PROJECT_ROOT / "datasets"
SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
DEFAULT_DATASET = str(DATASETS_DIR / "dwipada_augmented_dataset.json")


# ── Dataset Loading ──────────────────────────────────────────────────────────

def load_dataset(filepath: str) -> list[dict]:
    """Load a JSON dataset file and return the list of record dicts.

    Args:
        filepath: Path to a JSON file containing an array of records.

    Returns:
        List of record dictionaries.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def load_jsonl(filepath: str) -> list[dict]:
    """Load a JSONL file (one JSON object per line)."""
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ── Sentence Embedding Models (Level 3) ────────────────────────────────────

# Model registry: (display_name, model_id)
# - LaBSE: best cross-lingual (Telugu<->English), gap=0.39
# - mSBERT-mpnet: strong within-language discrimination, gap=0.58
# - L3Cube-IndicSBERT: best Indic within-language discrimination, gap=0.72
LEVEL3_MODELS = [
    ("LaBSE", "sentence-transformers/LaBSE"),
    ("mSBERT-mpnet", "paraphrase-multilingual-mpnet-base-v2"),
    ("L3Cube-IndicSBERT", "l3cube-pune/indic-sentence-bert-nli"),
]


def _load_sbert_model(model_id: str, display_name: str):
    """Load a SentenceTransformer model by ID."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError(
            "sentence-transformers is required for Level 3 (semantic fidelity).\n"
            "Install it with: pip install sentence-transformers"
        )
    print(f"Loading {display_name} ({model_id})...")
    model = SentenceTransformer(model_id)
    print(f"{display_name} loaded.")
    return model


def load_labse_model():
    """Load the sentence-transformers/LaBSE model for multilingual embeddings."""
    return _load_sbert_model("sentence-transformers/LaBSE", "LaBSE")


def load_msbert_mpnet_model():
    """Load the paraphrase-multilingual-mpnet-base-v2 model."""
    return _load_sbert_model("paraphrase-multilingual-mpnet-base-v2", "mSBERT-mpnet")


def load_indic_sbert_model():
    """Load the l3cube-pune/indic-sentence-bert-nli model."""
    return _load_sbert_model("l3cube-pune/indic-sentence-bert-nli", "L3Cube-IndicSBERT")


def batch_encode(model, texts: list[str], batch_size: int = 256) -> np.ndarray:
    """Encode a list of texts into embeddings using the given model.

    Returns:
        numpy array of shape (len(texts), embedding_dim).
    """
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    return embeddings


def cosine_similarity_paired(emb_a: np.ndarray, emb_b: np.ndarray) -> np.ndarray:
    """Compute row-wise cosine similarity between two embedding matrices.

    Returns:
        1D numpy array of length N with cosine similarities in [-1, 1].
    """
    norm_a = emb_a / (np.linalg.norm(emb_a, axis=1, keepdims=True) + 1e-10)
    norm_b = emb_b / (np.linalg.norm(emb_b, axis=1, keepdims=True) + 1e-10)
    return np.sum(norm_a * norm_b, axis=1)


# ── Gemma 3 Tokenizer (Level 4) ─────────────────────────────────────────────

def load_gemma_tokenizer():
    """Load the Gemma 3 tokenizer for TTR computation."""
    from transformers import AutoTokenizer

    model_id = "google/gemma-3-4b-it"
    print(f"Loading Gemma 3 tokenizer ({model_id})...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    print(f"Tokenizer loaded. Vocabulary size: {tokenizer.vocab_size}")
    return tokenizer


def count_telugu_tokens(tokenizer) -> int:
    """Count the number of Telugu tokens in the tokenizer's vocabulary."""
    import re
    telugu_re = re.compile(r'[\u0C00-\u0C7F]')
    count = 0
    for i in range(tokenizer.vocab_size):
        if telugu_re.search(tokenizer.decode([i])):
            count += 1
    return count


def gemma_tokenize(tokenizer, text: str) -> list[int]:
    """Tokenize text using the Gemma 3 tokenizer (no special tokens)."""
    return tokenizer.encode(text, add_special_tokens=False)


# ── Report Formatting ────────────────────────────────────────────────────────

def format_section_header(title: str, width: int = 80) -> str:
    """Return a formatted section header with -- borders."""
    prefix = f"── {title} "
    return prefix + "─" * max(0, width - len(prefix))


def format_stat_line(label: str, value, total: int = None, label_width: int = 30) -> str:
    """Format a single statistic line."""
    if total is not None:
        pct = value / total * 100 if total > 0 else 0.0
        return f"  {label:<{label_width}} {value:>8,} / {total:>8,}  ({pct:.1f}%)"
    elif isinstance(value, float):
        return f"  {label:<{label_width}} {value:.4f}"
    else:
        return f"  {label:<{label_width}} {value}"


def format_histogram(values: list[float], edges: list[float], width: int = 40) -> list[str]:
    """Create a text histogram from a list of float values."""
    counts = [0] * (len(edges) - 1)
    for v in values:
        for i in range(len(edges) - 1):
            if edges[i] <= v < edges[i + 1]:
                counts[i] += 1
                break
        else:
            if v >= edges[-1]:
                counts[-1] += 1

    max_count = max(counts) if counts else 1
    lines = []
    for i in range(len(counts)):
        lo = edges[i]
        hi = edges[i + 1]
        label = f"  {lo:>5.2f} - {hi:>5.2f}"
        bar_len = int(counts[i] / max_count * width) if max_count > 0 else 0
        bar = "\u2588" * bar_len
        lines.append(f"{label}:  {counts[i]:>7,}  {bar}")
    return lines


def write_report(report_lines: list[str], output_path: str):
    """Write report lines to a file and also print to stdout."""
    report_text = "\n".join(report_lines)
    print(report_text)

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(report_text + "\n")
    print(f"\nReport saved to: {path}")


# ── Metadata Helper ─────────────────────────────────────────────────────────

def save_results(results: dict, output_path: str, meta: dict):
    """Save results dict with _meta block to a JSON file."""
    output = {"_meta": meta}
    output.update(results)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False)
    print(f"  Results saved to: {path}")


def load_results(filepath: str) -> dict | None:
    """Load a results JSON file. Returns None if file doesn't exist."""
    path = Path(filepath)
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
