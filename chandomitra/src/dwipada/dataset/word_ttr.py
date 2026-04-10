#!/usr/bin/env python3
"""Word-level (literary) TTR analysis for Telugu dwipada poems.

Computes Type-Token Ratio at the word level (space-separated), not subword
tokens, providing a linguistically meaningful measure of lexical diversity.

Usage:
    python -m dwipada.dataset.word_ttr
    python -m dwipada.dataset.word_ttr --dataset path/to/data.jsonl --top-n 30
"""

import argparse
import json
import statistics
from collections import Counter
from datetime import datetime

from dwipada.dataset.validation_utils import (
    format_histogram,
    format_section_header,
    format_stat_line,
    write_report,
)
from dwipada.paths import DATASETS_DIR, PROJECT_ROOT

DEFAULT_DATASET = str(DATASETS_DIR / "dwipada_master_deduplicated.jsonl")
DEFAULT_OUTPUT = str(PROJECT_ROOT / "report" / "word_ttr_report.md")


def load_jsonl(filepath: str) -> list[dict]:
    """Load a JSONL file (one JSON object per line)."""
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def poem_to_words(poem: str) -> list[str]:
    """Split a poem into words: replace newlines with spaces, then split on whitespace."""
    return poem.replace("\n", " ").split()


def compute_ttr(words: list[str]) -> float:
    """Compute Type-Token Ratio: unique words / total words."""
    if not words:
        return 0.0
    return len(set(words)) / len(words)


def analyze_corpus(records: list[dict], top_n: int = 25) -> dict:
    """Run full word-level TTR analysis on the corpus."""
    per_poem_ttrs = []
    per_poem_word_counts = []
    total_lines = 0
    corpus_counter = Counter()
    source_data = {}  # source -> {"poems": int, "total": int, "unique": set}
    outliers = []  # (index, ttr, word_count, poem_text)
    skipped = 0

    for i, record in enumerate(records):
        poem = record.get("poem", "")
        if not poem or not poem.strip():
            skipped += 1
            continue

        words = poem_to_words(poem)
        if not words:
            skipped += 1
            continue

        ttr = compute_ttr(words)
        per_poem_ttrs.append(ttr)
        per_poem_word_counts.append(len(words))
        total_lines += len(poem.strip().split("\n"))
        corpus_counter.update(words)

        source = record.get("source", "unknown")
        if source not in source_data:
            source_data[source] = {"poems": 0, "total": 0, "unique": set()}
        source_data[source]["poems"] += 1
        source_data[source]["total"] += len(words)
        source_data[source]["unique"].update(words)

        truncated = poem.replace("\n", " ")
        if len(truncated) > 60:
            truncated = truncated[:57] + "..."
        outliers.append((i, ttr, len(words), truncated))

    total_words = sum(per_poem_word_counts)
    vocab_size = len(corpus_counter)
    num_poems = len(per_poem_ttrs)

    # Word lengths (character count)
    word_lengths = [len(w) for w in corpus_counter.elements()]

    # Hapax legomena
    hapax = [w for w, c in corpus_counter.items() if c == 1]

    # Sort outliers by TTR
    outliers.sort(key=lambda x: x[1])

    return {
        "num_poems": num_poems,
        "skipped": skipped,
        "total_words": total_words,
        "vocab_size": vocab_size,
        "total_lines": total_lines,
        "corpus_ttr": vocab_size / total_words if total_words > 0 else 0.0,
        "per_poem_ttrs": per_poem_ttrs,
        "per_poem_word_counts": per_poem_word_counts,
        "avg_words_per_poem": total_words / num_poems if num_poems > 0 else 0.0,
        "avg_words_per_line": total_words / total_lines if total_lines > 0 else 0.0,
        "hapax_count": len(hapax),
        "hapax_ratio": len(hapax) / vocab_size if vocab_size > 0 else 0.0,
        "top_words": corpus_counter.most_common(top_n),
        "word_lengths": word_lengths,
        "source_data": source_data,
        "lowest_ttr": outliers[:5],
        "highest_ttr": outliers[-5:][::-1],
    }


def build_report(results: dict, dataset_path: str) -> list[str]:
    """Build report lines from analysis results."""
    lines = []
    w = 80

    # Header
    lines.append("=" * w)
    lines.append("  WORD-LEVEL LITERARY TTR ANALYSIS")
    lines.append(f"  Dataset: {dataset_path}")
    lines.append(f"  Poems analysed: {results['num_poems']:,}")
    if results["skipped"]:
        lines.append(f"  Skipped (empty): {results['skipped']:,}")
    lines.append(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("=" * w)
    lines.append("")

    # Section 1: Corpus Overview
    lines.append(format_section_header("1. CORPUS OVERVIEW"))
    lines.append(format_stat_line("Total poems", results["num_poems"]))
    lines.append(format_stat_line("Total words", results["total_words"]))
    lines.append(format_stat_line("Vocabulary size (unique words)", results["vocab_size"]))
    lines.append(format_stat_line("Total lines", results["total_lines"]))
    lines.append(format_stat_line("Avg words per poem", results["avg_words_per_poem"]))
    lines.append(format_stat_line("Avg words per line", results["avg_words_per_line"]))
    lines.append("")

    # Section 2: Corpus-Level TTR
    lines.append(format_section_header("2. CORPUS-LEVEL TTR"))
    lines.append(format_stat_line("Corpus TTR (V/N)", results["corpus_ttr"]))
    lines.append(f"  (V = {results['vocab_size']:,} unique words / N = {results['total_words']:,} total words)")
    lines.append("")

    # Section 3: Per-Poem TTR Distribution
    ttrs = results["per_poem_ttrs"]
    lines.append(format_section_header("3. PER-POEM TTR DISTRIBUTION"))
    lines.append(format_stat_line("Mean TTR", statistics.mean(ttrs)))
    lines.append(format_stat_line("Median TTR", statistics.median(ttrs)))
    lines.append(format_stat_line("Min TTR", min(ttrs)))
    lines.append(format_stat_line("Max TTR", max(ttrs)))
    lines.append(format_stat_line("Std Dev", statistics.stdev(ttrs) if len(ttrs) > 1 else 0.0))
    lines.append("")
    lines.append("  Distribution:")
    edges = [0.0, 0.3, 0.5, 0.7, 0.85, 1.01]
    for h in format_histogram(ttrs, edges):
        lines.append(h)
    lines.append("")

    # Section 4: Hapax Legomena
    lines.append(format_section_header("4. HAPAX LEGOMENA"))
    lines.append(format_stat_line("Hapax count (freq=1)", results["hapax_count"]))
    lines.append(format_stat_line("Hapax / vocabulary", results["hapax_ratio"]))
    lines.append("")

    # Section 5: Top N Most Frequent Words
    lines.append(format_section_header(f"5. TOP {len(results['top_words'])} MOST FREQUENT WORDS"))
    lines.append(f"  {'Rank':<6} {'Word':<30} {'Freq':>8} {'%':>8}")
    lines.append(f"  {'─'*6} {'─'*30} {'─'*8} {'─'*8}")
    for rank, (word, freq) in enumerate(results["top_words"], 1):
        pct = freq / results["total_words"] * 100
        lines.append(f"  {rank:<6} {word:<30} {freq:>8,} {pct:>7.2f}%")
    lines.append("")

    # Section 6: Word Length Distribution
    wl = results["word_lengths"]
    lines.append(format_section_header("6. WORD LENGTH DISTRIBUTION (characters)"))
    lines.append(format_stat_line("Mean word length", statistics.mean(wl)))
    lines.append(format_stat_line("Median word length", statistics.median(wl)))
    lines.append("")
    lines.append("  Distribution:")
    wl_edges = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 30]
    for h in format_histogram(wl, wl_edges):
        lines.append(h)
    lines.append("")

    # Section 7: Per-Source TTR Breakdown
    lines.append(format_section_header("7. PER-SOURCE TTR BREAKDOWN"))
    src = results["source_data"]
    src_rows = []
    for name, d in sorted(src.items()):
        s_ttr = len(d["unique"]) / d["total"] if d["total"] > 0 else 0.0
        src_rows.append((name, d["poems"], d["total"], len(d["unique"]), s_ttr))
    src_rows.sort(key=lambda x: x[4], reverse=True)

    lines.append(f"  {'Source':<35} {'Poems':>7} {'Words':>9} {'Unique':>8} {'TTR':>8}")
    lines.append(f"  {'─'*35} {'─'*7} {'─'*9} {'─'*8} {'─'*8}")
    for name, poems, total, unique, s_ttr in src_rows:
        lines.append(f"  {name:<35} {poems:>7,} {total:>9,} {unique:>8,} {s_ttr:>8.4f}")
    lines.append("")

    # Section 8: TTR Outliers
    lines.append(format_section_header("8. TTR OUTLIERS"))
    lines.append("")
    lines.append("  Lowest TTR (most repetitive):")
    lines.append(f"  {'Index':>7} {'TTR':>8} {'Words':>7}  Poem")
    lines.append(f"  {'─'*7} {'─'*8} {'─'*7}  {'─'*40}")
    for idx, ttr, wc, text in results["lowest_ttr"]:
        lines.append(f"  {idx:>7} {ttr:>8.4f} {wc:>7}  {text}")
    lines.append("")
    lines.append("  Highest TTR (most diverse):")
    lines.append(f"  {'Index':>7} {'TTR':>8} {'Words':>7}  Poem")
    lines.append(f"  {'─'*7} {'─'*8} {'─'*7}  {'─'*40}")
    for idx, ttr, wc, text in results["highest_ttr"]:
        lines.append(f"  {idx:>7} {ttr:>8.4f} {wc:>7}  {text}")
    lines.append("")

    return lines


def main():
    parser = argparse.ArgumentParser(
        description="Word-level literary TTR analysis for Telugu dwipada poems."
    )
    parser.add_argument(
        "--dataset", "-d", default=DEFAULT_DATASET,
        help="Path to the JSONL dataset file.",
    )
    parser.add_argument(
        "--output", "-o", default=DEFAULT_OUTPUT,
        help="Path for the output report.",
    )
    parser.add_argument(
        "--top-n", type=int, default=25,
        help="Number of most frequent words to show (default: 25).",
    )
    args = parser.parse_args()

    print(f"Loading dataset: {args.dataset}")
    records = load_jsonl(args.dataset)
    print(f"Loaded {len(records):,} records.")

    print("Analysing word-level TTR...")
    results = analyze_corpus(records, top_n=args.top_n)

    report_lines = build_report(results, args.dataset)
    write_report(report_lines, args.output)


if __name__ == "__main__":
    main()
