"""Unified CLI for the dwipada package.

Usage:
    python -m dwipada <command> [options]
    dwipada <command> [options]

Commands:
    crawl       Crawl poetry sources
    clean       Clean crawled data
    consolidate Consolidate text files to JSON
    stats       Dataset statistics
    augment     Augment dataset with chandassu analysis
    combine     Combine real + synthetic datasets
    validate    4-level dataset validation
    prepare     Prepare training data
    train       Fine-tune Gemma model with LoRA
    generate    Generate poems with trained model
    generate-constrained  Generate poems with constrained decoding
    batch       Gemini batch API operations
    batch-vertex Vertex AI batch operations
    analyze     Analyze a dwipada poem
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        prog="dwipada",
        description="Telugu Dwipada Poetry Generation Toolkit",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- Crawl ---
    crawl_parser = subparsers.add_parser("crawl", help="Crawl poetry sources")
    crawl_parser.add_argument(
        "source",
        choices=[
            "basava_puranam", "dwipada_bhagavatam",
            "ranganatha_ramayanam", "palanati_veera_charitra",
            "srirama_parinayamu",
        ],
        help="Source to crawl",
    )

    # --- Clean ---
    clean_parser = subparsers.add_parser("clean", help="Clean crawled data")
    clean_parser.add_argument(
        "source",
        choices=[
            "basava_puranam", "dwipada_bhagavatam",
            "palanati_veera_charitra", "srirama_parinayamu",
            "poems",
        ],
        help="Source to clean",
    )

    # --- Consolidate ---
    subparsers.add_parser("consolidate", help="Consolidate text files to JSON")

    # --- Stats ---
    stats_parser = subparsers.add_parser("stats", help="Dataset statistics")
    stats_parser.add_argument("--input", help="Input JSON file")
    stats_parser.add_argument("--by-source", action="store_true", help="Show per-source breakdown")
    stats_parser.add_argument("--write-filtered", help="Write filtered perfect dataset to file")
    stats_parser.add_argument("--exclude", nargs="*", help="Sources to exclude")

    # --- Augment ---
    subparsers.add_parser("augment", help="Augment dataset with chandassu analysis")

    # --- Combine ---
    subparsers.add_parser("combine", help="Combine real + synthetic datasets")

    # --- Prepare synthetic ---
    subparsers.add_parser("prepare-synthetic", help="Prepare synthetic dataset")

    # --- Validate ---
    validate_parser = subparsers.add_parser("validate", help="4-level dataset validation")
    validate_parser.add_argument("--dataset", help="Path to dataset JSON")
    validate_parser.add_argument("--skip-levels", nargs="*", type=int, help="Levels to skip")
    validate_parser.add_argument("--limit", type=int, help="Max records to validate")
    validate_parser.add_argument("--fresh", action="store_true", help="Ignore checkpoints")

    # --- Prepare training data ---
    subparsers.add_parser("prepare", help="Prepare training data")

    # --- Train ---
    train_parser = subparsers.add_parser("train", help="Fine-tune Gemma model")
    train_parser.add_argument("--model", default="google/gemma-3-1b-it", help="Base model")
    train_parser.add_argument("--epochs", type=int, default=3)
    train_parser.add_argument("--batch-size", type=int, default=2)
    train_parser.add_argument("--lora-rank", type=int, default=16)
    train_parser.add_argument("--merge", action="store_true", help="Merge adapter after training")

    # --- Generate ---
    gen_parser = subparsers.add_parser("generate", help="Generate poems")
    gen_parser.add_argument("prompt", nargs="?", help="Prompt for poem generation")
    gen_parser.add_argument("--interactive", action="store_true")
    gen_parser.add_argument("--batch", help="Batch prompts file")
    gen_parser.add_argument("--num-poems", type=int, default=1)
    gen_parser.add_argument("--no-validate", action="store_true")

    # --- Batch ---
    batch_parser = subparsers.add_parser("batch", help="Gemini batch operations")
    batch_parser.add_argument("--prepare", help="Prepare N or START-END requests")
    batch_parser.add_argument("--submit", help="Submit batch JSONL file")
    batch_parser.add_argument("--status", help="Check batch job status")
    batch_parser.add_argument("--results", help="Download batch results")

    # --- Batch Vertex ---
    subparsers.add_parser("batch-vertex", help="Vertex AI batch operations")

    # --- Generate Constrained ---
    gen_c_parser = subparsers.add_parser(
        "generate-constrained", help="Generate poems with constrained decoding"
    )
    gen_c_parser.add_argument("prompt", nargs="?", help="Prompt for poem generation")
    gen_c_parser.add_argument("--interactive", action="store_true")
    gen_c_parser.add_argument("--batch", help="Batch prompts file")
    gen_c_parser.add_argument("--base-model", default="google/gemma-3-1b-it")
    gen_c_parser.add_argument("--adapter", help="Path to LoRA adapter directory")
    gen_c_parser.add_argument("--merged-model", help="Path to merged model")
    gen_c_parser.add_argument("--top-k-constraint", type=int, default=25)
    gen_c_parser.add_argument("--no-prasa", action="store_true")
    gen_c_parser.add_argument("--no-yati", action="store_true")
    gen_c_parser.add_argument("--do-sample", action="store_true")
    gen_c_parser.add_argument("--temperature", type=float, default=0.7)
    gen_c_parser.add_argument("--no-validate", action="store_true")

    # --- Analyze ---
    analyze_parser = subparsers.add_parser("analyze", help="Analyze a dwipada poem")
    analyze_parser.add_argument("poem", nargs="?", help="Poem text (2 lines separated by newline)")
    analyze_parser.add_argument("--file", help="Read poem from file")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    _dispatch(args)


def _dispatch(args):
    """Route CLI commands to their implementations."""
    if args.command == "crawl":
        crawler = __import__(
            f"dwipada.data.crawlers.{args.source}", fromlist=["main"]
        )
        crawler.main()

    elif args.command == "clean":
        cleaner = __import__(
            f"dwipada.data.cleaners.{args.source}", fromlist=["main"]
        )
        cleaner.main()

    elif args.command == "consolidate":
        from dwipada.data.consolidate import main
        main()

    elif args.command == "stats":
        from dwipada.dataset.stats import main as stats_main
        sys.argv = ["dwipada stats"]
        if args.input:
            sys.argv.extend(["--input", args.input])
        if args.by_source:
            sys.argv.append("--by-source")
        if args.write_filtered:
            sys.argv.extend(["--write-filtered", args.write_filtered])
        if args.exclude:
            sys.argv.extend(["--exclude"] + args.exclude)
        stats_main()

    elif args.command == "augment":
        from dwipada.dataset.augment import main
        main()

    elif args.command == "combine":
        from dwipada.dataset.combine import main
        main()

    elif args.command == "prepare-synthetic":
        from dwipada.dataset.prepare_synthetic import main
        main()

    elif args.command == "validate":
        from dwipada.dataset.validate import main as validate_main
        extra_args = []
        if args.dataset:
            extra_args.extend(["--dataset", args.dataset])
        if args.skip_levels:
            extra_args.extend(["--skip-levels"] + [str(l) for l in args.skip_levels])
        if args.limit:
            extra_args.extend(["--limit", str(args.limit)])
        if args.fresh:
            extra_args.append("--fresh")
        sys.argv = ["dwipada validate"] + extra_args
        validate_main()

    elif args.command == "prepare":
        from dwipada.training.prepare_data import main
        main()

    elif args.command == "train":
        from dwipada.training.train import main as train_main
        sys.argv = ["dwipada train"]
        if args.model:
            sys.argv.extend(["--model", args.model])
        if args.epochs:
            sys.argv.extend(["--epochs", str(args.epochs)])
        if args.batch_size:
            sys.argv.extend(["--batch_size", str(args.batch_size)])
        if args.lora_rank:
            sys.argv.extend(["--lora_rank", str(args.lora_rank)])
        if args.merge:
            sys.argv.append("--merge")
        train_main()

    elif args.command == "generate":
        from dwipada.training.generate import main as gen_main
        sys.argv = ["dwipada generate"]
        if args.prompt:
            sys.argv.append(args.prompt)
        if args.interactive:
            sys.argv.append("--interactive")
        if args.batch:
            sys.argv.extend(["--batch", args.batch])
        if args.num_poems > 1:
            sys.argv.extend(["--num-poems", str(args.num_poems)])
        if args.no_validate:
            sys.argv.append("--no-validate")
        gen_main()

    elif args.command == "batch":
        from dwipada.batch.gemini import main as batch_main
        sys.argv = ["dwipada batch"]
        if args.prepare:
            sys.argv.extend(["--prepare", args.prepare])
        if args.submit:
            sys.argv.extend(["--submit", args.submit])
        if args.status:
            sys.argv.extend(["--status", args.status])
        if args.results:
            sys.argv.extend(["--results", args.results])
        batch_main()

    elif args.command == "batch-vertex":
        from dwipada.batch.vertex import main
        main()

    elif args.command == "generate-constrained":
        from dwipada.training.generate_constrained import main as gen_c_main
        sys.argv = ["dwipada generate-constrained"]
        if args.prompt:
            sys.argv.append(args.prompt)
        if args.interactive:
            sys.argv.append("--interactive")
        if args.batch:
            sys.argv.extend(["--batch", args.batch])
        if args.base_model:
            sys.argv.extend(["--base-model", args.base_model])
        if args.adapter:
            sys.argv.extend(["--adapter", args.adapter])
        if args.merged_model:
            sys.argv.extend(["--merged-model", args.merged_model])
        if args.top_k_constraint != 25:
            sys.argv.extend(["--top-k-constraint", str(args.top_k_constraint)])
        if args.no_prasa:
            sys.argv.append("--no-prasa")
        if args.no_yati:
            sys.argv.append("--no-yati")
        if args.do_sample:
            sys.argv.append("--do-sample")
        if args.temperature != 0.7:
            sys.argv.extend(["--temperature", str(args.temperature)])
        if args.no_validate:
            sys.argv.append("--no-validate")
        gen_c_main()

    elif args.command == "analyze":
        _run_analyze(args)


def _run_analyze(args):
    """Analyze a dwipada poem from CLI input or file."""
    from dwipada.core.analyzer import analyze_dwipada, format_analysis_report

    if args.file:
        with open(args.file, 'r', encoding='utf-8') as f:
            poem = f.read().strip()
    elif args.poem:
        poem = args.poem.replace("\\n", "\n")
    else:
        print("Enter poem (2 lines, press Enter twice to finish):")
        lines = []
        while True:
            line = input()
            if not line:
                break
            lines.append(line)
        poem = "\n".join(lines)

    try:
        analysis = analyze_dwipada(poem)
        print(format_analysis_report(analysis))
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
