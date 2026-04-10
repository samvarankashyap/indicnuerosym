#!/usr/bin/env python3
"""
Batch process dwipada couplets through Gemini 3 Flash using the Gemini Batch API.

Subcommands:
    --prepare N       Create an N-entry subset JSONL in Gemini batch format
    --submit FILE     Upload JSONL and submit a batch job
    --status JOB      Check batch job status
    --results JOB     Download results from a completed batch job
"""

import argparse
import json
import sys
import time
from pathlib import Path

from google import genai
from google.genai import types

from dwipada.paths import OUTPUT_DIR
from dwipada.batch.config import load_api_key

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_INPUT = OUTPUT_DIR / "batch_requests.jsonl"
DEFAULT_OUTPUT_DIR = OUTPUT_DIR

MODEL = "gemini-3-flash-preview"


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def create_client() -> genai.Client:
    return genai.Client(api_key=load_api_key())


def convert_to_batch_format(entry: dict) -> dict:
    """Convert Vertex AI request format to Gemini batch API format.

    Input:  {"request": {"contents": [...]}, "metadata": {"work": "...", "couplet_number": N, ...}}
    Output: {"key": "work__N", "request": {"contents": [{"parts": [...]}]}}
    """
    metadata = entry["metadata"]
    key = f"{metadata['work']}__{metadata['couplet_number']}"

    contents = entry["request"]["contents"]
    # Strip "role" from contents if present (batch API uses parts only)
    batch_contents = []
    for content in contents:
        batch_contents.append({"parts": content["parts"]})

    return {
        "key": key,
        "request": {"contents": batch_contents},
    }


# ─────────────────────────────────────────────────────────────────────────────
# SUBCOMMANDS
# ─────────────────────────────────────────────────────────────────────────────

def parse_range(value: str):
    """Parse prepare argument as count or range.

    '100'     -> (0, 100, '100')       # first 100 lines (0-based start, exclusive end, suffix)
    '101-200' -> (100, 200, '101-200') # lines 101-200 (1-based input -> 0-based start, exclusive end)
    """
    if "-" in value:
        parts = value.split("-", 1)
        start_1based = int(parts[0])
        end_1based = int(parts[1])
        if start_1based < 1 or end_1based < start_1based:
            print(f"Error: Invalid range '{value}'. Use START-END where START >= 1 and END >= START.", file=sys.stderr)
            sys.exit(1)
        return start_1based - 1, end_1based, value
    else:
        count = int(value)
        return 0, count, str(count)


def cmd_prepare(args):
    """Create a subset JSONL in Gemini batch format (by count or range)."""
    start, end, suffix = parse_range(args.prepare)
    input_path = Path(args.input) if args.input else DEFAULT_INPUT
    output_path = DEFAULT_OUTPUT_DIR / f"batch_requests_{suffix}.jsonl"

    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    written = 0
    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line_num, line in enumerate(fin):
            if line_num >= end:
                break
            if line_num < start:
                continue
            entry = json.loads(line)
            batch_entry = convert_to_batch_format(entry)
            fout.write(json.dumps(batch_entry, ensure_ascii=False) + "\n")
            written += 1

    print(f"Wrote {written} entries to {output_path}")


def cmd_submit(args):
    """Upload JSONL file and submit a batch job."""
    input_path = Path(args.submit)
    if not input_path.exists():
        print(f"Error: File not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    client = create_client()

    # Upload file
    print(f"Uploading {input_path}...")
    uploaded = client.files.upload(
        file=str(input_path),
        config=types.UploadFileConfig(
            display_name=input_path.stem,
            mime_type="jsonl",
        ),
    )
    print(f"Uploaded as: {uploaded.name}")

    # Create batch job
    print(f"Submitting batch job with model={MODEL}...")
    job = client.batches.create(
        model=MODEL,
        src=uploaded.name,
        config={"display_name": f"dwipada-{input_path.stem}"},
    )
    print(f"Batch job created: {job.name}")
    print(f"State: {job.state}")
    print(f"\nTo check status:  python -m dwipada.batch.gemini --status \"{job.name}\"")
    print(f"To get results:   python -m dwipada.batch.gemini --results \"{job.name}\"")


def cmd_status(args):
    """Check batch job status."""
    job_name = args.status
    client = create_client()

    job = client.batches.get(name=job_name)
    print(f"Job: {job.name}")
    print(f"State: {job.state}")

    if hasattr(job, "batch_stats") and job.batch_stats:
        stats = job.batch_stats
        print(f"Total requests:  {getattr(stats, 'total_request_count', 'N/A')}")
        print(f"Succeeded:       {getattr(stats, 'succeeded_request_count', 'N/A')}")
        print(f"Failed:          {getattr(stats, 'failed_request_count', 'N/A')}")


def cmd_results(args):
    """Download results from a completed batch job."""
    job_name = args.results
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = DEFAULT_OUTPUT_DIR / "batch_responses.jsonl"
    client = create_client()

    job = client.batches.get(name=job_name)
    print(f"Job: {job.name}")
    print(f"State: {job.state}")

    if job.state.name != "JOB_STATE_SUCCEEDED":
        print(f"Job has not succeeded yet (state: {job.state}). Try again later.")
        sys.exit(1)

    # Download result file
    if job.dest and job.dest.file_name:
        print(f"Downloading results from {job.dest.file_name}...")
        file_content = client.files.download(file=job.dest.file_name)
        raw = file_content if isinstance(file_content, bytes) else file_content.encode("utf-8")

        with open(output_path, "wb") as f:
            f.write(raw)
        print(f"Results saved to {output_path}")

        # Count lines
        line_count = raw.decode("utf-8").count("\n")
        print(f"Total response lines: {line_count}")

    elif job.dest and job.dest.inlined_responses:
        print("Downloading inline results...")
        with open(output_path, "w", encoding="utf-8") as f:
            for resp in job.dest.inlined_responses:
                entry = {}
                if resp.response:
                    entry["response"] = resp.response.text
                    entry["status"] = "success"
                elif resp.error:
                    entry["error"] = str(resp.error)
                    entry["status"] = "error"
                if hasattr(resp, "key") and resp.key:
                    entry["key"] = resp.key
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"Results saved to {output_path}")
    else:
        print("No results found in the job response.")
        sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Batch process dwipada couplets through Gemini 3 Flash.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m dwipada.batch.gemini --prepare 100                         # first 100 entries
  python -m dwipada.batch.gemini --prepare 101-200                     # entries 101 to 200
  python -m dwipada.batch.gemini --submit output/batch_requests_101-200.jsonl
  python -m dwipada.batch.gemini --status "batches/abc123"
  python -m dwipada.batch.gemini --results "batches/abc123" -o output/batch_responses_101-200.jsonl
        """,
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare", type=str, metavar="N_or_RANGE",
                       help="Create a subset JSONL: N for first N entries, or START-END for a range (1-based)")
    group.add_argument("--submit", metavar="FILE",
                       help="Upload JSONL and submit a batch job")
    group.add_argument("--status", metavar="JOB_NAME",
                       help="Check batch job status")
    group.add_argument("--results", metavar="JOB_NAME",
                       help="Download results from a completed batch job")

    parser.add_argument("--input", "-i", metavar="FILE",
                        help=f"Source JSONL for --prepare (default: {DEFAULT_INPUT})")
    parser.add_argument("--output", "-o", metavar="FILE",
                        help="Output file path for --results (default: output/batch_responses.jsonl)")

    args = parser.parse_args()

    if args.prepare is not None:
        cmd_prepare(args)
    elif args.submit:
        cmd_submit(args)
    elif args.status:
        cmd_status(args)
    elif args.results:
        cmd_results(args)


if __name__ == "__main__":
    main()
