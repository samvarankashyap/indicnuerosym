#!/usr/bin/env python3
"""
Batch process dwipada couplets through Gemini on Vertex AI Batch Prediction.

The input JSONL must already be in Vertex AI format:
  {"request": {"contents": [{"role": "user", "parts": [{"text": "..."}]}]}}

Subcommands:
    --upload FILE     Upload a local JSONL to GCS and submit a batch job
    --status JOB_ID   Check batch job status
    --results JOB_ID  Download results from a completed batch job to local disk

Config (config.yaml):
    vertex:
        project_id: "your-gcp-project"
        location: "us-central1"
        gcs_bucket: "your-gcs-bucket"
        service_account_key: "path/to/service_account.json"  # optional; omit to use ADC
        model: "publishers/google/models/gemini-2.0-flash-001"
"""

import argparse
import json
import sys
import os
import yaml
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
CONFIG_FILE = PROJECT_ROOT / "config.yaml"
DEFAULT_INPUT = PROJECT_ROOT / "output" / "batch_requests.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output"

DEFAULT_MODEL = "publishers/google/models/gemini-3-flash"


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG & AUTH
# ─────────────────────────────────────────────────────────────────────────────

def load_vertex_config() -> dict:
    with open(CONFIG_FILE) as f:
        config = yaml.safe_load(f)
    vertex = config.get("vertex", {})
    required = ["project_id", "location", "gcs_bucket"]
    missing = [k for k in required if not vertex.get(k)]
    if missing:
        print(
            f"Error: Missing required fields in config.yaml under 'vertex': {missing}\n"
            "Add the following to config.yaml:\n"
            "  vertex:\n"
            "    project_id: your-gcp-project\n"
            "    location: us-central1\n"
            "    gcs_bucket: your-gcs-bucket\n"
            "    service_account_key: path/to/service_account.json  # optional\n"
            "    model: publishers/google/models/gemini-3-flash-preview  # optional",
            file=sys.stderr,
        )
        sys.exit(1)
    return vertex


def init_vertex(vertex_cfg: dict):
    """Initialise Vertex AI SDK, optionally using a service account JSON key."""
    import vertexai

    sa_key = vertex_cfg.get("service_account_key")
    credentials = None

    if sa_key:
        from google.oauth2 import service_account as sa_module

        key_path = Path(sa_key)
        if not key_path.is_absolute():
            key_path = PROJECT_ROOT / key_path
        if not key_path.exists():
            print(f"Error: Service account key not found: {key_path}", file=sys.stderr)
            sys.exit(1)
        credentials = sa_module.Credentials.from_service_account_file(
            str(key_path),
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )
        print(f"Using service account: {key_path.name}")
    else:
        print("Using Application Default Credentials (ADC).")

    vertexai.init(
        project=vertex_cfg["project_id"],
        location=vertex_cfg["location"],
        credentials=credentials,
    )


def get_gcs_client(vertex_cfg: dict):
    """Return a GCS client, optionally with service account credentials."""
    from google.cloud import storage

    sa_key = vertex_cfg.get("service_account_key")
    if sa_key:
        from google.oauth2 import service_account as sa_module

        key_path = PROJECT_ROOT / sa_key if not Path(sa_key).is_absolute() else Path(sa_key)
        creds = sa_module.Credentials.from_service_account_file(str(key_path))
        return storage.Client(project=vertex_cfg["project_id"], credentials=creds)
    return storage.Client(project=vertex_cfg["project_id"])


# ─────────────────────────────────────────────────────────────────────────────
# GCS HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def upload_to_gcs(local_path: Path, bucket_name: str, gcs_prefix: str, gcs_client) -> str:
    """Upload a local file to GCS and return the gs:// URI."""
    bucket = gcs_client.bucket(bucket_name)
    blob_name = f"{gcs_prefix}/{local_path.name}"
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(str(local_path))
    gcs_uri = f"gs://{bucket_name}/{blob_name}"
    print(f"Uploaded {local_path} → {gcs_uri}")
    return gcs_uri


def download_from_gcs(gcs_uri: str, local_path: Path, gcs_client):
    """Download a GCS file (gs://bucket/blob) to a local path."""
    assert gcs_uri.startswith("gs://"), f"Expected gs:// URI, got: {gcs_uri}"
    path_part = gcs_uri[len("gs://"):]
    bucket_name, blob_name = path_part.split("/", 1)
    bucket = gcs_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.download_to_filename(str(local_path))
    print(f"Downloaded {gcs_uri} → {local_path}")


def list_gcs_prefix(gcs_uri_prefix: str, gcs_client) -> list[str]:
    """List all blobs under a GCS URI prefix, returning gs:// URIs."""
    path_part = gcs_uri_prefix.removeprefix("gs://")
    bucket_name, prefix = path_part.split("/", 1)
    bucket = gcs_client.bucket(bucket_name)
    blobs = list(bucket.list_blobs(prefix=prefix))
    return [f"gs://{bucket_name}/{b.name}" for b in blobs if not b.name.endswith("/")]


# ─────────────────────────────────────────────────────────────────────────────
# SUBCOMMANDS
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_jsonl(input_path: Path) -> Path:
    """Strip 'metadata' from each line, keeping only 'request'.

    Saves a sidecar <stem>_metadata.jsonl with the original metadata indexed
    by line number so results can be correlated after download.
    Returns path to the cleaned temp file.
    """
    clean_path = input_path.with_stem(input_path.stem + "_clean")
    meta_path = input_path.with_stem(input_path.stem + "_metadata")

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(clean_path, "w", encoding="utf-8") as fclean, \
         open(meta_path, "w", encoding="utf-8") as fmeta:
        for i, line in enumerate(fin):
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            fclean.write(json.dumps({"request": entry["request"]}, ensure_ascii=False) + "\n")
            fmeta.write(json.dumps({"line": i, "metadata": entry.get("metadata", {})}, ensure_ascii=False) + "\n")

    print(f"Preprocessed {input_path.name} → {clean_path.name} (metadata saved to {meta_path.name})")
    return clean_path


def cmd_upload(args):
    """Upload local JSONL to GCS and submit a Vertex AI batch prediction job."""
    from vertexai.preview.batch_prediction import BatchPredictionJob

    input_path = Path(args.upload)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    vertex_cfg = load_vertex_config()
    init_vertex(vertex_cfg)
    gcs_client = get_gcs_client(vertex_cfg)

    bucket = vertex_cfg["gcs_bucket"]
    model = args.model or vertex_cfg.get("model", DEFAULT_MODEL)

    # Strip metadata field — Vertex AI only accepts request field
    clean_path = preprocess_jsonl(input_path)

    # Upload cleaned JSONL to GCS
    gcs_input_uri = upload_to_gcs(
        clean_path,
        bucket_name=bucket,
        gcs_prefix="batch_inputs",
        gcs_client=gcs_client,
    )
    clean_path.unlink(missing_ok=True)  # delete temp file after upload

    # GCS output prefix (Vertex AI writes result files here)
    job_name_hint = input_path.stem
    gcs_output_prefix = f"gs://{bucket}/batch_outputs/{job_name_hint}/"

    print(f"Submitting batch job: model={model}")
    print(f"  Input:  {gcs_input_uri}")
    print(f"  Output: {gcs_output_prefix}")

    job = BatchPredictionJob.submit(
        source_model=model,
        input_dataset=gcs_input_uri,
        output_uri_prefix=gcs_output_prefix,
    )

    print(f"\nBatch job submitted:")
    print(f"  Job name: {job.name}")
    print(f"  State:    {job.state}")
    print(f"\nTo check status:  python run_batch_vertex.py --status \"{job.name}\"")
    print(f"To get results:   python run_batch_vertex.py --results \"{job.name}\"")


def cmd_list(args):
    """List recent Vertex AI batch prediction jobs."""
    from vertexai.preview.batch_prediction import BatchPredictionJob

    vertex_cfg = load_vertex_config()
    init_vertex(vertex_cfg)

    STATE_NAMES = {
        0: "UNSPECIFIED", 1: "QUEUED", 2: "PENDING", 3: "RUNNING",
        4: "SUCCEEDED", 5: "FAILED", 6: "CANCELLING", 7: "CANCELLED",
        8: "PAUSED", 9: "EXPIRED", 10: "UPDATING", 11: "PARTIALLY_SUCCEEDED",
    }

    jobs = list(BatchPredictionJob.list())
    if not jobs:
        print("No batch prediction jobs found.")
        return

    print(f"{'ID':<25} {'STATE':<22} {'NAME'}")
    print("-" * 80)
    for job in jobs:
        job_id = job.name.split("/")[-1]
        state_int = int(job.state)
        state_label = STATE_NAMES.get(state_int, str(state_int))
        display_name = getattr(job, "display_name", "") or ""
        print(f"{job_id:<25} {state_label:<22} {display_name}")


def cmd_status(args):
    """Check Vertex AI batch job status."""
    from vertexai.preview.batch_prediction import BatchPredictionJob

    vertex_cfg = load_vertex_config()
    init_vertex(vertex_cfg)

    job = BatchPredictionJob(args.status)
    job.refresh()

    STATE_NAMES = {
        0: "UNSPECIFIED", 1: "QUEUED", 2: "PENDING", 3: "RUNNING",
        4: "SUCCEEDED ✓", 5: "FAILED ✗", 6: "CANCELLING", 7: "CANCELLED",
        8: "PAUSED", 9: "EXPIRED", 10: "UPDATING", 11: "PARTIALLY_SUCCEEDED",
    }
    state_int = int(job.state)
    state_label = STATE_NAMES.get(state_int, str(state_int))

    print(f"Job:   {job.name}")
    print(f"State: {state_int} ({state_label})")

    # Print stats if available
    stats = getattr(job._gca_resource, "completion_stats", None)
    if stats:
        succeeded = getattr(stats, 'successful_count', 0) or 0
        failed    = getattr(stats, 'failed_count', 0) or 0
        incomplete = getattr(stats, 'incomplete_count', 0) or 0
        total = succeeded + failed + incomplete
        print(f"Succeeded:  {succeeded} / {total}")
        print(f"Failed:     {failed} / {total}")
        print(f"Incomplete: {incomplete} / {total}")

    if hasattr(job, "output_location") and job.output_location:
        print(f"Output:    {job.output_location}")

    if state_int == 5:  # FAILED
        error = getattr(job._gca_resource, "error", None)
        if error:
            print(f"\nError code:    {getattr(error, 'code', 'N/A')}")
            print(f"Error message: {getattr(error, 'message', 'N/A')}")
        partial = getattr(job._gca_resource, "partial_failures", [])
        if partial:
            print(f"\nPartial failures ({len(partial)}):")
            for pf in partial[:5]:
                print(f"  - {pf}")
    elif state_int < 4:
        print(f"\nStill in progress. Re-run --status to check again.")


def cmd_results(args):
    """Download results from a completed Vertex AI batch job."""
    from vertexai.preview.batch_prediction import BatchPredictionJob

    vertex_cfg = load_vertex_config()
    init_vertex(vertex_cfg)
    gcs_client = get_gcs_client(vertex_cfg)

    job = BatchPredictionJob(args.results)
    job.refresh()

    print(f"Job:   {job.name}")
    print(f"State: {job.state}")

    state_name = job.state.name if hasattr(job.state, "name") else str(job.state)
    if "SUCCEEDED" not in state_name and "DONE" not in state_name:
        print(f"Job has not succeeded yet (state: {job.state}). Try again later.")
        sys.exit(1)

    output_location = getattr(job, "output_location", None)
    if not output_location:
        print("Error: No output location found in the job.", file=sys.stderr)
        sys.exit(1)

    print(f"Output location: {output_location}")

    # List output files
    output_files = list_gcs_prefix(output_location, gcs_client)
    if not output_files:
        print("No output files found at the output location.")
        sys.exit(1)

    print(f"Found {len(output_files)} output file(s).")

    # Determine local output path
    if args.output:
        local_out = Path(args.output)
    else:
        DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        job_id = args.results.rstrip("/").split("/")[-1]
        local_out = DEFAULT_OUTPUT_DIR / f"batch_responses_vertex_{job_id}.jsonl"

    # Merge all output files into one local JSONL
    with open(local_out, "w", encoding="utf-8") as fout:
        for gcs_file in output_files:
            if not gcs_file.endswith(".jsonl") and not gcs_file.endswith(".json"):
                continue
            tmp = local_out.with_suffix(".tmp")
            download_from_gcs(gcs_file, tmp, gcs_client)
            with open(tmp, "r", encoding="utf-8") as fin:
                for line in fin:
                    line = line.strip()
                    if line:
                        fout.write(line + "\n")
            tmp.unlink(missing_ok=True)

    # Count lines
    with open(local_out, "r", encoding="utf-8") as f:
        line_count = sum(1 for _ in f)

    print(f"Results saved to: {local_out}")
    print(f"Total response lines: {line_count}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Batch process dwipada couplets through Gemini on Vertex AI.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_batch_vertex.py --upload output/batch_requests_100.jsonl
  python run_batch_vertex.py --upload output/batch_requests_laj.jsonl --model publishers/google/models/gemini-3.1-pro-preview
  python run_batch_vertex.py --status "projects/123/locations/us-central1/batchPredictionJobs/456"
  python run_batch_vertex.py --results "projects/123/locations/us-central1/batchPredictionJobs/456"
  python run_batch_vertex.py --results "projects/123/.../456" -o output/batch_responses_100.jsonl

Config (config.yaml):
  vertex:
    project_id: your-gcp-project
    location: us-central1
    gcs_bucket: your-gcs-bucket
    service_account_key: path/to/service_account.json   # omit to use ADC
    model: publishers/google/models/gemini-3-flash-preview  # optional
        """,
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--upload", metavar="FILE",
        help="Upload local JSONL to GCS and submit a Vertex AI batch prediction job",
    )
    group.add_argument(
        "--list", action="store_true",
        help="List all batch prediction jobs",
    )
    group.add_argument(
        "--status", metavar="JOB_NAME",
        help="Check batch job status (full resource name or numeric ID)",
    )
    group.add_argument(
        "--results", metavar="JOB_NAME",
        help="Download results from a completed batch job",
    )

    parser.add_argument(
        "--output", "-o", metavar="FILE",
        help="Local output file for --results (default: output/batch_responses_vertex.jsonl)",
    )
    parser.add_argument(
        "--model", metavar="MODEL",
        help=(
            "Override the model for --upload "
            "(e.g. publishers/google/models/gemini-3.1-pro-preview). "
            "Takes precedence over config.yaml."
        ),
    )

    args = parser.parse_args()

    if args.upload:
        cmd_upload(args)
    elif args.list:
        cmd_list(args)
    elif args.status:
        cmd_status(args)
    elif args.results:
        cmd_results(args)


if __name__ == "__main__":
    main()
