# dwipada.batch (chandomitra fork) — Gemini + Vertex AI Batch APIs

Cloud batch-API integration for sending Telugu *dvipada* couplets to
Google's LLMs for meaning extraction and the LLM-as-Judge pass.
Snapshotted into the chandomitra benchmark folder. Adds a `vertex.py`
that the canonical `../../../../src/dwipada/batch/` does not have.

## Files

| File | Purpose |
| --- | --- |
| `gemini.py` | Gemini Batch API orchestrator (submit, status, results). |
| `vertex.py` | **Fork-only.** Vertex AI Batch Prediction API client. Uploads request JSONLs to GCS, submits the batch job, polls status, and downloads results back to local disk. Run via `python -m dwipada.batch.vertex --upload/--status/--results`. |
| `client.py` | Low-level Gemini API client wrapper (HTTP, retries, auth). |
| `config.py` | Loads `config.yaml` from the project root. |
| `generate_requests.py` | Walk through `data/*/` text files and emit a single JSONL of batch requests with metadata. |

## Why two batch backends

- **`gemini.py`** authenticates against the AI Studio API with a
  single API key and is the simplest path for one-off batch runs.
- **`vertex.py`** uses the full Vertex AI Batch Prediction API, which
  requires a GCP project with the Vertex AI API enabled, a GCS
  bucket for file staging, and either a service-account key or
  Application Default Credentials. The Vertex path is what the
  paper's LLM-as-Judge evaluation (Section 5.5) ran on.

The two backends consume the same input JSONL format (from
`generate_requests.py`) and write the same output schema, so they
are interchangeable from the caller's perspective.

## Usage

### Gemini Batch API

```bash
python -m dwipada.batch.generate_requests
dwipada batch --submit output/batch_requests.jsonl
dwipada batch --status "batches/abc123"
dwipada batch --results "batches/abc123"
```

### Vertex AI Batch Prediction (fork-only)

```bash
python -m dwipada.batch.vertex --upload output/batch_requests.jsonl
python -m dwipada.batch.vertex --status "projects/123/locations/us-central1/batchPredictionJobs/456"
python -m dwipada.batch.vertex --results "projects/123/locations/us-central1/batchPredictionJobs/456"
```

## Authentication

`config.yaml` at the project root holds both auth blocks:

```yaml
api_key: "your-gemini-ai-studio-api-key"

vertex:
  project_id: "your-gcp-project"
  location: "us-central1"
  gcs_bucket: "your-gcs-bucket"
  service_account_key: "serviceaccount.json"  # optional; omit to use ADC
  model: "publishers/google/models/gemini-3-flash-preview"  # optional
```

`config.yaml` and `serviceaccount.json` are gitignored.

## Related

- Canonical version (no `vertex.py`): `../../../../src/dwipada/batch/`
- `../../../../llm_as_judge/` — uses the Vertex backend for the
  paper's LLM-as-Judge evaluation pass
- `../../../../output/` — where request and response JSONLs land
- Paper Section 5.5 (LLM-as-a-Judge evaluation, Gemini 3 Pro on
  Vertex AI Batch API)
