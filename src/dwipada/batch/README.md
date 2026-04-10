# dwipada.batch — Gemini Batch API Integration

Cloud batch-API integration for sending Telugu *dvipada* couplets to
Google's Gemini API for meaning extraction (*bhāvam* and
*pratipadārtham*). Used to enrich the corpus with LLM-generated
annotations and to run the LLM-as-Judge evaluation referenced in the
paper's Section 5.5.

## Files

| File | Purpose |
| --- | --- |
| `gemini.py` | High-level Gemini Batch API orchestrator: submit jobs, poll status, download results. Exposed via `dwipada batch --submit/--status/--results`. |
| `client.py` | Low-level Gemini API client wrapper (HTTP, retries, auth). |
| `config.py` | Loads `config.yaml` from the project root and validates required fields (`api_key`, optional `vertex` block). |
| `generate_requests.py` | Walk through `data/*/` text files and emit a single JSONL of batch requests in Vertex/Gemini format, one per couplet, with metadata for back-correlation. |

## How requests are shaped

Each request asks Gemini to act as a Telugu/Sanskrit scholar and
return *bhāvam* (Telugu prose meaning) + *pratipadārtham* (English
meaning) for the couplet:

```jsonl
{"request": {"contents": [{"role": "user", "parts": [{"text": "Assume role of a telugu and sanskrit scholar...\nPoem:\nసౌధాగ్రముల యందు సదనంబు లందు\nవీధుల యందును వెఱవొప్ప నిలిచి"}]}]}, "metadata": {"source_file": "data/ranganatha_ramayanam/01_BalaKanda/001_*.txt", "work": "ranganatha_ramayanam", "couplet_number": 1}}
```

The `metadata` block is stripped before upload and saved as a sidecar
file so results can be joined back to source couplets.

## Usage

```bash
# 1. Generate the request JSONL from raw text
python -m dwipada.batch.generate_requests
# Output: output/batch_requests.jsonl

# 2. Submit (returns a job name)
dwipada batch --submit output/batch_requests.jsonl

# 3. Check status
dwipada batch --status "batches/abc123"

# 4. Download results
dwipada batch --results "batches/abc123"
# Output: output/batch_responses.jsonl
```

## Authentication

Add an AI Studio API key to `config.yaml` at the project root:

```yaml
api_key: "your-gemini-api-key"
```

`config.yaml` is gitignored. Never commit it.

## Vertex AI counterpart

The chandomitra fork of this sub-package
(`../../../chandomitra/src/dwipada/batch/`) additionally contains a
`vertex.py` script that talks to the Vertex AI Batch Prediction API
instead of the AI Studio API. The two are independent tools — pick
whichever fits your authentication setup. See the project root README
for the full Vertex AI setup walk-through.

## Related

- `../../../output/` — where the request and response JSONLs land
- `../../../llm_as_judge/` — uses the same batch infrastructure for
  the LLM-as-Judge evaluation pass
- `../../../config.yaml` — the gitignored API key file
- Paper Section 5.5 (LLM-as-a-Judge evaluation, Gemini 3 Pro on
  Vertex AI Batch API)
