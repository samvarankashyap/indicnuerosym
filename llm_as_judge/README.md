# LLM-as-Judge

Evaluation framework for assessing dataset quality using LLM-based judgment (inspired by G-Eval and GEMBA).

An LLM evaluates Telugu/English meaning quality using structured rubrics, providing automated quality scores for the augmented dataset.

## Files

| File | Description |
|---|---|
| `generate_batch_requests_for_laj.py` | Generate evaluation batch requests for Vertex AI |
| `batch_requests_laj.jsonl` | Generated batch requests (~1M) |
| `batch_requests_laj_metadata.jsonl` | Metadata tracking for batch requests |
| `batch_responses_vertex_*.jsonl` | LLM evaluation results (~3.6M) |
