# dwipada

Main Python package for Indic-NeuroSym. Provides prosody analysis,
data pipelines, and training tools for Telugu Dwipada poetry.

## Modules

| Module | Description |
|---|---|
| `cli.py` | Unified CLI dispatcher (`dwipada` command) |
| `paths.py` | Centralized path resolution for all project directories |
| `__main__.py` | Enables `python -m dwipada` invocation |

## Sub-packages

Each sub-package has its own README with file inventory, public API,
and pipeline order.

| Sub-package | README | One-line summary |
|---|---|---|
| `core/` | [core/README.md](core/README.md) | Prosody analysis engine — syllable splitting, guru/laghu classification, gana/yati/prasa validation, scoring |
| `data/` | [data/README.md](data/README.md) | Crawl → clean → consolidate pipeline for the 5 Telugu sources |
| `dataset/` | [dataset/README.md](dataset/README.md) | Dataset creation, augmentation, deduplication, statistics |
| `batch/` | [batch/README.md](batch/README.md) | Gemini Batch API integration for LLM annotation and LLM-as-Judge |

## Two copies of this package

There are two byte-similar but **non-identical** copies of this
package in the repo:

- **Canonical:** `src/dwipada/` (this folder) — used by `tests/`,
  the top-level CLI, and the offline analyser. This is the
  authoritative version.
- **Snapshot fork:** `chandomitra/src/dwipada/` — bundled inside
  the chandomitra benchmark folder so the chandomitra benchmark
  scripts can run self-contained. The fork adds extra files
  (constrained-generation utilities, an extra `vertex.py` in
  `batch/`, three extra files in `dataset/`, the `crawl_base.py` +
  `crawlers/` subtree in `data/`, and an entire `training/`
  sub-package). It does not add new files to `core/`.

The two READMEs at `src/dwipada/README.md` and
`chandomitra/src/dwipada/README.md` are intentionally kept identical
in content; the per-sub-package READMEs in each tree call out the
fork-specific differences.

## Usage

```bash
# CLI
dwipada analyze "poem text"
dwipada crawl ranganatha_ramayanam
dwipada prepare --split 80 10 10

# Python
from dwipada.core import analyze_dwipada, format_analysis_report
result = analyze_dwipada("సౌధాగ్రముల యందు సదనంబు లందు\nవీధుల యందును వెఱవొప్ప నిలిచి")
print(format_analysis_report(result))
```

## Related

- Tests: `../../tests/README.md`
- Streaming FST + NFA mirror of `core/`: `../../nfa_for_dwipada/README.md`
- Standalone validation pipeline: `../../dataset_validation_scripts/README.md`
- Top-level project README: `../../README.md`
