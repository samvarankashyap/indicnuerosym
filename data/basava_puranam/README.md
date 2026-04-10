# బసవపురాణము — Basava Puranam (raw text)

Raw crawled text of *Basava Purāṇamu* by పాల్కురికి సోమనాథుడు
(Pālkuriki Somanāthuḍu, 13th century), the canonical biography of
the Vīraśaiva saint Basaveśvara, written in Telugu *dvipada* metre.
Crawled from te.wikisource.org.

## Contents

- **47 `.txt` files** organised into **3 ఆశ్వాసము (book) subdirectories**:
  - `001_ప్రథమాశ్వాసము/` — First book
  - `002_ద్వితీయాశ్వాసము/` — Second book
  - `003_తృతీయాశ్వాసము/` — Third book
- Each `.txt` file is one section, named `NNN_<section title>.txt`
- Encoding: UTF-8 Telugu
- Each file starts with metadata header lines (`# గ్రంథము:`, `# ఆశ్వాసము:`, `# విభాగము:`, `# శీర్షిక:`) followed by the verse content

## Provenance

- **Crawler:** `../../crawlers/basava_puranam.py`
- **Source URL:** te.wikisource.org (3 separate Wikisource pages, one per ఆశ్వాసము)
- **Re-crawl:** `python ../../crawlers/basava_puranam.py` (idempotent)

## Downstream

This raw text is consolidated into `../consolidated_dwipada.json` by
`src/dwipada/data/consolidate.py`, then enters the validation
pipeline in `dataset_validation_scripts/`. The `Basava Puranam` row of
the master corpus contributes **2,454 raw couplets** (1,859 after
prosodic filtering, 75.8% purity) to the 27,881-couplet
`dwipada_master_dataset.json`.

## Related

- Parent: `../README.md` (`data/`)
- Crawler: `../../crawlers/README.md`
- Paper Section 4 (`tab:sources`)
