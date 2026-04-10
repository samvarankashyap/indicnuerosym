# శ్రీరమాపరిణయము — Srirama Parinayamu (raw text)

Raw crawled text of *Śrī Rāma Pariṇayamu* by తరిగొండ వెంగమాంబ
(Tarigoṇḍa Veṅgamāmba, 18th century), a *dvipada*-metre devotional
account of the marriage of Rāma and Sītā. The smallest single source
in the project's corpus but with the highest prosodic purity rate.
Crawled from te.wikisource.org.

## Contents

- **28 `.txt` files** in a single flat directory (no subdirectories)
- File naming: `NNN_<chapter title>.txt`
- Encoding: UTF-8 Telugu
- Each file starts with metadata header lines (`# గ్రంథము:`,
  `# రచయిత:`, `# అధ్యాయము:`, `# శీర్షిక:`) followed by the verse content

## Provenance

- **Crawler:** `../../crawlers/srirama_parinayamu.py`
- **Source URL:** te.wikisource.org (single Wikisource page covering
  all 28 chapters; the crawler validates chapter titles against an
  expected list)
- **Re-crawl:** `python ../../crawlers/srirama_parinayamu.py`

## Downstream

This raw text is consolidated into `../consolidated_dwipada.json` by
`src/dwipada/data/consolidate.py`. The `Srirama Parinayamu` row of
the master corpus contributes **392 raw couplets** (374 after
prosodic filtering, **95.4% purity** — the highest of any source) to
the 27,881-couplet `dwipada_master_dataset.json`.

## Related

- Parent: `../README.md` (`data/`)
- Crawler: `../../crawlers/README.md`
- Paper Section 4 (`tab:sources`)
