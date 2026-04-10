# ద్విపదభాగవతము (secondary dump) — raw text

Secondary single-file dump of the Telugu *Dwipada Bhāgavatamu*, kept
separately from `../dwipada_bhagavatam/` to preserve the alternate
source's text. Crawled from te.wikisource.org by an earlier crawler
pass that produced one file per *kāṇḍa* rather than per *aśvāsa*.

## Contents

- **3 `.txt` files** (flat directory, no subdirectories):
  - `1_madhurakanda.txt` — Madhurā Kāṇḍa (entire kanda in one file)
  - `1_kalyanakanda.txt` — Kalyāṇa Kāṇḍa (entire kanda in one file)
  - `1_jagadabhirakshakanda.txt` — Jagadabhirakṣa Kāṇḍa (entire kanda in one file)
- Encoding: UTF-8 Telugu
- Each file uses `# <heading>` markers between sections (no per-file
  metadata block); couplets are paired and separated by blank lines

## Provenance

- **Crawler:** `../../crawlers/dwipada_bhagavatam.py` (an earlier
  output mode that wrote one file per kanda; the current per-aśvāsa
  output lives in `../dwipada_bhagavatam/`)
- **Source URL:** te.wikisource.org

## Downstream

These three files are consolidated into `../consolidated_dwipada.json`
by `src/dwipada/data/consolidate.py`. They are included alongside the
larger `../dwipada_bhagavatam/` dump for completeness; the master
dataset deduplicates between the two during consolidation.

## Related

- Parent: `../README.md` (`data/`)
- Primary dump: `../dwipada_bhagavatam/`
- Crawler: `../../crawlers/README.md`
- Paper Section 4 (`tab:sources`)
