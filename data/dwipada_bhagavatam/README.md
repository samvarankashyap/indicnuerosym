# ద్విపదభాగవతము — Dwipada Bhagavatam (raw text)

Raw crawled text of the Telugu *Dwipada Bhāgavatamu*, a *dvipada*-metre
adaptation of the Sanskrit *Bhāgavata Purāṇa*. The corpus is split
into three *kāṇḍas* (books). Crawled from te.wikisource.org.

## Contents

- **209 `.txt` files** organised into **3 *kāṇḍa* subdirectories**:
  - `01_MadhuraKanda/` — Madhurā Kāṇḍa (Krishna's Mathura years)
  - `02_KalyanaKanda/` — Kalyāṇa Kāṇḍa
  - `03_JagadabhirakshaKanda/` — Jagadabhirakṣa Kāṇḍa
- Plus 3 single-file dumps at the directory root (`1_madhurakanda.txt`,
  `1_kalyanakanda.txt`, `1_jagadabhirakshakanda.txt`) — one
  whole-kanda dump per file from an earlier crawl pass
- Encoding: UTF-8 Telugu
- Files start with metadata header lines (`# గ్రంథము:`, `# కాండము:`,
  `# విభాగము:`, `# శీర్షిక:`) followed by the verse content

## Provenance

- **Crawler:** `../../crawlers/dwipada_bhagavatam.py`
- **Source URL:** te.wikisource.org (3 separate Wikisource pages, one per kanda)
- **Re-crawl:** `python ../../crawlers/dwipada_bhagavatam.py`

## Downstream

This raw text is consolidated into `../consolidated_dwipada.json` by
`src/dwipada/data/consolidate.py`. The `Dvipada Bhagavatamu` row of
the master corpus contributes **3,157 raw couplets** (2,002 after
prosodic filtering, 63.4% purity) to the 27,881-couplet
`dwipada_master_dataset.json`. Two of the three kandas survive
filtering at usable rates; the third is significantly noisier.

## Related

- Parent: `../README.md` (`data/`)
- Sibling source: `../dwipada_bhagavatam2/` (a smaller secondary dump)
- Crawler: `../../crawlers/README.md`
- Paper Section 4 (`tab:sources`)
