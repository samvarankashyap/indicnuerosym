# రంగనాథ రామాయణము — Ranganatha Ramayanam (raw text)

Raw crawled text of *Raṅganātha Rāmāyaṇamu* by గోన బుద్ధారెడ్డి
(Gōna Buddhā Reḍḍi, 13th century), the earliest *dvipada*-metre
adaptation of the Vālmīki *Rāmāyaṇa* into Telugu. The largest single
source in the project's corpus by an order of magnitude. Crawled
from andhrabharati.com.

## Contents

- **405 `.txt` files** organised into **7 *kāṇḍa* (book) subdirectories**:

  | Subdirectory | Kāṇḍa | Chapters |
  | --- | --- | --- |
  | `01_BalaKanda/` | Bāla Kāṇḍa | 31 |
  | `02_AyodhyaKanda/` | Ayodhyā Kāṇḍa | 35 |
  | `03_AranyaKanda/` | Araṇya Kāṇḍa | 28 |
  | `04_KishkindhaKanda/` | Kiṣkindhā Kāṇḍa | 25 |
  | `05_SundaraKanda/` | Sundara Kāṇḍa | 27 |
  | `06_YuddhaKanda/` | Yuddha Kāṇḍa | 170 |
  | `07_UttaraKanda/` | Uttara Kāṇḍa | 89 |

- File naming: `NNN_<chapter title>.txt`
- Encoding: UTF-8 Telugu
- Each file starts with metadata header lines (`# కాండము:`,
  `# అధ్యాయము:`, `# శీర్షిక:`) followed by the verse content
- Footnotes (where present) appear after a `--- పాదసూచికలు (Footnotes):`
  separator at the end of each file

## Provenance

- **Crawler:** `../../crawlers/ranganatha_ramayanam.py`
- **Source URL:** andhrabharati.com (405 individual chapter pages)
- **Re-crawl:** `python ../../crawlers/ranganatha_ramayanam.py`
  (resumable — `checkpoint.json` tracks downloaded chapters and the
  crawler skips them on re-run)

## Downstream

This raw text is consolidated into `../consolidated_dwipada.json` by
`src/dwipada/data/consolidate.py`. The `Ranganatha Ramayanamu` row of
the master corpus contributes **26,296 raw couplets** (21,828 after
prosodic filtering, 83.0% purity) — by far the largest single source,
making up roughly 78% of the 27,881-couplet
`dwipada_master_dataset.json`.

## Related

- Parent: `../README.md` (`data/`)
- Crawler: `../../crawlers/README.md` (with checkpoint/resume details)
- Paper Section 4 (`tab:sources`)
