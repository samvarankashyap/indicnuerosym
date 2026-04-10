# dwipada.data (chandomitra fork) — Crawl, Clean, Consolidate

Crawl → clean → consolidate pipeline for the five Telugu classical
literature sources, snapshotted into the chandomitra benchmark
folder. Adds a `crawl_base.py` and a `crawlers/` subdirectory that
the canonical `../../../../src/dwipada/data/` does not have, so the
chandomitra benchmarks can run end-to-end without depending on the
project-root `crawlers/` folder.

## Files

| File | Purpose |
| --- | --- |
| `crawl_base.py` | **Fork-only.** Shared crawler utilities (HTTP retry, HTML cleaning, filename sanitisation). Used by every per-source crawler in `crawlers/`. |
| `clean_base.py` | Shared text-cleaning utilities (character removal, whitespace normalisation, footnote stripping). |
| `consolidate.py` | Walk every `data/<source>/*.txt` file and emit one JSON record per couplet into `data/consolidated_dwipada.json`. |
| `cleaners/` | Per-source cleaners (one per work). |
| `crawlers/` | **Fork-only.** Per-source web crawlers (one per work) — mirrors the project-root `crawlers/` folder. |

## Per-source cleaners

| Cleaner | Source work |
| --- | --- |
| `cleaners/basava_puranam.py` | బసవపురాణము |
| `cleaners/dwipada_bhagavatam.py` | ద్విపదభాగవతము |
| `cleaners/palanati_veera_charitra.py` | పల్నాటివీరచరిత్ర |
| `cleaners/srirama_parinayamu.py` | శ్రీరమాపరిణయము |
| `cleaners/poems.py` | Generic cleaner for non-source-specific edge cases |

## Per-source crawlers (fork-only)

| Crawler | Source URL |
| --- | --- |
| `crawlers/basava_puranam.py` | te.wikisource.org |
| `crawlers/dwipada_bhagavatam.py` | te.wikisource.org |
| `crawlers/palanati_veera_charitra.py` | sahityasourabham.blogspot.com |
| `crawlers/ranganatha_ramayanam.py` | andhrabharati.com (with checkpoint resume) |
| `crawlers/srirama_parinayamu.py` | te.wikisource.org |

These mirror the standalone scripts at `../../../../crawlers/`. The
canonical version of this sub-package
(`../../../../src/dwipada/data/`) does not bundle them — it expects
the user to run the project-root crawlers separately.

## Usage

```bash
# Run all crawlers (this fork's bundled copies)
python -m dwipada.data.crawlers.basava_puranam
python -m dwipada.data.crawlers.dwipada_bhagavatam
python -m dwipada.data.crawlers.ranganatha_ramayanam
python -m dwipada.data.crawlers.palanati_veera_charitra
python -m dwipada.data.crawlers.srirama_parinayamu

# Clean
dwipada clean basava_puranam
dwipada clean dwipada_bhagavatam
# ...

# Consolidate
dwipada consolidate
# Output: data/consolidated_dwipada.json
```

## Related

- Canonical version (no `crawl_base.py`, no `crawlers/`):
  `../../../../src/dwipada/data/`
- Project-root standalone crawlers: `../../../../crawlers/`
- Downstream sub-package: `../dataset/`
- Paper Section 4 (Dataset Construction)
