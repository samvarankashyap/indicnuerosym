# dwipada.data — Data Acquisition and Cleaning

The crawl → clean → consolidate pipeline that turns five Telugu
classical literature sources into a single
`data/consolidated_dwipada.json`. The output of this sub-package
feeds the dataset preparation in `../dataset/`.

## Files

| File | Purpose |
| --- | --- |
| `clean_base.py` | Shared text-cleaning utilities (character removal, whitespace normalisation, line stripping, footnote removal). Used by every per-source cleaner. |
| `consolidate.py` | Walk every `data/<source>/*.txt` file and emit one JSON record per couplet into `data/consolidated_dwipada.json`. Records include the source name, original file path, line numbers, and the raw couplet text. |
| `cleaners/` | Per-source cleaners — one per literary work (see below). |

## Per-source cleaners

| Cleaner | Source work |
| --- | --- |
| `cleaners/basava_puranam.py` | బసవపురాణము |
| `cleaners/dwipada_bhagavatam.py` | ద్విపదభాగవతము |
| `cleaners/palanati_veera_charitra.py` | పల్నాటివీరచరిత్ర |
| `cleaners/srirama_parinayamu.py` | శ్రీరమాపరిణయము |
| `cleaners/poems.py` | Generic cleaner that handles non-source-specific edge cases |

The crawlers themselves live at the project root in `../../../crawlers/`,
not in this sub-package. The chandomitra-fork copy of this folder
(`../../../chandomitra/src/dwipada/data/`) additionally bundles
`crawl_base.py` and a `crawlers/` subdirectory that mirrors the root
`crawlers/` folder, so chandomitra benchmarks can run end-to-end
without external paths.

## Usage

```bash
# Run an individual cleaner against an already-crawled source
python -m dwipada.data.cleaners.basava_puranam

# Or clean every source via the unified CLI
dwipada clean basava_puranam
dwipada clean dwipada_bhagavatam
dwipada clean palanati_veera_charitra
dwipada clean srirama_parinayamu
dwipada clean poems         # generic cleaner

# Then consolidate everything
dwipada consolidate
# Output: data/consolidated_dwipada.json
```

## Output schema

Each record in `data/consolidated_dwipada.json`:

```json
{
  "poem": "line 1\nline 2",
  "source_file": "data/ranganatha_ramayanam/01_BalaKanda/005_*.txt",
  "work": "ranganatha_ramayanam",
  "couplet_number": 5
}
```

## Related

- `../../../crawlers/` — the web crawlers that produce the raw `data/*/`
  text files this sub-package consumes
- `../../../data/` — output of the crawlers and input to this sub-package
- `../dataset/` — downstream sub-package that turns the consolidated
  JSON into training-ready datasets
- Paper Section 4 (Dataset Construction)
