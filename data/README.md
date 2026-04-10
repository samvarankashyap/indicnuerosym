# Data

Raw crawled Telugu poetry texts organized by source. Each source
subdirectory has its own README with crawler / file-count / kanda
breakdown.

## Sources

| Directory | Source | Raw Couplets | Validated | Purity | Sub-README |
|---|---|---:|---:|---:|---|
| `ranganatha_ramayanam/` | Ranganatha Ramayanam | 26,296 | 21,828 | 83.0% | [README](ranganatha_ramayanam/README.md) |
| `basava_puranam/` | Basava Puranam | 2,454 | 1,859 | 75.8% | [README](basava_puranam/README.md) |
| `dwipada_bhagavatam/` | Dwipada Bhagavatam | 3,157 | 2,002 | 63.4% | [README](dwipada_bhagavatam/README.md) |
| `dwipada_bhagavatam2/` | Dwipada Bhagavatam (secondary dump) | — | — | — | [README](dwipada_bhagavatam2/README.md) |
| `palanati_veera_charitra/` | Palanati Veera Charitra (mañjari variant) | 783 | 65 | 8.3% | [README](palanati_veera_charitra/README.md) |
| `srirama_parinayamu/` | Srirama Parinayamu | 392 | 374 | 95.4% | [README](srirama_parinayamu/README.md) |
| `synthetic_data/` | LLM-generated synthetic poems | 3,496 | 1,753 | 50.1% | [README](synthetic_data/README.md) |

(Numbers match the paper's `tab:sources`. The Palanati low pass rate
is expected — that work uses the *mañjari dvipada* variant which
relaxes the *prāsa* constraint.)

## Consolidated

| File | Description |
|---|---|
| `consolidated_dwipada.json` | Merged JSON of all sources (~26 MB, ~36K raw couplets before validation) |

The validated 27,881-couplet `dwipada_master_dataset.json` lives in
`../datasets/`, not here.
