# Phase 1: Data Collection — Data Sources Summary

## Context
This phase covers the data collection process for building dwipada and ragale datasets. Data was gathered through web crawling (for Telugu dwipada texts) and manual curation (for Kannada ragale texts).

## Dwipada Data Sources (Telugu — Web Crawled)

### 1. రంగనాథ రామాయణము (Ranganatha Ramayanam)
- **Author**: గోన బుద్ధారెడ్డి (Gona Budda Reddy)
- **Source**: AndhaBharati.com
- **URL**: `https://www.andhrabharati.com/itihAsamulu/RanganathaRamayanamu/`
- **Structure**: 7 Kandas, 405 chapters total
  - బాలకాండము (31), అయోధ్యాకాండము (35), అరణ్యకాండము (28), కిష్కింధాకాండము (25), సుందరకాండము (27), యుద్ధకాండము (170), ఉత్తరకాండము (89)
- **Couplets**: 26,296 (83.5% pure)
- **Crawler**: `src/dwipada/data/crawlers/ranganatha_ramayanam.py`

### 2. బసవపురాణము (Basava Puranam)
- **Source**: Telugu Wikisource (te.wikisource.org)
- **URL**: `https://te.wikisource.org/wiki/బసవపురాణము/`
- **Sub-URLs**:
  - `https://te.wikisource.org/wiki/బసవపురాణము/ప్రథమాశ్వాసము`
  - `https://te.wikisource.org/wiki/బసవపురాణము/ద్వితీయాశ్వాసము`
  - `https://te.wikisource.org/wiki/బసవపురాణము/తృతీయాశ్వాసము`
- **Structure**: 3 ఆశ్వాసములు (Ashvasams)
- **Couplets**: 2,454 (76.3% pure)
- **Crawler**: `src/dwipada/data/crawlers/basava_puranam.py`

### 3. ద్విపద భాగవతము (Dwipada Bhagavatam)
- **Source**: Telugu Wikisource (te.wikisource.org)
- **URL**: `https://te.wikisource.org/wiki/ద్విపదభాగవతము/`
- **Structure**: 3 Kandas — మధురాకాండము, కల్యాణకాండము, జగదభిరక్షకాండము
- **Couplets**: 737 + 2,420 = 3,157 (two crawl passes; 87.5% and 82.8% pure)
- **Crawler**: `src/dwipada/data/crawlers/dwipada_bhagavatam.py`

### 4. పలనాటి వీర చరిత్ర (Palanati Veera Charitra)
- **Author**: శ్రీనాథభట్ట (Sri Natha Bhatta) — Manjari Dwipada variant
- **Source**: Sahitya Sourabham Blog (sahityasourabham.blogspot.com)
- **URL pattern**: `https://sahityasourabham.blogspot.com/search?q=శ్రీనాధభట్టకృత " పల్నాటివీరచరిత్ర " -- ద్విపదకావ్యం - {1..33}`
- **Structure**: 33 blog posts
- **Couplets**: 783 (8.4% pure — lowest purity due to Manjari variant)
- **Crawler**: `src/dwipada/data/crawlers/palanati_veera_charitra.py`

### 5. శ్రీరమాపరిణయము (Sri Rama Parinayamu)
- **Author**: తరిగొండ వెంగమాంబ (Tarigonda Vangamamba)
- **Source**: Telugu Wikisource (te.wikisource.org)
- **URL**: `https://te.wikisource.org/wiki/శ్రీరమాపరిణయము/పాఠం`
- **Structure**: 28 chapters (single page)
- **Couplets**: 392 (96.2% pure — highest purity)
- **Crawler**: `src/dwipada/data/crawlers/srirama_parinayamu.py`

### Dwipada Totals
| Dataset | Source | Couplets | Pure | Purity |
|---------|--------|-------:|-----:|-------:|
| రంగనాథ రామాయణము | AndhaBharati.com | 26,296 | 21,947 | 83.5% |
| బసవపురాణము | te.wikisource.org | 2,454 | 1,871 | 76.3% |
| ద్విపద భాగవతము | te.wikisource.org | 3,157 | 2,649 | ~84% |
| పలనాటి వీర చరిత్ర | sahityasourabham.blogspot.com | 783 | 66 | 8.4% |
| శ్రీరమాపరిణయము | te.wikisource.org | 392 | 377 | 96.2% |
| **Total** | | **33,082** | **26,910** | **81.3%** |

## Ragale Data Sources (Kannada)

### 1. KN Ningaiah Blog (Web Crawled)
- **Source**: knningaiah.blogspot.com
- **URL**: `https://knningaiah.blogspot.com/`
- **Content**: Kannada ragale poems collected from the blog

### 2. Manually Curated / Synthetic Data
- `synthetic_data/kannada_ragale.json` — Kannada poems with ragale prosodic analysis (gana breakdown, yati, prasa, anuprasa)
- `synthetic_data/nlp_favi_kannadadataset.txt` — Kannada poetry dataset with ragale analysis

## Synthetic Datasets

### Telugu Dwipada (LLM-Generated via Gemini API)

Each contributor generated ~1,000 poems across 100 topics:

| Dataset | Source | Couplets | Pure | Purity |
|---------|--------|-------:|-----:|-------:|
| Samvaran | Gemini API | 1,544 | — | — |
| Mahesh | Gemini API | 984 | — | — |
| Pradeep | Gemini API | 1,004 | — | — |
| **Combined (after dedup)** | Gemini API | **3,496** | **2,433** | **69.6%** |

### Kannada Ragale (LLM-Generated, reported by Favi)

| Dataset | Source | Couplets | Pure | Purity |
|---------|--------|-------:|-----:|-------:|
| kannada_ragale.json | Gemini API | 110 | — | — |
| nlp_favi_kannadadataset.txt | Gemini API | 120 | — | — |
| **Total** | | **230** | — | — |

### Final Combined Dataset (`dwipada_augmented_dataset.json`)

| Component | Entries | % of Total |
|-----------|-------:|----------:|
| Real (classical corpus, metrically pure) | 26,910 | 91.7% |
| Synthetic (LLM-generated, filtered) | 2,433 | 8.3% |
| **Total** | **29,343** | **100%** |

**Quality Validation (final dataset):**
- Metrical purity (chandass integrity): 100%
- Gana sequence correctness: 100%
- Prasa matching: 100%
- Yati matching: 100%

## Crawler Infrastructure
- **Base crawler**: `src/dwipada/data/crawl_base.py` (HTTP fetch with retry/backoff, HTML cleaning)
- **Base cleaner**: `src/dwipada/data/clean_base.py` (punctuation removal, arasunna handling, metadata-aware cleaning)
- **Output**: Raw data in `data/`, cleaned datasets in `datasets/`, consolidated in `data/consolidated_dwipada.json`
