# Telugu Poetry Crawlers

Standalone web crawlers for collecting classical Telugu poetry texts (primarily in Dwipada verse form) from various online sources. Each crawler is an independent Python script that can be run without installing any project packages.

## Overview

These crawlers collect Telugu classical poetry from 5 different sources across 3 websites. The collected texts are used for building datasets for Telugu poetry generation research.

| Crawler | Literary Work | Author | Source Website | Chapters |
|---------|--------------|--------|---------------|----------|
| `palanati_veera_charitra.py` | పల్నాటివీరచరిత్ర | శ్రీనాథభట్ట | sahityasourabham.blogspot.com | 33 blog posts |
| `srirama_parinayamu.py` | శ్రీరమాపరిణయము | తరిగొండ వెంగమాంబ | te.wikisource.org | 28 chapters |
| `ranganatha_ramayanam.py` | రంగనాథ రామాయణము | గోన బుద్ధారెడ్డి | andhrabharati.com | 405 chapters (7 kandas) |
| `basava_puranam.py` | బసవపురాణము | పాల్కురికి సోమనాథుడు | te.wikisource.org | 3 ఆశ్వాసములు |
| `dwipada_bhagavatam.py` | ద్విపదభాగవతము | — | te.wikisource.org | 3 kandas |

## Folder Structure

```
crawlers/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── crawl_base.py                      # Shared utilities (HTTP, HTML cleaning, etc.)
├── palanati_veera_charitra.py         # Crawler: Palanati Veera Charitra
├── srirama_parinayamu.py              # Crawler: Sri Rama Parinayamu
├── ranganatha_ramayanam.py            # Crawler: Ranganatha Ramayanam
├── basava_puranam.py                  # Crawler: Basava Puranam
├── dwipada_bhagavatam.py              # Crawler: Dwipada Bhagavatam
└── data/                              # Created automatically on first run
    ├── palanati_veera_charitra/        # Flat: NNN_<title>.txt files
    ├── srirama_parinayamu/             # Flat: NNN_<title>.txt files
    ├── ranganatha_ramayanam/           # Nested by kanda
    │   ├── checkpoint.json             # Resumable download state
    │   ├── 01_BalaKanda/
    │   │   ├── 001_<title>.txt
    │   │   ├── 002_<title>.txt
    │   │   └── ...
    │   ├── 02_AyodhyaKanda/
    │   ├── 03_AranyaKanda/
    │   ├── 04_KishkindhaKanda/
    │   ├── 05_SundaraKanda/
    │   ├── 06_YuddhaKanda/
    │   └── 07_UttaraKanda/
    ├── basava_puranam/                 # Nested by ఆశ్వాసము
    │   ├── 001_ప్రథమాశ్వాసము/
    │   │   ├── 001_<title>.txt
    │   │   └── ...
    │   ├── 002_ద్వితీయాశ్వాసము/
    │   └── 003_తృతీయాశ్వాసము/
    └── dwipada_bhagavatam2/            # One file per kanda
        ├── 1_madhurakanda.txt
        ├── 1_kalyanakanda.txt
        └── 1_jagadabhirakshakanda.txt
```

## Prerequisites

- Python 3.8 or higher
- Internet connection (to fetch pages from source websites)

## Quick Start

1. **Install dependencies:**

   ```bash
   cd crawlers
   pip install -r requirements.txt
   ```

2. **Run a crawler:**

   ```bash
   python srirama_parinayamu.py
   ```

   This will create a `data/srirama_parinayamu/` folder with the crawled text files.

3. **Run all crawlers:**

   ```bash
   python palanati_veera_charitra.py
   python srirama_parinayamu.py
   python ranganatha_ramayanam.py
   python basava_puranam.py
   python dwipada_bhagavatam.py
   ```

> **Note:** The `data/` folder is created automatically the first time any crawler is run. You do not need to create it manually.

---

## Crawler Details

### 1. Palanati Veera Charitra (`palanati_veera_charitra.py`)

**Source:** `sahityasourabham.blogspot.com`
**Work:** పల్నాటివీరచరిత్ర by శ్రీనాథభట్ట
**Type:** Blogspot search results (33 blog posts)

```bash
python palanati_veera_charitra.py
```

**How it works:**
- Constructs search query URLs for each of the 33 blog posts
- Extracts content from `post-body` div elements
- Identifies section headings by centered red/large text or h2/h3 tags
- Preserves dwipada verse line structure

**Output:** `data/palanati_veera_charitra/`
- Flat directory with files named `NNN_<section_title>.txt`
- 1-second delay between page requests

**Output file format:**
```
# గ్రంథము: పల్నాటివీరచరిత్ర
# రచయిత: శ్రీనాథభట్ట
# భాగము: 001
# శీర్షిక: <section title>

<verse content>
```

---

### 2. Sri Rama Parinayamu (`srirama_parinayamu.py`)

**Source:** `te.wikisource.org`
**Work:** శ్రీరమాపరిణయము by తరిగొండ వెంగమాంబ
**Type:** Single Wikisource page with 28 chapters

```bash
python srirama_parinayamu.py
```

**How it works:**
- Fetches a single page (only 1 HTTP request needed)
- Identifies chapter boundaries using centered `tiInherit` divs
- Extracts content from `<div class="poem">` sections and `<p>` tags
- Validates against a list of 28 expected chapter titles
- SSL verification is disabled (Wikisource certificate issue)

**Output:** `data/srirama_parinayamu/`
- Flat directory with files named `NNN_<chapter_title>.txt`

**Output file format:**
```
# గ్రంథము: శ్రీరమాపరిణయము
# రచయిత: తరిగొండ వెంగమాంబ
# అధ్యాయము: 001
# శీర్షిక: <chapter title>

<verse content>
```

---

### 3. Ranganatha Ramayanam (`ranganatha_ramayanam.py`)

**Source:** `andhrabharati.com`
**Work:** రంగనాథ రామాయణము
**Type:** Structured website with 405 chapters across 7 kandas

```bash
python ranganatha_ramayanam.py
```

**How it works:**
- Crawls 405 individual chapter pages across 7 kandas (books):
  1. BalaKanda (31 chapters)
  2. AyodhyaKanda (35 chapters)
  3. AranyaKanda (28 chapters)
  4. KishkindhaKanda (25 chapters)
  5. SundaraKanda (27 chapters)
  6. YuddhaKanda (170 chapters)
  7. UttaraKanda (89 chapters)
- Extracts title from `chapter_hdr` div, content from `wmsect` div
- Separately extracts footnotes from `fnlist` divs
- Removes footnote markers, superscripts, and navigation elements
- **Checkpoint system:** saves progress to `checkpoint.json` after each chapter
- **Resumable:** re-running the script skips already-downloaded chapters
- **Rate limited:** 1.5-second delay between requests
- **Interrupt-safe:** Ctrl+C saves progress; re-run to resume

**Output:** `data/ranganatha_ramayanam/`
- Nested directory structure: one folder per kanda
- Files named `NNN_<chapter_title>.txt`
- `checkpoint.json` tracks download progress

**Output file format:**
```
# కాండము: బాలకాండము
# అధ్యాయము: 001
# శీర్షిక: <chapter title>

<verse content>

---
పాదసూచికలు (Footnotes):
[1] footnote text
[2] footnote text
```

---

### 4. Basava Puranam (`basava_puranam.py`)

**Source:** `te.wikisource.org`
**Work:** బసవపురాణము by పాల్కురికి సోమనాథుడు
**Type:** 3 separate Wikisource pages (one per ఆశ్వాసము)

```bash
python basava_puranam.py
```

**How it works:**
- Fetches 3 pages (one per ఆశ్వాసము/book):
  1. ప్రథమాశ్వాసము (First Ashvasam)
  2. ద్వితీయాశ్వాసము (Second Ashvasam)
  3. తృతీయాశ్వాసము (Third Ashvasam)
- Detects section headings using 3 methods (h2/h3 headers, centered divs, bold paragraphs)
- Extracts content from poem divs and paragraph tags
- SSL verification is disabled (Wikisource certificate issue)
- 1-second delay between pages

**Output:** `data/basava_puranam/`
- Nested directory structure: one folder per ఆశ్వాసము
- Files named `NNN_<section_title>.txt`

**Output file format:**
```
# గ్రంథము: బసవపురాణము
# ఆశ్వాసము: ప్రథమాశ్వాసము
# విభాగము: 001
# శీర్షిక: <section title>

<verse content>
```

---

### 5. Dwipada Bhagavatam (`dwipada_bhagavatam.py`)

**Source:** `te.wikisource.org`
**Work:** ద్విపదభాగవతము
**Type:** 3 separate Wikisource pages (one per kanda)

```bash
python dwipada_bhagavatam.py
```

**How it works:**
- Fetches 3 pages (one per kanda/book):
  1. మధురాకాండము (Madhura Kanda)
  2. కల్యాణకాండము (Kalyana Kanda)
  3. జగదభిరక్షకాండము (Jagadabhiraksha Kanda)
- Separates section headings from verse content within poem divs
- **Couplet formatting:** groups verse lines into pairs (2-line couplets) with blank lines between
- Cleans verse lines: removes footnote markers `[N]`, parentheses, quotation marks, trailing page numbers
- SSL verification is disabled (Wikisource certificate issue)

**Output:** `data/dwipada_bhagavatam2/`
- One file per kanda (not per chapter)
- Section headings marked with `# <heading>`

**Output file format:**
```
# <section heading>

verse line 1
verse line 2

verse line 3
verse line 4

# <next section heading>

...
```

---

## Shared Utilities (`crawl_base.py`)

All crawlers import shared functions from `crawl_base.py`. This module provides:

| Function | Purpose |
|----------|---------|
| `fetch_page(url, retries, verify_ssl, timeout)` | HTTP GET with retry (exponential backoff: 1s, 2s, 4s) |
| `sanitize_filename(name, max_length)` | Remove invalid characters, limit length to 50 chars |
| `clean_html_content(soup)` | Remove page numbers, footnotes, edit links from parsed HTML |
| `clean_text(text)` | Remove citations `[N]`, page numbers, normalize whitespace |
| `find_content_div(soup)` | Locate `mw-parser-output` or `prp-pages-output` div |
| `extract_section_content(soup)` | Extract clean text from poem divs and paragraph tags |
| `suppress_ssl_warnings()` | Disable SSL verification warnings |

**Constants:**
- `DATA_DIR` — path to `crawlers/data/` (auto-resolved relative to script location)
- `TIMEOUT` — default request timeout (60 seconds)
- `HEADERS` — User-Agent header (Chrome 120)

## Output Format Conventions

All output `.txt` files follow these conventions:

- **Encoding:** UTF-8
- **Metadata headers:** Lines starting with `#` at the top of each file (work name, author, chapter/section info)
- **Blank line separator:** One blank line between metadata headers and content
- **Verse preservation:** Original line breaks within verses are preserved
- **File naming:** `NNN_<sanitized_title>.txt` where NNN is a zero-padded number
- **Filename sanitization:** special characters (`<>:"/\|?*`) removed, whitespace normalized, max 50 chars

## Troubleshooting

**SSL errors with Wikisource:**
The crawlers for Wikisource sources (srirama_parinayamu, basava_puranam, dwipada_bhagavatam) disable SSL verification due to intermittent certificate issues with `te.wikisource.org`. This is handled automatically.

**Ranganatha Ramayanam interrupted:**
If the crawler is interrupted (Ctrl+C or network failure), simply re-run it. The checkpoint system (`data/ranganatha_ramayanam/checkpoint.json`) tracks which chapters have been downloaded and skips them on the next run.

**No content extracted:**
If a crawler reports "No sections found" or "No content extracted", the source website's HTML structure may have changed. Check the source URL manually in a browser to verify.

**Rate limiting:**
All crawlers include polite delays between requests (1-1.5 seconds). Do not remove these delays, as aggressive crawling may result in being blocked by the source websites.
