# పల్నాటివీరచరిత్ర — Palanati Veera Charitra (raw text)

Raw crawled text of *Palnāṭi Vīra Charitra* by శ్రీనాథభట్ట
(Śrīnātha Bhaṭṭa, 14th century), the heroic ballad of the Palnadu
warriors. Composed in the *mañjari dvipada* variant of the metre,
which relaxes the *prāsa* (rhyme) requirement, so the prosodic
purity rate is much lower than the *samanya dvipada* sources.
Crawled from sahityasourabham.blogspot.com.

## Contents

- **140 `.txt` files** in a single flat directory (no subdirectories)
- File naming: `NNN_<section title>.txt` where `NNN` is a zero-padded number
- Encoding: UTF-8 Telugu
- Each file starts with metadata header lines (`# గ్రంథము:`,
  `# రచయిత:`, `# భాగము:`, `# శీర్షిక:`) followed by the verse content

## Provenance

- **Crawler:** `../../crawlers/palanati_veera_charitra.py`
- **Source URL:** sahityasourabham.blogspot.com (33 blog posts in
  the original; the crawler expands them into 140 sections)
- **Re-crawl:** `python ../../crawlers/palanati_veera_charitra.py`

## Downstream

This raw text is consolidated into `../consolidated_dwipada.json` by
`src/dwipada/data/consolidate.py`. The `Palanati Vira Charitra` row
of the master corpus contributes **783 raw couplets** (only **65**
after prosodic filtering, **8.3% purity**) to the 27,881-couplet
`dwipada_master_dataset.json`. The very low pass rate is **expected
and not a defect**: this work uses *mañjari dvipada*, which relaxes
the *prāsa* constraint enforced by the chandass scanner. The paper
flags this in Section 4.1.

## Related

- Parent: `../README.md` (`data/`)
- Crawler: `../../crawlers/README.md`
- Paper Section 4 (`tab:sources`) — note on the 91.7% rejection rate
