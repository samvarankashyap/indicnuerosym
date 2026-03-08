# Phase 3: Data Validation — Consolidated Report

## Context

This phase validates the quality, consistency, and diversity of the final dataset (`datasets/dwipada_augmented_dataset.json`) comprising **29,343** Telugu dwipada couplets with Telugu and English meanings. Validation is structured as a four-level pipeline complemented by an LLM-as-a-Judge evaluation on a stratified sample.

### Dataset Composition

| Component | Records | % of Total |
|-----------|--------:|----------:|
| Classical corpus (metrically pure) | 26,910 | 91.7% |
| Synthetic / LLM-generated (filtered) | 2,433 | 8.3% |
| **Total** | **29,343** | **100%** |

---

## 1. Length Ratio Analysis

We enforce a character-count ratio between the Telugu prose meaning (P) and the poetic verse (V) such that **0.8 ≤ len(P) / len(V) ≤ 2.5**. Ratios outside this band signal either overly terse or excessively verbose translations.

### Results

| Metric | Value |
|--------|------:|
| **Pass rate** | 29,085 / 29,343 (**99.1%**) |
| Mean ratio | 1.64 |
| Median ratio | 1.61 |
| Min ratio | 0.79 |
| Max ratio | 5.11 |

### Length Ratio Distribution

| Range | Count | Share |
|-------|------:|------:|
| 0.00 – 0.50 | 0 | 0.0% |
| 0.50 – 0.80 | 1 | < 0.1% |
| 0.80 – 1.00 | 78 | 0.3% |
| 1.00 – 1.50 | 9,698 | 33.1% |
| 1.50 – 2.00 | 16,371 | 55.8% |
| 2.00 – 5.11 | 3,195 | 10.9% |

The bulk of the dataset (88.9%) falls within the 1.0–2.0 band, with a long but thin right tail. Only **258 records** (0.9%) fall outside the enforced bounds.

### Top 5 Outliers

| Index | Ratio | Poem (truncated) |
|------:|------:|------------------|
| #13150 | 5.11 | నతులలంకాపురాటన కాళరాత్రి … |
| #834 | 4.73 | ససమసాహసు గంసు నవలీల జంపి … |
| #15898 | 4.71 | చందనపున్నాగ సహకారతరుల … |
| #10819 | 4.65 | నరలోకనాయక నాగకన్యకలు … |
| #3360 | 4.36 | స్థావరజంగ మాత్మకమైన ధరణి … |

---

## 2. Vector Space Similarity (Semantic Fidelity)

We measure cosine similarity between three text pairs using three multilingual sentence-embedding models. The primary quality gate is **LaBSE Telugu Meaning ↔ English Meaning ≥ 0.65**.

### Cross-Model Summary

| Model | Primary Pair | Pass (≥ 0.65) | Pass Rate | Mean | Median |
|-------|-------------|---------------:|----------:|-----:|-------:|
| **LaBSE** | Telugu ↔ English Meaning | 22,634 | **77.1%** | 0.7212 | 0.7338 |
| mSBERT-mpnet | Poem ↔ Telugu Meaning | 18,699 | 63.7% | 0.6848 | 0.7014 |
| L3Cube-IndicSBERT | Poem ↔ Telugu Meaning | 11,759 | 40.1% | 0.6136 | 0.6183 |

### Detailed Results by Model

#### 2.1 LaBSE

| Pair | Pass (≥ 0.65) | Pass % | Mean | Median | Min | Max |
|------|---------------:|-------:|-----:|-------:|----:|----:|
| Poem ↔ Telugu Meaning | 471 | 1.6% | 0.2851 | 0.2738 | −0.17 | 0.89 |
| Poem ↔ English Meaning | 271 | 0.9% | 0.2811 | 0.2735 | −0.20 | 0.87 |
| Telugu ↔ English Meaning | 22,634 | **77.1%** | 0.7212 | 0.7338 | 0.27 | 0.95 |

**LaBSE Telugu ↔ English Meaning Distribution**

| Similarity Range | Count | Share |
|-----------------|------:|------:|
| 0.00 – 0.30 | 3 | < 0.1% |
| 0.30 – 0.50 | 592 | 2.0% |
| 0.50 – 0.70 | 10,620 | 36.2% |
| 0.70 – 0.85 | 16,294 | 55.5% |
| 0.85 – 1.00 | 1,834 | 6.3% |

#### 2.2 mSBERT-mpnet

| Pair | Pass (≥ 0.65) | Pass % | Mean | Median | Min | Max |
|------|---------------:|-------:|-----:|-------:|----:|----:|
| Poem ↔ Telugu Meaning | 18,699 | 63.7% | 0.6848 | 0.7014 | 0.12 | 0.96 |
| Poem ↔ English Meaning | 1,295 | 4.4% | 0.4244 | 0.4221 | −0.06 | 0.89 |
| Telugu ↔ English Meaning | 6,963 | 23.7% | 0.5009 | 0.5154 | −0.10 | 0.98 |

#### 2.3 L3Cube-IndicSBERT

| Pair | Pass (≥ 0.65) | Pass % | Mean | Median | Min | Max |
|------|---------------:|-------:|-----:|-------:|----:|----:|
| Poem ↔ Telugu Meaning | 11,759 | 40.1% | 0.6136 | 0.6183 | 0.01 | 0.95 |
| Poem ↔ English Meaning | 9 | 0.0% | 0.2546 | 0.2506 | −0.13 | 0.67 |
| Telugu ↔ English Meaning | 6,160 | 21.0% | 0.5787 | 0.5845 | 0.15 | 0.86 |

### Interpretation

- **LaBSE** excels at cross-lingual (Telugu ↔ English) alignment, making it the strongest quality gate for translation fidelity. 77.1% of pairs exceed the 0.65 threshold.
- **mSBERT-mpnet** shows the highest intra-language (Poem ↔ Telugu Meaning) alignment at 63.7%, reflecting its strength in same-script semantic similarity.
- **L3Cube-IndicSBERT**, tuned for Indic languages, performs moderately on Poem ↔ Telugu (40.1%) but poorly on cross-lingual pairs, consistent with its monolingual design.
- The low Poem ↔ Meaning scores across all models are expected: classical verse uses archaic, compressed Telugu that differs structurally from modern prose paraphrases.

---

## 3. Lexical Diversity (Corpus Health)

### Type-Token Ratio (TTR) Filtering

We calculate the TTR for the aggregate dataset to measure vocabulary richness and identify duplicate or near-duplicate verses.

| Metric | Value |
|--------|------:|
| Total tokens (Gemma 3 tokenizer) | 845,701 |
| Telugu tokens in vocab | 1,784 |
| Telugu tokens used in corpus | 1,331 |
| **Telugu token coverage** | **74.6%** |
| **Avg per-poem TTR** | **0.8975** |

A per-poem TTR of ~0.90 indicates high lexical variety within individual couplets — each verse uses a largely distinct vocabulary.

### Duplicate Detection

| Category | Groups | Poems Affected |
|----------|-------:|---------------:|
| Exact duplicates | 946 | 2,227 (7.6%) |
| Near duplicates | 1,024 | 2,386 (8.1%) |

**Sample Exact Duplicate Groups (top 5 by frequency)**

| Frequency | Poem (truncated) |
|----------:|------------------|
| 6× | తలపు నీ మీదిది తప్పదు నాకు / బలుకు లేటికి నిన్నుబట్టెద వెదకి |
| 6× | దీపము వెలుగులు దిశలెల్ల నిండు / పాపము తొలగంగ పరుగెత్తి రాదె |
| 6× | వెలయగ చీకట్లు విడిపోవ చూడు / కలతలు మాపంగ కనులార నవ్వు |
| 6× | పొరుగున చీకటి పోవంగ చేసె / మరుగున ఉన్నట్టి మహిమలు చూపె |
| 6× | మితముగ చమురును మింగేను దీక్ష / సతతము వెలుగును సంతోష మొసగ |

The duplicates are concentrated in the synthetic portion of the dataset. These can be pruned in downstream fine-tuning data preparation.

---

## 4. LLM-as-a-Judge Evaluation

### Setup

- **Model**: Gemini 3.1 Pro (via Vertex AI Batch API)
- **Sample size**: 250 poems (stratified random sampling, seed=42)
- **Sampling strategy**: Stratified across 6 sources (≈42 per source), with `dwipada_bhagavatam` excluded
- **Rubric**: 5-dimensional evaluation (1–5 scale per dimension), inspired by G-Eval and GEMBA frameworks

### Source Distribution

| Source | Samples |
|--------|--------:|
| రంగనాథ రామాయణము (ranganatha_ramayanam) | 42 |
| బసవపురాణము (basava_puranam) | 42 |
| ద్విపద భాగవతము 2 (dwipada_bhagavatam2) | 42 |
| శ్రీరమాపరిణయము (srirama_parinayamu) | 42 |
| పలనాటి వీర చరిత్ర (palanati_veera_charitra) | 41 |
| Synthetic (LLM-generated) | 41 |
| **Total** | **250** |

### Dimension-wise Scores

| Dimension | Mean | 5/5 | 4/5 | 3/5 | 2/5 | 1/5 |
|-----------|-----:|----:|----:|----:|----:|----:|
| **Semantic Fidelity** | 4.73 | 200 (80.0%) | 36 (14.4%) | 10 (4.0%) | 4 (1.6%) | 0 (0.0%) |
| **Completeness** | 4.76 | 202 (80.8%) | 39 (15.6%) | 7 (2.8%) | 2 (0.8%) | 0 (0.0%) |
| **Cultural & Contextual Accuracy** | 4.89 | 234 (93.6%) | 7 (2.8%) | 6 (2.4%) | 3 (1.2%) | 0 (0.0%) |
| **Telugu Linguistic Quality** | 4.98 | 245 (98.0%) | 4 (1.6%) | 1 (0.4%) | 0 (0.0%) | 0 (0.0%) |
| **English Linguistic Quality** | 4.90 | 226 (90.4%) | 22 (8.8%) | 2 (0.8%) | 0 (0.0%) | 0 (0.0%) |

### Aggregate Score Distribution (Total / 25)

| Score | Count | Share |
|------:|------:|------:|
| 25/25 | 172 | 68.8% |
| 24/25 | 33 | 13.2% |
| 23/25 | 22 | 8.8% |
| 22/25 | 8 | 3.2% |
| 21/25 | 6 | 2.4% |
| 20/25 | 1 | 0.4% |
| 19/25 | 2 | 0.8% |
| 18/25 | 4 | 1.6% |
| 17/25 | 1 | 0.4% |
| 16/25 | 1 | 0.4% |

**Mean total score: 24.25 / 25 (97.0%)**

### Overall Verdicts

| Verdict | Count | Share |
|---------|------:|------:|
| **Excellent** | 226 | 90.4% |
| Good | 16 | 6.4% |
| Acceptable | 7 | 2.8% |
| Poor | 1 | 0.4% |

### Key Takeaways

- **98.0%** of samples received a Telugu linguistic quality score of 5/5, confirming high-quality Telugu prose generation.
- **93.6%** scored 5/5 on cultural and contextual accuracy, indicating reliable handling of mythological and philosophical content.
- Only **1 sample** (0.4%) was rated "Poor" overall, demonstrating robust translation quality across the dataset.

---

## 5. Cross-Level Validation Summary

| Validation Level | Description | Pass Rate |
|-------------------|-------------|----------:|
| Level 1: Chandass Scanner | Prosodic integrity (gana, prasa, yati) | 29,343 / 29,343 (**100.0%**) |
| Level 2: Sanity Checks | Length ratio, non-empty fields | 26,658 / 29,343 (**90.8%**) |
| Level 3: Semantic Fidelity | LaBSE Telugu ↔ English ≥ 0.65 | 22,634 / 29,343 (**77.1%**) |
| Level 4: Lexical Diversity | TTR and duplicate detection | Avg TTR 0.90 |
| LLM-as-a-Judge | 5-dim rubric (n=250) | 226 / 250 Excellent (**90.4%**) |

### Combined Pass Rates

| Category | Count | Rate |
|----------|------:|-----:|
| Pass ALL levels (1–3) | 20,207 / 29,343 | **68.9%** |
| Fail Level 2 only | 2,427 / 29,343 | 8.3% |
| Fail Level 3 only | 6,451 / 29,343 | 22.0% |
| Fail Level 1 only | 0 / 29,343 | 0.0% |

---

## Conclusion

The dataset demonstrates strong quality across all validation dimensions:

1. **Structural integrity is perfect** — every couplet passes prosodic validation (gana sequences, prasa, yati).
2. **Translation alignment is high** — 99.1% of records have appropriate length ratios, and 77.1% exceed the LaBSE cross-lingual similarity threshold.
3. **Lexical diversity is healthy** — per-poem TTR of 0.90 with 74.6% Telugu token coverage in the Gemma 3 vocabulary.
4. **Human-comparable quality** — LLM-as-a-Judge rates 90.4% of sampled translations as "Excellent" with a mean score of 24.25/25.
5. **Known issues** — 7.6% exact duplicates (concentrated in synthetic data) and a right-tail of verbose translations (10.9% with ratio > 2.0) are candidates for downstream pruning.

*Validation runtime: 7m 17s on 29,343 records (2026-03-06).*
