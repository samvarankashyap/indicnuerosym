# Phase 4: Instruct Fine-Tuning Dataset Preparation

## Context

This phase transforms the validated master dataset (`datasets/dwipada_augmented_dataset.json`, 29,343 entries) into instruction-completion pairs suitable for supervised fine-tuning (SFT) of Gemma 3. The pipeline is invoked via `dwipada prepare` and implemented in `src/dwipada/training/prepare_data.py`.

---

## 1. Input Schema

Each entry in the augmented dataset contains:

```json
{
  "poem": "మా రాముబాణనిర్మథితమాంసముల \nకీ రాదె నీ నాక మేల యిచ్చెదవు",
  "telugu_meaning": "...",
  "english_meaning": "...",
  "source": "ranganatha_ramayanam",
  "is_synthetic_data": false,
  "chandassu_analysis": { "line_1": {...}, "line_2": {...} },
  "word_to_word_meaning": {...}
}
```

---

## 2. Pipeline

```
dwipada_augmented_dataset.json
        │
        ▼
┌─ 1. Load & group by source ──────────────────────────┐
│   Group entries into source buckets                   │
│   (ranganatha_ramayanam, basava_puranam, etc.)        │
└───────────────────────────────────────────────────────┘
        │
        ▼
┌─ 2. Metrical purity filter ──────────────────────────┐
│   analyze_dwipada(poem) → keep only 100% overall     │
│   score (gana + prasa + yati all correct)             │
└───────────────────────────────────────────────────────┘
        │
        ▼
┌─ 3. Deduplication ───────────────────────────────────┐
│   Normalize whitespace, track seen poems, skip dupes  │
└───────────────────────────────────────────────────────┘
        │
        ▼
┌─ 4. Instruction generation ──────────────────────────┐
│   Randomly select from 4 template categories          │
│   (see §3 below)                                      │
└───────────────────────────────────────────────────────┘
        │
        ▼
┌─ 5. Format: prepend DWIPADA_RULES_BLOCK ─────────────┐
│   input = rules_block + "\n\n" + instruction          │
│   output = poem                                       │
└───────────────────────────────────────────────────────┘
        │
        ▼
┌─ 6. Shuffle (seed=42) & split 80/10/10 ──────────────┐
│   → train.jsonl / val.jsonl / test.jsonl              │
└───────────────────────────────────────────────────────┘
```

---

## 3. Instruction Templates

Each sample is assigned one instruction at random from the available categories based on its metadata:

| Category | Condition | Example |
|----------|-----------|---------|
| **Generic** | Always available | `ద్విపదలో ఒక పద్యం వ్రాయండి.` / `Write a Telugu dwipada couplet following the above rules.` |
| **Work-style** | Non-synthetic source | `రంగనాథ రామాయణము శైలిలో ద్విపద వ్రాయండి.` / `Compose a dwipada in the style of Basava Puranam.` |
| **Bhavam (Telugu)** | Has `telugu_meaning` | `ఈ భావంతో ద్విపద వ్రాయండి: {meaning}` |
| **Bhavam (English)** | Has `english_meaning` | `Write a dwipada that expresses: {meaning}` |

This diversity ensures the model learns to respond to varied prompt styles — generic, style-conditioned, and meaning-conditioned — in both Telugu and English.

---

## 4. Rules Block

Every training sample's `input` is prefixed with the dwipada rules in Telugu, covering:

- **Structure**: 2 lines, each with 4 ganas (3 Indra + 1 Surya)
- **Indra ganas**: నల (IIII), నగ (IIIU), సల (IIUI), భ (UII), ర (UIU), త (UUI)
- **Surya ganas**: న (III), హ/గల (UI)
- **Guru/Laghu**: deerga vowels, anusvara, visarga, conjuncts → Guru; short vowels → Laghu
- **Prasa**: 2nd syllable consonant must match across both lines
- **Yati**: 1st gana's first syllable must match 3rd gana's first syllable (maitri allowed)

This embeds the metrical specification directly in the prompt so the model always has the rules in context during training.

---

## 5. Output

| File | Content |
|------|---------|
| `training_data/train.jsonl` | 80% of filtered samples |
| `training_data/val.jsonl` | 10% |
| `training_data/test.jsonl` | 10% (held-out) |
| `training_data/data_stats.json` | Per-source counts, pass rates, split sizes |

### Sample JSONL Entry

```json
{
  "input": "ద్విపద నియమాలు (Dwipada Rules):\n- ద్విపద = 2 పాదాలు ...\n\nరంగనాథ రామాయణము శైలిలో ద్విపద వ్రాయండి.",
  "output": "మా రాముబాణనిర్మథితమాంసముల \nకీ రాదె నీ నాక మేల యిచ్చెదవు"
}
```

---

## 6. Usage

```bash
dwipada prepare
```

Reads from `datasets/dwipada_augmented_dataset.json`, writes to `training_data/`. Requires the dwipada package to be installed (`pip install -e .`).
