#!/usr/bin/env python3
"""
Large-scale Constrained Dwipada Generation Benchmark.

Generates 500+ poems across diverse prompts from the dwipada corpus,
validates each, and produces a comprehensive evaluation report.

Prompts are sampled from the 28K+ Telugu meanings in the training corpus
to ensure topical diversity. Each prompt is run with multiple seeds.

Usage:
    # Full benchmark (500 poems, ~8 min):
    python domino/constrained_generate_v2.py

    # Quick test (50 poems):
    python domino/constrained_generate_v2.py --num-poems 50

    # Custom settings:
    python domino/constrained_generate_v2.py --num-poems 1000 --seeds-per-prompt 10

    # Include unconstrained baseline:
    python domino/constrained_generate_v2.py --with-baseline
"""

import argparse
import json
import os
import random
import sys
import time
from collections import Counter

import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Gemma3ForCausalLM,
    Gemma3ForConditionalGeneration,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
NFA_DIR = os.path.join(PROJECT_DIR, "nfa_for_dwipada")
sys.path.insert(0, NFA_DIR)

# Import from the main script
sys.path.insert(0, SCRIPT_DIR)
from constrained_generate import (
    generate_poem,
    validate_poem,
    VALID_LINE_LENGTHS,
    MAX_LINE_LENGTH,
    TOP_K,
)


###############################################################################
# 1) DIVERSE PROMPT GENERATION
###############################################################################

# Category 1: Simple topics (one-word or short phrase) — 34 prompts
TOPIC_PROMPTS = [
    "ప్రకృతి", "సముద్రము", "సూర్యోదయము", "చంద్రుడు", "వర్షాకాలము",
    "పర్వతములు", "నదులు", "అడవి", "వసంతకాలము", "రాత్రి",
    "తల్లి", "గురువు", "స్నేహము", "ధైర్యము", "విద్య",
    "దయ", "శాంతి", "కీర్తి", "భక్తి", "వీరత్వము",
    "దేశభక్తి", "కృతజ్ఞత", "సంగీతము", "నృత్యము", "కవిత్వము",
    "యుద్ధము", "ప్రేమ", "వివేకము", "సాహసము", "త్యాగము",
    "ధర్మము", "న్యాయము", "సత్యము", "కరుణ",
]

# Category 2: Mythological/epic themes — 34 prompts
MYTHOLOGICAL_PROMPTS = [
    "హనుమంతుడు సముద్రమును దాటి లంకను చేరెను",
    "శ్రీరాముడు సీతాదేవిని రక్షించుటకు లంకకు వెళ్ళెను",
    "కృష్ణుడు అర్జునునకు గీతోపదేశము చేసెను",
    "రావణుడు సీతను అపహరించి లంకకు తీసుకొనిపోయెను",
    "భగవంతుడు సర్వ ప్రాణులను రక్షించును",
    "లక్ష్మణుడు రామునితో అడవికి వెళ్ళెను",
    "భరతుడు రాముని పాదుకలను తెచ్చెను",
    "సుగ్రీవుడు రామునికి సహాయము చేసెను",
    "విభీషణుడు రాముని శరణు జొచ్చెను",
    "దశరథుడు రాముని వియోగమున మరణించెను",
    "ద్రౌపది వస్త్రాపహరణ సమయమున కృష్ణుడు రక్షించెను",
    "భీముడు బకాసురుని సంహరించెను",
    "శివుడు గంగను శిరమున ధరించెను",
    "పార్వతి శివుని వివాహమాడెను",
    "నారదుడు భక్తితో విష్ణువును స్తుతించెను",
    "అర్జునుడు కిరాతుని ఎదుర్కొని పాశుపతాస్త్రము పొందెను",
    "సీత అగ్నిప్రవేశము చేసి పవిత్రత నిరూపించెను",
    "వాలిని సంహరించి రాముడు సుగ్రీవునికి రాజ్యమిచ్చెను",
    "కుంభకర్ణుడు నిద్రనుండి మేల్కొని యుద్ధమునకు వచ్చెను",
    "జటాయువు సీతను రక్షించుటకు రావణునితో పోరాడెను",
    "వామనుడు మూడడుగులతో మూడు లోకములను కొలిచెను",
    "నరసింహుడు హిరణ్యకశిపుని సంహరించి ప్రహ్లాదుని రక్షించెను",
    "గంగావతరణము శివుని జటాజూటముపై జరిగెను",
    "సముద్రమథనమున అమృతము లభించెను",
    "రాముడు శివధనుస్సును విరిచి సీతను వివాహమాడెను",
    "కర్ణుడు కవచకుండలములను దానమిచ్చెను",
    "ఏకలవ్యుడు బొటనవ్రేలిని గురుదక్షిణగా ఇచ్చెను",
    "శకుంతల దుష్యంతుని కలిసెను",
    "సావిత్రి యమధర్మరాజుతో వాదించి భర్తను రక్షించెను",
    "ప్రహ్లాదుడు నారాయణుని నామము జపించి రక్షింపబడెను",
    "బలిచక్రవర్తి వామనునకు మూడడుగులు దానమిచ్చెను",
    "కృష్ణుడు గోవర్ధన పర్వతమును ఎత్తెను",
    "రాముడు అహల్యను శాపవిమోచనము చేసెను",
    "హనుమంతుడు సంజీవని పర్వతమును తెచ్చెను",
]

# Category 3: Philosophical/moral themes — 34 prompts
PHILOSOPHICAL_PROMPTS = [
    "తల్లి ప్రేమ అన్నింటికంటే గొప్పది",
    "విజయమునకు కఠోర పరిశ్రమ అవసరము",
    "సత్యమే ధర్మమునకు మూలము",
    "దానము చేయుట గొప్ప ధర్మము",
    "క్షమ వీరుల ఆభరణము",
    "విద్య వినయమును నేర్పును",
    "అహంకారము వినాశనమునకు కారణము",
    "కాలము ఎవరికొరకు ఆగదు",
    "మనసు నిర్మలమైనచో జగమంతయు సుందరము",
    "సంతోషము ధనముకంటే విలువైనది",
    "పెద్దలను గౌరవించుట విధి",
    "అసత్యము చివరకు ఓటమి చెందును",
    "పరోపకారము మానవ ధర్మము",
    "జ్ఞానము అజ్ఞానమును నశింపజేయును",
    "ఐకమత్యమే మహాబలము",
    "ధర్మమార్గమున నడచుట శ్రేయస్కరము",
    "సహనము మానవునికి గొప్ప ఆయుధము",
    "గురువు బోధనలు జీవితమునకు దారిచూపును",
    "మాటలకంటే చేతలు గొప్పవి",
    "లోభము దుఃఖమునకు మూలము",
    "స్నేహము జీవితమునకు ఆధారము",
    "కష్టములలో ధైర్యము కలిగియుండవలెను",
    "మంచివారి సహవాసము జీవితమును మార్చును",
    "అజ్ఞానమే అన్ని కష్టములకు కారణము",
    "నిజమైన బలము మనసులో ఉండును",
    "క్రోధము శత్రువులకంటే ప్రమాదకరము",
    "సేవ చేయుట మానవ జన్మకు సార్థకత",
    "నిరాశలో కూడా ఆశ కలిగియుండవలెను",
    "మంచి మాట మందుకంటే మేలు",
    "కర్మఫలము తప్పక అనుభవించవలసినదే",
    "సంయమము జీవితమునకు అవసరము",
    "ధనము శాశ్వతము కాదు ధర్మమే శాశ్వతము",
    "ప్రతి మనిషికి కర్తవ్యము ముఖ్యము",
    "భయము జయించినవాడే నిజమైన వీరుడు",
]

# Category 4: Nature/descriptive themes — 34 prompts
NATURE_PROMPTS = [
    "సూర్యోదయమున ప్రకృతి అందముగా వెలుగును",
    "నదులు పర్వతముల నుండి సముద్రమునకు ప్రవహించును",
    "వర్షాకాలమున మేఘములు గర్జించును",
    "చంద్రుడు రాత్రిని వెలుగించును",
    "పూలు వసంతమున వికసించును",
    "సముద్రపు అలలు తీరమును తాకును",
    "పక్షులు ప్రభాతమున గానము చేయును",
    "పర్వతశిఖరమున మంచు కరుగును",
    "అడవిలో చెట్లు ఆకాశమును తాకును",
    "వేసవిలో భూమి తపించును",
    "వెన్నెల రాత్రిలో ప్రకృతి ప్రశాంతముగా ఉండును",
    "నదీతీరమున చెట్లు నీడనిచ్చును",
    "వర్షపు చినుకులు భూమిని తడుపును",
    "ఉదయకాలమున పక్షులు గూడు విడిచి ఎగురును",
    "శరత్కాలమున ఆకాశము నిర్మలముగా ఉండును",
    "హేమంతకాలమున మంచు భూమిని కప్పును",
    "సాయంకాలమున సూర్యుడు ఎరుపు రంగులో అస్తమించును",
    "తామరపూలు సరస్సులో వికసించును",
    "గాలి చెట్ల ఆకులను ఊపును",
    "ఇంద్రధనుస్సు వర్షము తరువాత ఆకాశమున కనబడును",
    "కోయిల వసంతమున మధురముగా పాడును",
    "నెమలి వర్షాకాలమున నాట్యము చేయును",
    "జలపాతము పర్వతము నుండి క్రిందికి దూకును",
    "సముద్రతీరమున ఇసుక తిన్నెలు ఏర్పడును",
    "చెట్ల నీడలో ప్రయాణికులు విశ్రమించును",
    "మేఘములు ఆకాశమున తేలియాడును",
    "చేపలు నదిలో ఈదును",
    "తుమ్మెదలు పూలచుట్టూ తిరుగును",
    "సూర్యకాంతిలో మంచుబిందువులు మెరయును",
    "వనములో జింకలు స్వేచ్ఛగా తిరుగును",
    "సంధ్యాకాలమున ఆకాశము రంగులతో నిండును",
    "శీతాకాలమున రాత్రులు పొడవుగా ఉండును",
    "వసంతమున మామిడి చెట్లు పూయును",
    "అలలపై నావ తేలియాడును",
]

# Category 5: Bare instructions — 34 prompts
BARE_PROMPTS = [
    "ఒక ద్విపద పద్యము రాయుము",
    "ఒక అందమైన ద్విపద రచించుము",
    "శ్రీరాముని గురించి ద్విపద చెప్పుము",
    "ఒక పద్యము వ్రాయుము",
    "తెలుగులో ద్విపద పద్యము చెప్పుము",
    "భక్తి రసముతో ఒక ద్విపద రాయుము",
    "వీర రసముతో ద్విపద రచించుము",
    "శృంగార రసముతో ద్విపద చెప్పుము",
    "కరుణ రసముతో ద్విపద రాయుము",
    "శాంత రసముతో ద్విపద రచించుము",
    "హాస్య రసముతో ద్విపద చెప్పుము",
    "రామాయణము నుండి ఒక ద్విపద చెప్పుము",
    "భాగవతము నుండి ఒక ద్విపద రాయుము",
    "మహాభారతము నుండి ఒక ద్విపద రచించుము",
    "ఒక నీతి పద్యము ద్విపదలో చెప్పుము",
    "ఒక ప్రకృతి వర్ణన ద్విపదలో రాయుము",
    "దేవుని స్తుతి ద్విపదలో రాయుము",
    "యుద్ధ వర్ణన ద్విపదలో చెప్పుము",
    "ప్రేమ గురించి ద్విపద రాయుము",
    "విరహ వేదన ద్విపదలో రచించుము",
    "సూర్యుని గురించి ద్విపద చెప్పుము",
    "చంద్రుని గురించి ద్విపద రాయుము",
    "సముద్రము గురించి ద్విపద రచించుము",
    "అడవి గురించి ద్విపద చెప్పుము",
    "తల్లి గురించి ద్విపద రాయుము",
    "గురువు గురించి ద్విపద రచించుము",
    "స్నేహము గురించి ద్విపద చెప్పుము",
    "దేశభక్తి గురించి ద్విపద రాయుము",
    "వీరుడి గురించి ద్విపద రచించుము",
    "రైతు గురించి ద్విపద చెప్పుము",
    "జీవితము గురించి ద్విపద రాయుము",
    "మరణము గురించి ద్విపద రచించుము",
    "కాలము గురించి ద్విపద చెప్పుము",
    "ధర్మము గురించి ద్విపద రాయుము",
]

# Category 6: Sampled from corpus (Telugu meanings of real poems)
def load_corpus_prompts(num=50, seed=42):
    """Sample Telugu meanings from the dwipada corpus."""
    dataset_path = os.path.join(PROJECT_DIR, "datasets", "dwipada_augmented_dataset.json")
    with open(dataset_path) as f:
        data = json.load(f)

    meanings = []
    seen = set()
    for item in data:
        m = item.get("telugu_meaning", "").strip()
        if m.startswith("తెలుగు:"):
            m = m[len("తెలుగు:"):].strip()
        if not m or len(m) < 20 or len(m) > 120:
            continue
        prefix = m[:30]
        if prefix in seen:
            continue
        seen.add(prefix)
        meanings.append(m)

    rng = random.Random(seed)
    rng.shuffle(meanings)
    return meanings[:num]


def build_diverse_prompts(num_prompts=100, seed=42):
    """Build a diverse prompt set from all categories.

    Distribution: equal across all 6 categories (34 each for 200 prompts).
    """
    rng = random.Random(seed)

    n_per_cat = num_prompts // 6
    n_topic = min(n_per_cat, len(TOPIC_PROMPTS))
    n_myth = min(n_per_cat, len(MYTHOLOGICAL_PROMPTS))
    n_phil = min(n_per_cat, len(PHILOSOPHICAL_PROMPTS))
    n_nature = min(n_per_cat, len(NATURE_PROMPTS))
    n_bare = min(n_per_cat, len(BARE_PROMPTS))
    n_corpus = num_prompts - n_topic - n_myth - n_phil - n_nature - n_bare

    prompts = []

    def sample(pool, n):
        if n >= len(pool):
            return list(pool)
        return rng.sample(pool, n)

    prompts.extend([(p, "topic") for p in sample(TOPIC_PROMPTS, n_topic)])
    prompts.extend([(p, "mythological") for p in sample(MYTHOLOGICAL_PROMPTS, n_myth)])
    prompts.extend([(p, "philosophical") for p in sample(PHILOSOPHICAL_PROMPTS, n_phil)])
    prompts.extend([(p, "nature") for p in sample(NATURE_PROMPTS, n_nature)])
    prompts.extend([(p, "bare") for p in sample(BARE_PROMPTS, n_bare)])

    corpus = load_corpus_prompts(num=n_corpus, seed=seed)
    prompts.extend([(p, "corpus") for p in corpus])

    rng.shuffle(prompts)
    return prompts


###############################################################################
# 2) BENCHMARK RUNNER
###############################################################################

def run_large_benchmark(
    model, tokenizer, prompts_with_categories, seeds_per_prompt=5,
    constrained=True, label="CONSTRAINED",
):
    """Generate poems for all prompts × seeds and collect results.

    Args:
        prompts_with_categories: list of (prompt_text, category) tuples
    """
    total_poems = len(prompts_with_categories) * seeds_per_prompt
    categories = Counter(cat for _, cat in prompts_with_categories)

    print(f"\n{'='*72}")
    print(f"  {label}")
    print(f"  Prompts: {len(prompts_with_categories)}, Seeds/prompt: {seeds_per_prompt}")
    print(f"  Total poems: {total_poems}")
    print(f"  Prompt categories: {dict(categories)}")
    print(f"{'='*72}\n")

    all_results = []
    total_valid = 0
    total_lines = 0
    valid_lines_count = 0
    per_category_valid = Counter()
    per_category_total = Counter()
    start_time = time.time()

    for pi, (topic, category) in enumerate(prompts_with_categories):
        prompt_valid = 0

        for si in range(seeds_per_prompt):
            seed = 42 + si * 7
            poem_idx = pi * seeds_per_prompt + si + 1

            result = generate_poem(
                model, tokenizer, topic,
                seed=seed, constrained=constrained,
                temperature=0.7, top_p=0.9, max_new_tokens=150,
            )
            result["category"] = category
            all_results.append(result)

            per_category_total[category] += 1
            if result["all_valid"]:
                total_valid += 1
                prompt_valid += 1
                per_category_valid[category] += 1

            for vl in result["valid_lines"]:
                total_lines += 1
                if vl["valid"]:
                    valid_lines_count += 1

            # Print every poem as it's generated
            status = "✓" if result["all_valid"] else "✗"
            bt = result.get("backtracks", 0)
            print(f"\n  {status} seed={seed} [{category}] ({result['elapsed']:.1f}s, {bt} bt) topic: {topic[:45]}")
            for vl in result["valid_lines"]:
                mark = "✓" if vl["valid"] else "✗"
                print(f"    {mark} {vl['line'][:70]}")
            if not result["all_valid"]:
                for vl in result["valid_lines"]:
                    if not vl["valid"]:
                        print(f"      markers: {vl['markers']}")

        # Progress
        elapsed = time.time() - start_time
        rate = (pi + 1) * seeds_per_prompt / elapsed if elapsed > 0 else 0
        pct = total_valid / ((pi + 1) * seeds_per_prompt) * 100

        print(
            f"\r  [{poem_idx:4d}/{total_poems}] "
            f"prompt {pi+1}/{len(prompts_with_categories)} "
            f"[{category:12s}] "
            f"| valid: {total_valid}/{(pi+1)*seeds_per_prompt} ({pct:.1f}%) "
            f"| {rate:.1f} poems/s",
            end="", flush=True,
        )

    elapsed = time.time() - start_time
    print(f"\n\n  Done in {elapsed:.1f}s ({total_poems/elapsed:.1f} poems/s)")

    return all_results, {
        "label": label,
        "constrained": constrained,
        "total_poems": total_poems,
        "total_valid": total_valid,
        "poem_accuracy": total_valid / total_poems * 100,
        "total_lines": total_lines,
        "valid_lines": valid_lines_count,
        "line_accuracy": valid_lines_count / total_lines * 100 if total_lines > 0 else 0,
        "elapsed": elapsed,
        "per_category": {
            cat: {
                "total": per_category_total[cat],
                "valid": per_category_valid[cat],
                "accuracy": per_category_valid[cat] / per_category_total[cat] * 100
                if per_category_total[cat] > 0 else 0,
            }
            for cat in per_category_total
        },
    }


###############################################################################
# 3) ANALYSIS & REPORTING
###############################################################################

def analyze_results(all_results, stats, prompts):
    """Produce detailed analysis of benchmark results."""

    print(f"\n{'='*72}")
    print(f"  EVALUATION REPORT: {stats['label']}")
    print(f"{'='*72}")

    # ── Overall accuracy ─────────────────────────────────────────────
    print(f"\n  OVERALL ACCURACY:")
    print(f"    Poem-level (both lines valid): {stats['total_valid']}/{stats['total_poems']} "
          f"({stats['poem_accuracy']:.1f}%)")
    print(f"    Line-level (individual lines): {stats['valid_lines']}/{stats['total_lines']} "
          f"({stats['line_accuracy']:.1f}%)")

    # ── Line breakdown ───────────────────────────────────────────────
    both_valid = 0
    l1_only = 0
    l2_only = 0
    neither = 0
    incomplete = 0

    for r in all_results:
        vl = r["valid_lines"]
        if len(vl) < 2:
            incomplete += 1
        else:
            v1, v2 = vl[0]["valid"], vl[1]["valid"]
            if v1 and v2: both_valid += 1
            elif v1: l1_only += 1
            elif v2: l2_only += 1
            else: neither += 1

    print(f"\n  LINE BREAKDOWN:")
    print(f"    Both lines valid:  {both_valid:4d}  ({both_valid/len(all_results)*100:.1f}%)")
    print(f"    Line 1 only valid: {l1_only:4d}  ({l1_only/len(all_results)*100:.1f}%)")
    print(f"    Line 2 only valid: {l2_only:4d}  ({l2_only/len(all_results)*100:.1f}%)")
    print(f"    Neither valid:     {neither:4d}  ({neither/len(all_results)*100:.1f}%)")
    if incomplete:
        print(f"    Incomplete (<2 lines): {incomplete}")

    # ── Per-prompt accuracy ──────────────────────────────────────────
    seeds_per = len(all_results) // len(prompts) if prompts else 1
    per_prompt = []
    for pi, topic in enumerate(prompts):
        chunk = all_results[pi * seeds_per : (pi + 1) * seeds_per]
        valid = sum(1 for r in chunk if r["all_valid"])
        per_prompt.append({"topic": topic, "valid": valid, "total": len(chunk),
                           "rate": valid / len(chunk) * 100})

    # Sort by accuracy
    per_prompt.sort(key=lambda x: x["rate"], reverse=True)

    print(f"\n  PER-PROMPT ACCURACY (sorted):")
    for ps in per_prompt[:15]:
        bar = "█" * int(ps["rate"] / 10) + "░" * (10 - int(ps["rate"] / 10))
        print(f"    {bar} {ps['rate']:5.1f}%  {ps['topic'][:50]}")
    if len(per_prompt) > 15:
        print(f"    ... ({len(per_prompt) - 15} more prompts)")

    # Accuracy distribution
    rates = [ps["rate"] for ps in per_prompt]
    rate_bins = Counter()
    for r in rates:
        if r == 100: rate_bins["100%"] += 1
        elif r >= 80: rate_bins["80-99%"] += 1
        elif r >= 60: rate_bins["60-79%"] += 1
        elif r >= 40: rate_bins["40-59%"] += 1
        elif r >= 20: rate_bins["20-39%"] += 1
        elif r > 0: rate_bins["1-19%"] += 1
        else: rate_bins["0%"] += 1

    print(f"\n  ACCURACY DISTRIBUTION ACROSS PROMPTS:")
    for bucket in ["100%", "80-99%", "60-79%", "40-59%", "20-39%", "1-19%", "0%"]:
        count = rate_bins.get(bucket, 0)
        bar = "█" * count
        print(f"    {bucket:>7s}: {bar} ({count})")

    # ── Per-category accuracy ────────────────────────────────────────
    if "per_category" in stats:
        print(f"\n  PER-CATEGORY ACCURACY:")
        for cat, cs in sorted(stats["per_category"].items(), key=lambda x: -x[1]["accuracy"]):
            bar = "█" * int(cs["accuracy"] / 10) + "░" * (10 - int(cs["accuracy"] / 10))
            print(f"    {bar} {cs['accuracy']:5.1f}%  {cat:15s} ({cs['valid']}/{cs['total']})")

    # ── Gana pattern distribution ────────────────────────────────────
    gana_patterns = Counter()
    for r in all_results:
        for vl in r["valid_lines"]:
            if vl["valid"]:
                gana_patterns[vl["partition"]] += 1

    print(f"\n  TOP 10 GANA PATTERNS (valid lines):")
    for pattern, count in gana_patterns.most_common(10):
        print(f"    {count:4d}×  {pattern}")

    # ── Timing stats ─────────────────────────────────────────────────
    times = [r["elapsed"] for r in all_results]
    print(f"\n  TIMING:")
    print(f"    Total:   {stats['elapsed']:.1f}s")
    print(f"    Average: {sum(times)/len(times):.2f}s per poem")
    print(f"    Min:     {min(times):.2f}s")
    print(f"    Max:     {max(times):.2f}s")
    print(f"    Median:  {sorted(times)[len(times)//2]:.2f}s")

    if stats["constrained"]:
        bts = [r.get("backtracks", 0) for r in all_results]
        chks = [r.get("nfa_checks", 0) for r in all_results]
        print(f"    Avg backtracks:  {sum(bts)/len(bts):.1f}")
        print(f"    Avg NFA checks:  {sum(chks)/len(chks):.0f}")

    # ── Novelty check ────────────────────────────────────────────────
    print(f"\n  NOVELTY CHECK:")
    dataset_path = os.path.join(PROJECT_DIR, "datasets", "dwipada_augmented_dataset.json")
    with open(dataset_path) as f:
        corpus = json.load(f)
    corpus_lines = set()
    for item in corpus:
        for line in item["poem"].strip().split("\n"):
            corpus_lines.add(line.strip())

    generated_lines = []
    for r in all_results:
        for vl in r["valid_lines"]:
            generated_lines.append(vl["line"])

    found_in_corpus = sum(1 for l in generated_lines if l in corpus_lines)
    print(f"    Generated lines:       {len(generated_lines)}")
    print(f"    Found in corpus:       {found_in_corpus}")
    print(f"    Novel lines:           {len(generated_lines) - found_in_corpus} "
          f"({(len(generated_lines) - found_in_corpus)/len(generated_lines)*100:.1f}%)")

    return per_prompt


def save_results(all_results, stats, per_prompt, output_path):
    """Save full results to JSON."""
    data = {
        **stats,
        "per_prompt": per_prompt,
        "poems": [
            {
                "topic": r["topic"],
                "category": r.get("category", "unknown"),
                "seed": r["seed"],
                "all_valid": r["all_valid"],
                "elapsed": r["elapsed"],
                "backtracks": r.get("backtracks", 0),
                "nfa_checks": r.get("nfa_checks", 0),
                "tokens_generated": r["tokens_generated"],
                "lines": [
                    {
                        "text": vl["line"],
                        "valid": vl["valid"],
                        "markers": vl["markers"],
                        "partition": vl["partition"],
                    }
                    for vl in r["valid_lines"]
                ],
            }
            for r in all_results
        ],
    }
    with open(output_path, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\n  Saved: {output_path} ({size_mb:.1f} MB)")


###############################################################################
# 4) MAIN
###############################################################################

###############################################################################
# MODEL LOADING
###############################################################################

# Available model choices:
#   gemma3-1b-merged  — Gemma 3 1B IT, merged (text-only, in dwipada_merged_model/)
#   gemma3-1b-lora    — Gemma 3 1B IT + LoRA adapter (dwipada_lora_adapter/)
#   gemma3-4b-lora    — Gemma 3 4B IT + LoRA adapter (best-checkpoint-2450)
#   gemma3-1b-base    — Gemma 3 1B IT, no fine-tuning
#   gemma3-4b-base    — Gemma 3 4B IT, no fine-tuning
#   gemma4-e4b-base   — Gemma 4 E2B IT, no fine-tuning (4-bit quantized)

# Models that were fine-tuned on dwipada data use a simple system prompt.
# Base models need a detailed system prompt with explicit gana rules.
FINETUNED_MODELS = {"gemma3-1b-merged", "gemma3-1b-lora", "gemma3-4b-lora"}

SYSTEM_PROMPT_FINETUNED = (
    "You are a Telugu and Sanskrit scholar specialising in Dwipada poetry."
)

SYSTEM_PROMPT_BASE = (
    "You are a Telugu and Sanskrit scholar specialising in Dwipada poetry. "
    "A Dwipada has exactly 2 lines. Each line has 3 Indra ganas + 1 Surya gana. "
    "Indra ganas: Nala(IIII), Naga(IIIU), Sala(IIUI), Bha(UII), Ra(UIU), Ta(UUI). "
    "Surya ganas: Na(III), Ha/Gala(UI). "
    "Output ONLY the 2-line poem, nothing else."
)


def is_finetuned_model(model_choice):
    """Check if a model choice is a fine-tuned dwipada model.

    Fine-tuned models output a 'ద్విపద:' prefix before the poem.
    Base models output poem text directly — no prefix to strip.
    """
    return model_choice in FINETUNED_MODELS


def build_prompt_for_model(topic, tokenizer, model_choice):
    """Build the generation prompt, adapting the system prompt to the model type.

    Fine-tuned models (merged/lora) use a simple system prompt matching
    their training format. Base models get a detailed prompt with explicit
    gana rules to guide generation.
    """
    is_finetuned = model_choice in FINETUNED_MODELS
    system_prompt = SYSTEM_PROMPT_FINETUNED if is_finetuned else SYSTEM_PROMPT_BASE

    if is_finetuned:
        user_prompt = (
            "క్రింది తెలుగు భావానికి అనుగుణంగా ఒక ద్విపద పద్యం రచించండి. "
            "ప్రతి పాదంలో 3 ఇంద్ర గణాలు + 1 సూర్య గణం ఉండాలి.\n\n"
            f"తెలుగు భావం: {topic}"
        )
    else:
        user_prompt = (
            "క్రింది తెలుగు భావానికి అనుగుణంగా ఒక ద్విపద పద్యం రచించండి.\n\n"
            f"భావం: {topic}\n\n"
            "ద్విపద:"
        )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

MODEL_CHOICES = [
    "gemma3-1b-merged",
    "gemma3-1b-lora",
    "gemma3-4b-lora",
    "gemma3-1b-base",
    "gemma3-4b-base",
    "gemma4-e4b-base",
]


def _get_bnb_config():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def _extract_gemma4_text_weights(base_id="google/gemma-4-E2B-it"):
    """Extract and cache text-only weights from the multimodal Gemma 4 E2B IT.

    Same approach as Gemma 3 4B: load full model on CPU, extract language_model
    weights, strip prefix, save to a lean checkpoint directory.
    """
    import gc
    from safetensors.torch import save_file as safetensors_save

    # Cache dir named after model
    safe_name = base_id.replace("/", "_").replace("-", "_")
    text_model_dir = os.path.join(PROJECT_DIR, "train_models", f"_text_only_{safe_name}")
    if os.path.exists(os.path.join(text_model_dir, "model.safetensors")):
        print(f"  Using cached text-only weights from {text_model_dir}")
        return text_model_dir

    print(f"  Extracting text-only weights from {base_id} (one-time)...")
    from transformers import Gemma4ForConditionalGeneration
    full_model = Gemma4ForConditionalGeneration.from_pretrained(
        base_id,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )
    text_state = {}
    for k, v in full_model.state_dict().items():
        if k.startswith("model.language_model."):
            text_state[k.replace("model.language_model.", "model.")] = v
        elif k.startswith("lm_head."):
            text_state[k] = v

    text_config = full_model.config.text_config
    del full_model
    gc.collect()

    os.makedirs(text_model_dir, exist_ok=True)
    text_state.pop("lm_head.weight", None)  # tied with embed_tokens
    safetensors_save(text_state, os.path.join(text_model_dir, "model.safetensors"))
    text_config.tie_word_embeddings = True
    text_config.save_pretrained(text_model_dir)
    del text_state
    gc.collect()
    print(f"  Text-only weights saved to {text_model_dir}")
    return text_model_dir


def _extract_gemma3_4b_text_weights():
    """Extract and cache text-only weights from the multimodal Gemma 3 4B IT.

    The 4B model bundles a vision tower; we strip it out and save the text-only
    weights so subsequent loads go straight to the lean checkpoint.
    """
    import gc
    from safetensors.torch import save_file as safetensors_save

    text_model_dir = os.path.join(PROJECT_DIR, "train_models", "_text_only_gemma3_4b")
    if os.path.exists(os.path.join(text_model_dir, "model.safetensors")):
        print(f"  Using cached text-only weights from {text_model_dir}")
        return text_model_dir

    print("  Extracting text-only weights from gemma-3-4b-it (one-time)...")
    full_model = Gemma3ForConditionalGeneration.from_pretrained(
        "google/gemma-3-4b-it",
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )
    text_state = {}
    for k, v in full_model.state_dict().items():
        if k.startswith("model.language_model."):
            text_state[k.replace("model.language_model.", "model.")] = v
        elif k.startswith("lm_head."):
            text_state[k] = v

    text_config = full_model.config.text_config
    del full_model
    gc.collect()

    os.makedirs(text_model_dir, exist_ok=True)
    text_state.pop("lm_head.weight", None)  # tied with embed_tokens
    safetensors_save(text_state, os.path.join(text_model_dir, "model.safetensors"))
    text_config.tie_word_embeddings = True
    text_config.save_pretrained(text_model_dir)
    del text_state
    gc.collect()
    print(f"  Text-only weights saved to {text_model_dir}")
    return text_model_dir


def load_model(choice):
    """Load a model+tokenizer pair based on the --model CLI choice."""
    import gc

    print(f"\n  Loading model: {choice}")

    if choice == "gemma3-1b-merged":
        path = os.path.join(PROJECT_DIR, "train_models", "dwipada_merged_model")
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="sdpa",
        )

    elif choice == "gemma3-1b-lora":
        base_id = "google/gemma-3-1b-it"
        adapter_path = os.path.join(PROJECT_DIR, "train_models", "dwipada_lora_adapter")
        tokenizer = AutoTokenizer.from_pretrained(base_id)
        model = AutoModelForCausalLM.from_pretrained(
            base_id, quantization_config=_get_bnb_config(), device_map="auto", attn_implementation="sdpa",
        )
        model = PeftModel.from_pretrained(model, adapter_path)
        print(f"  LoRA adapter: {adapter_path}")

    elif choice == "gemma3-4b-lora":
        adapter_path = os.path.join(PROJECT_DIR, "train_models", "checkpoints_gemma4b", "best-checkpoint-2450")
        text_model_dir = _extract_gemma3_4b_text_weights()
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-4b-it")
        model = Gemma3ForCausalLM.from_pretrained(
            text_model_dir, quantization_config=_get_bnb_config(), device_map="auto", attn_implementation="sdpa",
        )
        gc.collect()
        torch.cuda.empty_cache()
        model = PeftModel.from_pretrained(model, adapter_path)
        print(f"  LoRA adapter: {adapter_path}")

    elif choice == "gemma3-1b-base":
        base_id = "google/gemma-3-1b-it"
        tokenizer = AutoTokenizer.from_pretrained(base_id)
        model = AutoModelForCausalLM.from_pretrained(
            base_id, quantization_config=_get_bnb_config(), device_map="auto", attn_implementation="sdpa",
        )

    elif choice == "gemma3-4b-base":
        text_model_dir = _extract_gemma3_4b_text_weights()
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-4b-it")
        model = Gemma3ForCausalLM.from_pretrained(
            text_model_dir, quantization_config=_get_bnb_config(), device_map="auto", attn_implementation="sdpa",
        )
        gc.collect()
        torch.cuda.empty_cache()

    elif choice == "gemma4-e4b-base":
        base_id = "google/gemma-4-E2B-it"
        tokenizer = AutoTokenizer.from_pretrained(base_id)
        # Monkey-patch to avoid OOM during caching allocator warmup
        import transformers.modeling_utils as _mu
        _mu.caching_allocator_warmup = lambda *args, **kwargs: None
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            llm_int8_enable_fp32_cpu_offload=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            base_id,
            quantization_config=bnb_config,
            device_map="auto",
            max_memory={0: "7GiB", "cpu": "16GiB"},
            low_cpu_mem_usage=True,
        )

    else:
        raise ValueError(f"Unknown model choice: {choice}")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    print(f"  Model ready: {choice}")
    return model, tokenizer


def main():
    import constrained_generate
    p = argparse.ArgumentParser(description="Large-scale Dwipada Benchmark")
    p.add_argument("--model", type=str, default="gemma3-4b-lora", choices=MODEL_CHOICES,
                   help="Model to use (default: gemma3-4b-lora)")
    p.add_argument("--num-poems", type=int, default=1000,
                   help="Total poems to generate (default 1000)")
    p.add_argument("--seeds-per-prompt", type=int, default=5,
                   help="Seeds per prompt (default 5)")
    p.add_argument("--with-baseline", action="store_true",
                   help="Also run unconstrained baseline")
    p.add_argument("--baseline-only", action="store_true",
                   help="Run ONLY unconstrained baseline (no constrained)")
    p.add_argument("--top-k", type=int, default=50)
    args = p.parse_args()

    constrained_generate.TOP_K = args.top_k

    num_prompts = args.num_poems // args.seeds_per_prompt
    if num_prompts < 1:
        num_prompts = 1

    print("=" * 72)
    print("Large-scale Constrained Dwipada Generation Benchmark")
    print(f"  Model:             {args.model}")
    print(f"  Target poems:      {num_prompts * args.seeds_per_prompt}")
    print(f"  Prompts:           {num_prompts}")
    print(f"  Seeds per prompt:  {args.seeds_per_prompt}")
    print(f"  Top-K:             {args.top_k}")
    print("=" * 72)

    # Build diverse prompts
    print("\n  Building diverse prompt set...")
    prompts_with_cats = build_diverse_prompts(num_prompts=num_prompts, seed=42)
    categories = Counter(cat for _, cat in prompts_with_cats)
    print(f"  Selected {len(prompts_with_cats)} prompts:")
    for cat, count in sorted(categories.items()):
        print(f"    {cat:15s}: {count}")

    model, tokenizer = load_model(args.model)

    prompts_texts = [p for p, _ in prompts_with_cats]

    stats_c = None
    stats_u = None

    # ── Run constrained benchmark ────────────────────────────────────
    if not args.baseline_only:
        results_c, stats_c = run_large_benchmark(
            model, tokenizer, prompts_with_cats,
            seeds_per_prompt=args.seeds_per_prompt,
            constrained=True, label="FINE-TUNED — CONSTRAINED",
        )
        per_prompt_c = analyze_results(results_c, stats_c, prompts_texts)
        save_results(results_c, stats_c, per_prompt_c,
                     os.path.join(SCRIPT_DIR, "benchmark_constrained_1000.json"))

    # ── Run unconstrained baseline ───────────────────────────────────
    if args.with_baseline or args.baseline_only:
        results_u, stats_u = run_large_benchmark(
            model, tokenizer, prompts_with_cats,
            seeds_per_prompt=args.seeds_per_prompt,
            constrained=False, label="FINE-TUNED — UNCONSTRAINED",
        )
        per_prompt_u = analyze_results(results_u, stats_u, prompts_texts)
        save_results(results_u, stats_u, per_prompt_u,
                     os.path.join(SCRIPT_DIR, "benchmark_unconstrained_1000.json"))

    # ── Side-by-side comparison ──────────────────────────────────────
    if stats_c and stats_u:
        print(f"\n{'='*72}")
        print(f"  COMPARISON: CONSTRAINED vs UNCONSTRAINED")
        print(f"{'='*72}")
        print(f"                        Constrained    Unconstrained")
        print(f"  Poem accuracy:        {stats_c['poem_accuracy']:6.1f}%        {stats_u['poem_accuracy']:6.1f}%")
        print(f"  Line accuracy:        {stats_c['line_accuracy']:6.1f}%        {stats_u['line_accuracy']:6.1f}%")
        print(f"  Avg time/poem:        {stats_c['elapsed']/stats_c['total_poems']:6.2f}s        {stats_u['elapsed']/stats_u['total_poems']:6.2f}s")
        improvement = stats_c["poem_accuracy"] - stats_u["poem_accuracy"]
        print(f"  Improvement:          +{improvement:.1f} percentage points")

    print("\n" + "=" * 72)
    print("Done.")


if __name__ == "__main__":
    main()
