PROMPT_TEMPLATE = """You are an expert Kannada Scholar and Maha Kavi (Great Poet). You possess deep knowledge of **Kannada Chandassu (Prosody)**. You understand that poetic meter (Ganas) flows through the sound, and **Gana boundaries do not need to align with word boundaries**.


Generate 10 2-line Utsaha Ragale poems in Kannada on topic: <TOPIC> following
these strict rules:

*Metrical Rules:*
- Each line must have exactly 12 syllables (aksharas)
- Each line is divided into 4 ganas (metrical feet)
- Allowed gana patterns:
  - III (short-short-short, 3 syllables, 3 matras) — Primary, high frequency, keeps tempo fast
  - IIU (short-short-long, 3 syllables, 4 matras) — Standard, commonly used
  - UI (long-short, 2 syllables, 3 matras) — Standard, used frequently as variation
- Forbidden gana pattern:
  - IU (short-long) — Breaks the rhythmic flow of the Ragale
- Any mix of the allowed ganas is acceptable, but the total syllable count must be exactly 12
- Both lines must end on a Guru (long/heavy syllable)
- Ādi Prāsa: the 2nd syllable's consonant must match in both lines

*Syllable Weight Reference:*
- Laghu (I/short): short vowel (ಅ ಇ ಉ ಎ ಒ) without anusvara/visarga
- Guru (U/long): long vowel (ಆ ಈ ಊ ಏ ಓ ಔ ಐ), or has anusvara (ಂ), visarga (ಃ), or followed by conjunct consonant

*Output Format (strictly follow this JSON):*
{
  "poem_kannada": "<line 1 in Kannada>\\n<line 2 in Kannada>",
  "syllable_map": "<gana1> <gana2> | <gana3> <gana4>\\n<gana1> <gana2> | <gana3> <gana4>",
  "meaning_kannada": "<meaning of the couplet in Kannada>",
  "meaning_english": "<meaning of the couplet in English>",
  "theme": "<TOPIC>",
  "prasa_syllable": "<the matching Ādi Prāsa consonant>"
}"""

with open("topics_list.txt", "r") as f:
    topics = [line.strip() for line in f if line.strip()]

SEP = "=" * 30

with open("prompts_list.txt", "w") as out:
    for i, topic in enumerate(topics, 1):
        prompt = PROMPT_TEMPLATE.replace("<TOPIC>", topic)
        out.write(f"{SEP}\n{i}\n{SEP}\n\n{prompt}\n\n")

print(f"Generated {len(topics)} prompts in prompts_list.txt")
