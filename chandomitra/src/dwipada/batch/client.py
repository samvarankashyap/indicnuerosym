"""Gemini client for interactive single-prompt generation.

Usage:
    python -m dwipada.batch.client
"""

import yaml
from google import genai
from datetime import datetime
from pathlib import Path

from dwipada.paths import CONFIG_FILE, OUTPUT_DIR

# === CONFIGURE THESE ===
MODEL = "gemini-3-pro-preview"
OUTPUT_FILE = OUTPUT_DIR / "gemini_responses.txt"
PROMPT = """
The Updated Prompt
Role: You are an expert Telugu and Sanskrit Scholar and Maha Kavi (Great Poet) with deep knowledge of Telugu Chandassu (Prosody), specifically the Dwipada meter.

Task: Compose a Dwipada (ద్విపద) poem in Telugu on the topic: [INSERT TOPIC HERE].
Strict Rules for Construction:
Structure: The poem must have exactly 2 lines (Padas).
Gana Sequence: Each line must strictly follow this sequence:
Indra Gana - Indra Gana - Indra Gana - Surya Gana (Total 4 Ganas per line).

Prasa (Rhyme): The second letter (hallu) of the first line must match the second letter of the second line (Prasa Niyamam).

Yati (Caesura):

Rule: ఈ యతి స్థానంలో ఉండే అక్షరం పాదం మొదటి అక్షరంతో "యతి మైత్రి"లో ఉండాలనేది నియమం.
Explanation: The first letter of the 3rd Gana (Yati Sthanam) must match the first letter of the 1st Gana (Pradhama Aksharam) according to Yati Maitri rules.

Reference Data (Use strict adherence):
Guru (U) / Laghu (I) Rules:
Short vowels = I (Laghu)

Long vowels / Dheergham = U (Guru)

Letter before a Samyukta/Dvitva aksharam (conjunct consonant) = U
Sunna (Anuswara) / Visarga included = U
Pollu Hallu (ending with 'n', 'l') = U

Allowed Ganas:

Surya Ganas:

Na (III)
Ha / Gala (UI)

Indra Ganas:

Nala (IIII)
Naga (IIIU)
Sala (IIUI)
Bha (UII)
Ra (UIU)
Ta (UUI)

Yati Maitri Groups (Vargas - for matching 1st Gana start & 3rd Gana start):

(అ, ఆ, ఐ, ఔ, హ, య, అం, అః)
(ఇ, ఈ, ఎ, ఏ, ఋ)
(ఉ, ఊ, ఒ, ఓ)
(క, ఖ, గ, ఘ, క్ష)
(చ, ఛ, జ, ఝ, శ, ష, స)
(ట, ఠ, డ, ఢ)
(త, థ, ద, ధ)
(ప, ఫ, బ, భ, వ)
(ర, ల, ఱ, ళ)
(న, ణ)
(మ, పు, ఫు, బు, భు, ము...)

Output Format:
The Poem: Present the 2-line Telugu poem clearly.
The language should be very understandable and simple.
Chandassu Analysis (Verification):
Break down each line into Ganas.
Mark the Guru (U) and Laghu (I) below the text.
Identify the Gana names (e.g., Bha, Ra, Nala, Ha).
Explicitly point out the Yati match (Prove that the 1st letter of Gana 1 & 1st letter of Gana 3 are in the same Varga).
Explicitly point out the Prasa match (2nd letter of Line 1 & Line 2).

"""
# =======================


def load_api_key():
    with open(CONFIG_FILE) as f:
        config = yaml.safe_load(f)
    return config["api_key"]


def generate_response(model_name: str, prompt: str) -> str:
    client = genai.Client(api_key=load_api_key())
    response = client.models.generate_content(
        model=model_name,
        contents=prompt
    )
    return response.text


def append_to_log(prompt: str, response: str, model_name: str):
    """Append prompt and response to log file in structured format."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    entry = f"""
{'='*80}
TIMESTAMP: {timestamp}
MODEL: {model_name}
{'='*80}

--- PROMPT ---
{prompt.strip()}

--- RESPONSE ---
{response.strip()}

{'='*80}
"""

    with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
        f.write(entry)

    print(f"Response appended to {OUTPUT_FILE}")


if __name__ == "__main__":
    result = generate_response(MODEL, PROMPT)
    print(result)
    append_to_log(PROMPT, result, MODEL)
