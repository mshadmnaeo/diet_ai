"""
diet_classifier.py
------------------
Uses google/flan-t5-xl from HuggingFace for nutrition analysis.
No API key needed. Fully local. 100% open-source.

WHY FLAN-T5-XL over FLAN-T5-LARGE?
  - Large : 770MB  ~1GB RAM   — good accuracy
  - XL    : 3GB    ~4GB RAM   — significantly better accuracy
  - XL has 3 billion parameters vs Large's 770 million
  - Much better at following complex JSON instructions
  - More accurate nutrition estimates
  - Still fast enough on CPU (30-60 seconds per query)

SYSTEM REQUIREMENTS for XL:
  - RAM         : 6GB+ recommended (4GB minimum)
  - Disk space  : 6GB free for model cache
  - CPU         : Any modern CPU works (GPU makes it faster)
  - Streamlit   : Paid tier or local machine only (free tier = ~1GB RAM)

IF YOU WANT TO GO BACK TO FREE TIER:
  Change MODEL_NAME to "google/flan-t5-large" (770MB)
"""

import json
import re
import streamlit as st
from transformers import pipeline
import torch

# ── Fallback data if model fails ──────────────────────────────────────────────
FALLBACK_NUTRITION = {
    "diet_type":        "Unknown",
    "calories_per_100g": 200,
    "protein_g":         10.0,
    "carbs_g":           25.0,
    "fat_g":              8.0,
    "fiber_g":            3.0,
    "sugar_g":            5.0,
    "sodium_mg":         300,
    "is_vegan":          False,
    "is_vegetarian":     False,
    "is_gluten_free":    False,
    "is_keto_friendly":  False,
    "health_score":       5,
    "health_tips":       "Could not analyze. Please try again.",
    "allergens":          [],
}

DIET_TYPES = [
    "Keto", "Vegan", "Vegetarian", "Mediterranean",
    "Paleo", "High-Protein", "Low-Carb", "Balanced", "Junk Food",
]

# ── Model selection ───────────────────────────────────────────────────────────
# Change this one line to swap models:
#   "google/flan-t5-large"   770MB  ~1GB RAM   Streamlit Cloud free tier
#   "google/flan-t5-xl"      3GB    ~4GB RAM   Local / paid tier  ← THIS FILE
#   "google/flan-t5-xxl"     11GB   ~14GB RAM  Local with GPU only
MODEL_NAME = "google/flan-t5-xl"


@st.cache_resource(show_spinner="Loading Flan-T5-XL (~3GB, first run only)...")
def load_nutrition_model():
    """
    Load Flan-T5-XL from HuggingFace.
    Downloads ~3GB on first run, then uses local cache forever.
    @st.cache_resource ensures it loads exactly once per session.

    torch_dtype=torch.float32 is used because float16 can cause
    issues on some CPUs — XL is accurate enough without it.
    """
    print(f"Loading model: {MODEL_NAME}")

    pipe = pipeline(
        task="text2text-generation",
        model=MODEL_NAME,
        tokenizer=MODEL_NAME,
        max_new_tokens=400,      # XL can handle longer outputs
        device="cpu",            # change to device=0 if you have a GPU
        torch_dtype=torch.float32,
    )
    return pipe


def classify_diet(food_name: str) -> dict:
    """
    Ask Flan-T5-XL to analyze the food and return structured nutrition data.

    Strategy: three focused prompts instead of one big prompt.
    XL is smart enough to handle more detail in each prompt.
      1. Macros (calories, protein, carbs, fat)
      2. Micros (fiber, sugar, sodium)
      3. Qualitative (diet type, vegan, health score, tips, allergens)

    Results are merged into one complete nutrition dict.
    """
    pipe = load_nutrition_model()

    # ── Prompt 1: Macronutrients ──────────────────────────────────────────────
    macro_prompt = f"""As a certified nutritionist, provide the macronutrient content of "{food_name}" per 100 grams.
Be precise and realistic. Answer in JSON only, no explanation:
{{"calories_per_100g": <integer>, "protein_g": <decimal>, "carbs_g": <decimal>, "fat_g": <decimal>}}"""

    # ── Prompt 2: Micronutrients ──────────────────────────────────────────────
    micro_prompt = f"""As a certified nutritionist, provide the micronutrient content of "{food_name}" per 100 grams.
Be precise and realistic. Answer in JSON only, no explanation:
{{"fiber_g": <decimal>, "sugar_g": <decimal>, "sodium_mg": <integer>}}"""

    # ── Prompt 3: Qualitative analysis ───────────────────────────────────────
    qual_prompt = f"""As a certified nutritionist, classify the food "{food_name}".
Answer in JSON only, no explanation:
{{"diet_type": "<one of: {', '.join(DIET_TYPES)}>", "is_vegan": <true/false>, "is_vegetarian": <true/false>, "is_gluten_free": <true/false>, "is_keto_friendly": <true/false>, "health_score": <integer 1-10 where 10 is healthiest>, "health_tips": "<one practical tip under 20 words>", "allergens": ["<common allergens as list>"]}}"""

    try:
        macro_raw = pipe(macro_prompt)[0]["generated_text"].strip()
        micro_raw = pipe(micro_prompt)[0]["generated_text"].strip()
        qual_raw  = pipe(qual_prompt)[0]["generated_text"].strip()

        macro_data = _parse_json(macro_raw)
        micro_data = _parse_json(micro_raw)
        qual_data  = _parse_json(qual_raw)

        # Merge all three into one complete dict
        merged = FALLBACK_NUTRITION.copy()
        for source in [macro_data, micro_data, qual_data]:
            if source:
                merged.update(source)

        return _validate(merged)

    except Exception as e:
        print(f"Flan-T5-XL error for '{food_name}': {e}")
        fallback = FALLBACK_NUTRITION.copy()
        fallback["health_tips"] = f"Model error: {str(e)[:100]}"
        return fallback


def _parse_json(text: str) -> dict:
    """
    Safely extract JSON from model output.
    Handles extra text, markdown fences, and partial responses.
    """
    if not text:
        return {}

    # Strip markdown code fences
    text = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Find {...} block anywhere in the text
    match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Last resort: regex extraction of individual numeric values
    result = {}
    patterns = {
        "calories_per_100g": r'"?calories_per_100g"?\s*:\s*(\d+\.?\d*)',
        "protein_g":         r'"?protein_g"?\s*:\s*(\d+\.?\d*)',
        "carbs_g":           r'"?carbs_g"?\s*:\s*(\d+\.?\d*)',
        "fat_g":             r'"?fat_g"?\s*:\s*(\d+\.?\d*)',
        "fiber_g":           r'"?fiber_g"?\s*:\s*(\d+\.?\d*)',
        "sugar_g":           r'"?sugar_g"?\s*:\s*(\d+\.?\d*)',
        "sodium_mg":         r'"?sodium_mg"?\s*:\s*(\d+)',
        "health_score":      r'"?health_score"?\s*:\s*(\d+)',
    }
    for key, pattern in patterns.items():
        m = re.search(pattern, text)
        if m:
            result[key] = float(m.group(1))

    return result


def _validate(data: dict) -> dict:
    """Ensure all fields exist with correct types. Fill gaps from fallback."""
    for key, default in FALLBACK_NUTRITION.items():
        if key not in data or data[key] is None:
            data[key] = default

    # Clamp health_score 1-10
    data["health_score"] = max(1, min(10, int(data.get("health_score", 5))))

    # Ensure numeric fields
    for field in ["calories_per_100g", "protein_g", "carbs_g",
                  "fat_g", "fiber_g", "sugar_g", "sodium_mg"]:
        try:
            data[field] = float(data[field])
        except (TypeError, ValueError):
            data[field] = FALLBACK_NUTRITION[field]

    # Ensure booleans
    for field in ["is_vegan", "is_vegetarian", "is_gluten_free", "is_keto_friendly"]:
        val = data.get(field, False)
        if isinstance(val, str):
            data[field] = val.lower() in ("true", "yes", "1")
        else:
            data[field] = bool(val)

    # Ensure allergens is a list
    if not isinstance(data.get("allergens"), list):
        data["allergens"] = []

    return data


def get_diet_summary(nutrition: dict) -> str:
    return (
        f"{nutrition.get('diet_type', 'Unknown')} diet  ·  "
        f"{nutrition.get('calories_per_100g', '?'):.0f} kcal/100g  ·  "
        f"Health score: {nutrition.get('health_score', '?')}/10"
    )
