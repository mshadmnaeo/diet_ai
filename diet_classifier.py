"""
diet_classifier.py
------------------
Uses a lightweight text model from HuggingFace for nutrition analysis.
No API key needed. Fully local. 100% open-source.

This project uses a small, broadly compatible model so it works with
newer transformers versions where `text2text-generation` is unavailable.
"""

import json
import os
import re
import streamlit as st
import torch

# Use project-local cache by default so model downloads do not fail on ~/.cache perms.
project_hf_cache = os.path.join(os.getcwd(), ".hf_cache")
os.makedirs(project_hf_cache, exist_ok=True)
os.environ.setdefault("HF_HOME", project_hf_cache)
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.path.join(project_hf_cache, "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(project_hf_cache, "transformers"))
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

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
# Flan-T5-Large gives better instruction following than Base while keeping
# compatibility with direct seq2seq generation.
MODEL_NAME = "google/flan-t5-large"


@st.cache_resource(show_spinner="Loading nutrition model (first run only)...")
def load_nutrition_model():
    """
    Load a seq2seq instruction model from HuggingFace.
    Downloads once, then uses local cache.
    @st.cache_resource ensures it loads exactly once per session.
    """
    print(f"Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
    model.eval()
    return model, tokenizer


def classify_diet(food_name: str) -> dict:
    """
    Ask the model to analyze the food and return structured nutrition data.

    Strategy: three focused prompts instead of one big prompt.
      1. Macros (calories, protein, carbs, fat)
      2. Micros (fiber, sugar, sodium)
      3. Qualitative (diet type, vegan, health score, tips, allergens)

    Results are merged into one complete nutrition dict.
    """
    model, tokenizer = load_nutrition_model()

    # ── Prompt 1: Macronutrients ──────────────────────────────────────────────
    macro_prompt = f"""Return ONLY valid minified JSON (no prose, no markdown) for the food "{food_name}" per 100g.
Use realistic nutrition values.
Never output all zeros. If uncertain, return reasonable estimates for a typical preparation.
Schema:
{{"calories_per_100g":0,"protein_g":0.0,"carbs_g":0.0,"fat_g":0.0}}
JSON:"""

    # ── Prompt 2: Micronutrients ──────────────────────────────────────────────
    micro_prompt = f"""Return ONLY valid minified JSON (no prose, no markdown) for the food "{food_name}" per 100g.
Use realistic nutrition values.
Never output all zeros. If uncertain, return reasonable estimates for a typical preparation.
Schema:
{{"fiber_g":0.0,"sugar_g":0.0,"sodium_mg":0}}
JSON:"""

    # ── Prompt 3: Qualitative analysis ───────────────────────────────────────
    qual_prompt = f"""Return ONLY valid minified JSON (no prose, no markdown) classifying "{food_name}".
diet_type must be one of: {', '.join(DIET_TYPES)}.
Schema:
{{"diet_type":"Balanced","is_vegan":false,"is_vegetarian":false,"is_gluten_free":false,"is_keto_friendly":false,"health_score":5,"health_tips":"", "allergens":[]}}
Rules:
- health_score is integer 1-10.
- allergens is an array of short strings.
JSON:"""

    try:
        macro_raw = _generate_json_text(macro_prompt, model, tokenizer)
        micro_raw = _generate_json_text(micro_prompt, model, tokenizer)
        qual_raw  = _generate_json_text(qual_prompt, model, tokenizer)

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
        print(f"Nutrition model error for '{food_name}': {e}")
        fallback = FALLBACK_NUTRITION.copy()
        fallback["health_tips"] = f"Model error: {str(e)[:100]}"
        return fallback


def _generate_json_text(prompt: str, model, tokenizer) -> str:
    """
    Run deterministic generation for structured JSON-like responses.
    """
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=220,
            do_sample=False,
            num_beams=1,
            temperature=None,
            top_p=None,
        )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()


def _parse_json(text: str) -> dict:
    """
    Safely extract JSON from model output.
    Handles extra text, markdown fences, and partial responses.
    """
    if not text:
        return {}

    # Strip markdown code fences
    text = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()
    # Remove leading label-style prefixes like "JSON:" or "Answer:"
    text = re.sub(r"^(json|answer)\s*:\s*", "", text, flags=re.IGNORECASE).strip()

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Find {...} block anywhere in the text
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Try to sanitize common JSON issues (single quotes, trailing commas, Python booleans)
    sanitized = text
    sanitized = sanitized.replace("'", '"')
    sanitized = re.sub(r"\bTrue\b", "true", sanitized)
    sanitized = re.sub(r"\bFalse\b", "false", sanitized)
    sanitized = re.sub(r",\s*([}\]])", r"\1", sanitized)
    match = re.search(r"\{.*\}", sanitized, re.DOTALL)
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

    # Parse likely string fields when present in raw text.
    m = re.search(r'"?diet_type"?\s*:\s*"([^"]+)"', text, re.IGNORECASE)
    if m:
        result["diet_type"] = m.group(1).strip()

    m = re.search(r'"?health_tips"?\s*:\s*"([^"]+)"', text, re.IGNORECASE)
    if m:
        result["health_tips"] = m.group(1).strip()

    for field in ["is_vegan", "is_vegetarian", "is_gluten_free", "is_keto_friendly"]:
        m = re.search(rf'"?{field}"?\s*:\s*(true|false|yes|no|1|0)', text, re.IGNORECASE)
        if m:
            result[field] = m.group(1).lower() in ("true", "yes", "1")

    # Prose-style fallback parsing (common with instruction models).
    prose_patterns = {
        "calories_per_100g": r"(\d+\.?\d*)\s*(?:kcal|calories?)",
        "protein_g": r"(\d+\.?\d*)\s*(?:g|grams?)\s*(?:of\s*)?protein",
        "carbs_g": r"(\d+\.?\d*)\s*(?:g|grams?)\s*(?:of\s*)?(?:carbs?|carbohydrates?)",
        "fat_g": r"(\d+\.?\d*)\s*(?:g|grams?)\s*(?:of\s*)?fat",
        "fiber_g": r"(\d+\.?\d*)\s*(?:g|grams?)\s*(?:of\s*)?fiber",
        "sugar_g": r"(\d+\.?\d*)\s*(?:g|grams?)\s*(?:of\s*)?sugar",
        "sodium_mg": r"(\d+\.?\d*)\s*mg\s*(?:of\s*)?sodium",
    }
    for key, pattern in prose_patterns.items():
        if key not in result:
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                result[key] = float(m.group(1))

    # Standalone diet type answer (e.g., "Balanced")
    if "diet_type" not in result:
        for d in DIET_TYPES:
            if re.search(rf"\b{re.escape(d)}\b", text, re.IGNORECASE):
                result["diet_type"] = d
                break

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

    # Guard against degenerate model outputs (e.g., all zeros).
    for field in ["calories_per_100g", "protein_g", "carbs_g", "fat_g", "sodium_mg"]:
        if data[field] <= 0:
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
