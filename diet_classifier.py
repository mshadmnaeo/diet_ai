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

        # Dedicated pass for core metrics that often fail JSON parsing.
        core_prompt = f"""For "{food_name}", return exactly one line:
calories_per_100g=<number>; health_score=<1-10 integer>
Use realistic values and no extra text."""
        core_raw = _generate_json_text(core_prompt, model, tokenizer)
        core_data = _parse_core_metrics(core_raw)

        # Merge parsed outputs into one dict (validate fills missing fields).
        merged = {}
        for source in [macro_data, micro_data, qual_data]:
            if source:
                merged.update(source)
        if core_data:
            merged.update(core_data)

        return _validate(merged, food_name)

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

    # Health score in prose (e.g., "health score 7/10" or "score: 6")
    if "health_score" not in result:
        m = re.search(r"(?:health\s*score|score)\s*[:=]?\s*(\d{1,2})(?:\s*/\s*10)?", text, re.IGNORECASE)
        if m:
            result["health_score"] = float(m.group(1))

    return result


def _parse_core_metrics(text: str) -> dict:
    """Parse simple key=value model output for calories and health score."""
    out = {}
    if not text:
        return out
    c = re.search(r"calories_per_100g\s*[:=]\s*(\d+\.?\d*)", text, re.IGNORECASE)
    h = re.search(r"health_score\s*[:=]\s*(\d+\.?\d*)", text, re.IGNORECASE)
    if c:
        out["calories_per_100g"] = float(c.group(1))
    if h:
        out["health_score"] = float(h.group(1))
    return out


def _validate(data: dict, food_name: str = "") -> dict:
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

    _apply_rule_based_enrichment(data, food_name)
    _apply_metric_estimates_if_default(data, food_name)
    return data


def _apply_rule_based_enrichment(data: dict, food_name: str) -> None:
    """
    Improve weak model output with deterministic food-name heuristics.
    This prevents generic "Could not analyze" UI results.
    """
    name = (food_name or "").lower()

    vegan_keywords = [
        "tofu", "tempeh", "lentil", "dal", "chickpea", "falafel", "vegan", "salad",
        "rajma", "chana", "sambar", "idli", "poha", "upma", "aloo gobi", "bhindi",
    ]
    vegetarian_keywords = [
        "paneer", "cheese", "egg", "yogurt", "milk", "vegetarian", "palak paneer",
        "kadai paneer", "shahi paneer", "malai kofta", "veg pulao", "vegetable biryani",
    ]
    non_veg_keywords = [
        "chicken", "mutton", "lamb", "beef", "pork", "fish", "shrimp",
        "prawn", "tuna", "salmon", "meat", "bacon", "ham", "turkey",
        "chicken curry", "chicken biryani", "butter chicken", "tandoori chicken",
        "rogan josh", "keema", "fish curry", "prawn curry", "egg curry",
    ]
    gluten_keywords = [
        "bread", "pasta", "pizza", "burger", "noodle", "cake", "cookie", "donut",
        "naan", "roti", "chapati", "paratha", "kulcha", "puri", "bhatura",
    ]
    keto_keywords = ["avocado", "egg", "salmon", "chicken", "steak", "keto"]
    junk_keywords = [
        "pizza", "burger", "fries", "fried", "donut", "cake", "soda", "nachos",
        "pakora", "samosa", "jalebi", "gulab jamun", "kachori", "chole bhature",
    ]
    healthy_keywords = [
        "salad", "grilled", "steamed", "fruit", "vegetable", "oatmeal", "yogurt", "smoothie",
        "dal", "rajma", "chana", "khichdi", "idli", "sambar", "sprouts", "millet",
    ]
    has_non_veg_signal = any(k in name for k in non_veg_keywords)
    dairy_signals = ["paneer", "butter", "ghee", "lassi", "curd", "yogurt", "kheer", "milk"]

    if not data.get("is_vegan", False):
        data["is_vegan"] = any(k in name for k in vegan_keywords)
    if not data.get("is_vegetarian", False):
        data["is_vegetarian"] = data["is_vegan"] or any(k in name for k in vegetarian_keywords)
    if has_non_veg_signal:
        # Hard override for clearly non-vegetarian foods.
        data["is_vegan"] = False
        data["is_vegetarian"] = False
    if data.get("is_gluten_free", False) is False:
        if any(k in name for k in gluten_keywords):
            data["is_gluten_free"] = False
    if not data.get("is_keto_friendly", False):
        data["is_keto_friendly"] = any(k in name for k in keto_keywords)

    # Indian cuisine-specific hard overrides for common dishes.
    if "chicken biryani" in name or "mutton biryani" in name:
        data["is_vegan"] = False
        data["is_vegetarian"] = False
        data["diet_type"] = "High-Protein"
    elif "vegetable biryani" in name or "veg biryani" in name:
        data["is_vegan"] = False
        data["is_vegetarian"] = True
        data["diet_type"] = "Vegetarian"
    elif "fish curry" in name or "prawn curry" in name:
        data["is_vegan"] = False
        data["is_vegetarian"] = False
        data["diet_type"] = "High-Protein"
    elif "paneer" in name:
        data["is_vegan"] = False
        data["is_vegetarian"] = True
        data["diet_type"] = "Vegetarian"
    elif "chole bhature" in name:
        data["is_vegan"] = False
        data["is_vegetarian"] = True
        data["diet_type"] = "Junk Food"
        data["is_gluten_free"] = False
    elif "idli sambar" in name:
        data["is_vegetarian"] = True
        data["diet_type"] = "Vegetarian"

    allergens_l = [str(a).lower() for a in data.get("allergens", [])]
    if any(k in name for k in dairy_signals):
        if "milk" not in allergens_l:
            data["allergens"].append("milk")
            allergens_l.append("milk")
    if any(k in name for k in ["roti", "naan", "paratha", "chapati", "kulcha", "bhatura", "seviyan"]):
        if "gluten" not in allergens_l:
            data["allergens"].append("gluten")
            allergens_l.append("gluten")
    if "peanut" in name and "peanut" not in allergens_l:
        data["allergens"].append("peanut")
        allergens_l.append("peanut")
    if "egg" in name and "egg" not in allergens_l:
        data["allergens"].append("egg")

    # Remove likely false dairy allergen when dish has no dairy signals.
    if "milk" in allergens_l and not any(k in name for k in dairy_signals):
        data["allergens"] = [a for a in data["allergens"] if str(a).lower() != "milk"]

    # Prefer deterministic category signals from food name.
    if any(k in name for k in junk_keywords):
        data["diet_type"] = "Junk Food"
    elif has_non_veg_signal and data.get("diet_type") in ("Vegetarian", "Vegan"):
        data["diet_type"] = "High-Protein"
    elif data.get("diet_type", "Unknown") in ("Unknown", ""):
        if data["is_vegan"]:
            data["diet_type"] = "Vegan"
        elif data["is_vegetarian"]:
            data["diet_type"] = "Vegetarian"
        elif data["is_keto_friendly"]:
            data["diet_type"] = "Keto"
        elif any(k in name for k in healthy_keywords):
            data["diet_type"] = "Mediterranean"
        else:
            data["diet_type"] = "Balanced"
    elif data["diet_type"] == "Balanced" and any(k in name for k in healthy_keywords):
        data["diet_type"] = "Mediterranean"

    generic_tip = FALLBACK_NUTRITION["health_tips"]
    if not data.get("health_tips") or data["health_tips"] == generic_tip:
        if data["diet_type"] == "Junk Food":
            data["health_tips"] = "Add vegetables and reduce portion size to improve nutrient balance."
        elif data["diet_type"] in ("Vegan", "Vegetarian"):
            data["health_tips"] = "Pair with a complete protein source and include vitamin B12-rich foods."
        elif data["is_keto_friendly"]:
            data["health_tips"] = "Balance fats with fiber-rich vegetables and stay hydrated."
        else:
            data["health_tips"] = "Pair this meal with vegetables and watch sodium for better balance."


def _apply_metric_estimates_if_default(data: dict, food_name: str) -> None:
    """
    Replace static fallback-like metrics with cuisine-aware estimates when model
    output is weak. This avoids the same numbers for every food.
    """
    name = (food_name or "").lower()
    is_default_cal = abs(float(data.get("calories_per_100g", 0)) - float(FALLBACK_NUTRITION["calories_per_100g"])) < 1e-6
    is_default_macro_signature = (
        abs(float(data.get("protein_g", 0)) - float(FALLBACK_NUTRITION["protein_g"])) < 1e-6
        and abs(float(data.get("carbs_g", 0)) - float(FALLBACK_NUTRITION["carbs_g"])) < 1e-6
        and abs(float(data.get("fat_g", 0)) - float(FALLBACK_NUTRITION["fat_g"])) < 1e-6
    )
    current_score = int(data.get("health_score", 5))
    is_default_score = current_score == int(FALLBACK_NUTRITION["health_score"])
    suspicious_low_score = current_score <= 2 and data.get("diet_type") != "Junk Food"

    if is_default_cal or is_default_macro_signature:
        estimates = [
            (
                ["chicken biryani", "mutton biryani"],
                {"calories_per_100g": 180, "protein_g": 9, "carbs_g": 24, "fat_g": 7, "fiber_g": 1.8, "sugar_g": 1.5, "sodium_mg": 430},
            ),
            (
                ["veg biryani", "vegetable biryani"],
                {"calories_per_100g": 160, "protein_g": 4.5, "carbs_g": 26, "fat_g": 4.5, "fiber_g": 2.6, "sugar_g": 2.2, "sodium_mg": 360},
            ),
            (
                ["idli sambar"],
                {"calories_per_100g": 95, "protein_g": 3.8, "carbs_g": 17, "fat_g": 1.2, "fiber_g": 2.4, "sugar_g": 1.3, "sodium_mg": 260},
            ),
            (
                ["chole bhature"],
                {"calories_per_100g": 280, "protein_g": 8.5, "carbs_g": 33, "fat_g": 13, "fiber_g": 4.8, "sugar_g": 3.0, "sodium_mg": 520},
            ),
            (
                ["paneer tikka", "palak paneer", "kadai paneer", "paneer"],
                {"calories_per_100g": 240, "protein_g": 13, "carbs_g": 7, "fat_g": 17, "fiber_g": 1.2, "sugar_g": 3.0, "sodium_mg": 420},
            ),
            (
                ["butter chicken"],
                {"calories_per_100g": 230, "protein_g": 14, "carbs_g": 7, "fat_g": 16, "fiber_g": 1.0, "sugar_g": 3.0, "sodium_mg": 480},
            ),
            (
                ["fish curry", "prawn curry"],
                {"calories_per_100g": 170, "protein_g": 16, "carbs_g": 6, "fat_g": 9, "fiber_g": 1.0, "sugar_g": 2.0, "sodium_mg": 390},
            ),
            (
                ["dal", "rajma", "chana"],
                {"calories_per_100g": 130, "protein_g": 7.5, "carbs_g": 17, "fat_g": 3.0, "fiber_g": 5.0, "sugar_g": 2.0, "sodium_mg": 300},
            ),
            (
                ["pizza"],
                {"calories_per_100g": 260, "protein_g": 11, "carbs_g": 31, "fat_g": 10, "fiber_g": 2.2, "sugar_g": 3.8, "sodium_mg": 620},
            ),
            (
                ["burger"],
                {"calories_per_100g": 255, "protein_g": 12, "carbs_g": 29, "fat_g": 11, "fiber_g": 1.8, "sugar_g": 4.5, "sodium_mg": 590},
            ),
            (
                ["salad"],
                {"calories_per_100g": 70, "protein_g": 2.5, "carbs_g": 8.0, "fat_g": 3.0, "fiber_g": 3.5, "sugar_g": 3.0, "sodium_mg": 120},
            ),
        ]
        for keys, profile in estimates:
            if any(k in name for k in keys):
                for k, v in profile.items():
                    data[k] = float(v)
                break

    if is_default_score or suspicious_low_score:
        score = 6
        if data.get("diet_type") == "Junk Food":
            score = 3
        elif data.get("diet_type") in ("Vegan", "Vegetarian", "Mediterranean"):
            score = 7
        elif data.get("diet_type") == "High-Protein":
            score = 6

        sodium = float(data.get("sodium_mg", 0) or 0)
        sugar = float(data.get("sugar_g", 0) or 0)
        fiber = float(data.get("fiber_g", 0) or 0)
        calories = float(data.get("calories_per_100g", 0) or 0)

        if sodium > 500:
            score -= 1
        if sugar > 15:
            score -= 1
        if fiber >= 4:
            score += 1
        if calories > 280:
            score -= 1
        if calories < 120 and fiber >= 2:
            score += 1

        data["health_score"] = max(1, min(10, int(score)))


def get_diet_summary(nutrition: dict) -> str:
    return (
        f"{nutrition.get('diet_type', 'Unknown')} diet  ·  "
        f"{nutrition.get('calories_per_100g', '?'):.0f} kcal/100g  ·  "
        f"Health score: {nutrition.get('health_score', '?')}/10"
    )
