"""
diet_classifier.py
------------------
Uses Mistral-7B-Instruct to classify diet type and generate nutrition data
from a food name. The LLM acts as a nutrition expert, returning structured
JSON that our app can parse and display.

Why Mistral?
  - Fully open-source (Apache 2.0 license)
  - Runs locally on your machine (no API key needed)
  - Strong instruction-following capability
  - 7B parameters is a good balance of speed vs quality
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import re


# ── Default fallback data if LLM fails ───────────────────────────────────────
FALLBACK_NUTRITION = {
    "diet_type": "Unknown",
    "calories_per_100g": 200,
    "protein_g": 10,
    "carbs_g": 25,
    "fat_g": 8,
    "fiber_g": 3,
    "sugar_g": 5,
    "sodium_mg": 300,
    "is_vegan": False,
    "is_vegetarian": False,
    "is_gluten_free": False,
    "is_keto_friendly": False,
    "health_score": 5,
    "health_tips": "Unable to analyze. Please try again.",
    "allergens": [],
}

# ── Diet type descriptions (used in the prompt for better accuracy) ───────────
DIET_TYPES = [
    "Keto", "Vegan", "Vegetarian", "Mediterranean", "Paleo",
    "High-Protein", "Low-Carb", "Balanced", "Junk Food", "Raw Food",
]


def load_llm_model(model_name: str = "mistralai/Mistral-7B-Instruct-v0.2"):
    """
    Downloads and loads Mistral 7B model + tokenizer.
    
    - AutoTokenizer: converts text → numbers (tokens) the model understands
    - AutoModelForCausalLM: the actual language model that generates text
    - device_map="auto": uses GPU if available, else CPU
    - torch_dtype=float16: half-precision to save ~50% RAM
    
    This is slow the first time (downloads ~14GB) but cached after that.
    """
    print(f"Loading LLM: {model_name}... (first time may take several minutes)")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",          # auto-detect GPU/CPU
        torch_dtype=torch.float16,  # float16 = half precision = less RAM
        low_cpu_mem_usage=True,     # load model layer by layer to avoid RAM spike
    )
    model.eval()
    return tokenizer, model


def classify_diet(food_name: str, tokenizer, model) -> dict:
    """
    Takes a food name and returns a nutrition dictionary.

    The prompt is carefully structured:
      1. We tell Mistral to act as a nutrition expert
      2. We ask for ONLY JSON (no extra text) — makes parsing reliable
      3. We specify exact keys we need
      4. We use [INST]...[/INST] which is Mistral's instruction format
    
    Returns a dict with all nutrition fields, or FALLBACK_NUTRITION on error.
    """
    prompt = f"""[INST] You are a professional nutritionist and dietitian.
Analyze the food item: "{food_name}"

Return ONLY a valid JSON object. No explanation, no markdown, no extra text.
Use exactly these keys:

{{
  "diet_type": "one of: {', '.join(DIET_TYPES)}",
  "calories_per_100g": <integer>,
  "protein_g": <float per 100g>,
  "carbs_g": <float per 100g>,
  "fat_g": <float per 100g>,
  "fiber_g": <float per 100g>,
  "sugar_g": <float per 100g>,
  "sodium_mg": <integer per 100g>,
  "is_vegan": <true or false>,
  "is_vegetarian": <true or false>,
  "is_gluten_free": <true or false>,
  "is_keto_friendly": <true or false>,
  "health_score": <integer 1-10, where 10 is healthiest>,
  "health_tips": "<one short sentence tip>",
  "allergens": ["list", "of", "common", "allergens"]
}}
[/INST]"""

    # Convert text prompt → token IDs (numbers)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=400,    # max length of the response
            temperature=0.1,       # low temperature = more deterministic/consistent
            do_sample=True,        # enables temperature-based sampling
            pad_token_id=tokenizer.eos_token_id,
        )

    # Convert token IDs back to text
    raw_output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    # Extract only the part after [/INST] — that's the model's actual response
    if "[/INST]" in raw_output:
        raw_output = raw_output.split("[/INST]")[-1].strip()

    return _parse_json_response(raw_output, food_name)


def _parse_json_response(raw_text: str, food_name: str) -> dict:
    """
    Safely extracts JSON from the LLM response.
    LLMs sometimes add extra words — this handles that gracefully.
    """
    # Try to find a JSON block {...}
    json_match = re.search(r'\{[^{}]*\}', raw_text, re.DOTALL)
    
    if json_match:
        try:
            data = json.loads(json_match.group())
            # Validate required fields exist, fill missing ones from fallback
            for key, val in FALLBACK_NUTRITION.items():
                if key not in data:
                    data[key] = val
            return data
        except json.JSONDecodeError:
            pass

    # If JSON parsing fails entirely, return fallback
    print(f"Warning: Could not parse LLM response for '{food_name}'. Using fallback.")
    fallback = FALLBACK_NUTRITION.copy()
    fallback["health_tips"] = f"Could not analyze '{food_name}'. Try a more specific name."
    return fallback


def get_diet_summary(nutrition: dict) -> str:
    """
    Returns a human-readable one-line summary of the diet classification.
    """
    diet = nutrition.get("diet_type", "Unknown")
    score = nutrition.get("health_score", "?")
    cal = nutrition.get("calories_per_100g", "?")
    return f"{diet} diet · {cal} kcal/100g · Health score: {score}/10"
