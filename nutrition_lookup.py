"""
nutrition_lookup.py
-------------------
Fetches real nutrition data from Open Food Facts.
Free, open-source, no API key required.
Used to cross-check and improve the AI's estimates.
"""

import requests
from typing import Optional

BASE_URL = "https://world.openfoodfacts.org/cgi/search.pl"


def fetch_from_open_food_facts(food_name: str) -> Optional[dict]:
    """Search Open Food Facts and return the top result's nutrition data."""
    params = {
        "search_terms": food_name,
        "search_simple": 1,
        "action": "process",
        "json": 1,
        "page_size": 1,
        "fields": "product_name,nutriments",
    }

    try:
        response = requests.get(BASE_URL, params=params, timeout=8)
        response.raise_for_status()
        products = response.json().get("products", [])

        if not products:
            return None

        n = products[0].get("nutriments", {})

        return {
            "source":            "Open Food Facts",
            "product_name":      products[0].get("product_name", food_name),
            "calories_per_100g": n.get("energy-kcal_100g"),
            "protein_g":         n.get("proteins_100g"),
            "carbs_g":           n.get("carbohydrates_100g"),
            "fat_g":             n.get("fat_100g"),
            "fiber_g":           n.get("fiber_100g"),
            "sugar_g":           n.get("sugars_100g"),
            "sodium_mg":         int(n["sodium_100g"] * 1000)
                                 if n.get("sodium_100g") else None,
        }

    except requests.RequestException as e:
        print(f"Open Food Facts error for '{food_name}': {e}")
        return None


def merge_nutrition_data(llm_data: dict, api_data: Optional[dict]) -> dict:
    """
    Merge LLM and API data.
    API values are used for numeric fields when available (more accurate).
    LLM values are kept for qualitative fields (diet_type, is_vegan, etc.).
    """
    if api_data is None:
        return llm_data

    merged = llm_data.copy()

    for field in ["calories_per_100g", "protein_g", "carbs_g",
                  "fat_g", "fiber_g", "sugar_g", "sodium_mg"]:
        val = api_data.get(field)
        if val is not None and float(val) > 0:
            merged[field] = round(float(val), 1)

    merged["data_source"] = "Open Food Facts + Flan-T5 AI"
    return merged
