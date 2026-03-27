"""
nutrition_lookup.py
-------------------
Fetches real nutrition data from Open Food Facts — a free, open-source
food database with over 3 million products. This is used as a secondary
source to verify or supplement what the LLM returns.

Open Food Facts API: https://world.openfoodfacts.org/
No API key required. Completely free and open-source.
"""

import requests
from typing import Optional


BASE_URL = "https://world.openfoodfacts.org/cgi/search.pl"


def fetch_from_open_food_facts(food_name: str) -> Optional[dict]:
    """
    Searches Open Food Facts for a food item and returns
    the first result's nutrition data.
    
    Returns None if no result found or request fails.
    """
    params = {
        "search_terms": food_name,
        "search_simple": 1,
        "action": "process",
        "json": 1,
        "page_size": 1,           # we only need the top result
        "fields": "product_name,nutriments,categories_tags",
    }

    try:
        response = requests.get(BASE_URL, params=params, timeout=8)
        response.raise_for_status()
        data = response.json()

        products = data.get("products", [])
        if not products:
            return None

        product = products[0]
        nutriments = product.get("nutriments", {})

        return {
            "source": "Open Food Facts",
            "product_name": product.get("product_name", food_name),
            "calories_per_100g": nutriments.get("energy-kcal_100g", None),
            "protein_g":         nutriments.get("proteins_100g", None),
            "carbs_g":           nutriments.get("carbohydrates_100g", None),
            "fat_g":             nutriments.get("fat_100g", None),
            "fiber_g":           nutriments.get("fiber_100g", None),
            "sugar_g":           nutriments.get("sugars_100g", None),
            "sodium_mg":         _convert_sodium(nutriments.get("sodium_100g", None)),
        }

    except requests.RequestException as e:
        print(f"Open Food Facts lookup failed: {e}")
        return None


def _convert_sodium(sodium_g: Optional[float]) -> Optional[int]:
    """
    Open Food Facts stores sodium in grams per 100g.
    We convert to milligrams (more standard for nutrition labels).
    """
    if sodium_g is None:
        return None
    return int(sodium_g * 1000)


def merge_nutrition_data(llm_data: dict, api_data: Optional[dict]) -> dict:
    """
    Merges LLM nutrition data with Open Food Facts data.
    
    Strategy: prefer real API data for numeric values (more accurate),
    keep LLM data for qualitative fields (diet_type, is_vegan, etc.)
    """
    if api_data is None:
        return llm_data

    merged = llm_data.copy()

    # For each numeric field, use API value if available and non-zero
    numeric_fields = ["calories_per_100g", "protein_g", "carbs_g",
                      "fat_g", "fiber_g", "sugar_g", "sodium_mg"]

    for field in numeric_fields:
        api_val = api_data.get(field)
        if api_val is not None and api_val > 0:
            merged[field] = round(api_val, 1)

    merged["data_source"] = "Open Food Facts + AI Analysis"
    return merged
