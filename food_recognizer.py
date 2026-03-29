"""
food_recognizer.py
------------------
Uses CLIP (openai/clip-vit-base-patch32) to identify food from an image.
~600MB — fits comfortably on Streamlit Cloud free tier.
"""

from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch

FOOD_LABELS = [
    "pizza", "salad", "grilled chicken", "sushi", "burger",
    "pasta", "fruit bowl", "oatmeal", "steak", "vegetable curry",
    "sandwich", "soup", "fried rice", "tacos", "smoothie",
    "pancakes", "waffles", "ramen", "biryani", "dal",
    "idli sambar", "dosa", "chapati", "paratha", "butter chicken",
    "fish and chips", "caesar salad", "greek salad", "avocado toast",
    "scrambled eggs", "fried eggs", "boiled eggs", "yogurt parfait",
    "granola", "muesli", "protein shake", "wrap", "quesadilla",
    "nachos", "spring rolls", "dumplings", "pad thai", "pho",
    "hummus", "falafel", "shawarma", "kebab", "roast chicken",
    "mashed potato", "french fries", "sweet potato", "corn",
    "broccoli", "stir fry vegetables", "tofu", "tempeh",
    "chocolate cake", "ice cream", "cheesecake", "apple pie",
    "cookie", "brownie", "muffin", "donut",
]


def load_clip_model():
    """Load CLIP model and processor. Cached by Streamlit."""
    print("Loading CLIP model (openai/clip-vit-base-patch32)...")
    try:
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    except Exception as e:
        msg = str(e)
        if "ProxyError" in msg or "403 Forbidden" in msg:
            raise RuntimeError(
                "Hugging Face download blocked by proxy (403). "
                "Disable HTTP(S)_PROXY for this app or add Hugging Face domains "
                "to NO_PROXY, then restart Streamlit."
            ) from e
        raise
    model.eval()
    return model, processor


def identify_food(image: Image.Image, model, processor) -> tuple[str, float]:
    """
    Identify the food in an image using CLIP zero-shot classification.
    Returns (food_name, confidence_score 0.0-1.0).
    """
    text_inputs = [f"a photo of {label}" for label in FOOD_LABELS]

    inputs = processor(
        text=text_inputs,
        images=image,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probs   = outputs.logits_per_image.softmax(dim=1)
    top_idx = probs.argmax(dim=1).item()
    return FOOD_LABELS[top_idx], float(probs[0][top_idx])


def get_top_n_foods(image: Image.Image, model, processor, n: int = 3) -> list[dict]:
    """Return the top-N food predictions with confidence scores."""
    text_inputs = [f"a photo of {label}" for label in FOOD_LABELS]

    inputs = processor(
        text=text_inputs,
        images=image,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probs       = outputs.logits_per_image.softmax(dim=1)[0]
    top_indices = probs.topk(n).indices.tolist()

    return [
        {"food": FOOD_LABELS[idx], "confidence": float(probs[idx])}
        for idx in top_indices
    ]
