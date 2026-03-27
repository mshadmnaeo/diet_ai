"""
food_recognizer.py
------------------
Uses CLIP (openai/clip-vit-base-patch32) to identify food from an image.
CLIP is a zero-shot model — no custom training needed. It compares the image
against a list of text labels and picks the best match.
"""

from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch

# ── Extended food label list ──────────────────────────────────────────────────
# Add or remove items here to expand/reduce what the app can detect.
FOOD_LABELS = [
    "pizza", "salad", "grilled chicken", "sushi", "burger",
    "pasta", "fruit bowl", "oatmeal", "steak", "vegetable curry",
    "sandwich", "soup", "fried rice", "tacos", "smoothie",
    "pancakes", "waffles", "ramen", "biryani", "dal",
    "idli sambar", "dosa", "chapati", "paratha", "butter chicken",
    "fish and chips", "caesar salad", "Greek salad", "avocado toast",
    "scrambled eggs", "fried eggs", "boiled eggs", "yogurt parfait",
    "granola", "muesli", "protein shake", "wrap", "quesadilla",
    "nachos", "spring rolls", "dumplings", "pad thai", "pho",
    "hummus", "falafel", "shawarma", "kebab", "roast chicken",
    "mashed potato", "french fries", "sweet potato", "corn on cob",
    "broccoli", "stir fry vegetables", "tofu", "tempeh",
    "chocolate cake", "ice cream", "cheesecake", "apple pie",
]


def load_clip_model():
    """
    Downloads and returns the CLIP model + processor.
    Called once and cached by Streamlit via @st.cache_resource.
    """
    print("Loading CLIP model... (first time only)")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()
    return model, processor


def identify_food(image: Image.Image, model, processor) -> tuple[str, float]:
    """
    Takes a PIL Image and returns:
      - food_name (str): the best-matching food label
      - confidence (float): probability 0.0 to 1.0

    How it works:
      1. Convert each label to "a photo of <food>" text
      2. Convert image to pixel tensors
      3. CLIP computes similarity between image and each text
      4. softmax turns raw scores into probabilities
      5. argmax picks the highest probability
    """
    text_inputs = [f"a photo of {label}" for label in FOOD_LABELS]

    inputs = processor(
        text=text_inputs,
        images=image,
        return_tensors="pt",   # pt = PyTorch tensors
        padding=True,
        truncation=True,
    )

    with torch.no_grad():      # no_grad = faster, uses less memory (we're not training)
        outputs = model(**inputs)

    # logits_per_image: raw similarity scores, shape [1, num_labels]
    probs = outputs.logits_per_image.softmax(dim=1)  # convert to probabilities
    top_idx = probs.argmax(dim=1).item()             # index of highest probability

    food_name = FOOD_LABELS[top_idx]
    confidence = float(probs[0][top_idx])

    return food_name, confidence


def get_top_n_foods(image: Image.Image, model, processor, n: int = 3) -> list[dict]:
    """
    Returns the top-N food predictions with their confidence scores.
    Useful for showing alternative guesses in the UI.
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

    probs = outputs.logits_per_image.softmax(dim=1)[0]
    top_indices = probs.topk(n).indices.tolist()

    results = []
    for idx in top_indices:
        results.append({
            "food": FOOD_LABELS[idx],
            "confidence": float(probs[idx]),
        })
    return results
