"""
food_recognizer.py
------------------
Food recognition using a 2-stage Hugging Face ensemble:
  1) prithivMLmods/Indian-Western-Food-34 (India-focused + Western)
  2) nateraw/food (Food-101 fallback for broader categories)
"""

from PIL import Image
from transformers import pipeline

PRIMARY_MODEL = "prithivMLmods/Indian-Western-Food-34"
FALLBACK_MODEL = "nateraw/food"
PRIMARY_CONFIDENCE_THRESHOLD = 0.55
_LAST_DECISION = {}


def load_clip_model():
    """
    Backward-compatible loader name used by app.py.
    Returns image-classification pipelines for primary and fallback models.
    """
    print(f"Loading primary food model: {PRIMARY_MODEL}")
    primary = pipeline("image-classification", model=PRIMARY_MODEL, device="cpu")
    print(f"Loading fallback food model: {FALLBACK_MODEL}")
    fallback = pipeline("image-classification", model=FALLBACK_MODEL, device="cpu")
    return primary, fallback


def _normalize_preds(preds: list[dict]) -> list[dict]:
    out = []
    for p in preds:
        label = str(p.get("label", "")).replace("_", " ").strip().lower()
        score = float(p.get("score", 0.0))
        if label:
            out.append({"food": label, "confidence": score})
    return out


def _choose_predictions(primary_preds: list[dict], fallback_preds: list[dict]) -> list[dict]:
    if primary_preds and primary_preds[0]["confidence"] >= PRIMARY_CONFIDENCE_THRESHOLD:
        return primary_preds
    return fallback_preds if fallback_preds else primary_preds


def _run_ensemble(image: Image.Image, primary_pipe, fallback_pipe, top_k: int = 5) -> tuple[list[dict], str]:
    primary = _normalize_preds(primary_pipe(image, top_k=max(5, top_k)))
    fallback = _normalize_preds(fallback_pipe(image, top_k=max(5, top_k)))
    chosen = _choose_predictions(primary, fallback)
    source = "primary"
    if chosen == fallback and fallback:
        source = "fallback"
    if not chosen:
        source = "none"

    global _LAST_DECISION
    _LAST_DECISION = {
        "source": source,
        "primary_top_confidence": primary[0]["confidence"] if primary else None,
        "fallback_top_confidence": fallback[0]["confidence"] if fallback else None,
        "threshold": PRIMARY_CONFIDENCE_THRESHOLD,
    }
    return chosen, source


def get_last_decision() -> dict:
    """Expose latest routing decision for UI/debugging."""
    return dict(_LAST_DECISION)


def identify_food(image: Image.Image, model, processor) -> tuple[str, float]:
    """
    Identify food using primary model and confidence-based fallback.
    Returns (food_name, confidence_score 0.0-1.0).
    """
    chosen, _ = _run_ensemble(image, model, processor, top_k=5)
    if not chosen:
        return "unknown food", 0.0
    return chosen[0]["food"], chosen[0]["confidence"]


def get_top_n_foods(image: Image.Image, model, processor, n: int = 3) -> list[dict]:
    """Return top-N predictions from confidence-selected model path."""
    chosen, _ = _run_ensemble(image, model, processor, top_k=max(5, n))
    return chosen[:n]
