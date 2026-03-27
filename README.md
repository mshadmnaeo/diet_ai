# Diet AI — Flan-T5-XL Version

Better quality nutrition analysis using Google's Flan-T5-XL (3 billion parameters).
No API keys. No accounts. 100% HuggingFace open-source.

## Models used

| Model | Size | Purpose | RAM needed |
|---|---|---|---|
| openai/clip-vit-base-patch32 | ~600MB | Food image recognition | ~800MB |
| google/flan-t5-xl | ~3GB | Nutrition analysis | ~4GB |
| Open Food Facts | Free API | Real nutrition data | 0 |

**Total RAM needed: ~5GB**

---

## Why XL over Large?

| | Flan-T5-Large | Flan-T5-XL |
|---|---|---|
| Parameters | 770M | 3B |
| Size | 770MB | 3GB |
| RAM needed | ~1GB | ~4GB |
| Streamlit Cloud | Free tier | Paid tier / local |
| Nutrition accuracy | Good | Much better |
| JSON instruction following | Decent | Excellent |
| Speed on CPU | ~20s/query | ~45s/query |

---

## Quick start

```bash
# 1. Make sure you have 6GB+ free RAM and 6GB+ free disk

# 2. Go into the project folder
cd diet_ai_xl

# 3. Create virtual environment
python -m venv venv
source venv/bin/activate      # Mac/Linux
venv\Scripts\activate         # Windows

# 4. Install packages
pip install -r requirements.txt

# 5. Run
streamlit run app.py
# First run downloads ~3.6GB total — takes 5-10 minutes once
```

---

## Swap models easily

Change one line in diet_classifier.py:

```python
# Free tier / fast:
MODEL_NAME = "google/flan-t5-large"    # 770MB, ~1GB RAM

# This version (better quality):
MODEL_NAME = "google/flan-t5-xl"       # 3GB, ~4GB RAM

# Maximum quality (GPU recommended):
MODEL_NAME = "google/flan-t5-xxl"      # 11GB, ~14GB RAM
```

---

## GPU acceleration (optional but faster)

If you have an NVIDIA GPU, change one line in diet_classifier.py:

```python
# Change:
device="cpu"

# To:
device=0    # uses your first GPU
```

Also install CUDA PyTorch:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `Killed` or out of memory | Close other apps, need 5GB+ free RAM |
| Very slow (>2 min/query) | Normal on CPU — use GPU or switch to flan-t5-large |
| First run hangs at 0% | Slow internet — wait, model is 3GB |
| `ModuleNotFoundError` | `pip install -r requirements.txt` |
| Streamlit Cloud crashes | XL needs paid tier — use flan-t5-large for free tier |
