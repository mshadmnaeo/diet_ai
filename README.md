# Diet AI — Nutrition Predictor

An open-source Python app that identifies food from images and generates
a detailed nutrition chart using AI models — all running locally on your machine.

---

## What it does

1. You upload a food photo (or type a food name)
2. **CLIP** (vision AI) identifies what food it is
3. **Mistral 7B** (language AI) analyzes diet type and nutrition
4. **Open Food Facts** cross-checks the nutrition data
5. Streamlit displays pie charts, bar charts, a health gauge, and a data table

---

## Prerequisites

| Requirement       | Minimum        | Recommended     |
|-------------------|----------------|-----------------|
| Python            | 3.9            | 3.11            |
| RAM               | 16 GB          | 32 GB           |
| GPU VRAM          | None (CPU ok)  | 8 GB+ (faster)  |
| Disk space        | 20 GB free     | 30 GB free      |
| Internet (1st run)| Required       | —               |

> First run downloads ~14GB of model weights. After that the app runs fully offline.

---

## Installation

### Step 1 — Clone / download the project

```bash
# If you have git:
git clone <your-repo-url>
cd diet_ai

# Or just place all .py files in a folder called diet_ai/
```

### Step 2 — Create a virtual environment (recommended)

```bash
# Create it
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — (GPU users) Install CUDA-enabled PyTorch

If you have an NVIDIA GPU, replace the torch install with:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

---

## Running the app

```bash
streamlit run app.py
```

Streamlit will open your browser automatically at `http://localhost:8501`

---

## Project structure

```
diet_ai/
├── app.py                # Main Streamlit UI — entry point
├── food_recognizer.py    # CLIP food detection from images
├── diet_classifier.py    # Mistral 7B nutrition/diet analysis
├── nutrition_lookup.py   # Open Food Facts API integration
├── chart_generator.py    # Matplotlib chart generation
├── requirements.txt      # Python dependencies
└── README.md
```

---

## Models used

| Model                            | Purpose               | Size   | License     |
|----------------------------------|-----------------------|--------|-------------|
| `openai/clip-vit-base-patch32`   | Food image detection  | ~600MB | MIT         |
| `mistralai/Mistral-7B-Instruct`  | Nutrition reasoning   | ~14GB  | Apache 2.0  |

Both download automatically from HuggingFace on first run.

---

## Configuration

### Add more food labels (food_recognizer.py)

```python
FOOD_LABELS = [
    "pizza", "salad", ...
    "your_food_here",   # just add to this list
]
```

### Use a smaller/faster LLM (diet_classifier.py)

Replace Mistral with a lighter model if you have less RAM:

```python
# ~4GB RAM instead of 14GB
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# ~8GB RAM, faster than Mistral
model_name = "microsoft/phi-2"
```

### Use llama.cpp for even less RAM (optional)

```bash
pip install llama-cpp-python
```

Then use a GGUF quantized version of Mistral (~4-8GB):
Download from: https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF

---

## Troubleshooting

**Out of memory error:**
- Reduce model precision: change `torch.float16` to `torch.float8` (if supported)
- Use a smaller model (TinyLlama or Phi-2)
- Close other applications to free RAM

**Slow on CPU:**
- This is expected — Mistral 7B can take 2-5 minutes per query on CPU
- For speed, use a GPU or switch to a smaller model

**Model download stuck:**
- Check your internet connection
- HuggingFace sometimes rate-limits — wait and retry
- Set `HF_HUB_OFFLINE=1` to use cached models only

**Streamlit port conflict:**
```bash
streamlit run app.py --server.port 8502
```

---

## License

All models and libraries used are open-source:
- CLIP: MIT License
- Mistral 7B: Apache 2.0 License
- Open Food Facts data: Open Database License (ODbL)
- Streamlit, PyTorch, HuggingFace Transformers: Apache 2.0 / BSD
