"""
app.py
------
Main Streamlit application — the only file you run.

    streamlit run app.py

Uses:
  - CLIP          : food image recognition    (~600MB, HuggingFace)
  - Flan-T5-Large : nutrition analysis        (~770MB, HuggingFace, Google)
  - Open Food Facts: real nutrition database  (free, no key needed)

No API keys. No accounts. 100% HuggingFace open-source models.
Works on Streamlit Cloud free tier.
"""

import os   # ✅ MUST be first
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import traceback

# ── Load Hugging Face token from Streamlit secrets ───────────────────────────
HF_TOKEN = None

try:
    # Option 1: flat structure
    HF_TOKEN = st.secrets["HF_TOKEN"]
except KeyError:
    try:
        # Option 2: nested structure
        HF_TOKEN = st.secrets["huggingface"]["token"]
    except KeyError:
        HF_TOKEN = None

# Set as environment variable (used by transformers / huggingface_hub)
if HF_TOKEN:
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN
    os.environ["HF_TOKEN"] = HF_TOKEN

# Local modules
from food_recognizer import load_clip_model, identify_food, get_top_n_foods
from diet_classifier  import load_nutrition_model, classify_diet, get_diet_summary
from nutrition_lookup  import fetch_from_open_food_facts, merge_nutrition_data
from chart_generator   import build_nutrition_df, plot_macro_pie, plot_nutrient_bars, plot_health_gauge

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Diet AI",
    page_icon="🥗",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .badge {
        display: inline-block;
        padding: 3px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        margin: 3px 4px 3px 0;
    }
    .badge-yes { background: #EAF3DE; color: #27500A; }
    .badge-no  { background: #FCEBEB; color: #791F1F; }
</style>
""", unsafe_allow_html=True)


# ── Load both models at startup (cached — loads only once) ────────────────────
@st.cache_resource(show_spinner="Loading vision model CLIP (~600MB)...")
def get_clip():
    """CLIP for food image recognition. ~600MB download on first run."""
    return load_clip_model()


@st.cache_resource(show_spinner="Loading nutrition model Flan-T5-XL (~3GB)...")
def get_nutrition_model():
    """Flan-T5-Large for nutrition analysis. ~770MB download on first run."""
    return load_nutrition_model()


# ── Header ────────────────────────────────────────────────────────────────────
st.title("🥗 Diet AI — Nutrition Predictor")
st.caption(
    "Upload a food photo or type a food name · "
    "Powered by CLIP + Flan-T5-XL (Google) · "
    "No API keys needed · 100% HuggingFace open-source"
)

# Show a one-time info message about first-run download
if "models_loaded" not in st.session_state:
    st.info(
        "First run: downloading CLIP (~600MB) and Flan-T5-XL (~3GB) from HuggingFace. "
        "This takes 5-10 minutes once. After that the app is fast.",
        icon="ℹ️",
    )

st.divider()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Input")

    input_mode = st.radio(
        "Choose input method",
        ["Upload image", "Type food name"],
        horizontal=True,
    )

    uploaded_file = None
    manual_food   = None

    if input_mode == "Upload image":
        uploaded_file = st.file_uploader(
            "Upload a food photo",
            type=["jpg", "jpeg", "png", "webp"],
            help="Clear, well-lit photos give the best results.",
        )
    else:
        manual_food = st.text_input(
            "Food name",
            placeholder="e.g. grilled salmon, biryani, avocado toast",
        )

    st.divider()
    st.subheader("Settings")

    portion_g = st.slider(
        "Portion size",
        min_value=50,
        max_value=600,
        value=100,
        step=25,
        format="%dg",
        help="All nutrition values scale with this slider.",
    )

    use_api = st.toggle(
        "Cross-check with Open Food Facts",
        value=True,
        help="Fetches real nutrition data to verify AI estimates.",
    )

    st.divider()
    analyze_btn = st.button(
        "Analyze",
        type="primary",
        use_container_width=True,
    )

    st.divider()
    st.caption("Models used:")
    st.caption("• CLIP (OpenAI) — vision")
    st.caption("• Flan-T5-XL (Google) — nutrition")
    st.caption("• Open Food Facts — real data")
    st.caption("All 100% free & open-source.")


# ── Main analysis ─────────────────────────────────────────────────────────────
if analyze_btn:

    if not uploaded_file and not manual_food:
        st.warning("Please upload a food image or type a food name.")
        st.stop()

    food_name  = None
    confidence = 1.0

    # ── Step 1: Food recognition via CLIP ─────────────────────────────────────
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")

        img_col, info_col = st.columns([1, 2])

        with img_col:
            st.image(image, use_container_width=True)

        with info_col:
            with st.spinner("Identifying food with CLIP..."):
                try:
                    clip_model, clip_proc = get_clip()
                    food_name, confidence = identify_food(
                        image, clip_model, clip_proc
                    )
                    top_guesses = get_top_n_foods(
                        image, clip_model, clip_proc, n=3
                    )
                    st.session_state["models_loaded"] = True
                except Exception as e:
                    st.error(f"Image recognition failed: {e}")
                    st.code(traceback.format_exc())
                    st.stop()

            st.success(
                f"Detected: **{food_name.title()}** "
                f"({confidence:.0%} confidence)"
            )
            if len(top_guesses) > 1:
                st.caption("Other possibilities:")
                for g in top_guesses[1:]:
                    st.caption(
                        f"  • {g['food'].title()} — {g['confidence']:.0%}"
                    )

    else:
        food_name = manual_food.strip()
        if not food_name:
            st.warning("Please enter a food name.")
            st.stop()
        st.info(f"Analyzing: **{food_name.title()}**")

    # ── Step 2: Nutrition analysis via Flan-T5 ────────────────────────────────
    with st.spinner(
        f"Analyzing '{food_name}' with Flan-T5-XL (Google)... "
        "This may take 20-40 seconds on CPU."
    ):
        try:
            # Preload both models (cached — only slow on very first run)
            get_clip()
            get_nutrition_model()
            st.session_state["models_loaded"] = True

            nutrition = classify_diet(food_name)
        except Exception as e:
            st.error(f"Nutrition analysis failed: {e}")
            st.code(traceback.format_exc())
            st.stop()

    # ── Step 3: Open Food Facts cross-check ──────────────────────────────────
    if use_api:
        with st.spinner("Cross-checking with Open Food Facts..."):
            api_data  = fetch_from_open_food_facts(food_name)
            nutrition = merge_nutrition_data(nutrition, api_data)
            if api_data:
                st.caption(
                    f"Data source: {nutrition.get('data_source', 'Flan-T5-XL AI')}"
                )

    st.divider()

    # ── Step 4: Diet summary ──────────────────────────────────────────────────
    st.subheader("Diet classification")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Diet type",     nutrition.get("diet_type", "Unknown"))
    c2.metric("Calories/100g", f"{nutrition.get('calories_per_100g', 0):.0f} kcal")
    c3.metric("Health score",  f"{nutrition.get('health_score', '?')}/10")
    c4.metric(
        f"Calories ({portion_g}g)",
        f"{round(nutrition.get('calories_per_100g', 0) * portion_g / 100)} kcal",
    )

    # Badges
    badge_html = ""
    for label, key in [
        ("Vegan",         "is_vegan"),
        ("Vegetarian",    "is_vegetarian"),
        ("Gluten-free",   "is_gluten_free"),
        ("Keto-friendly", "is_keto_friendly"),
    ]:
        val  = nutrition.get(key, False)
        css  = "badge-yes" if val else "badge-no"
        icon = "✓" if val else "✗"
        badge_html += f'<span class="badge {css}">{icon} {label}</span>'
    st.markdown(badge_html, unsafe_allow_html=True)

    tip = nutrition.get("health_tips", "")
    if tip:
        st.info(f"💡 {tip}")

    allergens = nutrition.get("allergens", [])
    if allergens:
        st.warning(f"⚠️ Common allergens: {', '.join(allergens)}")

    st.divider()

    # ── Step 5: Charts ────────────────────────────────────────────────────────
    st.subheader("Nutrition charts")

    ch1, ch2, ch3 = st.columns([1.2, 1.6, 1])

    with ch1:
        st.caption("Macronutrient split")
        fig = plot_macro_pie(nutrition, food_name, portion_g)
        st.pyplot(fig)
        plt.close(fig)

    with ch2:
        st.caption("% of daily recommended value")
        fig = plot_nutrient_bars(nutrition, portion_g)
        st.pyplot(fig)
        plt.close(fig)

    with ch3:
        st.caption("Health score")
        fig = plot_health_gauge(int(nutrition.get("health_score", 5)))
        st.pyplot(fig)
        plt.close(fig)

    st.divider()

    # ── Step 6: Nutrition table + CSV download ────────────────────────────────
    st.subheader("Full nutrition breakdown")

    df = build_nutrition_df(nutrition, portion_g)
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Nutrient":          st.column_config.TextColumn("Nutrient", width="medium"),
            "Per 100g":          st.column_config.NumberColumn("Per 100g",           format="%.1f"),
            f"Per {portion_g}g": st.column_config.NumberColumn(f"Per {portion_g}g", format="%.1f"),
            "Unit":              st.column_config.TextColumn("Unit", width="small"),
        },
    )

    st.download_button(
        label="Download as CSV",
        data=df.to_csv(index=False),
        file_name=f"{food_name.replace(' ', '_')}_nutrition.csv",
        mime="text/csv",
    )


# ── Empty state ───────────────────────────────────────────────────────────────
else:
    st.markdown("""
    ### How to use
    1. Choose **Upload image** or **Type food name** in the sidebar
    2. Adjust the **portion size** slider (default 100g)
    3. Click **Analyze**

    ---
    **Models (all free, no account needed):**
    - **CLIP** by OpenAI — identifies food from your photo (~600MB)
    - **Flan-T5-Large** by Google — nutrition analysis (~770MB)
    - **Open Food Facts** — real nutrition database (cross-check)

    Both models download from HuggingFace on first run and are cached locally after that.
    """)
