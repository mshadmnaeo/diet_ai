"""
app.py
------
Main Streamlit application entry point.

Run with:  streamlit run app.py

Flow:
  1. User uploads an image or types a food name (sidebar)
  2. CLIP model identifies the food from the image
  3. Mistral LLM generates detailed nutrition data as JSON
  4. Open Food Facts API cross-checks/enriches the data
  5. Charts and tables are displayed in the main area
"""

import streamlit as st
from PIL import Image
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
from diet_classifier import load_llm_model, classify_diet, get_diet_summary
from nutrition_lookup import fetch_from_open_food_facts, merge_nutrition_data
from chart_generator import (
    build_nutrition_dataframe,
    plot_macro_pie,
    plot_nutrient_bars,
    plot_health_gauge,
)

# ── Page configuration (must be first Streamlit call) ────────────────────────
st.set_page_config(
    page_title="Diet AI",
    page_icon="🥗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS for minor style improvements ───────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 12px 16px;
        text-align: center;
        border: 1px solid #e9ecef;
    }
    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 13px;
        font-weight: 600;
        margin: 3px;
    }
    .badge-yes  { background: #d4edda; color: #155724; }
    .badge-no   { background: #f8d7da; color: #721c24; }
    .section-divider { margin: 20px 0; border-top: 1px solid #e9ecef; }
</style>
""", unsafe_allow_html=True)


# ── Model loading (cached — loads only once per session) ─────────────────────
@st.cache_resource(show_spinner="Loading vision model (CLIP)...")
def get_clip():
    """Cache the CLIP model so it loads only once."""
    return load_clip_model()


@st.cache_resource(show_spinner="Loading language model (Mistral 7B)...")
def get_llm():
    """Cache Mistral so it loads only once. First load: ~14GB download."""
    return load_llm_model()


# ── Header ────────────────────────────────────────────────────────────────────
st.title("🥗 Diet AI — Nutrition Predictor")
st.caption(
    "Upload a food photo (or type a food name) to get instant diet classification "
    "and a detailed nutrition breakdown."
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
            help="Supported formats: JPG, PNG, WEBP",
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
        help="All nutrition values scale with this slider",
    )

    use_api = st.toggle(
        "Enrich with Open Food Facts",
        value=True,
        help="Fetches real data from Open Food Facts database to verify AI results",
    )

    st.divider()

    analyze_btn = st.button(
        "Analyze",
        type="primary",
        use_container_width=True,
        icon="🔍",
    )

    st.caption("*First run downloads models (~14GB). Subsequent runs are fast.*")


# ── Main analysis logic ───────────────────────────────────────────────────────
if analyze_btn:

    # ── Validate input
    if not uploaded_file and not manual_food:
        st.warning("Please upload a food image or type a food name to continue.")
        st.stop()

    # ── Step 1: Identify the food
    food_name  = None
    confidence = 1.0
    top_guesses = []

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")

        col_img, col_info = st.columns([1, 2])
        with col_img:
            st.image(image, caption="Uploaded image", use_container_width=True)

        with col_info:
            with st.spinner("Identifying food with CLIP..."):
                try:
                    clip_model, clip_processor = get_clip()
                    food_name, confidence = identify_food(image, clip_model, clip_processor)
                    top_guesses = get_top_n_foods(image, clip_model, clip_processor, n=3)
                except Exception as e:
                    st.error(f"Image recognition failed: {e}")
                    st.stop()

            st.success(f"**Detected:** {food_name.title()}  ({confidence:.0%} confidence)")

            if top_guesses:
                st.caption("Other possibilities:")
                for guess in top_guesses[1:]:
                    st.caption(f"  • {guess['food'].title()} — {guess['confidence']:.0%}")

    else:
        food_name = manual_food.strip()
        st.info(f"Analyzing: **{food_name.title()}**")

    if not food_name:
        st.error("Could not determine food name.")
        st.stop()

    # ── Step 2: Get nutrition from LLM
    with st.spinner(f"Analyzing nutrition for '{food_name}' with Mistral..."):
        try:
            tokenizer, llm_model = get_llm()
            nutrition = classify_diet(food_name, tokenizer, llm_model)
        except Exception as e:
            st.error(f"Nutrition analysis failed: {e}")
            st.code(traceback.format_exc())
            st.stop()

    # ── Step 3: Optionally enrich with Open Food Facts
    if use_api:
        with st.spinner("Cross-checking with Open Food Facts..."):
            api_data = fetch_from_open_food_facts(food_name)
            nutrition = merge_nutrition_data(nutrition, api_data)
            if api_data:
                st.caption(f"Data source: {nutrition.get('data_source', 'AI Analysis')}")

    st.divider()

    # ── Step 4: Display diet summary badges
    st.subheader("Diet classification")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Diet type",     nutrition.get("diet_type", "Unknown"))
    m2.metric("Calories/100g", f"{nutrition.get('calories_per_100g', 0)} kcal")
    m3.metric("Health score",  f"{nutrition.get('health_score', '?')}/10")
    m4.metric("Portion cals",  f"{round(nutrition.get('calories_per_100g', 0) * portion_g / 100)} kcal")

    # Diet badges
    badges_html = ""
    badge_map = {
        "Vegan":        nutrition.get("is_vegan", False),
        "Vegetarian":   nutrition.get("is_vegetarian", False),
        "Gluten-free":  nutrition.get("is_gluten_free", False),
        "Keto-friendly": nutrition.get("is_keto_friendly", False),
    }
    for label, value in badge_map.items():
        css_class = "badge-yes" if value else "badge-no"
        icon = "✓" if value else "✗"
        badges_html += f'<span class="badge {css_class}">{icon} {label}</span>'

    st.markdown(badges_html, unsafe_allow_html=True)

    # Health tip
    tip = nutrition.get("health_tips", "")
    if tip:
        st.info(f"Tip: {tip}")

    # Allergens
    allergens = nutrition.get("allergens", [])
    if allergens:
        st.warning(f"Common allergens: {', '.join(allergens)}")

    st.divider()

    # ── Step 5: Charts
    st.subheader("Nutrition charts")

    chart_col1, chart_col2, chart_col3 = st.columns([1.2, 1.5, 1])

    with chart_col1:
        fig_pie = plot_macro_pie(nutrition, food_name.title(), portion_g)
        st.pyplot(fig_pie)
        plt_close = __import__("matplotlib.pyplot", fromlist=["close"])
        plt_close.close(fig_pie)

    with chart_col2:
        fig_bars = plot_nutrient_bars(nutrition, portion_g)
        st.pyplot(fig_bars)
        plt_close.close(fig_bars)

    with chart_col3:
        health_score = int(nutrition.get("health_score", 5))
        fig_gauge = plot_health_gauge(health_score)
        st.pyplot(fig_gauge)
        plt_close.close(fig_gauge)

    st.divider()

    # ── Step 6: Detailed nutrition table
    st.subheader("Full nutrition breakdown")

    df = build_nutrition_dataframe(nutrition, portion_g)
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Nutrient":         st.column_config.TextColumn("Nutrient", width="medium"),
            "Per 100g":         st.column_config.NumberColumn("Per 100g", format="%.1f"),
            f"Per {portion_g}g": st.column_config.NumberColumn(
                f"Per {portion_g}g", format="%.1f"
            ),
            "Unit":             st.column_config.TextColumn("Unit", width="small"),
        },
    )

    # Download button for the nutrition data
    csv_data = df.to_csv(index=False)
    st.download_button(
        label="Download nutrition data (CSV)",
        data=csv_data,
        file_name=f"{food_name.replace(' ', '_')}_nutrition.csv",
        mime="text/csv",
    )

# ── Empty state (no button clicked yet) ──────────────────────────────────────
else:
    st.markdown("""
    ### How to use

    1. Choose **Upload image** or **Type food name** in the sidebar
    2. Adjust the **portion size** slider if needed
    3. Click **Analyze**

    The app will:
    - Identify the food using the **CLIP** vision model
    - Classify the diet type and generate nutrition data using **Mistral 7B**
    - Cross-check values with the **Open Food Facts** database
    - Show macro charts, daily value bars, and a health score gauge
    """)

    st.image(
        "https://images.openfoodfacts.org/images/misc/openfoodfacts-logo-en-178x150.png",
        width=150,
        caption="Powered by Open Food Facts",
    )
