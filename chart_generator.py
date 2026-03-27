"""
chart_generator.py
------------------
All chart/visualization logic lives here.
Uses Matplotlib and Pandas to generate:
  1. Macro pie chart (protein / carbs / fat split)
  2. Full nutrient horizontal bar chart
  3. Daily value % progress bars
  4. Nutrition dataframe for st.dataframe()

Keeping charts in a separate file makes app.py cleaner
and makes it easy to swap chart libraries later.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np
from typing import Optional


# ── Color palette (matches a clean, health-focused aesthetic) ─────────────────
COLORS = {
    "protein": "#5DCAA5",   # teal-green
    "carbs":   "#7F77DD",   # purple
    "fat":     "#F0997B",   # coral
    "fiber":   "#FAC775",   # amber
    "sugar":   "#ED93B1",   # pink
    "sodium":  "#B4B2A9",   # gray
}

# ── Recommended Daily Values (based on 2000 kcal diet) ───────────────────────
DAILY_VALUES = {
    "calories_per_100g": 2000,
    "protein_g":         50,
    "carbs_g":           275,
    "fat_g":             78,
    "fiber_g":           28,
    "sugar_g":           50,
    "sodium_mg":         2300,
}


def build_nutrition_dataframe(nutrition: dict, portion_g: int) -> pd.DataFrame:
    """
    Creates a Pandas DataFrame with per-100g and per-portion columns.
    Used by st.dataframe() to show a sortable table.
    """
    scale = portion_g / 100

    rows = [
        {"Nutrient": "Calories (kcal)", "Per 100g": nutrition.get("calories_per_100g", 0),
         f"Per {portion_g}g": round(nutrition.get("calories_per_100g", 0) * scale, 1),
         "Unit": "kcal"},
        {"Nutrient": "Protein",         "Per 100g": nutrition.get("protein_g", 0),
         f"Per {portion_g}g": round(nutrition.get("protein_g", 0) * scale, 1),
         "Unit": "g"},
        {"Nutrient": "Carbohydrates",   "Per 100g": nutrition.get("carbs_g", 0),
         f"Per {portion_g}g": round(nutrition.get("carbs_g", 0) * scale, 1),
         "Unit": "g"},
        {"Nutrient": "Fat",             "Per 100g": nutrition.get("fat_g", 0),
         f"Per {portion_g}g": round(nutrition.get("fat_g", 0) * scale, 1),
         "Unit": "g"},
        {"Nutrient": "Fiber",           "Per 100g": nutrition.get("fiber_g", 0),
         f"Per {portion_g}g": round(nutrition.get("fiber_g", 0) * scale, 1),
         "Unit": "g"},
        {"Nutrient": "Sugar",           "Per 100g": nutrition.get("sugar_g", 0),
         f"Per {portion_g}g": round(nutrition.get("sugar_g", 0) * scale, 1),
         "Unit": "g"},
        {"Nutrient": "Sodium",          "Per 100g": nutrition.get("sodium_mg", 0),
         f"Per {portion_g}g": round(nutrition.get("sodium_mg", 0) * scale, 1),
         "Unit": "mg"},
    ]

    return pd.DataFrame(rows)


def plot_macro_pie(nutrition: dict, food_name: str, portion_g: int) -> plt.Figure:
    """
    Draws a pie chart showing protein / carbs / fat split.
    The slice sizes are scaled to the selected portion size.
    """
    scale = portion_g / 100

    protein = nutrition.get("protein_g", 0) * scale
    carbs   = nutrition.get("carbs_g", 0) * scale
    fat     = nutrition.get("fat_g", 0) * scale

    # Only include macros with value > 0 to avoid empty slices
    labels, sizes, colors = [], [], []
    for label, value, color in [
        ("Protein", protein, COLORS["protein"]),
        ("Carbs",   carbs,   COLORS["carbs"]),
        ("Fat",     fat,     COLORS["fat"]),
    ]:
        if value > 0:
            labels.append(f"{label}\n{value:.1f}g")
            sizes.append(value)
            colors.append(color)

    fig, ax = plt.subplots(figsize=(5, 5))
    fig.patch.set_alpha(0)       # transparent background
    ax.set_facecolor("none")

    if sum(sizes) > 0:
        wedges, texts, autotexts = ax.pie(
            sizes,
            labels=labels,
            colors=colors,
            autopct="%1.1f%%",
            startangle=140,
            pctdistance=0.75,
            wedgeprops={"linewidth": 1.5, "edgecolor": "white"},
        )
        for at in autotexts:
            at.set_fontsize(10)
            at.set_fontweight("bold")
    else:
        ax.text(0, 0, "No data", ha="center", va="center", fontsize=14)

    ax.set_title(
        f"Macros — {food_name} ({portion_g}g)",
        fontsize=12, fontweight="bold", pad=15
    )
    plt.tight_layout()
    return fig


def plot_nutrient_bars(nutrition: dict, portion_g: int) -> plt.Figure:
    """
    Draws a horizontal bar chart showing each nutrient as a % of daily value.
    Color coding: green = good, amber = moderate, red = over limit.
    """
    scale = portion_g / 100

    nutrients = {
        "Calories": (nutrition.get("calories_per_100g", 0) * scale,
                     DAILY_VALUES["calories_per_100g"], "kcal"),
        "Protein":  (nutrition.get("protein_g", 0) * scale,
                     DAILY_VALUES["protein_g"], "g"),
        "Carbs":    (nutrition.get("carbs_g", 0) * scale,
                     DAILY_VALUES["carbs_g"], "g"),
        "Fat":      (nutrition.get("fat_g", 0) * scale,
                     DAILY_VALUES["fat_g"], "g"),
        "Fiber":    (nutrition.get("fiber_g", 0) * scale,
                     DAILY_VALUES["fiber_g"], "g"),
        "Sugar":    (nutrition.get("sugar_g", 0) * scale,
                     DAILY_VALUES["sugar_g"], "g"),
        "Sodium":   (nutrition.get("sodium_mg", 0) * scale,
                     DAILY_VALUES["sodium_mg"], "mg"),
    }

    names, values, dv_pcts, bar_colors, value_labels = [], [], [], [], []

    for name, (val, dv, unit) in nutrients.items():
        pct = min((val / dv * 100) if dv > 0 else 0, 150)  # cap at 150%
        names.append(name)
        values.append(val)
        dv_pcts.append(pct)
        value_labels.append(f"{val:.1f} {unit}")

        # Color: green < 30%, amber 30-70%, red > 70% of daily value
        if pct < 30:
            bar_colors.append(COLORS["protein"])
        elif pct < 70:
            bar_colors.append(COLORS["fiber"])
        else:
            bar_colors.append(COLORS["fat"])

    fig, ax = plt.subplots(figsize=(6, 4.5))
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")

    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, dv_pcts, color=bar_colors,
                   edgecolor="white", linewidth=0.8, height=0.6)

    # Add value labels inside/next to bars
    for i, (bar, label) in enumerate(zip(bars, value_labels)):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                label, va="center", ha="left", fontsize=9, color="#555")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel("% of daily recommended value", fontsize=9)
    ax.set_xlim(0, 160)
    ax.axvline(x=100, color="#ccc", linestyle="--", linewidth=1, label="Daily value")
    ax.set_title(f"% Daily value ({portion_g}g portion)",
                 fontsize=12, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="both", labelsize=9)

    plt.tight_layout()
    return fig


def plot_health_gauge(health_score: int) -> plt.Figure:
    """
    Draws a simple semicircular gauge showing the health score (1-10).
    Green = healthy, amber = moderate, red = unhealthy.
    """
    fig, ax = plt.subplots(figsize=(4, 2.5), subplot_kw={"projection": "polar"})
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")

    # Draw colored arc segments
    segments = [
        (0, np.pi / 3, "#E24B4A"),        # red: 0-3
        (np.pi / 3, 2 * np.pi / 3, "#EF9F27"),  # amber: 3-6
        (2 * np.pi / 3, np.pi, "#5DCAA5"),  # green: 7-10
    ]
    for start, end, color in segments:
        theta = np.linspace(start, end, 100)
        ax.plot(theta, [1] * 100, color=color, linewidth=10, solid_capstyle="round")

    # Draw needle
    angle = (health_score / 10) * np.pi
    ax.annotate("", xy=(angle, 0.95), xytext=(angle, 0.05),
                arrowprops={"arrowstyle": "->", "color": "#333", "lw": 2})

    ax.set_ylim(0, 1.3)
    ax.set_xlim(0, np.pi)
    ax.axis("off")
    ax.set_title(f"Health score: {health_score}/10",
                 fontsize=12, fontweight="bold", y=0.05)

    return fig
