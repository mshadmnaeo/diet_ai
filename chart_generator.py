"""
chart_generator.py
------------------
All chart generation: pie, bar, gauge, and nutrition DataFrame.
Uses Matplotlib and Pandas — no extra dependencies.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

COLORS = {
    "protein": "#5DCAA5",
    "carbs":   "#7F77DD",
    "fat":     "#F0997B",
    "fiber":   "#FAC775",
    "sugar":   "#ED93B1",
    "sodium":  "#B4B2A9",
}

DAILY_VALUES = {
    "calories_per_100g": 2000,
    "protein_g":          50,
    "carbs_g":           275,
    "fat_g":              78,
    "fiber_g":            28,
    "sugar_g":            50,
    "sodium_mg":        2300,
}


def get_portion_column_name(portion_g: int) -> str:
    """Return a unique portion column label for nutrition tables."""
    return "Per portion (100g)" if portion_g == 100 else f"Per {portion_g}g"


def build_nutrition_df(nutrition: dict, portion_g: int) -> pd.DataFrame:
    """Build a Pandas DataFrame scaled to the selected portion size."""
    scale = portion_g / 100
    portion_col = get_portion_column_name(portion_g)
    rows = [
        {"Nutrient": "Calories (kcal)", "Per 100g": nutrition.get("calories_per_100g", 0), "Unit": "kcal"},
        {"Nutrient": "Protein",         "Per 100g": nutrition.get("protein_g", 0),         "Unit": "g"},
        {"Nutrient": "Carbohydrates",   "Per 100g": nutrition.get("carbs_g", 0),            "Unit": "g"},
        {"Nutrient": "Fat",             "Per 100g": nutrition.get("fat_g", 0),              "Unit": "g"},
        {"Nutrient": "Fiber",           "Per 100g": nutrition.get("fiber_g", 0),            "Unit": "g"},
        {"Nutrient": "Sugar",           "Per 100g": nutrition.get("sugar_g", 0),            "Unit": "g"},
        {"Nutrient": "Sodium",          "Per 100g": nutrition.get("sodium_mg", 0),          "Unit": "mg"},
    ]
    df = pd.DataFrame(rows)
    df[portion_col] = df["Per 100g"].apply(lambda x: round(x * scale, 1))
    return df[["Nutrient", "Per 100g", portion_col, "Unit"]]


def plot_macro_pie(nutrition: dict, food_name: str, portion_g: int) -> plt.Figure:
    """Pie chart: protein / carbs / fat split."""
    scale   = portion_g / 100
    protein = nutrition.get("protein_g", 0) * scale
    carbs   = nutrition.get("carbs_g", 0)   * scale
    fat     = nutrition.get("fat_g", 0)     * scale

    labels, sizes, colors = [], [], []
    for name, val, color in [
        ("Protein", protein, COLORS["protein"]),
        ("Carbs",   carbs,   COLORS["carbs"]),
        ("Fat",     fat,     COLORS["fat"]),
    ]:
        if val > 0:
            labels.append(f"{name}\n{val:.1f}g")
            sizes.append(val)
            colors.append(color)

    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")

    if sum(sizes) > 0:
        wedges, texts, autotexts = ax.pie(
            sizes, labels=labels, colors=colors,
            autopct="%1.1f%%", startangle=140, pctdistance=0.78,
            wedgeprops={"linewidth": 1.5, "edgecolor": "white"},
        )
        for at in autotexts:
            at.set_fontsize(9)
            at.set_fontweight("bold")
        for t in texts:
            t.set_fontsize(9)
    else:
        ax.text(0, 0, "No data", ha="center", va="center", fontsize=12)

    ax.set_title(f"{food_name.title()} ({portion_g}g)",
                 fontsize=11, fontweight="bold", pad=12)
    plt.tight_layout()
    return fig


def plot_nutrient_bars(nutrition: dict, portion_g: int) -> plt.Figure:
    """Horizontal bar chart: each nutrient as % of daily recommended value."""
    scale = portion_g / 100

    nutrients = {
        "Calories": (nutrition.get("calories_per_100g", 0) * scale, DAILY_VALUES["calories_per_100g"], "kcal"),
        "Protein":  (nutrition.get("protein_g", 0) * scale,         DAILY_VALUES["protein_g"],         "g"),
        "Carbs":    (nutrition.get("carbs_g", 0) * scale,            DAILY_VALUES["carbs_g"],            "g"),
        "Fat":      (nutrition.get("fat_g", 0) * scale,              DAILY_VALUES["fat_g"],              "g"),
        "Fiber":    (nutrition.get("fiber_g", 0) * scale,            DAILY_VALUES["fiber_g"],            "g"),
        "Sugar":    (nutrition.get("sugar_g", 0) * scale,            DAILY_VALUES["sugar_g"],            "g"),
        "Sodium":   (nutrition.get("sodium_mg", 0) * scale,          DAILY_VALUES["sodium_mg"],          "mg"),
    }

    names, pcts, bar_colors, labels = [], [], [], []
    for name, (val, dv, unit) in nutrients.items():
        pct = min((val / dv * 100) if dv > 0 else 0, 150)
        names.append(name)
        pcts.append(pct)
        labels.append(f"{val:.1f} {unit}")
        bar_colors.append(
            COLORS["protein"] if pct < 25 else
            COLORS["fiber"]   if pct < 60 else
            COLORS["fat"]
        )

    fig, ax = plt.subplots(figsize=(5.5, 4))
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")

    y = np.arange(len(names))
    bars = ax.barh(y, pcts, color=bar_colors,
                   edgecolor="white", linewidth=0.8, height=0.55)

    for bar, label in zip(bars, labels):
        ax.text(bar.get_width() + 1.5, bar.get_y() + bar.get_height() / 2,
                label, va="center", ha="left", fontsize=8, color="#555555")

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("% of daily value", fontsize=8)
    ax.set_xlim(0, 165)
    ax.axvline(x=100, color="#cccccc", linestyle="--", linewidth=1)
    ax.set_title(f"% Daily value  ({portion_g}g portion)",
                 fontsize=10, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="both", labelsize=8)
    plt.tight_layout()
    return fig


def plot_health_gauge(health_score: int) -> plt.Figure:
    """Semicircle gauge showing health score 1-10."""
    score = max(1, min(10, int(health_score)))

    fig, ax = plt.subplots(figsize=(3.5, 2.2),
                            subplot_kw={"projection": "polar"})
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")

    for start, end, color in [
        (0,          np.pi / 3,      "#E24B4A"),   # red  1-3
        (np.pi / 3,  2*np.pi / 3,   "#EF9F27"),   # amber 4-6
        (2*np.pi/3,  np.pi,          "#5DCAA5"),   # green 7-10
    ]:
        theta = np.linspace(start, end, 100)
        ax.plot(theta, [1] * 100, color=color,
                linewidth=12, solid_capstyle="round")

    # Map low scores to left and high scores to right.
    needle = ((10 - score) / 10) * np.pi
    ax.annotate("", xy=(needle, 0.9), xytext=(needle, 0.1),
                arrowprops={"arrowstyle": "->", "color": "#333333", "lw": 2.5})

    ax.text(np.pi / 2, 0.38, str(score), ha="center", va="center",
            fontsize=18, fontweight="bold", color="#333333",
            transform=ax.transData)

    ax.set_ylim(0, 1.3)
    ax.set_xlim(0, np.pi)
    ax.axis("off")
    ax.set_title(f"Health score: {score}/10",
                 fontsize=10, fontweight="bold", y=0.02)
    plt.tight_layout()
    return fig
