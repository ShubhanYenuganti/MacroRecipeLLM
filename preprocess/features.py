import json
import math
import pandas as pd
import numpy as np
import re

LIST_COLS = [
    "Images",
    "Keywords",
    "RecipeIngredientQuantities",
    "RecipeIngredientParts",
    "RecipeInstructions",
]

def coerce_list(x):
    """Return a Python list for many possible encodings; NaN -> []"""
    if x is None:
        return []
    # pandas NaN is a float
    if isinstance(x, float) and math.isnan(x):
        return []
    s = x if isinstance(x, str) else str(x)

    s_stripped = s.strip()
    if s_stripped == "":
        return []

    # JSON array like ["a","b"]
    if s_stripped.startswith("[") and s_stripped.endswith("]"):
        try:
            return json.loads(s_stripped)
        except Exception:
            pass

    # R vector like c("a","b")
    if s_stripped.startswith("c("):
        return re.findall(r'"(.*?)"', s_stripped)

    # Already a Python-list-like string '["a","b"]' or "['a','b']"
    # last resort: try json, then regex for quoted strings
    try:
        return json.loads(s_stripped.replace("'", '"'))
    except Exception:
        pass

    # If nothing matched, treat as single-item list
    return [s_stripped]

def feature_eng(csv_path):
    df = pd.read_csv(csv_path)

    # Ensure list-like columns become real Python lists
    for col in LIST_COLS:
        if col in df.columns:
            df[col] = df[col].apply(coerce_list)

    # If TotalTime is in minutes already, keep it; else coerce numeric minutes
    if "TotalTime" in df.columns:
        df["TotalTime"] = pd.to_numeric(df["TotalTime"], errors="coerce")

    # Calculate features
    df["complexity_score"] = calculate_complexity(df)
    df["batch_friendliness_score"] = calculate_batch_friendliness(df)
    df["difficulty_level"] = categorize_difficulty(df)

    # Nutrition per serving (avoid div-by-zero)
    for num, newcol in [
        ("Calories", "calories_per_serving"),
        ("FatContent", "fat_per_serving"),
        ("SaturatedFatContent", "saturated_fat_per_serving"),
        ("CholesterolContent", "cholesterol_per_serving"),
        ("SodiumContent", "sodium_per_serving"),
        ("CarbohydrateContent", "carbohydrate_per_serving"),
        ("FiberContent", "fiber_per_serving"),
        ("SugarContent", "sugar_per_serving"),
        ("ProteinContent", "protein_per_serving"),
    ]:
        if num in df.columns and "RecipeServings" in df.columns:
            df[newcol] = (
                pd.to_numeric(df[num], errors="coerce") /
                pd.to_numeric(df["RecipeServings"], errors="coerce").replace(0, np.nan)
            )

    # Quality score
    df["quality_score"] = calculate_quality_score(df)

    return df

def calculate_complexity(df):
    scores = []
    for _, recipe in df.iterrows():
        ingredients = recipe.get("RecipeIngredientQuantities", [])
        instructions = recipe.get("RecipeInstructions", [])
        # Guard types
        if not isinstance(ingredients, list):
            ingredients = coerce_list(ingredients)
        if not isinstance(instructions, list):
            instructions = coerce_list(instructions)

        ingredient_count = len(ingredients)
        ingredient_score = min(ingredient_count / 15, 1.0)

        instruction_count = len(instructions)
        instruction_score = min(instruction_count / 20, 1.0)

        total_time = pd.to_numeric(recipe.get("TotalTime", np.nan), errors="coerce")
        time_score = min((total_time or 0) / 180, 1.0) if pd.notna(total_time) else 0.0

        techniques_score = count_cooking_techniques(instructions)

        complexity = (
            0.3 * ingredient_score +
            0.3 * instruction_score +
            0.2 * time_score +
            0.2 * techniques_score
        )
        scores.append(complexity)
    return scores

def count_cooking_techniques(instructions):
    complex_techniques = [
        "sauté","braise","sear","deglaze","julienne","blanch",
        "poach","broil","grill","steam","roast","bake",
        "whisk","fold","knead","caramelize"
    ]
    text = " ".join(instructions).lower()
    technique_count = sum(1 for t in complex_techniques if t in text)
    return min(technique_count / 5, 1.0)

def categorize_difficulty(df):
    levels = []
    for _, r in df.iterrows():
        c = r.get("complexity_score", 0.5)
        if c < 0.3: levels.append("Easy")
        elif c < 0.6: levels.append("Medium")
        elif c < 0.8: levels.append("Hard")
        else: levels.append("Expert")
    return levels

def calculate_batch_friendliness(df):
    scores = []
    for _, recipe in df.iterrows():
        instructions = recipe.get("RecipeInstructions", [])
        if not isinstance(instructions, list):
            instructions = coerce_list(instructions)
        text = " ".join(instructions).lower()

        score = 0.5
        pos = ["double","triple","freeze","store","reheat","make ahead","batch","large batch"]
        neg = ["immediately","fresh","serve hot","don't store","tempura","soufflé","meringue"]

        score += sum(0.1 for term in pos if term in text)
        score -= sum(0.1 for term in neg if term in text)

        category = str(recipe.get("RecipeCategory", "")).lower()
        if any(k in category for k in ["soup","stew","casserole","curry"]):
            score += 0.2
        elif any(k in category for k in ["dessert","bread","cake"]):
            score += 0.1

        scores.append(max(0, min(1, score)))
    return scores

def calculate_quality_score(df):
    if "AggregatedRating" not in df.columns or "ReviewCount" not in df.columns:
        return pd.Series([np.nan] * len(df))
    rating_norm = pd.to_numeric(df["AggregatedRating"], errors="coerce") / 5.0
    review_log = np.log1p(pd.to_numeric(df["ReviewCount"], errors="coerce"))
    denom = review_log.max() if pd.notna(review_log.max()) and review_log.max() > 0 else 1
    review_norm = review_log / denom
    return (rating_norm * 0.7 + review_norm * 0.3)

# Usage:
df = feature_eng('recipes_clean.csv')  # Prefer the cleaned CSV
df.to_csv('recipes_feature_eng.csv', index=False)
