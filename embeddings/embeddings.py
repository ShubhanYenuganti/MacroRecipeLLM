import pandas as pd
import numpy as np
import json

def transform_for_bedrock_kb(csv_path, output_csv_path):
    """Transforms CSV of recipes into an embedding friendly format for Bedrock Knowledge Base format"""
    
    df = pd.read_csv(csv_path)
    
    # embedding text fields -- semantic_text, nutritional_text, procedural_text
    # semantic_text -- name, description, cuisine style, category, keywords, etc.
    df['semantic_text'] = df.apply(create_semantic_text, axis = 1)
    df['nutritional_text'] = df.apply(create_nutritional_text, axis = 1)
    df['procedural_text'] = df.apply(create_procedural_text, axis = 1)
    
    # create comprehensive embedding text
    df['comprehensive_text'] = df.apply(create_comprehensive_text, axis=1)
    
    df['metadata'] = df.apply(create_metadata_json, axis=1)
    
    # Select columns for Bedrock KB
    kb_df = df[[
        'RecipeId',           # Unique identifier
        'Name',               # Recipe name  
        'comprehensive_text', # Main text for embedding
        'metadata'           # Structured metadata
    ]].copy()
    
    # Rename columns to Bedrock KB conventions
    kb_df.columns = ['recipe_id', 'title', 'text', 'metadata']
    
    # Save the transformed CSV
    kb_df.to_csv(output_csv_path, index=False)
    
    print(f"Transformed {len(df)} recipes for Bedrock Knowledge Base")
    print(f"Output saved to: {output_csv_path}")
    
    return kb_df

def create_semantic_text(row):
    """Creates semantic embedding text focusing on recipe identity"""
    
    category = str(row.get('category', 'Unknown')).title()
    description = str(row.get('Description', ''))[:200] # Truncate long descriptions
    keywords = str(row.get('Keywords', ''))
    flavor_profile = str(row.get('flavor_profile', '')).title()
    dietary_tags = create_dietary_tags(row)
    
    text = f"""
    Recipe: {row['name']}
    Category: {category}
    Description: {description}
    Keywords: {keywords}
    Cuisine Style: {row['predicted_cuisine']}
    Dietary Profile: {', '.join(dietary_tags)}
    Flavor Characteristics: {flavor_profile}
    Meal Type: {row['predicted_meal_category']}
    """.strip()
    
    return text

def create_nutritional_text(row):
    """Creates nutritional embedding text focusing on diet compatibility"""
    
    calories = row.get('calories_per_serving', 0)
    protein = row.get('protein_per_serving', 0)
    carbs = row.get('carbs_per_serving', 0)
    fat = row.get('fat_per_serving', 0)

    calorie_category = categorize_calories(calories)
    protein_category = categorize_protein(protein)
    carb_category = categorize_carbs(carbs)
    fat_category = categorize_fats(fat)
    
    diet_compatibility = determine_diet_fit(row)
    macro_ratio = describe_macro_ratio(protein, carbs, fat)
    
    text = f"""
    Nutritional Profile: {row['Name']}
    Calories per serving: {calories:.0f} ({calorie_category})
    Protein: {protein:.1f}g ({protein_category})
    Carbohydrates: {carbs:.1f}g ({carb_category})  
    Fat: {fat:.1f}g ({fat_category})
    Macro Ratio: {macro_ratio}
    Diet Compatibility: {', '.join(diet_compatibility)}
    Nutritional Density: {calculate_nutrient_density(row)}
    Serving Size: {row.get('RecipeServings', 1)} servings
    """.strip()
    
    return text

def create_procedural_text(row):
    """Creates procedural embedding text focusing on cookin process"""
    
    total_time = row.get('TotalTime', 0)
    difficulty = row.get("difficulty_level", "Medium")
    complexity = row.get("complexity_score", 0.5)
    batch_friendliness = row.get("batch_friendliness_score", 0.5)
    
    time_category = categorize_cooking_time(total_time)
    complexity_desc = describe_complexity(complexity)
    batch_friendly = "Yes" if batch_friendliness > 0.7 else "Moderate" if batch_friendliness > 0.4 else "No"
    
    text = f"""
    Cooking Process: {row['Name']}
    Total Time: {total_time:.0f} minutes ({time_category})
    Difficulty Level: {difficulty}
    Complexity: {complexity_desc} (Score: {complexity:.2f})
    Batch Cooking Friendly: {batch_friendly}
    Skill Level Required: {determine_skill_level(complexity, difficulty)}
    """.strip()
    
    return text

def create_comprehensive_text(row):
    """Combine all aspects into one comprehensive text for embedding"""
    
    semantic = create_semantic_text(row)
    nutritional = create_nutritional_text(row)
    procedural = create_procedural_text(row)
    
    comprehensive = f"""
    {semantic}
    
    {nutritional}
    
    {procedural}
    
    Quality Score: {row.get('quality_score', 0.5):.2f}
    Overall Rating: {row.get('AggregatedRating', 0):.1f}/5.0
    Number of Reviews: {row.get('ReviewCount', 0)} reviews
    """.strip()
    
    return comprehensive
   
def create_metadata_json(row):
    """Create structured metadata for filtering and display"""
    
    metadata = {
        "recipe_id": str(row.get('RecipeId', '')),
        "name": str(row.get('Name', '')),
        "category": str(row.get('RecipeCategory', '')),
        "servings": int(row.get('RecipeServings', 1)),
        
        # Timing information
        "total_time_minutes": float(row.get('TotalTime', 0)),
        "prep_time": float(row.get('PrepTime', 0)),
        "cook_time": float(row.get('CookTime', 0)),
        
        # Difficulty and complexity
        "difficulty_level": str(row.get('difficulty_level', 'Medium')),
        "complexity_score": float(row.get('complexity_score', 0.5)),
        
        # Nutritional information
        "calories_per_serving": float(row.get('calories_per_serving', 0)),
        "protein_per_serving": float(row.get('protein_per_serving', 0)),
        "carbs_per_serving": float(row.get('carbs_per_serving', 0)),
        "fat_per_serving": float(row.get('fat_per_serving', 0)),
        
        # Quality indicators
        "quality_score": float(row.get('quality_score', 0.5)),
        "popularity_score": float(row.get('popularity_score', 0.5)),
        "rating": float(row.get('AggregatedRating', 0)),
        "review_count": int(row.get('ReviewCount', 0)),
        
        # Batch cooking
        "batch_friendly_score": float(row.get('batch_friendly_score', 0.5)),
        "batch_friendly": row.get('batch_friendly_score', 0.5) > 0.7,
        
        # Dietary tags
        "dietary_tags": create_dietary_tags(row),
        "diet_compatibility": determine_diet_fit(row)
    }
    
    return json.dumps(metadata)

def create_dietary_tags(row):
    tags = []
    
    # Protein-based tags
    if row.get('protein_per_serving', 0) > 20:
        tags.append("High Protein")
    
    # Carb-based tags  
    if row.get('carbs_per_serving', 0) < 20:
        tags.append("Low Carb")
    elif row.get('carbs_per_serving', 0) < 10:
        tags.append("Keto Friendly")
        
    # Fat-based tags
    if row.get('fat_per_serving', 0) < 10:
        tags.append("Low Fat")
        
    # Calorie-based tags
    if row.get('calories_per_serving', 0) < 300:
        tags.append("Light")
    elif row.get('calories_per_serving', 0) > 600:
        tags.append("Hearty")
        
    return tags

# Helper functions for categorization
def categorize_calories(calories):
    if calories < 200: return "Very Low Calorie"
    elif calories < 400: return "Low Calorie" 
    elif calories < 600: return "Moderate Calorie"
    elif calories < 800: return "High Calorie"
    else: return "Very High Calorie"

def categorize_protein(protein):
    if protein < 10: return "Low Protein"
    elif protein < 20: return "Moderate Protein"
    elif protein < 30: return "High Protein"
    else: return "Very High Protein"

def categorize_carbs(carbs):
    if carbs < 15: return "Very Low Carb"
    elif carbs < 30: return "Low Carb"
    elif carbs < 45: return "Moderate Carb"
    else: return "High Carb"

def categorize_fats(fat):
    if fat < 5: return "Very Low Fat"
    elif fat < 15: return "Low Fat"
    elif fat < 25: return "Moderate Fat"
    else: return "High Fat"

def categorize_cooking_time(minutes):
    if minutes < 20: return "Quick"
    elif minutes < 45: return "Moderate"
    elif minutes < 90: return "Long"
    else: return "Extended"

def determine_diet_fit(row):
    """Determine which diets this recipe might fit"""
    diets = []
    
    protein = row.get('protein_per_serving', 0)
    carbs = row.get('carbs_per_serving', 0)
    fat = row.get('fat_per_serving', 0)
    calories = row.get('calories_per_serving', 0)
    
    # High protein diets
    if protein > 25:
        diets.extend(["High Protein", "Muscle Building"])
    
    # Low carb diets
    if carbs < 20:
        diets.append("Low Carb")
    if carbs < 10:
        diets.append("Keto")
        
    # Weight management
    if calories < 400:
        diets.append("Weight Loss")
        
    # Balanced diets
    if 15 <= protein <= 30 and 20 <= carbs <= 50 and 10 <= fat <= 25:
        diets.append("Balanced")
        
    return diets

def describe_macro_ratio(protein, carbs, fat):
    total = protein + carbs + fat
    if total == 0:
        return "No Macros"
    
    p_pct = (protein / total) * 100
    c_pct = (carbs / total) * 100
    f_pct = (fat / total) * 100
    
    return f"P:{p_pct:.0f}%, C:{c_pct:.0f}%, F:{f_pct:.0f}%"

def calculate_nutrient_density(row):
    """
    Calculates a nutrient density score (0–1) for each row of a nutrition DataFrame.
    """
    # Define weights (positive = beneficial, negative = limiting)
    weights = {
        'protein_per_serving':  0.3,
        'fiber_per_serving':    0.2,
        'carbohydrate_per_serving': 0.05,  # light contribution
        'calories_per_serving':       -0.1,
        'fat_per_serving':     -0.1,
        'saturated_fat_per_serving': -0.1,
        'cholesterol_per_serving':  -0.05,
        'sodium_per_serving':  -0.05,
        'sugar_per_serving':   -0.1,
    }
    
    # Min-max normalize each nutrient column
    normed = pd.DataFrame()
    for col in weights:
        col_min, col_max = row[col].min(), row[col].max()
        if col_max > col_min:  # avoid div0
            normed[col] = (row[col] - col_min) / (col_max - col_min)
        else:
            normed[col] = 0.0
    
    # Apply weights
    scores = np.zeros(len(row))
    for col, w in weights.items():
        scores += normed[col] * w
    
    # Shift to positive range if negatives exist
    min_score, max_score = scores.min(), scores.max()
    if max_score > min_score:
        scores = (scores - min_score) / (max_score - min_score)
    else:
        scores = np.ones(len(row))  # all identical rows
    
    return scores

def categorize_cooking_time(minutes):
    if minutes < 20: return "Quick"
    elif minutes < 45: return "Moderate"
    elif minutes < 90: return "Long"
    else: return "Extended"
    
def describe_complexity(score):
    if score < 0.3: return "Simple"
    elif score < 0.6: return "Moderate"
    else: return "Complex"

def determine_skill_level(complexity, difficulty):
    if complexity < 0.3 and difficulty == "Easy":
        return "Beginner"
    elif complexity < 0.6 or difficulty == "Medium":
        return "Intermediate"
    else:
        return "Advanced"
