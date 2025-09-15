import pandas as pd
import numpy as np
import json
import ast
from datetime import datetime
import isodate
import re

def preprocess(csv_path):
    """Process the Food.com dataset from a CSV file with enhanced features"""
    
    df = pd.read_csv(csv_path)
    
    # drop rows
    df.drop(columns=['AuthorId', 'AuthorName', 'RecipeYield'], inplace=True)
    
    # drop NAs
    df = df.dropna().copy()
    
    # refactor cooktime, preptime, totaltime
    for col in ["CookTime", "PrepTime", "TotalTime"]:
        df[col] = df[col].astype(str).apply(duration_minutes).astype("Int64")
    
    # convert list items
    for col in ["Images", "Keywords", "RecipeIngredientQuantities", "RecipeIngredientParts", "RecipeInstructions"]:
        df[col] = df[col].apply(r_vector_to_list)

    # JSON-encode list columns so they round-trip when saving to CSV
    for col in ["Images", "Keywords", "RecipeIngredientQuantities", "RecipeIngredientParts", "RecipeInstructions"]:
        df[col] = df[col].apply(json.dumps)
    
    # convert appropriate objects to strings
    for col in ["Name", "Description", "RecipeCategory"]:
        df[col] = df[col].astype(str)
        
    return df
    
def duration_minutes(x):
    if str(x).strip() == "":
        return pd.NA
    try:
        d = isodate.parse_duration(str(x))
        secs = d.total_seconds() if hasattr(d, "total_seconds") else d.totimedelta().total_seconds()
        return round(secs / 60)
    except Exception:
        return pd.NA

def r_vector_to_list(x):
    if pd.isna(x):
        return []
    items = re.findall(r'"(.*?)"', str(x))
    return [str(i) for i in items]  


df = preprocess('recipes.csv')
df.to_csv("recipes_clean.csv", index=False)