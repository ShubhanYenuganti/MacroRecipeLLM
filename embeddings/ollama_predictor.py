import requests
import json
import pandas as pd
from tqdm import tqdm
import time
import os
import ast

class OllamaCuisinePredictor:
    def __init__(self, model="llama2", api_url="http://localhost:11434"):
        self.model = model
        self.api_url = api_url
        
        self.cuisine_labels = [
            'Italian', 'Mexican', 'Chinese', 'Indian', 'American', 
            'French', 'Thai', 'Mediterranean', 'Japanese', 'Korean',
            'Middle Eastern', 'German', 'British', 'Spanish', 'Greek',
            'Vietnamese', 'Lebanese', 'Moroccan', 'Brazilian', 'Russian'
        ]
    
    def check_status(self):
        try:
            response = requests.get(f"{self.api_url}/api/tags")
            if response.status_code == 200:
                models = [model['name'] for model in response.json()['models']]
                if self.model in models:
                    print(f"Ollama is running with {self.model} model.")
                    return True
                else:
                    print(f"Model {self.model} not found. Available models: {models}")
                    return False
            else:
                print("Ollama API is not reachable.")
                return False
        except requests.exceptions.ConnectionError:
            print("Cannot connect to Ollama. Is it running?")
            print("Start Ollama: ollama serve")
            return False

    def create_cuisine_prompt(self, recipe_row):
        name = str(recipe_row.get("Name", "")).strip()
        description = str(recipe_row.get("Description", "")).strip()
        category = str(recipe_row.get("RecipeCategory", "")).strip()
        
        keywords = recipe_row.get("Keywords", [])
        if isinstance(keywords, str):
            try:
                keywords = ast.literal_eval(keywords)
            except:
                keywords = [key.strip() for key in keywords.split(",")]
            
        ingredients = recipe_row.get('RecipeIngredientParts', [])
        if isinstance(ingredients, str):
            try:
                ingredients = ast.literal_eval(ingredients)
            except:
                ingredients = [ing.strip() for ing in ingredients.split(',')]
        
        key_ingredients = ', '.join([str(ing).strip() for ing in ingredients[:8]])

        prompt = f"""Classify the cuisine style of this recipe. Choose ONLY ONE from this exact list:
{', '.join(self.cuisine_labels)}

Recipe Information:
- Name: {name}
- Description: {description}  
- Category: {category}
- Key Ingredients: {key_ingredients}
- Keywords: {keywords}

Based on the ingredients, cooking style, and recipe name, this recipe's cuisine is:

Answer with just the cuisine name from the list above. If uncertain, choose the most likely option."""
        return prompt
    
    def query_ollama(self, prompt, max_retries=3):
        for attempt in range(max_retries):
            try:
                payload = {
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1, 
                        "top_p": 0.9, 
                        "max_tokens": 50
                    }
                }
                
                response = requests.post(
                    f"{self.api_url}/api/generate",
                    json=payload,
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return result['response'].strip()
                else:
                    print(f"API error {response.status_code}: {response.text}")
            except requests.exceptions.Timeout:
                print(f"Timeout on attempt {attempt + 1}")
            except Exception as e:
                print(f"Error on attempt {attempt + 1}: {e}")
            
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
        
        return "Error"
    
    def extract_cuisine_from_response(self, response):
        response_clean = response.strip().replace(".",'').replace(',','')
        for cuisine in self.cuisine_labels:
            if cuisine.lower() == response_clean.lower():
                return cuisine, 0.9
            elif cuisine.lower() in response_clean.lower():
                return cuisine, 0.7
        
        variations = {
            'italian': 'Italian',
            'mexico': 'Mexican',
            'china': 'Chinese',
            'india': 'Indian',
            'america': 'American',
            'france': 'French',
            'thailand': 'Thai',
            'asia': 'Chinese',
            'mediterranean': 'Mediterranean'
        }
        
        response_low = response_clean.lower()
        for variant, cuisine in variations.items():
            if variant in response_low:
                return cuisine, 0.6
        
        return "Unknown", 0.0
    
    def predict_batch(self, df, save_progress=True):
        results = []
        progress_file = f"ollama_progress_{self.model.replace(':', '_')}.json"
        start_idx = 0
        if save_progress and os.path.exists(progress_file):
            try:
                with open(progress_file, 'r') as f:
                    existing_results = json.load(f)
                results = existing_results
                start_idx = len(existing_results)
                print(f"Resuming from index {start_idx}")
            except:
                print("Error loading progress file. Starting from scratch.")
        
        for i in tqdm(range(start_idx, len(df)), desc=f"Classifying with {self.model}"):
            recipe = df.iloc[i]
            prompt = self.create_cuisine_prompt(recipe)
            
            prediction = self.query_ollama(prompt)
            cuisine, confidence = self.extract_cuisine_from_response(prediction)
            
            result = {
                "recipe_index": i,
                "cuisine": cuisine,
                "confidence": confidence,
                "raw_response": prediction,
                "recipe_name": recipe.get("Name", "")
            }
            
            results.append(result)
            
            if save_progress:
                with open(progress_file, 'w') as f:
                    json.dump(results, f, indent=2)
                print(f"Progress saved at index {i + 1}")
            
            time.sleep(0.1)
        
        return results
