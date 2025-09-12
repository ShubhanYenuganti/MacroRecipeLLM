import requests
import json
import pandas as pd
from tqdm import tqdm
import time
import os
import ast

class MealCategoryPredictor:
    def __init__(self, model="llama2", api_url="http://localhost:11434"):
        self.model = model
        self.api_url = api_url
        
        self.meal_labels = [
            'Breakfast', 'Lunch', 'Dinner', 'Snack'
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
    
    def create_meal_prompt(self, recipe_row):
        name = str(recipe_row.get("Name", "")).strip()
        description = str(recipe_row.get("Description", "")).strip()
        category = str(recipe_row.get("RecipeCategory", "")).strip()
        
        keywords = recipe_row.get("Keywords", [])
        if isinstance(keywords, str):
            try:
                keywords = ast.literal_eval(keywords)
            except:
                keywords = [key.strip() for key in keywords.split(",")]
            
        
        prompt = f"""Classify the meal type of the following recipe. Choose ONLY ONE from this exact list: {', '.join(self.meal_labels)}
        Recipe Information:
        - Name: {name}
        - Description: {description}
        - Category: {category}
        - Keywords: {keywords}
        Based on the name, description, category and keywords describing the food, classify its meal type.
        Answer with just the cuisine name from the list above. If uncertain, choose the most likely option.
        """
        
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

    def extract_meal_type_from_response(self, response):
        response_clean = response.strip().replace(".",'').replace(",",'')
        for meal in self.meal_labels:
            if meal.lower() == response_clean.lower():
                return meal, 0.9
            elif meal.lower() in response_clean.lower():
                return meal, 0.7
        
        variations = {
            # breakfast
            "breakfast": "breakfast",
            "bfast": "breakfast",
            "bkfst": "breakfast",
            "brkfst": "breakfast",
            "brekkie": "breakfast",
            "breaky": "breakfast",
            "brunch": "breakfast",   # change to "lunch" if you prefer

            # lunch
            "lunch": "lunch",
            "luncheon": "lunch",
            "lunchtime": "lunch",

            # dinner
            "dinner": "dinner",
            "dinnertime": "dinner",
            "supper": "dinner",

            # snack
            "snack": "snack",
            "snacks": "snack",
            "snacking": "snack",
        }

        response_low = response_clean.lower()
        
        for variant, standard in variations.items():
            if variant in response_low:
                return standard, 0.6
        
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
            prompt = self.create_meal_prompt(recipe)
            
            prediction = self.query_ollama(prompt)
            meal_type, confidence = self.extract_meal_type_from_response(prediction)
            
            result = {
                "recipe_index": i,
                "meal_type": meal_type,
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