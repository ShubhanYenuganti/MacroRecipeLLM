import pandas as pd
import requests
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import re
from collections import defaultdict
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FastMealCategoryPredictor:
    def __init__(self, ollama_url="http://localhost:11434", model_name="llama3.2:3b", max_workers=8):
        self.ollama_url = ollama_url
        self.model_name = model_name
        self.max_workers = max_workers
        self.session = requests.Session()
        
        # Cache for similar recipes to avoid redundant API calls
        self.prediction_cache = {}
        
        # Predefined meal categories for consistent outputs
        self.meal_categories = ["Breakfast", "Lunch", "Dinner", "Snack"]
        
        # Fast keyword-based pre-filtering for obvious cases
        self.keyword_patterns = self._build_keyword_patterns()
        
    def _build_keyword_patterns(self):
        """Build regex patterns for fast meal category matching"""
        patterns = {
            "Breakfast": r'\b(breakfast|morning|cereal|oatmeal|pancake|waffle|french toast|eggs|bacon|sausage|muffin|bagel|toast|coffee|smoothie|omelet|frittata|granola|yogurt)\b',
            
            "Lunch": r'\b(lunch|sandwich|wrap|salad|soup|burger|panini|quesadilla|pasta salad|cold|light meal|midday)\b',
            
            "Dinner": r'\b(dinner|supper|evening|main course|entree|roast|steak|chicken breast|salmon|casserole|pasta dinner|rice bowl|hearty|family meal)\b',
            
            "Snack": r'\b(snack|appetizer|finger food|bite|chip|dip|nuts|trail mix|energy|bar|cookie|cracker|popcorn|pretzel|quick bite|on-the-go)\b'
        }
        return {meal: re.compile(pattern, re.IGNORECASE) for meal, pattern in patterns.items()}
    
    def calculate_total_time(self, row):
        """Calculate total time from prep time and cook time"""
        prep_time = 0
        cook_time = 0
        
        # Try different possible column names for prep time
        prep_columns = ['prep_time', 'PrepTime', 'Prep_Time', 'preparation_time', 'PrepTimeMinutes']
        for col in prep_columns:
            if col in row and pd.notna(row[col]):
                try:
                    prep_time = float(row[col])
                    break
                except (ValueError, TypeError):
                    continue
        
        # Try different possible column names for cook time
        cook_columns = ['cook_time', 'CookTime', 'Cook_Time', 'cooking_time', 'CookTimeMinutes', 'TotalTimeMinutes']
        for col in cook_columns:
            if col in row and pd.notna(row[col]):
                try:
                    cook_time = float(row[col])
                    break
                except (ValueError, TypeError):
                    continue
        
        # If we have a TotalTime column already, use that
        total_columns = ['TotalTime', 'total_time', 'Total_Time', 'TotalTimeMinutes']
        for col in total_columns:
            if col in row and pd.notna(row[col]):
                try:
                    return float(row[col])
                except (ValueError, TypeError):
                    continue
        
        return prep_time + cook_time
    
    def prefilter_obvious_meals(self, text):
        """Fast keyword-based meal detection for obvious cases"""
        
        # Keyword matching (most reliable)
        for meal, pattern in self.keyword_patterns.items():
            if pattern.search(text):
                return meal
        

        return None
    
    def create_cache_key(self, name, total_time, keywords):
        """Create a simple cache key for similar recipes"""
        # Use first few words of name + time range + first keyword
        name_words = ' '.join(name.lower().split()[:3]) if name else ''
        time_range = self.get_time_range(total_time)
        first_keyword = keywords.split()[0].lower() if keywords.strip() else ''
        return f"{name_words}|{time_range}|{first_keyword}"
    
    def get_time_range(self, total_time):
        """Convert total time to range category for caching"""
        if total_time <= 15:
            return "quick"
        elif total_time <= 30:
            return "short"
        elif total_time <= 60:
            return "medium"
        elif total_time <= 120:
            return "long"
        else:
            return "very_long"
    
    def prepare_batch_data(self, df):
        """Prepare data in batches with pre-filtering"""
        batches = []
        cache_hits = 0
        prefiltered = 0
        
        logger.info("Preparing batches with pre-filtering...")
        
        for idx, row in df.iterrows():
            name = str(row.get('Name', '')).strip()
            description = str(row.get('Description', '')).strip()[:200]  # Limit description length
            keywords = str(row.get('Keywords', ''))
            total_time = float(row.get('TotalTime', 0))
            
            # Store total_time back to dataframe
            df.at[idx, 'TotalTime'] = total_time
            
            # Create combined text for analysis
            combined_text = f"{name} {description} {keywords}"
            
            # Try cache first
            cache_key = self.create_cache_key(name, total_time, keywords)
            if cache_key in self.prediction_cache:
                df.at[idx, 'predicted_meal_category'] = self.prediction_cache[cache_key]
                cache_hits += 1
                continue
            
            # Try keyword pre-filtering
            obvious_meal = self.prefilter_obvious_meals(combined_text)
            if obvious_meal:
                df.at[idx, 'predicted_meal_category'] = obvious_meal
                self.prediction_cache[cache_key] = obvious_meal
                prefiltered += 1
                continue
            
            # Add to batch for Ollama processing
            batch_item = {
                'index': idx,
                'name': name,
                'description': description,
                'keywords': keywords[:150],  # Limit keywords length
                'total_time': total_time,
                'cache_key': cache_key,
                'text': combined_text[:400]  # Shorter for meal category task
            }
            batches.append(batch_item)
        
        logger.info(f"Cache hits: {cache_hits}, Pre-filtered: {prefiltered}, Ollama needed: {len(batches)}")
        return batches
    
    def create_optimized_prompt(self, batch_items):
        """Create a batch prompt for multiple recipes focused on meal categories"""
        
        prompt = f"""Classify these recipes into meal categories: {', '.join(self.meal_categories)}

Consider these factors:
- Breakfast: Morning foods, quick preparation, cereals, eggs, pastries, coffee drinks
- Lunch: Midday meals, sandwiches, salads, light to moderate preparation time
- Dinner: Evening meals, hearty portions, main courses, longer cooking times, family meals
- Snack: Small portions, finger foods, quick bites, minimal preparation

Use the recipe name, description, keywords, and total cooking time as context.

For each recipe, respond with only the meal category (one word).

Recipes:
"""
        
        for i, item in enumerate(batch_items, 1):
            time_info = f" ({int(item['total_time'])}min)" if item['total_time'] > 0 else ""
            keywords_short = item['keywords'][:80] + "..." if len(item['keywords']) > 80 else item['keywords']
            prompt += f"{i}. {item['name']}{time_info} - {keywords_short}\n"
        
        prompt += "\nRespond with only the meal categories, one per line:"
        
        return prompt
    
    def call_ollama_batch(self, batch_items, batch_size=10):
        """Call Ollama with batch of recipes"""
        try:
            prompt = self.create_optimized_prompt(batch_items)
            
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,  # Low temperature for consistent results
                    "top_p": 0.9,
                    "top_k": 10,
                    "num_predict": batch_size * 2,  # Limit response length
                }
            }
            
            response = self.session.post(
                f"{self.ollama_url}/api/generate",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()['response'].strip()
                return self.parse_batch_response(result, batch_items)
            else:
                logger.error(f"Ollama API error: {response.status_code}")
                return self.fallback_predictions(batch_items)
                
        except Exception as e:
            logger.error(f"Batch processing error: {e}")
            return self.fallback_predictions(batch_items)
    
    def parse_batch_response(self, response_text, batch_items):
        """Parse batch response and match with input items"""
        lines = [line.strip() for line in response_text.split('\n') if line.strip()]
        predictions = []
        
        for i, item in enumerate(batch_items):
            if i < len(lines):
                # Clean and validate prediction
                prediction = self.clean_prediction(lines[i])
                if prediction in self.meal_categories:
                    predictions.append((item, prediction))
                else:
                    predictions.append((item, self.keyword_based_fallback(item['text'])))
            else:
                predictions.append((item, self.keyword_based_fallback(item['text'])))
        
        return predictions
    
    def clean_prediction(self, prediction):
        """Clean and standardize prediction"""
        prediction = prediction.strip().title()
        
        # Handle common variations
        mapping = {
            "Morning": "Breakfast",
            "Evening": "Dinner", 
            "Midday": "Lunch",
            "Appetizer": "Snack",
            "Dessert": "Snack",
            "Side": "Snack"
        }
        
        return mapping.get(prediction, prediction)
    
    def keyword_based_fallback(self, text):
        """Fallback prediction based purely on keywords - no calories"""
        text = text.lower()
        
        # Keyword-based fallback only
        if any(word in text for word in ['breakfast', 'morning', 'cereal', 'eggs', 'pancake', 'waffle', 'toast', 'coffee', 'smoothie', 'omelet', 'muffin', 'bagel', 'yogurt', 'granola']):
            return "Breakfast"
        elif any(word in text for word in ['lunch', 'sandwich', 'wrap', 'salad', 'soup', 'burger', 'panini', 'quesadilla', 'midday']):
            return "Lunch"
        elif any(word in text for word in ['dinner', 'supper', 'evening', 'main', 'entree', 'roast', 'steak', 'salmon', 'casserole', 'hearty', 'family']):
            return "Dinner"
        elif any(word in text for word in ['snack', 'appetizer', 'bite', 'chip', 'dip', 'nuts', 'trail', 'energy', 'bar', 'cookie', 'cracker', 'popcorn', 'pretzel']):
            return "Snack"
        else:
            # Default fallback when no keywords match
            return "Lunch"  # Most neutral category as default
    
    def fallback_predictions(self, batch_items):
        """Fallback predictions based on keywords only - no calorie logic"""
        predictions = []
        for item in batch_items:
            text = item['text']
            meal = self.keyword_based_fallback(text)
            predictions.append((item, meal))
        
        return predictions
    
    def process_batch_worker(self, batch):
        """Worker function for parallel processing"""
        return self.call_ollama_batch(batch)
    
    def predict_meal_categories(self, df, batch_size=10):
        """Main function to predict meal categories for entire dataset"""
        
        logger.info(f"Starting meal category prediction for {len(df)} recipes...")
        start_time = time.time()
        
        # Initialize prediction column
        df['predicted_meal_category'] = "Unknown"
        
        # Prepare batches with pre-filtering
        ollama_batches = self.prepare_batch_data(df)
        
        if not ollama_batches:
            logger.info("All recipes handled by cache/pre-filtering!")
            return df
        
        # Group into batches for Ollama
        batches = [ollama_batches[i:i + batch_size] for i in range(0, len(ollama_batches), batch_size)]
        
        logger.info(f"Processing {len(batches)} batches with Ollama...")
        
        # Process batches in parallel
        processed_count = 0
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_batch = {executor.submit(self.process_batch_worker, batch): batch for batch in batches}
            
            for future in as_completed(future_to_batch):
                try:
                    predictions = future.result()
                    
                    # Update dataframe with predictions
                    for item, meal_category in predictions:
                        df.at[item['index'], 'predicted_meal_category'] = meal_category
                        self.prediction_cache[item['cache_key']] = meal_category
                    
                    processed_count += len(predictions)
                    
                    if processed_count % (batch_size * 10) == 0:  # Progress update every 100 items
                        elapsed = time.time() - start_time
                        rate = processed_count / elapsed
                        logger.info(f"Processed {processed_count}/{len(ollama_batches)} items. Rate: {rate:.1f} items/sec")
                
                except Exception as e:
                    logger.error(f"Batch processing failed: {e}")
                    # Handle failed batch with keyword-based fallbacks only
                    batch = future_to_batch[future]
                    for item in batch:
                        fallback = self.keyword_based_fallback(item['text'])
                        df.at[item['index'], 'predicted_meal_category'] = fallback
        
        total_time = time.time() - start_time
        logger.info(f"Completed meal category prediction in {total_time:.2f} seconds")
        logger.info(f"Average rate: {len(df) / total_time:.1f} recipes/second")
        
        return df

# Usage function
def predict_meal_categories(input_csv_path, output_csv_path, model_name="llama3.2:3b"):
    """
    Main function to predict meal categories for recipe dataset
    
    Args:
        input_csv_path: Path to input CSV with recipe data
        output_csv_path: Path to save results
        model_name: Ollama model to use (3b model recommended for speed)
    """
    
    # Load data
    logger.info(f"Loading dataset from {input_csv_path}")
    df = pd.read_csv(input_csv_path)
    
    # Initialize predictor
    predictor = FastMealCategoryPredictor(model_name=model_name, max_workers=8)
    
    # Run predictions
    df_with_predictions = predictor.predict_meal_categories(df, batch_size=12)
    
    # Save results
    df_with_predictions.to_csv(output_csv_path, index=False)
    logger.info(f"Results saved to {output_csv_path}")
    
    # Print summary
    meal_counts = df_with_predictions['predicted_meal_category'].value_counts()
    logger.info("Meal category distribution:")
    for meal, count in meal_counts.items():
        logger.info(f"  {meal}: {count} recipes ({count/len(df)*100:.1f}%)")
    
    # Show some examples by time ranges
    logger.info("\nSample predictions by time range:")
    for category in ["Breakfast", "Lunch", "Dinner", "Snack"]:
        sample = df_with_predictions[df_with_predictions['predicted_meal_category'] == category].head(3)
        logger.info(f"\n{category} examples:")
        for _, row in sample.iterrows():
            total_time = row.get('TotalTime', 0)
            logger.info(f"  - {row.get('Name', 'Unknown')} ({int(total_time)} min)")
    
    return df_with_predictions

if __name__ == "__main__":
    # Configuration
    INPUT_CSV = "/Users/shubhan/MacroRecipe/MacroRecipeLLM/recipes_with_cuisine_predictions.csv"
    OUTPUT_CSV = "/Users/shubhan/MacroRecipe/MacroRecipeLLM/ready_for_embeddings.csv"
    MODEL = "llama3.2:3b"  # Fast 3B model
    
    # Run prediction
    result_df = predict_meal_categories(INPUT_CSV, OUTPUT_CSV, MODEL)