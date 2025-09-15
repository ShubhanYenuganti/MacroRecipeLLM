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
logging.basicConfig(level = logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CuisinePredictor:
    def __init__(self, ollama_url = "http://localhost:11434", model_name="llama3.2:3b", max_workers = 8):
        self.ollama_url = ollama_url
        self.model_name = model_name
        self.max_workers = max_workers
        self.session = requests.Session()
        
        # Cache for similar recipes to avoid redundant API calls
        self.prediction_cache = {}

        # Predefined cuisine categories for consistent outputs
        self.cuisine_categories = [
            "American", "Italian", "Mexican", "Chinese", "Japanese", "Thai", "Indian", 
            "French", "Greek", "Mediterranean", "Middle Eastern", "Korean", "Vietnamese",
            "British", "German", "Spanish", "Portuguese", "Brazilian", "Cajun",
            "Caribbean", "African", "Fusion", "International"
        ]
        
        # Fast keyword-based pre-filtering for obvious cases
        self.keyword_patterns = self._build_keyword_patterns()
    
    def _build_keyword_patterns(self):
        """Build regex patterns for fast keyword matching"""
        patterns = {
            "Italian": r'\b(pasta|pizza|risotto|lasagna|parmesan|marinara|alfredo|pesto|caprese|bruschetta)\b',
            "Mexican": r'\b(tacos?|burrito|quesadilla|salsa|guacamole|enchilada|fajita|tortilla|jalapeño|cilantro)\b',
            "Chinese": r'\b(stir.?fry|wok|soy sauce|bok choy|dim sum|kung pao|sweet and sour|lo mein)\b',
            "Japanese": r'\b(sushi|sashimi|teriyaki|tempura|miso|ramen|udon|wasabi|sake|bento)\b',
            "Thai": r'\b(pad thai|curry|coconut milk|lemongrass|basil|lime|fish sauce|tom yum)\b',
            "Indian": r'\b(curry|naan|basmati|masala|tandoor|turmeric|cumin|cardamom|biryani|dal)\b',
            "French": r'\b(coq au vin|ratatouille|croissant|baguette|brie|bordeaux|provence|burgundy)\b',
            "Greek": r'\b(feta|olive|tzatziki|gyros?|moussaka|spanakopita|oregano|mediterranean)\b',
        }
        return {cuisine: re.compile(pattern, re.IGNORECASE) for cuisine, pattern in patterns.items()}

    def prefilter_obvious_cuisines(self, text):
        """Fast keyword-based cuisine detection for obvious cases"""
        for cuisine, pattern in self.keyword_patterns.items():
            if pattern.search(text):
                return cuisine
        return None
    
    def create_cache_key(self, name, category):
        """Create a simple cache key for similar recipes"""
        # Use first few words of name + category as cache key
        name_words = ' '.join(name.lower().split()[:3]) if name else ''
        category_clean = category.lower() if category else ''
        return f"{name_words}|{category_clean}"
    
    def prepare_batch_data(self, df):
        """Prepare data in batches with pre-filtering"""
        batches = []
        cache_hits = 0
        prefiltered = 0
        
        logger.info("Preparing batches with pre-filtering ... ")
        
        for idx, row in df.iterrows():
            name = str(row.get('Name', '')).strip()
            description = str(row.get('Description', '')).strip()[:200]
            category = str(row.get('RecipeCategory', '')).strip()
            keywords = str(row.get('Keywords', '')).strip()
            
            # Create combined text for analysis
            combined_text = f"{name} {description} {category} {keywords}"
            
            # Try cache first
            cache_key = self.create_cache_key(name, category)
            
            if cache_key in self.prediction_cache:
                df.at[idx, 'predicted_cuisine'] = self.prediction_cache[cache_key]
                cache_hits += 1
                continue
            
            # Try keyword pre-filtering
            obvious_cousine = self.prefilter_obvious_cuisines(combined_text)
            if obvious_cousine:
                df.at[idx, 'predicted_cuisine'] = obvious_cousine
                self.prediction_cache[cache_key] = obvious_cousine
                prefiltered += 1
                continue
                
            # Add to batch for Ollama processing
            batch_item = {
                'index': idx,
                'name': name,
                'description': description,
                'category': category,
                'keywords': keywords,
                'cache_key': cache_key,
                'text': combined_text[:500]
            }
            batches.append(batch_item)
        
        logger.info(f"Cache hits: {cache_hits}, Pre-filtered: {prefiltered}, Ollama needed: {len(batches)}")
        return batches

    def create_optimized_prompt(self, batch_items):
        """Create a batch prompt for multiple recipes"""
        prompt = f"""Predict the cuisine type for these recipes. Choose from: {', '.join(self.cuisine_categories)}

For each recipe, respond with only the cuisine type (one word). If uncertain, choose "Unknown".

Recipes:
"""
        for i, item in enumerate(batch_items, 1):
            prompt += f"{i}. {item['name']} - {item['category']} - {item['keywords'][:100]}\n"
        
        prompt += "\nRespond with only the cuisine names, one per line:"
        
        return prompt
    
    def call_ollama_batch(self, batch_items, batch_size = 10):
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
                if prediction in self.cuisine_categories:
                    predictions.append((item, prediction))
                else:
                    predictions.append((item, "Unknown"))  # Fallback
            else:
                predictions.append((item, "Unknown"))  # Fallback
        
        return predictions

    def clean_prediction(self, prediction):
        """Clean and standardize prediction"""
        prediction = prediction.strip().title()
        
        # Handle common variations
        mapping = {
            "Asian": "Chinese",
            "European": "International", 
            "Latin": "Mexican",
            "Southern": "American",
            "Tex-Mex": "Mexican",
            "Pan-Asian": "Fusion"
        }
        
        return mapping.get(prediction, prediction)

    def fallback_predictions(self, batch_items):
        """Fallback predictions based on simple rules"""
        predictions = []
        for item in batch_items:
            # Simple keyword fallback
            text = item['text'].lower()
            if any(word in text for word in ['pasta', 'pizza', 'italian']):
                cuisine = "Italian"
            elif any(word in text for word in ['taco', 'salsa', 'mexican']):
                cuisine = "Mexican"
            elif any(word in text for word in ['curry', 'spice', 'indian']):
                cuisine = "Indian"
            else:
                cuisine = "American"  # Default fallback
            
            predictions.append((item, cuisine))
        
        return predictions

    def process_batch_worker(self, batch):
        """Worker function for parallel processing"""
        return self.call_ollama_batch(batch)
    
    def predict_cuisines(self, df, batch_size = 10):
        """Main function to predict cuisines for entire dataset"""
        
        logger.info(f"Starting cuisine prediction for {len(df)} recipes...")
        start_time = time.time()
        
        # Initialize prediction column
        df['predicted_cuisine'] = "Unknown"
        
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
                    
                    for item, cuisine in predictions:
                        df.at[item['index'], 'predicted_cuisine'] = cuisine
                        self.prediction_cache[item['cache_key']] = cuisine
                    
                    processed_count += len(predictions)
                    
                    if processed_count % (batch_size * 10) == 0:  # Progress update every 100 items
                        elapsed = time.time() - start_time
                        rate = processed_count / elapsed
                        logger.info(f"Processed {processed_count}/{len(ollama_batches)} items. Rate: {rate:.1f} items/sec")

                except Exception as e:
                    logger.error(f"Batch processing failed: {e}")
                    # Handle failed batch with fallbacks
                    batch = future_to_batch[future]
                    for item in batch:
                        df.at[item['index'], 'predicted_cuisine'] = "Unknown"
                
            total_time = time.time() - start_time
            logger.info(f"Completed cuisine prediction in {total_time:.2f} seconds")
            logger.info(f"Average rate: {len(df) / total_time:.1f} recipes/second")
            
            return df

def predict_recipe_cuisines(input_csv_path, output_csv_path, model_name="llama3.2:3b"):
    # Load data
    logger.info(f"Loading dataset from {input_csv_path}")
    df = pd.read_csv(input_csv_path)
    
    # Initialize predictor
    predictor = CuisinePredictor(model_name=model_name, max_workers=8)
    
    df_with_predictions = predictor.predict_cuisines(df, batch_size=12)

    # Save results
    df_with_predictions.to_csv(output_csv_path, index=False)
    logger.info(f"Results saved to {output_csv_path}")
    
    # Print summary
    cuisine_counts = df_with_predictions['predicted_cuisine'].value_counts()
    logger.info("Cuisine distribution:")
    for cuisine, count in cuisine_counts.head(10).items():
        logger.info(f"  {cuisine}: {count} recipes ({count/len(df)*100:.1f}%)")
    
    return df_with_predictions

if __name__ == "__main__":
    # Uncomment to test keyword parsing first
    # test_keyword_parsing()
    
    # Configuration
    INPUT_CSV = "/Users/shubhan/MacroRecipe/MacroRecipeLLM/recipes_with_flavors.csv"
    OUTPUT_CSV = "/Users/shubhan/MacroRecipe/MacroRecipeLLM/recipes_with_cuisine_predictions.csv" 
    MODEL = "llama3.2:3b"  # Fast 3B model - download with: ollama pull llama3.2:3b
    
    # Run prediction
    result_df = predict_recipe_cuisines(INPUT_CSV, OUTPUT_CSV, MODEL)