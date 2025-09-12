#!/bin/bash
echo "Starting Ollama batch cuisine classification..."

# Ensure model is available
ollama pull llama3.2:1b

python3 << 'EOF'
import pandas as pd
from embeddings.ollama_predictor import OllamaCuisinePredictor
import json
import os

csv_path = r"/Users/shubhan/MacroRecipe/MacroRecipeLLM/preprocess/recipes_clean.csv"
output_csv = r"/Users/shubhan/MacroRecipe/MacroRecipeLLM/recipes_with_cuisine_ollama.csv"

df = pd.read_csv(csv_path)
print(f"Loaded {len(df)} recipes")

predictor = OllamaCuisinePredictor(model="llama3.2:1b")
results = predictor.predict_batch(df)

df['cuisine_style'] = [r['cuisine'] for r in results]
df['cuisine_confidence'] = [r['confidence'] for r in results]

df.to_csv(output_csv, index=False)

with open(os.path.join(os.path.dirname(output_csv), "ollama_detailed_results.json"), "w") as f:
    json.dump(results, f, indent=2)

print("Batch processing complete!")
print(df['cuisine_style'].value_counts())
EOF

echo "Ollama batch job finished"
