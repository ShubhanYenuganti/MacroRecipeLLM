import subprocess
import os
import pandas as pd
from ollama.meal_category_predictor import MealCategoryPredictor

def setup_ollama_batch_job(csv_path, model_name="llama3.2:1b"):
    output_dir = os.path.abspath("./ollama_meal_batch")
    os.makedirs(output_dir, exist_ok=True)
    
    batch_script_path = os.path.join(output_dir, "run_ollama_meal_batch.sh")
    abs_csv_path = os.path.abspath(csv_path)
    output_csv = os.path.abspath("recipes_with_meal_category_ollama.csv")
    
    script_content = f"""#!/bin/bash
    echo "Starting Ollama batch cuisine classification..."

# Ensure model is available
ollama pull {model_name}

python3 << 'EOF'
import pandas as pd
from ollama.meal_category_predictor import MealCategoryPredictor
import json
import os

csv_path = r"{abs_csv_path}"
output_csv = r"{output_csv}"

df = pd.read_csv(csv_path)
print(f"Loaded {{len(df)}} recipes")

predictor = MealCategoryPredictor(model="{model_name}")
results = predictor.predict_batch(df)

df['meal_type'] = [r['meal_type'] for r in results]
df['meal_confidence'] = [r['confidence'] for r in results]

df.to_csv(output_csv, index=False)

with open(os.path.join(os.path.dirname(output_csv), "ollama_detailed_results.json"), "w") as f:
    json.dump(results, f, indent=2)

print("Batch processing complete!")
print(df['cuisine_style'].value_counts())
EOF

echo "Ollama batch job finished"
"""
    with open(batch_script_path, "w") as f:
        f.write(script_content)

    os.chmod(batch_script_path, 0o755)
    print(f"Batch script created: {batch_script_path}")
    return batch_script_path

if __name__ == "__main__":
    csv_input = "preprocess/recipes_clean.csv"
    csv_output = "recipes_with_cuisine_ollama.csv"

    script_path = setup_ollama_batch_job(csv_input, "llama3.2:1b")
    print("\nRunning batch job script...")
    subprocess.run([script_path], check=True)

    df_with_meal_type = pd.read_csv(os.path.abspath(csv_output))
    print("\Meal Classification Results (Option 2):")
    print(df_with_meal_type['meal_type'].value_counts())
    print(f"High confidence: {(df_with_meal_type['confidence'] >= 0.7).sum()}")
