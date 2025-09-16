# MacroRecipeLLM
Attempt at fine-tuning an LLM to be able to provide an end-user a recipe recommendation based on their desired macros and other factors. 

# Dataset
Utilizes following Kaggle dataset [Food.com - Recipes and Reviews](https://www.kaggle.com/datasets/irkaal/foodcom-recipes-and-reviews?resource=download).

## Original Schema
Kaggle dataset originally contained following metadata:  
**Cook Time**, **PrepTime**, **TotalTime**, **Description**, **RecipeCategory**, **RecipeCategory**, **Keywords**, **RecipeIngredientQuantities**, **RecipeIngredientParts**, **AggregatedRating**, **ReviewCount**, **Calories**, **Calories**, **FatContent**, **SaturatedFatContent**, **Cholestrol**, **SodiumContent**, **CarbohydrateContent**, **FiberContent**, **SugarContent**, **ProteinContent**, **RecipeInstructions**

# Transformation
features.py adds **Complexity Score**, **Batch Friendliness Score**, **Difficulty Level** as additional columns to the dataset.  
- the complexity score is meant to encapsulate the difficulty of preparing the recipe based on the count of ingredients, instructions and time to prepare along with complex techniques detected in the recipe instructions.
- the difficulty level is a labeling of the complexity score based on thresholds
- the batch friendliness score is compiled by assessing the instructions of the recipe to generate a score indicating whether the food is viable for meal prepping.

`flavor_predictor.py` is a multi-layered flavor predictor that identifies keywords across the description, keywords, category, ingredients and instructions that may indicate towards a flavor label. It generates a score based on a normalized summations of scores for each keyword found based on in which label (description, keywords, category, etc.) it was found in.

`cuisine_predictor.py` runs a querying predictor on ollama's open-source llama3.2:3b model to predict a cuisine label for a recipe based on its associated description, category and keywords.

`meal_category_predictor.py` runs a querying predictor on ollama's open-source llama3.2:3b model to predict a label (Breakfast, Lunch, Dinner, Snack) for a recipe based on its associated name, description, keywords and total time to cooking and prepare.

# New Schema for Efficient Embeddings
