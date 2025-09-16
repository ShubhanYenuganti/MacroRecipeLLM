# MacroRecipeLLM
Attempt at fine-tuning an LLM to be able to provide an end-user a recipe recommendation based on their desired macros and other factors. 

# Dataset
Utilizes following Kaggle dataset [Food.com - Recipes and Reviews](https://www.kaggle.com/datasets/irkaal/foodcom-recipes-and-reviews?resource=download).

## Original Schema
Kaggle dataset originally contained following metadata:  
**Cook Time**, **PrepTime**, **TotalTime**, **Description**, **RecipeCategory**, **RecipeCategory**, **Keywords**, **RecipeIngredientQuantities**, **RecipeIngredientParts**, **AggregatedRating**, **ReviewCount**, **Calories**, **Calories**, **FatContent**, **SaturatedFatContent**, **Cholestrol**, **SodiumContent**, **CarbohydrateContent**, **FiberContent**, **SugarContent**, **ProteinContent**, **RecipeInstructions**

# AWS Workflow: S3 → OpenSearch Serverless → Bedrock Knowledge Base → Bedrock Agent (Nova Premier) → Lambda URL API
We store pre-chunked CSV exports of the embedded-ready CSVs in **Amazon S3**.  
**Amazon Bedrock Knowledge Bases** ingests those objects, generates embeddings, and stores vectors in **Amazon OpenSearch Serverless**.  
A **Bedrock Agent** (LLM: **Amazon Nova Premier**) uses the Knowledge Base for retrieval and produces **structured JSON** outputs.  
An **AWS Lambda** function exposes a **Function URL** that calls to query the Agent and returns the output.

# Data Transformation
features.py adds **Complexity Score**, **Batch Friendliness Score**, **Difficulty Level** as additional columns to the dataset.  
- the complexity score is meant to encapsulate the difficulty of preparing the recipe based on the count of ingredients, instructions and time to prepare along with complex techniques detected in the recipe instructions.
- the difficulty level is a labeling of the complexity score based on thresholds
- the batch friendliness score is compiled by assessing the instructions of the recipe to generate a score indicating whether the food is viable for meal prepping.

`flavor_predictor.py` is a multi-layered flavor predictor that identifies keywords across the description, keywords, category, ingredients and instructions that may indicate towards a flavor label. It generates a score based on a normalized summations of scores for each keyword found based on in which label (description, keywords, category, etc.) it was found in.

`cuisine_predictor.py` runs a querying predictor on ollama's open-source llama3.2:3b model to predict a cuisine label for a recipe based on its associated description, category and keywords.

`meal_category_predictor.py` runs a querying predictor on ollama's open-source llama3.2:3b model to predict a label (Breakfast, Lunch, Dinner, Snack) for a recipe based on its associated name, description, keywords and total time to cooking and prepare.

# New Schema for Efficient Embeddings
This schema is used to transform the dataset into a viable format for Amazon OpenSearch to generate more effective embeddings. 
RECIPE: <Name>

=== INGREDIENTS & INSTRUCTIONS ===

INGREDIENTS (<N> items):
• <qty> <ingredient>
• ...

COOKING INSTRUCTIONS:
1. <step one>
2. <step two>
...

=== RECIPE PROFILE ===

SEMANTIC PROFILE:
    Recipe Name: <Name>
    Category: <RecipeCategory>
    Description: <shortened description>
    Keywords: [<keywords>]
    Cuisine Style: <predicted_cuisine>
    Dietary Profile: <derived dietary tags>
    Flavor Profile: <flavor_profile>
    Meal Type: <predicted_meal_category>

NUTRITIONAL PROFILE:
    Total Calories: <value> (<category>)
    Total Protein: <g> (<category>)
    Total Carbohydrates: <g> (<category>)
    Total Fat: <g> (<category>)
    Total Sugars: <g> (<category>)
    Macro Ratio: P:<%>, C:<%>, F:<%>
    Diet Compatibility: <list>
    Servings: <servings>
    Nutritional Density Score: <0–1>

PROCEDURAL PROFILE:
    Total Time: <minutes> (<bucket>)
    Prep Time: <minutes>
    Cook Time: <minutes>
    Difficulty Level: <Easy|Medium|Hard>
    Complexity Score: <descriptor> <0–1>
    Skill Level Required: <Beginner|Intermediate|Advanced>
    Batch Cooking Friendly: <Yes|Moderate|No>

QUALITY & RATINGS:
    Overall Rating: <rating>/5.0 stars
    Reviews: <count> reviews
    Quality Score: <0–1>
