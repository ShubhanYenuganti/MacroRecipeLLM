import re
import pandas as pd
from collections import defaultdict
import math

def infer_flavor_profile(row):
    """Infers flavor profile from recipe metadata"""
    
    # Uses name, description, keywords, category, ingredients, instructions to infer flavor profile
    
    name = str(row.get('Name', '')).lower()
    description = str(row.get('Description', '')).lower()
    keywords = str(row.get('Keywords', '')).lower()
    category = str(row.get('RecipeCategory', '')).lower()
    ingredients = str(row.get('RecipeIngredientParts', '')).lower()
    instructions = str(row.get('RecipeInstructions', '')).lower()
    
    all_text = f"{name} {description} {keywords} {category} {ingredients} {instructions}"
    
    keyword_flavors = detect_keyword_flavors(all_text) # 0.4 weight
    ingredient_flavors = detect_ingredient_flavors(ingredients) # 0.3 weight
    method_flavors = detect_method_flavors(instructions) # 0.1 weight
    name_flavors = detect_name_flavors(name) # 0.2 weight 
    
    combined_scores = combine_flavor_scores(keyword_flavors, ingredient_flavors, method_flavors, name_flavors)

    return format_flavor_profile(combined_scores)

def detect_keyword_flavors(text):
    """Detect flavors based on explicit keywords"""
    
    flavor_keywords = {
        'spicy': ['spicy', 'hot', 'fiery', 'jalapeño', 'habanero', 'cayenne', 'chili', 'sriracha', 'tabasco'],
        'sweet': ['sweet', 'honey', 'maple', 'caramel', 'chocolate', 'vanilla', 'sugar', 'syrup'],
        'savory': ['savory', 'umami', 'meaty', 'rich', 'hearty', 'satisfying'],
        'tangy': ['tangy', 'tart', 'acidic', 'citrus', 'lemon', 'lime', 'vinegar', 'sour'],
        'smoky': ['smoky', 'grilled', 'barbecue', 'bbq', 'charred', 'wood-fired'],
        'creamy': ['creamy', 'rich', 'smooth', 'velvety', 'buttery', 'silky'],
        'fresh': ['fresh', 'bright', 'crisp', 'light', 'refreshing', 'zesty'],
        'earthy': ['earthy', 'rustic', 'mushroom', 'truffle', 'woody'],
        'aromatic': ['aromatic', 'fragrant', 'herbal', 'perfumed'],
        'bold': ['bold', 'intense', 'robust', 'strong', 'powerful'],
        'mild': ['mild', 'gentle', 'subtle', 'delicate', 'light'],
        'complex': ['complex', 'layered', 'nuanced', 'sophisticated']
    }
    
    flavor_scores = defaultdict(float)
    
    for flavor, keywords in flavor_keywords.items():
        for kw in keywords:
            count = len(re.findall(r'\b' + re.escape(kw) + r'\b', text))
            if count > 0:
                flavor_scores[flavor] += count
    
    return dict(flavor_scores)

def detect_ingredient_flavors(ingredients):
    """Infer flavors from ingredient analysis"""
    
    flavor_scores = defaultdict(float)
    
    spicy_ingredients = {
        'jalapeño': 0.9, 'habanero': 1.0, 'cayenne pepper': 0.95,
        'chili flakes': 0.85, 'sriracha': 0.9, 'tabasco': 0.85,
        'black pepper': 0.6, 'ginger': 0.5, 'mustard': 0.4
    }
    
    sweet_ingredients = {
        'sugar': 1.0, 'honey': 0.9, 'maple syrup': 0.9,
        'caramel': 0.85, 'chocolate': 0.8, 'vanilla': 0.7,
        'fruit': 0.6, 'molasses': 0.8, 'syrup': 0.85
    }
    
    savory_ingredients = {
        'soy sauce': 0.9, 'miso': 0.9, 'beef': 0.85,
        'chicken stock': 0.8, 'parmesan': 0.8,
        'mushrooms': 0.75, 'bacon': 0.9, 'anchovy': 0.9
    }
    
    tangy_ingredients = {
        'lemon juice': 1.0, 'lime juice': 0.95,
        'vinegar': 0.9, 'tamarind': 0.85, 'yogurt': 0.6,
        'pickles': 0.75, 'cranberry': 0.7, 'green apple': 0.7
    }
    
    smoky_ingredients = {
        'smoked paprika': 0.9, 'chipotle': 0.9, 'liquid smoke': 1.0,
        'bacon': 0.8, 'smoked salmon': 0.85, 'charcoal-grilled meat': 0.95,
        'bbq sauce': 0.8, 'roasted peppers': 0.7
    }
    
    creamy_ingredients = {
        'cream': 0.9, 'milk': 0.6, 'butter': 0.8, 'cheese': 0.7,
        'yogurt': 0.6, 'coconut milk': 0.8, 'heavy cream': 1.0, 'sour cream': 0.7,
        'mascarpone': 0.9, 'ricotta': 0.7, 'mayonnaise': 0.8
    }


    fresh_ingredients = {
        'mint': 0.9, 'basil': 0.8, 'parsley': 0.8,
        'cucumber': 0.75, 'cilantro': 0.85,
        'lemon zest': 0.8, 'lime zest': 0.8, 'arugula': 0.7
    }
    
    earthy_ingredients = {
        'mushrooms': 0.9, 'truffle': 1.0, 'beets': 0.85,
        'lentils': 0.75, 'potatoes': 0.7, 'carrots': 0.6,
        'turnips': 0.7, 'wild rice': 0.7
    }
    
    aromatic_ingredients = {
        'garlic': 0.9, 'onion': 0.8, 'ginger': 0.8,
        'cinnamon': 0.75, 'cardamom': 0.8,
        'rosemary': 0.85, 'thyme': 0.8, 'lemongrass': 0.85
    }
    
    bold_ingredients = {
        'blue cheese': 1.0, 'anchovies': 0.9,
        'wasabi': 0.95, 'horseradish': 0.9,
        'garlic': 0.8, 'dark chocolate': 0.75,
        'espresso': 0.8, 'fish sauce': 0.9
    }
    
    mild_ingredients = {
        'milk': 0.7, 'rice': 0.6, 'tofu': 0.65,
        'zucchini': 0.6, 'chicken breast': 0.7,
        'cauliflower': 0.65, 'white bread': 0.6, 'oatmeal': 0.65
    }
    
    complex_ingredients = {
        'wine': 0.9, 'soy sauce': 0.85, 'balsamic vinegar': 0.9,
        'curry powder': 0.9, 'garam masala': 0.9,
        'miso': 0.9, 'aged cheese': 0.9, 'dark chocolate': 0.85
    }
    
    # score ingredients
    ingredient_categories = {
        'spicy': spicy_ingredients,
        'sweet': sweet_ingredients,
        'savory': savory_ingredients,
        'tangy': tangy_ingredients,
        'smoky': smoky_ingredients,
        'creamy': creamy_ingredients,
        'fresh': fresh_ingredients,
        'earthy': earthy_ingredients,
        'aromatic': aromatic_ingredients,
        'bold': bold_ingredients,
        'mild': mild_ingredients,
        'complex': complex_ingredients
    }
    
    for flavor, ingredients_dict in ingredient_categories.items():
        for ingredient, score in ingredients_dict.items():
            if re.search(r'\b' + re.escape(ingredient) + r'\b', ingredients):
                flavor_scores[flavor] += score 

    return dict(flavor_scores)

def detect_method_flavors(instructions):
    """Detect flavors based on cooking instructions"""
    
    flavor_scores = defaultdict(float)
    
    cooking_method_flavor_map = {
        # Dry-heat methods
        'grilling':      {'smoky': 0.9, 'savory': 0.8, 'bold': 0.7, 'earthy': 0.6},
        'roasting':      {'savory': 0.8, 'earthy': 0.7, 'aromatic': 0.6, 'sweet': 0.5},
        'baking':        {'sweet': 0.9, 'creamy': 0.6, 'aromatic': 0.6, 'complex': 0.7},
        'broiling':      {'savory': 0.8, 'bold': 0.7, 'smoky': 0.6},
        'toasting':      {'aromatic': 0.7, 'earthy': 0.6, 'bold': 0.5},

        # Frying methods
        'pan-frying':    {'savory': 0.8, 'bold': 0.6, 'smoky': 0.5, 'aromatic': 0.6},
        'deep-frying':   {'savory': 0.8, 'bold': 0.7, 'creamy': 0.5, 'complex': 0.6},
        'stir-frying':   {'fresh': 0.8, 'aromatic': 0.7, 'bold': 0.6, 'spicy': 0.6},
        'sautéing':      {'aromatic': 0.8, 'savory': 0.7, 'fresh': 0.6},

        # Moist-heat methods
        'boiling':       {'mild': 0.7, 'fresh': 0.5},
        'steaming':      {'fresh': 0.9, 'mild': 0.8, 'delicate': 0.6},
        'poaching':      {'mild': 0.9, 'delicate': 0.8, 'fresh': 0.7},
        'simmering':     {'savory': 0.7, 'complex': 0.7, 'aromatic': 0.6},
        'braising':      {'savory': 0.9, 'complex': 0.8, 'earthy': 0.7, 'bold': 0.6},
        'stewing':       {'savory': 0.9, 'complex': 0.8, 'hearty': 0.7},

        # Specialized
        'smoking':       {'smoky': 1.0, 'bold': 0.8, 'savory': 0.7},
        'barbecuing':    {'smoky': 0.9, 'savory': 0.8, 'spicy': 0.6, 'bold': 0.8},
        'fermenting':    {'tangy': 0.9, 'complex': 0.9, 'aromatic': 0.7},
        'pickling':      {'tangy': 1.0, 'bold': 0.6, 'aromatic': 0.5},
        'caramelizing':  {'sweet': 0.9, 'aromatic': 0.7, 'complex': 0.7},
        'glazing':       {'sweet': 0.8, 'bold': 0.6, 'aromatic': 0.5},
        'marinating':    {'tangy': 0.7, 'spicy': 0.7, 'aromatic': 0.6, 'complex': 0.7},
        'curing':        {'savory': 0.8, 'bold': 0.8, 'complex': 0.7},
        'sous-vide':     {'mild': 0.9, 'creamy': 0.7, 'delicate': 0.8},
    }
    
    for method, flavors in cooking_method_flavor_map.items():
        if re.search(r'\b' + re.escape(method) + r'\b', instructions):
            for flavor, score in flavors.items():
                flavor_scores[flavor] += score
                
    return dict(flavor_scores)

def detect_name_flavors(name):
    
    flavor_scores = defaultdict(float)
    
    keyword_flavor_map = {
        # Spicy
        'spicy': 'spicy', 'hot': 'spicy', 'fiery': 'spicy',
        'jalapeño': 'spicy', 'habanero': 'spicy', 'cayenne': 'spicy',
        'chili': 'spicy', 'sriracha': 'spicy', 'tabasco': 'spicy',
        'pepper': 'spicy', 'wasabi': 'spicy',

        # Sweet
        'sweet': 'sweet', 'honey': 'sweet', 'maple': 'sweet',
        'caramel': 'sweet', 'chocolate': 'sweet', 'vanilla': 'sweet',
        'sugar': 'sweet', 'syrup': 'sweet', 'candy': 'sweet',
        'dessert': 'sweet', 'frosting': 'sweet',

        # Savory
        'savory': 'savory', 'umami': 'savory', 'meaty': 'savory',
        'rich': 'savory', 'hearty': 'savory', 'satisfying': 'savory',
        'bacon': 'savory', 'beef': 'savory', 'pork': 'savory',
        'anchovy': 'savory', 'broth': 'savory',

        # Tangy
        'tangy': 'tangy', 'tart': 'tangy', 'acidic': 'tangy',
        'citrus': 'tangy', 'lemon': 'tangy', 'lime': 'tangy',
        'vinegar': 'tangy', 'sour': 'tangy', 'pickled': 'tangy',
        'cranberry': 'tangy', 'yogurt': 'tangy',

        # Smoky
        'smoky': 'smoky', 'smoked': 'smoky', 'grilled': 'smoky',
        'barbecue': 'smoky', 'bbq': 'smoky', 'charred': 'smoky',
        'wood-fired': 'smoky', 'chipotle': 'smoky',

        # Creamy
        'creamy': 'creamy', 'rich': 'creamy', 'smooth': 'creamy',
        'velvety': 'creamy', 'buttery': 'creamy', 'silky': 'creamy',
        'cream': 'creamy', 'cheese': 'creamy', 'milk': 'creamy',
        'yogurt': 'creamy', 'mayo': 'creamy',

        # Fresh
        'fresh': 'fresh', 'bright': 'fresh', 'crisp': 'fresh',
        'light': 'fresh', 'refreshing': 'fresh', 'zesty': 'fresh',
        'herb': 'fresh', 'mint': 'fresh', 'basil': 'fresh',
        'cucumber': 'fresh', 'garden': 'fresh',

        # Earthy
        'earthy': 'earthy', 'rustic': 'earthy', 'mushroom': 'earthy',
        'truffle': 'earthy', 'woody': 'earthy', 'beet': 'earthy',
        'lentil': 'earthy', 'root': 'earthy',

        # Aromatic
        'aromatic': 'aromatic', 'fragrant': 'aromatic', 'herbal': 'aromatic',
        'perfumed': 'aromatic', 'spiced': 'aromatic', 'ginger': 'aromatic',
        'garlic': 'aromatic', 'onion': 'aromatic', 'cinnamon': 'aromatic',
        'cardamom': 'aromatic', 'rosemary': 'aromatic',

        # Bold
        'bold': 'bold', 'intense': 'bold', 'robust': 'bold',
        'strong': 'bold', 'powerful': 'bold',
        'espresso': 'bold', 'dark': 'bold', 'blue cheese': 'bold',
        'fish sauce': 'bold',

        # Mild
        'mild': 'mild', 'gentle': 'mild', 'subtle': 'mild',
        'delicate': 'mild', 'plain': 'mild', 'simple': 'mild',
        'light': 'mild',

        # Complex
        'complex': 'complex', 'layered': 'complex', 'nuanced': 'complex',
        'sophisticated': 'complex', 'aged': 'complex',
        'fermented': 'complex', 'wine': 'complex',
        'balsamic': 'complex', 'curry': 'complex'
    }

    for keyword, flavor in keyword_flavor_map.items():
        if re.search(r'\b' + re.escape(keyword) + r'\b', name, re.IGNORECASE):
            flavor_scores[flavor] += 1.0

    return dict(flavor_scores)

def _transform(dict, *, mode="log1p", alpha=1.0, cap=None, binarize=False):
    """Transform raw detector scores into comparable magnitudes"""
    out = {}
    for k, v in dict.items():
        x = float(v)
        if cap is not None:
            x = min(x, cap)
        if binarize:
            x = 1.0 if x > 0 else 0.0
        if mode == "log1p":
            x = math.log1p(max(x, 0.0))
        elif mode == "power":
            x = max(x, 0.0) ** alpha
        elif mode == "none":
            x = max(x, 0.0)
        out[k] = x
    
    return out

def _normalize(d, *, method = "l1", temperature = 1.0):
    """Normalize a dict of scores to a comparable scale"""
    if not d:
        return {}
    vals = list(d.values())
    
    if method == "l1":
        s = sum(vals)
        if s == 0:
            return {k: 0.0 for k in d}
        return {k: v / s for k, v in d.items()}
    elif method == "softmax":
        # temperate < 1 sharpens, > 1 flattens
        t = max(1e-6, float(temperature))
        exps = {k: math.exp(v / t) for k, v in d.items()}
        z = sum(exps.values())
        return {k: (exps[k] / z) if z > 0 else 0.0 for k in d}
    elif method == "max":
        m = max(vals)
        if m == 0:
            return {k: 0.0 for k in d}
        return {k: v / m for k, v in d.items()}
    else:
        return d  # no normalization
           

def combine_flavor_scores(keyword_flavors, ingredient_flavors, method_flavors, name_flavors):
    # 1) Transform each detector to tame scale differences
    # Tune these knobs if needed:
    kw_t   = _transform(keyword_flavors,   mode="log1p")                       # dampen raw counts
    ingr_t = _transform(ingredient_flavors, mode="power", alpha=0.7, cap=1.0)  # cap & gentle compression
    meth_t = _transform(method_flavors,    mode="power", alpha=0.7, cap=1.0)   # same idea as ingredients
    name_t = _transform(name_flavors,      mode="none", binarize=True)         # presence/absence

    # 2) Normalize each detector independently (probability-like vectors)
    kw_n   = _normalize(kw_t,   method="l1")
    ingr_n = _normalize(ingr_t, method="l1")
    meth_n = _normalize(meth_t, method="l1")
    name_n = _normalize(name_t, method="l1")

    # 3) Apply global weights (these sum to 1.0)
    weights = {
        'keyword':   0.4,
        'ingredient':0.3,
        'method':    0.1,
        'name':      0.2
    }

    combined = defaultdict(float)
    for d, w in [(kw_n, weights['keyword']),
                 (ingr_n, weights['ingredient']),
                 (meth_n, weights['method']),
                 (name_n, weights['name'])]:
        for flavor, score in d.items():
            combined[flavor] += w * score

    # Optional: final normalization to keep outputs comparable across rows
    combined = _normalize(combined, method="l1")
    return dict(combined)

def format_flavor_profile(flavor_scores, max_flavors=4, min_score=0.1):
    """Format the final flavor profile"""
    
    # Filter and sort flavors
    significant_flavors = {
        flavor: score for flavor, score in flavor_scores.items() 
        if score >= min_score
    }
    
    # Sort by score and take top flavors
    sorted_flavors = sorted(significant_flavors.items(), key=lambda x: x[1], reverse=True)
    top_flavors = [flavor for flavor, score in sorted_flavors[:max_flavors]]
    
    # Create descriptive flavor profile
    if not top_flavors:
        return "Balanced"
    elif len(top_flavors) == 1:
        return top_flavors[0].title()
    elif len(top_flavors) == 2:
        return f"{top_flavors[0].title()} and {top_flavors[1].title()}"
    else:
        return f"{', '.join(flavor.title() for flavor in top_flavors[:-1])} and {top_flavors[-1].title()}"
    
def analyze_recipe_flavors(df):
    """Analyze flavor profiles for entire dataset"""
    
    for idx, row in df.iterrows():                 # row is a Series
        flavor_profile = infer_flavor_profile(row) # your infer_* uses .get OK
        print(f"Recipe: {row.get('Name', '')} => Flavor Profile: {flavor_profile}")
        df.at[idx, 'flavor_profile'] = flavor_profile
    
    return df

df = pd.read_csv("recipes_feature_eng.csv")

analyze_recipe_flavors(df).to_csv("recipes_with_flavors.csv", index=False)
