"""
Configuration for model evaluation.
"""

# Model providers
PROVIDER_OPENAI = "openai"
PROVIDER_FIREWORKS = "fireworks"

# Default models by provider
DEFAULT_MODELS = {
    PROVIDER_OPENAI: "gpt-4o-mini",
    PROVIDER_FIREWORKS: "accounts/fireworks/models/llama-v3p1-8b-instruct"
}

# Paths
EVAL_DATA_DIR = "hexcolor_vf"
TEST_NAMES_FILE = "test_names_eval.pkl"
TEST_PALETTES_FILE = "test_palettes_rgb_eval.pkl"
RESULTS_DIR = "ml/evaluator/results"

# Rate limiting defaults
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY = 1.0  # seconds
DEFAULT_RETRY_BACKOFF = 2.0  # exponential backoff multiplier
DEFAULT_REQUEST_DELAY = 0.1  # delay between requests (seconds)

# Prompts
# SYSTEM_PROMPT = """
# You are an expert Color Theorist and UI Designer. Your task is to generate a cohesive, aesthetically pleasing color palette based on a user's text query.

# Generate a JSON response with the following format:
# {
#     "palette_hex": ["#FFFFFF", "#FF0000", "#FFF200", "#54E5FF", "#000000"]
# }
# """

# USER_PROMPT_TEMPLATE = """
# What's the best color palette consisting of five colors to describe the text {query}?
# Provide the color values using text (hex) format in ascending order.

# Here are some associate text-palette pairs for reference:
# ### REFERENCE PALETTES
# {examples}
# """
SYSTEM_PROMPT = """
You are an expert Color Theorist and UI Designer. Your task is to generate a cohesive, aesthetically pleasing color palette based on a user's text query.

Generate a JSON response with the following format:
{
    "palette_text": ["Titanium White", "Classic Red", "Bright Yellow", "Sky Blue", "Black"],
    "palette_hex": ["#FFFFFF", "#FF0000", "#FFF200", "#54E5FF", "#000000"]
}
"""

USER_PROMPT_TEMPLATE = """
What's the best color palette consisting of five colors to describe the text {query}?
Provide the color values using text (hex) format in ascending order.

Here are some associate text-palette pairs for reference:
### REFERENCE PALETTES
{examples}
"""
