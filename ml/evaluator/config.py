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
RESULTS_DIR = "ml/evaluator/results2"

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
SYSTEM_PROMPT_HSL = """
You are an expert Color Theorist and UI Designer. Your task is to generate a cohesive, aesthetically pleasing color palette based on a user's text query.

Generate a JSON response with the following format:
{
    "palette_text": ["color_name1", "color_name2", "color_name3", "color_name4", "color_name5"],
    "palette_hex": ["(H, S%, L%)", "(H, S%, L%)", "(H, S%, L%)", "(H, S%, L%)", "(H, S%, L%)"]
}
where H is between [0, 360), and S and L are between [0, 100]. No decimals.
As a reminder, H stands for hue, S stands for saturation, and L stands for lightness
"""

USER_PROMPT_TEMPLATE_HSL = """
What's the best color palette consisting of five colors to describe the text "{query}"?
Provide the color values using hsl format.

Here are some associate text-palette pairs for reference:
### REFERENCE PALETTES
{examples}
"""

SYSTEM_PROMPT_CIELAB = """
You are an expert Color Theorist and UI Designer. Your task is to generate a cohesive, aesthetically pleasing color palette based on a user's text query.

Generate a JSON response with the following format:
{
    "palette_text": ["color_name1", "color_name2", "color_name3", "color_name4", "color_name5"],
    "palette_lab": ["L(L, a, b)", "L(L, a, b)", "L(L, a, b)", "L(L, a, b)", "L(L, a, b)"]
}
where L is between [0, 100], a is between [-128, 127], and b is between [-128, 127]. No decimals.
As a reminder, L stands for lightness, a stands for green-red axis, and b stands for blue-yellow axis.
"""

USER_PROMPT_TEMPLATE_CIELAB = """
What's the best color palette consisting of five colors to describe the text "{query}"?
Provide the color values using CIELAB format.

Here are some associate text-palette pairs for reference:
### REFERENCE PALETTES
{examples}
"""

SYSTEM_PROMPT = """
You are an expert Color Theorist and UI Designer. Your task is to generate a cohesive, aesthetically pleasing color palette based on a user's text query.

Generate a JSON response with the following format:
{
    "palette_text": ["color_name1", "color_name2", "color_name3", "color_name4", "color_name5"],
    "palette_hex": ["#XXXXXX", "#XXXXXX", "#XXXXXX", "#XXXXXX", "#XXXXXX"]
}
"""

USER_PROMPT_TEMPLATE = """
What's the best color palette consisting of five colors to describe the text "{query}"?
Provide the color values using text (hex) format in ascending order.

Here are some associate text-palette pairs for reference:
### REFERENCE PALETTES
{examples}
"""
