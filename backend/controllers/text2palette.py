from flask import jsonify
from errors import BadRequestError
import openai
import re
import colorsys
from dotenv import load_dotenv

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

load_dotenv()
client = openai.OpenAI()

def generate_test_palette_from_query(user_query: str, retrieved_examples: str):

    response = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_HSL},
            {"role": "user", "content": USER_PROMPT_TEMPLATE_HSL.format(query= user_query, examples= retrieved_examples)},
        ],
        response_format={"type": "json_object"}
    )

    content = response.choices[0].message.content

    hsl_pattern = r'\(\s*(\d{1,3})\s*,\s*(\d{1,3})%\s*,\s*(\d{1,3})%\s*\)'
    matches = re.findall(hsl_pattern, content)
    
    hsl_colors = []
    for h, s, l in matches:
        h_int, s_int, l_int = int(h), int(s), int(l)
        if 0 <= h_int < 360 and 0 <= s_int <= 100 and 0 <= l_int <= 100:
            r, g, b = colorsys.hls_to_rgb(h_int/360, l_int/100, s_int/100)
            hsl_colors.append([int(r * 255 + 0.5), int(g * 255 + 0.5), int(b * 255 + 0.5)])
        if len(hsl_colors) >= 5:
            break

    res = {
        "msg": "Test palette generated from user query!",
        "user_query": user_query,
        "retrieved_examples": retrieved_examples,
        "palette": hsl_colors
    }

    return jsonify(res), 200


__all__ = [
    'generate_test_palette_from_query'
]