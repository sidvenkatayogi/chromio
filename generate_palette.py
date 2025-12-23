import os
import subprocess
import json
import argparse
import openai
from dotenv import load_dotenv

def get_examples_from_query_db(query: str) -> str:
    """
    Runs the query_db.py script and returns its output.
    """
    try:
        result = subprocess.run(
            ["python", "query_db.py", query],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error running query_db.py: {e}")
        print(f"Stderr: {e.stderr}")
        return ""
    except FileNotFoundError:
        print("Error: 'python' command not found. Make sure Python is in your PATH.")
        return ""


def generate_palette_with_gpt4o(user_query: str, examples: str) -> dict:
    """
    Uses GPT-4o to generate a color palette.
    """
    client = openai.OpenAI()

    system_prompt = """
You are an expert color palette generator. Your task is to create a color palette based on a user's query.
You will be given a user query and some examples of palettes with their descriptions.
Use these examples to understand the kind of palettes the user is looking for.

Generate a palette of 5 colors in hex code format.

Your output MUST be a JSON object with a single key "palette" which is an array of 5 hex code strings.
Do not include any other text or explanations in your response.

Example response:
{
  "palette": [
    "#FFFFFF",
    "#000000",
    "#FF0000",
    "#00FF00",
    "#0000FF"
  ]
}
"""

    user_message = f"""Here are some examples of palettes:
---
{examples}
---

Based on those examples, generate a color palette for the following user query:
User Query: "{user_query}"""
    try:
        response = client.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            response_format={"type": "json_object"}
        )
        palette_json = response.choices[0].message.content
        print(palette_json)
        return json.loads(palette_json)
    except openai.APIError as e:
        print(f"OpenAI API Error: {e}")
    except json.JSONDecodeError:
        print("Error: Failed to decode JSON from GPT-5-mini response.")
    return {}

def main():
    """
    Main function to run the script.
    """
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: The OPENAI_API_KEY environment variable is not set.")
        print("Please create a .env file and add your OpenAI API key to it.")
        print("Example content for .env file:")
        print("OPENAI_API_KEY='your-key-here'")
        return

    parser = argparse.ArgumentParser(description="Generate a color palette using GPT5-mini.")
    parser.add_argument("query", help="Text query to generate a palette for.")
    args = parser.parse_args()

    print(f"Generating palette for: '{args.query}'")
    
    print("1. Getting examples from the local database...")
    examples = get_examples_from_query_db(args.query)

    if not examples.strip():
        print("Could not get examples from query_db.py. Proceeding without them.")

    print("2. Calling GPT-4o to generate a new palette...")
    generated_palette = generate_palette_with_gpt4o(args.query, examples)

    if generated_palette and "palette" in generated_palette:
        print("\n--- Generated Palette ---")
        for color in generated_palette["palette"]:
            print(f"  - {color}")
        print("\n")
    else:
        print("Could not generate a palette.")


if __name__ == "__main__":
    main()