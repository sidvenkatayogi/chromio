#!/usr/bin/env python3
"""
Main evaluation script for color palette generation models using direct requests library.

Usage:
    python -m ml.evaluator.evaluate2 --provider fireworks --model accounts/sidvenkatayogi/deployedModels/rft-fhiaqwgu-0-1-ka2pgv73
"""

import os
import sys

# Suppress tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import pickle
import json
import requests
import time
import re
import logging
from dotenv import load_dotenv

# Suppress ChromaDB verbose output
logging.getLogger("chromadb").setLevel(logging.ERROR)

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from ml.evaluator.config import (
    PROVIDER_OPENAI, PROVIDER_FIREWORKS, DEFAULT_MODELS,
    EVAL_DATA_DIR, TEST_NAMES_FILE, TEST_PALETTES_FILE, RESULTS_DIR,
    DEFAULT_MAX_RETRIES, DEFAULT_RETRY_DELAY, DEFAULT_REQUEST_DELAY,
    SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
)
from ml.grader.evaluation import Color, ColorPalette, DccwMeasurer, normalize_inv_map, harmonic_mean

import chromadb
from chromadb.utils import embedding_functions

CHROMA_PATH = "chroma_db"
CHROMA_PATH_HSL = "chroma_db_hsl"
CHROMA_PATH_CIELAB = "chroma_db_cielab"
COLLECTION_NAME = "pat"
MODEL_NAME = "all-mpnet-base-v2"


def extract_hsl_from_string(content: str) -> list:
    """
    Extract HSL colors from string using regex.
    Matches format: (H, S%, L%) where H is 0-360, S and L are 0-100%
    """
    # Pattern to match (H, S%, L%) format
    hsl_pattern = r'\(\s*(\d{1,3})\s*,\s*(\d{1,3})%\s*,\s*(\d{1,3})%\s*\)'
    matches = re.findall(hsl_pattern, content)
    
    hsl_colors = []
    for h, s, l in matches:
        h_int, s_int, l_int = int(h), int(s), int(l)
        # Validate ranges
        if 0 <= h_int < 360 and 0 <= s_int <= 100 and 0 <= l_int <= 100:
            hsl_colors.append(f"({h_int}, {s_int}%, {l_int}%)")
        if len(hsl_colors) >= 5:
            break
    
    return hsl_colors


def hsl_to_hex(hsl_str: str) -> str:
    """
    Convert HSL string format "(H, S%, L%)" to hex color "#RRGGBB".
    """
    # Parse HSL values
    match = re.match(r'\(\s*(\d{1,3})\s*,\s*(\d{1,3})%\s*,\s*(\d{1,3})%\s*\)', hsl_str)
    if not match:
        return "#000000"
    
    h, s, l = int(match.group(1)), int(match.group(2)), int(match.group(3))
    
    # Convert to 0-1 range
    h = h / 360.0
    s = s / 100.0
    l = l / 100.0
    
    # HSL to RGB conversion
    def hue_to_rgb(p, q, t):
        if t < 0:
            t += 1
        if t > 1:
            t -= 1
        if t < 1/6:
            return p + (q - p) * 6 * t
        if t < 1/2:
            return q
        if t < 2/3:
            return p + (q - p) * (2/3 - t) * 6
        return p
    
    if s == 0:
        r = g = b = l
    else:
        q = l * (1 + s) if l < 0.5 else l + s - l * s
        p = 2 * l - q
        r = hue_to_rgb(p, q, h + 1/3)
        g = hue_to_rgb(p, q, h)
        b = hue_to_rgb(p, q, h - 1/3)
    
    # Convert to 0-255 range and format as hex
    r_int = int(round(r * 255))
    g_int = int(round(g * 255))
    b_int = int(round(b * 255))
    
    return f"#{r_int:02x}{g_int:02x}{b_int:02x}"


def extract_lab_from_string(content: str) -> list:
    """
    Extract CIELAB colors from string using regex.
    Matches format: L(L, a, b) where L is 0-100, a and b are -128 to 127
    """
    # Pattern to match L(L, a, b) format
    lab_pattern = r'L\(\s*(\d{1,3})\s*,\s*(-?\d{1,3})\s*,\s*(-?\d{1,3})\s*\)'
    matches = re.findall(lab_pattern, content)
    
    lab_colors = []
    for L, a, b in matches:
        L_int, a_int, b_int = int(L), int(a), int(b)
        # Validate ranges
        if 0 <= L_int <= 100 and -128 <= a_int <= 127 and -128 <= b_int <= 127:
            lab_colors.append(f"L({L_int}, {a_int}, {b_int})")
        if len(lab_colors) >= 5:
            break
    
    return lab_colors


def lab_to_hex(lab_str: str) -> str:
    """
    Convert CIELAB string format "L(L, a, b)" to hex color "#RRGGBB".
    """
    import numpy as np
    from skimage import color as sk_color
    
    # Parse LAB values
    match = re.match(r'L\(\s*(\d{1,3})\s*,\s*(-?\d{1,3})\s*,\s*(-?\d{1,3})\s*\)', lab_str)
    if not match:
        return "#000000"
    
    L, a, b = float(match.group(1)), float(match.group(2)), float(match.group(3))
    
    # Convert LAB to RGB
    lab_array = np.array([[[L, a, b]]])
    rgb_array = sk_color.lab2rgb(lab_array)[0][0]
    
    # Convert to 0-255 range and format as hex
    r_int = int(round(np.clip(rgb_array[0] * 255, 0, 255)))
    g_int = int(round(np.clip(rgb_array[1] * 255, 0, 255)))
    b_int = int(round(np.clip(rgb_array[2] * 255, 0, 255)))
    
    return f"#{r_int:02x}{g_int:02x}{b_int:02x}"


class RequestsModelClient:
    """Direct requests-based model client for Fireworks AI."""
    
    def __init__(self, model: str, api_key: str, 
                 max_retries: int = DEFAULT_MAX_RETRIES,
                 retry_delay: float = DEFAULT_RETRY_DELAY,
                 request_delay: float = DEFAULT_REQUEST_DELAY,
                 color_format: str = 'hex'):
        self.model = model
        self.api_key = api_key
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.request_delay = request_delay
        self.retry_backoff = 2.0
        self._last_request_time = 0
        self.url = "https://api.fireworks.ai/inference/v1/chat/completions"
        self.color_format = color_format.lower()
    
    def _extract_palette_text(self, data: dict) -> list:
        """
        Recursively extract palette_text from potentially nested JSON.
        """
        if isinstance(data, dict):
            if 'palette_text' in data:
                return data['palette_text']
            
            for key, value in data.items():
                if isinstance(value, dict):
                    result = self._extract_palette_text(value)
                    if result:
                        return result
        return []
    
    def _extract_hex_from_string(self, content: str) -> list:
        """
        Extract hex colors directly from string using regex.
        """
        hex_pattern = r'#[0-9a-fA-F]{6}'
        matches = re.findall(hex_pattern, content)
        # Return first 5 unique colors
        seen = []
        for m in matches:
            if m.lower() not in [s.lower() for s in seen]:
                seen.append(m)
            if len(seen) >= 5:
                break
        return seen
    
    def _wait_for_rate_limit(self):
        """Wait if needed to respect rate limits."""
        if self.request_delay > 0:
            elapsed = time.time() - self._last_request_time
            if elapsed < self.request_delay:
                time.sleep(self.request_delay - elapsed)
        self._last_request_time = time.time()
    
    def _call_with_retry(self, api_call_func):
        """Execute API call with retry logic and exponential backoff."""
        last_exception = None
        delay = self.retry_delay
        
        for attempt in range(self.max_retries):
            try:
                self._wait_for_rate_limit()
                return api_call_func()
            except requests.exceptions.RequestException as e:
                last_exception = e
                print(f"Request error (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    print(f"Retrying in {delay:.1f}s...")
                    time.sleep(delay)
                    delay *= self.retry_backoff
            except Exception as e:
                last_exception = e
                print(f"Unexpected error (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    print(f"Retrying in {delay:.1f}s...")
                    time.sleep(delay)
                    delay *= self.retry_backoff
        
        print(f"Max retries ({self.max_retries}) exceeded. Last error: {last_exception}")
        return {}, [], {}
    
    def generate_palette(self, query: str, examples: str) -> tuple:
        """
        Generate a color palette using the Fireworks API.
        
        Returns:
            tuple: (palette_hex, palette_text, raw_content)
        """
        user_message = USER_PROMPT_TEMPLATE.format(query=query, examples=examples)
        
        def api_call():
            payload = {
                "model": self.model,
                # "max_tokens": 4000,
                # "top_p": 1,
                # "top_k": 4,
                "temperature": 0,
                "messages": [
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT
                    },
                    {
                        "role": "user",
                        "content": user_message
                    }
                ]
            }
            
            headers = {
                "Accept": "application/json",
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            response = requests.post(self.url, headers=headers, data=json.dumps(payload))
            response.raise_for_status()
            
            response_json = response.json()
            content = response_json['choices'][0]['message']['content']
            
            # Extract colors based on format
            if self.color_format == 'hsl':
                palette_hsl = extract_hsl_from_string(content)
                # Convert HSL to hex for evaluation
                palette_hex = [hsl_to_hex(hsl) for hsl in palette_hsl]
            elif self.color_format == 'cielab' or self.color_format == 'lab':
                palette_lab = extract_lab_from_string(content)
                # Convert LAB to hex for evaluation
                palette_hex = [lab_to_hex(lab) for lab in palette_lab]
            else:
                palette_hex = self._extract_hex_from_string(content)
            
            try:
                content_json = json.loads(content)
                palette_text = self._extract_palette_text(content_json)
            except (json.JSONDecodeError, TypeError):
                palette_text = []
                
            return palette_hex, palette_text, content
        
        return self._call_with_retry(api_call)


def run_evaluation(provider: str, model: str = None, output_dir: str = None, limit: int = None,
                   max_retries: int = DEFAULT_MAX_RETRIES, retry_delay: float = DEFAULT_RETRY_DELAY,
                   request_delay: float = DEFAULT_REQUEST_DELAY, color_format: str = 'hex'):
    """
    Run the full evaluation pipeline.
    
    Args:
        provider: Model provider ('openai' or 'fireworks')
        model: Specific model name (required for this version)
        output_dir: Directory to save results (optional)
        limit: Maximum number of samples to evaluate (optional, for testing)
        max_retries: Maximum number of retries on rate limit errors
        retry_delay: Initial delay between retries (seconds)
        request_delay: Delay between requests to avoid rate limits (seconds)
        color_format: Color format to use ('hex' or 'hsl')
    """
    # Load environment variables
    load_dotenv()
    
    # Setup paths
    data_dir = os.path.join(PROJECT_ROOT, EVAL_DATA_DIR)
    output_dir = output_dir or os.path.join(PROJECT_ROOT, RESULTS_DIR)
    os.makedirs(output_dir, exist_ok=True)
    
    # Get model name
    model_name = model or DEFAULT_MODELS.get(provider)
    
    if not model_name:
        raise ValueError(f"Model name is required for this evaluation script")
    
    # Get API key
    api_key = os.getenv("FIREWORKS_API_KEY")
    if not api_key:
        raise ValueError("FIREWORKS_API_KEY environment variable is not set")
    
    print(f"=== Model Evaluation (Direct Requests) ===")
    print(f"Provider: {provider}")
    print(f"Model: {model_name}")
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Load evaluation data
    print("Loading evaluation data...")
    
    with open(os.path.join(data_dir, TEST_NAMES_FILE), 'rb') as f:
        test_names = pickle.load(f)
    
    with open(os.path.join(data_dir, TEST_PALETTES_FILE), 'rb') as f:
        test_palettes = pickle.load(f)
    
    if limit:
        test_names = test_names[:limit]
        test_palettes = test_palettes[:limit]
    
    print(f"Loaded {len(test_names)} test samples")
    print()
    
    # Initialize model client
    print("Initializing model client...")
    print(f"Color format: {color_format}")
    print(f"Rate limiting: max_retries={max_retries}, retry_delay={retry_delay}s, request_delay={request_delay}s")
    client = RequestsModelClient(
        model=model_name,
        api_key=api_key,
        max_retries=max_retries,
        retry_delay=retry_delay,
        request_delay=request_delay,
        color_format=color_format
    )
    
    print("Initializing example fetcher (loading embedding model)...")
    if color_format == 'cielab' or color_format == 'lab':
        chroma_path = os.path.join(PROJECT_ROOT, CHROMA_PATH_CIELAB)
    elif color_format == 'hsl':
        chroma_path = os.path.join(PROJECT_ROOT, CHROMA_PATH_HSL)
    else:
        chroma_path = os.path.join(PROJECT_ROOT, CHROMA_PATH)
        
    # Initialize client and embedding function once
    chroma_client = chromadb.PersistentClient(path=chroma_path)
    emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=MODEL_NAME
    )
    collection = chroma_client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=emb_fn
    )
    print()
    
    # Phase 1: Prepare all prompts
    print("=== Phase 1: Retrieving examples and preparing prompts ===")
    prepared_samples = []
    
    for i, (query_words, gt_rgb_palette) in enumerate(tqdm(zip(test_names, test_palettes), total=len(test_names), desc="Preparing prompts")):
        query_str = " ".join(query_words)
        gt_hex_palette = ColorPalette(gt_rgb_palette)
        
        retrieval_results = collection.query(
            query_texts=[query_str],
            n_results=3
        )
        
        output_lines = []
        if retrieval_results['documents'] and retrieval_results['documents'][0]:
            for j, result_doc in enumerate(retrieval_results['documents'][0], start=1):
                data = json.loads(result_doc)
                output_lines.append(f"Palette {j}:")
                output_lines.append(f"Description: {data['description']}")
                for color in data['palette']:
                    output_lines.append(f"  - {color}")
                output_lines.append("")
        
        examples = "\n".join(output_lines)
        
        prepared_samples.append({
            'index': i,
            'query_str': query_str,
            'query_words': query_words,
            'gt_rgb_palette': gt_rgb_palette,
            'gt_hex_palette': gt_hex_palette,
            'examples': examples
        })
    
    print(f"\n✓ Prepared {len(prepared_samples)} prompts")
    print("\n" + "="*50)
    print("Ready to start API calls.")
    print(f"This will make {len(prepared_samples)} requests to the model.")
    print(f"Current model: {model_name}")
    print("="*50)
    
    # Wait for user confirmation and optional model change
    user_input = input("\nEnter a different model name to use, or press Enter to continue with current model (Ctrl+C to cancel): ").strip()
    
    # Update model if user provided one
    if user_input:
        model_name = user_input
        print(f"\n✓ Switching to model: {model_name}")
        client = RequestsModelClient(
            model=model_name,
            api_key=api_key,
            max_retries=max_retries,
            retry_delay=retry_delay,
            request_delay=request_delay,
            color_format=color_format
        )
    print()
    
    # Phase 2: Run API calls
    print("=== Phase 2: Running API calls ===")
    results = []
    
    for sample in tqdm(prepared_samples, desc="Generating palettes"):
        i = sample['index']
        query_str = sample['query_str']
        gt_rgb_palette = sample['gt_rgb_palette']
        gt_hex_palette = sample['gt_hex_palette']
        examples = sample['examples']
        
        generated_hex, generated_text, response = client.generate_palette(query_str, examples)
        
        if len(generated_hex) != 5:
            print(f"Warning: Sample {i} generated {len(generated_hex)} colors instead of 5")
            # Pad or truncate to 5 colors
            if len(generated_hex) < 5:
                generated_hex = generated_hex + ['#000000'] * (5 - len(generated_hex))
            else:
                generated_hex = generated_hex[:5]
        
        # Calculate metrics using grader.py directly
        try:
            dccw_measurer = DccwMeasurer(
                source=generated_hex,
                source_option='hex',
                target=gt_hex_palette.to_hex_list(),
                target_option='hex'
            )
            
            dccw_score_no_cycle = dccw_measurer.measure_dccw(reflect_cycle=False)
            source_diversity = dccw_measurer.calculate_source_diversity()
            target_diversity = dccw_measurer.calculate_target_diversity()
            
            # Using constants compatible with grader.py
            norm_D = normalize_inv_map(abs(target_diversity - source_diversity), tau=15.0, k=0.2)
            norm_S = normalize_inv_map(dccw_score_no_cycle, tau=26.0, k=0.15)
            score_R = harmonic_mean(norm_D, norm_S)
            
            metrics = {
                'norm_D': norm_D,
                'norm_S': norm_S,
                'score_R': score_R,
                'gt_diversity': target_diversity,
                'gen_diversity': source_diversity,
                'raw_dccw': dccw_score_no_cycle,
                'diversity_diff': abs(target_diversity - source_diversity)
            }
        except Exception as e:
            print(f"Error calculating metrics sample {i}: {e}")
            metrics = {
                'norm_D': float('nan'),
                'norm_S': float('nan'),
                'score_R': float('nan'),
                'gt_diversity': float('nan'),
                'gen_diversity': float('nan'),
                'raw_dccw': float('nan'),
                'diversity_diff': float('nan')
            }
        
        # Store result
        result = {
            'sample_index': i,
            'text_query': query_str,
            'ground_truth_palette': str(gt_hex_palette),
            'ground_truth_rgb': str(gt_rgb_palette),
            'generated_palette': str(generated_hex),
            'generated_text': ' '.join(str(response).splitlines()),
            'generated_palette_text': str(generated_text),
            'norm_D': metrics['norm_D'],
            'norm_S': metrics['norm_S'],
            'score_R': metrics['score_R'],
            'model': model_name,
            'provider': provider,
            'gt_diversity': metrics['gt_diversity'],
            'gen_diversity': metrics['gen_diversity'],
            'raw_dccw': metrics['raw_dccw'],
            'diversity_diff': metrics['diversity_diff']
        }
        results.append(result)
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Calculate summary statistics
    print("\n=== Summary Statistics ===")
    print(f"Total samples: {len(df)}")
    print(f"Mean norm_D: {df['norm_D'].mean():.4f}")
    print(f"Mean norm_S: {df['norm_S'].mean():.4f}")
    print(f"Mean score_R: {df['score_R'].mean():.4f}")
    print(f"Std norm_D: {df['norm_D'].std():.4f}")
    print(f"Std norm_S: {df['norm_S'].std():.4f}")
    print(f"Std score_R: {df['score_R'].std():.4f}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_safe_name = model_name.replace("/", "_").replace(":", "_")
    filename = f"eval_{provider}_{model_safe_name}_{timestamp}.csv"
    filepath = os.path.join(output_dir, filename)
    
    df.to_csv(filepath, index=False)
    print(f"\nResults saved to: {filepath}")
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate color palette generation models using direct requests.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m ml.evaluator.evaluate2 --provider fireworks --model accounts/sidvenkatayogi/deployedModels/rft-fhiaqwgu-0-1-ka2pgv73
    python -m ml.evaluator.evaluate2 --provider fireworks --model accounts/sidvenkatayogi/deployedModels/rft-fhiaqwgu-0-1-ka2pgv73 --limit 10
        """
    )
    parser.add_argument(
        "--provider",
        type=str,
        required=True,
        choices=[PROVIDER_OPENAI, PROVIDER_FIREWORKS],
        help="Model provider (currently only fireworks supported)"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name (required)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save results (default: ml/evaluator/results)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of samples to evaluate (for testing)"
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help=f"Max retries on rate limit errors (default: {DEFAULT_MAX_RETRIES})"
    )
    parser.add_argument(
        "--retry-delay",
        type=float,
        default=DEFAULT_RETRY_DELAY,
        help=f"Initial retry delay in seconds (default: {DEFAULT_RETRY_DELAY})"
    )
    parser.add_argument(
        "--request-delay",
        type=float,
        default=DEFAULT_REQUEST_DELAY,
        help=f"Delay between requests in seconds (default: {DEFAULT_REQUEST_DELAY})"
    )
    parser.add_argument(
        "--color-format",
        type=str,
        default="hex",
        choices=["hex", "hsl", "cielab", "lab"],
        help="Color format to use for extraction (default: hex)"
    )
    
    args = parser.parse_args()
    
    run_evaluation(
        provider=args.provider,
        model=args.model,
        output_dir=args.output_dir,
        limit=args.limit,
        max_retries=args.max_retries,
        retry_delay=args.retry_delay,
        request_delay=args.request_delay,
        color_format=args.color_format
    )


if __name__ == "__main__":
    main()
