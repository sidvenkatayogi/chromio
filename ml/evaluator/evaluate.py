#!/usr/bin/env python3
"""
Main evaluation script for color palette generation models.

Usage:
    python -m ml.evaluator.evaluate --provider openai --model gpt-4o-mini
    python -m ml.evaluator.evaluate --provider fireworks --model accounts/fireworks/models/qwen3-8b
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

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from ml.evaluator.config import (
    PROVIDER_OPENAI, PROVIDER_FIREWORKS, DEFAULT_MODELS,
    EVAL_DATA_DIR, TEST_NAMES_FILE, TEST_PALETTES_FILE, RESULTS_DIR,
    DEFAULT_MAX_RETRIES, DEFAULT_RETRY_DELAY, DEFAULT_REQUEST_DELAY
)
from ml.evaluator.model_clients import get_model_client
from ml.grader.evaluation import Color, ColorPalette, DccwMeasurer, normalize_inv_map, harmonic_mean

import chromadb
from chromadb.utils import embedding_functions

CHROMA_PATH = "chroma_db"
CHROMA_PATH_HSL = "chroma_db_hsl"
COLLECTION_NAME = "pat"
MODEL_NAME = "all-mpnet-base-v2"


def run_evaluation(provider: str, model: str = None, output_dir: str = None, limit: int = None,
                   max_retries: int = DEFAULT_MAX_RETRIES, retry_delay: float = DEFAULT_RETRY_DELAY,
                   request_delay: float = DEFAULT_REQUEST_DELAY, color_format: str = 'hex'):
    """
    Run the full evaluation pipeline.
    
    Args:
        provider: Model provider ('openai' or 'fireworks')
        model: Specific model name (optional, uses default if not provided)
        output_dir: Directory to save results (optional)
        limit: Maximum number of samples to evaluate (optional, for testing)
        max_retries: Maximum number of retries on rate limit errors
        retry_delay: Initial delay between retries (seconds)
        request_delay: Delay between requests to avoid rate limits (seconds)
        color_format: Color format to use ('hex' or 'hsl')
    """
    # Setup paths
    data_dir = os.path.join(PROJECT_ROOT, EVAL_DATA_DIR)
    output_dir = output_dir or os.path.join(PROJECT_ROOT, RESULTS_DIR)
    os.makedirs(output_dir, exist_ok=True)
    
    # Get model name
    model_name = model or DEFAULT_MODELS.get(provider)

    # if provider == "fireworks" and model:
    #     model_name = "accounts/fireworks/models/" + model

    if not model_name:
        raise ValueError(f"No default model for provider: {provider}")
    
    print(f"=== Model Evaluation ===")
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
    
    # return test_names, test_palettes
    # test_names, test_palettes = load_eval_data(data_dir, TEST_NAMES_FILE, TEST_PALETTES_FILE)
    
    if limit:
        test_names = test_names[:limit]
        test_palettes = test_palettes[:limit]
    
    print(f"Loaded {len(test_names)} test samples")
    print()
    
    # Initialize model client
    print("Initializing model client...")
    print(f"Color format: {color_format}")
    print(f"Rate limiting: max_retries={max_retries}, retry_delay={retry_delay}s, request_delay={request_delay}s")
    client = get_model_client(
        provider, model_name,
        max_retries=max_retries,
        retry_delay=retry_delay,
        request_delay=request_delay,
        color_format=color_format
    )
    
    print("Initializing example fetcher (loading embedding model)...")
    chroma_path = os.path.join(PROJECT_ROOT, CHROMA_PATH_HSL if color_format == 'hsl' else CHROMA_PATH)
        
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
    
    # Run evaluation
    print("Running evaluation...")
    results = []
    
    for i, (query_words, gt_rgb_palette) in enumerate(tqdm(zip(test_names, test_palettes), total=len(test_names))):
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
        description="Evaluate color palette generation models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python evaluate.py --provider openai --model gpt-4o-mini
    python evaluate.py --provider fireworks --model accounts/fireworks/models/llama-v3p1-8b-instruct
    python evaluate.py --provider openai --limit 10  # Test with first 10 samples
        """
    )
    parser.add_argument(
        "--provider",
        type=str,
        required=True,
        choices=[PROVIDER_OPENAI, PROVIDER_FIREWORKS],
        help="Model provider (openai or fireworks)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name (uses default for provider if not specified)"
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
        choices=["hex", "hsl"],
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
