#!/usr/bin/env python3
"""
Create JSONL evaluation dataset from test samples.

Usage:
    python -m ml.evaluator.create_eval_dataset
    python -m ml.evaluator.create_eval_dataset --limit 10
"""

import os
import sys

# Suppress tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import pickle
import json
from datetime import datetime
from tqdm import tqdm
import logging

# Suppress ChromaDB verbose output
logging.getLogger("chromadb").setLevel(logging.ERROR)

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from ml.evaluator.config import (
    EVAL_DATA_DIR, TEST_NAMES_FILE, TEST_PALETTES_FILE, RESULTS_DIR,
    SYSTEM_PROMPT
)
from ml.grader.evaluation import ColorPalette, DccwMeasurer

import chromadb
from chromadb.utils import embedding_functions

CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "pat"
MODEL_NAME = "all-mpnet-base-v2"


def create_dataset(output_dir: str = None, limit: int = None):
    """
    Create JSONL dataset from test samples with retrieved examples.
    
    Args:
        output_dir: Directory to save dataset (optional)
        limit: Maximum number of samples to process (optional, for testing)
    """
    # Setup paths
    data_dir = os.path.join(PROJECT_ROOT, EVAL_DATA_DIR)
    output_dir = output_dir or os.path.join(PROJECT_ROOT, RESULTS_DIR)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"=== Creating Evaluation Dataset ===")
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Load evaluation data
    print("Loading test samples...")
    
    with open(os.path.join(data_dir, TEST_NAMES_FILE), 'rb') as f:
        test_names = pickle.load(f)
    
    with open(os.path.join(data_dir, TEST_PALETTES_FILE), 'rb') as f:
        test_palettes = pickle.load(f)
    
    if limit:
        test_names = test_names[:limit]
        test_palettes = test_palettes[:limit]
    
    print(f"Loaded {len(test_names)} test samples")
    print()
    
    # Initialize ChromaDB
    print("Initializing ChromaDB (loading embedding model)...")
    chroma_path = os.path.join(PROJECT_ROOT, CHROMA_PATH)
    
    chroma_client = chromadb.PersistentClient(path=chroma_path)
    emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=MODEL_NAME
    )
    collection = chroma_client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=emb_fn
    )
    print()
    
    # Process samples and create dataset
    print("Retrieving examples and creating dataset...")
    dataset_entries = []
    
    for i, (query_words, gt_rgb_palette) in enumerate(tqdm(zip(test_names, test_palettes), total=len(test_names), desc="Processing")):
        query_str = " ".join(query_words)
        gt_hex_palette = ColorPalette(gt_rgb_palette)
        
        # Retrieve similar examples
        retrieval_results = collection.query(
            query_texts=[query_str],
            n_results=3
        )
        
        # Format examples
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
        
        # Format user message
        user_content = f"\nWhat's the best color palette consisting of five colors to describe the text {query_str}?\nProvide the color values using text (hex) format in ascending order.\n\nHere are some associate text-palette pairs for reference:\n### REFERENCE PALETTES\n{examples}\n"
        
        # Calculate ground truth diversity
        try:
            dccw_measurer = DccwMeasurer(
                source=gt_hex_palette.to_hex_list(),
                source_option='hex',
                target=gt_hex_palette.to_hex_list(),
                target_option='hex'
            )
            gt_diversity = dccw_measurer.calculate_source_diversity()
        except:
            gt_diversity = 0.0
        
        # Create JSONL entry
        entry = {
            "messages": [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": user_content
                }
            ],
            "gt_palette": gt_hex_palette.to_hex_list(),
            "gt_diversity": gt_diversity
        }
        dataset_entries.append(entry)
    
    # Save dataset
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    jsonl_filename = f"eval_prompts_{timestamp}.jsonl"
    jsonl_filepath = os.path.join(output_dir, jsonl_filename)
    
    with open(jsonl_filepath, 'w') as f:
        for entry in dataset_entries:
            f.write(json.dumps(entry) + '\n')
    
    print(f"\n✓ Dataset created successfully!")
    print(f"✓ Saved {len(dataset_entries)} entries to: {jsonl_filepath}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Create JSONL evaluation dataset from test samples.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m ml.evaluator.create_eval_dataset
    python -m ml.evaluator.create_eval_dataset --limit 10
    python -m ml.evaluator.create_eval_dataset --output-dir custom/path
        """
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save dataset (default: ml/evaluator/results2)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of samples to process (for testing)"
    )
    
    args = parser.parse_args()
    
    create_dataset(
        output_dir=args.output_dir,
        limit=args.limit
    )


if __name__ == "__main__":
    main()
