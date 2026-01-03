"""
Data loading utilities for evaluation.
"""

import pickle
import os
from typing import List, Tuple


def load_eval_data(data_dir: str, names_file: str, palettes_file: str) -> Tuple[List[List[str]], List[List[Tuple[int, int, int]]]]:
    """
    Load evaluation data from pickle files.
    
    Args:
        data_dir: Directory containing the data files
        names_file: Filename for test names/queries
        palettes_file: Filename for ground truth RGB palettes
    
    Returns:
        Tuple of (test_names, test_palettes)
        - test_names: List of lists of words (queries)
        - test_palettes: List of lists of RGB tuples
    """
    names_path = os.path.join(data_dir, names_file)
    palettes_path = os.path.join(data_dir, palettes_file)
    
    with open(names_path, 'rb') as f:
        test_names = pickle.load(f)
    
    with open(palettes_path, 'rb') as f:
        test_palettes = pickle.load(f)
    
    return test_names, test_palettes


def query_to_string(query_words: List[str]) -> str:
    """Convert a list of query words to a single string."""
    return " ".join(query_words)


def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """Convert hex color to RGB tuple."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    """Convert RGB tuple to hex color."""
    return "#{:02x}{:02x}{:02x}".format(rgb[0], rgb[1], rgb[2])
