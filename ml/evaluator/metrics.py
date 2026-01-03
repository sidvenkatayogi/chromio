"""
Metrics calculation using the grader module.
"""

import sys
import os
import math

# Add parent directory to path to import grader
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from grader.grader import DccwMeasurer, normalize_inv_map, harmonic_mean


def calculate_metrics(generated_hex_list: list, ground_truth_hex_list: list) -> dict:
    """
    Calculate norm_D, norm_S, and score_R metrics.
    
    Args:
        generated_hex_list: List of 5 hex color strings from the model
        ground_truth_hex_list: List of 5 hex color strings (ground truth)
    
    Returns:
        Dictionary with norm_D, norm_S, and score_R
    """
    try:
        dccw_measurer = DccwMeasurer(
            source=generated_hex_list,
            source_option='hex',
            target=ground_truth_hex_list,
            target_option='hex'
        )
        
        dccw_score_no_cycle = dccw_measurer.measure_dccw(reflect_cycle=False)
        source_diversity = dccw_measurer.calculate_source_diversity()
        target_diversity = dccw_measurer.calculate_target_diversity()
        
        norm_D = normalize_inv_map(abs(target_diversity - source_diversity), tau=10.0, k=0.2)
        norm_S = normalize_inv_map(dccw_score_no_cycle, tau=26.0, k=0.15)
        score_R = harmonic_mean(norm_D, norm_S)
        
        return {
            'norm_D': norm_D,
            'norm_S': norm_S,
            'score_R': score_R
        }
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        return {
            'norm_D': float('nan'),
            'norm_S': float('nan'),
            'score_R': float('nan')
        }


def rgb_palette_to_hex(rgb_palette: list) -> list:
    """Convert a list of RGB tuples to hex strings."""
    hex_list = []
    for rgb in rgb_palette:
        hex_color = "#{:02x}{:02x}{:02x}".format(int(rgb[0]), int(rgb[1]), int(rgb[2]))
        hex_list.append(hex_color)
    return hex_list
