"""
Script to add raw DCCW and diversity scores to evaluation result CSVs.

Adds 4 new fields:
- gt_diversity: Ground truth palette diversity score
- gen_diversity: Generated palette diversity score  
- raw_dccw: Raw DCCW score between ground truth and generated palettes
- diversity_diff: Absolute difference in diversity scores
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
import ast
from pathlib import Path

from ml.grader.evaluation import ColorPalette, DccwMeasurer


RESULTS_DIR = Path(__file__).parent / 'results'


def parse_palette(palette_str):
    """Parse palette string to list of hex colors."""
    try:
        palette = ast.literal_eval(palette_str)
        return palette
    except:
        return None


def calculate_scores(gt_palette_str, gen_palette_str):
    """Calculate raw DCCW and diversity scores for a palette pair."""
    gt_palette = parse_palette(gt_palette_str)
    gen_palette = parse_palette(gen_palette_str)
    
    if gt_palette is None or gen_palette is None:
        return None, None, None, None
    
    try:
        # Create DccwMeasurer to calculate scores
        dccw_measurer = DccwMeasurer(
            source=gt_palette,
            source_option='hex',
            target=gen_palette,
            target_option='hex'
        )
        
        # Get diversity scores
        gt_diversity = dccw_measurer.calculate_source_diversity()
        gen_diversity = dccw_measurer.calculate_target_diversity()
        
        # Get raw DCCW score
        raw_dccw = dccw_measurer.measure_dccw(reflect_cycle=False)
        
        # Calculate diversity difference
        diversity_diff = abs(gen_diversity - gt_diversity)
        
        return gt_diversity, gen_diversity, raw_dccw, diversity_diff
        
    except Exception as e:
        print(f"Error calculating scores: {e}")
        return None, None, None, None


def process_csv(csv_path):
    """Process a single CSV file and add raw score columns."""
    print(f"\nProcessing: {csv_path.name}")
    
    df = pd.read_csv(csv_path)
    
    # Initialize new columns
    gt_diversities = []
    gen_diversities = []
    raw_dccws = []
    diversity_diffs = []
    
    for idx, row in df.iterrows():
        gt_diversity, gen_diversity, raw_dccw, diversity_diff = calculate_scores(
            row['ground_truth_palette'],
            row['generated_palette']
        )
        
        gt_diversities.append(gt_diversity)
        gen_diversities.append(gen_diversity)
        raw_dccws.append(raw_dccw)
        diversity_diffs.append(diversity_diff)
        
        if (idx + 1) % 20 == 0:
            print(f"  Processed {idx + 1}/{len(df)} samples")
    
    # Add new columns
    df['gt_diversity'] = gt_diversities
    df['gen_diversity'] = gen_diversities
    df['raw_dccw'] = raw_dccws
    df['diversity_diff'] = diversity_diffs
    
    # Save back to CSV
    df.to_csv(csv_path, index=False)
    print(f"  Saved with {len(df)} samples")
    
    # Print summary stats
    valid_mask = df['gt_diversity'].notna()
    if valid_mask.sum() > 0:
        print(f"  GT Diversity: {df.loc[valid_mask, 'gt_diversity'].mean():.2f} (mean)")
        print(f"  Gen Diversity: {df.loc[valid_mask, 'gen_diversity'].mean():.2f} (mean)")
        print(f"  Raw DCCW: {df.loc[valid_mask, 'raw_dccw'].mean():.2f} (mean)")


def main():
    csv_files = sorted(RESULTS_DIR.glob('*.csv'))
    print(f"Found {len(csv_files)} CSV files to process")
    
    for csv_file in csv_files:
        process_csv(csv_file)
    
    print("\n✓ All files processed successfully!")


if __name__ == "__main__":
    main()
