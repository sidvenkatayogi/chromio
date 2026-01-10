#!/usr/bin/env python3
"""
Script to rescore diversity metric (norm_D) in evaluation results.
"""

import os
import sys
import pandas as pd
import glob

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from ml.grader.evaluation import normalize_inv_map


def rescore_diversity(csv_path: str, tau: float = 15.0, k: float = 0.2):
    """
    Rescore the diversity metric (norm_D) in the evaluation results CSV.
    
    Args:
        csv_path: Path to the CSV file
        tau: Tau parameter for normalize_inv_map
        k: K parameter for normalize_inv_map
    """
    print(f"\n{'='*60}")
    print(f"Processing: {os.path.basename(csv_path)}")
    print(f"{'='*60}")
    
    try:
        df = pd.read_csv(csv_path)
        
        print(f"Total rows: {len(df)}")
        print(f"Rescoring with tau={tau}, k={k}")
        
        # Recalculate norm_D using diversity_diff
        df['norm_D'] = df['diversity_diff'].apply(
            lambda x: normalize_inv_map(x, tau=tau, k=k)
        )
        
        # Recalculate score_R (harmonic mean of norm_D and norm_S)
        from ml.grader.evaluation import harmonic_mean
        df['score_R'] = df.apply(
            lambda row: harmonic_mean(row['norm_D'], row['norm_S']),
            axis=1
        )
        
        # Print summary statistics
        print("\n=== Updated Summary Statistics ===")
        print(f"Mean norm_D: {df['norm_D'].mean():.4f}")
        print(f"Mean norm_S: {df['norm_S'].mean():.4f}")
        print(f"Mean score_R: {df['score_R'].mean():.4f}")
        print(f"Std norm_D: {df['norm_D'].std():.4f}")
        print(f"Std norm_S: {df['norm_S'].std():.4f}")
        print(f"Std score_R: {df['score_R'].std():.4f}")
        
        # Save updated CSV
        df.to_csv(csv_path, index=False)
        print(f"✓ Updated and saved successfully")
        
        return df
    except Exception as e:
        print(f"✗ Error processing file: {e}")
        return None


if __name__ == "__main__":
    results_dir = os.path.join(PROJECT_ROOT, "ml/evaluator/results")
    
    # Get all CSV files in results directory
    csv_files = glob.glob(os.path.join(results_dir, "*.csv"))
    
    print(f"Found {len(csv_files)} CSV files to process in {results_dir}")
    
    processed = 0
    failed = 0
    
    for csv_path in sorted(csv_files):
        result = rescore_diversity(csv_path, tau=15.0, k=0.2)
        if result is not None:
            processed += 1
        else:
            failed += 1
    
    print(f"\n{'='*60}")
    print(f"SUMMARY: Processed {processed} files, {failed} failed")
    print(f"{'='*60}")
