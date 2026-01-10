#!/usr/bin/env python3
"""
Process batch results JSONL file and create CSV with evaluation metrics.

Usage:
    python -m ml.evaluator.process_batch_results --input ml/evaluator/batch_results/BIJOutputSet.jsonl
"""

import os
import sys
import argparse
import json
import pandas as pd
import re
from datetime import datetime

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from ml.grader.evaluation import ColorPalette, DccwMeasurer, normalize_inv_map, harmonic_mean


def extract_hex_from_string(content: str) -> list:
    """Extract hex colors directly from string using regex."""
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


def extract_palette_text(data: dict) -> list:
    """Recursively extract palette_text from potentially nested JSON."""
    if isinstance(data, dict):
        if 'palette_text' in data:
            return data['palette_text']
        
        for key, value in data.items():
            if isinstance(value, dict):
                result = extract_palette_text(value)
                if result:
                    return result
    return []


def extract_query_text(user_message: str) -> str:
    """Extract the query text from the user message."""
    # Pattern: "describe the text <QUERY>?"
    match = re.search(r'describe the text (.+?)\?', user_message)
    if match:
        return match.group(1).strip()
    return "unknown"


def process_batch_results(input_file: str, output_dir: str = None, provider: str = "fireworks", 
                         model: str = "batch_model", color_format: str = "hex"):
    """
    Process batch results JSONL and create evaluation CSV.
    
    Args:
        input_file: Path to the batch results JSONL file
        output_dir: Directory to save results CSV (optional)
        provider: Provider name for the model
        model: Model name
        color_format: Color format to use ('hex' or 'hsl')
    """
    # Setup output directory
    if output_dir is None:
        output_dir = os.path.join(PROJECT_ROOT, "ml", "evaluator", "results")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"=== Processing Batch Results ===")
    print(f"Input file: {input_file}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Read JSONL file
    print("Reading batch results...")
    entries = []
    with open(input_file, 'r') as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))
    
    print(f"Loaded {len(entries)} entries")
    print()
    
    # Process each entry
    print("Processing entries and calculating metrics...")
    results = []
    
    for i, entry in enumerate(entries):
        # Extract ground truth palette (handle both hex and HSL formats)
        gt_palette_raw = entry['gt_palette']
        
        # Check if ground truth is in HSL or hex format
        if isinstance(gt_palette_raw, list) and len(gt_palette_raw) > 0:
            # Check if first item looks like HSL format
            first_item = str(gt_palette_raw[0])
            if first_item.startswith('(') and '%' in first_item:
                # HSL format - convert to hex
                gt_hex_palette = [hsl_to_hex(hsl) for hsl in gt_palette_raw]
            else:
                # Already hex format
                gt_hex_palette = gt_palette_raw
        else:
            gt_hex_palette = gt_palette_raw
        
        gt_palette_obj = ColorPalette([tuple(int(h[j:j+2], 16) for j in (1, 3, 5)) for h in gt_hex_palette])
        
        # Extract query text from user message
        user_message = entry['original_messages'][1]['content']
        query_str = extract_query_text(user_message)
        
        # Extract generated palette from assistant response
        assistant_content = entry['responses']['content']
        
        # Extract colors based on format
        if color_format.lower() == 'hsl':
            generated_hsl = extract_hsl_from_string(assistant_content)
            # Convert HSL to hex for evaluation
            generated_hex = [hsl_to_hex(hsl) for hsl in generated_hsl]
        else:
            generated_hex = extract_hex_from_string(assistant_content)
        
        # Extract palette text names
        try:
            content_json = json.loads(assistant_content.split('\n\n')[1] if '\n\n' in assistant_content else assistant_content)
            generated_text = extract_palette_text(content_json)
        except:
            generated_text = []
        
        # Pad or truncate to 5 colors if needed
        if len(generated_hex) != 5:
            print(f"Warning: Entry {i} generated {len(generated_hex)} colors instead of 5")
            if len(generated_hex) < 5:
                generated_hex = generated_hex + ['#000000'] * (5 - len(generated_hex))
            else:
                generated_hex = generated_hex[:5]
        
        # Calculate metrics
        try:
            dccw_measurer = DccwMeasurer(
                source=generated_hex,
                source_option='hex',
                target=gt_hex_palette,
                target_option='hex'
            )
            
            dccw_score_no_cycle = dccw_measurer.measure_dccw(reflect_cycle=False)
            source_diversity = dccw_measurer.calculate_source_diversity()
            target_diversity = dccw_measurer.calculate_target_diversity()
            
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
            print(f"Error calculating metrics for entry {i}: {e}")
            metrics = {
                'norm_D': float('nan'),
                'norm_S': float('nan'),
                'score_R': float('nan'),
                'gt_diversity': float('nan'),
                'gen_diversity': float('nan'),
                'raw_dccw': float('nan'),
                'diversity_diff': float('nan')
            }
        
        # Convert ground truth hex to RGB for display
        gt_rgb = [tuple(int(h[j:j+2], 16) for j in (1, 3, 5)) for h in gt_hex_palette]
        
        # Store result
        result = {
            'sample_index': i,
            'text_query': query_str,
            'ground_truth_palette': str(gt_hex_palette),
            'ground_truth_rgb': str(gt_rgb),
            'generated_palette': str(generated_hex),
            'generated_text': str(generated_text),
            'norm_D': metrics['norm_D'],
            'norm_S': metrics['norm_S'],
            'score_R': metrics['score_R'],
            'model': model,
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
    model_safe_name = model.replace("/", "_").replace(":", "_")
    filename = f"eval_{provider}_{model_safe_name}_{timestamp}.csv"
    filepath = os.path.join(output_dir, filename)
    
    df.to_csv(filepath, index=False)
    print(f"\nResults saved to: {filepath}")
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Process batch results JSONL and create evaluation CSV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m ml.evaluator.process_batch_results --input ml/evaluator/batch_results/BIJOutputSet.jsonl
    python -m ml.evaluator.process_batch_results --input ml/evaluator/batch_results/f.jsonl --model accounts/fireworks/models/qwen3-8b_HSL
        """
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to batch results JSONL file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save results CSV (default: ml/evaluator/results)"
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="fireworks",
        help="Provider name (default: fireworks)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="batch_model",
        help="Model name (default: batch_model)"
    )
    parser.add_argument(
        "--color-format",
        type=str,
        default="hex",
        choices=["hex", "hsl"],
        help="Color format to use for extraction (default: hex)"
    )
    
    args = parser.parse_args()
    
    process_batch_results(
        input_file=args.input,
        output_dir=args.output_dir,
        provider=args.provider,
        model=args.model,
        color_format=args.color_format
    )


if __name__ == "__main__":
    main()
