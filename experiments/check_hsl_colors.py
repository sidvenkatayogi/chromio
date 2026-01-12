#!/usr/bin/env python3
"""Check for valid HSL colors in qwen.jsonl assistant responses."""

import json
import re
import sys

def is_valid_hsl(color_str):
    """Check if a color string is in valid HSL format: (H, S%, L%)"""
    pattern = r'^\((\d+),\s*(\d+)%,\s*(\d+)%\)$'
    match = re.match(pattern, color_str)
    
    if not match:
        return False, "Invalid format"
    
    h, s, l = map(int, match.groups())
    
    if not (0 <= h < 360):
        return False, f"H={h} out of range [0, 360)"
    if not (0 <= s <= 100):
        return False, f"S={s} out of range [0, 100]"
    if not (0 <= l <= 100):
        return False, f"L={l} out of range [0, 100]"
    
    return True, "Valid"

def check_line(line_num, line):
    """Check a single line for valid HSL colors."""
    try:
        data = json.loads(line)
    except json.JSONDecodeError as e:
        return f"Line {line_num}: JSON parse error - {e}"
    
    # Get assistant response
    responses = data.get('responses', {})
    content = responses.get('content', '')
    
    # Try to extract JSON from content
    try:
        # Find JSON in the content
        json_match = re.search(r'\{[^}]*"palette_hex"[^}]*\}', content, re.DOTALL)
        if not json_match:
            return f"Line {line_num}: No palette_hex found in response"
        
        palette_json = json.loads(json_match.group(0))
        palette_hex = palette_json.get('palette_hex', [])
        
        if not isinstance(palette_hex, list):
            return f"Line {line_num}: palette_hex is not a list"
        
        if len(palette_hex) != 5:
            return f"Line {line_num}: Expected 5 colors, got {len(palette_hex)}"
        
        # Check each color
        invalid_colors = []
        for i, color in enumerate(palette_hex):
            is_valid, reason = is_valid_hsl(color)
            if not is_valid:
                invalid_colors.append(f"Color {i+1}: '{color}' - {reason}")
        
        if invalid_colors:
            return f"Line {line_num}: Invalid colors:\n  " + "\n  ".join(invalid_colors)
        
        return None  # Valid
        
    except json.JSONDecodeError as e:
        return f"Line {line_num}: Could not parse palette JSON - {e}"
    except Exception as e:
        return f"Line {line_num}: Error - {e}"

def main():
    input_file = sys.argv[1] if len(sys.argv) > 1 else 'ml/evaluator/batch_results/qwen.jsonl'
    
    print(f"Checking {input_file} for invalid HSL colors...\n")
    
    invalid_count = 0
    total_count = 0
    
    with open(input_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            total_count += 1
            error = check_line(line_num, line)
            
            if error:
                invalid_count += 1
                print(error)
                print()
    
    print(f"\n{'='*60}")
    print(f"Total lines: {total_count}")
    print(f"Invalid lines: {invalid_count}")
    print(f"Valid lines: {total_count - invalid_count}")

if __name__ == '__main__':
    main()
