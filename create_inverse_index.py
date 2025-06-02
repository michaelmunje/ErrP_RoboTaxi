#!/usr/bin/env python3
"""
Create inverse index from i2i mapping file.
Maps 10, 20, 30, ..., 210 to first occurrence of corresponding first index.
This is used to create an additional x axis when plotting the intervention to cumulative reward
Curve. We could use a second x axis because only a subset of the intervention steps are from true
interventions.
"""

import sys
import re

def create_inverse_index(i2i_file_path):
    """Create inverse index from i2i mapping file."""
    
    # Create output filename
    output_file = i2i_file_path.replace('_i2i.log', '_inverse_index.txt')
    
    # Target values: 10, 20, 30, ..., 210
    target_values = list(range(10, 211, 10))
    inverse_index = {}
    
    with open(i2i_file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
                
            # Parse (noisy_count, non_noisy_count)
            match = re.match(r'\((\d+),(\d+)\)', line)
            if match:
                noisy_count = int(match.group(1))
                non_noisy_count = int(match.group(2))
                
                # Check if this non_noisy_count is one of our targets
                # and we haven't seen it before
                if non_noisy_count in target_values and non_noisy_count not in inverse_index:
                    inverse_index[non_noisy_count] = noisy_count
    
    # Write results
    with open(output_file, 'w') as outfile:
        for target in target_values:
            if target in inverse_index:
                outfile.write(f"{inverse_index[target]}\n")
            else:
                outfile.write("N/A\n")  # In case target value never reached
    
    print(f"Inverse index written to: {output_file}")
    
    # Also print to console
    print("\nInverse index mapping:")
    for target in target_values:
        if target in inverse_index:
            print(f"{target} -> {inverse_index[target]}")
        else:
            print(f"{target} -> N/A")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python create_inverse_index.py <i2i_file>")
        sys.exit(1)
    
    create_inverse_index(sys.argv[1]) 