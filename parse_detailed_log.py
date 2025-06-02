#!/usr/bin/env python3
"""
Simple parser for detailed.log files to create mapping between 
cumulative non-zero noisy rewards and non-zero non-noisy rewards.
"""

import re
import sys

def parse_log_and_create_mapping(log_file_path):
    """Parse log file and create the requested mapping."""
    
    # Create output filename with _i2i.log suffix
    output_file = log_file_path.replace('_detailed.log', '_detailed_i2i.log')
    
    cumulative_noisy = 0
    cumulative_non_noisy = 0
    
    # Pattern to extract: np.int64(noisy_reward), non_noisy_reward)
    # non_noisy_reward can be either np.int64(value) or just a plain integer
    pattern = r'np\.int64\((-?\d+)\),\s*(?:np\.int64\((-?\d+)\)|(-?\d+))\)'
    
    with open(log_file_path, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            if not line.startswith('[state, action, reward]'):
                continue
                
            match = re.search(pattern, line)
            if match:
                noisy_reward = int(match.group(1))
                # non_noisy_reward can be in group 2 (np.int64) or group 3 (plain int)
                non_noisy_reward = int(match.group(2) if match.group(2) is not None else match.group(3))
                
                if noisy_reward != 0:
                    cumulative_noisy += 1
                if non_noisy_reward != 0:
                    cumulative_non_noisy += 1
                    
                outfile.write(f"({cumulative_noisy},{cumulative_non_noisy})\n")
    
    print(f"Output written to: {output_file}")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python parse_detailed_log.py <log_file>")
        sys.exit(1)
    
    parse_log_and_create_mapping(sys.argv[1]) 