#!/usr/bin/env python3
"""
Split error log into punctuation errors and other errors.
"""
import re
from pathlib import Path
from typing import Set

# Define common Urdu and ASCII punctuation marks
PUNCTUATION_MARKS: Set[str] = {
    '،', '؟', '۔', '!', '.', ',', ':', ';', '?', '!', 
    '-', '—', '–', '"', '"', "'", ''', ''', '(', ')', 
    '[', ']', '{', '}', '/', '\\', '|', '*', '&', '@',
    '#', '$', '%', '^', '`', '~', '<', '>'
}

def is_punctuation_error(line: str) -> bool:
    """
    Determine if an error line is related to punctuation.
    
    Checks for:
    - Inserted/deleted punctuation marks
    - Substitution to/from punctuation
    """
    # Check for inserted word being punctuation
    if "inserted word:" in line:
        match = re.search(r'inserted word: (.+?) at', line)
        if match:
            word = match.group(1).strip()
            return word in PUNCTUATION_MARKS
    
    # Check for deleted word being punctuation
    if "deleted word:" in line:
        match = re.search(r'deleted word: (.+?) at', line)
        if match:
            word = match.group(1).strip()
            return word in PUNCTUATION_MARKS
    
    # Check for substitution involving punctuation
    if "substitution:" in line:
        match = re.search(r'substitution: (.+?) -> (.+?) at', line)
        if match:
            src_word = match.group(1).strip()
            tgt_word = match.group(2).strip()
            # If either side is pure punctuation, consider it punctuation error
            return (src_word in PUNCTUATION_MARKS or 
                    tgt_word in PUNCTUATION_MARKS or
                    all(c in PUNCTUATION_MARKS for c in src_word) or
                    all(c in PUNCTUATION_MARKS for c in tgt_word))
    
    return False

def split_error_log(input_path: Path, 
                   punctuation_output: Path, 
                   other_output: Path) -> tuple:
    """
    Split error log into punctuation and non-punctuation errors.
    
    Returns:
        (punctuation_count, other_count)
    """
    punctuation_errors = []
    other_errors = []
    
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')
            if is_punctuation_error(line):
                punctuation_errors.append(line)
            else:
                other_errors.append(line)
    
    # Write punctuation errors
    with open(punctuation_output, 'w', encoding='utf-8') as f:
        f.write('\n'.join(punctuation_errors))
        if punctuation_errors:
            f.write('\n')
    
    # Write other errors
    with open(other_output, 'w', encoding='utf-8') as f:
        f.write('\n'.join(other_errors))
        if other_errors:
            f.write('\n')
    
    return len(punctuation_errors), len(other_errors)

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Split error log into punctuation and other errors'
    )
    parser.add_argument(
        '--input',
        type=Path,
        default=Path('logs/gold_errors.log'),
        help='Input error log file'
    )
    parser.add_argument(
        '--punctuation-output',
        type=Path,
        default=Path('logs/gold_errors_punctuation.log'),
        help='Output file for punctuation errors'
    )
    parser.add_argument(
        '--other-output',
        type=Path,
        default=Path('logs/gold_errors_non_punctuation.log'),
        help='Output file for non-punctuation errors'
    )
    
    args = parser.parse_args()
    
    punct_count, other_count = split_error_log(
        args.input,
        args.punctuation_output,
        args.other_output
    )
    
    print(f"Punctuation errors: {punct_count}")
    print(f"Other errors: {other_count}")
    print(f"Total errors: {punct_count + other_count}")
    print(f"\nOutput files:")
    print(f"  Punctuation: {args.punctuation_output}")
    print(f"  Other: {args.other_output}")
