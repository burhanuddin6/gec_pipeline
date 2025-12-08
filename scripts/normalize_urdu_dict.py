#!/usr/bin/env python3
"""
Normalize all Urdu characters in makhzan_wordFrequency.json dictionary.

This script applies Unicode NFC normalization to all keys in the dictionary,
merging entries that become identical after normalization.

Usage:
    python scripts/normalize_urdu_dict.py
    python scripts/normalize_urdu_dict.py --input data/makhzan_wordFrequency.json --output data/makhzan_wordFrequency_normalized.json
"""
import argparse
import json
import unicodedata
from pathlib import Path
from typing import Dict


def normalize_characters(text: str) -> str:
    """
    Normalize text using Unicode NFC normalization.
    """
    return unicodedata.normalize('NFC', text)


def normalize_dictionary(input_dict: Dict[str, int]) -> Dict[str, int]:
    """
    Normalize all keys in the dictionary and merge duplicate entries.
    
    Args:
        input_dict: Dictionary mapping words to frequencies
    
    Returns:
        Normalized dictionary with merged frequencies
    """
    normalized_dict = {}
    merge_count = 0
    
    for word, frequency in input_dict.items():
        normalized_word = normalize_characters(word)
        
        if normalized_word in normalized_dict:
            # Merge frequencies for words that normalize to the same form
            normalized_dict[normalized_word] += frequency
            merge_count += 1
        else:
            normalized_dict[normalized_word] = frequency
    
    return normalized_dict, merge_count


def main():
    parser = argparse.ArgumentParser(
        description='Normalize Urdu characters in word frequency dictionary'
    )
    parser.add_argument(
        '--input',
        type=Path,
        default=Path('data/makhzan_wordFrequency.json'),
        help='Input dictionary file'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('data/makhzan_wordFrequency_normalized.json'),
        help='Output normalized dictionary file'
    )
    parser.add_argument(
        '--in-place',
        action='store_true',
        help='Overwrite input file (use with caution!)'
    )
    
    args = parser.parse_args()
    
    # Load input dictionary
    print(f"Loading dictionary from: {args.input}")
    with open(args.input, 'r', encoding='utf-8') as f:
        input_dict = json.load(f)
    
    print(f"Original entries: {len(input_dict)}")
    
    # Normalize
    print("Normalizing Urdu characters...")
    normalized_dict, merge_count = normalize_dictionary(input_dict)
    
    print(f"Normalized entries: {len(normalized_dict)}")
    print(f"Entries merged: {merge_count}")
    
    # Determine output path
    output_path = args.input if args.in_place else args.output
    
    # Save normalized dictionary
    print(f"Saving normalized dictionary to: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(normalized_dict, f, ensure_ascii=False, indent=2)
    
    print("Done!")
    
    # Show sample of changes
    if merge_count > 0:
        print("\nSample of normalized words (first 10 that changed):")
        count = 0
        for original, freq in list(input_dict.items())[:100]:
            normalized = normalize_characters(original)
            if original != normalized:
                print(f"  {original} -> {normalized} (freq: {freq})")
                count += 1
                if count >= 10:
                    break


if __name__ == '__main__':
    main()
