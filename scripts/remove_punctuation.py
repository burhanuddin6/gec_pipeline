#!/usr/bin/env python3
"""
Remove all punctuation from text files.

Usage:
    python scripts/remove_punctuation.py --input shuffled_sampled_correct.txt --output shuffled_sampled_correct_no_punct.txt
    python scripts/remove_punctuation.py --correct shuffled_sampled_correct.txt --incorrect shuffled_sampled_incorrect.txt
"""
import argparse
import re
from pathlib import Path
import unicodedata
from typing import Set

# Define comprehensive punctuation marks (Urdu and ASCII)
# Note: Excluded ZWNJ (‌) and ZWJ (‍) as they are text formatting, not punctuation
PUNCTUATION_MARKS: Set[str] = {
    '،', '؟', '۔', '!', '.', ',', ':', ';', '?', '!', 
    '-', '—', '–', '"', '"', "'", '(', ')', 
    '[', ']', '{', '}', '/', '\\', '|', '*', '&', '@',
    '#', '$', '%', '^', '`', '~', '<', '>', '«', '»',
    '…', '·', '•', '؛', '٪', '٫', '٬'
}

def normalize_urdu(text: str) -> str:
    # NFC will combine characters and diacritics consistently
    text = unicodedata.normalize('NFC', text)
    return text

def remove_punctuation(text: str) -> str:
    """
    Remove all punctuation from text, normalizing whitespace.
    Removes leading/trailing spaces and collapses multiple spaces to one.
    """
    # Create regex pattern from punctuation set
    pattern = '[' + re.escape(''.join(PUNCTUATION_MARKS)) + ']'
    
    # Remove punctuation
    cleaned = re.sub(pattern, '', text)
    
    # Normalize spaces: multiple spaces to single space
    # cleaned = re.sub(r' +', ' ', cleaned)
    
    # Remove leading and trailing spaces from each line
    cleaned = cleaned.strip()    
    
    return normalize_urdu(cleaned)

def process_file(input_path: Path, output_path: Path) -> tuple:
    """
    Process a file to remove punctuation.
    
    Returns:
        (lines_processed, chars_removed)
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    original_chars = sum(len(line) for line in lines)
    cleaned_lines = []
    
    for line in lines:
        cleaned = remove_punctuation(line)
        if cleaned:  # Only add non-empty lines
            cleaned_lines.append(cleaned)
    
    # Write output
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(cleaned_lines))
        if cleaned_lines:
            f.write('\n')
    
    new_chars = sum(len(line) for line in cleaned_lines)
    chars_removed = original_chars - new_chars
    
    return len(cleaned_lines), chars_removed

def main():
    parser = argparse.ArgumentParser(
        description='Remove punctuation from text files'
    )
    
    # Option 1: Single file mode
    parser.add_argument(
        '--input',
        type=Path,
        help='Input file path'
    )
    parser.add_argument(
        '--output',
        type=Path,
        help='Output file path'
    )
    
    # Option 2: Batch mode for correct/incorrect pairs
    parser.add_argument(
        '--correct',
        type=Path,
        default=Path('data/consolidated_gold_correct.txt'),
        help='Correct sentences file'
    )
    parser.add_argument(
        '--incorrect',
        type=Path,
        default=Path('data/consolidated_gold_incorrect.txt'),
        help='Incorrect sentences file'
    )
    parser.add_argument(
        '--suffix',
        type=str,
        default='_no_punct',
        help='Suffix to add to output filenames (default: _no_punct)'
    )
    
    args = parser.parse_args()
    
    # Single file mode
    if args.input and args.output:
        lines, chars = process_file(args.input, args.output)
        print(f"Processed: {args.input}")
        print(f"  Lines: {lines}")
        print(f"  Characters removed: {chars}")
        print(f"  Output: {args.output}")
        return
    
    # Batch mode (default)
    files_to_process = []
    
    if args.correct.exists():
        output_correct = args.correct.parent / f"{args.correct.stem}{args.suffix}{args.correct.suffix}"
        files_to_process.append((args.correct, output_correct, "Correct"))
    
    if args.incorrect.exists():
        output_incorrect = args.incorrect.parent / f"{args.incorrect.stem}{args.suffix}{args.incorrect.suffix}"
        files_to_process.append((args.incorrect, output_incorrect, "Incorrect"))
    
    if not files_to_process:
        print("No files found to process!")
        print(f"Looking for:")
        print(f"  - {args.correct}")
        print(f"  - {args.incorrect}")
        return
    
    print("Processing files...")
    print()
    
    total_lines = 0
    total_chars = 0
    
    for input_path, output_path, label in files_to_process:
        lines, chars = process_file(input_path, output_path)
        total_lines += lines
        total_chars += chars
        
        print(f"{label} file: {input_path}")
        print(f"  Lines processed: {lines}")
        print(f"  Characters removed: {chars}")
        print(f"  Output: {output_path}")
        print()
    
    print(f"Total lines processed: {total_lines}")
    print(f"Total characters removed: {total_chars}")

if __name__ == '__main__':
    main()