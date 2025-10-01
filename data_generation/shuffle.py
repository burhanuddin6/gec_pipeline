import random

def shuffle_paired_files(correct_file, incorrect_file, ids_file=None, 
                        output_correct=None, output_incorrect=None, output_ids=None, 
                        seed=42):
    """
    Shuffles corresponding lines in correct, incorrect, and optionally ids files using a seed.
    
    :param correct_file: Path to the correct sentences file
    :param incorrect_file: Path to the incorrect sentences file
    :param ids_file: Path to the ids file (optional)
    :param output_correct: Output path for shuffled correct file (if None, overwrites input)
    :param output_incorrect: Output path for shuffled incorrect file (if None, overwrites input)
    :param output_ids: Output path for shuffled ids file (if None, overwrites input)
    :param seed: Random seed for reproducible shuffling
    """
    # Set the random seed
    random.seed(seed)
    
    # Read all files
    with open(correct_file, 'r', encoding='utf-8') as f:
        correct_lines = f.readlines()
    
    with open(incorrect_file, 'r', encoding='utf-8') as f:
        incorrect_lines = f.readlines()
    
    ids_lines = None
    if ids_file:
        with open(ids_file, 'r', encoding='utf-8') as f:
            ids_lines = f.readlines()
    
    # Validate that all files have the same number of lines
    if len(correct_lines) != len(incorrect_lines):
        raise ValueError("Correct and incorrect files must have the same number of lines")
    
    if ids_lines and len(ids_lines) != len(correct_lines):
        raise ValueError("IDs file must have the same number of lines as correct/incorrect files")
    
    # Create indices and shuffle them
    indices = list(range(len(correct_lines)))
    random.shuffle(indices)
    
    # Reorder lines according to shuffled indices
    shuffled_correct = [correct_lines[i] for i in indices]
    shuffled_incorrect = [incorrect_lines[i] for i in indices]
    shuffled_ids = [ids_lines[i] for i in indices] if ids_lines else None
    
    # Set output paths (default to input paths if not specified)
    output_correct = output_correct or correct_file
    output_incorrect = output_incorrect or incorrect_file
    output_ids = output_ids or ids_file
    
    # Write shuffled files
    with open(output_correct, 'w', encoding='utf-8') as f:
        f.writelines(shuffled_correct)
    
    with open(output_incorrect, 'w', encoding='utf-8') as f:
        f.writelines(shuffled_incorrect)
    
    if shuffled_ids and output_ids:
        with open(output_ids, 'w', encoding='utf-8') as f:
            f.writelines(shuffled_ids)
    
    print(f"Files shuffled with seed {seed}")
    print(f"Shuffled {len(correct_lines)} line pairs")
    print(f"Output files:")
    print(f"  Correct: {output_correct}")
    print(f"  Incorrect: {output_incorrect}")
    if output_ids:
        print(f"  IDs: {output_ids}")

# Example usage - shuffle to new files with custom seed
shuffle_paired_files(
    correct_file='data/out/sampled_correct.txt',
    incorrect_file='data/out/sampled_incorrect.txt',
    ids_file='data/out/sampled_error_id.txt',
    output_correct='data/out/shuffled_sampled_correct.txt',
    output_incorrect='data/out/shuffled_sampled_incorrect.txt',
    output_ids='data/out/shuffled_sampled_error_id.txt',
    seed=42
)