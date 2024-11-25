import json
import random

def create_sampled_files(ids_file, correct_file, incorrect_file, json_file, output_ids, output_correct, output_incorrect):
    """
    Select lines from the input files based on error type percentages in the JSON.
    
    :param ids_file: Path to the input `ids` file
    :param correct_file: Path to the input `correct` file
    :param incorrect_file: Path to the input `incorrect` file
    :param json_file: Path to the JSON file with error type percentages
    :param output_ids: Path to the output `ids` file
    :param output_correct: Path to the output `correct` file
    :param output_incorrect: Path to the output `incorrect` file
    """
    # Load the JSON with percentages
    with open(json_file, 'r', encoding='utf-8') as f:
        percentages = json.load(f)
    print(percentages)
    
    # Read input files
    with open(ids_file, 'r', encoding='utf-8') as f_ids, open(correct_file, 'r', encoding='utf-8') as f_correct, open(incorrect_file, 'r', encoding='utf-8') as f_incorrect:
        ids_lines = f_ids.readlines()
        correct_lines = f_correct.readlines()
        incorrect_lines = f_incorrect.readlines()
    
    # Group lines by error type
    grouped_lines = {}
    for i, id_line in enumerate(ids_lines):
        error_type = id_line.strip()
        if error_type not in grouped_lines:
            grouped_lines[error_type] = []
        grouped_lines[error_type].append((id_line, correct_lines[i], incorrect_lines[i]))
    
    # Calculate total lines to sample
    total_lines = len(ids_lines)
    
    # Sample lines based on percentages
    sampled_lines = []
    for error_type, lines in grouped_lines.items():
        # Find the percentage for this error type
        percentage = percentages[error_type]['percentage']
        # Calculate the number of lines to sample
        num_samples = round((percentage / 100) * total_lines)
        sampled_lines.extend(random.sample(lines, min(num_samples, len(lines))))
    
    print(sampled_lines)
    # Write sampled lines to output files
    with open(output_ids, 'w', encoding='utf-8') as f_out_ids, open(output_correct, 'w', encoding='utf-8') as f_out_correct, open(output_incorrect, 'w', encoding='utf-8') as f_out_incorrect:
        for id_line, correct_line, incorrect_line in sampled_lines:
            f_out_ids.write(id_line)
            f_out_correct.write(correct_line)
            f_out_incorrect.write(incorrect_line)
    
    print(f"Sampled files created: {output_ids}, {output_correct}, {output_incorrect}")


# Example usage
create_sampled_files(
    ids_file='data/out copy/error_id.txt',
    correct_file='data/out copy/correct.txt',
    incorrect_file='data/out copy/incorrect.txt',
    json_file='data/annotations_dict.json',
    output_ids='data/out copy/sampled_error_id.txt',
    output_correct='data/out copy/sampled_correct.txt',
    output_incorrect='data/out copy/sampled_incorrect.txt'
)
