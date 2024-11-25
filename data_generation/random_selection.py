import random

def shrink_training_data_random(ids_file, correct_file, incorrect_file, output_ids, output_correct, output_incorrect):
    """
    Shrinks training data by randomly selecting one instance from consecutive duplicate IDs.

    :param ids_file: Path to the file containing IDs
    :param correct_file: Path to the file containing correct sentences
    :param incorrect_file: Path to the file containing incorrect sentences
    :param output_ids: Path to the output file for filtered IDs
    :param output_correct: Path to the output file for filtered correct sentences
    :param output_incorrect: Path to the output file for filtered incorrect sentences
    """
    try:
        # Read all three input files
        with open(ids_file, 'r', encoding='utf-8') as f_ids, open(correct_file, 'r', encoding='utf-8') as f_correct, open(incorrect_file, 'r', encoding='utf-8') as f_incorrect:
            ids = [line.strip() for line in f_ids.readlines()]
            correct_sentences = [line.strip() for line in f_correct.readlines()]
            incorrect_sentences = [line.strip() for line in f_incorrect.readlines()]
        
        # Validate lengths of the files
        if not (len(ids) == len(correct_sentences) == len(incorrect_sentences)):
            raise ValueError("All input files must have the same number of lines!")
        
        # Initialize output data structures
        unique_ids = []
        unique_correct = []
        unique_incorrect = []
        
        # Process IDs and handle consecutive duplicates
        buffer = []  # Temporary storage for consecutive duplicate indices
        for i, current_id in enumerate(ids):
            if buffer and ids[buffer[-1]] != current_id:  # New ID encountered
                # Randomly select one index from the buffer
                selected_idx = random.choice(buffer)
                unique_ids.append(ids[selected_idx])
                unique_correct.append(correct_sentences[selected_idx])
                unique_incorrect.append(incorrect_sentences[selected_idx])
                buffer = []  # Reset the buffer
            
            buffer.append(i)  # Add the current index to the buffer
        
        # Handle the last buffer if any
        if buffer:
            selected_idx = random.choice(buffer)
            unique_ids.append(ids[selected_idx])
            unique_correct.append(correct_sentences[selected_idx])
            unique_incorrect.append(incorrect_sentences[selected_idx])
        
        # Write filtered data to output files
        with open(output_ids, 'w', encoding='utf-8') as f_out_ids, \
             open(output_correct, 'w', encoding='utf-8') as f_out_correct, \
             open(output_incorrect, 'w', encoding='utf-8') as f_out_incorrect:
            f_out_ids.write('\n'.join(unique_ids) + '\n')
            f_out_correct.write('\n'.join(unique_correct) + '\n')
            f_out_incorrect.write('\n'.join(unique_incorrect) + '\n')

        print(f"Processed files written to {output_ids}, {output_correct}, and {output_incorrect}")

    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
shrink_training_data_random(
    ids_file="data/out/error_id.txt",                # Input IDs file
    correct_file="data/out/correct.txt",        # Input correct sentences file
    incorrect_file="data/out/incorrect.txt",    # Input incorrect sentences file
    output_ids="data/out/error_id.txt",     # Output filtered IDs file
    output_correct="data/out/correct.txt",  # Output filtered correct sentences file
    output_incorrect="data/out/incorrect.txt"  # Output filtered incorrect sentences file
)
