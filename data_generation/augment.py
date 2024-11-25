import random
def augment_data(correct_file, incorrect_file, clean_file, output_correct, output_incorrect):
    """
    Augment correct and incorrect data files with clean data where source and target are the same.
    
    :param correct_file: Path to the input `correct` file
    :param incorrect_file: Path to the input `incorrect` file
    :param clean_file: Path to the input `clean` file
    :param output_correct: Path to the output `correct` file
    :param output_incorrect: Path to the output `incorrect` file
    """
    try:
        # Read existing data
        with open(correct_file, 'r', encoding='utf-8') as f_correct, open(incorrect_file, 'r', encoding='utf-8') as f_incorrect:
            correct_lines = f_correct.readlines()
            incorrect_lines = f_incorrect.readlines()
        
        # Read clean data
        with open(clean_file, 'r', encoding='utf-8') as f_clean:
            clean_lines = f_clean.readlines()
        
        # Augment with clean data
        augmented_correct = correct_lines + clean_lines
        augmented_incorrect = incorrect_lines + clean_lines

        #zip and shuffle
        zipped = list(zip(augmented_correct, augmented_incorrect))
        random.shuffle(zipped)
        augmented_correct, augmented_incorrect = zip(*zipped)
        
        # Write augmented data
        with open(output_correct, 'w', encoding='utf-8') as f_out_correct, open(output_incorrect, 'w', encoding='utf-8') as f_out_incorrect:
            f_out_correct.writelines(augmented_correct)
            f_out_incorrect.writelines(augmented_incorrect)
        
        print(f"Augmented files created: {output_correct}, {output_incorrect}")
    
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
augment_data(
    correct_file='data/out copy/sampled_correct.txt',
    incorrect_file='data/out copy/sampled_incorrect.txt',
    clean_file='data/data_01.txt',
    output_correct='data/out copy/sampled_augmented_correct.txt',
    output_incorrect='data/out copy/sampled_augmented_incorrect.txt'
)
