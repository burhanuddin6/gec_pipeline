# Sample text with lines
textfiles = ['correct.txt', 'incorrect.txt']
for file in textfiles:
    text = open(file, 'r', encoding='utf-8').read()
    # Remove periods at the end of each line
    lines = text.strip().split('\n')  # Split text into lines
    cleaned_lines = [line.rstrip('.') for line in lines]  # Remove trailing periods

    # Join lines back into a single text
    cleaned_text = '\n'.join(cleaned_lines)

    # Write cleaned text back to file
    with open(file, 'w', encoding='utf-8') as f:
        f.write(cleaned_text)
