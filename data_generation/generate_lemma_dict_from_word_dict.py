import json

def generate_lemma_word_dict(input_path="data/urdu_word_dict.json", output_path="data/lemma_word_dict.json"):
    """
    Creates a dictionary mapping each lemma to a list of its word forms.

    This function reads a word dictionary (JSON file) where each word is mapped
    to a list of its grammatical forms (as tokens). It then creates a reverse
    mapping from each lemma to all unique word forms associated with it.

    Args:
        input_path (str): The path to the input JSON file (urdu_word_dict.json).
        output_path (str): The path to save the output JSON file (lemma_word_dict.json).
    """
    print(f"Loading word dictionary from {input_path}...")
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            word_dict = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_path}")
        print("Please ensure 'urdu_word_dict.json' exists.")
        return

    print("Generating lemma-to-word dictionary...")
    lemma_dict = {}
    for word, tokens in word_dict.items():
        for token in tokens:
            lemma = token.get('lemma')
            if lemma:
                if lemma not in lemma_dict:
                    lemma_dict[lemma] = set()
                lemma_dict[lemma].add(word)

    # Convert sets to lists for JSON serialization
    for lemma in lemma_dict:
        lemma_dict[lemma] = list(lemma_dict[lemma])

    print(f"Saving lemma dictionary to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(lemma_dict, f, ensure_ascii=False, indent=4)

    print(f"Process completed. Lemma dictionary saved with {len(lemma_dict)} entries.")

if __name__ == "__main__":
    generate_lemma_word_dict()
