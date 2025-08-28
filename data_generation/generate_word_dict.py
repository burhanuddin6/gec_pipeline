import subprocess
import json
import urduhack
from urduhack.conll import CoNLL

def download_data():
    """
    Downloads the necessary CoNLL-U files from the Universal Dependencies repository.
    """
    # URLs for the CoNLL-U files
    base_url = "https://raw.githubusercontent.com/UniversalDependencies/UD_Urdu-UDTB/master/"
    files_to_download = [
        "ur_udtb-ud-train.conllu",
        "ur_udtb-ud-test.conllu",
        "ur_udtb-ud-dev.conllu"
    ]

    print("Downloading CoNLL-U data files...")
    for file_name in files_to_download:
        try:
            # Clean up old file if it exists
            subprocess.run(["rm", file_name], check=False, stderr=subprocess.DEVNULL)
        except FileNotFoundError:
            pass
        # Download the file
        subprocess.run(["wget", base_url + file_name], check=True)
    print("Downloads complete.")


def init_urduhack_pipeline():
    """
    Downloads and initializes the UrduHack pipeline models.
    """
    print("Initializing UrduHack pipeline...")
    urduhack.download()
    print("UrduHack initialization complete.")


def build_urdu_word_dict_from_conllu():
    """
    Builds a word dictionary directly from CoNLL-U files.

    This function iterates through the training, testing, and development
    CoNLL-U files. For each word, it stores a list of unique linguistic
    forms by appending the raw `urduhack.conll.Token` object.

    Returns:
        dict: A dictionary containing words and their corresponding token data.
    """
    words_dict = {}
    conllu_files = [
        'ur_udtb-ud-train.conllu',
        'ur_udtb-ud-test.conllu',
        'ur_udtb-ud-dev.conllu'
    ]

    print("Building word dictionary from CoNLL-U files...")
    for file_path in conllu_files:
        print(f"Processing {file_path}...")
        conll_data = CoNLL.load_file(file_path)
        for sentence in conll_data:
            _sent_meta, tokens = sentence
            for token in tokens:
                word = token['text']

                # If the word is not yet in our dictionary, add it with its first token form.
                if word not in words_dict:
                    # **CRITICAL CHANGE**: Appending the raw token object itself
                    words_dict[word] = [token]
                    continue

                # If word exists, check if this specific grammatical form is already recorded.
                is_duplicate = False
                for existing_token in words_dict[word]:
                    if (existing_token['upos'] == token['upos'] and
                        existing_token['xpos'] == token['xpos'] and
                        existing_token['feats'] == token['feats']):
                        is_duplicate = True
                        break
                
                # If this form is not a duplicate, add it.
                if not is_duplicate:
                    # **CRITICAL CHANGE**: Appending the raw token object itself
                    words_dict[word].append(token)
    
    print(f"Dictionary built. Total unique words: {len(words_dict)}")
    return words_dict


def generate_word_dict():
    """
    Main function to orchestrate the dictionary generation process.
    """
    # To run this from scratch, uncomment the next two lines.
    # print("Step 1: Downloading data...")
    # download_data()
    
    print("Step 1: Initializing UrduHack...")
    init_urduhack_pipeline()
    
    print("Step 2: Building the dictionary from CoNLL-U files...")
    final_word_dict = build_urdu_word_dict_from_conllu()
    
    print("Step 3: Saving the final dictionary...")
    output_filename = 'urdu_word_dict.json'
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(final_word_dict, f, ensure_ascii=False, indent=4)
        
    print(f"Process completed successfully. Dictionary saved to {output_filename}")


if __name__ == "__main__":
    generate_word_dict()