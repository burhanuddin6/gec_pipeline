import json
from constants import *
from urduhack.normalization import normalize_characters
from annotation import *

def extract_clean_sentence_features(correct, storage):
    for i in range(len(correct.words)):
        value = extract_window_features(correct, i)
        storage[str(value)] = value

    
if __name__ == '__main__':

    # Global pipeline instance
    stanza_pipeline = StanzaPipeline()
    nlp = stanza_pipeline.get_pipeline()

    clean_text = open('data/wikiedits/train_correct.txt', 'r', encoding='utf-8').read()
    print("clean text: ", clean_text)
    clean_text = normalize_characters(clean_text)
    
    try:
        num_processed_lines = open('logs/num_processed_lines.txt', 'r').read()
        num_processed_lines = int(num_processed_lines)
    except:
        num_processed_lines = 0
    clean_text = clean_text.split('\n')[num_processed_lines:]
    
    if num_processed_lines == 0:
        annotations = {}
    else:
        annotations = json.load(open('data/valid_grammar_features.json', 'r', encoding='utf-8'), object_hook=custom_decoder)

    print(f"Starting from line number: {num_processed_lines}")
    print(f"annotations: {annotations}")
    
    for sentence in clean_text:
        doc = nlp(sentence)
        # inefficient but had to do this way cuz there is no exception handling in the urduhack library
        for orig in doc.sentences:
            extract_clean_sentence_features(orig, annotations)
        num_processed_lines += 1
        if num_processed_lines % 1000 == 0:
            with open('logs/num_processed_lines.txt', 'w') as f:
                f.write(str(num_processed_lines))
            # write in a json file
            with open('data/valid_grammar_features.json', 'w', encoding='utf-8') as f:
                json.dump(annotations, f, ensure_ascii=False, indent=4, cls=UPOSFeatsEncoder)
