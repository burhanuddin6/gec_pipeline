import subprocess
import numpy as np
import time
import urduhack
from urduhack import CoNLL
from urduhack.normalization import normalize_characters



def downlaod_data():
    subprocess.run(["rm", "ur_udtb-ud-train.conllu"])
    subprocess.run(['rm', 'words.txt'])
    subprocess.run(["wget", "https://raw.githubusercontent.com/UniversalDependencies/UD_Urdu-UDTB/refs/heads/master/ur_udtb-ud-train.conllu"])
    subprocess.run(["wget", "https://github.com/urduhack/urdu-words/raw/refs/heads/master/words.txt"])


def init_urduhack_pipeline():
    # Downloading models
    urduhack.download()

    # Initializing the pipeline
    nlp = urduhack.Pipeline()

def init_urdu_word_dict():
    with open('words.txt', 'r') as file:
        words = file.read().splitlines()

    return {word: None for word in words}

def build_urdu_word_dict(words_dict):
    CONLL_DATA = CoNLL.load_file('ur_udtb-ud-train.conllu')
    for sentence in CONLL_DATA:
        sent_meta, tokens = sentence
        print(f"Sentence ID: {sent_meta['sent_id']}")
        print(f"Sentence Text: {sent_meta['text']}")
        for token in tokens:
            try:
                if words_dict[token['text']] is None:
                    words_dict[token['text']] = []
                # words_dict[token['text']].append(token)
                append=False
                for word_token in words_dict[token['text']]:
                    if word_token['upos'] != token['upos'] or word_token['xpos'] != token['xpos'] or word_token['feats'] != token['feats']:
                        append = True
                        break
                if append:
                    words_dict[token['text']].append(token)
            except:
                continue

def clean_urdu_word_dict(words_dict):
    words_dict2 = {}
    for word in words_dict.keys():
        if words_dict[word] is not None:
            words_dict2[word] = words_dict[word]
    return words_dict2

def generate_word_dicts():
    downlaod_data()
    init_urduhack_pipeline()
    words_dict = init_urdu_word_dict()
    words_dict = build_urdu_word_dict(words_dict)
    words_dict2 = clean_urdu_word_dict(words_dict)
    # save dict to files
    with open('urdu_word_dict.txt', 'w') as file:
        file.write(str(words_dict))
    with open('urdu_word_dict2.txt', 'w') as file:
        file.write(str(words_dict2))