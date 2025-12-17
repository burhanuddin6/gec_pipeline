from alignment import Alignment
import numpy as np
import time
# import urduhack
# from data_generation.generate_word_dict import generate_word_dict
import json
from constants import *
from urduhack.normalization import normalize_characters
import config
import contextlib
import stanza
import os

NUM_SPELLING_ISSUES = 0

class UPOSFeats:
    def __init__(self, dictionary: dict):
        self.upos = dictionary['upos']
        self.feats = dictionary['feats']

    def __init__(self, upos, feats):
        self.upos = upos
        self.feats = feats

    def to_dict(self):
        return {'upos': self.upos, 'feats': self.feats}
    
    def __str__(self):
        return f"upos: {self.upos}, feats: {self.feats}"

    def __repr__(self):
        return f"upos: {self.upos}, feats: {self.feats}"
    
    def __eq__(self, value):
        return self.upos == value.upos and self.feats == value.feats
    
class UPOSFeatsEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, UPOSFeats):
            return obj.to_dict()
        return super().default(obj)
    
def custom_decoder(dct: dict):
    """Custom decoder to reconstruct UPOSFeats objects from JSON"""
    for key, value in dct.items():
        if key == 'kernel_feats' and isinstance(value, list):
            # Convert list of dictionaries back to UPOSFeats objects
            dct[key] = []
            for item in value:
                if isinstance(item, dict) and 'upos' in item and 'feats' in item:
                    dct[key].append(UPOSFeats(item['upos'], item['feats']))
                else:
                    dct[key].append(item)  # Keep None values as is
        elif key in ['incorrect_feats', 'correct_feats'] and isinstance(value, dict):
            if 'upos' in value and 'feats' in value:
                dct[key] = UPOSFeats(value['upos'], value['feats'])
    return dct


def log(error):
    with open('logs/errors.log', 'a', encoding='utf-8') as f:
        f.write(f"{error}\n")

def insertion_error_exist(t_annot, type_annotation):
    if t_annot['type'] != INSERTION:
        return False
    if not all([(type_annotation['kernel_feats'][i] == t_annot['kernel_feats'][i]) for i in range(KERNEL_SIZE)]):
        return False
    return True
    
def substitution_error_exist(t_annot, type_annotation):
    
    if t_annot['type'] != SUBSTITUTION:
        return False
    # check if the features and POS are same for the substituted words
    if (not type_annotation['incorrect_feats'] == t_annot['incorrect_feats']) or \
    (not type_annotation['correct_feats'] == t_annot['correct_feats']):
        return False
    return True

def deletion_error_exist(t_annot, type_annotation):
    if t_annot['type'] != DELETION:
        return False
    if t_annot['deleted_words'] != type_annotation['deleted_words']:
        return False
    if not all([
                (type_annotation['kernel_feats'][i] == t_annot['kernel_feats'][i])
                for i in range(KERNEL_SIZE)
                ]):
        return False
    return True
    

def set_kernel(i_minus_one, i, i_plus_one, sequence, type):
    feats = [NONE_LABEL, NONE_LABEL, NONE_LABEL]
    kernel = [NONE_LABEL, NONE_LABEL, NONE_LABEL]
    if i_minus_one >= 0:
        kernel[0] = sequence[i_minus_one].upos
        if type != SUBSTITUTION:
            feats[0] = UPOSFeats(sequence[i_minus_one].upos, sequence[i_minus_one].feats)
    if type != DELETION: # deletion errors will have a smaller kernel
        kernel[1] = sequence[i].upos
        if type != SUBSTITUTION:
            feats[1] = UPOSFeats(sequence[i].upos, sequence[i].feats)
    if i_plus_one < len(sequence):
        kernel[2] = sequence[i_plus_one].upos
        if type != SUBSTITUTION:
            feats[2] = UPOSFeats(sequence[i_plus_one].upos, sequence[i].feats)
    return ((kernel, feats) if type != SUBSTITUTION else kernel)


def annotate(incorrect, correct, kernel_sorted_annotations):
    '''
    incorrect and correct are both of class Sentence from urduhack library
    '''
    alignment = Alignment(incorrect, correct)
    seq = alignment.align_seq
    type_annotation = {}
    
    for op, i1, i2, j1, j2 in seq:
        kernel = [NONE_LABEL, NONE_LABEL, NONE_LABEL]
        if op == SUBSTITUTION:
            try:
                # check if there is not a spelling issue:
                incorrect_word = incorrect.words[i1].text
                correct_word = correct.words[j1].text
                if incorrect_word not in config.word_dict or correct_word not in config.word_dict:
                    print(f"Word not found in the word_dict: {incorrect_word} or {correct_word}")
                    continue

                incorrect_feats = UPOSFeats(incorrect.words[i1].upos, incorrect.words[i1].feats)
                correct_feats = UPOSFeats(correct.words[j1].upos, correct.words[j1].feats)
                kernel = set_kernel(j1-1, j1, j1+1, correct.words, SUBSTITUTION)
                tup_kernel = " ".join(kernel) + '_' + SUBSTITUTION

                # if not a spelling issue then its an easy substitution in which case we would add the information of the substitution
                type_annotation = {
                    'type': SUBSTITUTION,
                    'incorrect_word_upos': incorrect.words[i1].upos,
                    'correct_word_upos': correct.words[j1].upos,
                    'incorrect_feats': incorrect_feats,
                    'correct_feats': correct_feats,
                    'occurence': 1
                }
                
                if tup_kernel in kernel_sorted_annotations:
                    for t_annot in kernel_sorted_annotations[tup_kernel]:
                        if substitution_error_exist(t_annot, type_annotation):
                            t_annot['occurence'] += 1
                            break
                    else:
                        kernel_sorted_annotations[tup_kernel].append(type_annotation.copy())
                else:
                    kernel_sorted_annotations[tup_kernel] = [type_annotation.copy()]
            except ValueError as e:
                log(f"Skipping substitution due to error: {e}")
                continue

        elif op == DELETION:
            deleted_word = incorrect.words[i1].text
            try:
                kernel, kernel_feats = set_kernel(i1-1, None, i2, incorrect.words, DELETION)
                tup_kernel = " ".join(kernel) + '_' + DELETION
            except Exception as e:
                log(f"Error in setting kernel for deletion: {e}")
                continue
            # if not a spelling issue then its an easy substitution in which case we would add the information of the substitution
            type_annotation = {
                'type': DELETION,
                'deleted_words': [deleted_word],
                'kernel_feats': kernel_feats,
                'occurence': 1,
                'incorrect_text': incorrect.text,
                'correct_text': correct.text,
                'alignment': "  ".join([",".join(str(tup_element)) for tup_element in alignment.align_seq])
            }
            if tup_kernel in kernel_sorted_annotations:
                for t_annot in kernel_sorted_annotations[tup_kernel]:
                    if deletion_error_exist(t_annot, type_annotation):
                        t_annot['occurence'] += 1
                        t_annot['deleted_words'].append(deleted_word)
                        break
                else:
                    kernel_sorted_annotations[tup_kernel].append(type_annotation.copy())
            else:
                kernel_sorted_annotations[tup_kernel] = [type_annotation.copy()]

            print(f"Deletion: {incorrect.words[i1].text}")
            print(f"Incorrect text: {incorrect.text}")
            print(f"Correct text: {correct.text}")
            print(f"alignment: {alignment.align_seq}")

        elif op == INSERTION:
            inserted_word = correct.words[j1].text
            try:
                kernel, kernel_feats = set_kernel(j1-1, j1, j1+1, correct.words, INSERTION)
                tup_kernel = " ".join(kernel) + '_' + INSERTION
            except Exception as e:
                log(f"Error in setting kernel for insertion: {e}")
                continue
            # if not a spelling issue then its an easy substitution in which case we would add the information of the substitution
            type_annotation = {
                'type': INSERTION,
                'inserted_word': inserted_word,
                'inserted_word_upos': correct.words[j1].upos,
                'kernel_feats': kernel_feats,
                'occurence': 1
            }
            if tup_kernel in kernel_sorted_annotations:
                for t_annot in kernel_sorted_annotations[tup_kernel]:
                    if insertion_error_exist(t_annot, type_annotation):
                        t_annot['occurence'] += 1
                        break
                else:
                    kernel_sorted_annotations[tup_kernel].append(type_annotation.copy())
            else:
                kernel_sorted_annotations[tup_kernel] = [type_annotation.copy()]

            print(f"Insertion: {correct.words[j1].text}")
            print(f"Incorrect text: {incorrect.text}")
            print(f"Correct text: {correct.text}")
            print(f"alignment: {alignment.align_seq}")


    return kernel_sorted_annotations
    
class StanzaPipeline:
    """Singleton class to manage Stanza pipeline"""
    _instance = None
    _pipeline = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(StanzaPipeline, cls).__new__(cls)
        return cls._instance
    
    def get_pipeline(self):
        if self._pipeline is None:
            print("Initializing Stanza pipeline...")
            # Suppress Stanza's verbose output during initialization
            with contextlib.redirect_stdout(open(os.devnull, 'w')):
                self._pipeline = stanza.Pipeline(
                    lang="ur", 
                    verbose=False, 
                    processors='tokenize,pos,lemma'
                )
            print("Stanza pipeline initialized successfully.")

        return self._pipeline
    
if __name__ == '__main__':

    # Global pipeline instance
    stanza_pipeline = StanzaPipeline()
    nlp = stanza_pipeline.get_pipeline()

    config.word_dict = json.load(open('data/urdu_word_dict.json', 'r', encoding='utf-8'))

    orig_text = open('data/wikiedits/train_incorrect.txt', 'r', encoding='utf-8').read()
    cor_text = open('data/wikiedits/train_correct.txt', 'r', encoding='utf-8').read()

    orig_text = normalize_characters(orig_text)
    cor_text = normalize_characters(cor_text)
    
    try:
        num_processed_lines = open('logs/num_processed_lines.txt', 'r').read()
        num_processed_lines = int(num_processed_lines)
    except:
        num_processed_lines = 0
    orig_text = orig_text.split('\n')[num_processed_lines:]
    cor_text = cor_text.split('\n')[num_processed_lines:]
    
    if num_processed_lines == 0:
        annotations = {}
    else:
        annotations = json.load(open('data/sample.json', 'r', encoding='utf-8'), object_hook=custom_decoder)

    print(f"Starting from line number: {num_processed_lines}")
    print(f"annotations: {annotations}")
    for sentence1, sentence2 in zip(orig_text, cor_text):
        doc1 = nlp(sentence1)
        doc2 = nlp(sentence2)
        # inefficient but had to do this way cuz there is no exception handling in the urduhack library
        for orig, cor in zip(doc1.sentences, doc2.sentences):
            align = Alignment(orig, cor)
            print(align.align_seq)
            annotate(orig, cor, annotations)
        num_processed_lines += 1
        if num_processed_lines % 1000 == 0:
            with open('logs/num_processed_lines.txt', 'w') as f:
                f.write(str(num_processed_lines))
            # write in a json file
            with open('data/sample.json', 'w', encoding='utf-8') as f:
                json.dump(annotations, f, ensure_ascii=False, indent=4, cls=UPOSFeatsEncoder)
            exit()
