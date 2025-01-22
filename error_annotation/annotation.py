from alignment import Alignment
import numpy as np
import time
import urduhack
# from data_generation.generate_word_dict import generate_word_dict
import json
from constants import *
from urduhack.normalization import normalize_characters

word_dict = json.load(open('data/urdu_word_dict2.json', 'r', encoding='utf-8'))
words = open('data/urdu_words.txt', 'r', encoding='utf-8').read().split('\n')
dictionary = {word.strip(): None for word in words}

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
    
from collections.abc import MutableSequence

class WordUPOSFeats(MutableSequence):
    def __init__(self, word):
        if isinstance(word, str):
            if word not in word_dict:
                raise ValueError(f"Word {word} not found in the word_dict")
            self.usage_list = WordUPOSFeats([UPOSFeats(temp['upos'], temp['feats']) for temp in word_dict[word]])
            if self.is_empty():
                raise ValueError(f"Features not found for word {word}")
        elif isinstance(word, list):
            assert all([isinstance(temp, UPOSFeats) for temp in word]) and len(word) > 0
            self.usage_list = word
        else:
            raise ValueError(f"Invalid type for word: {type(word)}")
    def __getitem__(self, index):
        return self.usage_list[index]

    def __setitem__(self, index, value):
        self.usage_list[index] = value

    def __delitem__(self, index):
        del self.usage_list[index]

    def __len__(self):
        return len(self.usage_list)

    def insert(self, index, value):
        self.usage_list.insert(index, value)

    def to_dict(self):
        return [value.to_dict() for value in self.usage_list]
    
    def __eq__(self, value):
        return WordUPOSFeats.features_are_similar(self, value)
    
    def is_empty(self):
        return len(self.usage_list) == 0

    @staticmethod
    def features_are_similar(check_for_word, check_in_word):
        '''
        check_for_word: (str) features to check for
        check_in_word: str features to check check_for_word in/from
        '''
        assert isinstance(check_for_word, WordUPOSFeats) and isinstance(check_in_word, WordUPOSFeats)
        return all(word_characterstics in check_in_word for word_characterstics in check_for_word) or all(word_characterstics in check_for_word for word_characterstics in check_in_word)

import json
from typing import Any

class WordUPOSFeatsEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, WordUPOSFeats):
            # Serialize WordUPOSFeats into a dictionary of UPOSFeats
            return obj.to_dict()  # Uses the to_dict() method from WordUPOSFeats class
        elif isinstance(obj, UPOSFeats):
            # Serialize UPOSFeats directly as a dictionary
            return obj.to_dict()
        return super().default(obj)  # Delegate to the default encoder if it's not one of our classes
    
def custom_decoder(dct: dict):
    # Check if the dictionary contains any keys related to "*feats"
    for key, value in dct.items():
        if key == 'kernel_feats':
            # If the key ends with "*feats", we need to create a WordUPOSFeats instance
            dct[key] = []
            for value2 in value:
                if isinstance(value2, str):
                    # Convert the value into a WordUPOSFeats instance (assumes the value is a list of dictionaries for UPOSFeats)
                    dct[key].append(NONE_LABEL)
                else:
                    # Convert each element into WordUPOSFeats (assumes the value is a list of dictionaries for UPOSFeats)
                    dct[key].append(WordUPOSFeats([UPOSFeats(**item) for item in value2]))
        elif key == 'incorrect_feats' or key == 'correct_feats':
            # Convert the value into a WordUPOSFeats instance (assumes the value is a list of dictionaries for UPOSFeats)
            dct[key] = WordUPOSFeats([UPOSFeats(**item) for item in value])
    return dct


def check_spelling_issues(word):
    global NUM_SPELLING_ISSUES
    if word not in dictionary:
        NUM_SPELLING_ISSUES += 1
        return True
    return False

def log(error):
    with open('logs/errors.log', 'a') as f:
        f.write(f"{error}\n")

def insertion_error_exist(t_annot, type_annotation):
    if t_annot['type'] != INSERTION:
        return False
    if not all([(type_annotation['kernel_feats'][i] == t_annot['kernel_feats'][i]) for i in range(KERNEL_SIZE)]):
        return False
    return True
    
def find_substitute_potentials(deleted_word, feats):
    # find the potential substitutes for the deleted word
    potential_substitutes = []
    for word in word_dict:
        if word == deleted_word:
            continue
        word_characterstics = WordUPOSFeats(word)
        if WordUPOSFeats.features_are_similar(feats, word_characterstics):
            potential_substitutes.append(word)
    return potential_substitutes

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
                (WordUPOSFeats(type_annotation['kernel_feats'][i]) == WordUPOSFeats(t_annot['kernel_feats'][i]))
                for i in range(KERNEL_SIZE)
                ]):
        return False
    return True
    

def set_kernel(i_minus_one, i, i_plus_one, sequence, type):
    feats = [NONE_LABEL, NONE_LABEL, NONE_LABEL]
    kernel = [NONE_LABEL, NONE_LABEL, NONE_LABEL]
    try:
        if i_minus_one >= 0:
            kernel[0] = sequence[i_minus_one].upos
            if type != SUBSTITUTION:
                feats[0] = WordUPOSFeats(sequence[i_minus_one].text)
        if type != DELETION: # deletion errors will have a smaller kernel
            kernel[1] = sequence[i].upos
            if type != SUBSTITUTION:
                feats[1] = WordUPOSFeats(sequence[i].text)
        if i_plus_one < len(sequence):
            kernel[2] = sequence[i_plus_one].upos
            if type != SUBSTITUTION:
                feats[2] = WordUPOSFeats(sequence[i_plus_one].text)
    except:
        pass
        # raise Exception(f"Error in setting kernel, feats with args: {i_minus_one}, {i}, {i_plus_one}, {sequence}, {type}")
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
            
            if not check_spelling_issues(incorrect.words[i1].text):
                # check if there is not a spelling issue:
                incorrect_word = incorrect.words[i1].text
                correct_word = correct.words[j1].text
                if incorrect_word not in word_dict or correct_word not in word_dict:
                    print(f"Word not found in the word_dict: {incorrect_word} or {correct_word}")
                    continue

                incorrect_feats = WordUPOSFeats(incorrect_word)
                correct_feats = WordUPOSFeats(correct_word)
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

        elif op == DELETION:
            deleted_word = incorrect.words[i1].text
            if check_spelling_issues(deleted_word):
                continue
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
            if check_spelling_issues(inserted_word): # update this to check all words in kernel. It is not necessary that the inserted word is the only spelling issue
                continue
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
    

if __name__ == '__main__':
    # Initializing the pipeline
    nlp = urduhack.Pipeline()

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
        annotations = json.load(open('data/annotations.json', 'r', encoding='utf-8'), object_hook=custom_decoder)

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
        if num_processed_lines % 100 == 0:
            with open('logs/num_processed_lines.txt', 'w') as f:
                f.write(str(num_processed_lines))
            # write in a json file
            with open('data/annotations.json', 'w', encoding='utf-8') as f:
                json.dump(annotations, f, ensure_ascii=False, indent=4, cls=WordUPOSFeatsEncoder)
            exit()
            
