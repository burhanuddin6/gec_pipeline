from errant.alignment import Alignment
import numpy as np
import time
import urduhack
# from data_generation.generate_word_dict import generate_word_dict
import json
from constants import *
from urduhack.normalization import normalize_characters

# Downloading models
# urduhack.download()

# Initializing the pipeline
nlp = urduhack.Pipeline()

orig_text = open('incorrect.txt', 'r', encoding='utf-8').read()
cor_text = open('correct.txt', 'r', encoding='utf-8').read()

orig_text = normalize_characters(orig_text)
cor_text = normalize_characters(cor_text)

orig_text = orig_text.split('\n')
cor_text = cor_text.split('\n')

word_dict = json.load(open('urdu_word_dict2.json', 'r', encoding='utf-8'))
words = open('urdu_words.txt', 'r', encoding='utf-8').read().split('\n')
dictionary = {word.strip(): None for word in words}


def annotate(incorrect, correct, kernel_sorted_annotations):
    '''
    incorrect and correct are both of class Sentence from urduhack library
    '''
    alignment = Alignment(incorrect, correct)
    seq = alignment.align_seq
    type_annotation = {}
    spelling_issues_dataset = []
    
    for op, i1, i2, j1, j2 in seq:
        kernel = [NONE_LABEL, NONE_LABEL, NONE_LABEL]
        if op == SUBSTITUTION:
            # check if there is not a spelling issue:
            if incorrect.words[i1].text not in dictionary:
                spelling_issues_dataset.append((incorrect.text, correct.text))
                print(f"Spelling issue: {incorrect.words[i1].text}")
            else:
                # if not a spelling issue then its an easy substitution in which case we would add the information of the substitution
                type_annotation['type'] = SUBSTITUTION
                type_annotation['incorrect_sentence'] = incorrect.to_dict()
                type_annotation['correct_sentence'] = correct.to_dict()
                type_annotation['index'] = i1
                if i1 > 0:
                    kernel[0] = incorrect.words[i1-1].upos 
                kernel[1] = incorrect.words[i1].upos
                if i1 < len(incorrect.words)-1:
                    kernel[2] = incorrect.words[i1+1].upos
                tup_kernel = " ".join(kernel)
                if tup_kernel in kernel_sorted_annotations:
                    kernel_sorted_annotations[tup_kernel].append(type_annotation)
                else:
                    kernel_sorted_annotations[tup_kernel] = [type_annotation]
        elif op == DELETION:
            pass
        elif op == INSERTION:
            # its a bit counter intuitive. So I means that the correct sentence has an extra word.
            # But our task is to generate the incorrect sentence from the correct sentence, so we would delte the extra word from the correct sentence
            # when we get a similar structure in the correct sentence
            type_annotation['type'] = INSERTION
            type_annotation['incorrect_sentence'] = incorrect.to_dict()
            type_annotation['index'] = j1
            if j1 > 0:
                kernel[0] = correct.words[j1-1].upos
            kernel[1] = correct.words[j1].upos
            if j1 < len(correct.words)-1:
                kernel[2] = correct.words[j1+1].upos
            tup_kernel = " ".join(kernel)
            if tup_kernel in kernel_sorted_annotations:
                kernel_sorted_annotations[tup_kernel].append(type_annotation)
            else:
                kernel_sorted_annotations[tup_kernel] = [type_annotation]
    return kernel_sorted_annotations
    

abc = {}
for sentence1, sentence2 in zip(orig_text, cor_text):
    doc1 = nlp(sentence1)
    doc2 = nlp(sentence2)
    # inefficient but had to do this way cuz there is no exception handling in the urduhack library
    for orig, cor in zip(doc1.sentences, doc2.sentences):
        align = Alignment(orig, cor)
        print(align.align_seq)
        annotate(orig, cor, abc)

print(abc)

# write in a json file
with open('annotations.json', 'w', encoding='utf-8') as f:
    json.dump(abc, f, ensure_ascii=False, indent=4)