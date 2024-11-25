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

orig_text = open('data/wikiedits/incorrect.txt', 'r', encoding='utf-8').read()
cor_text = open('data/wikiedits/correct.txt', 'r', encoding='utf-8').read()

orig_text = normalize_characters(orig_text)
cor_text = normalize_characters(cor_text)

orig_text = orig_text.split('\n')
cor_text = cor_text.split('\n')

word_dict = json.load(open('data/urdu_word_dict2.json', 'r', encoding='utf-8'))
words = open('data/urdu_words.txt', 'r', encoding='utf-8').read().split('\n')
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

            if incorrect.words[i1].text not in dictionary:
                spelling_issues_dataset.append((incorrect.text, correct.text))
                print(f"Spelling issue: {incorrect.words[i1].text}")
            else:

                # check if there is not a spelling issue:
                incorrect_word_upos = incorrect.words[i1].upos
                correct_word_upos = correct.words[j1].upos
                incorrect_word = incorrect.words[i1].text
                correct_word = correct.words[j1].text
                if incorrect_word not in word_dict or correct_word not in word_dict:
                    print(f"Word not found in the word_dict: {incorrect_word} or {correct_word}")
                    continue
                incorrect_feats = [i for row in [temp["feats"].split("|") for temp in word_dict[incorrect_word]] for i in row]
                incorrect_feats = set(incorrect_feats); incorrect_feats = list(incorrect_feats)
                incorrect_feats = "|".join(incorrect_feats)
                correct_feats = [i for row in [temp["feats"].split("|") for temp in word_dict[correct_word]] for i in row]
                correct_feats = set(correct_feats); correct_feats = list(correct_feats)
                correct_feats = "|".join(correct_feats)

                # if not a spelling issue then its an easy substitution in which case we would add the information of the substitution
                type_annotation['type'] = SUBSTITUTION
                type_annotation['incorrect_word_upos'] = incorrect_word_upos
                type_annotation['correct_word_upos'] = correct_word_upos
                type_annotation['incorrect_feats'] = incorrect_feats
                type_annotation['correct_feats'] = correct_feats
                
                if i1 > 0:
                    kernel[0] = incorrect.words[i1-1].upos 
                kernel[1] = incorrect.words[i1].upos
                if i1 < len(incorrect.words)-1:
                    kernel[2] = incorrect.words[i1+1].upos
                tup_kernel = " ".join(kernel)
                if tup_kernel in kernel_sorted_annotations:
                    for t_annot in kernel_sorted_annotations[tup_kernel]:
                        if t_annot['type'] == SUBSTITUTION and t_annot['incorrect_word_upos'] == incorrect_word_upos and t_annot['correct_word_upos'] == correct_word_upos and (incorrect_feats in t_annot['incorrect_feats']) and (correct_feats in t_annot['correct_feats']):
                            t_annot['occurence'] += 1
                            break
                    else:
                        kernel_sorted_annotations[tup_kernel].append({
                            'type': SUBSTITUTION,
                            'incorrect_word_upos': type_annotation['incorrect_word_upos'],
                            'correct_word_upos': type_annotation['correct_word_upos'],
                            'incorrect_feats': type_annotation['incorrect_feats'],
                            'correct_feats': type_annotation['correct_feats'],
                            'occurence': 1
                        })
                else:
                    kernel_sorted_annotations[tup_kernel] = [{
                        'type': SUBSTITUTION,
                        'incorrect_word_upos': type_annotation['incorrect_word_upos'],
                        'correct_word_upos': type_annotation['correct_word_upos'],
                        'incorrect_feats': type_annotation['incorrect_feats'],
                        'correct_feats': type_annotation['correct_feats'],
                        'occurence': 1
                    }]
        elif op == DELETION:
            pass
        elif op == INSERTION:
            # its a bit counter intuitive. So I means that the correct sentence has an extra word.
            # But our task is to generate the incorrect sentence from the correct sentence, so we would delte the extra word from the correct sentence
            # when we get a similar structure in the correct sentence
            # check if there is not a spelling issue:
            change_word_upos = correct.words[j1].upos
            change_word = correct.words[j1].text
            if change_word not in word_dict:
                print(f"Word not found in the word_dict: {change_word}")
                continue
            change_feats = [i for row in [temp["feats"].split("|") for temp in word_dict[change_word]] for i in row]
            change_feats = set(change_feats); change_feats = list(change_feats)
            change_feats = "|".join(change_feats)

            # if not a spelling issue then its an easy substitution in which case we would add the information of the substitution
            type_annotation['type'] = INSERTION
            type_annotation['change_word_upos'] = change_word_upos
            type_annotation['change_feats'] = change_feats

            if j1 > 0:
                kernel[0] = correct.words[j1-1].upos
            kernel[1] = correct.words[j1].upos
            if j1 < len(correct.words)-1:
                kernel[2] = correct.words[j1+1].upos
            tup_kernel = " ".join(kernel)
            if tup_kernel in kernel_sorted_annotations:
                for t_annot in kernel_sorted_annotations[tup_kernel]:
                    if t_annot['type'] == INSERTION and t_annot['change_word_upos'] == change_word_upos and (change_feats in t_annot['change_feats']):
                        t_annot['occurence'] += 1
                        break
                else:
                    kernel_sorted_annotations[tup_kernel].append({
                        'type': INSERTION,
                        'change_word_upos': type_annotation['change_word_upos'],
                        'change_feats': type_annotation['change_feats'],
                        'occurence': 1
                    })
            else:
                kernel_sorted_annotations[tup_kernel] = [{
                    'type': INSERTION,
                    'change_word_upos': type_annotation['change_word_upos'],
                    'change_feats': type_annotation['change_feats'],
                    'occurence': 1
                }]
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
with open('data/annotations.json', 'w', encoding='utf-8') as f:
    json.dump(abc, f, ensure_ascii=False, indent=4)