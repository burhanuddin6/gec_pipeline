from errant.alignment import Alignment
import numpy as np
import time
import urduhack
# from data_generation.generate_word_dict import generate_word_dict
import json

# Downloading models
# urduhack.download()

# Initializing the pipeline
nlp = urduhack.Pipeline()

orig_text = """
میں یک بلی ہوں۔    
"""

cor_text = """
میں ایک بلی ہوں۔
"""

doc1 = nlp(orig_text)
doc2 = nlp(cor_text)


# algorithm(incorrect, correct):
# get pos tags for both
# get alignment
# compare the Alignment:
#     if substitution, then:
#         store the difference in pos tags, feats of the substitute generate_word_dict
#     if deletion, then:
#         store the pos tags, feats of that words and words on left and right
#     if insertion, then:
#         store the pos tags, feats of that words and words on left and right


word_dict = json.load(open('urdu_word_dict2.json', 'r', encoding='utf-8'))

def lookup_xpos_feats(word, upos):
    if word in word_dict:
        for token in word_dict[word]:
            if token['upos'] == upos:
                return token['xpos'], token['feats']
    else:
        return None, None



def annotate(incorrect, correct):
    '''
    incorrect and correct are both of class sentence
    '''
    alignment = Alignment(incorrect, correct)
    seq = alignment.align_seq
    type_annotation = {}
    for op, i1, i2, j1, j2 in seq:
        if op == 'S':
            type_annotation['type'] = 'S'
            type_annotation['prior_upos'] = incorrect.words[i1].upos
            type_annotation['prior_xpos'], type_annotation['prior_feats'] = lookup_xpos_feats(incorrect.words[i1].text, incorrect.words[i1].upos)
            type_annotation['post_upos'] = incorrect.words[i2].upos
            type_annotation['post_xpos'], type_annotation['post_feats'] = lookup_xpos_feats(incorrect.words[i2].text, incorrect.words[i2].upos)

            print(type_annotation)
        elif op == 'D':
            type_annotation['type'] = 'D'
            print(f"Deletion: {incorrect.words[i1].text}")
        elif op == 'I':
            type_annotation['type'] = 'I'
            print(f"Insertion: {correct.words[j1].text}")
    return type_annotation
    

for sentence1, sentence2 in zip(doc1.sentences, doc2.sentences):
    align = Alignment(sentence1, sentence2)
    print(align.align_seq)
    annotate(sentence1, sentence2)  