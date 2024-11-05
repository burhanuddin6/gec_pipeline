import urduhack
# from data_generation.generate_word_dict import generate_word_dict
import json
from constants import *


word_dict = json.load(open('urdu_word_dict2.json', 'r', encoding='utf-8'))
annotations = json.load(open('annotations.json', 'r', encoding='utf-8'))
words = open('urdu_words.txt', 'r', encoding='utf-8').read().split('\n')
dictionary = {word.strip(): None for word in words}
lemma_dict = json.load(open('lemma_word_dict.json', 'r', encoding='utf-8'))



def perform_infliction(sentence, word_ind, kernel, error_annotations):
    """
    sentence is of type Sentence from urduhack library
    word_ind is the index of the word in the sentence to be changed/deleted
    kernel is of type list of size 3 
    error_annotation is of type dict that stores the annotation information and index

    Performs the error infliction on the sentence based on the kernel and error_annotation

    returns the incorrect sentence
    """
    replacement = None
    #
    original_word = (sentence.words[word_ind].text).strip('۔').strip('،').strip('؟')

    # extract lemma of the original word
    lemma = sentence.words[word_ind].lemma.strip('۔').strip('،').strip('؟')

    # get different forms of the lemma of the original word
    word_forms_words = lemma_dict[lemma] if lemma in lemma_dict else []

    # populate word_forms with the different forms of the original word (with relevant info from word_dict)
    word_forms = []
    for word_form_word in word_forms_words: 
        if (word_form_word != original_word) and (word_form_word in word_dict):
            word_forms.extend(word_dict[word_form_word])
    if not word_forms:
        raise Exception(f"Word {original_word}'s other forms not found in the word_dict")
    
    replacements = []
    for error_annotation in error_annotations:
        if error_annotation['type'] == SUBSTITUTION:
            change_upos = error_annotation['correct_sentence'][error_annotation['index']]['upos']
            change_word = error_annotation['correct_sentence'][error_annotation['index']]['text']
            change_feats = [temp["feats"] for temp in word_dict[change_word]]
            change_feats = "|".join(change_feats)
            print(change_feats)
            for form in word_forms:
                form_features = form['feats'].split('|')
                if form['text'] != original_word and form['upos'] == change_upos and all([feature in change_feats for feature in form_features]):
                    replacement = form['text']
                    if replacement not in replacements:
                        replacements.append(replacement)
                

        elif error_annotation['type'] == INSERTION:
            pass
    
    return replacements


def inflict(correct_doc):
    '''
    correct_doc is of class document (sentence list) from urduhack library

    Performs error infliction based on the annotations.json file provided

    returns None
    '''
    for sentence in correct_doc.sentences: # this loop is slower (guess) therefore above
        for kernel in annotations:
            for i, token in enumerate(sentence.words):
                token_kernel = " ".join([sentence.words[i-1].upos if i > 0 else NONE_LABEL, token.upos, sentence.words[i+1].upos if i < len(sentence.words)-1 else NONE_LABEL])
                if token_kernel == kernel:
                    try:
                        replace = perform_infliction(sentence, i, kernel, annotations[kernel])
                        # print(f"Replacing {sentence.words[i].text} with {replace}")
                        # print(f"Replacing {sentence.words[i].text}")
                        print("pure replacemnet ", replace)
                    
                    except Exception as e:
                        print(e)
                        continue                    

if __name__ == '__main__':
    # Downloading models
    # urduhack.download()

    # Initializing the pipeline
    nlp = urduhack.Pipeline()

    correct_text = open('to_inflict.txt', 'r', encoding='utf-8').read()

    doc1 = nlp(correct_text)

    inflict(doc1)