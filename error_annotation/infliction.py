import urduhack
# from data_generation.generate_word_dict import generate_word_dict
import json
from constants import *
from urduhack.normalization import normalize_characters
from annotation import features_are_similar, get_features



word_dict = json.load(open('data/urdu_word_dict2.json', 'r', encoding='utf-8'))
annotations = json.load(open('data/annotations_with_ids.json', 'r', encoding='utf-8'))
words = open('data/urdu_words.txt', 'r', encoding='utf-8').read().split('\n')
dictionary = {word.strip(): None for word in words}
lemma_dict = json.load(open('data/lemma_word_dict.json', 'r', encoding='utf-8'))



def substitution_infliction(sentence, word_ind, kernel, sub_err_annotations):
    """
    sentence is of type Sentence from urduhack library
    word_ind is the index of the word in the sentence to be changed/deleted
    kernel is of type list of size 3 
    error_annotation is of type dict that stores the annotation information and index

    Performs the error infliction on the sentence based on the kernel and error_annotation

    returns the incorrect sentence
    """
    replacement = None
    
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
    for sub_err_annotation in sub_err_annotations:
        change_upos = sub_err_annotation['correct_word_upos']
        change_feats = sub_err_annotation['correct_feats']
        for form in word_forms:
            form_features = form['feats']
            if form['text'] != original_word and form['upos'] == change_upos and features_are_similar(change_feats, [form_features]):
                replacement = form['text']
                if replacement not in replacements:
                    replacements.append((replacement, sub_err_annotation['id']))
                    
    return replacements


def inflict(correct_doc):
    '''
    correct_doc is of class document (sentence list) from urduhack library

    Performs error infliction based on the annotations.json file provided

    returns None
    '''
    sentence_pairs = []
    for sentence in correct_doc.sentences: # this loop is slower (guess) therefore above
        for kernel in annotations:
            for i, token in enumerate(sentence.words):
                token_kernel = [sentence.words[i-1].upos if i > 0 else NONE_LABEL, token.upos, sentence.words[i+1].upos if i < len(sentence.words)-1 else NONE_LABEL]
                if token_kernel == kernel:
                    try:
                        replace = substitution_infliction(sentence, i, kernel, annotations[kernel])
                        print("pure replacemnet ", replace)
                        for replacement, error_id in replace:
                            print("SENTENCE", sentence.text)
                            print("REPLACEMENT", replacement)
                            err_sentence = sentence.text.replace(sentence.words[i].text, replacement)
                            sentence_pairs.append((sentence.text, err_sentence, error_id))

                    
                    except Exception as e:
                        print(e)
                        continue
    return sentence_pairs                    

if __name__ == '__main__':
    # Downloading models
    # urduhack.download()

    # Initializing the pipeline
    nlp = urduhack.Pipeline()

    correct_text = open('data/data_00.txt', 'r', encoding='utf-8').read()
    correct_text = normalize_characters(correct_text)
    chunked_text = correct_text.split('\n')


    corr_out_file = open('data/out/correct.txt', 'a', encoding='utf-8')
    incorr_out_file = open('data/out/incorrect.txt', 'a', encoding='utf-8')
    error_id_file = open('data/out/error_id.txt', 'a', encoding='utf-8')
    for i in range(0, len(chunked_text), 10):
        try:
            chunk = chunked_text[i:i+10]
            to_inflict = "\n".join(chunk)
            doc1 = nlp(to_inflict)

            pairs = inflict(doc1)
            
        except:
            continue