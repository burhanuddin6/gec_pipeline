import urduhack
# from data_generation.generate_word_dict import generate_word_dict
import json
from constants import *
from urduhack.normalization import normalize_characters
from annotation import WordUPOSFeats, custom_decoder
import random


word_dict = json.load(open('data/urdu_word_dict2.json', 'r', encoding='utf-8'))
annotations = json.load(open('data/annotations_with_ids.json', 'r', encoding='utf-8'), object_hook=custom_decoder)
words = open('data/urdu_words.txt', 'r', encoding='utf-8').read().split('\n')
dictionary = {word.strip(): None for word in words}
lemma_dict = json.load(open('data/lemma_word_dict.json', 'r', encoding='utf-8'))


def remove_punctuation(word):
    return word.strip('۔').strip('،').strip('؟')

def fits_kernel(sentence, word_ind, kernel, for_type=SUBSTITUTION):
    """
    sentence is of type Sentence from urduhack library
    word_ind is the index of the word in the sentence to be changed/deleted
    kernel is of type list of size 3 

    Checks if the kernel matches the token at word_ind in the sentence

    returns True if the kernel matches, False otherwise
    """
    if for_type == DELETION:
        token_kernel = [sentence.words[word_ind-1].upos if word_ind > 0 else NONE_LABEL, sentence.words[word_ind+1].upos if word_ind < len(sentence.words)-1 else NONE_LABEL]
        del_kernel = kernel.split('_')[0].split(' ')
        return token_kernel[0] == del_kernel[0] and token_kernel[1] == del_kernel[2]
    else:
        token_kernel = [sentence.words[word_ind-1].upos if word_ind > 0 else NONE_LABEL, sentence.words[word_ind].upos, sentence.words[word_ind+1].upos if word_ind < len(sentence.words)-1 else NONE_LABEL]
        token_kernel = ' '.join(token_kernel)
        return token_kernel in kernel

def substitution_infliction(sentence, word_ind, sub_err_annotations):
    """
    sentence is of type Sentence from urduhack library
    word_ind is the index of the word in the sentence to be changed/deleted
    kernel is of type list of size 3 
    error_annotation is of type dict that stores the annotation information and index

    Performs the error infliction on the sentence based on the kernel and error_annotation

    returns the incorrect sentence

    approach:
    1) we first check if the structure of the kernel confirm to the structure of our annotated error data
	2) then we check if the features of the center word match the corrected word (from annotated error data) by comparing this word 	features with the features of the corrected word
	3) if ok then also check if you can find a substitution of this word from its different lemma forms that match the incorrect word 	(from annotated error data) features set.
	4) if so then, replace the word and produce an erroneous pair.
    """
    replacement = None
    
    original_word = remove_punctuation(sentence.words[word_ind].text)

    # extract lemma of the original word
    lemma = remove_punctuation(sentence.words[word_ind].lemma)

    # get different forms of the lemma of the original word
    word_forms_words = lemma_dict[lemma] if lemma in lemma_dict else []

    # populate word_forms with the different forms of the original word (with relevant info from word_dict)
    word_forms = []
    for word_form_word in word_forms_words: 
        if (word_form_word != original_word) and (word_form_word in word_dict):
            word_forms.append(WordUPOSFeats(word_form_word))
    if not word_forms:
        print (f"Word {original_word}'s other forms not found in the word_dict")
        return []
    
    replacements = [] # a list of incorrectly substitutable words respecting the characterstics provided in the error annotation
    for sub_err_annotation in sub_err_annotations:
        change_word_characterstics = sub_err_annotation['correct_feats']
        for form in word_forms:
            assert isinstance(form, WordUPOSFeats) and isinstance(change_word_characterstics, WordUPOSFeats)
            if form.has_word and form.word != original_word and WordUPOSFeats.features_are_similar(change_word_characterstics, form):
                replacement = form.word
                replacements.append((replacement, sub_err_annotation['id']))

    # remove duplicates if any
    replacements = list(set(replacements))    
    return replacements


def insertion_infliction(sentence, word_ind, ins_err_annotations):
    """
    sentence is of type Sentence from urduhack library
    word_ind is the index of the word in the sentence to be changed/deleted
    kernel is of type list of size 3 
    error_annotation is of type dict that stores the annotation information and index

    Performs the error infliction on the sentence based on the kernel and error_annotation

    returns the incorrect sentence

    approach:
    1) same as step 1 of substitution
	2) check if the features of all three words that fit the kernel are similar to the features of all three corrected words (from annotated error data)
	3) if so, then delete the center word and produce an erroneous pair
	4) one observation: usually the insertion and deletion errors are produced on common words like pronouns and articles, one choice here is that we just limit our operation to words that we actually find in wikiedits. but one heuristic is also that ideally, the majority of insertion errors produced with our approach will contain only those words that are seen in the wikiedits errors.
    """
    replacements = []
    
    try:
        original_word = remove_punctuation(sentence.words[word_ind].text)
        left_word = remove_punctuation(sentence.words[word_ind-1].text) if word_ind > 0 else None
        right_word = remove_punctuation(sentence.words[word_ind+1].text) if word_ind < len(sentence.words)-1 else None
        original_word, left_word, right_word = WordUPOSFeats(original_word), WordUPOSFeats(left_word), WordUPOSFeats(right_word)
    except:
        return replacements
    for ins_err_annotation in ins_err_annotations:
        if all([isinstance(ins_err_annotation['kernel_feats'][i], WordUPOSFeats) for i in range(3)]):
            if WordUPOSFeats.features_are_similar(left_word, ins_err_annotation['kernel_feats'][0]) and WordUPOSFeats.features_are_similar(original_word, ins_err_annotation['kernel_feats'][1]) and WordUPOSFeats.features_are_similar(right_word, ins_err_annotation['kernel_feats'][2]):
                erroneous_sentence = sentence.text.replace(sentence.words[word_ind].text, ' ')
                erroneous_sentence = erroneous_sentence.replace('  ', ' ')
                replacements.append((erroneous_sentence, ins_err_annotation['id']))
    
    replacements = list(set(replacements))
    return replacements

def deletion_infliction(sentence, word_ind, del_err_annotations):
    """
    sentence is of type Sentence from urduhack library
    word_ind is the index of the word in the sentence to be changed/deleted
    kernel is of type list of size 3 
    error_annotation is of type dict that stores the annotation information and index

    Performs the error infliction on the sentence based on the kernel and error_annotation

    returns the new sentence and the error id

    approach:
    1) same as step 1 of substitution
	2) check if for two consecutive words match the left and right word of the annotation
    3) insert the deleted word (from annotation) in between the two words and produce an erroneous pair
    """
    replacements = []

    try:
        original_word = WordUPOSFeats(remove_punctuation(sentence.words[word_ind].text))
        next_word = remove_punctuation(sentence.words[word_ind+1].text) if word_ind < len(sentence.words)-1 else None
        next_word = WordUPOSFeats(next_word)
    except:
        return replacements

    for del_err_annotation in del_err_annotations:
        print(del_err_annotation['kernel_feats'][0], del_err_annotation['kernel_feats'][2])
        if all([isinstance(del_err_annotation['kernel_feats'][i], WordUPOSFeats) for i in [0, 2]]):
            if WordUPOSFeats.features_are_similar(original_word, del_err_annotation['kernel_feats'][0]) and WordUPOSFeats.features_are_similar(next_word, del_err_annotation['kernel_feats'][2]):
                insertion_index = word_ind + len(sentence.words[word_ind].text) + 1
                replace_word = del_err_annotation['deleted_words'][random.randint(0, len(del_err_annotation['deleted_words'])-1)]
                erroneous_sentence = sentence.text[:insertion_index] + replace_word + ' ' + sentence.text[insertion_index:]
                erroneous_sentence = erroneous_sentence.replace('  ', ' ')
                replacements.append((erroneous_sentence, del_err_annotation['id']))
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
            for i in range(len(sentence.words)): # check kernel on each index of the sentence
                if fits_kernel(sentence, i, kernel, for_type=DELETION):
                    try:
                        if SUBSTITUTION in kernel.split('_'):
                            replace = substitution_infliction(sentence, i, annotations[kernel])
                            print("pure replacemnet ", replace)
                            for replacement, error_id in replace:
                                print("SENTENCE", sentence.text)
                                print("REPLACEMENT", replacement)
                                err_sentence = sentence.text.replace(sentence.words[i].text, replacement)
                                sentence_pairs.append((sentence.text, err_sentence, error_id))
                        elif INSERTION in kernel.split('_'):
                            replacements = insertion_infliction(sentence, i, annotations[kernel])
                            for erroneous_sentence, error_id in replacements:
                                sentence_pairs.append((sentence.text, erroneous_sentence, error_id))
                        elif DELETION in kernel.split('_'):
                            replacements = deletion_infliction(sentence, i, annotations[kernel])
                            for erroneous_sentence, error_id in replacements:
                                sentence_pairs.append((sentence.text, erroneous_sentence, error_id))
                            else:
                                raise Exception("Invalid kernel type")
                    except Exception as e:
                        print(e)
                        continue
    return sentence_pairs                    

if __name__ == '__main__':
    nlp = urduhack.Pipeline()

    correct_text = open('data/cleaned_correct_corpus/data_00.txt', 'r', encoding='utf-8').read()
    correct_text = normalize_characters(correct_text)
    chunked_text = correct_text.split('\n')


    corr_out_file = open('data/out/correct.txt', 'a', encoding='utf-8')
    incorr_out_file = open('data/out/incorrect.txt', 'a', encoding='utf-8')
    error_id_file = open('data/out/error_id.txt', 'a', encoding='utf-8')
    step = 10
    for i in range(0, len(chunked_text), step):
        chunk = chunked_text[i:i+step]
        to_inflict = "\n".join(chunk)
        doc1 = nlp(to_inflict)
        pairs = inflict(doc1)
        print(pairs)
        exit()