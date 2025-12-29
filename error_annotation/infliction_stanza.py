import json
import os
import contextlib
import stanza
from functools import lru_cache
from typing import Dict, List, Any, Optional, Tuple

from error_annotation.constants import *
from misc.urduhack_normalization import normalize_characters
from error_annotation.annotation_stanza import UPOSFeats, StanzaPipeline, analyze_sentence_with_stanza, custom_decoder
from error_annotation import config

import tqdm

# Load annotations and lemma dictionary
annotations = json.load(open('data/annotations_with_ids.json', 'r', encoding='utf-8'), object_hook=custom_decoder)
lemma_dict = json.load(open('data/lemma_word_dict.json', 'r', encoding='utf-8'))

# Global pipeline instance
stanza_pipeline = StanzaPipeline()

def remove_punctuation(word):
    return word.strip('۔').strip('،').strip('؟')

def is_word_in_dict(word: str) -> bool:
    """Check if word exists in urdu_word_dict.json (OOV check)"""
    return word in config.word_dict

def fits_kernel(sentence_words: List[Dict], word_ind: int, kernel_upos: List[str], 
                kernel_feats: List[UPOSFeats], for_type: str = SUBSTITUTION) -> bool:
    """
    Check if the kernel matches the context at word_ind in the sentence
    
    Args:
        sentence_words: List of word dictionaries from Stanza analysis
        word_ind: Index of the word in the sentence
        kernel_upos: UPOS tags of the kernel
        kernel_feats: Feature objects of the kernel
        for_type: Type of error (SUBSTITUTION, INSERTION, DELETION)
    
    Returns:
        True if the kernel matches, False otherwise
    """
    if for_type == DELETION:
        # For deletion, check if we can fit the kernel around the deletion point
        left_ind = word_ind - 1
        right_ind = word_ind + 1
        
        # Check left context
        if left_ind >= 0:
            if (sentence_words[left_ind]['upos'] != kernel_upos[0] or 
                sentence_words[left_ind]['feats'] != kernel_feats[0].feats):
                return False
        elif kernel_upos[0] != NONE_LABEL:
            return False
            
        # Check right context
        if right_ind < len(sentence_words):
            if (sentence_words[right_ind]['upos'] != kernel_upos[2] or 
                sentence_words[right_ind]['feats'] != kernel_feats[2].feats):
                return False
        elif kernel_upos[2] != NONE_LABEL:
            return False
            
    else:  # SUBSTITUTION or INSERTION
        left_ind = word_ind - 1
        right_ind = word_ind + 1
        
        # Check left context
        if left_ind >= 0:
            if (sentence_words[left_ind]['upos'] != kernel_upos[0] or 
                sentence_words[left_ind]['feats'] != kernel_feats[0].feats):
                return False
        elif kernel_upos[0] != NONE_LABEL:
            return False
            
        # Check middle word
        if (sentence_words[word_ind]['upos'] != kernel_upos[1] or 
            sentence_words[word_ind]['feats'] != kernel_feats[1].feats):
            return False
            
        # Check right context
        if right_ind < len(sentence_words):
            if (sentence_words[right_ind]['upos'] != kernel_upos[2] or 
                sentence_words[right_ind]['feats'] != kernel_feats[2].feats):
                return False
        elif kernel_upos[2] != NONE_LABEL:
            return False
    
    return True

def substitution_infliction(sentence_words: List[Dict], word_ind: int, 
                           sub_err_annotations: List[Dict]) -> List[Tuple[str, str]]:
    """
    Perform substitution infliction using context-aware morphological analysis
    
    Args:
        sentence_words: List of word dictionaries from Stanza analysis
        word_ind: Index of the word to be substituted
        sub_err_annotations: List of substitution error annotations
    
    Returns:
        List of possible replacement words
    """
    replacements = []
    
    original_word = remove_punctuation(sentence_words[word_ind]['text'])
    lemma = remove_punctuation(sentence_words[word_ind]['lemma'])
    
    # Get different forms of the lemma
    word_forms_words = lemma_dict.get(lemma, [])
    # print(f"Word forms for lemma '{lemma}': {word_forms_words}")
    # print(f"Type of word_forms_words: {type(word_forms_words)}")
    # print(f"Type of entries in word_forms_words: {type(word_forms_words[0]) if word_forms_words else 'N/A'}")
    '''
    It was determined from above three lines of code that the type of `word_forms_words` is list of strings, we can perform a character-level comparison with the original word such that the original word and the other derived words of original word's lemma should have atleast one character same.

    The reason this is done here, is because: https://aistudio.google.com/prompts/1IPYH5urj6KAXt0WXe11Hr8NfxiRxsDNz
    THIS WILL BE IMPLEMENTED IF NEEDED LATER.
    '''
    
    # For each substitution annotation
    for sub_err_annotation in sub_err_annotations:
        try:
            # Check if the current word's features match the "correct" features in the annotation
            current_word_feats = UPOSFeats(sentence_words[word_ind]['upos'], 
                                          sentence_words[word_ind]['feats'])
            
            if (current_word_feats.upos == sub_err_annotation['correct_feats'].upos and
                current_word_feats.feats == sub_err_annotation['correct_feats'].feats):
                
                # Look for a form of the lemma that matches the "incorrect" features
                target_upos = sub_err_annotation['incorrect_feats'].upos
                target_feats = sub_err_annotation['incorrect_feats'].feats
                
                for word_form in word_forms_words:
                    if not is_word_in_dict(word_form):
                        continue
                        
                    # Check if this word form has the target features in urdu_word_dict
                    word_dict_entries = config.word_dict.get(word_form, [])
                    for entry in word_dict_entries:
                        if (entry.get('upos') == target_upos and 
                            entry.get('feats') == target_feats):
                            if word_form != original_word:  # Don't replace with the same word
                                replacements.append((word_form, sub_err_annotation['id']))
                            break
                            
        except Exception as e:
            continue
    
    # Remove duplicates
    return list(set(replacements))

def insertion_infliction(sentence_words: List[Dict], word_ind: int, 
                        ins_err_annotations: List[Dict]) -> List[Tuple[str, str]]:
    """
    Perform insertion infliction (actually deletion from correct sentence)
    
    Args:
        sentence_words: List of word dictionaries from Stanza analysis
        word_ind: Index of the word to be deleted
        ins_err_annotations: List of insertion error annotations
    
    Returns:
        List containing the word to delete (for consistency with other functions)
    """
    deletions = []
    
    try:
        # Check if the current context matches any insertion annotation
        for ins_err_annotation in ins_err_annotations:
            # Check if kernel fits
            if fits_kernel(sentence_words, word_ind, 
                          ins_err_annotation['kernel_upos'], 
                          ins_err_annotation['kernel_feats'], 
                          INSERTION):
                # If it fits, we can delete this word to create the error
                deletions.append((sentence_words[word_ind]['text'], ins_err_annotation['id']))
                break
                
    except Exception as e:
        pass
    
    return list(set(deletions))

def deletion_infliction(sentence_words: List[Dict], word_ind: int, 
                       del_err_annotations: List[Dict]) -> List[Tuple[str, str]]:
    """
    Perform deletion infliction (actually insertion into correct sentence)
    
    Args:
        sentence_words: List of word dictionaries from Stanza analysis
        word_ind: Index where to insert the word
        del_err_annotations: List of deletion error annotations
    
    Returns:
        List of possible words to insert
    """
    insertions = []
    
    try:
        # Check if the current context matches any deletion annotation
        for del_err_annotation in del_err_annotations:
            # Check if kernel fits (for deletion, we check the gap between words)
            if fits_kernel(sentence_words, word_ind, 
                          del_err_annotation['kernel_upos'], 
                          del_err_annotation['kernel_feats'], 
                          DELETION):
                # If it fits, we can insert the deleted word here
                for deleted_word in del_err_annotation['deleted_words']:
                    if is_word_in_dict(deleted_word):
                        insertions.append((deleted_word, del_err_annotation['id']))

    except Exception as e:
        pass
    
    return list(set(insertions))

def inflict(correct_text: str) -> List[Tuple[str, str]]:
    """
    Main infliction function using Stanza for context-aware morphological analysis
    
    Args:
        correct_text: Clean sentence to inflict errors on
    
    Returns:
        List of tuples: (incorrect_sentence, error_type)
    """
    # Normalize the input text
    correct_text = normalize_characters(correct_text.strip())
    
    if not correct_text:
        return []
    
    # Analyze sentence with Stanza
    try:
        sentence_words = analyze_sentence_with_stanza(correct_text)
    except Exception as e:
        return []
    
    if not sentence_words:
        return []
    
    inflicted_pairs = []
    
    # Iterate through each word position
    for word_ind in range(len(sentence_words)):
        current_word = sentence_words[word_ind]['text']
        
        # Skip if current word is not in dictionary (OOV check)
        if not is_word_in_dict(current_word):
            continue
        
        # Create kernel for current position
        left_ind = word_ind - 1 if word_ind > 0 else -1
        right_ind = word_ind + 1 if word_ind + 1 < len(sentence_words) else len(sentence_words)
        
        # Create kernel UPOS
        kernel_upos = [NONE_LABEL, NONE_LABEL, NONE_LABEL]
        kernel_feats = [None, None, None]
        
        if left_ind >= 0:
            kernel_upos[0] = sentence_words[left_ind]['upos']
            kernel_feats[0] = UPOSFeats(sentence_words[left_ind]['upos'], 
                                       sentence_words[left_ind]['feats'])
        
        kernel_upos[1] = sentence_words[word_ind]['upos']
        kernel_feats[1] = UPOSFeats(sentence_words[word_ind]['upos'], 
                                   sentence_words[word_ind]['feats'])
        
        if right_ind < len(sentence_words):
            kernel_upos[2] = sentence_words[right_ind]['upos']
            kernel_feats[2] = UPOSFeats(sentence_words[right_ind]['upos'], 
                                       sentence_words[right_ind]['feats'])
        
        kernel_key = str(kernel_upos)
        
        # Check if this kernel exists in annotations
        if kernel_key in annotations:
            # Group annotations by type
            substitutions = []
            insertions = []
            deletions = []
            
            for annotation in annotations[kernel_key]:
                if annotation['type'] == SUBSTITUTION:
                    substitutions.append(annotation)
                elif annotation['type'] == INSERTION:
                    insertions.append(annotation)
                elif annotation['type'] == DELETION:
                    deletions.append(annotation)
            
            # Try substitution infliction
            if substitutions:
                replacements = substitution_infliction(sentence_words, word_ind, substitutions)
                for replacement in replacements:
                    # Create incorrect sentence
                    replacement_word, error_id = replacement
                    incorrect_words = [w['text'] for w in sentence_words]
                    incorrect_words[word_ind] = replacement_word
                    incorrect_sentence = ' '.join(incorrect_words)
                    inflicted_pairs.append((incorrect_sentence, error_id))

            # Try insertion infliction (delete current word)
            if insertions:
                deletions_possible = insertion_infliction(sentence_words, word_ind, insertions)
                if deletions_possible:
                    # Create incorrect sentence by deleting current word
                    error_id = deletions_possible[0][1]                
                    incorrect_words = [w['text'] for i, w in enumerate(sentence_words) if i != word_ind]
                    incorrect_sentence = ' '.join(incorrect_words)
                    inflicted_pairs.append((incorrect_sentence, error_id))  # Using first error_id

        # Check for deletion infliction (insert between words)
        if word_ind < len(sentence_words) - 1:  # Not the last word
            # Create kernel for deletion (gap between current and next word)
            del_kernel_upos = [NONE_LABEL, NONE_LABEL, NONE_LABEL]
            del_kernel_feats = [None, None, None]
            
            del_kernel_upos[0] = sentence_words[word_ind]['upos']
            del_kernel_feats[0] = UPOSFeats(sentence_words[word_ind]['upos'], 
                                           sentence_words[word_ind]['feats'])
            
            del_kernel_upos[2] = sentence_words[word_ind + 1]['upos']
            del_kernel_feats[2] = UPOSFeats(sentence_words[word_ind + 1]['upos'], 
                                           sentence_words[word_ind + 1]['feats'])
            
            del_kernel_key = str(del_kernel_upos)
            
            if del_kernel_key in annotations:
                deletions = [ann for ann in annotations[del_kernel_key] if ann['type'] == DELETION]
                if deletions:
                    insertions_possible = deletion_infliction(sentence_words, word_ind, deletions)
                    for insertion in insertions_possible:
                        # Create incorrect sentence by inserting word
                        insertion_word, error_id = insertion
                        incorrect_words = [w['text'] for w in sentence_words]
                        incorrect_words.insert(word_ind + 1, insertion_word)
                        incorrect_sentence = ' '.join(incorrect_words)
                        inflicted_pairs.append((incorrect_sentence, error_id))

    return inflicted_pairs

if __name__ == '__main__':
    # Load word dictionary
    config.word_dict = json.load(open('makhzan_wordFrequency_normalized', 'r', encoding='utf-8'))

    # Open output files
    corr_out_file = open('data/out/correct.txt', 'a', encoding='utf-8')
    incorr_out_file = open('data/out/incorrect.txt', 'a', encoding='utf-8')
    error_id_file = open('data/out/error_id.txt', 'a', encoding='utf-8')
    
    # Read correct sentences
    correct_text = open('data/cleaned_correct_corpus/makhzan_sentences.txt', 'r', encoding='utf-8').read()
    correct_text = normalize_characters(correct_text)
    lines = correct_text.split('\n')

    # Iterate through each line and inflict errors
    for i in tqdm.tqdm(range(0, len(lines))):
        inflicted_results = inflict(lines[i])
        if inflicted_results:
            for incorrect_sent, error_id in inflicted_results:
                corr_out_file.write(lines[i] + '\n')
                incorr_out_file.write(incorrect_sent + '\n')
                error_id_file.write(error_id + '\n')

        else:
            print(f"No errors could be inflicted on sentence {i}: '{lines[i]}'")