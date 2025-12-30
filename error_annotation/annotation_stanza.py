import json
import os
import contextlib
import stanza
from functools import lru_cache
from typing import Dict, List, Any, Optional

from error_annotation.alignment import Alignment
from error_annotation.constants import *
from misc.urduhack_normalization import normalize_characters
from error_annotation import config

import tqdm

NUM_SPELLING_ISSUES = 0

class UPOSFeats:
    def __init__(self, upos: str, feats: str):
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

class StanzaPipeline:
    """Singleton class to manage Stanza pipeline with LRU cache"""
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
            print("✅ Stanza pipeline initialized successfully.")
        return self._pipeline

# Global pipeline instance
stanza_pipeline = StanzaPipeline()

@lru_cache(maxsize=10000)
def analyze_sentence_with_stanza(sentence_text: str) -> List[Dict[str, Any]]:
    """
    Analyze sentence with Stanza and return word information.
    Uses LRU cache to avoid reprocessing the same sentences.
    
    Returns: List of dictionaries with word info: [{'text': str, 'lemma': str, 'upos': str, 'feats': str}, ...]
    """
    nlp = stanza_pipeline.get_pipeline()
    doc = nlp(sentence_text)
    
    words_info = []
    for sentence in doc.sentences:
        for word in sentence.words:
            words_info.append({
                'text': word.text,
                'lemma': word.lemma,
                'upos': word.upos,
                'feats': word.feats if word.feats else 'None'
            })
    
    return words_info

def log(error):
    with open('logs/errors.log', 'a', encoding='utf-8') as f:
        f.write(f"{error}\n")

def is_word_in_dict(word: str) -> bool:
    """Check if word exists in urdu_word_dict.json (OOV check)"""
    return word in config.word_dict

def insertion_error_exist(t_annot, type_annotation):
    if t_annot['type'] != INSERTION:
        return False
    if not all([(type_annotation['kernel_upos'][i] == t_annot['kernel_upos'][i]) for i in range(KERNEL_SIZE)]):
        return False
    # Check if the context-specific features match
    if not all([(type_annotation['kernel_feats'][i].upos == t_annot['kernel_feats'][i].upos and 
                type_annotation['kernel_feats'][i].feats == t_annot['kernel_feats'][i].feats) 
                for i in range(KERNEL_SIZE)]):
        return False
    return True
    
def substitution_error_exist(t_annot, type_annotation):
    if t_annot['type'] != SUBSTITUTION:
        return False
    # Check if the kernel UPOS matches
    if not all([(type_annotation['kernel_upos'][i] == t_annot['kernel_upos'][i]) for i in range(KERNEL_SIZE)]):
        return False
    # Check if the context-specific features match for incorrect and correct words
    if (type_annotation['incorrect_feats'].upos != t_annot['incorrect_feats'].upos or
        type_annotation['incorrect_feats'].feats != t_annot['incorrect_feats'].feats or
        type_annotation['correct_feats'].upos != t_annot['correct_feats'].upos or
        type_annotation['correct_feats'].feats != t_annot['correct_feats'].feats):
        return False
    return True

def deletion_error_exist(t_annot, type_annotation):
    if t_annot['type'] != DELETION:
        return False
    if t_annot['deleted_words'] != type_annotation['deleted_words']:
        return False
    # Check if the context kernel matches (only left and right, middle is NONE for deletion)
    if not all([
                (type_annotation['kernel_upos'][i] == t_annot['kernel_upos'][i])
                for i in range(KERNEL_SIZE) if i != KERNEL_CENTER
                ]):
        return False
    # Check context features (excluding middle position)
    if not all([
                (type_annotation['kernel_feats'][i].upos == t_annot['kernel_feats'][i].upos and
                 type_annotation['kernel_feats'][i].feats == t_annot['kernel_feats'][i].feats)
                for i in range(KERNEL_SIZE) if i != KERNEL_CENTER
                ]):
        return False
    return True

def set_kernel_with_stanza(incorrect_words: List[Dict], correct_words: List[Dict], 
                          i_minus_one: int, i: int, i_plus_one: int, error_type: str):
    """
    Create kernel with context-aware features from Stanza analysis.
    Generalizes to support variable KERNEL_SIZE.
    
    Args:
        incorrect_words: List of word dictionaries from incorrect sentence
        correct_words: List of word dictionaries from correct sentence
        i_minus_one: Legacy parameter (kept for backward compatibility, represents i-1)
        i: Center position index
        i_plus_one: Legacy parameter (kept for backward compatibility, represents i+1)
        error_type: One of SUBSTITUTION, INSERTION, or DELETION
    
    Returns:
        For SUBSTITUTION: (kernel_upos, incorrect_feats, correct_feats)
        For INSERTION/DELETION: (kernel_upos, kernel_feats)
    """
    # Initialize kernel arrays with NONE_LABEL
    kernel_upos = [NONE_LABEL] * KERNEL_SIZE
    kernel_feats = [None] * KERNEL_SIZE
    
    # Build context window around position i
    for offset in range(-KERNEL_RADIUS, KERNEL_RADIUS + 1):
        kernel_idx = offset + KERNEL_RADIUS  # Map offset to kernel array index [0, KERNEL_SIZE-1]
        context_pos = i + offset
        
        # Handle center position specially based on error type
        if offset == 0:  # Center position
            if error_type == SUBSTITUTION:
                # For substitution, center stores the incorrect word's UPOS
                kernel_upos[kernel_idx] = incorrect_words[i]['upos']
                # Separate incorrect and correct features will be returned at the end
            elif error_type == INSERTION:
                # For insertion, center is the inserted word from correct sentence
                if i < len(correct_words):
                    kernel_upos[kernel_idx] = correct_words[i]['upos']
                    kernel_feats[kernel_idx] = UPOSFeats(correct_words[i]['upos'], 
                                                         correct_words[i]['feats'])
            # For DELETION, center remains NONE_LABEL
            
        else:  # Context positions (not center)
            if error_type == DELETION:
                # For DELETION, use incorrect_words for all context
                if 0 <= context_pos < len(incorrect_words):
                    kernel_upos[kernel_idx] = incorrect_words[context_pos]['upos']
                    kernel_feats[kernel_idx] = UPOSFeats(incorrect_words[context_pos]['upos'],
                                                         incorrect_words[context_pos]['feats'])
            elif error_type == INSERTION:
                # For INSERTION, use correct_words for context (grammatically correct context)
                if 0 <= context_pos < len(correct_words):
                    kernel_upos[kernel_idx] = correct_words[context_pos]['upos']
                    kernel_feats[kernel_idx] = UPOSFeats(correct_words[context_pos]['upos'],
                                                         correct_words[context_pos]['feats'])
            else:  # SUBSTITUTION
                # For SUBSTITUTION, use incorrect_words for context (indices align since same length)
                if 0 <= context_pos < len(incorrect_words):
                    kernel_upos[kernel_idx] = incorrect_words[context_pos]['upos']
                    kernel_feats[kernel_idx] = UPOSFeats(incorrect_words[context_pos]['upos'],
                                                         incorrect_words[context_pos]['feats'])
    
    # Return based on error type
    if error_type == SUBSTITUTION:
        # Return kernel UPOS and separate features for incorrect and correct words
        incorrect_feats = UPOSFeats(incorrect_words[i]['upos'], incorrect_words[i]['feats'])
        correct_feats = UPOSFeats(correct_words[i]['upos'], correct_words[i]['feats'])
        return kernel_upos, incorrect_feats, correct_feats
    else:
        # For INSERTION and DELETION, return kernel UPOS and features
        return kernel_upos, kernel_feats

def annotate(incorrect_text: str, correct_text: str, kernel_sorted_annotations: Dict):
    """
    Main annotation function using Stanza for context-aware morphological analysis
    """
    # Normalize the input texts
    incorrect_text = normalize_characters(incorrect_text)
    correct_text = normalize_characters(correct_text)
    
    # Analyze sentences with Stanza
    try:
        incorrect_words = analyze_sentence_with_stanza(incorrect_text)
        correct_words = analyze_sentence_with_stanza(correct_text)
    except Exception as e:
        log(f"Stanza analysis failed for sentences: {incorrect_text} | {correct_text} | Error: {e}")
        return kernel_sorted_annotations
    
    # Create simple word objects for alignment (compatibility with existing alignment code)
    class SimpleWord:
        def __init__(self, text, lemma, upos):
            self.text = text
            self.lemma = lemma
            self.upos = upos
    
    class SimpleSentence:
        def __init__(self, words_info):
            self.words = [SimpleWord(w['text'], w['lemma'], w['upos']) for w in words_info]
            self.text = ' '.join([w['text'] for w in words_info])
    
    incorrect_sentence = SimpleSentence(incorrect_words)
    correct_sentence = SimpleSentence(correct_words)
    
    # Perform alignment
    alignment = Alignment(incorrect_sentence, correct_sentence)
    seq = alignment.align_seq
    
    for op, i1, i2, j1, j2 in seq:
        if op == SUBSTITUTION:
            try:
                # OOV check for both incorrect and correct words
                incorrect_word = incorrect_words[i1]['text']
                correct_word = correct_words[j1]['text']
                
                if not is_word_in_dict(incorrect_word) or not is_word_in_dict(correct_word):
                    log(f"OOV check failed for substitution: {incorrect_word} -> {correct_word}")
                    continue
                
                # Get context indices
                i_minus_one = i1 - 1 if i1 > 0 else -1
                i_plus_one = i1 + 1 if i1 + 1 < len(incorrect_words) else len(incorrect_words)
                
                # Create kernel with context-aware features
                kernel_upos, incorrect_feats, correct_feats = set_kernel_with_stanza(
                    incorrect_words, correct_words, i_minus_one, i1, i_plus_one, SUBSTITUTION
                )
                
                kernel_key = str(kernel_upos)
                
                type_annotation = {
                    'type': SUBSTITUTION,
                    'kernel_upos': kernel_upos,
                    'incorrect_feats': incorrect_feats,
                    'correct_feats': correct_feats,
                    'occurrence': 1,
                    'incorrect_text': incorrect_text,
                    'correct_text': correct_text,
                    'alignment': "  ".join([",".join([str(elem) for elem in tup]) for tup in alignment.align_seq])
                }
                
                # Check if this exact substitution pattern exists
                if kernel_key in kernel_sorted_annotations:
                    found_existing = False
                    for existing_annotation in kernel_sorted_annotations[kernel_key]:
                        if substitution_error_exist(existing_annotation, type_annotation):
                            existing_annotation['occurrence'] += 1
                            found_existing = True
                            break
                    if not found_existing:
                        kernel_sorted_annotations[kernel_key].append(type_annotation)
                else:
                    kernel_sorted_annotations[kernel_key] = [type_annotation]
                    
            except Exception as e:
                log(f"Error processing substitution: {e}")
                continue

        elif op == DELETION:
            try:
                deleted_word = incorrect_words[i1]['text']
                
                # OOV check for the deleted word and context words
                if not is_word_in_dict(deleted_word):
                    log(f"OOV check failed for deleted word: {deleted_word}")
                    continue
                
                # OOV check for context words
                i_minus_one = i1 - 1 if i1 > 0 else -1
                i_plus_one = i1 + 1 if i1 + 1 < len(incorrect_words) else len(incorrect_words)
                
                context_words_valid = True
                if i_minus_one >= 0 and not is_word_in_dict(incorrect_words[i_minus_one]['text']):
                    context_words_valid = False
                if i_plus_one < len(incorrect_words) and not is_word_in_dict(incorrect_words[i_plus_one]['text']):
                    context_words_valid = False
                
                if not context_words_valid:
                    log(f"OOV check failed for deletion context around: {deleted_word}")
                    continue
                
                # Create kernel for deletion (middle position is NONE)
                kernel_upos, kernel_feats = set_kernel_with_stanza(
                    incorrect_words, correct_words, i_minus_one, i1, i_plus_one, DELETION
                )
                
                kernel_key = str(kernel_upos)
                
                type_annotation = {
                    'type': DELETION,
                    'kernel_upos': kernel_upos,
                    'kernel_feats': kernel_feats,
                    'deleted_words': [deleted_word],
                    'occurrence': 1,
                    'incorrect_text': incorrect_text,
                    'correct_text': correct_text,
                    'alignment': "  ".join([",".join([str(elem) for elem in tup]) for tup in alignment.align_seq])
                }
                
                # Check if this exact deletion pattern exists
                if kernel_key in kernel_sorted_annotations:
                    found_existing = False
                    for existing_annotation in kernel_sorted_annotations[kernel_key]:
                        if deletion_error_exist(existing_annotation, type_annotation):
                            existing_annotation['occurrence'] += 1
                            found_existing = True
                            break
                    if not found_existing:
                        kernel_sorted_annotations[kernel_key].append(type_annotation)
                else:
                    kernel_sorted_annotations[kernel_key] = [type_annotation]
                    
            except Exception as e:
                log(f"Error processing deletion: {e}")
                continue

        elif op == INSERTION:
            try:
                inserted_word = correct_words[j1]['text']
                
                # OOV check for inserted word and all context words in kernel
                if not is_word_in_dict(inserted_word):
                    log(f"OOV check failed for inserted word: {inserted_word}")
                    continue
                
                # For insertion, we need to check context in the correct sentence
                j_minus_one = j1 - 1 if j1 > 0 else -1
                j_plus_one = j1 + 1 if j1 + 1 < len(correct_words) else len(correct_words)
                
                context_words_valid = True
                if j_minus_one >= 0 and not is_word_in_dict(correct_words[j_minus_one]['text']):
                    context_words_valid = False
                if j_plus_one < len(correct_words) and not is_word_in_dict(correct_words[j_plus_one]['text']):
                    context_words_valid = False
                
                if not context_words_valid:
                    log(f"OOV check failed for insertion context around: {inserted_word}")
                    continue
                
                # Create kernel for insertion
                kernel_upos, kernel_feats = set_kernel_with_stanza(
                    incorrect_words, correct_words, j_minus_one, j1, j_plus_one, INSERTION
                )
                
                kernel_key = str(kernel_upos)
                
                type_annotation = {
                    'type': INSERTION,
                    'kernel_upos': kernel_upos,
                    'kernel_feats': kernel_feats,
                    'inserted_word': inserted_word,
                    'occurrence': 1,
                    'incorrect_text': incorrect_text,
                    'correct_text': correct_text,
                    'alignment': "  ".join([",".join([str(elem) for elem in tup]) for tup in alignment.align_seq])
                }
                
                # Check if this exact insertion pattern exists
                if kernel_key in kernel_sorted_annotations:
                    found_existing = False
                    for existing_annotation in kernel_sorted_annotations[kernel_key]:
                        if insertion_error_exist(existing_annotation, type_annotation):
                            existing_annotation['occurrence'] += 1
                            found_existing = True
                            break
                    if not found_existing:
                        kernel_sorted_annotations[kernel_key].append(type_annotation)
                else:
                    kernel_sorted_annotations[kernel_key] = [type_annotation]
                    
            except Exception as e:
                log(f"Error processing insertion: {e}")
                continue

    return kernel_sorted_annotations

# Custom JSON encoder for UPOSFeats objects
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

if __name__ == '__main__':
    # Load word dictionary
    config.word_dict = json.load(open('makhzan_wordFrequency_normalized.json', 'r', encoding='utf-8'))

    # Load input texts
    orig_text = open('data/wikiedits/train_incorrect.txt', 'r', encoding='utf-8').read()
    cor_text = open('data/wikiedits/train_correct.txt', 'r', encoding='utf-8').read()

    if len(orig_text.split('\n')) != len(cor_text.split('\n')):
        raise ValueError("Original and Correct files have different number of lines.")

    orig_text = normalize_characters(orig_text)
    cor_text = normalize_characters(cor_text)
    
    try:
        num_processed_lines = int(open('logs/num_processed_lines.txt', 'r').read())
    except:
        num_processed_lines = 0
        
    orig_text = orig_text.split('\n')[num_processed_lines:]
    cor_text = cor_text.split('\n')[num_processed_lines:]
    
    if num_processed_lines == 0:
        annotations = {}
    else:
        annotations = json.load(open('data/annotations.json', 'r', encoding='utf-8'), object_hook=custom_decoder)

    print(f"Starting from line number: {num_processed_lines}")
    print(f"Number of existing annotations: {len(annotations)}")
    
    # for sentence1, sentence2 in zip(orig_text, cor_text):
    for sentence1, sentence2 in tqdm.tqdm(zip(orig_text, cor_text), total=min(len(orig_text), len(cor_text))):
        if sentence1.strip() and sentence2.strip():  # Skip empty lines
            annotations = annotate(sentence1.strip(), sentence2.strip(), annotations)
        
        num_processed_lines += 1
        if num_processed_lines % STEP_COUNT == 0:
            # Save progress
            with open('data/annotations.json', 'w', encoding='utf-8') as f:
                json.dump(annotations, f, ensure_ascii=False, indent=2, cls=UPOSFeatsEncoder)
            with open('logs/num_processed_lines.txt', 'w') as f:
                f.write(str(num_processed_lines))
            print(f"Processed {num_processed_lines} lines, saved checkpoint")