import json
import os
import contextlib
import stanza
from functools import lru_cache
from typing import Dict, List, Any, Optional

from .alignment import Alignment
from .constants import *
from misc.urduhack_normalization import normalize_characters
from . import config

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
                    processors='tokenize,pos,lemma',
                    device="cpu"
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
    with open('logs/gold_errors.log', 'a', encoding='utf-8') as f:
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
    Create kernel with context-aware features from Stanza analysis
    
    Returns:
        For SUBSTITUTION: (kernel_upos, incorrect_feats, correct_feats)
        For INSERTION/DELETION: (kernel_upos, kernel_feats)
    """
    kernel_upos = [NONE_LABEL, NONE_LABEL, NONE_LABEL]
    kernel_feats = [None, None, None]
    
    # Set left context
    if i_minus_one >= 0:
        if error_type == DELETION:
            kernel_upos[0] = incorrect_words[i_minus_one]['upos']
            kernel_feats[0] = UPOSFeats(incorrect_words[i_minus_one]['upos'], 
                                       incorrect_words[i_minus_one]['feats'])
        else:  # SUBSTITUTION or INSERTION
            kernel_upos[0] = incorrect_words[i_minus_one]['upos']
            kernel_feats[0] = UPOSFeats(incorrect_words[i_minus_one]['upos'], 
                                       incorrect_words[i_minus_one]['feats'])
    
    # Set middle position
    if error_type == SUBSTITUTION:
        # For substitution, we store both incorrect and correct word features
        kernel_upos[1] = incorrect_words[i]['upos']
        # We'll return separate incorrect and correct features
    elif error_type == INSERTION:
        # For insertion, the middle word is from the correct sentence (inserted word)
        kernel_upos[1] = correct_words[i]['upos']
        kernel_feats[1] = UPOSFeats(correct_words[i]['upos'], correct_words[i]['feats'])
    # For DELETION, middle remains NONE_LABEL
    
    # Set right context
    if error_type == DELETION:
        if i_plus_one < len(incorrect_words):
            kernel_upos[2] = incorrect_words[i_plus_one]['upos']
            kernel_feats[2] = UPOSFeats(incorrect_words[i_plus_one]['upos'], 
                                       incorrect_words[i_plus_one]['feats'])
    else:  # SUBSTITUTION or INSERTION
        if i_plus_one < len(incorrect_words):
            kernel_upos[2] = incorrect_words[i_plus_one]['upos']
            kernel_feats[2] = UPOSFeats(incorrect_words[i_plus_one]['upos'], 
                                       incorrect_words[i_plus_one]['feats'])
    
    if error_type == SUBSTITUTION:
        # Return kernel UPOS and separate features for incorrect and correct words
        incorrect_feats = UPOSFeats(incorrect_words[i]['upos'], incorrect_words[i]['feats'])
        correct_feats = UPOSFeats(correct_words[i]['upos'], correct_words[i]['feats'])
        return kernel_upos, incorrect_feats, correct_feats
    else:
        # For INSERTION and DELETION, return kernel UPOS and features
        return kernel_upos, kernel_feats

def annotate(id :int, incorrect_text: str, correct_text: str, kernel_sorted_annotations: Dict):
    """
    Main annotation function using Stanza for context-aware morphological analysis
    """
    # Normalize the input texts
    incorrect_text = normalize_characters(incorrect_text)
    correct_text = normalize_characters(correct_text)
    
    # Track number of errors in this sentence
    errors_in_sentence = 0
    
    # Analyze sentences with Stanza
    try:
        incorrect_words = analyze_sentence_with_stanza(incorrect_text)
        correct_words = analyze_sentence_with_stanza(correct_text)
    except Exception as e:
        log(f"Stanza analysis failed for sentences: {incorrect_text} | {correct_text} | Error: {e}")
        return kernel_sorted_annotations, errors_in_sentence
    
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
                    log(f"OOV check failed for substitution: {incorrect_word} -> {correct_word} at sentence index {id}")
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
                
                errors_in_sentence += 1
                    
            except Exception as e:
                log(f"Error processing substitution: {e} at sentence index {id}")
                continue

        elif op == DELETION:
            try:
                deleted_word = incorrect_words[i1]['text']
                
                # OOV check for the deleted word and context words
                if not is_word_in_dict(deleted_word):
                    log(f"OOV check failed for deleted word: {deleted_word} at sentence index {id}")
                    continue
                
                # OOV check for context words
                i_minus_one = i1 - 1 if i1 > 0 else -1
                i_plus_one = i1 + 1 if i1 + 1 < len(incorrect_words) else len(incorrect_words)
                
                # Check left/right context individually and report which context words failed
                missing = []
                left_word = incorrect_words[i_minus_one]['text'] if i_minus_one >= 0 else None
                right_word = incorrect_words[i_plus_one]['text'] if i_plus_one < len(incorrect_words) else None

                if left_word is not None and not is_word_in_dict(left_word):
                    missing.append(("left", left_word))
                if right_word is not None and not is_word_in_dict(right_word):
                    missing.append(("right", right_word))

                if missing:
                    # Build informative message with context and which side(s) failed
                    context_repr = f"left='{left_word}'" if left_word is not None else "left=None"
                    context_repr += f", right='{right_word}'" if right_word is not None else ", right=None"
                    missing_str = ", ".join([f"{pos}='{w}'" for pos, w in missing])
                    log(f"OOV check failed for deletion around: '{deleted_word}' at sentence index {id}; context: {context_repr}; missing: {missing_str}")
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
                
                errors_in_sentence += 1
                    
            except Exception as e:
                log(f"Error processing deletion: {e} at sentence index {id}")
                continue

        elif op == INSERTION:
            try:
                inserted_word = correct_words[j1]['text']
                
                # OOV check for inserted word and all context words in kernel
                if not is_word_in_dict(inserted_word):
                    log(f"OOV check failed for inserted word: {inserted_word} at sentence index {id}")
                    continue
                
                # For insertion, we need to check context in the correct sentence
                j_minus_one = j1 - 1 if j1 > 0 else -1
                j_plus_one = j1 + 1 if j1 + 1 < len(correct_words) else len(correct_words)
                
                # Check left/right context individually and report which context words failed
                missing = []
                left_word = correct_words[j_minus_one]['text'] if j_minus_one >= 0 else None
                right_word = correct_words[j_plus_one]['text'] if j_plus_one < len(correct_words) else None

                if left_word is not None and not is_word_in_dict(left_word):
                    missing.append(("left", left_word))
                if right_word is not None and not is_word_in_dict(right_word):
                    missing.append(("right", right_word))

                if missing:
                    context_repr = f"left='{left_word}'" if left_word is not None else "left=None"
                    context_repr += f", right='{right_word}'" if right_word is not None else ", right=None"
                    missing_str = ", ".join([f"{pos}='{w}'" for pos, w in missing])
                    log(f"OOV check failed for insertion around: '{inserted_word}' at sentence index {id}; context: {context_repr}; missing: {missing_str}")
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
                
                errors_in_sentence += 1
                    
            except Exception as e:
                log(f"Error processing insertion: {e} at sentence index {id}")
                continue

    return kernel_sorted_annotations, errors_in_sentence

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
    config.word_dict = json.load(open('data/makhzan_wordFrequency_normalized.json', 'r', encoding='utf-8'))

    # Load input texts
    orig_text = open('data/consolidated_gold_incorrect_no_punct.txt', 'r', encoding='utf-8').read()
    cor_text = open('data/consolidated_gold_correct_no_punct.txt', 'r', encoding='utf-8').read()

    orig_text = normalize_characters(orig_text)
    cor_text = normalize_characters(cor_text)
    
    try:
        num_processed_lines = int(open('logs/gold_num_processed_lines.txt', 'r').read())
    except:
        num_processed_lines = 0
        
    orig_text = orig_text.split('\n')[num_processed_lines:]
    cor_text = cor_text.split('\n')[num_processed_lines:]
    
    if num_processed_lines == 0:
        annotations = {}
    else:
        annotations = json.load(open('data/gold_annotations.json', 'r', encoding='utf-8'), object_hook=custom_decoder)

    print(f"Starting from line number: {num_processed_lines}")
    print(f"Number of existing annotations: {len(annotations)}")
    
    # Statistics tracking: errors per sentence
    from collections import defaultdict
    error_statistics = defaultdict(int)  # {num_errors: count}
    
    for id, (sentence1, sentence2) in enumerate(zip(orig_text, cor_text)):
        if sentence1.strip() and sentence2.strip():  # Skip empty lines
            annotations, num_errors = annotate(id, sentence1.strip(), sentence2.strip(), annotations)
            error_statistics[num_errors] += 1
        
        num_processed_lines += 1
        if num_processed_lines % 1000 == 0:
            # Save progress
            with open('data/gold_annotations.json', 'w', encoding='utf-8') as f:
                json.dump(annotations, f, ensure_ascii=False, indent=2, cls=UPOSFeatsEncoder)
            with open('logs/gold_num_processed_lines.txt', 'w') as f:
                f.write(str(num_processed_lines))
            # Save error statistics
            with open('logs/gold_error_statistics.json', 'w', encoding='utf-8') as f:
                json.dump(dict(error_statistics), f, ensure_ascii=False, indent=2)
            print(f"Processed {num_processed_lines} lines, saved checkpoint")
        
        if num_processed_lines % 100 == 0:
            print(f"Total lines processed: {num_processed_lines}")
    
    # Save final error statistics
    with open('logs/gold_error_statistics.json', 'w', encoding='utf-8') as f:
        json.dump(dict(error_statistics), f, ensure_ascii=False, indent=2)
    
    # Print summary
    print("\n=== Error Statistics ===")
    print(f"Total sentences processed: {sum(error_statistics.values())}")
    for num_errors in sorted(error_statistics.keys()):
        count = error_statistics[num_errors]
        print(f"Sentences with {num_errors} error(s): {count}")