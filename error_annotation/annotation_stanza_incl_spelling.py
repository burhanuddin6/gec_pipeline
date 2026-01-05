# save as run_annotation_no_oov.py
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

# NOTE: removed openpyxl usage and all OOV-related code

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
    os.makedirs('logs', exist_ok=True)
    with open('logs/gold_errors.log', 'a', encoding='utf-8') as f:
        f.write(f"{error}\n")

def make_error_tag(op, incorrect_word=None, correct_word=None,
                   incorrect_feats=None, correct_feats=None):
    if op == SUBSTITUTION:
        pos = incorrect_feats.upos
        # incorrect_word and correct_word expected to be dicts with 'lemma'
        if incorrect_word.get('lemma') != correct_word.get('lemma'):
            return f"R:{pos}"
        else:
            return f"R:{pos}:INFL"

    elif op == DELETION:
        return f"M:{incorrect_word.get('upos')}"

    elif op == INSERTION:
        return f"U:{correct_word.get('upos')}"

def insertion_error_exist(t_annot, type_annotation):
    if t_annot['type'] != INSERTION:
        return False
    if not all([(type_annotation['kernel_upos'][i] == t_annot['kernel_upos'][i]) for i in range(KERNEL_SIZE)]):
        return False
    if not all([(type_annotation['kernel_feats'][i].upos == t_annot['kernel_feats'][i].upos and 
                type_annotation['kernel_feats'][i].feats == t_annot['kernel_feats'][i].feats) 
                for i in range(KERNEL_SIZE)]):
        return False
    return True

def substitution_error_exist(t_annot, type_annotation):
    if t_annot['type'] != SUBSTITUTION:
        return False
    if not all([(type_annotation['kernel_upos'][i] == t_annot['kernel_upos'][i]) for i in range(KERNEL_SIZE)]):
        return False
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
    if not all([
                (type_annotation['kernel_upos'][i] == t_annot['kernel_upos'][i])
                for i in range(KERNEL_SIZE) if i != KERNEL_CENTER
                ]):
        return False
    if not all([
                (type_annotation['kernel_feats'][i].upos == t_annot['kernel_feats'][i].upos and
                 type_annotation['kernel_feats'][i].feats == t_annot['kernel_feats'][i].feats)
                for i in range(KERNEL_SIZE) if i != KERNEL_CENTER
                ]):
        return False
    return True

def set_kernel_with_stanza(incorrect_words: List[Dict], correct_words: List[Dict], 
                          i_minus_one: int, i: int, i_plus_one: int, error_type: str):
    kernel_upos = [NONE_LABEL, NONE_LABEL, NONE_LABEL]
    kernel_feats = [None, None, None]
    
    # Set left context
    if i_minus_one >= 0:
        kernel_upos[0] = incorrect_words[i_minus_one]['upos']
        kernel_feats[0] = UPOSFeats(incorrect_words[i_minus_one]['upos'], 
                                   incorrect_words[i_minus_one]['feats'])
    
    # Middle position
    if error_type == SUBSTITUTION:
        kernel_upos[1] = incorrect_words[i]['upos']
    elif error_type == INSERTION:
        kernel_upos[1] = correct_words[i]['upos']
        kernel_feats[1] = UPOSFeats(correct_words[i]['upos'], correct_words[i]['feats'])
    
    # Right context
    if i_plus_one < len(incorrect_words):
        kernel_upos[2] = incorrect_words[i_plus_one]['upos']
        kernel_feats[2] = UPOSFeats(incorrect_words[i_plus_one]['upos'], 
                                   incorrect_words[i_plus_one]['feats'])
    
    if error_type == SUBSTITUTION:
        incorrect_feats = UPOSFeats(incorrect_words[i]['upos'], incorrect_words[i]['feats'])
        correct_feats = UPOSFeats(correct_words[i]['upos'], correct_words[i]['feats'])
        return kernel_upos, incorrect_feats, correct_feats
    else:
        return kernel_upos, kernel_feats

def annotate(id :int, incorrect_text: str, correct_text: str, kernel_sorted_annotations: Dict):
    """
    Main annotation function using Stanza for context-aware morphological analysis.
    OOV logic removed entirely.
    Returns:
      kernel_sorted_annotations,
      errors_in_sentence,
      sentence_error_tags (list of strings),
      []  (placeholder for oov events - kept for compatibility)
    """
    sentence_error_tags: List[str] = []
    sentence_oov_events: List[Dict[str, Any]] = []  # always empty now

    incorrect_text = normalize_characters(incorrect_text)
    correct_text = normalize_characters(correct_text)
    
    errors_in_sentence = 0
    
    try:
        incorrect_words = analyze_sentence_with_stanza(incorrect_text)
        correct_words = analyze_sentence_with_stanza(correct_text)
    except Exception as e:
        log(f"Stanza analysis failed for sentences: {incorrect_text} | {correct_text} | Error: {e}")
        return kernel_sorted_annotations, errors_in_sentence, sentence_error_tags, sentence_oov_events
    
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
    
    alignment = Alignment(incorrect_sentence, correct_sentence)
    seq = alignment.align_seq
    
    for op, i1, i2, j1, j2 in seq:
        if op == SUBSTITUTION:
            try:
                incorrect_w = incorrect_words[i1]
                correct_w = correct_words[j1]
                
                i_minus_one = i1 - 1 if i1 > 0 else -1
                i_plus_one = i1 + 1 if i1 + 1 < len(incorrect_words) else len(incorrect_words)
                
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
                    'alignment': "  ".join([",".join([str(elem) for elem in tup]) for tup in alignment.align_seq]),
                    'lemma_mismatch': incorrect_w.get('lemma') != correct_w.get('lemma'),
                    'incorrect_lemma': incorrect_w.get('lemma'),
                    'correct_lemma': correct_w.get('lemma')
                }
                
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
                
                try:
                    tag = make_error_tag(SUBSTITUTION, incorrect_word=incorrect_w, correct_word=correct_w,
                                         incorrect_feats=incorrect_feats, correct_feats=correct_feats)
                    sentence_error_tags.append(tag)
                except Exception as e:
                    log(f"Tagging failed for substitution at sentence {id}: {e}")
                
                errors_in_sentence += 1
                    
            except Exception as e:
                log(f"Error processing substitution: {e} at sentence index {id}")
                continue

        elif op == DELETION:
            try:
                deleted_w = incorrect_words[i1]
                
                i_minus_one = i1 - 1 if i1 > 0 else -1
                i_plus_one = i1 + 1 if i1 + 1 < len(incorrect_words) else len(incorrect_words)
                
                kernel_upos, kernel_feats = set_kernel_with_stanza(
                    incorrect_words, correct_words, i_minus_one, i1, i_plus_one, DELETION
                )
                
                kernel_key = str(kernel_upos)
                
                type_annotation = {
                    'type': DELETION,
                    'kernel_upos': kernel_upos,
                    'kernel_feats': kernel_feats,
                    'deleted_words': [deleted_w['text']],
                    'occurrence': 1,
                    'incorrect_text': incorrect_text,
                    'correct_text': correct_text,
                    'alignment': "  ".join([",".join([str(elem) for elem in tup]) for tup in alignment.align_seq])
                }
                
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
                
                try:
                    tag = make_error_tag(DELETION, incorrect_word=deleted_w)
                    sentence_error_tags.append(tag)
                except Exception as e:
                    log(f"Tagging failed for deletion at sentence {id}: {e}")
                
                errors_in_sentence += 1
                    
            except Exception as e:
                log(f"Error processing deletion: {e} at sentence index {id}")
                continue

        elif op == INSERTION:
            try:
                inserted_w = correct_words[j1]
                
                j_minus_one = j1 - 1 if j1 > 0 else -1
                j_plus_one = j1 + 1 if j1 + 1 < len(correct_words) else len(correct_words)
                
                kernel_upos, kernel_feats = set_kernel_with_stanza(
                    incorrect_words, correct_words, j_minus_one, j1, j_plus_one, INSERTION
                )
                
                kernel_key = str(kernel_upos)
                
                type_annotation = {
                    'type': INSERTION,
                    'kernel_upos': kernel_upos,
                    'kernel_feats': kernel_feats,
                    'inserted_word': inserted_w['text'],
                    'occurrence': 1,
                    'incorrect_text': incorrect_text,
                    'correct_text': correct_text,
                    'alignment': "  ".join([",".join([str(elem) for elem in tup]) for tup in alignment.align_seq])
                }
                
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
                
                try:
                    tag = make_error_tag(INSERTION, correct_word=inserted_w)
                    sentence_error_tags.append(tag)
                except Exception as e:
                    log(f"Tagging failed for insertion at sentence {id}: {e}")
                
                errors_in_sentence += 1
                    
            except Exception as e:
                log(f"Error processing insertion: {e} at sentence index {id}")
                continue

    return kernel_sorted_annotations, errors_in_sentence, sentence_error_tags, sentence_oov_events

# JSON encoder/decoder for UPOSFeats objects
class UPOSFeatsEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, UPOSFeats):
            return obj.to_dict()
        return super().default(obj)

def custom_decoder(dct: dict):
    for key, value in dct.items():
        if key == 'kernel_feats' and isinstance(value, list):
            dct[key] = []
            for item in value:
                if isinstance(item, dict) and 'upos' in item and 'feats' in item:
                    dct[key].append(UPOSFeats(item['upos'], item['feats']))
                else:
                    dct[key].append(item)
        elif key in ['incorrect_feats', 'correct_feats'] and isinstance(value, dict):
            if 'upos' in value and 'feats' in value:
                dct[key] = UPOSFeats(value['upos'], value['feats'])
    return dct

if __name__ == '__main__':
    # Load input texts
    orig_text = open('data/consolidated-gold-incorrect-nospell.txt', 'r', encoding='utf-8').read()
    cor_text = open('data/consolidated-gold-correct-nospell.txt', 'r', encoding='utf-8').read()

    orig_text = normalize_characters(orig_text)
    cor_text = normalize_characters(cor_text)
    
    # load manual spell tags (one per line), fallback to NONE if missing
    try:
        with open('data/consolidated-gold-spell_tags.txt', 'r', encoding='utf-8') as f:
            all_spell_lines = [ln.rstrip("\n") for ln in f.readlines()]
    except FileNotFoundError:
        all_spell_lines = []
    
    try:
        num_processed_lines = int(open('logs/gold_num_processed_lines.txt', 'r').read())
    except:
        num_processed_lines = 0
        
    orig_lines = orig_text.split('\n')
    cor_lines = cor_text.split('\n')

    # Align spell lines length with inputs (if available)
    spell_lines = all_spell_lines if all_spell_lines else ["NONE"] * max(len(orig_lines), len(cor_lines))
    if len(spell_lines) < max(len(orig_lines), len(cor_lines)):
        # pad
        spell_lines += ["NONE"] * (max(len(orig_lines), len(cor_lines)) - len(spell_lines))

    # Slice for resume support
    orig_lines = orig_lines[num_processed_lines:]
    cor_lines = cor_lines[num_processed_lines:]
    spell_lines = spell_lines[num_processed_lines:]
    
    if num_processed_lines == 0:
        annotations = {}
    else:
        annotations = json.load(open('data/gold_annotations_no_oov.json', 'r', encoding='utf-8'), object_hook=custom_decoder)

    print(f"Starting from line number: {num_processed_lines}")
    print(f"Number of existing annotations: {len(annotations)}")
    
    from collections import defaultdict
    error_statistics = defaultdict(int)

    # Prepare output tags file (overwrite if starting fresh, append if resuming)
    os.makedirs('data', exist_ok=True)
    output_tags_path = 'data/gold_auto_plus_spell_tags.txt'
    mode = 'a' if num_processed_lines > 0 and os.path.exists(output_tags_path) else 'w'
    out_f = open(output_tags_path, mode, encoding='utf-8')

    for id, (sentence1, sentence2, spell_line) in enumerate(zip(orig_lines, cor_lines, spell_lines)):
        if sentence1.strip() and sentence2.strip():
            annotations, num_errors, tags, oov_events = annotate(id, sentence1.strip(), sentence2.strip(), annotations)
            auto_tags_str = " ".join(tags) if tags else "NONE"

            # parse spell_line (space-separated R:spell tokens) — allow commas too
            spell_line_clean = (spell_line or "NONE").strip()
            if spell_line_clean.upper() == "NONE" or spell_line_clean == "":
                spell_tokens = []
            else:
                # allow either space or comma separated
                if "," in spell_line_clean and " " not in spell_line_clean:
                    spell_tokens = [t.strip() for t in spell_line_clean.split(",") if t.strip()]
                else:
                    spell_tokens = [t.strip() for t in spell_line_clean.replace(",", " ").split() if t.strip()]

            # Combine auto tags + spell tokens
            if auto_tags_str == "NONE" and not spell_tokens:
                combined = "NONE"
            elif auto_tags_str == "NONE" and spell_tokens:
                combined = " ".join(spell_tokens)
            elif auto_tags_str != "NONE" and not spell_tokens:
                combined = auto_tags_str
            else:
                # both exist -> append spells after auto tags
                combined = auto_tags_str + " " + " ".join(spell_tokens)

            # Write combined tags line
            out_f.write(combined + "\n")

            error_statistics[num_errors] += 1
        
        num_processed_lines += 1

        # periodic checkpoints (same intervals as before)
        if num_processed_lines % 1000 == 0:
            # Save annotations for resume
            with open('data/gold_annotations.json', 'w', encoding='utf-8') as f:
                json.dump(annotations, f, ensure_ascii=False, indent=2, cls=UPOSFeatsEncoder)
            with open('logs/gold_num_processed_lines.txt', 'w') as f:
                f.write(str(num_processed_lines))
            with open('logs/gold_error_statistics.json', 'w', encoding='utf-8') as f:
                json.dump(dict(error_statistics), f, ensure_ascii=False, indent=2)
            print(f"Checkpoint saved at {num_processed_lines} lines processed")

        if num_processed_lines % 100 == 0:
            print(f"Total lines processed: {num_processed_lines}")

    out_f.close()

    # Final save annotations and logs
    with open('data/gold_annotations.json', 'w', encoding='utf-8') as f:
        json.dump(annotations, f, ensure_ascii=False, indent=2, cls=UPOSFeatsEncoder)
    with open('logs/gold_num_processed_lines.txt', 'w') as f:
        f.write(str(num_processed_lines))
    with open('logs/gold_error_statistics.json', 'w', encoding='utf-8') as f:
        json.dump(dict(error_statistics), f, ensure_ascii=False, indent=2)

    # summary
    print("\n=== Error Statistics ===")
    print(f"Total sentences processed: {sum(error_statistics.values())}")
    for num_errors in sorted(error_statistics.keys()):
        count = error_statistics[num_errors]
        print(f"Sentences with {num_errors} error(s): {count}")

    print(f"\nWrote tags to: {output_tags_path}")
