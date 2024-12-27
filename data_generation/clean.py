import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from difflib import SequenceMatcher
import random

def similarity_ratio(a, b):
    return SequenceMatcher(None, a, b).ratio()

# import deepcopy
from copy import deepcopy
# Sample text with lines
def remove_duplicate_edits(corrfilename='data/wikiedits/correct.txt', incorrfilename='data/wikiedits/incorrect.txt'):
    corrfile, incorrfile = [open(file, 'r', encoding='utf-8').read().strip().split('\n') for file in [corrfilename, incorrfilename]]
    incorrfile_copy = deepcopy(incorrfile)
    store_removed = ""
    count = 0
    i = 0
    while (i < len(incorrfile)):
        try:
            line = incorrfile[i]
            i += 1
            index = corrfile.index(line)
            if index < i:
                # remove the duplicate edit only if it is before the current edit
                store_removed += incorrfile.pop(index) + '\n' + corrfile.pop(index) + '\n\n'
                count += 1
                i -= 1
                print("Removed duplicate edit ", i)
            else:
                print("Not removing duplicate edit")
        except:
            pass

    print(f'Removed {count} duplicate edits')
    # save back
    with open(corrfilename, 'w', encoding='utf-8') as f:
        f.write('\n'.join(corrfile))
    with open(incorrfilename, 'w', encoding='utf-8') as f:
        f.write('\n'.join(incorrfile))
    with open('data/wikiedits/duplicates.txt', 'w', encoding='utf-8') as f:
        f.write(f'{count} duplicate edits removed\n\n')
        f.write(store_removed)



def create_two_files_from_one(filename='wikiedits.txt', output_correct='data/wikiedits/correct.txt', output_incorrect='data/wikiedits/incorrect.txt'):
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.read().strip().split('\n\n')
        correct_lines = [line.split('\n')[1] for line in lines]
        incorrect_lines = [line.split('\n')[0] for line in lines]

    with open(output_correct, 'w', encoding='utf-8') as f:
        f.write('\n'.join(correct_lines))
    with open(output_incorrect, 'w', encoding='utf-8') as f:
        f.write('\n'.join(incorrect_lines))



def remove_similar_ones(corrfilename='data/wikiedits/correct.txt', incorrfilename='data/wikiedits/incorrect.txt', output_dir='data/wikiedits/'):
    corrlines = open(corrfilename, 'r', encoding='utf-8').read().strip().split('\n')
    incorrlines = open(incorrfilename, 'r', encoding='utf-8').read().strip().split('\n')
    remove_indices = []
    for i in range(len(corrlines)-1):
        print(i)
        j = i + 1
        # alignment = Alignment(lines[i], lines[j])
        # count_changes = sum(1 for op, _, _, _, _ in alignment.align_seq if op in [Alignment.SUBSTITUTION, Alignment.INSERTION, Alignment.DELETION])
        sim_ratio = similarity_ratio(corrlines[i], corrlines[j])
        if sim_ratio > 0.8:
            # sentences are too similar discard one
            print(f'{sim_ratio} {corrlines[i]}\n{corrlines[j]}')
            remove_indices.append(i)
    
    removed_lines = [f'{corrlines[i]}\n{incorrlines[i]}' for i in remove_indices]
    corrlines = [line for i, line in enumerate(corrlines) if i not in remove_indices]
    incorrlines = [line for i, line in enumerate(incorrlines) if i not in remove_indices]
    print(f'Removed {len(remove_indices)} similar edits')
    with open(output_dir + 'correct1.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(corrlines))
    with open(output_dir + 'incorrect1.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(incorrlines))
    with open(output_dir + 'similar.txt', 'w', encoding='utf-8') as f:
        f.write(f'{len(remove_indices)} similar edits removed\n\n')
        f.write('\n\n'.join(removed_lines))


    
'''
Issues with numeric characters:
1) Unnecessary spaces between numbers and commas, these sentences are reversed so need to correct that too
2) sentences starting with numbers have numbers at the end instead of the start
3) dates and times don't have - or : instead have spaces
4) some sentences only are dates so remove them by using ratio of numbers to total characters

Issues with punctuation:
1) Extra spaces after punctuation but that can be ignored
2) Only one quotation mark is present instead of two, so need to remove that

'''

import re

def clean_misc_issues(corrfilename='data/wikiedits/correct1.txt', incorrfilename='data/wikiedits/incorrect1.txt', numeric_ratio_threshold=0.3, output_dir='data/wikiedits/'):
    """
    Cleans a list of sentences based on numeric and punctuation-related issues.
    
    :param sentences: List of sentences to clean.
    :param numeric_ratio_threshold: Ratio threshold for numeric-only sentences.
    :return: List of cleaned sentences.
    """
    urdu_digits = "۰۱۲۳۴۵۶۷۸۹"
    english_to_urdu_map = str.maketrans("0123456789", urdu_digits)
    urdu_to_english_map = str.maketrans(urdu_digits, "0123456789")
    def fix_numeric_issues(sentence):
        # 2. Remove unnecessary spaces between numbers and commas
        sentence = re.sub(r'\s+([,])', r'\1', sentence)
        sentence = sentence.translate(english_to_urdu_map)
        return sentence

    
    def fix_punctuation_issues(sentence):
        # 1. Remove unmatched single quotation marks
        if sentence.count('"') == 1:
            sentence = sentence.replace('"', '')
        
        return sentence
    
    def is_numeric_dominant(sentence):
        # Calculate numeric character ratio
        total_chars = len(sentence)
        numeric_chars = sum(char.isdigit() for char in sentence)
        return (numeric_chars / total_chars) > numeric_ratio_threshold if total_chars > 0 else False

    def normalize(filename):
        from urduhack.normalization import normalize_characters
        text = open(filename, 'r', encoding='utf-8').read()
        normalized_text = normalize_characters(text)
        return normalized_text
        


    
    
    remove_indices = []
    corrsentences = normalize(corrfilename).strip().split('\n')
    incorrsentences = normalize(incorrfilename).strip().split('\n')

    for i in range(len(corrsentences)):
        # Remove numeric-dominant sentences
        if is_numeric_dominant(corrsentences[i]) or is_numeric_dominant(incorrsentences[i]):
            remove_indices.append(i)
            continue
        
    
        corrsentences[i] = fix_numeric_issues(corrsentences[i])
        incorrsentences[i] = fix_numeric_issues(incorrsentences[i])
        corrsentences[i] = fix_punctuation_issues(corrsentences[i])
        incorrsentences[i] = fix_punctuation_issues(incorrsentences[i])


    removed_lines = [f'{corrsentences[i]}\n{incorrsentences[i]}' for i in remove_indices]
    corrsentences = [line.strip() for i, line in enumerate(corrsentences) if i not in remove_indices]
    incorrsentences = [line.strip() for i, line in enumerate(incorrsentences) if i not in remove_indices]

    with open(output_dir + 'correct2.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(corrsentences))
    with open(output_dir + 'incorrect2.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(incorrsentences))
    with open(output_dir + 'clean_misc_issues.txt', 'w', encoding='utf-8') as f:
        f.write(f'{len(remove_indices)} numeric sentences removed\n\n')
        f.write('\n\n'.join(removed_lines))


def train_test_split(corrfilename='data/wikiedits/correct2.txt', incorrfilename='data/wikiedits/incorrect2.txt', output_dir='data/wikiedits/'):
    corrsentences = open(corrfilename, 'r', encoding='utf-8').read().strip().split('\n')
    incorrsentences = open(incorrfilename, 'r', encoding='utf-8').read().strip().split('\n')

    zipped = list(zip(corrsentences, incorrsentences))

    random.shuffle(zipped)

    corrsentences, incorrsentences = zip(*zipped)

    train_size = int(0.8 * len(corrsentences))
    with open(output_dir + 'train_correct.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(corrsentences[:train_size]))
    with open(output_dir + 'train_incorrect.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(incorrsentences[:train_size]))
    with open(output_dir + 'test_correct.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(corrsentences[train_size:]))
    with open(output_dir + 'test_incorrect.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(incorrsentences[train_size:]))

if __name__ == '__main__':
    # create_two_files_from_one()
    # remove_duplicate_edits()
    # remove_similar_ones()
    # clean_misc_issues()
    train_test_split()


