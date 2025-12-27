import subprocess
import os
import json
from annotation import custom_decoder

args_dict = {
    'bad_samples': {
        'word_dict': 'makhzan_wordFrequency_normalized.json',
        'valid_grammar': 'valid_grammar_features_makhzan_only.json',
        'error_log': 'logs/errors.log',
        'log_lines': 'logs/num_processed_lines.txt',
        'incorrect_file': 'data/test/bad_samples/original.txt',
        'correct_file': 'data/test/bad_samples/correct.txt',
        'annotations': 'data/test/bad_samples/annotations.json',
        'excluded': 'data/test/bad_samples/excluded_samples.json',
        'included': 'data/test/bad_samples/included_samples.json',
    },
    'good_samples': {
        'word_dict': 'makhzan_wordFrequency_normalized.json',
        'valid_grammar': 'valid_grammar_features_makhzan_only.json',
        'error_log': 'logs/errors.log',
        'log_lines': 'logs/num_processed_lines.txt',
        'incorrect_file': 'data/test/good_samples/original.txt',
        'correct_file': 'data/test/good_samples/correct.txt',
        'annotations': 'data/test/good_samples/annotations.json',
        'excluded': 'data/test/good_samples/excluded_samples.json',
        'included': 'data/test/good_samples/included_samples.json',
    },
    'spelling_samples': {
        'word_dict': 'makhzan_wordFrequency_normalized.json',
        'valid_grammar': 'valid_grammar_features_makhzan_only.json',
        'error_log': 'logs/errors.log',
        'log_lines': 'logs/num_processed_lines.txt',
        'incorrect_file': 'data/test/spelling_samples/original.txt',
        'correct_file': 'data/test/spelling_samples/correct.txt',
        'annotations': 'data/test/spelling_samples/annotations.json',
        'excluded': 'data/test/spelling_samples/excluded_samples.json',
        'included': 'data/test/spelling_samples/included_samples.json',
    },
}

def run_test(script_name, arguments):
    # Reset the processed lines file to "0"
    log_file_path = arguments['log_lines']
    print(f"--- Resetting {log_file_path} to 0 ---")
    with open(log_file_path, 'w') as f:
        f.write('0')

    # Construct the command list
    # Converts the dict into: ["--key", "value", "--key2", "value2"]
    cmd = ['python', script_name]
    for key, value in arguments.items():
        cmd.append(f'--{key}')
        cmd.append(str(value))

    print(f"--- Starting {script_name} ---")
    try:
        subprocess.run(cmd, check=True)
        print(f"--- {script_name} completed successfully ---\n")
    except subprocess.CalledProcessError as e:
        print(f"--- Error running {script_name}: {e} ---\n")

def metrics(script_name, args_dict):
    TP = len(json.load(open(args_dict['bad_samples']['excluded'], 'r', encoding='utf-8'), object_hook=custom_decoder))
    FN = len(json.load(open(args_dict['bad_samples']['included'], 'r', encoding='utf-8'), object_hook=custom_decoder))
    FP = len(json.load(open(args_dict['good_samples']['excluded'], 'r', encoding='utf-8'), object_hook=custom_decoder))
    TN = len(json.load(open(args_dict['good_samples']['included'], 'r', encoding='utf-8'), object_hook=custom_decoder))
    
    precision = TP/(TP + FP)
    recall = TP/(TP + FN)
    print(f"### Results for: {script_name} ###")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")

if __name__ == "__main__":
    scripts_to_test = ['error_annotation/annotation_stanza.py']
    
    for script in scripts_to_test:
        run_test(script, args_dict['bad_samples'])
        run_test(script, args_dict['good_samples'])
        metrics(script, args_dict)