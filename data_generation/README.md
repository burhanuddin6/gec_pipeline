# Data Generation Directory

This directory contains the following files and folders:

## Files
- `generate_word_dict.py`: Uses `data/urdu_words.txt` and `data/conllu/` for generating `data/word_dict2.json` .
- `clean.py`: Script to clean and preprocess the Wikipedia edits dataset.
- `augment.py`: Script to augment the generated corpus (after infliction) with clean corpus
- `assign_error_ids.py`: Script to assign unique IDs to error annotations. Produces `data/annotations_with_ids.json` from `data/annotations.json`.
- `calc_perc.py`: Script to calculate the percentage of errors in the annotations. Produces `data/annotations_dict.json` from `data/annotations_with_ids.json`.
- `sample.py`: Script to sample the generated corpus (after infliction) based on percentages of annotations calculated in `data/annotations_dict.json`.

