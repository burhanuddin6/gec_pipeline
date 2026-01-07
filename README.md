# Grammatical Error Correction (GEC) Data Pipeline

This repository contains the pipeline for generating a synthetic Grammatical Error Correction (GEC) dataset for Urdu. The pipeline consists of three main stages: Wikipedia Extraction & Cleaning, Error Annotation, and Error Infliction.

## Prerequisites

1.  **Python 3.x**
2.  **Dependencies**: Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: If `en_core_web_sm` is missing, install it via: `python -m spacy download en_core_web_sm`*

3.  **Required Data Files**:
    *   `data/makhzan_wordFrequency_normalized.json`: A dictionary of Urdu words for Out-of-vocabulary (OOV) checks.
    * `data/cleaned_correct_corpus/makhzan_sentences.txt`: A clean corpus generated using `data/cleaned_correct_corpus/clean_corpus_cloning.ipynb` notebook.
    

---

## Pipeline Steps

### 1. WikiEdits Extraction and Cleaning
**Script:** `data_generation/clean.py`

This step processes the raw Wikipedia edits to create clean training and testing datasets of corrected/incorrect sentence pairs. `data/raw_wikiedits.txt` is the extracted Wikipedia edits text file generated via `UrWikiEditsExtractor.ipynb` notebook

*   **Input**: `data/raw_wikiedits.txt`
*   **Execution**:
    ```bash
    python data_generation/clean.py
    ```
*   **Process**:
    1.  Splits raw edits into `correct.txt` and `incorrect.txt`.
    2.  Removes duplicate edits and highly similar sentence pairs.
    3.  Cleans miscellaneous issues (e.g., date-only changes).
    4.  (Optional) Splits into Train/Test sets if `train_test_split()` is enabled in the script.
*   **Outputs** (in `data/wikiedits/`):
    *   `correct2.txt`, `incorrect2.txt`
    *   `train_correct.txt`, `train_incorrect.txt` (if split)

**Configuration (Hardcoded Paths)**:
To use different input/output files, edit `data_generation/clean.py`:
*   Function `create_two_files_from_one(filename='data/raw_wikiedits.txt', ...)`: Change `filename` to your input path.
*   Function `remove_duplicate_edits(...)`: Verify `corrfilename` and `incorrfilename` match your data flow.

### 2. Error Annotation
**Script:** `error_annotation/annotation_stanza.py`

This step uses the clean aligned pairs to generate grammatical error annotations in JSON format using Stanza for more accurate morphological analysis.

*   **Input**: `data/wikiedits/incorrect2.txt` and `data/wikiedits/correct2.txt`.
*   **Execution**:
    ```bash
    python error_annotation/annotation_stanza.py
    ```
*   **Process**:
    *   The script aligns sentences and identifies error types (Substitution, Deletion, Insertion).
    *   It tracks progress in `logs/num_processed_lines.txt`. **Note**: If starting fresh, ensure this file contains `0` or is empty.
    *   It saves checkpoints every 1000 lines.
*   **Output**: `data/annotations.json`

**Configuration (Hardcoded Paths)**:
Edit `error_annotation/annotation_stanza.py` (near the end of the file) to point to your dataset:
*   `orig_text = open('data/wikiedits/incorrect2.txt', ...)`
*   `cor_text = open('data/wikiedits/correct2.txt', ...)`

#### 2b. Assign Error IDs
**Script:** `data_generation/assign_error_ids.py`

This intermediate step assigns unique IDs to the generated annotations, required for the infliction step.

*   **Input**: `data/annotations.json`
*   **Execution**:
    ```bash
    python data_generation/assign_error_ids.py
    ```
*   **Output**: `data/annotations_with_ids.json`

### 3. Error Infliction
**Script:** `error_annotation/infliction_stanza.py`

This step inflicts the annotated errors onto a new, clean corpus to generate a large-scale synthetic GEC dataset.

*   **Prerequisites**:
    1.  `data/annotations_with_ids.json`: From Step 2b.
    2.  `data/lemma_word_dict.json`: Generated from `urdu_word_dict.json`.
        *   Run `python data_generation/generate_lemma_dict_from_word_dict.py` if missing.
    3.  `data/cleaned_correct_corpus/makhzan_sentences.txt`: The clean target corpus.

*   **Execution**:
    ```bash
    python error_annotation/infliction_stanza.py
    ```
*   **Process**:
    *   Reads the clean corpus sentences.
    *   Uses Stanza to analyze sentences and matches them to potential error patterns from the annotations.
    *   substitutes/inserts/deletes words to create an "incorrect" version.
*   **Outputs** (in `data/out/`):
    *   `correct.txt`: The original clean sentences.
    *   `incorrect.txt`: The sentences with inflicted errors.
    *   `error_id.txt`: The IDs of the errors inflicted for each line.

**Configuration (Hardcoded Paths)**:
Edit `error_annotation/infliction_stanza.py` to change inputs/outputs:
*   Lines 15-16: Paths for `annotations_with_ids.json` and `lemma_word_dict.json`.
*   Main Block (bottom of file):
    *   Input Corpus: `open('data/cleaned_correct_corpus/makhzan_sentences.txt', ...)`
    *   Outputs: `open('data/out/correct.txt', ...)` etc.

## Helper Scripts

*   `data_generation/generate_lemma_dict_from_word_dict.py`: Converts `urdu_word_dict.json` to a lemma-based lookup `lemma_word_dict.json`.
*   `data_generation/assign_error_ids.py`: Adds IDs to the raw `annotations.json`.
