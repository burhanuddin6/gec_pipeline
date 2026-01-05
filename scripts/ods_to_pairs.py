import sys
import os
import pandas as pd

INPUT_FILE = sys.argv[1] if len(sys.argv) > 1 else "data\Manual_tagging_gold_ur_gec_dataset.ods"
SHEET_NAME = "manually_tagged_and_corrected"      # <<< CHANGE THIS
INCORRECT_TXT = "data\consolidated-gold-incorrect-nospell.txt"
CORRECT_TXT = "data\consolidated-gold-correct-nospell.txt"
SPELL_TXT = "data\consolidated-gold-spell_tags.txt"

SPELL_TAG = "R:SPELL"


def count_spelling_errors(row):
    """
    Priority:
    1. incorrect_word (comma/urdu-comma separated)
    2. correct_spellings
    3. manual_error_tags count of R:SPELL
    4. spelling_issue == YES → 1
    """
    def split_items(val):
        return [
            x.strip() for x in
            str(val)
            .replace("،", ",")
            .replace("؛", ",")
            .replace(";", ",")
            .replace("|", ",")
            .split(",")
            if x.strip()
        ]

    if pd.notna(row.get("incorrect_word")) and str(row["incorrect_word"]).strip():
        return len(split_items(row["incorrect_word"]))

    if pd.notna(row.get("correct_spellings")) and str(row["correct_spellings"]).strip():
        return len(split_items(row["correct_spellings"]))

    met = row.get("manual_error_tags")
    if pd.notna(met):
        cnt = str(met).lower().count("r:spell")
        if cnt > 0:
            return cnt

    si = row.get("spelling_issue")
    if pd.notna(si) and str(si).strip().upper() in {"YES", "Y", "1", "TRUE"}:
        return 1

    return 0


def main():
    ext = os.path.splitext(INPUT_FILE)[1].lower()
    read_kwargs = {}

    if ext == ".ods":
        read_kwargs["engine"] = "odf"

    df = pd.read_excel(INPUT_FILE, sheet_name=SHEET_NAME, **read_kwargs)

    required = {"incorrect_sentence", "correct_sentence"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    with open(INCORRECT_TXT, "w", encoding="utf-8") as f_inc, \
         open(CORRECT_TXT, "w", encoding="utf-8") as f_cor, \
         open(SPELL_TXT, "w", encoding="utf-8") as f_spell:

        for _, row in df.iterrows():
            inc = str(row["incorrect_sentence"]).strip()
            cor = str(row["correct_sentence"]).strip()

            if inc == "" and cor == "":
                continue

            f_inc.write(inc + "\n")
            f_cor.write(cor + "\n")

            n_spell = count_spelling_errors(row)

            if n_spell > 0:
                f_spell.write(" ".join([SPELL_TAG] * n_spell) + "\n")
            else:
                f_spell.write("NONE\n")

    print("Files written:")
    print(f"  {INCORRECT_TXT}")
    print(f"  {CORRECT_TXT}")
    print(f"  {SPELL_TXT}")
    print(f"Sheet used: {SHEET_NAME}")
    print(f"Total sentence pairs: {len(df)}")


if __name__ == "__main__":
    main()
