import sys
import os
import pandas as pd

INPUT_FILE = sys.argv[1] if len(sys.argv) > 1 else "data/Manual_tagging_gold_ur_gec_dataset.ods"
SHEET_NAME = "manually_tagged_and_corrected"   # <<< CHANGE THIS
INCORRECT_TXT = "data/consolided-gold-incorrect-nospell.txt"
CORRECT_TXT = "data/consolided-gold-correct-nospell.txt"

def main():
    ext = os.path.splitext(INPUT_FILE)[1].lower()

    read_kwargs = {}
    if ext == ".ods":
        read_kwargs["engine"] = "odf"

    df = pd.read_excel(INPUT_FILE, sheet_name=SHEET_NAME, **read_kwargs)

    required_cols = {"incorrect_sentence", "correct_sentence"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    with open(INCORRECT_TXT, "w", encoding="utf-8") as f_inc, \
         open(CORRECT_TXT, "w", encoding="utf-8") as f_cor:

        for inc, cor in zip(df["incorrect_sentence"], df["correct_sentence"]):
            inc = str(inc).strip()
            cor = str(cor).strip()

            if inc == "" and cor == "":
                continue

            f_inc.write(inc + "\n")
            f_cor.write(cor + "\n")

    print(f"Wrote {INCORRECT_TXT}")
    print(f"Wrote {CORRECT_TXT}")
    print(f"Sheet used: {SHEET_NAME}")
    print(f"Total sentence pairs: {len(df)}")

if __name__ == "__main__":
    main()
