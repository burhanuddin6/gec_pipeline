import json

INPUT_JSON = "data/consolidated_10-07_02-AM.json"      # your input file
INCORRECT_OUT = "data/consolidated-gold-incorrect.txt"
CORRECT_OUT = "data/consolidated-gold-correct.txt"

with open(INPUT_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)

with open(INCORRECT_OUT, "w", encoding="utf-8") as f_incorrect, \
     open(CORRECT_OUT, "w", encoding="utf-8") as f_correct:

    for item in data:
        f_incorrect.write(item["incorrect_pair"].strip() + "\n")
        f_correct.write(item["correct_pair"].strip() + "\n")
