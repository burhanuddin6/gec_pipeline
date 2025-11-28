#!/usr/bin/env python3
"""
Count entries and occurrences in `data/urdu_word_dict.json`.

This file maps words to lists of occurrence metadata. The script reports:
 - number of unique words (keys)
 - total occurrences (sum of list lengths)
 - top-N words by occurrence count

Usage:
    python scripts/count_words_in_urdu_dict.py --input data/urdu_word_dict.json --top 30
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from typing import Any, Dict


def main() -> int:
    p = argparse.ArgumentParser(description="Count words/occurrences in urdu_word_dict.json")
    p.add_argument("--input", "-i", default="data/urdu_word_dict.json")
    p.add_argument("--top", "-t", type=int, default=20, help="Top N words by occurrences")
    args = p.parse_args()

    inp = args.input
    if not os.path.exists(inp):
        print(f"Input file not found: {inp}", file=sys.stderr)
        return 2

    with open(inp, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    if isinstance(data, dict):
        unique_words = len(data)
        occ_counter = Counter()
        total_occurrences = 0
        for word, occurrences in data.items():
            if isinstance(occurrences, list):
                count = len(occurrences)
            elif isinstance(occurrences, dict):
                # if value is a dict, maybe contains counts
                # try to detect a numeric 'count' field
                if "count" in occurrences and isinstance(occurrences["count"], int):
                    count = occurrences["count"]
                else:
                    # fallback: treat as single occurrence
                    count = 1
            elif isinstance(occurrences, int):
                count = occurrences
            else:
                count = 1

            occ_counter[word] = count
            total_occurrences += count

        print(f"Input: {inp}")
        print(f"Unique word entries (keys): {unique_words}")
        print(f"Total occurrences (sum of list lengths / counts): {total_occurrences}")
        print("")
        print(f"Top {args.top} words by occurrences:")
        for word, cnt in occ_counter.most_common(args.top):
            print(f"{word}\t{cnt}")
        return 0

    # If top-level is a list or other structure, try to handle gracefully
    elif isinstance(data, list):
        print(f"Top-level JSON is a list with {len(data)} items; inspecting item types...")
        # collect strings if present
        strings = [x for x in data if isinstance(x, str)]
        if strings:
            unique = len(set(strings))
            print(f"String items: {len(strings)} total, {unique} unique")
        else:
            print("No string items found; cannot compute word counts for this structure.")
        return 0

    else:
        print("Unrecognized JSON structure; expected dict mapping words->list of occurrences.")
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
