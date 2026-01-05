#!/usr/bin/env python3
# compute_error_stats.py

import sys
import os
import json
import re
from collections import Counter, defaultdict

DEFAULT_PATH = 'data/gold_auto_plus_spell_tags.txt'

TAG_SPLIT_RE = re.compile(r'[,\s]+')  # split on whitespace or comma

def load_lines(path):
    with open(path, 'r', encoding='utf-8') as f:
        raw = [ln.rstrip('\n') for ln in f.readlines()]
    # keep lines that are not empty (empty means probably not processed)
    lines = [ln for ln in raw if ln is not None]  # preserve NONE lines
    return lines

def parse_tags(line):
    if line is None:
        return []
    s = line.strip()
    if s == "" or s.upper() == "NONE":
        return []
    tokens = [t.strip() for t in TAG_SPLIT_RE.split(s) if t.strip() and t.strip().upper() != "NONE"]
    return tokens

def percentage(part, whole):
    return 0.0 if whole == 0 else (part / whole) * 100.0

def main(path):
    if not os.path.exists(path):
        print(f"ERROR: file not found: {path}")
        return 1

    lines = load_lines(path)
    total_sentences = len([ln for ln in lines if ln.strip() != ""])  # count non-empty lines
    # If you want to count all lines including explicit 'NONE' lines, use len(lines)
    # We'll treat explicit 'NONE' as processed sentences (they are still lines), so:
    total_sentences = len(lines)

    per_sentence_tags = []
    tag_counter = Counter()
    errors_per_sentence_counter = Counter()

    for ln in lines:
        tags = parse_tags(ln)
        per_sentence_tags.append(tags)
        for t in tags:
            tag_counter[t] += 1
        errors_per_sentence_counter[len(tags)] += 1

    total_errors = sum(tag_counter.values())

    # Top 10 most common
    top10_most = tag_counter.most_common(10)

    # Top 10 least common (non-zero). Sort ascending by count then by tag name
    least_items = sorted(tag_counter.items(), key=lambda x: (x[1], x[0]))
    top10_least = least_items[:10]

    # Error-to-sentence dict with percentages
    error_to_sentence = {}
    for num_errors, sent_count in sorted(errors_per_sentence_counter.items()):
        pct = percentage(sent_count, total_sentences)
        error_to_sentence[num_errors] = {"sentence_count": sent_count, "percentage": pct}

    # Inflection errors: tags containing ':INFL' (case-insensitive)
    inflection_count = sum(cnt for tag, cnt in tag_counter.items() if ':INFL' in tag.upper())

    # Spelling errors: exact 'R:spell' (case-insensitive)
    spelling_count = sum(cnt for tag, cnt in tag_counter.items() if tag.lower() == 'r:spell')

    # Substitutions: R:*, excluding R:spell (case-insensitive)
    subs_count = sum(cnt for tag, cnt in tag_counter.items()
                     if tag.upper().startswith('R:') and tag.lower() != 'r:spell')

    # Insertions: U:*
    ins_count = sum(cnt for tag, cnt in tag_counter.items() if tag.upper().startswith('U:'))

    # Deletions: M:*
    del_count = sum(cnt for tag, cnt in tag_counter.items() if tag.upper().startswith('M:'))

    # Percentages relative to total_errors (avoid division by zero)
    stats = {
        "total_sentences": total_sentences,
        "total_errors": total_errors,
        "top10_most": [{"tag": t, "count": c, "percentage_of_errors": percentage(c, total_errors)} for t,c in top10_most],
        "top10_least": [{"tag": t, "count": c, "percentage_of_errors": percentage(c, total_errors)} for t,c in top10_least],
        "error_to_sentence": error_to_sentence,
        "inflection": {"count": inflection_count, "percentage_of_errors": percentage(inflection_count, total_errors)},
        "spelling": {"count": spelling_count, "percentage_of_errors": percentage(spelling_count, total_errors)},
        "substitution": {"count": subs_count, "percentage_of_errors": percentage(subs_count, total_errors)},
        "insertion": {"count": ins_count, "percentage_of_errors": percentage(ins_count, total_errors)},
        "deletion": {"count": del_count, "percentage_of_errors": percentage(del_count, total_errors)},
        "all_tag_counts": dict(tag_counter)
    }

    # Pretty-print summary to stdout
    print("=== SUMMARY ===")
    print(f"Total sentences processed: {total_sentences}")
    print(f"Total error tags (token count): {total_errors}")
    print()
    print("Top 10 most frequent error tags (tag, count, % of total errors):")
    for item in stats["top10_most"]:
        print(f"  {item['tag']}: {item['count']} ({item['percentage_of_errors']:.2f}%)")
    print()
    print("Top 10 least frequent error tags (tag, count, % of total errors):")
    for item in stats["top10_least"]:
        print(f"  {item['tag']}: {item['count']} ({item['percentage_of_errors']:.2f}%)")
    print()
    print("Error-count -> sentence-count (and percentage of sentences):")
    for num_errors, info in stats["error_to_sentence"].items():
        print(f"  {num_errors} errors: {info['sentence_count']} sentences ({info['percentage']:.2f}%)")
    print()
    print(f"Inflection errors (:INFL): {inflection_count} ({stats['inflection']['percentage_of_errors']:.2f}% of errors)")
    print(f"Spelling errors (R:spell): {spelling_count} ({stats['spelling']['percentage_of_errors']:.2f}% of errors)")
    print()
    print("By operation type (counts and % of errors):")
    print(f"  Substitutions (R:*, excluding R:spell): {subs_count} ({stats['substitution']['percentage_of_errors']:.2f}%)")
    print(f"  Insertions (U:*): {ins_count} ({stats['insertion']['percentage_of_errors']:.2f}%)")
    print(f"  Deletions (M:*): {del_count} ({stats['deletion']['percentage_of_errors']:.2f}%)")
    print()

    # Save stats to JSON file for later analysis
    outpath = 'data/gold_error_stats_summary.json'
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    with open(outpath, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"Wrote JSON summary -> {outpath}")

    return 0

if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PATH
    sys.exit(main(path))
