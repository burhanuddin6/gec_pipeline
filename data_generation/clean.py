# Sample text with lines
corrfile, incorrfile = [open(file, 'r', encoding='utf-8').read().strip().split('\n') for file in ['data/wikiedits/correct.txt', 'data/wikiedits/incorrect.txt']]
incorrfile_copy = incorrfile.copy()

for line in incorrfile_copy:
    try:
        index = corrfile.index(line)
        incorrfile.pop(index)
        corrfile.pop(index)
    except:
        pass

# save back
with open('data/wikiedits/correct.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(corrfile))
with open('data/wikiedits/incorrect.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(incorrfile))