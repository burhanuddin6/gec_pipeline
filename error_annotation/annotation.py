from errant.alignment import Alignment
import numpy as np
import time
import urduhack
from data_generation.generate_word_dict import generate_word_dict

# Downloading models
# urduhack.download()

# Initializing the pipeline
nlp = urduhack.Pipeline()

orig_text = """
میں میں بلی ہوں۔    
"""

cor_text = """
میں ایک بلی ہوں۔
"""

doc1 = nlp(orig_text)
doc2 = nlp(cor_text)

for sentence1, sentence2 in zip(doc1.sentences, doc2.sentences):
    print(Alignment(sentence1, sentence2))


algorithm(incorrect, correct):
get pos tags for both
get alignment
compare the Alignment:
    if substitution, then:
        store the difference in pos tags, feats of the substitute generate_word_dict
    if deletion, then:
        store the pos tags, feats of that words and words on left and right
        