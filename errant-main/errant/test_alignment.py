from errant.alignment import Alignment
import numpy as np
import time
import urduhack

# Downloading models
# urduhack.download()

# Initializing the pipeline
nlp = urduhack.Pipeline()

orig_text = """
میں بلی ہوں۔    
"""

cor_text = """
میں ایک بلی ہوں۔
"""

doc1 = nlp(orig_text)
doc2 = nlp(cor_text)

for sentence1, sentence2 in zip(doc1.sentences, doc2.sentences):
    print(Alignment(sentence1, sentence2))