import urduhack
from urduhack.models.lemmatizer.lemmatizer import lemma_lookup

class UrduStemmer:
    def __init__(self):
        # Downloading models
        urduhack.download()

        # Initializing the pipeline
        nlp = urduhack.Pipeline()
        
    def stem(self, word):
        return lemma_lookup(word)[0][1]