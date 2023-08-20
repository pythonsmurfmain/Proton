import nltk
from nltk.stem.porter import PorterStemmer
import numpy as np
stemmer = PorterStemmer()


def tokenize(sen):
    return nltk.word_tokenize(sen)


def stem(word):
    return stemmer.stem(word.lower())


def bgfw(tsen,all_words):
    tsen = [stem(w) for w in tsen]
    bag = np.zeros(len(all_words), dtype=np.float32)
    for index,w in enumerate(all_words):
        if w in tsen:
            bag[index] = 1.0
    return bag
