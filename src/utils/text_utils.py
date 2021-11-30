import re
import string
from collections import Counter
import pandas as pd
import nltk
from nltk.corpus import stopwords


def remove_punctuations(text):
    translator = str.maketrans("", "", string.punctuation)
    return text.translate(translator)


def remove_stopwords(text):
    all_stopwords = set(stopwords.words("english"))
    filtered_words = [word.lower() for word in text.split() if word.lower() not in all_stopwords]
    return " ".join(filtered_words)


def counter_word(text_column):
    count = Counter()
    for text in text_column.values:
        for word in text.split():
            count[word] += 1
    return count


def build_word_set(text_column: pd.Series):
    word_set = set()
    for sentence in text_column:
        for word in sentence.split():
            if word not in word_set:
                word_set.add(word)
    return word_set


def decode_sequence(reverse_word_index: dict, sequence):
    return " ".join([reverse_word_index.get(idx, "?") for idx in sequence])