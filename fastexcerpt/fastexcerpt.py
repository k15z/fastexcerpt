"""Main module."""
import typing
from collections import Counter

import numpy as np
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def enumerate_excerpts(doc: str, window_size: int) -> typing.List[str]:
    excerpts = []
    sentences = sent_tokenize(doc)
    for i in range(window_size, len(sentences) + 1):
        excerpts.append(" ".join(sentences[i - window_size : i]))
    return excerpts


class FastExcerpt:
    def __init__(self, window_size: int = 3, hash_size: int = 10000):
        self.hash_size = hash_size
        self.window_size = window_size

    def fit(self, docs: typing.List[str], labels: typing.List[int]) -> None:
        X, y = [], []
        for doc, label in zip(docs, labels):
            for excerpt in enumerate_excerpts(doc, self.window_size):
                X.append(excerpt)
                y.append(label)

        self.model = Pipeline([("vec", HashingVectorizer()), ("clf", LogisticRegression())])
        self.model.fit(X, y)
        self.mu = np.mean(y)

    def excerpts(self, doc: str, num_excerpts: int = 1) -> typing.List[str]:
        excerpt_to_score: typing.Counter = Counter()
        for excerpt in enumerate_excerpts(doc, self.window_size):
            excerpt_to_score[excerpt] = abs(self.model.predict_proba([excerpt])[0, 0] - self.mu)

        excerpts = []
        for excerpt, _ in excerpt_to_score.most_common(num_excerpts):
            excerpts.append(excerpt)
        return excerpts
