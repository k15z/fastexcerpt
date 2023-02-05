"""Main module."""
import typing
from collections import Counter

import numpy as np
from bpemb import BPEmb
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from tqdm import tqdm


def enumerate_excerpts(doc: str, window_size: int) -> typing.List[str]:
    excerpts = []
    sentences = sent_tokenize(doc)
    for i in range(window_size, len(sentences) + 1):
        excerpts.append(" ".join(sentences[i - window_size : i]))
    return excerpts


class FastExcerpt:
    def __init__(self, window_size: int = 5, hash_size: int = 10000, verbose: bool = False):
        self.verbose = verbose
        self.hash_size = hash_size
        self.window_size = window_size

    def fit(self, docs: typing.List[str], labels: typing.List[int]) -> None:
        X, y = [], []
        iterator = zip(docs, tqdm(labels)) if self.verbose else zip(docs, labels)
        for doc, label in iterator:
            for excerpt in enumerate_excerpts(doc, self.window_size):
                X.append(excerpt)
                y.append(label)

        self.model = Pipeline(
            [
                ("vec", HashingVectorizer(n_features=self.hash_size)),
                ("clf", LogisticRegression(verbose=1 if self.verbose else 0)),
            ]
        )
        self.model.fit(X, y)
        self.mu = np.mean(y)
        print(roc_auc_score(y, self.model.predict(X)))

    def excerpts(self, doc: str, num_excerpts: int = 1) -> typing.List[str]:
        excerpts = enumerate_excerpts(doc, self.window_size)
        y_pred = self.model.predict_proba(excerpts)
        excerpt_to_score: typing.Counter = Counter({k: np.max(v) for k, v in zip(excerpts, y_pred)})

        excerpts = []
        for excerpt, _ in excerpt_to_score.most_common(num_excerpts):
            excerpts.append(excerpt)
        return excerpts


class SubwordFastExcerpt:
    def __init__(self, window_size: int = 5, verbose: bool = False):
        self.verbose = verbose
        self.window_size = window_size
        self.encoder = BPEmb(lang="en", dim=300)

    def fit(self, docs: typing.List[str], labels: typing.List[int]) -> None:
        X, y = [], []
        iterator = zip(docs, tqdm(labels)) if self.verbose else zip(docs, labels)
        for doc, label in iterator:
            for excerpt in enumerate_excerpts(doc, self.window_size):
                X.append(self.encoder.embed(excerpt).mean(axis=0))
                y.append(label)
        self.mu = np.mean(y)

        self.model = LogisticRegression(verbose=1 if self.verbose else 0)
        self.model.fit(X, y)
        print(roc_auc_score(y, self.model.predict(X)))

    def excerpts(self, doc: str, num_excerpts: int = 1) -> typing.List[str]:
        excerpts = enumerate_excerpts(doc, self.window_size)
        embeddings = [self.encoder.embed(x).mean(axis=0) for x in excerpts]
        y_pred = self.model.predict_proba(embeddings)
        excerpt_to_score: typing.Counter = Counter({k: np.max(v) for k, v in zip(excerpts, y_pred)})

        excerpts = []
        for excerpt, _ in excerpt_to_score.most_common(num_excerpts):
            excerpts.append(excerpt)
        return excerpts
