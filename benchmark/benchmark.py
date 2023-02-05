"""Simple benchmarking for the excerpt models.

Num Excerpts: 1
  Random: 0.5796737683530137
  FastExcerpt: 0.641636210369364

Num Excerpts: 3
  Random: 0.6397643661794605
  FastExcerpt: 0.6802500333497639

Num Excerpts: 5
  Random: 0.6717437148703995
  FastExcerpt: 0.694652873898157

Num Excerpts: 10
  Random: 0.6882350548118742
  FastExcerpt: 0.70905571444655
"""
import bz2
import json
from random import choices

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from fastexcerpt import FastExcerpt
from fastexcerpt.fastexcerpt import enumerate_excerpts

# Extract some data for testing
docs, labels = [], []
with bz2.open("/Users/kevz/Downloads/ao3.jsonl.bz2", "rt") as fin:
    for line in fin:
        work = json.loads(line)
        if len(work["content"].split(" ")) <= 1000:
            continue
        docs.append(work["content"])
        if "Explicit" in work["tags"]["rating"]:
            labels.append(1)
        else:
            labels.append(0)
        if len(labels) == 10000:
            break
train_docs, test_docs, train_labels, test_labels = train_test_split(docs, labels)
print(np.mean(labels))

# Train an excerpt model
fe = FastExcerpt(verbose=True)
fe.fit(train_docs, train_labels, sampling_rate=None)


def evaluate(X_train, X_test, y_train, y_test):
    model = Pipeline(
        [
            ("vec", HashingVectorizer()),
            ("clf", LogisticRegression()),
        ]
    )
    model.fit(X_train, y_train)
    return roc_auc_score(y_test, model.predict(X_test))


for num_excerpts in [1, 3, 5, 10]:
    # Build a new training dataset for the downstream task
    train_excerpt_random = [
        " ".join(choices(enumerate_excerpts(doc, 5), k=num_excerpts)) for doc in train_docs
    ]
    train_excerpt_model = [" ".join(fe.excerpts(doc, num_excerpts)) for doc in train_docs]

    test_excerpt_random = [
        " ".join(choices(enumerate_excerpts(doc, 5), k=num_excerpts)) for doc in test_docs
    ]
    test_excerpt_model = [" ".join(fe.excerpts(doc, num_excerpts)) for doc in test_docs]

    # Compare the performance of models trained on random excerpts vs selected excerpts
    print("Num Excerpts:", num_excerpts)
    print(
        "  Random:", evaluate(train_excerpt_random, test_excerpt_random, train_labels, test_labels)
    )
    print(
        "  FastExcerpt:",
        evaluate(train_excerpt_model, test_excerpt_model, train_labels, test_labels),
    )
    print()
