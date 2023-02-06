"""Test module for fastexcerpt."""

import nltk

from fastexcerpt import FastExcerpt

nltk.download("punkt")


def test_fastexcerpt():
    docs = [
        "This is great. Lots of short sentences. Let's go!",
        "This is horrible. Lots of short sentences. Let's go!",
    ]
    fast_excerpt = FastExcerpt(window_size=1, hash_size=10)
    fast_excerpt.fit(docs, labels=[1, 0], sampling_rate=None)
    assert fast_excerpt.excerpts("Hello! This is great. What do you think?", 1) == [
        "This is great."
    ]


def test_fastexcerpt_with_sampling():
    docs = [
        "This is great. Lots of short sentences. Let's go!",
        "This is horrible. Lots of short sentences. Let's go!",
    ]
    fast_excerpt = FastExcerpt(window_size=1, hash_size=10)
    fast_excerpt.fit(docs, labels=[1, 0], sampling_rate=1.0)
    assert fast_excerpt.excerpts("Hello! This is great. What do you think?", 1) == [
        "This is great."
    ]


def test_fastexcerpt_generator():
    docs = [
        "This is great. Lots of short sentences. Let's go!",
        "This is horrible. Lots of short sentences. Let's go!",
    ]
    labels = [1, 9]

    def iterator():
        for doc, label in zip(docs, labels):
            yield doc, label

    fast_excerpt = FastExcerpt(window_size=1, hash_size=10)
    fast_excerpt.fit_iterator(iterator())
    assert fast_excerpt.excerpts("Hello! This is great. What do you think?", 1) == [
        "This is great."
    ]


def test_fastexcerpt_generator_samplling():
    docs = [
        "This is great. Lots of short sentences. Let's go!",
        "This is horrible. Lots of short sentences. Let's go!",
    ]
    labels = [1, 9]

    def iterator():
        for doc, label in zip(docs, labels):
            yield doc, label

    fast_excerpt = FastExcerpt(window_size=1, hash_size=10)
    fast_excerpt.fit_iterator(iterator(), sampling_rate=1.0)
    assert fast_excerpt.excerpts("Hello! This is great. What do you think?", 1) == [
        "This is great."
    ]
