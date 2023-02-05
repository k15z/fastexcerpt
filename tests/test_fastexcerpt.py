"""Test module for fastexcerpt."""

import nltk
from fastexcerpt import __author__, __email__, __version__, FastExcerpt, SubwordFastExcerpt

nltk.download("punkt")


def test_project_info():
    """Test __author__ value."""
    assert __author__ == "Kevin Alex Zhang"
    assert __email__ == "hello@kevz.dev"
    assert __version__ == "0.0.0"


def test_fastexcerpt():
    docs = [
        "This is great. Lots of short sentences. Let's go!",
        "This is horrible. Lots of short sentences. Let's go!",
    ]
    fast_excerpt = FastExcerpt(window_size=1, hash_size=10)
    fast_excerpt.fit(docs, labels=[1, 0])
    assert fast_excerpt.excerpts("Hello! This is great. What do you think?", 1) == [
        "This is great."
    ]

def test_subword_fastexcerpt():
    docs = [
        "This is great. Lots of short sentences. Let's go!",
        "This is horrible. Lots of short sentences. Let's go!",
    ]
    fast_excerpt = SubwordFastExcerpt(window_size=1)
    fast_excerpt.fit(docs, labels=[1, 0])
    assert fast_excerpt.excerpts("Hello! This is great. What do you think?", 1) == [
        "This is great."
    ]
