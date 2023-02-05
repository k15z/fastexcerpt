# FastExcerpt

Fast extraction of relevant excerpts for long-form text.

## Motivation
The motivation for this project was the need for a lightning fast library for extracting meaningful
excerpts from long-form text. This library is used by [AO3 Disco](https://ao3-disco.app), a 
fanfiction recommendation service which uses this library to extract excerpts for further analysis.

As an example, consider predicting whether a novel contains "time travel" as a plot device:

 - If you randomly select a few excerpts to analyze, it's quite possible you'll miss the relevant 
   passage in the text.
 - If you pass the entire work through a classic model (BoW, etc.), it's possible that you'll be 
   able to classify it, but the precision/recall might not be great.
 - If you pass the entire work through a deep learning model (transformers, etc.), it might be 
   able to achieve good results if it's given enough data, but would be too slow for many types 
   of applications.

The goal of `FastExcerpt` is to combine the strengths of these approaches by using fast models to
efficiently extract relevant excerpts for further analysis.

## Usage
Construct a FastExcerpt object:

```python
fe = FastExcerpt(
    window_size=3, # Look for 3-sentence long excerpts.
    max_hash_size=10000, # Hash size used for the ranking model.
)

# Extract 1 excerpt which is 3 sentences long.
excerpts = fe.excerpts("... a long document ...", k=1)
```

The standard use case is when you have a downstream binary classification task. You can simply pass
your data in as two lists:

```python
docs = [
    "This is a long document. It can be over 100,000 words long!",
    ... # more documents
]
labels = [
    1,
    ...
]
fe.fit(docs, labels)
```

In this case, the model will learn to select excerpts that are maximize the performance of a 
simple classifier on the specified classification task.

Consider an example where you are predicting whether the work should be rated "Explicit" - in 
this case, the model should learn to select excerpts of text that contain explicit language.
