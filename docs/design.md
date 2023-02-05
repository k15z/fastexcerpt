The motivation for this project was the need for a lightning fast library for extracting meaningful
excerpts of text from long-form content. An example where this library is used in production is 
[AO3 Disco](https://ao3-disco.app), a fanfiction recommendation service which uses this library to 
extract excerpts for relevance analysis.

As an example, consider predicting whether a novel contains "time travel" as a plot device:

 - If you randomly select a few excerpts to analyze, it's quite possible you'll miss the relevant 
   passage in the text.
 - If you pass the entire work through a classic model (BoW, etc.), it's possible that you'll be 
   able to classify it, but the precision/recall might not be great.
 - If you pass the entire work through a deep learning model (transformers, etc.), it might be 
   able to achieve good results if it's given enough data, but would be too slow for many types 
   of applications.

The goal of `FastExcerpt` is to combine the strengths of these approaches by providing different 
methods for extracting excerpts that are relevant for downstream tasks.

## Proposal
Construct a FastExcerpt obejct:

```python
fe = FastExcerpt(
    window_size=3, # Look for 3-sentence long excerpts.
    max_hash_size=10000, # Hash size used for the ranking model.
)

# Extract 1 excerpt which is 3 sentences long.
excerpts = fe.excerpts("... a long document ...", N=1)
```

### Supervised
The standard use case is when you have a downstream classification task. You can simply past
your data in as two lists of strings.

```python
docs = [
    "This is a long document. It can be over 100,000 words long!",
    ... # more documents
]
labels = [
    "Humor",
    ...
]
fe.fit(docs, labels)
```

In this case, the model will learn to select excerpts that are maximize the performance of a 
simple classifier on the specified classification task.

Consider an example where you are predicting whether the work should be rated "Explicit" - in 
this case, the model should learn to select excerpts of text that contain explicit language.

### Unsupervised
If your data does not have labels, you can still use FastExcerpt.

```python
fe.fit(docs)
```

In this case, the model will learn to select excerpts that can be used to tell the works in the
dataset apart.

Consider an example where you have two documents: Harry Potter and Game of Thrones. The selected 
excerpts would be passages where - if you just read that one passage, it would be clear which work 
it came from.
