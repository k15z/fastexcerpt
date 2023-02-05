This project is still in an exploratory stage. Here is my current thinking on 
the roadmap:

## v0.1 - MVP

 - Implement different ideas for excerpt extraction.
    - Scoring:
        - direct = predict the label, score = "confidence"
        - residual = predict the error, score = 1/error
    - Models:
        - Hashing vectorizer -> logistic regression.
        - Subword embeddings -> pooling -> logistic regression.
    - We'll only consider models that we can implement efficiently via some kind of sliding
      window trick later, but in the initial implementation, we'll brute-force it.
 - Add benchmark results focused on quality.
    - num excerpts vs accuracy
    - hypothesis: for our model, we can achieve same accuracy with fewer
      excerpts than random. furthermore, we asymptotically approach the 
      accuracy that can be achieved by looking at the full work
 - Add different selection methods?
     - greedy?
     - greedy non-overlapping?
     - maximal score via dynamic programming?
 - [Python] Start optimizing for speed - sliding window optimizations, etc.
 - Finalize API design and prepare a release.

## v0.2 - Scalability

 - Implement support for generators / batch training
    - For the full FF dataset, we probably won't be able to hold everything
      in memory without reserving a high-powered AWS instance.
 - Add speed to the benchmark report

## v1.0 - Usability

 - [C++/Rust] Consider rewriting some computationally intensive parts and 
   adding Python bindings.
 - Publish pre-trained models for general categories (i.e. Humor/Drama) as well
   as a large unsupervised model trained on our entire dataset.
