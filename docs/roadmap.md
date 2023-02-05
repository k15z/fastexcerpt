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
 - Implement support for generators / batch training
    - For the full FF dataset, we probably won't be able to hold everything
      in memory without reserving a high-powered AWS instance.
 - Enable iterative refinement
 - Implement benchmarks via Luigi for ease of running?
 - Finalize API design and prepare a release.

## v0.2 - Scalability

 - [Python] Start optimizing for speed - sliding window optimizations, etc.
 - Add speed to the benchmark report

## v0.3 - Functionality

  - Add support for other supervised losses
  - Add support for unsupervised losses
  - Publish pre-trained models for general categories (i.e. Humor/Drama) as well
    as a large unsupervised model trained on our entire dataset.
 - Add a basic cli
    - fastexcerpt train --dataset ao3.jsonl.bz2 --model model.bin
    - fastexcerpt excerpts --model model.bin --path_to_file example.txt
