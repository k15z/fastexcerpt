This project is still in an exploratory stage. Here is my current plan:

## v0.1 - MVP

 - Remove subword dependencies + model.
 - Consider switching to PyTorch backend.
   - Add support for other supervised losses
     - multi-class
     - multi-label
 - Code cleanup :)
 - Finalize API design and prepare a release.

## v0.2 - Scalability

 - [Python] Start optimizing for speed - sliding window optimizations, etc.
 - [Go/Rust] Move hashing vectorizer to Go/Rust?
 - Add speed to the benchmark report

## v0.3 - Functionality

  - Enable iterative refinement
  - Add support for unsupervised losses (?)
  - Publish pre-trained models for general categories (i.e. Humor/Drama) as well
    as a large unsupervised model trained on our entire dataset.
  - Add a basic cli
    - fastexcerpt train --dataset ao3.jsonl.bz2 --model model.bin
    - fastexcerpt excerpts --model model.bin --path_to_file example.txt
