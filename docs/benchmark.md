On this page, we summarize our results from applying FastExcerpt to an internal dataset of works 
on Archive of Our Own consisting of ~150,000 works that are tagged with various labels by the 
authors of the works. The lengths of the works range from 1,000 words to over 500,000 words and 
we test different methods of extracting 5-sentence excerpts for predicting the tags.


## Predicting Category
First, we train a model to predict the "F/M" category. We test out two policies for selecting 
excerpts:

1. Choose excerpts randomly from the entire work.
2. Use FastExcerpt with the default configuration after fitting on the train set.

To evaluate the quality of the chosen snippets, we train another model to predict the tag using
only the chosen excerpts as the input and report the area under the ROC curve. We report the 
performance when using the top 1 excerpt, the top 3 excerpts, and so on.

| # Excerpts  | Random | FastExcerpt |
| ----------- | ------ | ----------- |
| 1           | 0.58   | 0.64        |
| 3           | 0.64   | 0.68        |
| 5           | 0.67   | 0.69        |
| 10          | 0.69   | 0.70        |

As the above table shows, assuming we only have the capacity to process 1 excerpt, using 
`FastExcerpt` to select it significantly outperforms random selection. However, as the 
number of excerpts increases, the gains decrease.

## Predicting Rating
Next, we run the same evaluation procedure but with the "Explicit" tag.

| # Excerpts  | Random | FastExcerpt |
| ----------- | ------ | ----------- |
| 1           | 0.558  | 0.749       |
| 3           | 0.593  | 0.748       |
| 5           | 0.604  | 0.750       |
| 10          | 0.643  | 0.746       |

On this tag, we observe an even larger gap in performance between random excerpts and fastexcerpt
than the F/M tag. We hypothesize that this is due the fact that the presence of a F/M relationship 
can be inferred from many different parts of the work whereas explicit content might be limited to 
a specific paragraph/chapter which FastExcerpt is able to find.
