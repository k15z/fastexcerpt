"""Luigi-based benchmark runner.

The main entrypoint is:

> luigi --module benchmark.benchmark BenchmarkReport

Inside BenchmarkReport, various default methods for generating excerpts 
are configured:

```
EvaluateExcerpts(config={
    target: "predict_category",
    method: "FastExcerpt",
    window_size: ...
})
```

And the final report summarizes the performance.
"""
import os
import bz2
import luigi
import shutil
import json
from random import random, choices
import tempfile
from tqdm import tqdm
from fastexcerpt import FastExcerpt
from fastexcerpt.fastexcerpt import enumerate_excerpts
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

ROOT_DIR = "/Users/kevz/Downloads/"


class TrainTestSplit(luigi.Task):

    target = "predict_category"

    def output(self):
        return {
            "train": luigi.LocalTarget(os.path.join(ROOT_DIR, f"data/{self.target}_train.jsonl")),
            "test": luigi.LocalTarget(os.path.join(ROOT_DIR, f"data/{self.target}_test.jsonl")),
        }

    def run(self):
        train = tempfile.NamedTemporaryFile("wt")
        test = tempfile.NamedTemporaryFile("wt")
        with bz2.open(os.path.join(ROOT_DIR, "ao3.jsonl.bz2"), "rt") as fin:
            for line in tqdm(fin):
                work = json.loads(line)
                if len(work["content"].split(" ")) <= 1000:
                    continue
                row = json.dumps(
                    {
                        "label": 1 if "F/M" in work["tags"]["category"] else 0,
                        "content": work["content"],
                    }
                )
                if random() < 0.8:
                    train.write(row + "\n")
                else:
                    test.write(row + "\n")
                if random() < 0.0001:
                    break
        train.flush()
        test.flush()
        shutil.move(train.name, self.output()["train"].path)
        shutil.move(test.name, self.output()["test"].path)


class ExtractExcerpts(luigi.Task):

    method = luigi.Parameter("random")
    num_excerpts = luigi.IntParameter(5)

    def output(self):
        return {
            "train": luigi.LocalTarget(
                os.path.join(ROOT_DIR, f"excerpts/{self.method}_{self.num_excerpts}_train.jsonl")
            ),
            "test": luigi.LocalTarget(
                os.path.join(ROOT_DIR, f"excerpts/{self.method}_{self.num_excerpts}_test.jsonl")
            ),
        }

    def requires(self):
        return TrainTestSplit()

    def run(self):
        train = tempfile.NamedTemporaryFile("wt")
        test = tempfile.NamedTemporaryFile("wt")

        if self.method == "random":
            with open(self.input()["train"].path, "rt") as fin:
                for line in tqdm(fin, "Random"):
                    work = json.loads(line)
                    excerpts = enumerate_excerpts(work["content"], 5)
                    if len(excerpts) > self.num_excerpts:
                        excerpts = choices(excerpts, k=self.num_excerpts)
                    train.write(
                        json.dumps({"label": work["label"], "text": " ".join(excerpts)}) + "\n"
                    )

            with open(self.input()["test"].path, "rt") as fin:
                for line in tqdm(fin):
                    work = json.loads(line)
                    excerpts = enumerate_excerpts(work["content"], 5)
                    excerpts = choices(excerpts, k=self.num_excerpts)
                    test.write(
                        json.dumps({"label": work["label"], "text": " ".join(excerpts)}) + "\n"
                    )

        elif self.method == "fastexcerpt":
            with open(self.input()["train"].path, "rt") as fin:
                docs, labels = [], []
                for line in tqdm(fin, "FastExcerpt"):
                    work = json.loads(line)
                    docs.append(work["content"])
                    labels.append(work["label"])
                fe = FastExcerpt(verbose=True)
                fe.fit(docs, labels)

                for doc, label in zip(docs, labels):
                    train.write(
                        json.dumps({"label": label, "text": " ".join(fe.excerpts(doc))}) + "\n"
                    )

            with open(self.input()["test"].path, "rt") as fin:
                for line in tqdm(fin):
                    work = json.loads(line)
                    test.write(
                        json.dumps(
                            {"label": work["label"], "text": " ".join(fe.excerpts(work["content"]))}
                        )
                        + "\n"
                    )

        train.flush()
        test.flush()
        shutil.move(train.name, self.output()["train"].path)
        shutil.move(test.name, self.output()["test"].path)


class EvaluteExcerpts(luigi.Task):

    method = luigi.Parameter("random")
    num_excerpts = luigi.IntParameter(5)

    def output(self):
        return luigi.LocalTarget(
            os.path.join(ROOT_DIR, f"results/{self.method}_{self.num_excerpts}.json")
        )

    def requires(self):
        return ExtractExcerpts(self.method, self.num_excerpts)

    def run(self):
        model = Pipeline(
            [
                ("vec", HashingVectorizer()),
                ("clf", LogisticRegression()),
            ]
        )
        result = {}

        with open(self.input()["train"].path, "rt") as fin:
            X, y = [], []
            for line in fin:
                work = json.loads(line)
                X.append(work["text"])
                y.append(work["label"])
            model.fit(X, y)

        with open(self.input()["test"].path, "rt") as fin:
            X, y = [], []
            for line in fin:
                work = json.loads(line)
                X.append(work["text"])
                y.append(work["label"])
            print(len(X))
            result["test_auroc"] = roc_auc_score(y, model.predict(X))

        with self.output().open("w") as fout:
            json.dump(result, fout, indent=2)


class BenchmarkReport(luigi.Task):
    def requires(self):
        return [
            EvaluteExcerpts("random", 1),
            EvaluteExcerpts("fastexcerpt", 1),
        ]
