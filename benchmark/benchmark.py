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
import pandas as pd
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
import enum

ROOT_DIR = "/Users/kevz/Downloads/"


class PredictionTarget(enum.Enum):
    Category_FM = "category_FM"
    Rating_Explicit = "rating_explicit"


class TrainTestSplit(luigi.Task):
    """Extract labels and split into train/test file."""

    target = luigi.EnumParameter(enum=PredictionTarget)

    def output(self):
        return {
            "train": luigi.LocalTarget(
                os.path.join(ROOT_DIR, f"data/{self.target.value}_train.jsonl")
            ),
            "test": luigi.LocalTarget(
                os.path.join(ROOT_DIR, f"data/{self.target.value}_test.jsonl")
            ),
        }

    def run(self):
        train = tempfile.NamedTemporaryFile("wt")
        test = tempfile.NamedTemporaryFile("wt")
        with bz2.open(os.path.join(ROOT_DIR, "ao3.jsonl.bz2"), "rt") as fin:
            for line in tqdm(fin):
                work = json.loads(line)
                if len(work["content"].split(" ")) <= 1000:
                    continue

                if self.target == PredictionTarget.Category_FM:
                    label = 1 if "F/M" in work["tags"]["category"] else 0
                elif self.target == PredictionTarget.Rating_Explicit:
                    label = 1 if "Explicit" in work["tags"]["rating"] else 0

                row = json.dumps(
                    {
                        "label": label,
                        "content": work["content"],
                    }
                )

                if random() < 0.8:
                    train.write(row + "\n")
                else:
                    test.write(row + "\n")
                if random() < 0.001:
                    break
        train.flush()
        test.flush()
        shutil.move(train.name, self.output()["train"].path)
        shutil.move(test.name, self.output()["test"].path)


class ExtractExcerpts(luigi.Task):
    """Extract excerpts with specified method."""

    target = luigi.EnumParameter(enum=PredictionTarget)
    method = luigi.Parameter("random")
    num_excerpts = luigi.IntParameter(5)

    def output(self):
        return {
            "train": luigi.LocalTarget(
                os.path.join(
                    ROOT_DIR,
                    f"excerpts/{self.target.value}_{self.method}_{self.num_excerpts}_train.jsonl",
                )
            ),
            "test": luigi.LocalTarget(
                os.path.join(
                    ROOT_DIR,
                    f"excerpts/{self.target.value}_{self.method}_{self.num_excerpts}_test.jsonl",
                )
            ),
        }

    def requires(self):
        return TrainTestSplit(self.target)

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
    """Evaluate excerpts generated using the specified method."""

    target = luigi.EnumParameter(enum=PredictionTarget)
    method = luigi.Parameter("random")
    num_excerpts = luigi.IntParameter(5)

    def output(self):
        return luigi.LocalTarget(
            os.path.join(
                ROOT_DIR, f"results/{self.target.value}_{self.method}_{self.num_excerpts}.json"
            )
        )

    def requires(self):
        return ExtractExcerpts(self.target, self.method, self.num_excerpts)

    def run(self):
        model = Pipeline(
            [
                ("vec", HashingVectorizer()),
                ("clf", LogisticRegression()),
            ]
        )
        result = {
            "target": self.target.value,
            "method": self.method,
            "k": self.num_excerpts,
        }

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
    """Generate a summary report."""

    def requires(self):
        return [
            EvaluteExcerpts(PredictionTarget.Category_FM, "random", 1),
            EvaluteExcerpts(PredictionTarget.Category_FM, "fastexcerpt", 1),
            EvaluteExcerpts(PredictionTarget.Category_FM, "random", 3),
            EvaluteExcerpts(PredictionTarget.Category_FM, "fastexcerpt", 3),
            EvaluteExcerpts(PredictionTarget.Rating_Explicit, "random", 1),
            EvaluteExcerpts(PredictionTarget.Rating_Explicit, "fastexcerpt", 1),
            EvaluteExcerpts(PredictionTarget.Rating_Explicit, "random", 3),
            EvaluteExcerpts(PredictionTarget.Rating_Explicit, "fastexcerpt", 3),
        ]

    def run(self):
        rows = []
        for result in self.input():
            rows.append(json.load(result.open("r")))
        pd.DataFrame(rows).to_markdown("benchmark/benchmark.md")
