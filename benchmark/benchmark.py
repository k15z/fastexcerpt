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
import bz2
import enum
import json
import os
import shutil
import tempfile
from random import choices, random

import luigi
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from tqdm import tqdm

from fastexcerpt import FastExcerpt
from fastexcerpt.fastexcerpt import enumerate_excerpts

ROOT_DIR = "/Users/kevz/Downloads/"
WINDOW_SIZE = 5


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
        train = tempfile.NamedTemporaryFile("wt", delete=False)
        test = tempfile.NamedTemporaryFile("wt", delete=False)
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
        train = tempfile.NamedTemporaryFile("wt", delete=False)
        test = tempfile.NamedTemporaryFile("wt", delete=False)

        if self.method == "random":
            with open(self.input()["train"].path, "rt") as fin:
                for line in tqdm(fin, "Random"):
                    work = json.loads(line)
                    excerpts = enumerate_excerpts(work["content"], WINDOW_SIZE)
                    if len(excerpts) > self.num_excerpts:
                        excerpts = choices(excerpts, k=self.num_excerpts)
                    train.write(
                        json.dumps({"label": work["label"], "text": " ".join(excerpts)}) + "\n"
                    )

            with open(self.input()["test"].path, "rt") as fin:
                for line in tqdm(fin):
                    work = json.loads(line)
                    excerpts = enumerate_excerpts(work["content"], WINDOW_SIZE)
                    excerpts = choices(excerpts, k=self.num_excerpts)
                    test.write(
                        json.dumps({"label": work["label"], "text": " ".join(excerpts)}) + "\n"
                    )

        elif self.method == "fastexcerpt":

            def iterator():
                with open(self.input()["train"].path, "rt") as fin:
                    for line in fin:
                        work = json.loads(line)
                        yield work["content"], work["label"]

            fe = FastExcerpt(window_size=WINDOW_SIZE, verbose=True)
            fe.fit_iterator(tqdm(iterator(), "Fit"))

            for doc, label in tqdm(iterator(), "Predict"):
                train.write(
                    json.dumps(
                        {"label": label, "text": " ".join(fe.excerpts(doc, self.num_excerpts))}
                    )
                    + "\n"
                )

            with open(self.input()["test"].path, "rt") as fin:
                for line in tqdm(fin, "Test"):
                    work = json.loads(line)
                    test.write(
                        json.dumps(
                            {
                                "label": work["label"],
                                "text": " ".join(fe.excerpts(work["content"], self.num_excerpts)),
                            }
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
            result["test_size"] = len(X)
            idx = list(model.classes_).index(1)
            y_pred = model.predict_proba(X)[:, idx]
            result["test_auroc"] = roc_auc_score(y, y_pred)

        with self.output().open("w") as fout:
            json.dump(result, fout, indent=2)


class BenchmarkReport(luigi.Task):
    """Generate a summary report."""

    def requires(self):
        for target in [PredictionTarget.Category_FM, PredictionTarget.Rating_Explicit]:
            for num_excerpts in [1, 3, 5, 10]:
                yield from [
                    EvaluteExcerpts(target, "random", num_excerpts),
                    EvaluteExcerpts(target, "fastexcerpt", num_excerpts),
                ]

    def run(self):
        rows = []
        for result in self.input():
            rows.append(json.load(result.open("r")))
        pd.DataFrame(rows).to_markdown("benchmark/benchmark.md")
