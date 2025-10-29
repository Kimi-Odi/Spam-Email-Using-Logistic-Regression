#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Tuple

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

try:
    # Use stdlib for downloading to avoid extra deps
    from urllib.request import urlretrieve
except Exception:  # pragma: no cover
    urlretrieve = None


DATASET_URL_DEFAULT = (
    "https://raw.githubusercontent.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity/refs/heads/master/Chapter03/datasets/sms_spam_no_header.csv"
)


def ensure_parents(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def fetch_dataset(url: str, cache_path: Path) -> Path:
    """Download dataset to cache_path if not present. Returns local path.

    Gracefully handles no-network environments by using existing cache if available.
    """
    if cache_path.exists():
        return cache_path
    ensure_parents(cache_path)
    if urlretrieve is None:
        raise RuntimeError("urllib not available and cache file missing: cannot fetch dataset")
    try:
        urlretrieve(url, str(cache_path))
    except Exception as e:
        raise RuntimeError(
            f"Failed to download dataset from {url}. You can place the file manually at {cache_path}. Error: {e}"
        )
    return cache_path


def load_dataset(csv_path: Path) -> Tuple[pd.Series, pd.Series]:
    # Dataset expected format: label,text (no header)
    df = pd.read_csv(
        csv_path,
        header=None,
        names=["label", "text"],
        encoding="latin-1",
        dtype={0: str, 1: str},
    )
    df = df.dropna(subset=["label", "text"]).copy()
    df["text"] = df["text"].astype(str).str.strip()
    return df["text"], df["label"]


def train_and_evaluate(
    texts: pd.Series,
    labels: pd.Series,
    *,
    test_size: float,
    random_state: int,
    ngram_min: int,
    ngram_max: int,
    min_df: int,
):
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=test_size, random_state=random_state, stratify=labels
    )

    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(ngram_min, ngram_max),
        min_df=min_df,
    )
    X_train_vec = vectorizer.fit_transform(X_train)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train_vec, y_train)

    X_test_vec = vectorizer.transform(X_test)
    y_pred = clf.predict(X_test_vec)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision_weighted": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
    }

    report = classification_report(y_test, y_pred, digits=4)
    return vectorizer, clf, metrics, report


def persist_artifacts(output_dir: Path, vectorizer, model, pipeline):
    ensure_parents(output_dir / "dummy")
    joblib.dump(vectorizer, output_dir / "vectorizer.joblib")
    joblib.dump(model, output_dir / "model_lr.joblib")
    joblib.dump(pipeline, output_dir / "pipeline_lr.joblib")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Train baseline SMS spam classifier (Logistic Regression + TF-IDF)")
    parser.add_argument("--dataset-url", default=DATASET_URL_DEFAULT, help="CSV URL for SMS dataset")
    parser.add_argument("--cache-path", default=str(Path("data") / "sms_spam_no_header.csv"), help="Local cache path for dataset")
    parser.add_argument("--output-dir", default=str(Path("artifacts")), help="Directory to write artifacts and metrics")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test set fraction")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for reproducible split")
    parser.add_argument("--ngram-min", type=int, default=1, help="Minimum n-gram length")
    parser.add_argument("--ngram-max", type=int, default=2, help="Maximum n-gram length")
    parser.add_argument("--min-df", type=int, default=1, help="Minimum document frequency for TF-IDF")
    args = parser.parse_args(argv)

    cache_path = Path(args.cache_path)
    output_dir = Path(args.output_dir)

    try:
        local_csv = fetch_dataset(args.dataset_url, cache_path)
    except Exception as e:
        print(str(e), file=sys.stderr)
        if cache_path.exists():
            local_csv = cache_path
            print(f"Proceeding with existing cache at {local_csv}")
        else:
            return 2

    texts, labels = load_dataset(local_csv)

    vectorizer, model, metrics, report = train_and_evaluate(
        texts,
        labels,
        test_size=args.test_size,
        random_state=args.random_state,
        ngram_min=args.ngram_min,
        ngram_max=args.ngram_max,
        min_df=args.min_df,
    )

    pipeline = make_pipeline(vectorizer, model)
    persist_artifacts(output_dir, vectorizer, model, pipeline)

    # Write metrics and report
    ensure_parents(output_dir / "metrics.json")
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with open(output_dir / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    print("Training complete. Metrics:")
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

