#!/usr/bin/env python3
import json
from pathlib import Path
from typing import Optional, Tuple, List

import pandas as pd
import streamlit as st
import altair as alt
import joblib
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

try:
    from urllib.request import urlretrieve
except Exception:  # pragma: no cover
    urlretrieve = None


DEFAULT_DATASET_URL = (
    "https://raw.githubusercontent.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity/refs/heads/master/Chapter03/datasets/sms_spam_no_header.csv"
)
DEFAULT_ARTIFACTS_DIR = Path("artifacts")


@st.cache_data(show_spinner=False)
def fetch_dataset(url: str, cache_path: Path) -> Path:
    if cache_path.exists():
        return cache_path
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if urlretrieve is None:
        raise RuntimeError("Cannot download dataset (urllib unavailable). Place CSV at: " + str(cache_path))
    urlretrieve(url, str(cache_path))
    return cache_path


@st.cache_data(show_spinner=False)
def load_dataset(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        csv_path,
        header=None,
        names=["label", "text"],
        encoding="latin-1",
        dtype={0: str, 1: str},
    )
    df = df.dropna(subset=["label", "text"]).copy()
    df["text"] = df["text"].astype(str).str.strip()
    return df


def try_load_pipeline(artifacts_dir: Path) -> Optional[Pipeline]:
    pipe_path = artifacts_dir / "pipeline_lr.joblib"
    vec_path = artifacts_dir / "vectorizer.joblib"
    model_path = artifacts_dir / "model_lr.joblib"
    # Try full pipeline first, then fall back to vectorizer+model even if pipeline load fails
    if pipe_path.exists():
        try:
            return joblib.load(pipe_path)
        except Exception as e:
            st.warning(f"Failed to load pipeline_lr.joblib: {e}. Trying vectorizer+model fallback…")
    if vec_path.exists() and model_path.exists():
        try:
            vec = joblib.load(vec_path)
            model = joblib.load(model_path)
            return make_pipeline(vec, model)
        except Exception as e:
            st.warning(f"Failed to load vectorizer/model: {e}")
    return None


def get_vectorizer_and_model(pipeline: Pipeline):
    # make_pipeline names become lowercase class names
    # e.g., 'tfidfvectorizer' and 'logisticregression'
    steps = dict(pipeline.named_steps)
    vectorizer = steps.get("tfidfvectorizer")
    model = steps.get("logisticregression")
    return vectorizer, model


def compute_top_tokens(vectorizer, model, top_n: int = 20) -> pd.DataFrame:
    feature_names = vectorizer.get_feature_names_out()
    classes = list(model.classes_)
    coefs = model.coef_

    rows = []
    if coefs.shape[0] == 1:  # binary: positive class is classes[1]
        coef = coefs[0]
        # Top tokens for positive class
        top_pos_idx = coef.argsort()[::-1][:top_n]
        for i in top_pos_idx:
            rows.append({"class": classes[1], "token": feature_names[i], "weight": float(coef[i])})
        # Top tokens for negative class (most negative weights)
        top_neg_idx = coef.argsort()[:top_n]
        for i in top_neg_idx:
            rows.append({"class": classes[0], "token": feature_names[i], "weight": float(-coef[i])})
    else:
        # One-vs-rest per class
        for ci, cls in enumerate(classes):
            coef = coefs[ci]
            top_idx = coef.argsort()[::-1][:top_n]
            for i in top_idx:
                rows.append({"class": cls, "token": feature_names[i], "weight": float(coef[i])})
    return pd.DataFrame(rows)


def compute_token_frequencies(df: pd.DataFrame, vectorizer, top_n: int = 20) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute top token frequencies for ham and spam using CountVectorizer
    aligned to the trained TF‑IDF vectorizer's vocabulary and preprocessing.
    """
    vocab = getattr(vectorizer, "vocabulary_", None)
    ngram_range = getattr(vectorizer, "ngram_range", (1, 1))
    strip_accents = getattr(vectorizer, "strip_accents", None)
    lowercase = getattr(vectorizer, "lowercase", True)

    cv = CountVectorizer(
        vocabulary=vocab,
        ngram_range=ngram_range,
        strip_accents=strip_accents,
        lowercase=lowercase,
    )
    feature_names = cv.get_feature_names_out()

    df_ham = df[df["label"] == "ham"]["text"]
    df_spam = df[df["label"] == "spam"]["text"]

    X_ham = cv.transform(df_ham)
    X_spam = cv.transform(df_spam)

    counts_ham = X_ham.sum(axis=0).A1
    counts_spam = X_spam.sum(axis=0).A1

    top_h_idx = counts_ham.argsort()[::-1][:top_n]
    top_s_idx = counts_spam.argsort()[::-1][:top_n]

    df_top_ham = pd.DataFrame({
        "token": feature_names[top_h_idx],
        "count": counts_ham[top_h_idx],
        "class": "ham",
    })
    df_top_spam = pd.DataFrame({
        "token": feature_names[top_s_idx],
        "count": counts_spam[top_s_idx],
        "class": "spam",
    })
    return df_top_ham, df_top_spam


def plot_class_distribution(df: pd.DataFrame):
    counts = df["label"].value_counts().reset_index()
    counts.columns = ["label", "count"]
    chart = (
        alt.Chart(counts)
        .mark_bar()
        .encode(x=alt.X("label:N", sort="-y"), y="count:Q", tooltip=["label", "count"])
    )
    st.altair_chart(chart, use_container_width=True)


def plot_confusion_matrix(y_true: List[str], y_pred: List[str]):
    cm = confusion_matrix(y_true, y_pred, labels=sorted(list(set(y_true) | set(y_pred))))
    labels = sorted(list(set(y_true) | set(y_pred)))
    df_cm = pd.DataFrame(cm, index=labels, columns=labels)
    df_melt = df_cm.reset_index().melt(id_vars="index")
    df_melt.columns = ["Actual", "Predicted", "Count"]
    chart = (
        alt.Chart(df_melt)
        .mark_rect()
        .encode(
            x=alt.X("Predicted:N"),
            y=alt.Y("Actual:N"),
            color=alt.Color("Count:Q", scale=alt.Scale(scheme="blues")),
            tooltip=["Actual", "Predicted", "Count"],
        )
    )
    st.altair_chart(chart, use_container_width=True)


def plot_probability_hist(probas: pd.Series, positive_label: str):
    df = pd.DataFrame({"prob": probas})
    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(x=alt.X("prob:Q", bin=alt.Bin(maxbins=30), title=f"P({positive_label})"), y="count()")
    )
    st.altair_chart(chart, use_container_width=True)


def main():
    st.set_page_config(page_title="SMS Spam Classifier — Visualization", layout="wide")
    st.title("SMS Spam Classifier — Visualization")

    # Sidebar configuration
    st.sidebar.header("Configuration")
    artifacts_dir = Path(st.sidebar.text_input("Artifacts directory", str(DEFAULT_ARTIFACTS_DIR)))
    dataset_url = st.sidebar.text_input("Dataset URL", DEFAULT_DATASET_URL)
    cache_path = Path(st.sidebar.text_input("Dataset cache path", str(Path("data") / "sms_spam_no_header.csv")))
    test_size = st.sidebar.slider("Test size", 0.1, 0.5, 0.2, 0.05)
    random_state = st.sidebar.number_input("Random seed", value=42, step=1)

    # Load artifacts
    pipeline = try_load_pipeline(artifacts_dir)
    metrics_path = artifacts_dir / "metrics.json"
    report_path = artifacts_dir / "classification_report.txt"

    if pipeline is None:
        st.warning(
            "Trained artifacts not loaded. Ensure the sidebar 'Artifacts directory' points to the folder containing the files above (run training if needed)."
        )
    else:
        st.success("Loaded trained artifacts.")

    # Data Overview
    st.header("1. Data Overview")
    try:
        local_csv = fetch_dataset(dataset_url, cache_path)
        df = load_dataset(local_csv)
        c1, c2, c3 = st.columns([1, 1, 2])
        with c1:
            st.metric("Rows", len(df))
        with c2:
            st.metric("Classes", df["label"].nunique())
        with c3:
            st.write("Class distribution")
            plot_class_distribution(df)
        st.write("Sample rows")
        st.dataframe(df.sample(min(10, len(df))), use_container_width=True)
    except Exception as e:
        st.error(f"Failed to load dataset: {e}")
        df = None

    # Top Tokens by Class
    st.header("2. Top Tokens by Class")
    if pipeline is None:
        st.info("Load artifacts to compute top tokens.")
    else:
        vectorizer, model = get_vectorizer_and_model(pipeline)
        if vectorizer is None or model is None or not hasattr(model, "coef_"):
            st.warning("Could not access vectorizer/model for token importance.")
        else:
            top_n = st.slider("Top N tokens", 5, 50, 20, 1)
            top_df = compute_top_tokens(vectorizer, model, top_n=top_n)
            st.dataframe(top_df, use_container_width=True)
            # Simple bar chart for one class at a time
            classes = sorted(top_df["class"].unique().tolist())
            sel = st.selectbox("Show chart for class", classes)
            chart = (
                alt.Chart(top_df[top_df["class"] == sel])
                .mark_bar()
                .encode(
                    x=alt.X("weight:Q", title="weight"),
                    y=alt.Y("token:N", sort="-x"),
                    tooltip=["token", "weight"],
                )
            )
            st.altair_chart(chart, use_container_width=True)

            # Frequency plots by class (ham, spam)
            if df is not None:
                st.subheader("Token Frequency by Class")
                try:
                    freq_ham, freq_spam = compute_token_frequencies(df, vectorizer, top_n=top_n)
                    c1, c2 = st.columns(2)
                    with c1:
                        st.caption("Ham: Top token counts")
                        ch = (
                            alt.Chart(freq_ham)
                            .mark_bar()
                            .encode(x=alt.X("count:Q"), y=alt.Y("token:N", sort="-x"), tooltip=["token", "count"])
                        )
                        st.altair_chart(ch, use_container_width=True)
                    with c2:
                        st.caption("Spam: Top token counts")
                        cs = (
                            alt.Chart(freq_spam)
                            .mark_bar()
                            .encode(x=alt.X("count:Q"), y=alt.Y("token:N", sort="-x"), tooltip=["token", "count"])
                        )
                        st.altair_chart(cs, use_container_width=True)
                except Exception as e:
                    st.info(f"Could not compute token frequencies: {e}")

    # Model Performance
    st.header("3. Model Performance")
    if metrics_path.exists():
        try:
            metrics = json.loads(metrics_path.read_text())
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Accuracy", f"{metrics.get('accuracy', 0):.4f}")
            c2.metric("Precision (weighted)", f"{metrics.get('precision_weighted', 0):.4f}")
            c3.metric("Recall (weighted)", f"{metrics.get('recall_weighted', 0):.4f}")
            c4.metric("F1 (weighted)", f"{metrics.get('f1_weighted', 0):.4f}")
        except Exception as e:
            st.warning(f"Failed to read metrics.json: {e}")
    else:
        st.info("metrics.json not found; will recompute visuals using the dataset.")

    if report_path.exists():
        try:
            with st.expander("Classification Report"):
                st.code(report_path.read_text())
        except Exception as e:
            st.warning(f"Failed to read classification_report.txt: {e}")

    # Recompute visuals (confusion matrix and probability histogram) using the dataset
    if pipeline is not None and df is not None:
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                df["text"], df["label"], test_size=float(test_size), random_state=int(random_state), stratify=df["label"]
            )
            y_pred = pipeline.predict(X_test)
            st.subheader("Confusion Matrix")
            plot_confusion_matrix(list(y_test), list(y_pred))

            # Probability histogram (binary expected)
            st.subheader("Probability Histogram")
            proba = None
            if hasattr(pipeline, "predict_proba"):
                try:
                    proba_arr = pipeline.predict_proba(X_test)
                    # Choose positive label heuristically as the lexicographically greater of classes if binary
                    model = get_vectorizer_and_model(pipeline)[1]
                    if model is not None and hasattr(model, "classes_") and len(model.classes_) >= 2:
                        classes = list(model.classes_)
                        pos_index = classes.index(sorted(classes)[-1])
                    else:
                        pos_index = 1 if proba_arr.shape[1] > 1 else 0
                    proba = pd.Series(proba_arr[:, pos_index])
                    plot_probability_hist(proba, positive_label=str(sorted(set(df["label"]))[-1]))
                except Exception:
                    st.info("Model does not provide probability outputs for histogram.")
            else:
                st.info("Model does not support predict_proba; skipping histogram.")

            # ROC and Precision-Recall curves
            if hasattr(pipeline, "predict_proba") and proba is not None:
                try:
                    # Binary curves using chosen positive label (lexicographically last)
                    pos_label = sorted(set(df["label"]))[-1]
                    y_true_bin = (y_test == pos_label).astype(int)
                    fpr, tpr, _ = roc_curve(y_true_bin, proba)
                    roc_auc = auc(fpr, tpr)
                    roc_df = pd.DataFrame({"FPR": fpr, "TPR": tpr})
                    st.subheader("ROC Curve")
                    roc_chart = (
                        alt.Chart(roc_df)
                        .mark_line()
                        .encode(x="FPR:Q", y="TPR:Q")
                    )
                    st.altair_chart(roc_chart, use_container_width=True)
                    st.caption(f"AUC = {roc_auc:.4f}")

                    precision, recall, _ = precision_recall_curve(y_true_bin, proba)
                    ap = average_precision_score(y_true_bin, proba)
                    pr_df = pd.DataFrame({"Recall": recall, "Precision": precision})
                    st.subheader("Precision–Recall Curve")
                    pr_chart = (
                        alt.Chart(pr_df)
                        .mark_line()
                        .encode(x="Recall:Q", y="Precision:Q")
                    )
                    st.altair_chart(pr_chart, use_container_width=True)
                    st.caption(f"Average Precision = {ap:.4f}")
                except Exception as e:
                    st.info(f"Could not compute ROC/PR curves: {e}")
        except Exception as e:
            st.warning(f"Failed to recompute visuals: {e}")

    # Live Inference
    st.header("4. Live Inference")
    if pipeline is None:
        st.info("Load artifacts to run live inference.")
    else:
        tab1, tab2 = st.tabs(["Single Text", "Batch CSV"])
        with tab1:
            # Session state to allow sample buttons to populate text
            if "live_input" not in st.session_state:
                st.session_state["live_input"] = ""
            c1, c2 = st.columns(2)
            with c1:
                if st.button("Use sample ham"):
                    st.session_state["live_input"] = "Hey, are we still on for lunch today at 12?"
            with c2:
                if st.button("Use sample spam"):
                    st.session_state["live_input"] = "URGENT! You have won a $1000 prize. Reply WIN to claim now!"
            text = st.text_area("Enter an SMS message", key="live_input")
            if st.button("Predict") and text.strip():
                pred = pipeline.predict([text])[0]
                proba_arr = None
                if hasattr(pipeline, "predict_proba"):
                    try:
                        proba_arr = pipeline.predict_proba([text])[0]
                    except Exception:
                        proba_arr = None
                st.success(f"Prediction: {pred}")
                # Probability bars per class
                if proba_arr is not None:
                    model = get_vectorizer_and_model(pipeline)[1]
                    if model is not None and hasattr(model, "classes_"):
                        classes = list(model.classes_)
                        prob_df = pd.DataFrame({"class": classes, "prob": proba_arr})
                        pchart = (
                            alt.Chart(prob_df)
                            .mark_bar()
                            .encode(x=alt.X("prob:Q", scale=alt.Scale(domain=[0, 1])), y=alt.Y("class:N", sort="-x"), tooltip=["class", "prob"])
                        )
                        st.altair_chart(pchart, use_container_width=True)
        with tab2:
            up = st.file_uploader("Upload CSV with a 'text' column", type=["csv"])
            if up is not None:
                try:
                    df_up = pd.read_csv(up)
                    if "text" not in df_up.columns:
                        st.error("CSV must contain a 'text' column")
                    else:
                        preds = pipeline.predict(df_up["text"])  # type: ignore[arg-type]
                        out = pd.DataFrame({"text": df_up["text"], "prediction": preds})
                        if hasattr(pipeline, "predict_proba"):
                            try:
                                proba_arr = pipeline.predict_proba(df_up["text"])  # type: ignore[arg-type]
                                model = get_vectorizer_and_model(pipeline)[1]
                                if model is not None and hasattr(model, "classes_") and len(model.classes_) >= 2:
                                    classes = list(model.classes_)
                                    pos_index = classes.index(sorted(classes)[-1])
                                    out["prob_positive"] = proba_arr[:, pos_index]
                            except Exception:
                                pass
                        st.dataframe(out, use_container_width=True)
                        st.download_button(
                            "Download Predictions",
                            data=out.to_csv(index=False).encode("utf-8"),
                            file_name="predictions.csv",
                            mime="text/csv",
                        )
                except Exception as e:
                    st.error(f"Failed to process uploaded CSV: {e}")


if __name__ == "__main__":
    main()
