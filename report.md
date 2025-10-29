# Project Report — Spam Email Using Logistic Regression

## Overview
This project implements an end‑to‑end baseline for SMS spam detection using Logistic Regression with TF‑IDF features, an interactive Streamlit application for exploration and inference, and a one‑click deployment on Streamlit Cloud. The work follows an OpenSpec, spec‑driven workflow with documented proposals, tasks, and specs.

- Live app: https://spam-email-using-logistic-regression-7juxqhef7rkgydvvepajy6.streamlit.app/
- GitHub repo: https://github.com/Kimi-Odi/Spam-Email-Using-Logistic-Regression

## Data
- Source: SMS spam dataset (CSV, no header)
  - URL: https://raw.githubusercontent.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity/refs/heads/master/Chapter03/datasets/sms_spam_no_header.csv
- Schema: two columns (label, text), where label ∈ {ham, spam}.
- Handling: downloaded and cached locally; supports offline by placing the CSV at `data/sms_spam_no_header.csv`.

## Methodology
- Vectorization: TF‑IDF with lowercase, Unicode accent stripping; configurable n‑gram range and min_df.
- Model: Logistic Regression (`max_iter=1000`).
- Split: Stratified train/test (80/20) with fixed random seed (42).
- Metrics: Accuracy, Precision (weighted), Recall (weighted), F1 (weighted). Additional curves for ROC and Precision–Recall in the app.

## Training Pipeline
- Script: `train_spam_classifier.py`
  - Fetch/caches dataset; loads and cleans data
  - Fits TF‑IDF and Logistic Regression; evaluates on test set
  - Persists artifacts to `artifacts/`:
    - `pipeline_lr.joblib`, `vectorizer.joblib`, `model_lr.joblib`
    - `metrics.json`, `classification_report.txt`

## Results (local baseline)
- Accuracy: 0.9668
- Precision (weighted): 0.9680
- Recall (weighted): 0.9668
- F1 (weighted): 0.9648

## Streamlit Application
- Entry: `streamlit_app.py`
- Sections:
  1) Data Overview — dataset summary, class distribution, sample rows
  2) Top Tokens by Class — most informative tokens by LR coefficients; frequency plots for ham and spam
  3) Model Performance — metrics, classification report, confusion matrix, probability histogram, ROC curve (AUC), Precision–Recall curve (AP)
  4) Live Inference — single‑text prediction (probability bars), batch CSV upload with downloadable predictions, sample ham/spam buttons
- Configuration via sidebar: artifacts directory, dataset URL/cache path, test size, random seed.

## Deployment
- Platform: Streamlit Cloud (auto‑deploy on push enabled)
- Runtime: Python 3.11 (`runtime.txt`)
- Config: `.streamlit/config.toml` (headless server)
- Deployed URL: https://spam-email-using-logistic-regression-7juxqhef7rkgydvvepajy6.streamlit.app/

## Repository Structure (key files)
- `train_spam_classifier.py` — training and evaluation CLI
- `streamlit_app.py` — interactive visualization and inference UI
- `requirements.txt` — dependencies (scikit‑learn, pandas, numpy, scipy, joblib, streamlit)
- `runtime.txt` — Streamlit Cloud Python version pin (3.11)
- `.streamlit/config.toml` — Streamlit server/theme settings
- `artifacts/` — model/vectorizer/pipeline and metrics (checked in for cloud demo)
- `openspec/` — proposals, tasks, and specs for OpenSpec workflow

## OpenSpec Workflow Summary
- Baseline (Phase 1): `openspec/changes/add-spam-classifier-baseline/` — ADDED requirements, tasks, and proposal
- Visualization (Phase 2): `openspec/changes/add-spam-visualization-streamlit/` — ADDED requirements; full UI tasks completed
- Deployment: `openspec/changes/deploy-streamlit-cloud/` — ADDED requirements; repo prepared and app deployed
- MQTT Telemetry (proposal only): `openspec/changes/add-mqtt-telemetry/` — scoped but not implemented

## What’s Completed
- Training pipeline and artifacts for Logistic Regression + TF‑IDF
- Streamlit UI with data overview, token importance and frequencies, performance with ROC/PR, live inference with probability bars and samples
- Streamlit Cloud deployment (auto‑deploy on push)
- README with setup, run, and cloud link

## Future Work
- Data expansion: consider email spam corpora if targeting emails rather than SMS
- Modeling: add hyperparameter tuning, alternative models (e.g., linear SVM), calibrated probabilities
- Evaluation: add acceptance thresholds (e.g., minimum F1) and cross‑validation reporting
- UX: add filters for n‑gram ranges and stop‑word options; allow exporting token importance
- Ops: add CI for lint/tests and optional smoke test for the app startup
- Telemetry: implement the MQTT ingestion capability for future real‑time use cases

## Reproducibility
- Local training:
  - `python -m venv .venv`
  - Windows: `.\.venv\Scripts\python.exe -m pip install -r requirements.txt`
  - Windows: `.\.venv\Scripts\python.exe train_spam_classifier.py`
  - macOS/Linux: `./.venv/bin/python -m pip install -r requirements.txt && ./.venv/bin/python train_spam_classifier.py`
- Run app locally: `streamlit run streamlit_app.py` → http://localhost:8501
- Cloud: push to GitHub; Streamlit Cloud reads `requirements.txt` and `runtime.txt`

## References
- Dataset: Hands‑On AI for Cybersecurity, Chapter 3 — SMS spam dataset (GitHub link above)
- scikit‑learn documentation: Logistic Regression, TF‑IDF Vectorizer, metrics

