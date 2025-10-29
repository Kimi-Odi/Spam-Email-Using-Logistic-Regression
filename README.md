# Spam-Email-Using-Logistic-Regression

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://spam-email-using-logistic-regression-7juxqhef7rkgydvvepajy6.streamlit.app/)

Interactive baseline SMS spam classifier with Streamlit visualization and Streamlit Cloud deployment.

## Prerequisites
- Python 3.10+ (Windows/macOS/Linux)

## Setup
```bash
python -m venv .venv
# Windows
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
# macOS/Linux
./.venv/bin/python -m pip install -r requirements.txt
```

## Train Baseline (Logistic Regression + TF‑IDF)
```bash
# Windows
.\.venv\Scripts\python.exe train_spam_classifier.py
# macOS/Linux
./.venv/bin/python train_spam_classifier.py
```
Artifacts are written to `artifacts/`:
- `pipeline_lr.joblib`, `vectorizer.joblib`, `model_lr.joblib`
- `metrics.json`, `classification_report.txt`

If offline, place the dataset at `data/sms_spam_no_header.csv` manually.

## Run the Streamlit App
```bash
streamlit run streamlit_app.py

# Or explicitly via venv Python
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py   # Windows
./.venv/bin/python -m streamlit run streamlit_app.py            # macOS/Linux
```
Open: http://localhost:8501

Sidebar settings:
- Artifacts directory: `artifacts`
- Dataset URL/cache path: keep defaults or adjust if offline

## Features
- Data Overview: dataset summary, class distribution, sample rows
- Top Tokens by Class: LR coefficient importance + ham/spam token frequency plots
- Model Performance: metrics, classification report, confusion matrix, ROC, PR curve, probability histogram
- Live Inference: single-text prediction with probability bars; batch CSV upload; sample ham/spam buttons

## Project Report
- See the full report with methodology, results, and workflow: [report.md](report.md)

## Cloud Deploy
- Deployed app: https://spam-email-using-logistic-regression-7juxqhef7rkgydvvepajy6.streamlit.app/
- Auto-deploy on push is enabled via Streamlit Cloud settings.

## Troubleshooting
- If sections other than Data Overview don’t appear, ensure artifacts exist in `artifacts/` and re-run training.
- If offline, copy the dataset to `data/sms_spam_no_header.csv`.
- Change port: `streamlit run streamlit_app.py --server.port 8502`.
