# SMS Spam Classifier

Interactive baseline SMS spam classifier with Streamlit visualization.

## Prerequisites
- Python 3.10+ (works on Windows/macOS/Linux)

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
Artifacts will be written to `artifacts/`:
- `pipeline_lr.joblib`, `vectorizer.joblib`, `model_lr.joblib`
- `metrics.json`, `classification_report.txt`

If your machine is offline, place the dataset at `data/sms_spam_no_header.csv` manually; the script will use the cached file.

## Run the Streamlit App
```bash
# Using module (recommended)
streamlit run streamlit_app.py

# Or explicitly via venv Python
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py   # Windows
./.venv/bin/python -m streamlit run streamlit_app.py            # macOS/Linux
```
Then open: http://localhost:8501

In the sidebar, set:
- Artifacts directory: `artifacts`
- Dataset URL/cache path: keep defaults or adjust if offline

## Features
- Data Overview: dataset summary, class distribution, sample rows
- Top Tokens by Class: LR coefficient importance + ham/spam token frequency plots
- Model Performance: metrics, classification report, confusion matrix, ROC, PR curve, probability histogram
- Live Inference: single-text prediction with probability bars; batch CSV upload; sample ham/spam buttons

## Troubleshooting
- If sections other than Data Overview don’t appear, ensure artifacts exist in `artifacts/` and re-run training.
- If offline, copy the dataset to `data/sms_spam_no_header.csv`.
- To change ports: `streamlit run streamlit_app.py --server.port 8502`.

