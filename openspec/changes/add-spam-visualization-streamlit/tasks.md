## 1. Implementation (Streamlit)
- [x] 1.1 Add dependency: `streamlit`
- [x] 1.2 Implement app to load `artifacts/pipeline_lr.joblib` (or vectorizer+model)
- [x] 1.3 Read `artifacts/metrics.json` and `classification_report.txt` if present
- [x] 1.4 Re-fetch dataset and reproduce split (seed 42) when visuals require it
- [x] 1.5 Section: Data Overview — dataset stats, class distribution chart, sample rows
- [x] 1.6 Section: Top Tokens by Class — compute informative tokens via LR coefficients / TF‑IDF; add ham/spam token frequency plots
- [x] 1.7 Section: Model Performance — metrics, classification report, confusion matrix, probability histogram; add ROC and Precision‑Recall curves
- [x] 1.8 Section: Live Inference — single-text prediction + batch CSV upload; add probability bars for prediction and sample ham/spam buttons
- [x] 1.9 Error handling when artifacts missing; show setup instructions
- [x] 1.10 Add run instructions: `streamlit run streamlit_app.py`

## 2. Documentation
- [x] 2.1 Update README with prerequisites and how to launch
- [x] 2.2 Note dataset/network requirements and offline cache behavior

