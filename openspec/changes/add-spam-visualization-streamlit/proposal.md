## Why
Provide an easy, interactive way to visualize model results and try the classifier. A lightweight Streamlit app will help present Phase 1 outcomes and enable quick manual testing.

## What Changes
- Add a Streamlit UI to visualize SMS spam classifier results with the following sections:
  1) Data Overview — dataset summary, class distribution, sample rows
  2) Top Tokens by Class — show most informative tokens per class (based on LR coefficients / TF‑IDF)
  3) Model Performance — metrics, classification report, confusion matrix, probability histogram
  4) Live Inference — single-text prediction and batch CSV upload
- Load artifacts from Phase 1 (vectorizer/model/pipeline, metrics)
- Optionally recompute evaluation on the dataset for visuals with a fixed seed
- Provide a CLI command to launch the app locally

## Impact
- Affected specs: `spam-visualization`
- Affected code: UI (Streamlit), artifact loading, lightweight evaluation helpers
