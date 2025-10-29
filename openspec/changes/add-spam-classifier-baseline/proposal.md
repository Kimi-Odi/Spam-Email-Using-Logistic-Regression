## Why
Establish a baseline spam message classification pipeline to ground future iterations. This creates a working reference using a simple ML model and a publicly available dataset to measure improvements against.

## What Changes
- Add a new `spam-classifier` capability with a Phase 1 baseline
- Fetch SMS spam dataset from: https://raw.githubusercontent.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity/refs/heads/master/Chapter03/datasets/sms_spam_no_header.csv
- Preprocess text and vectorize with TF-IDF (or equivalent)
- Train a baseline Logistic Regression (LR) model and persist artifacts
- Provide a CLI/script to train and evaluate with reproducible split and metrics (accuracy, precision, recall, F1)
- Prepare placeholders for Phase 2 changes (empty for now)

## Impact
- Affected specs: `spam-classifier`
- Affected code: data ingestion, preprocessing, training, evaluation, model artifact IO (stack TBD)
