## ADDED Requirements
### Requirement: Streamlit SMS Spam Visualization
The system SHALL provide a Streamlit-based UI to visualize baseline results and run interactive predictions for the SMS spam classifier.

#### Scenario: Data overview section
- **WHEN** the app loads the dataset (or cached summary)
- **THEN** it displays basic stats (rows, class counts), a class distribution chart, and a few sample rows

#### Scenario: Top tokens by class section
- **WHEN** the trained Logistic Regression coefficients and TF‑IDF vocabulary are available
- **THEN** the app shows the most informative tokens per class (e.g., top-N by coefficient magnitude) with scores

#### Scenario: Load trained artifacts
- **WHEN** the app starts
- **THEN** it loads the saved pipeline (or vectorizer+model) from a configurable artifacts directory

#### Scenario: Show metrics and report
- **WHEN** artifacts include metrics and a classification report
- **THEN** the app displays accuracy, precision, recall, F1 and renders the report

#### Scenario: Recompute evaluation for visuals
- **WHEN** metrics are missing or a confusion matrix is requested
- **THEN** the app fetches the SMS dataset and reproduces the test split with a fixed random seed to compute confusion matrix and probability histogram

#### Scenario: Single-text inference
- **WHEN** a user inputs a message
- **THEN** the app displays the predicted label (spam/ham) and probability

#### Scenario: Batch CSV predictions
- **WHEN** a user uploads a CSV with a `text` column
- **THEN** the app returns a table with predicted labels and probabilities and allows download

#### Scenario: Missing artifacts handling
- **WHEN** required artifacts are not found
- **THEN** the app shows clear instructions to run training first and where to place artifacts
