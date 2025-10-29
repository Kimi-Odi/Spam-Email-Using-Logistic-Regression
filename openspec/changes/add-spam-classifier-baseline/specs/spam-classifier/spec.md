## ADDED Requirements
### Requirement: Baseline SMS Spam Classifier
The system SHALL train and evaluate a baseline SMS spam classifier using a simple ML model and a public dataset to establish a measurable starting point.

#### Scenario: Fetch SMS dataset from URL
- **WHEN** the training pipeline starts
- **THEN** it downloads or reads the SMS CSV from the configured URL and parses labels and messages

#### Scenario: Preprocess and vectorize
- **WHEN** messages are loaded
- **THEN** the pipeline normalizes text and vectorizes using TF-IDF with a configurable n-gram range

#### Scenario: Reproducible split
- **WHEN** splitting data
- **THEN** the pipeline uses a fixed random seed to create train/test sets

#### Scenario: Train baseline Logistic Regression
- **WHEN** training
- **THEN** the pipeline fits a Logistic Regression classifier on the training set

#### Scenario: Evaluate metrics
- **WHEN** evaluation runs on the test set
- **THEN** it outputs accuracy, precision, recall, and F1-score

#### Scenario: Persist artifacts
- **WHEN** training completes
- **THEN** the model and vectorizer artifacts are saved for later inference

#### Scenario: CLI entrypoint
- **WHEN** a user runs the training command
- **THEN** the pipeline performs fetch, preprocess, train, and evaluate, and exits with a non-error status on success
