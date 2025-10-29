## ADDED Requirements
### Requirement: Streamlit Cloud Deployment
The system SHALL be deployable on Streamlit Cloud from a GitHub repository and render all intended sections without local setup.

#### Scenario: Repository contains required files
- **WHEN** the app is deployed from GitHub
- **THEN** the repository includes `streamlit_app.py`, `requirements.txt`, and `runtime.txt` specifying a supported Python version

#### Scenario: Basic configuration present
- **WHEN** the app boots on Streamlit Cloud
- **THEN** it uses `.streamlit/config.toml` defaults and loads without manual flags

#### Scenario: Dataset availability
- **WHEN** the app runs in the cloud environment
- **THEN** it fetches the SMS dataset from the configured URL or uses a cached copy if present

#### Scenario: Artifacts availability
- **WHEN** the app starts
- **THEN** prebuilt artifacts in `artifacts/` are found so all sections render immediately; if missing, the app informs the user how to generate them

