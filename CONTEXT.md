# CONTEXT.md — AI-Powered Test Case Generator & Optimizer

This file is a concise orientation guide for AI coding agents working in this repository.

## What this project is

A Streamlit web app that turns a natural-language user story into:

1. 10–15 structured (Gherkin-style) manual test cases via an LLM (Groq / Llama 3.3 70B, or local Ollama Llama 3.2 3B).
2. Ready-to-run Playwright (Python, sync API) scripts.
3. ML-based risk scores/labels (Random Forest) used to prioritize test cases.
4. A downloadable ZIP with the scripts and framework scaffolding.
5. Optional JIRA Cloud integration: fetch an issue as the story source, and bulk-seed JIRA projects from CSV/JSON.

Target domain is fintech/SaaS quality engineering.

## Tech stack

| Layer | Technology |
|-------|------------|
| UI | Streamlit |
| LLM (cloud) | Groq SDK, `llama-3.3-70b-versatile` / `llama-3.1-8b-instant` |
| LLM (local) | Ollama, `llama3.2:3b` |
| ML | scikit-learn `RandomForestClassifier` + `LabelEncoder` |
| Explainability | SHAP (`TreeExplainer`, waterfall plots) |
| Data | pandas, numpy (<2.0), joblib |
| HTTP/integrations | requests, JIRA Cloud REST API v3 |
| Notebooks | Jupyter (`notebook/`) |

Python 3.10+. Dependencies are in `requirements.txt` (note: `pyyaml` is not listed; GitHub Actions uses pip).

## Repository layout

```
app.py                      # Entire Streamlit UI & orchestration
services/
  llm_service.py            # Groq + Ollama prompt/response handling, normalization
  risk_service.py           # Feature extraction, model loading, risk scoring
  export_service.py         # Builds downloadable Playwright POM ZIP
providers/
  jira_client.py            # Fetch a JIRA issue (REST API v3, Basic auth)
scripts/
  seed_jira.py              # Bulk-create JIRA issues from JSON template
  jira_seed_from_csv.py     # Bulk-create JIRA issues from CSV (assets/StoryExample.csv)
  README.md                 # Seed script docs
notebook/
  riskModelTraining.ipynb   # Generates synthetic data, trains + serializes model
  US-priority.ipynb         # Offline backlog prioritization demo
data/
  risk_model.pkl            # Trained RandomForest model (committed)
  label_encoder.pkl         # Label encoder for High/Medium/Low (committed)
  historical_defects.csv    # Training data — NOT committed (*.csv gitignored)
assets/
  jira-seed-template.json   # Sample seed template (5 signup stories)
  logo.svg                  # App logo
  StoryExample.csv          # Referenced by app/seed script — NOT currently present in repo
images/                     # README screenshots (jira-seed.png, MLRiskDashboard.png)
init-scripts/               # Bash helpers: repo init + venv activation
.github/workflows/seed-jira.yml  # Optional scheduled/manual JIRA seeding
jira-conf-testing.py        # Standalone JIRA connectivity/JQL smoke test
```

## Application flow (app.py)

1. `load_dotenv()`; reads JIRA env vars; configures page + logo.
2. Reads/sets Streamlit `session_state` (API keys, JIRA config, user story, textarea versioning).
3. Groq API key input (or selects local Ollama model).
4. Story source radio: **Manual Text** (with quick-start examples) or **Jira Cloud** (issue key fetch).
5. Two tabs: manual input & quick starts; batch CSV upload.
6. Batch CSV: user uploads backlog; app finds a column containing `story`/`description`; `score_backlog` ranks rows by risk; a row can be loaded into the story textarea.
7. On "Generate Tests + Risk Scoring":
   - Groq path -> `generate_with_groq` (strict JSON) -> `apply_risk_scoring`.
   - Ollama path -> `generate_with_ollama` (raw markdown Gherkin + Playwright) -> no structured risk table.
8. Results render in tabs: Risk Prioritized Table, Playwright Scripts, Gherkin Summary, ML Risk Dashboard, SHAP Explainability.
9. Download button builds a ZIP via `create_framework_zip`.

Model files are loaded with `@st.cache_resource`; a missing model aborts the app with an error.

## Key module contracts

### `services/llm_service.py`
- `generate_with_groq(client, user_story, selected_model) -> dict`
  - Prompts for exact JSON (`test_cases`, `gherkin`, `scripts`, `estimated_coverage`, `summary`).
  - Raises `ValueError(raw_output)` if JSON parsing fails.
  - `_normalize_groq_payload` guarantees types/defaults for test cases and scripts.
- `generate_with_ollama(user_story, temperature=0.4, max_tokens=5000) -> dict`
  - Uses POM/Playwright-only system prompt; returns `raw_response`, `scripts:[text]`, empty `test_cases`.

### `services/risk_service.py`
- `load_risk_model(model_path="data/risk_model.pkl", encoder_path="data/label_encoder.pkl") -> (clf, le)`; raises `FileNotFoundError` with training guidance.
- `extract_features(tc)` accepts either a string or a test-case dict and returns integer features:
  `loc`, `cyclomatic_complexity`, `prev_defects`, `negative_tests`, `edge_tests`, `money_related`, `security_related`.
  Features are heuristic (text length/word count, keyword presence: transfer/payment/money, 2fa/password/auth/token/otp).
- `apply_risk_scoring(test_cases, clf, le)` adds `risk_score` (max predicted probability × 100) and `risk_label`, then sorts descending by score.
- `score_backlog(df, story_col, clf, le)` adds `Risk Score` and `Risk Label` columns, sorted descending.

### `services/export_service.py`
- `create_framework_zip(data) -> bytes`: ZIP containing `README.md`, `requirements.txt` (pytest, pytest-playwright), `pytest.ini`, and `tests/test_NNN_generated.py` per script.

### `providers/jira_client.py`
- `fetch_jira_story(issue_key, acceptance_field_id=None, base_url=None, username=None, api_token=None) -> (story_text, meta)`
  - GET `/rest/api/3/issue/{key}`; Basic auth (`email:token` base64).
  - Returns the **summary** as story text; `meta` has `key` and `acceptance_criteria`.
  - Note: the ADF-to-text helper `_adf_to_text` exists but is currently unused by `fetch_jira_story`.
- `validate_jira_connection(...) -> bool`: GET `/rest/api/3/project`.
- `JiraFetchError` class exists but is not raised by the fetch implementation.

### JIRA seed scripts
- `scripts/seed_jira.py`: reads `assets/jira-seed-template.json` (fields: `project_key`, `issue_type`, `labels`, `tickets[]`), supports `--template` and `--dry-run`. Uses `/rest/api/3/issue` and `/rest/api/3/myself`.
- `scripts/jira_seed_from_csv.py`: requires `--project`; reads `assets/StoryExample.csv` (columns `id`, `story`) by default; `--labels`, `--csv`, `--dry-run`. Derives a summary from the "I want …" clause and adds the lowercased id as a label.
- Both create descriptions in Atlassian Document Format (ADF) and map priority names to IDs (`Highest=1 … Lowest=5`).

## ML model details

- Training notebook: `notebook/riskModelTraining.ipynb`.
- If `data/historical_defects.csv` is missing, it generates 200 synthetic samples (seed 42) across fintech modules.
- Features: `loc`, `cyclomatic_complexity`, `prev_defects`, `negative_tests`, `edge_tests`, `money_related`, `security_related`.
- Target: `risk_label` derived by rules into `High`/`Medium`/`Low`; encoded with `LabelEncoder`.
- Model: `RandomForestClassifier(n_estimators=150, max_depth=10, class_weight='balanced', random_state=42)`, 75/25 stratified split.
- Serialized to `data/risk_model.pkl` and `data/label_encoder.pkl` (both committed).
- The notebook saves with relative paths (`../data/...`), so it must be run from `notebook/`.

## Configuration / environment variables

`.env` at repo root (never commit; see `.env.example`):

| Variable | Purpose |
|----------|---------|
| `GROQ_API_KEY` | Groq cloud LLM access |
| `JIRA_API_URL` | JIRA base URL used by `app.py` and `providers/jira_client.py` (**not** `JIRA_BASE_URL`) |
| `JIRA_BASE_URL` | JIRA base URL used by the seed scripts and `.env.example` |
| `JIRA_EMAIL` | JIRA account email |
| `JIRA_API_TOKEN` | JIRA API token |
| `JIRA_ACCEPTANCE_FIELD` | Optional custom field id for acceptance criteria (default placeholder `customfield_XXXXX`) |
| `JIRA_SEED_PROJECT` | Default project for the CSV seed (documented, not read by code) |
| `JIRA_SEED_LABELS` | Default labels for the CSV seed (documented, not read by code) |

**Important inconsistency:** the UI/JIRA fetch path reads `JIRA_API_URL`, while the seed scripts and `.env.example` use `JIRA_BASE_URL`. Set both in `.env` to be safe, or standardize the code.

## Running the app

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt

# one-time: train model if data/*.pkl are absent
jupyter notebook notebook/riskModelTraining.ipynb

python -m streamlit run app.py
# http://localhost:8501
```

Local Ollama mode (optional): `ollama pull llama3.2:3b` then `ollama serve`.

JIRA seeding:

```bash
python scripts/jira_seed_from_csv.py --project KAN --dry-run
python scripts/jira_seed_from_csv.py --project KAN
python scripts/seed_jira.py --template assets/jira-seed-template.json --dry-run
python scripts/seed_jira.py
```

## Known gaps / gotchas

- `assets/StoryExample.csv` is referenced by the UI and the CSV seed script but is not currently present in the repo (and `*.csv` is gitignored). The "Download Sample CSV Template" button checks for its existence; the seed script raises `CSV file not found`.
- `data/historical_defects.csv` is not committed; it is regenerated by the training notebook.
- `app.py` uses `st.session_state.textarea_version` around the JIRA fetch block before it may be initialized (initialization occurs later), which can raise on first JIRA fetch.
- The JIRA fetch UI injects the story via `st.session_state.user_story` + `textarea_version` + `st.rerun()`; preserve this pattern when editing.
- Groq mode produces structured `test_cases` and enables risk/SHAP tabs; Ollama mode only shows raw Gherkin/Playwright and disables the structured dashboards.
- `providers/jira_client.py` logs request headers at INFO level, including the Authorization header — avoid enabling verbose logging in shared environments.
- `.gitignore` ignores `*.csv`, `.env`, `.claude/`, `docs/`, and `.ipynb_checkpoints/`.

## Conventions

- Plain Python modules under `services/` and `providers/`; no package manager beyond pip/requirements.
- UI text and code comments are mixed English/Spanish; keep user-facing copy consistent with surrounding code.
- No test suite or lint/typecheck config exists in the repo.
- Commit style in history: short imperative/`fix:`/`feat:`/`docs:` messages.
