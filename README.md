# AI Backlog Risk Prioritizer (Fintech/SaaS)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.38%2B-FF4B4B)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5%2B-F7931E)](https://scikit-learn.org/)
[![SHAP](https://img.shields.io/badge/SHAP-Explainability-blueviolet)](https://shap.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A web application for **risk-based backlog prioritization** using Machine Learning, designed for QA/Product teams in fintech and SaaS.

---

## What It Solves

- Classifies stories as **High / Medium / Low risk**
- Calculates a **Risk Score (0-100)** per story
- Automatically ranks backlog items by criticality
- Helps teams decide what to validate first

---

## Key Features

### ML-Driven Prioritization
- Upload a backlog CSV (`story` or `description` column)
- Automatic scoring using `RandomForestClassifier`
- Ranked table sorted by descending risk score

### Risk Dashboard
- Risk distribution overview
- Fast visibility into critical-item concentration

### Model Transparency
- Explainability section to understand why a story scores higher or lower
- Based on heuristic features + trained model artifacts

---

## Demo

![AI Backlog Risk Dashboard](./images/MLRiskDashboard.png "AI Backlog Risk Prioritizer")

---

## Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/crisemy/ai-test-generator.git
   cd ai-test-generator
   ```

2. **Create and activate a virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # macOS/Linux
   .venv\Scripts\Activate.ps1 # Windows PowerShell
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **(Optional) Set a Groq API key**
   Only needed if you want to use LLM-assisted mode.
   ```bash
   GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
   ```

5. **(Optional) Retrain model artifacts**
   ```bash
   jupyter notebook notebook/riskModelTraining.ipynb
   ```

6. **Run the app**
   ```bash
   streamlit run app.py
   ```
   Open [http://localhost:8501](http://localhost:8501).

---

## Project Structure

```text
ai-test-generator/
├── app.py
├── services/
│   ├── llm_service.py
│   ├── risk_service.py
│   └── export_service.py
├── assets/
│   ├── logo.svg
│   └── StoryExample.csv
├── data/
│   ├── risk_model.pkl
│   └── label_encoder.pkl
├── notebook/
│   ├── riskModelTraining.ipynb
│   └── US-priority.ipynb
├── requirements.txt
└── README.md
```

---

## How It Works

1. Load stories manually or through CSV.
2. The feature engine transforms story text into risk predictors.
3. The ML model returns risk label and score per story.
4. The app ranks stories from highest to lowest risk.
5. Teams use the ranking for planning and execution focus.

---

## Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| ML | scikit-learn (`RandomForestClassifier`) |
| Data | pandas + joblib |
| Explainability | SHAP |
| Config | python-dotenv |

---

## Requirements

```text
streamlit>=1.38.0
groq>=0.9.0
python-dotenv>=1.0.1
scikit-learn>=1.5.0
pandas>=2.2.0
numpy>=1.26.0,<2.0.0
shap>=0.40.0,<0.50.0
matplotlib>=3.7.0
```

---

## Author

**Cristian N.**

QA Engineer focused on quality architecture, risk-based prioritization, and AI/ML applications for QA.