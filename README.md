# AI-Powered Test Case Generator & Optimizer

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.38%2B-FF4B4B)](https://streamlit.io/)
[![Groq](https://img.shields.io/badge/Groq-Llama%203.3-orange)](https://groq.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An intelligent web tool that automatically generates high-quality test cases, Playwright automation scripts, and prioritizes them using ML-based risk scoring — built for fintech and SaaS quality engineering.

**Key Features**
- Generate 10–15 manual test cases (Gherkin-style) from a natural language user story
- Produce ready-to-run Playwright Python scripts for high-priority scenarios
- Apply **ML risk scoring** (Random Forest Classified) to prioritize tests based on predicted defect likelihood
- Export everything in a clean ZIP file (Markdown table + .py scripts + README)
- Simple, fast MVP with Streamlit + Groq LLM + scikit-learn

## Demo

![AI-Powered Test Case Generator & Optimizer](./images/MLRiskDashboard.png "AI Test Generator")  

## Quick Start

1. Clone the repository
   ```bash
   git clone https://github.com/yourusername/ai-test-generator.git
   cd ai-test-generator

Note: Provided a init-scripts folder containing .sh scripts for initializing the INIT Repo

2. Create and activate virtual environment
```bash
    python -m venv .venv
    source .venv/bin/activate   # macOS/Linux # or .venv\Scripts\activate  # Windows
```
Note: Provided a init-scripts folder containing .sh scripts for initializing the .env for MAC users
3. Install dependencies
```bash
    pip install -r requirements.txt
```
4. Set up your Groq API key
Create a .env file in the root:

GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

5. (One-time) Train the risk model
```bash
    jupyter notebook notebooks/risk_model_training.ipynb
```

6. Launch the app
```bash
    python -m streamlit run app.py
```
Open http://localhost:8501 in your browser.

## Project Structure
```bash
ai-test-generator/
├── app.py                        # Main Streamlit application
├── notebooks/
│   └── risk_model_training.ipynb # ML training, EDA and model serialization
├── data/
│   ├── historical_defects.csv    # Training dataset (synthetic or real)
│   ├── risk_model.pkl            # Trained RandomForest model
│   └── label_encoder.pkl         # Label encoder for risk categories
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```
## How it works

1.  Input — Paste a user story or use quick fintech/SaaS examples.
2.  Generation — Groq LLM (Llama 3.3 70B) creates structured test cases + Playwright scripts.
3.  Risk Scoring — Features extracted from each test case → Random Forest predicts risk label (High/Medium/Low) and score (0–100).
4.  Prioritization — Tests sorted by descending risk score.
5.  Export — Download ZIP with Markdown table (including risk), .py scripts and instructions.

## Tech Stack

Frontend: Streamlit
LLM: Groq API (Llama 3.3 70B Versatile recommended)
ML: scikit-learn (RandomForestClassifier)
Automation: Playwright (Python sync API)
Data/Model Persistence: pandas + joblib

## Requirements
```bash
streamlit>=1.38.0
groq>=0.9.0
python-dotenv>=1.0.1
pandas>=2.2.0
numpy>=1.26.0
scikit-learn>=1.5.0
joblib>=1.4.0
```
## Author

Cristian N.

QA Engineer with 20+ years of experience in software testing and automation.

MSc Candidate in Data Science & Artificial Intelligence.

Research interests include:

* Experimental QA engineering
* QA Architecture
* Reliability testing
* AI-assisted quality assurance
* Data-driven software stability analysis