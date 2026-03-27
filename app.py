import streamlit as st
from groq import Groq
import os
import json
import zipfile
import io
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import ollama
from typing import Dict, Any

load_dotenv()

st.set_page_config(page_title="AI Test Case Generator & Optimizer", layout="wide")
st.logo("assets/logo.svg")
st.title("AI-Powered Test Case Generator & Optimizer")
st.markdown("**Project 1** — Senior QA Engineer & Test Architect | Fintech/SaaS + ML Risk Scoring")

def generate_with_ollama(user_story: str, temperature: float = 0.7, max_tokens: int = 4096) -> Dict[str, Any]:
    """
    Generates Playwright tests using Qwen 3.5 locally via Ollama.
    Returns a compatible format with your existing Groq flow to minimize changes.
    """
    try:
        system_prompt = """You are a Senior QA Automation Engineer with +15 years of experience.
Specialist in Playwright with Python, Page Object Model (POM), sync API, robust locators, and good waits.
Generate clean, professional code, well commented in Spanish, and ready for production."""

        response = ollama.chat(
            model="qwen3.5",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"""User Story / Requirement:\n\n{user_story}\n\n
Please generate:
1. Test cases in Gherkin format (Given-When-Then)
2. Complete Playwright test code using Page Object Model (POM)
3. Separate Page Object files when possible.
Include clear comments in Spanish."""}
            ],
            options={
                "temperature": temperature,
                "num_ctx": 8192,
                "num_predict": max_tokens
            }
        )

        generated_text = response['message']['content']

        return {
            "status": "success",
            "model_used": "qwen3.5 (Ollama Local)",
            "raw_response": generated_text,
            "gherkin": "",           # You can parse this later if needed
            "playwright_code": generated_text,
            "error": None
        }

    except Exception as e:
        return {
            "status": "error",
            "model_used": "qwen3.5 (Ollama Local)",
            "raw_response": "",
            "error": str(e)
        }


# API Key handling
if "GROQ_API_KEY" not in st.session_state:
    st.session_state.GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

api_key = st.text_input("Groq API Key", value=st.session_state.GROQ_API_KEY, type="password")
if api_key:
    st.session_state.GROQ_API_KEY = api_key
    client = Groq(api_key=api_key)
else:
    st.warning("Enter your Groq API Key to continue.")
    st.stop()

# Load ML model (cached)
@st.cache_resource
def load_risk_model():
    model_path = "data/risk_model.pkl"
    encoder_path = "data/label_encoder.pkl"
    
    if not (os.path.exists(model_path) and os.path.exists(encoder_path)):
        st.error("ML model files not found. Please run the training notebook first and ensure files are saved in /data/")
        st.stop()
    
    clf = joblib.load(model_path)
    le = joblib.load(encoder_path)
    return clf, le

model_rf, label_encoder = load_risk_model()

# Feature extraction: Supports both dict (test case) and string (raw story)
def extract_features(tc):
    if isinstance(tc, str):
        text = tc.lower()
        neg = 0
        edge = 0
    else:
        text = " ".join([str(tc.get(k, "")) for k in ["scenario", "given", "when", "then"]]).lower()
        neg = 1 if tc.get('type') == 'Negative' else 0
        edge = 1 if tc.get('type') == 'Edge' else 0
        
    return {
        'loc': len(text) // 10,                     
        'cyclomatic_complexity': len(text.split()) // 5 + 5,
        'prev_defects': 4 if any(w in text for w in ['transfer', 'payment', 'money', 'withdraw']) else 1,
        'negative_tests': neg,
        'edge_tests': edge,
        'money_related': 1 if any(w in text for w in ['amount', 'money', 'transfer', 'currency', '$']) else 0,
        'security_related': 1 if any(w in text for w in ['2fa', 'password', 'auth', 'security', 'token', 'otp']) else 0
    }

# Model selection
model = st.selectbox(
    "LLM Model",
    ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"],
    index=0
)

# Initialize session state for user story persistence
if "user_story" not in st.session_state:
    st.session_state.user_story = ""
if "textarea_version" not in st.session_state:
    st.session_state.textarea_version = 0

st.divider()

# Input Method Tabs
tab_manual, tab_batch = st.tabs(["Manual Input & Quick Starts", "Batch CSV Upload (ML Filter)"])

with tab_manual:
    st.subheader("Quick Start Examples")
    cols = st.columns(3)
    quick_stories = [
        ("Fintech: Money Transfer with 2FA", 
         "As a registered bank customer, I want to transfer money between my accounts with mandatory 2FA verification, so that transfers are secure."),
        ("SaaS: User Signup", 
         "As a new user, I want to sign up with email, password, and company name validation, so I can create an account quickly and securely."),
        ("Edge Case: Large Amount Transfer", 
         "As a premium user, I want to initiate a high-value international transfer (> $100,000) with additional approval steps."),
    ]
    for col, (label, text) in zip(cols, quick_stories):
        if col.button(label, key=f"quick_{label}"):
            st.session_state.user_story = text
            st.session_state.textarea_version += 1
            if "generated_data" in st.session_state:
                del st.session_state["generated_data"]
            st.rerun()

with tab_batch:
    st.subheader("Prioritize Backlog with ML")
    st.markdown("Upload a CSV with your backlog (requires a column `story` or `description`). The ML model will score them so you generate tests only for high risks.")
    
    with open("assets/StoryExample.csv", "rb") as f:
        st.download_button(
            label="📥 Download Sample CSV Template (StoryExample.csv)",
            data=f,
            file_name="StoryExample.csv",
            mime="text/csv",
            help="Download this file, add your own User Stories in the 'story' column, then upload it below."
        )
    
    uploaded_file = st.file_uploader("Upload CSV Backlog", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df_backlog = pd.read_csv(uploaded_file)
            story_col = next((col for col in df_backlog.columns if 'story' in col.lower() or 'description' in col.lower()), None)
            
            if story_col:
                if st.button("Score Backlog & Prioritize", type="secondary"):
                    scores, labels = [], []
                    for text in df_backlog[story_col].fillna(""):
                        feats = extract_features(str(text))
                        X_pred = pd.DataFrame([feats])
                        pred_encoded = model_rf.predict(X_pred)[0]
                        risk_prob = max(model_rf.predict_proba(X_pred)[0]) * 100
                        scores.append(round(risk_prob, 1))
                        labels.append(label_encoder.inverse_transform([pred_encoded])[0])
                        
                    df_backlog['Risk Score'] = scores
                    df_backlog['Risk Label'] = labels
                    df_sorted = df_backlog.sort_values(by="Risk Score", ascending=False)
                    st.session_state.scored_backlog = df_sorted
            else:
                st.error("Could not find a 'story' or 'description' column in the CSV.")
                
        except Exception as e:
            st.error(f"Error processing CSV: {e}")
            
    if "scored_backlog" in st.session_state:
        st.success("Backlog prioritized! Select a row below to load its story into the generator.")
        # We use dataframe selection
        event = st.dataframe(
            st.session_state.scored_backlog, 
            use_container_width=True, 
            selection_mode="single-row", 
            on_select="rerun"
        )
        if event.selection.rows:
            selected_idx = event.selection.rows[0]
            selected_story_text = st.session_state.scored_backlog.iloc[selected_idx][story_col]
            if st.button("Load Selected Story for Generation", type="primary"):
                st.session_state.user_story = str(selected_story_text)
                st.session_state.textarea_version += 1
                if "generated_data" in st.session_state:
                    del st.session_state["generated_data"]
                st.rerun()

st.divider()

# Main input (LLM Generation target)
user_story = st.text_area(
    "Paste your User Story or Feature Description (English) to generate tests:",
    value=st.session_state.user_story,
    height=120,
    key=f"user_story_textarea_{st.session_state.textarea_version}",
    placeholder="Click a Quick Start button, select from your Batch CSV, or paste your own..."
)

col_clear, _ = st.columns([1, 5])
with col_clear:
    if st.button("Clear User Story"):
        st.session_state.user_story = ""
        st.session_state.textarea_version += 1
        if "generated_data" in st.session_state:
            del st.session_state["generated_data"]
        st.rerun()

# Generate button
if st.button("Generate Tests + Risk Scoring", type="primary"):
    if not user_story.strip():
        st.warning("Please enter a User Story first.")
        st.stop()

    with st.spinner("Generating tests with LLM + applying ML Risk Scoring..."):
        system_prompt = """You are an expert QA Test Architect with 20+ years in fintech/SaaS.
Specialize in risk-based testing and automation-first.
Respond ONLY with valid JSON. No extra text."""

        user_prompt = f"""Given this user story/feature:
"{user_story}"

Generate EXACTLY this JSON:

{{
  "test_cases": [
    {{"id": "TC-001", "scenario": "str", "given": "str", "when": "str", "then": "str", "type": "Positive|Negative|Edge", "priority": "High|Medium|Low", "risk_reason": "str"}}
    // 10-15 cases: >=4 positive, >=4 negative, >=3 edge/boundary
  ],
  "scripts": [
    "Full Python Playwright sync code string for top 4-6 risk cases\\nfrom playwright.sync_api import Page, expect\\ndef test_... (page: Page): ..."
  ],
  "estimated_coverage": "XX%",
  "summary": "Brief coverage and risk focus summary"
}}

Prioritize fintech risks: security, funds movement, privacy, boundaries.
Respond with valid JSON ONLY."""

        response = client.chat.completions.create(
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            model=model,
            temperature=0.2,
            max_tokens=6000,
            response_format={"type": "json_object"}
        )

        raw_json = response.choices[0].message.content
        try:
            data = json.loads(raw_json)
        except json.JSONDecodeError:
            st.error("Invalid JSON from LLM. Try again or switch model.")
            with st.expander("Raw LLM response"):
                st.code(raw_json, language="json")
            st.stop()

    st.success("Generation + Risk Scoring complete!")

    # Apply ML risk scoring
    for tc in data.get("test_cases", []):
        feats = extract_features(tc)
        X_pred = pd.DataFrame([feats])
        pred_encoded = model_rf.predict(X_pred)[0]
        risk_label = label_encoder.inverse_transform([pred_encoded])[0]
        risk_proba = model_rf.predict_proba(X_pred)[0]
        risk_score = max(risk_proba) * 100  
        tc['risk_score'] = round(risk_score, 1)
        tc['risk_label'] = risk_label

    # Sort test cases by risk score descending
    data["test_cases"] = sorted(data["test_cases"], key=lambda x: x.get('risk_score', 0), reverse=True)
    
    st.session_state.generated_data = data

if "generated_data" in st.session_state:
    data = st.session_state.generated_data

    # Tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Risk Prioritized Table",
        "Playwright Scripts",
        "Gherkin Summary",
        "Raw JSON",
        "ML Risk Dashboard",
        "SHAP Explainability"
    ])

    with tab1:
        st.subheader("Test Cases Prioritized by ML Risk Score")
        if data.get("test_cases"):
            df_tc = pd.DataFrame(data["test_cases"])
            st.dataframe(
                df_tc[['id', 'scenario', 'type', 'priority', 'risk_label', 'risk_score', 'risk_reason']],
                use_container_width=True
            )

    with tab2:
        st.subheader("Automated Playwright Scripts (Python)")
        for i, script in enumerate(data.get("scripts", []), 1):
            with st.expander(f"Script {i}", expanded=(i <= 2)):
                st.code(script, language="python")

    with tab3:
        st.subheader("Gherkin-Style Summary")
        st.markdown(data.get("summary", "No summary available."))
        st.info(f"Estimated Coverage: {data.get('estimated_coverage', 'N/A')}")

    with tab4:
        st.subheader("Raw JSON Response")
        st.json(data)

    with tab5:
        st.subheader("ML Risk Dashboard")
        if data.get("test_cases"):
            df_scores = pd.DataFrame(data["test_cases"])
            st.bar_chart(
                df_scores['risk_score'].value_counts(bins=5).sort_index(),
                x_label="Risk Score Range",
                y_label="Number of Test Cases"
            )
            st.markdown("""
            **How to read this chart:**
            - **80–100**: Very high risk — prioritize these tests first.
            - **60–80**: High risk — critical scenarios, negative/edge cases.
            - **40–60**: Medium risk — standard paths.
            - **0–40**: Low risk — basic validation.
            """)
                
        else:
            st.info("Generate some tests to see the risk distribution.")

    with tab6:
        st.subheader("Model Decision Transparency (SHAP)")
        if data.get("test_cases"):
            # Select test case by ID and Scenario
            tc_options = {f"{tc['id']} - {tc['scenario']}": tc for tc in data["test_cases"]}
            selected_tc_key = st.selectbox("Select a Test Case to explain:", list(tc_options.keys()))
            selected_tc = tc_options[selected_tc_key]
            
            feats = extract_features(selected_tc)
            X_pred = pd.DataFrame([feats])
            
            with st.spinner("Generating SHAP explanation..."):
                try:
                    explainer = shap.TreeExplainer(model_rf)
                    shap_values = explainer(X_pred)
                    
                    val_to_plot = shap_values[0]
                    if len(shap_values.shape) == 3:
                        val_to_plot = shap_values[0, :, 1] if shap_values.shape[2] > 1 else shap_values[0]

                    col_shap1, col_shap2, col_shap3 = st.columns([1, 3, 1])
                    with col_shap2:
                        fig, ax = plt.subplots(figsize=(6, 4))
                        shap.plots.waterfall(val_to_plot, show=False)
                        # Setting use_container_width=True within a narrow column limits the absolute size
                        st.pyplot(fig, use_container_width=True)
                    
                    st.success(f"**Predicted Risk Label:** {selected_tc.get('risk_label')} (Score: {selected_tc.get('risk_score')})")
                except Exception as e:
                    st.error(f"Could not render SHAP plot: {e}")
        else:
            st.info("Generate some tests first to analyze their risk score drivers.")

    # POM ZIP Download
    def create_zip():
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            # Docs
            tc_md = "| ID | Scenario | Type | Priority | Risk Label | Risk Score | Risk Reason |\n|---|---------|------|----------|------------|------------|-------------|\n"
            for tc in data.get("test_cases", []):
                tc_md += f"| {tc.get('id','')} | {tc.get('scenario','')} | {tc.get('type','')} | {tc.get('priority','')} | {tc.get('risk_label','')} | {tc.get('risk_score','')} | {tc.get('risk_reason','')} |\n"
            zip_file.writestr("docs/test_cases.md", tc_md)
            zip_file.writestr("docs/summary.md", data.get("summary", "No summary") + f"\nEstimated Coverage: {data.get('estimated_coverage', 'N/A')}")

            # POM Structure
            base_page = '''class BasePage:
    def __init__(self, page):
        self.page = page
        
    def navigate(self, url):
        self.page.goto(url)
        self.page.wait_for_load_state('networkidle')
'''
            zip_file.writestr("pages/__init__.py", "")
            zip_file.writestr("pages/base_page.py", base_page)

            zip_file.writestr("tests/__init__.py", "")
            for i, script in enumerate(data.get("scripts", []), 1):
                safe_name = data["test_cases"][i-1].get("scenario", f"script_{i}")[:30].replace(" ", "_").replace("/", "_")
                filename = f"tests/test_{i:03d}_{safe_name}.py"
                zip_file.writestr(filename, script)

            # Config
            pytest_ini = '''[pytest]
addopts = --headed --browser chromium
testpaths = tests
'''
            zip_file.writestr("pytest.ini", pytest_ini)
            
            reqs = "pytest\npytest-playwright\n"
            zip_file.writestr("requirements.txt", reqs)

            # CI/CD GitHub Actions
            gh_action = '''name: QA Playwright Regression
on:
  push:
    branches: [ main, master ]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Install Playwright browsers
        run: playwright install --with-deps chromium
      - name: Run Pytest
        run: pytest --tracing=retain-on-failure
      - uses: actions/upload-artifact@v4
        if: always()
        with:
          name: playwright-traces
          path: test-results/
'''
            zip_file.writestr(".github/workflows/qa-regression.yml", gh_action)

            # README
            readme = "# Playwright POM Framework\n\nRun locally:\n```bash\npip install -r requirements.txt\nplaywright install\npytest\n```\n"
            zip_file.writestr("README.md", readme)

        zip_buffer.seek(0)
        return zip_buffer.getvalue()

    st.download_button(
        label="⬇️ Download Ready-to-run POM Framework (Tests + Config + CI)",
        data=create_zip(),
        file_name="playwright_pom_framework.zip",
        mime="application/zip",
        type="primary"
    )import streamlit as st
from groq import Groq
import os
import json
import zipfile
import io
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import ollama
from typing import Dict, Any

load_dotenv()

st.set_page_config(page_title="AI Test Case Generator & Optimizer", layout="wide")
st.logo("assets/logo.svg")
st.title("AI-Powered Test Case Generator & Optimizer")
st.markdown("**Project 1** — Senior QA Engineer & Test Architect | Fintech/SaaS + ML Risk Scoring")


# ==================== OLLAMA FUNCTION ====================
def generate_with_ollama(user_story: str, temperature: float = 0.7, max_tokens: int = 4096) -> Dict[str, Any]:
    """
    Generates Playwright tests using Qwen 3.5 locally via Ollama.
    Returns a compatible format with your existing Groq flow.
    """
    try:
        system_prompt = """You are a Senior QA Automation Engineer with +15 years of experience.
Specialist in Playwright with Python, Page Object Model (POM), sync API, robust locators, and good waits.
Generate clean, professional code, well commented in Spanish, and ready for production."""

        response = ollama.chat(
            model="qwen3.5",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"""User Story / Requirement:\n\n{user_story}\n\n
Please generate:
1. Test cases in Gherkin format (Given-When-Then)
2. Complete Playwright test code using Page Object Model (POM)
3. Separate Page Object files when possible.
Include clear comments in Spanish."""}
            ],
            options={
                "temperature": temperature,
                "num_ctx": 8192,
                "num_predict": max_tokens
            }
        )

        generated_text = response['message']['content']

        return {
            "status": "success",
            "model_used": "qwen3.5 (Ollama Local)",
            "raw_response": generated_text,
            "gherkin": "",
            "playwright_code": generated_text,
            "error": None
        }

    except Exception as e:
        return {
            "status": "error",
            "model_used": "qwen3.5 (Ollama Local)",
            "raw_response": "",
            "error": str(e)
        }


# ==================== GROQ CLIENT ====================
if "GROQ_API_KEY" not in st.session_state:
    st.session_state.GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

api_key = st.text_input("Groq API Key", value=st.session_state.GROQ_API_KEY, type="password")

if api_key:
    st.session_state.GROQ_API_KEY = api_key
    client = Groq(api_key=api_key)
else:
    st.warning("Enter your Groq API Key to continue.")
    st.stop()


# Load ML model (cached)
@st.cache_resource
def load_risk_model():
    model_path = "data/risk_model.pkl"
    encoder_path = "data/label_encoder.pkl"
    
    if not (os.path.exists(model_path) and os.path.exists(encoder_path)):
        st.error("ML model files not found. Please run the training notebook first and ensure files are saved in /data/")
        st.stop()
    
    clf = joblib.load(model_path)
    le = joblib.load(encoder_path)
    return clf, le

model_rf, label_encoder = load_risk_model()


# Feature extraction
def extract_features(tc):
    if isinstance(tc, str):
        text = tc.lower()
        neg = 0
        edge = 0
    else:
        text = " ".join([str(tc.get(k, "")) for k in ["scenario", "given", "when", "then"]]).lower()
        neg = 1 if tc.get('type') == 'Negative' else 0
        edge = 1 if tc.get('type') == 'Edge' else 0
        
    return {
        'loc': len(text) // 10,                     
        'cyclomatic_complexity': len(text.split()) // 5 + 5,
        'prev_defects': 4 if any(w in text for w in ['transfer', 'payment', 'money', 'withdraw']) else 1,
        'negative_tests': neg,
        'edge_tests': edge,
        'money_related': 1 if any(w in text for w in ['amount', 'money', 'transfer', 'currency', '$']) else 0,
        'security_related': 1 if any(w in text for w in ['2fa', 'password', 'auth', 'security', 'token', 'otp']) else 0
    }


# ==================== MODEL SELECTION ====================
model_options = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "mixtral-8x7b-32768",
    "Local - Qwen 3.5 (Ollama)"
]

selected_model = st.selectbox(
    "LLM Model",
    model_options,
    index=0
)

is_local_ollama = "Qwen 3.5 (Ollama)" in selected_model


# Initialize session state
if "user_story" not in st.session_state:
    st.session_state.user_story = ""
if "textarea_version" not in st.session_state:
    st.session_state.textarea_version = 0


st.divider()

# ====================== INPUT TABS ======================
tab_manual, tab_batch = st.tabs(["Manual Input & Quick Starts", "Batch CSV Upload (ML Filter)"])

with tab_manual:
    st.subheader("Quick Start Examples")
    cols = st.columns(3)
    quick_stories = [
        ("Fintech: Money Transfer with 2FA", 
         "As a registered bank customer, I want to transfer money between my accounts with mandatory 2FA verification, so that transfers are secure."),
        ("SaaS: User Signup", 
         "As a new user, I want to sign up with email, password, and company name validation, so I can create an account quickly and securely."),
        ("Edge Case: Large Amount Transfer", 
         "As a premium user, I want to initiate a high-value international transfer (> $100,000) with additional approval steps."),
    ]
    for col, (label, text) in zip(cols, quick_stories):
        if col.button(label, key=f"quick_{label}"):
            st.session_state.user_story = text
            st.session_state.textarea_version += 1
            if "generated_data" in st.session_state:
                del st.session_state["generated_data"]
            st.rerun()

with tab_batch:
    st.subheader("Prioritize Backlog with ML")
    st.markdown("Upload a CSV with your backlog (requires a column `story` or `description`).")
    
    with open("assets/StoryExample.csv", "rb") as f:
        st.download_button(
            label="📥 Download Sample CSV Template",
            data=f,
            file_name="StoryExample.csv",
            mime="text/csv"
        )
    
    uploaded_file = st.file_uploader("Upload CSV Backlog", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df_backlog = pd.read_csv(uploaded_file)
            story_col = next((col for col in df_backlog.columns if 'story' in col.lower() or 'description' in col.lower()), None)
            
            if story_col:
                if st.button("Score Backlog & Prioritize", type="secondary"):
                    scores, labels = [], []
                    for text in df_backlog[story_col].fillna(""):
                        feats = extract_features(str(text))
                        X_pred = pd.DataFrame([feats])
                        pred_encoded = model_rf.predict(X_pred)[0]
                        risk_prob = max(model_rf.predict_proba(X_pred)[0]) * 100
                        scores.append(round(risk_prob, 1))
                        labels.append(label_encoder.inverse_transform([pred_encoded])[0])
                        
                    df_backlog['Risk Score'] = scores
                    df_backlog['Risk Label'] = labels
                    df_sorted = df_backlog.sort_values(by="Risk Score", ascending=False)
                    st.session_state.scored_backlog = df_sorted
            else:
                st.error("Could not find a 'story' or 'description' column in the CSV.")
                
        except Exception as e:
            st.error(f"Error processing CSV: {e}")
            
    if "scored_backlog" in st.session_state:
        st.success("Backlog prioritized! Select a row below to load its story.")
        event = st.dataframe(
            st.session_state.scored_backlog, 
            use_container_width=True, 
            selection_mode="single-row", 
            on_select="rerun"
        )
        if event.selection.rows:
            selected_idx = event.selection.rows[0]
            selected_story_text = st.session_state.scored_backlog.iloc[selected_idx][story_col]
            if st.button("Load Selected Story for Generation", type="primary"):
                st.session_state.user_story = str(selected_story_text)
                st.session_state.textarea_version += 1
                if "generated_data" in st.session_state:
                    del st.session_state["generated_data"]
                st.rerun()

st.divider()

# ====================== MAIN INPUT ======================
user_story = st.text_area(
    "Paste your User Story or Feature Description (English) to generate tests:",
    value=st.session_state.user_story,
    height=120,
    key=f"user_story_textarea_{st.session_state.textarea_version}",
    placeholder="Click a Quick Start button, select from your Batch CSV, or paste your own..."
)

col_clear, _ = st.columns([1, 5])
with col_clear:
    if st.button("Clear User Story"):
        st.session_state.user_story = ""
        st.session_state.textarea_version += 1
        if "generated_data" in st.session_state:
            del st.session_state["generated_data"]
        st.rerun()


# ====================== GENERATE BUTTON ======================
if st.button("Generate Tests + Risk Scoring", type="primary"):
    if not user_story.strip():
        st.warning("Please enter a User Story first.")
        st.stop()

    with st.spinner(f"Generating tests with {'Qwen 3.5 (Local Ollama)' if is_local_ollama else selected_model}..."):
        
        if is_local_ollama:
            # === OLLAMA PATH ===
            result = generate_with_ollama(user_story, temperature=0.7)
            
            if result["status"] == "error":
                st.error(f"Ollama Error: {result['error']}")
                st.stop()
            
            # Adapt Ollama output to the expected structure
            data = {
                "test_cases": [],
                "scripts": [result["playwright_code"]],
                "estimated_coverage": "TBD (Local)",
                "summary": "Generated locally with Qwen 3.5 via Ollama - No structured JSON parsing yet.",
                "model_used": result["model_used"]
            }
            
        else:
            # === GROQ PATH (your original logic) ===
            system_prompt = """You are an expert QA Test Architect with 20+ years in fintech/SaaS.
Specialize in risk-based testing and automation-first.
Respond ONLY with valid JSON. No extra text."""

            user_prompt = f"""Given this user story/feature:
"{user_story}"

Generate EXACTLY this JSON:

{{
  "test_cases": [
    {{"id": "TC-001", "scenario": "str", "given": "str", "when": "str", "then": "str", "type": "Positive|Negative|Edge", "priority": "High|Medium|Low", "risk_reason": "str"}}
  ],
  "scripts": [
    "Full Python Playwright sync code string for top 4-6 risk cases\\nfrom playwright.sync_api import Page, expect\\ndef test_... (page: Page): ..."
  ],
  "estimated_coverage": "XX%",
  "summary": "Brief coverage and risk focus summary"
}}

Prioritize fintech risks: security, funds movement, privacy, boundaries.
Respond with valid JSON ONLY."""

            response = client.chat.completions.create(
                messages=[{"role": "system", "content": system_prompt}, 
                         {"role": "user", "content": user_prompt}],
                model=selected_model,
                temperature=0.2,
                max_tokens=6000,
                response_format={"type": "json_object"}
            )

            raw_json = response.choices[0].message.content
            try:
                data = json.loads(raw_json)
            except json.JSONDecodeError:
                st.error("Invalid JSON from LLM. Try again or switch model.")
                with st.expander("Raw LLM response"):
                    st.code(raw_json, language="json")
                st.stop()

    st.success(f"Generation complete using {data.get('model_used', selected_model)}!")

    # Apply ML Risk Scoring
    for tc in data.get("test_cases", []):
        feats = extract_features(tc)
        X_pred = pd.DataFrame([feats])
        pred_encoded = model_rf.predict(X_pred)[0]
        risk_label = label_encoder.inverse_transform([pred_encoded])[0]
        risk_proba = model_rf.predict_proba(X_pred)[0]
        risk_score = max(risk_proba) * 100  
        tc['risk_score'] = round(risk_score, 1)
        tc['risk_label'] = risk_label

    data["test_cases"] = sorted(data["test_cases"], key=lambda x: x.get('risk_score', 0), reverse=True)
    
    st.session_state.generated_data = data


# ====================== DISPLAY RESULTS ======================
if "generated_data" in st.session_state:
    data = st.session_state.generated_data

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Risk Prioritized Table",
        "Playwright Scripts",
        "Gherkin Summary",
        "Raw JSON",
        "ML Risk Dashboard",
        "SHAP Explainability"
    ])

    with tab1:
        st.subheader("Test Cases Prioritized by ML Risk Score")
        if data.get("test_cases"):
            df_tc = pd.DataFrame(data["test_cases"])
            st.dataframe(
                df_tc[['id', 'scenario', 'type', 'priority', 'risk_label', 'risk_score', 'risk_reason']],
                use_container_width=True
            )
        else:
            st.info("No structured test cases available (Ollama mode returns raw response for now).")

    with tab2:
        st.subheader("Automated Playwright Scripts (Python)")
        scripts = data.get("scripts", [])
        if scripts:
            for i, script in enumerate(scripts, 1):
                with st.expander(f"Script {i}", expanded=(i <= 2)):
                    st.code(script, language="python")
        else:
            st.info("No scripts generated yet.")

    with tab3:
        st.subheader("Gherkin-Style Summary")
        st.markdown(data.get("summary", "No summary available."))
        st.info(f"Estimated Coverage: {data.get('estimated_coverage', 'N/A')}")

    with tab4:
        st.subheader("Raw JSON Response")
        st.json(data)

    with tab5:
        st.subheader("ML Risk Dashboard")
        if data.get("test_cases"):
            df_scores = pd.DataFrame(data["test_cases"])
            st.bar_chart(
                df_scores['risk_score'].value_counts(bins=5).sort_index(),
                x_label="Risk Score Range",
                y_label="Number of Test Cases"
            )

    with tab6:
        st.subheader("Model Decision Transparency (SHAP)")
        if data.get("test_cases"):
            tc_options = {f"{tc['id']} - {tc['scenario']}": tc for tc in data["test_cases"]}
            selected_tc_key = st.selectbox("Select a Test Case to explain:", list(tc_options.keys()))
            selected_tc = tc_options[selected_tc_key]
            
            feats = extract_features(selected_tc)
            X_pred = pd.DataFrame([feats])
            
            with st.spinner("Generating SHAP explanation..."):
                try:
                    explainer = shap.TreeExplainer(model_rf)
                    shap_values = explainer(X_pred)
                    val_to_plot = shap_values[0]

                    col_shap1, col_shap2, col_shap3 = st.columns([1, 3, 1])
                    with col_shap2:
                        fig, ax = plt.subplots(figsize=(6, 4))
                        shap.plots.waterfall(val_to_plot, show=False)
                        st.pyplot(fig, use_container_width=True)
                    
                    st.success(f"**Predicted Risk Label:** {selected_tc.get('risk_label')} (Score: {selected_tc.get('risk_score')})")
                except Exception as e:
                    st.error(f"Could not render SHAP plot: {e}")

    # ====================== DOWNLOAD ZIP ======================
    def create_zip():
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            # Docs
            tc_md = "| ID | Scenario | Type | Priority | Risk Label | Risk Score | Risk Reason |\n|---|---------|------|----------|------------|------------|-------------|\n"
            for tc in data.get("test_cases", []):
                tc_md += f"| {tc.get('id','')} | {tc.get('scenario','')} | {tc.get('type','')} | {tc.get('priority','')} | {tc.get('risk_label','')} | {tc.get('risk_score','')} | {tc.get('risk_reason','')} |\n"
            zip_file.writestr("docs/test_cases.md", tc_md)
            zip_file.writestr("docs/summary.md", data.get("summary", "No summary") + f"\nEstimated Coverage: {data.get('estimated_coverage', 'N/A')}")

            # POM Structure
            base_page = '''class BasePage:
    def __init__(self, page):
        self.page = page
        
    def navigate(self, url):
        self.page.goto(url)
        self.page.wait_for_load_state('networkidle')
'''
            zip_file.writestr("pages/__init__.py", "")
            zip_file.writestr("pages/base_page.py", base_page)
            zip_file.writestr("tests/__init__.py", "")

            for i, script in enumerate(data.get("scripts", []), 1):
                safe_name = f"script_{i}" if not data.get("test_cases") else data["test_cases"][i-1].get("scenario", f"script_{i}")[:30].replace(" ", "_").replace("/", "_")
                filename = f"tests/test_{i:03d}_{safe_name}.py"
                zip_file.writestr(filename, script)

            # Config files
            zip_file.writestr("pytest.ini", '''[pytest]\naddopts = --headed --browser chromium\ntestpaths = tests\n''')
            zip_file.writestr("requirements.txt", "pytest\npytest-playwright\n")
            
            # GitHub Actions
            gh_action = '''name: QA Playwright Regression
on:
  push:
    branches: [ main, master ]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Install Playwright browsers
        run: playwright install --with-deps chromium
      - name: Run Pytest
        run: pytest --tracing=retain-on-failure
      - uses: actions/upload-artifact@v4
        if: always()
        with:
          name: playwright-traces
          path: test-results/
'''
            zip_file.writestr(".github/workflows/qa-regression.yml", gh_action)

            readme = "# Playwright POM Framework\n\nRun locally:\n```bash\npip install -r requirements.txt\nplaywright install\npytest\n```\n"
            zip_file.writestr("README.md", readme)

        zip_buffer.seek(0)
        return zip_buffer.getvalue()

    st.download_button(
        label="⬇️ Download Ready-to-run POM Framework (Tests + Config + CI)",
        data=create_zip(),
        file_name="playwright_pom_framework.zip",
        mime="application/zip",
        type="primary"
    )