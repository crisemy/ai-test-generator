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

from providers.jira_client import fetch_jira_story, JiraFetchError

load_dotenv()

DEFAULT_JIRA_BASE_URL = os.getenv("JIRA_BASE_URL", "")
DEFAULT_JIRA_EMAIL = os.getenv("JIRA_EMAIL", "")
DEFAULT_JIRA_API_TOKEN = os.getenv("JIRA_API_TOKEN", "")
DEFAULT_JIRA_ACCEPTANCE_FIELD = os.getenv("JIRA_ACCEPTANCE_FIELD", "")

st.set_page_config(page_title="AI Test Case Generator & Optimizer", layout="wide")
st.logo("assets/logo.svg")
st.title("AI-Powered Test Case Generator & Optimizer")
st.markdown("**Project 1** — Senior QA Engineer & Test Architect | Fintech/SaaS + ML Risk Scoring")

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
if "jira_base_url" not in st.session_state:
    st.session_state.jira_base_url = DEFAULT_JIRA_BASE_URL
if "jira_email" not in st.session_state:
    st.session_state.jira_email = DEFAULT_JIRA_EMAIL
if "jira_api_token" not in st.session_state:
    st.session_state.jira_api_token = DEFAULT_JIRA_API_TOKEN
if "jira_issue_key" not in st.session_state:
    st.session_state.jira_issue_key = ""
if "jira_ac_field" not in st.session_state:
    st.session_state.jira_ac_field = DEFAULT_JIRA_ACCEPTANCE_FIELD

# Story Source
st.subheader("Story source")
source = st.radio("Select the source", ["Manual Text", "Jira Cloud"], horizontal=True)

if source == "Jira Cloud":
    col_a, col_b = st.columns(2)
    with col_a:
        jira_base_url = st.text_input("Jira Base URL (https://tu-org.atlassian.net)", value=st.session_state.jira_base_url)
        jira_issue_key = st.text_input("Issue Key (ej: QA-123)", value=st.session_state.jira_issue_key)
        jira_ac_field = st.text_input("AC Field (customfield_XXXXX, opcional)", value=st.session_state.jira_ac_field)
    with col_b:
        jira_email = st.text_input("Jira email/usuaer", value=st.session_state.jira_email)
        jira_api_token = st.text_input("Jira API token", value=st.session_state.jira_api_token, type="password")
        fetch_jira = st.button("Story source from Jira", key="btn_fetch_jira")

    if fetch_jira:
        try:
            story_text, meta = fetch_jira_story(
                base_url=jira_base_url,
                email=jira_email,
                api_token=jira_api_token,
                issue_key=jira_issue_key,
                acceptance_field_id=jira_ac_field or None,
            )
        except JiraFetchError as exc:
            st.error(str(exc))
        else:
            st.session_state.user_story = story_text
            st.session_state.textarea_version += 1
            st.session_state.jira_base_url = jira_base_url
            st.session_state.jira_email = jira_email
            st.session_state.jira_api_token = jira_api_token
            st.session_state.jira_issue_key = jira_issue_key
            st.session_state.jira_ac_field = jira_ac_field
            if "generated_data" in st.session_state:
                del st.session_state["generated_data"]
            st.success(f"Story {meta.get('issue_key')} uploaded from Jira")
            st.rerun()

# Quick Start Examples
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
    if col.button(label):
        st.session_state.user_story = text
        st.session_state.textarea_version += 1
        if "generated_data" in st.session_state:
            del st.session_state["generated_data"]
        st.rerun()

# Main input
user_story = st.text_area(
    "Paste your User Story or Feature Description (English):",
    value=st.session_state.user_story,
    height=180,
    key=f"user_story_textarea_{st.session_state.textarea_version}",
    placeholder="Click a Quick Start button or paste your own..."
)

col_clear, _ = st.columns([1, 3])
with col_clear:
    if st.button("Clear User Story"):
        st.session_state.user_story = ""
        st.session_state.textarea_version += 1
        if "generated_data" in st.session_state:
            del st.session_state["generated_data"]
        st.rerun()

# Load ML model (cached)
@st.cache_resource
def load_risk_model():
    model_path = "data/risk_model.pkl"
    encoder_path = "data/label_encoder.pkl"
    
    if not (os.path.exists(model_path) and os.path.exists(encoder_path)):
        st.error("ML model files not found. Please run the training notebook first (notebooks/risk_model_training.ipynb) and ensure files are saved in /data/")
        st.stop()
    
    clf = joblib.load(model_path)
    le = joblib.load(encoder_path)
    return clf, le

model_rf, label_encoder = load_risk_model()

# Feature extraction from generated test case
def extract_features(tc):
    text = " ".join([str(tc.get(k, "")) for k in ["scenario", "given", "when", "then"]]).lower()
    return {
        'loc': len(text) // 10,                     # Proxy complexity (aprox lines of code)
        'cyclomatic_complexity': len(text.split()) // 5 + 5,  # proxy simple
        'prev_defects': 4 if any(w in text for w in ['transfer', 'payment', 'money', 'withdraw']) else 1,
        'negative_tests': 1 if tc.get('type') == 'Negative' else 0,
        'edge_tests': 1 if tc.get('type') == 'Edge' else 0,
        'money_related': 1 if any(w in text for w in ['amount', 'money', 'transfer', 'currency', '$']) else 0,
        'security_related': 1 if any(w in text for w in ['2fa', 'password', 'auth', 'security', 'token', 'otp']) else 0
    }

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
        risk_score = max(risk_proba) * 100  # confidence del label predicho
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
            
            # Gráfico principal
            st.bar_chart(
                df_scores['risk_score'].value_counts(bins=5).sort_index(),
                x_label="Risk Score Range",
                y_label="Number of Test Cases"
            )
            
            # Explicación clara y visible
            st.markdown("""
            **How to read this chart:**
            - Each bar shows how many generated test cases fall into that risk score range.
            - **Risk Score** (0–100) represents the model's predicted likelihood of high defect probability.
            - **80–100**: Very high risk — prioritize these tests first (likely security/money movement issues)
            - **60–80**: High risk — critical scenarios, negative/edge cases
            - **40–60**: Medium risk — standard but potentially problematic
            - **20–40**: Low-medium — mostly positive/happy paths
            - **0–20**: Low risk — basic validation, low impact expected
            
            The model was trained on historical defect patterns (complexity, previous bugs, security/money keywords, etc.).
            Higher score = test case more likely to uncover real defects based on past data.
            """)
            
            # Expand it even more
            with st.expander("More about the Risk Model (Advanced)"):
                st.markdown("""
                - **Model**: Random Forest Classifier
                - **Key features used**:
                - Estimated complexity (text length proxy)
                - Presence of money/transfer keywords
                - Security-related terms (2FA, password, auth...)
                - Test type (Negative/Edge → higher risk)
                - **Training data**: Synthetic fintech defects (can be replaced with real datasets later)
                - **Output**: Probability of "High" risk class × 100
                """)
                
        else:
            st.info("Generate some tests to see the risk distribution.")

    with tab6:
        st.subheader("Model Decision Transparency (SHAP)")
        st.markdown("Understand EXACTLY why the ML model assigned a specific Risk Score to a test case.")
        
        if data.get("test_cases"):
            # Select test case by ID and Scenario
            tc_options = {f"{tc['id']} - {tc['scenario']}": tc for tc in data["test_cases"]}
            selected_tc_key = st.selectbox("Select a Test Case to explain:", list(tc_options.keys()))
            selected_tc = tc_options[selected_tc_key]
            
            # Reconstruct feature vector
            feats = extract_features(selected_tc)
            X_pred = pd.DataFrame([feats])
            
            with st.spinner("Generating SHAP explanation..."):
                try:
                    # Initialize tree explainer
                    explainer = shap.TreeExplainer(model_rf)
                    shap_values = explainer(X_pred)
                    
                    # Target class logic: model_rf outputs predicting risk classes.
                    # Usually, index 1 is the positive/high risk class, but we need to pass the appropriate index.
                    # To keep it robust, if shap_values handles multi-class, we pick the first instance and the predicted class.
                    # With RandomForest in scikit-learn, shap_values might have shape (n_samples, n_features, n_classes).
                    val_to_plot = shap_values[0]
                    if len(shap_values.shape) == 3:
                        # Extract the array corresponding to the specific predicted class logic, or default to class index 1 (usually High Risk)
                        # We just plot the index 1 for highest probability class
                        val_to_plot = shap_values[0, :, 1] if shap_values.shape[2] > 1 else shap_values[0]

                    fig, ax = plt.subplots(figsize=(8, 5))
                    shap.plots.waterfall(val_to_plot, show=False)
                    st.pyplot(fig)
                    
                    st.success(f"**Predicted Risk Label:** {selected_tc.get('risk_label')} (Score: {selected_tc.get('risk_score')})")
                    st.markdown("""
                    **How to read the SHAP Waterfall Plot:**
                    - **f(x)** is the model output (before probability scaling) for this specific test case.
                    - **E[f(x)]** is the baseline average output across the training data.
                    - Red bars (+): Features that *increased* the risk (pushed it higher).
                    - Blue bars (-): Features that *decreased* the risk (pushed it lower).
                    """)
                except Exception as e:
                    st.error(f"Could not render SHAP plot: {e}")
        else:
            st.info("Generate some tests first to analyze their risk score drivers.")

    # ZIP Download
    def create_zip():
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            # test_cases.md with risk columns
            tc_md = "| ID | Scenario | Type | Priority | Risk Label | Risk Score | Risk Reason |\n|---|---------|------|----------|------------|------------|-------------|\n"
            for tc in data.get("test_cases", []):
                tc_md += f"| {tc.get('id','')} | {tc.get('scenario','')} | {tc.get('type','')} | {tc.get('priority','')} | {tc.get('risk_label','')} | {tc.get('risk_score','')} | {tc.get('risk_reason','')} |\n"
            zip_file.writestr("test_cases.md", tc_md)

            # scripts
            for i, script in enumerate(data.get("scripts", []), 1):
                safe_name = data["test_cases"][i-1].get("scenario", f"script_{i}")[:30].replace(" ", "_").replace("/", "_")
                filename = f"test_{i:03d}_{safe_name}.py"
                zip_file.writestr(filename, script)

            # summary
            zip_file.writestr("summary.md", data.get("summary", "No summary") + f"\nEstimated Coverage: {data.get('estimated_coverage', 'N/A')}")

            # README
            readme = "# Generated Tests with Risk Scoring\n\nRun scripts:\n```bash\npip install pytest pytest-playwright\nplaywright install\npytest *.py\n```\n"
            zip_file.writestr("README_tests.md", readme)

        zip_buffer.seek(0)
        return zip_buffer.getvalue()

    st.download_button(
        label="⬇️ Download ZIP (Tests + Scripts + README)",
        data=create_zip(),
        file_name="ai_generated_tests_with_risk.zip",
        mime="application/zip"
    )
