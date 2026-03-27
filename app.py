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
        st.error("ML model files not found. Please run the training notebook first.")
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
        st.success("Backlog prioritized!")
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

# ====================== MAIN USER STORY INPUT ======================
user_story = st.text_area(
    "Paste your User Story or Feature Description (English) to generate tests:",
    value=st.session_state.user_story,
    height=120,
    key=f"user_story_textarea_{st.session_state.textarea_version}",
    placeholder="Click a Quick Start button, select from your Batch CSV, or paste your own..."
)

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
            # OLLAMA PATH
            result = generate_with_ollama(user_story, temperature=0.7)
            
            if result["status"] == "error":
                st.error(f"Ollama Error: {result['error']}")
                st.stop()
            
            data = {
                "test_cases": [],
                "scripts": [result["playwright_code"]],
                "estimated_coverage": "TBD (Local)",
                "summary": "Generated locally with Qwen 3.5 via Ollama",
                "model_used": result["model_used"]
            }
            
        else:
            # GROQ PATH
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
    "Full Python Playwright sync code..."
  ],
  "estimated_coverage": "XX%",
  "summary": "Brief coverage and risk focus summary"
}}

Prioritize fintech risks. Respond with valid JSON ONLY."""

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
                st.error("Invalid JSON from LLM. Try again.")
                with st.expander("Raw response"):
                    st.code(raw_json, language="json")
                st.stop()

    st.success(f"Generation complete using {data.get('model_used', selected_model)}!")

    # ML Risk Scoring
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


# ====================== RESULTS SECTION ======================
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
            st.dataframe(df_tc[['id', 'scenario', 'type', 'priority', 'risk_label', 'risk_score', 'risk_reason']], use_container_width=True)
        else:
            st.info("Ollama mode currently returns raw response. Structured test cases coming soon.")

    with tab2:
        st.subheader("Automated Playwright Scripts")
        for i, script in enumerate(data.get("scripts", []), 1):
            with st.expander(f"Script {i}", expanded=True):
                st.code(script, language="python")

    with tab3:
        st.subheader("Summary")
        st.markdown(data.get("summary", "No summary available."))
        st.info(f"Estimated Coverage: {data.get('estimated_coverage', 'N/A')}")

    with tab4:
        st.subheader("Raw JSON")
        st.json(data)

    with tab5:
        st.subheader("ML Risk Dashboard")
        if data.get("test_cases"):
            df_scores = pd.DataFrame(data["test_cases"])
            st.bar_chart(df_scores['risk_score'].value_counts(bins=5).sort_index())

    with tab6:
        st.subheader("SHAP Explainability")
        st.info("SHAP visualization available when structured test cases are generated (Groq mode).")

    # Download ZIP
    def create_zip():
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            zip_file.writestr("README.md", "# Playwright POM Framework\nGenerated with AI Test Generator")
            zip_file.writestr("requirements.txt", "pytest\npytest-playwright\n")
            zip_file.writestr("pytest.ini", "[pytest]\naddopts = --headed --browser chromium\n")
            
            for i, script in enumerate(data.get("scripts", []), 1):
                filename = f"tests/test_{i:03d}.py"
                zip_file.writestr(filename, script)
        
        zip_buffer.seek(0)
        return zip_buffer.getvalue()

    st.download_button(
        label="⬇️ Download Ready-to-run POM Framework",
        data=create_zip(),
        file_name="playwright_pom_framework.zip",
        mime="application/zip",
        type="primary"
    )