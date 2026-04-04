import os

import matplotlib.pyplot as plt
import pandas as pd
import shap
import streamlit as st
from dotenv import load_dotenv
from groq import Groq

from services.export_service import create_framework_zip
from services.llm_service import generate_with_groq, generate_with_ollama
from services.risk_service import apply_risk_scoring, extract_features, load_risk_model, score_backlog
from providers.jira_client import fetch_jira_story, JiraFetchError

# Load environment variables
load_dotenv()

# Use environment variables directly
JIRA_API_URL = os.getenv("JIRA_API_URL")
JIRA_EMAIL = os.getenv("JIRA_EMAIL")
JIRA_API_TOKEN = os.getenv("JIRA_API_TOKEN")
JIRA_ACCEPTANCE_FIELD = os.getenv("JIRA_ACCEPTANCE_FIELD", "customfield_XXXXX")

# Jira config is optional - validated only when user selects "Jira Cloud" source
JIRA_CONFIGURED = bool(JIRA_API_URL and JIRA_EMAIL and JIRA_API_TOKEN)

st.set_page_config(page_title="AI Test Case Generator & Optimizer", layout="wide")
st.logo("assets/logo.svg")
st.title("AI-Powered Test Case Generator & Optimizer")
st.markdown("**Project 1** - Senior QA Engineer & Test Architect | Fintech/SaaS + ML Risk Scoring")

# Usar las configuraciones cargadas en lugar de constantes no definidas
st.session_state.jira_base_url = JIRA_API_URL
st.session_state.jira_email = JIRA_EMAIL
st.session_state.jira_api_token = JIRA_API_TOKEN
st.session_state.jira_ac_field = JIRA_ACCEPTANCE_FIELD

if "GROQ_API_KEY" not in st.session_state:
    st.session_state.GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
if "selected_story_col" not in st.session_state:
    st.session_state.selected_story_col = None
if "jira_base_url" not in st.session_state:
    st.session_state.jira_base_url = JIRA_API_URL
if "jira_email" not in st.session_state:
    st.session_state.jira_email = JIRA_EMAIL
if "jira_api_token" not in st.session_state:
    st.session_state.jira_api_token = JIRA_API_TOKEN
if "jira_issue_key" not in st.session_state:
    st.session_state.jira_issue_key = ""
if "jira_ac_field" not in st.session_state:
    st.session_state.jira_ac_field = JIRA_ACCEPTANCE_FIELD

# Define a default value for the Jira acceptance field if not provided in the .env file
DEFAULT_JIRA_ACCEPTANCE_FIELD = os.getenv("JIRA_ACCEPTANCE_FIELD", "customfield_XXXXX")

# Initialize the Jira acceptance field in the session state
if "jira_ac_field" not in st.session_state:
    st.session_state.jira_ac_field = DEFAULT_JIRA_ACCEPTANCE_FIELD

api_key = st.text_input("Groq API Key", value=st.session_state.GROQ_API_KEY, type="password")
client = None
if api_key:
    st.session_state.GROQ_API_KEY = api_key
    client = Groq(api_key=api_key)
else:
    st.warning("Enter your Groq API Key to continue (or use local Ollama mode).")

# Story Source
st.subheader("Story source")
source = st.radio("Select the source", ["Manual Text", "Jira Cloud"], horizontal=True)

if source == "Jira Cloud":
    if not JIRA_CONFIGURED:
        st.warning("Jira credentials not configured in .env file. Please set JIRA_API_URL, JIRA_EMAIL, and JIRA_API_TOKEN to use this feature.")
    else:
        col_a, col_b = st.columns(2)
        with col_a:
            jira_base_url = st.text_input(
                "Jira Base URL (https://tu-org.atlassian.net)",
                value=st.session_state.jira_base_url,
                disabled=not bool(JIRA_API_URL)
            )
            jira_issue_key = st.text_input(
                "Issue Key (ej: QA-123 or KAN-10)",
                value=st.session_state.jira_issue_key or "KAN-1",
                placeholder="e.g. KAN-10",
                key="jira_issue_key_input"
            )
            st.session_state.jira_issue_key = jira_issue_key
            jira_ac_field = st.text_input(
                "AC Field (customfield_XXXXX, opcional)",
                value=st.session_state.jira_ac_field or "customfield_XXXXX"
            )
        with col_b:
            jira_email = st.text_input(
                "Jira email/usuario",
                value=st.session_state.jira_email,
                disabled=not bool(JIRA_EMAIL)
            )
            jira_api_token = st.text_input(
                "Jira API token",
                value=st.session_state.jira_api_token,
                type="password",
                disabled=not bool(JIRA_API_TOKEN)
            )
            fetch_jira = st.button("Story source from Jira", key="btn_fetch_jira")

        # Fetch Jira story only when button is clicked
        if fetch_jira:
            base_url = jira_base_url or JIRA_API_URL
            email = jira_email or JIRA_EMAIL
            api_token = jira_api_token or JIRA_API_TOKEN

            if not base_url or not email or not api_token:
                st.warning("Please fill in all JIRA fields: Base URL, Email, and API Token.")
            else:
                try:
                    issue_key = st.session_state.jira_issue_key or jira_issue_key
                    if not issue_key or not issue_key.strip():
                        st.warning("No issue key provided. Enter a JIRA issue key (e.g. KAN-10) to fetch.")
                        st.stop()
                    story_text, meta = fetch_jira_story(
                        issue_key=issue_key.strip(),
                        base_url=base_url,
                        username=email,
                        api_token=api_token,
                        acceptance_field_id=jira_ac_field if jira_ac_field and jira_ac_field != "customfield_XXXXX" else None,
                    )
                    # Inject fetched story into session state so generation pipeline uses it
                    st.session_state.user_story = story_text
                    st.session_state.textarea_version += 1
                    st.success(f"Story {meta.get('issue_key')} loaded from JIRA")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error fetching JIRA story: {e}")


@st.cache_resource
def load_cached_risk_model():
    return load_risk_model()


try:
    model_rf, label_encoder = load_cached_risk_model()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()


model_options = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "Local - Qwen 3.5 (Ollama)",
]

selected_model = st.selectbox("LLM Model", model_options, index=0)
is_local_ollama = "Qwen 3.5 (Ollama)" in selected_model

if "user_story" not in st.session_state:
    st.session_state.user_story = ""
if "textarea_version" not in st.session_state:
    st.session_state.textarea_version = 0

st.divider()

tab_manual, tab_batch = st.tabs(["Manual Input & Quick Starts", "Batch CSV Upload (ML Filter)"])

with tab_manual:
    st.subheader("Quick Start Examples")
    cols = st.columns(3)
    quick_stories = [
        (
            "Fintech: Money Transfer with 2FA",
            "As a registered bank customer, I want to transfer money between my accounts with mandatory 2FA verification, so that transfers are secure.",
        ),
        (
            "SaaS: User Signup",
            "As a new user, I want to sign up with email, password, and company name validation, so I can create an account quickly and securely.",
        ),
        (
            "Edge Case: Large Amount Transfer",
            "As a premium user, I want to initiate a high-value international transfer (> $100,000) with additional approval steps.",
        ),
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

    example_path = "assets/StoryExample.csv"
    if os.path.exists(example_path):
        with open(example_path, "rb") as f:
            st.download_button(
                label="Download Sample CSV Template",
                data=f,
                file_name="StoryExample.csv",
                mime="text/csv",
            )
    else:
        st.info("Sample CSV not found. Please create a CSV with a 'story' or 'description' column.")

    uploaded_file = st.file_uploader("Upload CSV Backlog", type=["csv"])

    if uploaded_file is not None:
        try:
            df_backlog = pd.read_csv(uploaded_file)
            story_col = next(
                (col for col in df_backlog.columns if "story" in col.lower() or "description" in col.lower()),
                None,
            )

            if story_col:
                st.session_state.selected_story_col = story_col
                if st.button("Score Backlog & Prioritize", type="secondary"):
                    st.session_state.scored_backlog = score_backlog(df_backlog, story_col, model_rf, label_encoder)
            else:
                st.error("Could not find a 'story' or 'description' column in the CSV.")

        except Exception as exc:
            st.error(f"Error processing CSV: {exc}")

    if "scored_backlog" in st.session_state:
        st.success("Backlog prioritized!")
        event = st.dataframe(
            st.session_state.scored_backlog,
            use_container_width=True,
            selection_mode="single-row",
            on_select="rerun",
        )

        if event.selection.rows and st.session_state.selected_story_col:
            selected_idx = event.selection.rows[0]
            selected_story_text = st.session_state.scored_backlog.iloc[selected_idx][st.session_state.selected_story_col]
            if st.button("Load Selected Story for Generation", type="primary"):
                st.session_state.user_story = str(selected_story_text)
                st.session_state.textarea_version += 1
                if "generated_data" in st.session_state:
                    del st.session_state["generated_data"]
                st.rerun()

st.divider()

user_story = st.text_area(
    "Paste your User Story or Feature Description (English) to generate tests:",
    value=st.session_state.user_story,
    height=120,
    key=f"user_story_textarea_{st.session_state.textarea_version}",
    placeholder="Click a Quick Start button, select from your Batch CSV, or paste your own...",
)

if st.button("Clear User Story"):
    st.session_state.user_story = ""
    st.session_state.textarea_version += 1
    if "generated_data" in st.session_state:
        del st.session_state["generated_data"]
    st.rerun()

if st.button("Generate Tests + Risk Scoring", type="primary"):
    if not user_story.strip():
        st.warning("Please enter a User Story first.")
        st.stop()

    with st.spinner(f"Generating tests with {'Qwen 3.5 (Local Ollama)' if is_local_ollama else selected_model}..."):
        if is_local_ollama:
            try:
                result = generate_with_ollama(user_story)
            except Exception as exc:
                st.error(f"Ollama Error: {exc}")
                st.stop()

            data = {
                "test_cases": [],
                "scripts": result.get("scripts", []),
                "estimated_coverage": "TBD (Local)",
                "summary": result.get("summary", "Generated locally with Ollama"),
                "raw_response": result.get("raw_response", ""),
                "model_used": result.get("model_used", "llama3.2:3b (Ollama)"),
            }
        else:
            if client is None:
                st.warning("Groq API key is required for cloud models.")
                st.stop()

            try:
                data = generate_with_groq(client, user_story, selected_model)
            except ValueError as raw_output:
                st.error("Invalid JSON from LLM. Try again.")
                with st.expander("Raw response"):
                    st.code(str(raw_output), language="json")
                st.stop()
            except Exception as exc:
                st.error(f"Groq Error: {exc}")
                st.stop()

    st.success(f"Generation complete using {data.get('model_used', selected_model)}!")

    if data.get("test_cases"):
        data["test_cases"] = apply_risk_scoring(data.get("test_cases", []), model_rf, label_encoder)
        if len(data["test_cases"]) < 8:
            st.warning(
                f"The model returned only {len(data['test_cases'])} structured test cases. "
                "Try generating again to get a fuller risk table."
            )

    st.session_state.generated_data = data

if "generated_data" in st.session_state:
    data = st.session_state.generated_data
    test_cases = data.get("test_cases", [])

    tab1, tab2, tab3, tab5, tab6 = st.tabs(
        [
            "Risk Prioritized Table",
            "Playwright Scripts",
            "Gherkin Summary",
            "ML Risk Dashboard",
            "SHAP Explainability",
        ]
    )

    with tab1:
        if not is_local_ollama and data.get("test_cases"):
            # === MODO GROQ ===
            st.subheader("Test Cases Prioritized by ML Risk Score")
            df_tc = pd.DataFrame(data["test_cases"])
            st.dataframe(
                df_tc[['id', 'scenario', 'type', 'priority', 'risk_label', 'risk_score', 'risk_reason']],
                use_container_width=True,
                height=400
            )
        else:
            # === MODO OLLAMA ===
            st.subheader("Generated Content (Ollama)")
            st.caption(f"Model: {data.get('model_used', 'Llama 3.2 3B')}")

            raw_text = data.get("raw_response", "")

            if raw_text:
                col_gherkin, col_code = st.columns(2)

                with col_gherkin:
                    st.markdown("**Gherkin Test Cases**")
                    # Extraemos la parte de Gherkin si existe
                    if "```gherkin" in raw_text:
                        gherkin_part = raw_text.split("```gherkin")[-1].split("```")[0]
                    else:
                        gherkin_part = raw_text[:2000]  # fallback
                    st.text_area("Gherkin", gherkin_part.strip(), height=450, key="gherkin_area")

                with col_code:
                    st.markdown("**Playwright Code**")
                    # Intentamos extraer solo código Python
                    if "```python" in raw_text:
                        python_part = raw_text.split("```python")[-1].split("```")[0]
                        st.code(python_part.strip(), language="python")
                    else:
                        st.text_area("Raw Code", raw_text[-2500:], height=450, key="code_area")
            else:
                st.warning("No content was generated.")

    with tab2:
        st.subheader("Playwright Scripts (Python)")
        scripts = data.get("scripts", [])
        if scripts:
            for i, script in enumerate(scripts, 1):
                with st.expander(f"Script {i}", expanded=True):
                    # Normalize line breaks: convert literal \n to actual newlines
                    script_clean = script.replace("\\n", "\n")

                    if "```python" in script_clean:
                        code_part = script_clean.split("```python")[1].split("```")[0]
                        st.code(code_part.strip(), language="python")
                    else:
                        st.code(script_clean, language="python")
        else:
            st.warning("No Playwright scripts were generated.")

    with tab3:
        st.subheader("Gherkin Summary")
        st.markdown(data.get("summary", "No summary available."))
        st.info(f"Estimated Coverage: {data.get('estimated_coverage', 'N/A')}")

    with tab5:
        st.subheader("ML Risk Dashboard")
        if test_cases:
            df_scores = pd.DataFrame(test_cases)
            bins = [0, 20, 40, 60, 80, 100]
            score_buckets = pd.cut(df_scores["risk_score"], bins=bins, include_lowest=True, right=True)
            distribution = score_buckets.value_counts(sort=False)

            col_risk1, col_risk2, col_risk3 = st.columns([1, 3, 1])
            with col_risk2:
                fig, ax = plt.subplots(figsize=(7, 4))
                distribution.plot(kind="bar", ax=ax, color="#1f77b4", edgecolor="black")
                ax.set_xlabel("Risk Score Range")
                ax.set_ylabel("Number of Test Cases")
                ax.set_title("Risk Score Distribution")
                ax.tick_params(axis="x", rotation=30)
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

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
            st.info("ML Risk Dashboard is only available when using Groq models (structured JSON output).")

    with tab6:
        st.subheader("SHAP Explainability")
        if test_cases:
            st.markdown(
                """
SHAP explains *why* the selected test case got its risk score.
- Bars pushing to the **right** increase predicted risk.
- Bars pushing to the **left** decrease predicted risk.
- Larger absolute bars mean stronger influence on the model decision.
"""
            )
            tc_options = {
                f"{tc.get('id', 'TC')} - {tc.get('scenario', 'No scenario')}": tc
                for tc in test_cases
            }
            selected_tc_key = st.selectbox("Select a Test Case to explain:", list(tc_options.keys()))
            selected_tc = tc_options[selected_tc_key]

            feats = extract_features(selected_tc)
            x_pred = pd.DataFrame([feats])

            with st.spinner("Generating SHAP explanation..."):
                try:
                    explainer = shap.TreeExplainer(model_rf)
                    shap_values = explainer(x_pred)

                    value_to_plot = shap_values[0]
                    if len(shap_values.shape) == 3:
                        value_to_plot = shap_values[0, :, 1] if shap_values.shape[2] > 1 else shap_values[0]

                    col_shap1, col_shap2, col_shap3 = st.columns([1, 3, 1])
                    with col_shap2:
                        fig = plt.figure(figsize=(7, 4))
                        shap.plots.waterfall(value_to_plot, show=False)
                        st.pyplot(fig, use_container_width=True)
                        plt.close(fig)

                    st.success(
                        f"Predicted Risk Label: {selected_tc.get('risk_label')} "
                        f"(Score: {selected_tc.get('risk_score')})"
                    )
                except Exception as exc:
                    st.error(f"Could not render SHAP plot: {exc}")
        else:
            st.info("SHAP visualization is only available when structured test cases are generated (Groq mode).")

    st.download_button(
        label="Download Ready-to-run POM Framework",
        data=create_framework_zip(data),
        file_name="playwright_pom_framework.zip",
        mime="application/zip",
        type="primary",
    )
