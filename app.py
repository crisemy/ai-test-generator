import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from groq import Groq

from services.export_service import create_framework_zip
from services.llm_service import generate_with_groq, generate_with_ollama
from services.risk_service import apply_risk_scoring, load_risk_model, score_backlog

load_dotenv()

st.set_page_config(page_title="AI Test Case Generator & Optimizer", layout="wide")
st.logo("assets/logo.svg")
st.title("AI-Powered Test Case Generator & Optimizer")
st.markdown("**Project 1** - Senior QA Engineer & Test Architect | Fintech/SaaS + ML Risk Scoring")

if "GROQ_API_KEY" not in st.session_state:
    st.session_state.GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
if "selected_story_col" not in st.session_state:
    st.session_state.selected_story_col = None

api_key = st.text_input("Groq API Key", value=st.session_state.GROQ_API_KEY, type="password")
client = None
if api_key:
    st.session_state.GROQ_API_KEY = api_key
    client = Groq(api_key=api_key)
else:
    st.warning("Enter your Groq API Key to continue (or use local Ollama mode).")


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
                    if "```python" in script:
                        code_part = script.split("```python")[1].split("```")[0]
                        st.code(code_part.strip(), language="python")
                    else:
                        st.code(script, language="python")
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
            st.bar_chart(df_scores["risk_score"].value_counts(bins=5).sort_index())
        else:
            st.info("ML Risk Dashboard is only available when using Groq models (structured JSON output).")

    with tab6:
        st.subheader("SHAP Explainability")
        st.info("SHAP visualization is only available when structured test cases are generated (Groq mode).")

    st.download_button(
        label="Download Ready-to-run POM Framework",
        data=create_framework_zip(data),
        file_name="playwright_pom_framework.zip",
        mime="application/zip",
        type="primary",
    )