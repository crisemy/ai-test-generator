import json
from typing import Any, Dict

import ollama

def generate_with_ollama(user_story: str, temperature: float = 0.4, max_tokens: int = 5000) -> Dict[str, Any]:
    """
    Genera Gherkin + Playwright puro usando Llama 3.2 3B.
    """
    try:
        model_name = "llama3.2:3b"

        system_prompt = """You are a Senior QA Automation Engineer specialized ONLY in Playwright with Python.
Rules you MUST follow:
- Use ONLY Playwright: `from playwright.sync_api import sync_playwright, Page, expect`
- NEVER use Selenium, webdriver, or any other library.
- Always use Page Object Model (POM) pattern.
- Use good waits and expect assertions.
- Write clean, professional code with comments in Spanish."""

        user_prompt = f"""User Story:
{user_story}

Generate clearly separated sections:

1. **Gherkin Test Cases** (5 to 8 scenarios maximum, Given-When-Then)
2. **Playwright Code using POM** (create page classes when possible)

Use markdown for separation:
- Start Gherkin with ```gherkin
- Start code with ```python

Be direct and production-ready."""

        response = ollama.chat(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            options={"temperature": temperature, "num_ctx": 8192}
        )

        generated_text = response['message']['content']

        return {
            "status": "success",
            "model_used": f"{model_name} (Ollama Local)",
            "raw_response": generated_text,
            "test_cases": [],
            "scripts": [generated_text],
            "estimated_coverage": "TBD (Local)",
            "summary": "Generated locally with Llama 3.2 3B via Ollama",
            "error": None
        }

    except Exception as e:
        return {
            "status": "error",
            "model_used": "Ollama",
            "error": str(e),
            "scripts": []
        }

def generate_with_groq(client: Any, user_story: str, selected_model: str) -> Dict[str, Any]:
    system_prompt = """You are an expert QA Test Architect with 20+ years in fintech/SaaS.
Specialize in risk-based testing and automation-first.
Respond ONLY with valid JSON. No extra text."""

    user_prompt = f"""Given this user story/feature:
\"{user_story}\"

Generate EXACTLY this JSON:

{{
  \"test_cases\": [
    {{\"id\": \"TC-001\", \"scenario\": \"str\", \"given\": \"str\", \"when\": \"str\", \"then\": \"str\", \"type\": \"Positive|Negative|Edge\", \"priority\": \"High|Medium|Low\", \"risk_reason\": \"str\"}}
  ],
  \"scripts\": [
    \"Full Python Playwright sync code string...\"
  ],
  \"estimated_coverage\": \"XX%\",
  \"summary\": \"Brief coverage and risk focus summary\"
}}

Prioritize fintech risks. Respond with valid JSON ONLY."""

    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        model=selected_model,
        temperature=0.2,
        max_tokens=6000,
        response_format={"type": "json_object"},
    )

    raw_json = response.choices[0].message.content
    try:
        return json.loads(raw_json)
    except json.JSONDecodeError as exc:
        raise ValueError(raw_json) from exc