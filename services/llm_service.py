import json
from typing import Any, Dict, List

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
  \"gherkin\": \"Full Gherkin test cases in ```gherkin``` format with Feature and Scenario keywords\",
  \"scripts\": [
    \"Full Python Playwright sync code string...\"
  ],
  \"estimated_coverage\": \"XX%\",
  \"summary\": \"Brief coverage and risk focus summary\"
}}

Rules:
- Return 10 to 15 test cases.
- Include at least 4 Positive, 4 Negative, and 2 Edge cases.
- `test_cases` must be an array of objects (no markdown, no prose).
- `scripts` must be an array of plain Python code strings.
- `gherkin` must be a single string with properly formatted Gherkin (Feature, Scenario, Given/When/Then).
- Do not place test cases in `summary`.

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
        data = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        raise ValueError(raw_json) from exc

    return _normalize_groq_payload(data)


def _normalize_groq_payload(data: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(data, dict):
        raise ValueError("Invalid JSON payload type from Groq.")

    test_cases_raw = data.get("test_cases", [])
    scripts_raw = data.get("scripts", [])

    if not isinstance(test_cases_raw, list):
        test_cases_raw = []
    if not isinstance(scripts_raw, list):
        scripts_raw = []

    normalized_cases: List[Dict[str, Any]] = []
    for idx, tc in enumerate(test_cases_raw, 1):
        if not isinstance(tc, dict):
            continue
        normalized_cases.append(
            {
                "id": str(tc.get("id", f"TC-{idx:03d}")),
                "scenario": str(tc.get("scenario", "")).strip(),
                "given": str(tc.get("given", "")).strip(),
                "when": str(tc.get("when", "")).strip(),
                "then": str(tc.get("then", "")).strip(),
                "type": str(tc.get("type", "Positive")).strip(),
                "priority": str(tc.get("priority", "Medium")).strip(),
                "risk_reason": str(tc.get("risk_reason", "")).strip(),
            }
        )

    normalized_scripts = [str(script) for script in scripts_raw if isinstance(script, (str, int, float))]

    return {
        "test_cases": normalized_cases,
        "scripts": normalized_scripts,
        "estimated_coverage": str(data.get("estimated_coverage", "N/A")),
        "summary": str(data.get("summary", "No summary available.")),
    }
