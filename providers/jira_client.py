from typing import Optional, Tuple, Dict, Any

import requests


class JiraFetchError(Exception):
    """Custom error to make UI messaging cleaner."""


def _adf_to_text(node: Any) -> str:
    """Rudimentary conversion from Atlassian Document Format to plain text."""
    if node is None:
        return ""
    if isinstance(node, str):
        return node
    if isinstance(node, list):
        return "\n".join(filter(None, (_adf_to_text(n) for n in node)))
    if isinstance(node, dict):
        node_type = node.get("type")
        text = node.get("text")
        if text:
            return text
        children = node.get("content", [])
        joined = "\n".join(filter(None, (_adf_to_text(c) for c in children)))
        if node_type in {"paragraph", "heading"}:
            return joined + "\n"
        return joined
    return ""


def fetch_jira_story(
    base_url: str,
    email: str,
    api_token: str,
    issue_key: str,
    acceptance_field_id: Optional[str] = None,
) -> Tuple[str, Dict[str, str]]:
    """
    Fetch a Jira issue (Cloud v3 API) and return plain-text story + metadata.

    Returns: (story_text, metadata)
    Raises: JiraFetchError on any failure that should be surfaced to the UI.
    """
    if not (base_url and email and api_token and issue_key):
        raise JiraFetchError("Faltan datos para conectarse a Jira (URL, email, token o issue key).")

    url = base_url.rstrip("/") + f"/rest/api/3/issue/{issue_key}"
    params = {"fields": "summary,description" + ("," + acceptance_field_id if acceptance_field_id else "")}

    try:
        resp = requests.get(url, params=params, auth=(email, api_token), timeout=15)
    except requests.RequestException as exc:
        raise JiraFetchError(f"No se pudo contactar a Jira: {exc}") from exc

    if resp.status_code == 401:
        raise JiraFetchError("Jira devolvió 401 (credenciales inválidas o token vencido).")
    if resp.status_code == 404:
        raise JiraFetchError(f"Issue {issue_key} no encontrado o sin permisos.")
    if not resp.ok:
        raise JiraFetchError(f"Error {resp.status_code} al leer Jira: {resp.text[:200]}")

    data = resp.json()
    fields = data.get("fields", {})
    summary = fields.get("summary", "")
    description = fields.get("description", "")
    acceptance = None

    if acceptance_field_id:
        acceptance = fields.get(acceptance_field_id)

    story_parts = []
    if summary:
        story_parts.append(summary)
    if description:
        story_parts.append(_adf_to_text(description).strip())
    if acceptance:
        story_parts.append("Acceptance Criteria:\n" + _adf_to_text(acceptance).strip())

    story_text = "\n\n".join(filter(None, story_parts)).strip()
    if not story_text:
        raise JiraFetchError("La issue no tiene summary/description legibles.")

    metadata = {
        "summary": summary,
        "issue_key": issue_key,
        "url": base_url.rstrip("/") + f"/browse/{issue_key}",
    }
    return story_text, metadata
