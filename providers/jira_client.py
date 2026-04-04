from typing import Optional, Tuple, Dict, Any

import requests
import base64
from dotenv import load_dotenv
import os
import logging

# Load environment variables from .env file
load_dotenv()

# Fetch Jira credentials and base URL from environment variables
JIRA_API_URL = os.getenv("JIRA_API_URL")
JIRA_EMAIL = os.getenv("JIRA_EMAIL")  # Updated to use JIRA_EMAIL for consistency
JIRA_API_TOKEN = os.getenv("JIRA_API_TOKEN")


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
    issue_key: str,
    acceptance_field_id: Optional[str] = None,
    base_url: Optional[str] = None,
    username: Optional[str] = None,
    api_token: Optional[str] = None,
) -> Tuple[str, Dict[str, str]]:
    """
    Fetch a Jira story by issue key.

    Args:
        issue_key (str): The Jira issue key (e.g., "CRISE7-1").
        acceptance_field_id (str, optional): The custom field ID for acceptance criteria.
        base_url (str): The base URL for the Jira API.
        username (str): The Jira username (email).
        api_token (str): The Jira API token.

    Returns:
        tuple: The story text and metadata.

    Raises:
        Exception: If the request fails or the response is invalid.
    """
    if not base_url or not username or not api_token:
        raise ValueError("Missing required parameters: base_url, username, or api_token.")

    url = f"{base_url}/rest/api/3/issue/{issue_key}"
    headers = {
        "Authorization": f"Basic {base64.b64encode(f'{username}:{api_token}'.encode()).decode()}",
        "Content-Type": "application/json",
    }

    logging.info(f"Fetching Jira story from URL: {url}")
    logging.info(f"Headers: {headers}")
    logging.info(f"Issue Key: {issue_key}")

    try:
        resp = requests.get(url, headers=headers)
        if resp.status_code != 200:
            logging.error(
                f"Failed to fetch Jira story. Status code: {resp.status_code}, Response: {resp.text}"
            )
            raise Exception(
                f"Failed to fetch Jira story. Status code: {resp.status_code}"
            )

        try:
            data = resp.json()
        except ValueError as e:
            logging.error(
                f"Failed to decode JSON response. Error: {e}, Response: {resp.text}"
            )
            raise Exception(
                f"Failed to decode JSON response from Jira. Error: {e}, Response: {resp.text}"
            )

        story_text = data.get("fields", {}).get("summary", "")
        meta = {
            "key": issue_key,
            "acceptance_criteria": data.get("fields", {}).get(
                acceptance_field_id, "" if not acceptance_field_id else None
            ),
        }
        return story_text, meta

    except requests.RequestException as e:
        logging.error(f"Request to Jira API failed. Error: {e}")
        raise Exception(f"Request to Jira API failed. Error: {e}")


def validate_jira_connection(
    base_url: Optional[str] = None, username: Optional[str] = None, api_token: Optional[str] = None
) -> bool:
    """
    Validate the connection to Jira by making a simple request to the Jira API.

    Args:
        base_url (str): The base URL for the Jira API.
        username (str): The Jira username (email).
        api_token (str): The Jira API token.

    Returns:
        bool: True if the connection is successful, False otherwise.

    Raises:
        Exception: If the request fails or the response is invalid.
    """
    if not base_url or not username or not api_token:
        raise ValueError("Missing required parameters: base_url, username, or api_token.")

    url = f"{base_url}/rest/api/3/project"
    headers = {
        "Authorization": f"Basic {base64.b64encode(f'{username}:{api_token}'.encode()).decode()}",
        "Content-Type": "application/json",
    }

    logging.info(f"Validating Jira connection with URL: {url}")

    try:
        resp = requests.get(url, headers=headers)
        if resp.status_code == 200:
            logging.info("Jira connection validated successfully.")
            return True
        else:
            logging.error(
                f"Failed to validate Jira connection. Status code: {resp.status_code}, Response: {resp.text}"
            )
            raise Exception(
                f"Failed to validate Jira connection. Status code: {resp.status_code}, Response: {resp.text}"
            )
    except requests.RequestException as e:
        logging.error(f"Request to Jira API failed. Error: {e}")
        raise Exception(f"Request to Jira API failed. Error: {e}")
