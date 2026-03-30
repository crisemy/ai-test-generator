#!/usr/bin/env python3
"""
JIRA Seed Script - Create bulk test tickets from JSON template
Usage: python scripts/seed_jira.py [--template assets/jira-seed-template.json] [--dry-run]
"""

import json
import os
import sys
import argparse
import requests
from pathlib import Path
from typing import Dict, List, Any
from dotenv import load_dotenv


class JiraSeedError(Exception):
    """Custom exception for JIRA seed operations"""
    pass


class JiraSeedClient:
    """Client for creating JIRA issues from template"""

    def __init__(self, base_url: str, email: str, api_token: str):
        """Initialize JIRA client with credentials"""
        if not all([base_url, email, api_token]):
            raise JiraSeedError(
                "Missing JIRA configuration. Set JIRA_BASE_URL, JIRA_EMAIL, JIRA_API_TOKEN in .env"
            )

        self.base_url = base_url.rstrip("/")
        self.email = email
        self.api_token = api_token
        self.session = requests.Session()
        self.session.auth = (self.email, self.api_token)
        self.session.headers.update({"Accept": "application/json"})

    def test_connection(self) -> bool:
        """Test JIRA API connection"""
        try:
            resp = self.session.get(f"{self.base_url}/rest/api/3/myself", timeout=10)
            if resp.status_code == 401:
                raise JiraSeedError("Invalid JIRA credentials (401)")
            if resp.status_code == 404:
                raise JiraSeedError("JIRA base URL not found (404)")
            if not resp.ok:
                raise JiraSeedError(f"JIRA API error: {resp.status_code}")
            return True
        except requests.RequestException as e:
            raise JiraSeedError(f"Cannot connect to JIRA: {e}")

    def create_issue(
        self,
        project_key: str,
        issue_type: str,
        summary: str,
        description: str,
        labels: List[str],
        priority: str = "Medium",
    ) -> Dict[str, Any]:
        """
        Create a JIRA issue via REST API
        Returns: dict with issue_key and url
        """
        # Map priority text to JIRA priority ID
        priority_map = {
            "Lowest": 5,
            "Low": 4,
            "Medium": 3,
            "High": 2,
            "Highest": 1,
        }
        priority_id = priority_map.get(priority, 3)

        # Build acceptance criteria as bullet points
        acceptance_text = ""
        # Note: acceptance_criteria would come from the ticket if needed

        # Construct issue payload
        payload = {
            "fields": {
                "project": {"key": project_key},
                "issuetype": {"name": issue_type},
                "summary": summary,
                "description": {
                    "version": 1,
                    "type": "doc",
                    "content": [
                        {
                            "type": "paragraph",
                            "content": [{"type": "text", "text": description}],
                        }
                    ],
                },
                "labels": labels,
                "priority": {"id": str(priority_id)},
            }
        }

        try:
            resp = self.session.post(
                f"{self.base_url}/rest/api/3/issue",
                json=payload,
                timeout=15,
            )

            if resp.status_code == 201:
                data = resp.json()
                issue_key = data.get("key", "UNKNOWN")
                return {
                    "status": "success",
                    "issue_key": issue_key,
                    "url": f"{self.base_url}/browse/{issue_key}",
                    "message": f"Created {issue_key}",
                }
            elif resp.status_code == 400:
                error_msg = resp.json().get("errorMessages", ["Invalid request"])[0]
                return {
                    "status": "error",
                    "issue_key": None,
                    "message": f"Validation error: {error_msg}",
                }
            elif resp.status_code == 401:
                raise JiraSeedError("JIRA authentication failed (401)")
            else:
                return {
                    "status": "error",
                    "issue_key": None,
                    "message": f"JIRA API error {resp.status_code}: {resp.text[:200]}",
                }

        except requests.RequestException as e:
            return {
                "status": "error",
                "issue_key": None,
                "message": f"Request failed: {e}",
            }

    def seed_from_template(
        self, template_path: str, dry_run: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Load template and create issues
        Returns: list of results for each ticket
        """
        if not os.path.exists(template_path):
            raise JiraSeedError(f"Template file not found: {template_path}")

        with open(template_path, "r", encoding="utf-8") as f:
            template = json.load(f)

        project_key = template.get("project_key")
        issue_type = template.get("issue_type", "Story")
        labels = template.get("labels", [])
        tickets = template.get("tickets", [])

        if not project_key:
            raise JiraSeedError("Template missing 'project_key'")

        results = []

        for ticket in tickets:
            summary = ticket.get("summary")
            description = ticket.get("description")
            priority = ticket.get("priority", "Medium")
            ticket_labels = labels + [ticket.get("key", "").lower()]

            if not summary or not description:
                results.append(
                    {
                        "status": "skipped",
                        "issue_key": ticket.get("key"),
                        "message": "Missing summary or description",
                    }
                )
                continue

            print(f"[{'DRY RUN' if dry_run else 'CREATING'}] {ticket.get('key')}: {summary}")

            if not dry_run:
                result = self.create_issue(
                    project_key=project_key,
                    issue_type=issue_type,
                    summary=summary,
                    description=description,
                    labels=ticket_labels,
                    priority=priority,
                )
                results.append(result)
            else:
                results.append(
                    {
                        "status": "dry-run",
                        "issue_key": ticket.get("key"),
                        "message": f"Would create {summary}",
                    }
                )

        return results


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Create JIRA issues from JSON template (no LLM required)"
    )
    parser.add_argument(
        "--template",
        default="assets/jira-seed-template.json",
        help="Path to JSON template file",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be created without making API calls",
    )

    args = parser.parse_args()

    # Load environment variables
    load_dotenv()

    jira_base_url = os.getenv("JIRA_BASE_URL", "").strip()
    jira_email = os.getenv("JIRA_EMAIL", "").strip()
    jira_api_token = os.getenv("JIRA_API_TOKEN", "").strip()

    if not jira_base_url:
        print("❌ Error: JIRA_BASE_URL not set in .env")
        sys.exit(1)

    try:
        client = JiraSeedClient(jira_base_url, jira_email, jira_api_token)

        print(f"🔗 Testing JIRA connection to {jira_base_url}...")
        client.test_connection()
        print("✅ JIRA connection successful")

        print(f"\n📋 Loading template from {args.template}...")
        results = client.seed_from_template(args.template, dry_run=args.dry_run)

        # Print results summary
        print("\n" + "=" * 60)
        print("SEED RESULTS")
        print("=" * 60)

        success_count = sum(1 for r in results if r["status"] == "success")
        error_count = sum(1 for r in results if r["status"] == "error")
        skipped_count = sum(1 for r in results if r["status"] == "skipped")
        dry_run_count = sum(1 for r in results if r["status"] == "dry-run")

        for result in results:
            status = result["status"].upper()
            key = result.get("issue_key", "N/A")
            msg = result.get("message", "")

            if result["status"] == "success":
                url = result.get("url", "")
                print(f"✅ {status:8} | {key:12} | {msg}")
                print(f"           → {url}")
            elif result["status"] == "error":
                print(f"❌ {status:8} | {key:12} | {msg}")
            elif result["status"] == "dry-run":
                print(f"🔄 {status:8} | {key:12} | {msg}")
            else:
                print(f"⏭️  {status:8} | {key:12} | {msg}")

        print("\n" + "=" * 60)
        print(
            f"Summary: {success_count} created, {error_count} errors, "
            f"{skipped_count} skipped, {dry_run_count} dry-run"
        )
        print("=" * 60)

        if args.dry_run:
            print("\n💡 This was a DRY RUN. Run without --dry-run to create issues.")

        sys.exit(0 if error_count == 0 else 1)

    except JiraSeedError as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
