#!/usr/bin/env python3
"""
JIRA Seed from CSV - Create JIRA issues from StoryExample.csv

Usage:
    python scripts/jira_seed_from_csv.py --project KAN --dry-run
    python scripts/jira_seed_from_csv.py --project KAN
    python scripts/jira_seed_from_csv.py --project KAN --labels ai-seed,sprint-1
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional

import pandas as pd
import requests
from dotenv import load_dotenv


class JiraSeedCSVError(Exception):
    """Custom exception for JIRA seed CSV operations."""
    pass


class JiraSeedCSVClient:
    """Client for creating JIRA issues from CSV source."""

    DEFAULT_LABELS = ["ai-seed", "fintech"]

    def __init__(self, base_url: str, email: str, api_token: str):
        """Initialize JIRA client with credentials."""
        if not all([base_url, email, api_token]):
            raise JiraSeedCSVError(
                "Missing JIRA configuration. Set JIRA_BASE_URL, JIRA_EMAIL, JIRA_API_TOKEN in .env"
            )

        self.base_url = base_url.rstrip("/")
        self.auth = (email, api_token)
        self.headers = {"Accept": "application/json", "Content-Type": "application/json"}

    def test_connection(self) -> bool:
        """Test JIRA API connection."""
        try:
            resp = requests.get(
                f"{self.base_url}/rest/api/3/myself",
                auth=self.auth,
                headers=self.headers,
                timeout=10,
            )
            if resp.status_code == 401:
                raise JiraSeedCSVError("Invalid JIRA credentials (401)")
            if resp.status_code == 404:
                raise JiraSeedCSVError("JIRA base URL not found (404)")
            if not resp.ok:
                raise JiraSeedCSVError(f"JIRA API error: {resp.status_code}")
            return True
        except requests.RequestException as e:
            raise JiraSeedCSVError(f"Cannot connect to JIRA: {e}")

    def extract_summary_from_story(self, story: str) -> str:
        """
        Extract summary from user story.
        Format: 'As a X, I want Y so that Z' -> 'Y' (the 'I want' clause)
        Format: 'As a X, I want Y so I can Z' -> 'Y' (before purpose clause)
        Falls back to first 80 chars if parsing fails.
        """
        story_clean = story.strip()
        story_lower = story_clean.lower()

        # Define clause markers that indicate an incomplete purpose (end of summary)
        purpose_markers = ["so that", "so i can", "so i could", "so i will", "so i would"]
        for marker in purpose_markers:
            if marker in story_lower:
                idx = story_lower.find(marker)
                before_marker = story_clean[:idx].strip()
                if "i want" in before_marker.lower():
                    summary = before_marker.lower().split("i want")[1].strip()
                    return self._clean_punctuation(summary).capitalize()

        # Handle "As a X, I want Y." format (no purpose clause)
        if "i want" in story_lower:
            parts = story_lower.split("i want")[1].strip()
            # Take until period or end
            for sep in [".", "\n"]:
                if sep in parts:
                    summary = parts.split(sep)[0].strip()
                    return self._clean_punctuation(summary).capitalize()
            return self._clean_punctuation(parts).strip().capitalize()

        # Fallback: first 80 chars
        return story_clean[:80] + ("..." if len(story_clean) > 80 else "")

    def _clean_punctuation(self, text: str) -> str:
        """Remove trailing commas and clean punctuation from extracted summary."""
        text = text.strip()
        # Remove trailing commas, prepositions
        while text.endswith(",") or text.endswith(" to"):
            text = text[:-1].strip()
        return text

    def extract_id_label(self, id_value: Any) -> str:
        """Convert CSV id to JIRA label format (lowercase, no spaces)."""
        return str(id_value).lower().replace(" ", "-")

    def create_issue(
        self,
        project_key: str,
        story_text: str,
        issue_id: str,
        labels: List[str],
        priority: str = "Medium",
    ) -> Dict[str, Any]:
        """
        Create a JIRA issue via REST API.
        Returns: dict with status, issue_key, url, message
        """
        priority_map = {"Lowest": 5, "Low": 4, "Medium": 3, "High": 2, "Highest": 1}
        priority_id = priority_map.get(priority, 3)

        summary = self.extract_summary_from_story(story_text)

        payload = {
            "fields": {
                "project": {"key": project_key},
                "issuetype": {"name": "Story"},
                "summary": summary,
                "description": {
                    "version": 1,
                    "type": "doc",
                    "content": [
                        {
                            "type": "paragraph",
                            "content": [{"type": "text", "text": story_text}],
                        }
                    ],
                },
                "labels": labels,
                "priority": {"id": str(priority_id)},
            }
        }

        try:
            resp = requests.post(
                f"{self.base_url}/rest/api/3/issue",
                json=payload,
                auth=self.auth,
                headers=self.headers,
                timeout=15,
            )

            if resp.status_code == 201:
                data = resp.json()
                issue_key = data.get("key", "UNKNOWN")
                return {
                    "status": "success",
                    "issue_key": issue_key,
                    "url": f"{self.base_url}/browse/{issue_key}",
                    "message": summary,
                }
            elif resp.status_code == 400:
                error_msg = resp.json().get("errorMessages", ["Invalid request"])[0]
                return {
                    "status": "error",
                    "issue_key": None,
                    "message": f"Validation error: {error_msg}",
                }
            elif resp.status_code == 401:
                raise JiraSeedCSVError("JIRA authentication failed (401)")
            else:
                return {
                    "status": "error",
                    "issue_key": None,
                    "message": f"JIRA API error {resp.status_code}: {resp.text[:200]}",
                }

        except requests.RequestException as e:
            return {"status": "error", "issue_key": None, "message": f"Request failed: {e}"}

    def seed_from_csv(
        self,
        csv_path: str,
        project_key: str,
        extra_labels: Optional[List[str]] = None,
        dry_run: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Read CSV and create JIRA issues.
        Returns: list of results for each story.
        """
        if not os.path.exists(csv_path):
            raise JiraSeedCSVError(f"CSV file not found: {csv_path}")

        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            raise JiraSeedCSVError(f"Error reading CSV: {e}")

        required_cols = {"id", "story"}
        if not required_cols.issubset(df.columns):
            missing = required_cols - set(df.columns)
            raise JiraSeedCSVError(f"CSV missing required columns: {missing}")

        if len(df) == 0:
            raise JiraSeedCSVError("CSV has no data rows")

        base_labels = extra_labels if extra_labels else self.DEFAULT_LABELS
        results = []

        for idx, row in df.iterrows():
            issue_id = str(row["id"]).strip()
            story_text = str(row["story"]).strip()
            id_label = self.extract_id_label(issue_id)
            labels = base_labels + [id_label]

            if not story_text or story_text.lower() == "nan":
                results.append(
                    {
                        "status": "skipped",
                        "issue_key": issue_id,
                        "message": "Empty story text",
                    }
                )
                continue

            summary = self.extract_summary_from_story(story_text)
            print(f"[{'DRY RUN' if dry_run else 'CREATING'}] {issue_id}: {summary}")

            if not dry_run:
                result = self.create_issue(
                    project_key=project_key,
                    story_text=story_text,
                    issue_id=issue_id,
                    labels=labels,
                )
                results.append(result)
            else:
                results.append(
                    {
                        "status": "dry-run",
                        "issue_key": issue_id,
                        "message": f"Would create: {summary}",
                        "url": f"(dry-run, no URL)",
                    }
                )

        return results


def print_summary(results: List[Dict[str, Any]], dry_run: bool) -> None:
    """Print formatted summary of seed results."""
    success = sum(1 for r in results if r["status"] == "success")
    errors = sum(1 for r in results if r["status"] == "error")
    skipped = sum(1 for r in results if r["status"] == "skipped")
    dry = sum(1 for r in results if r["status"] == "dry-run")

    print("\n" + "=" * 60)
    if dry_run:
        print("DRY RUN Summary")
    else:
        print("SEED RESULTS")
    print("=" * 60)

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
    if dry_run:
        print(f"Summary: {len(results)} issues ready to seed (dry run)")
        print("💡 Run without --dry-run to create issues in JIRA")
    else:
        print(f"Summary: {success} created, {errors} errors, {skipped} skipped")
        if success == len(results):
            print("🎉 All stories seeded successfully!")
    print("=" * 60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Create JIRA stories from assets/StoryExample.csv"
    )
    parser.add_argument(
        "--project",
        required=True,
        help="JIRA project key (e.g., KAN)",
    )
    parser.add_argument(
        "--csv",
        default="assets/StoryExample.csv",
        help="Path to CSV file (default: assets/StoryExample.csv)",
    )
    parser.add_argument(
        "--labels",
        default=None,
        help="Comma-separated extra labels (default: ai-seed,fintech)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview what would be created without making API calls",
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

    extra_labels = None
    if args.labels:
        extra_labels = [l.strip() for l in args.labels.split(",") if l.strip()]

    try:
        client = JiraSeedCSVClient(jira_base_url, jira_email, jira_api_token)

        print(f"🔗 Testing JIRA connection to {jira_base_url}...")
        client.test_connection()
        print("✅ JIRA connection validated")

        print(f"\n📖 Reading stories from {args.csv}...")
        results = client.seed_from_csv(
            csv_path=args.csv,
            project_key=args.project,
            extra_labels=extra_labels,
            dry_run=args.dry_run,
        )

        print_summary(results, args.dry_run)

        sys.exit(0 if all(r["status"] in ("success", "skipped", "dry-run") for r in results) else 1)

    except JiraSeedCSVError as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n❌ Operation cancelled by user")
        sys.exit(130)
    except Exception as e:
        print(f"❌ Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
