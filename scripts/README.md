# JIRA Seed Script

Utility to create bulk test issues in JIRA from a JSON template without using LLM API tokens.

## Features

✅ Create issues directly from JSON template  
✅ No LLM overhead (no token usage)  
✅ Dry-run mode to preview changes  
✅ Label tickets with `ai-seed` for easy filtering  
✅ Full error handling and logging  
✅ Run manually or via GitHub Actions/cron  

## Setup

### 1. Ensure `.env` has JIRA credentials:

```bash
JIRA_BASE_URL=https://your-domain.atlassian.net
JIRA_EMAIL=your-email@example.com
JIRA_API_TOKEN=your-api-token-from-https://id.atlassian.com/manage-profile/security/api-tokens
```

### 2. Create or edit template file:

See [`assets/jira-seed-template.json`](../assets/jira-seed-template.json) for the expected format.

Template structure:
```json
{
  "project_key": "SAAS",
  "issue_type": "Story",
  "labels": ["ai-seed", "saas"],
  "tickets": [
    {
      "key": "SIGNUP-001",
      "summary": "User Signup with Email and Password",
      "description": "As a new user...",
      "acceptance_criteria": [...],
      "priority": "High",
      "reporter": "QA Lead"
    }
  ]
}
```

## Usage

### Dry Run (Preview)
```bash
python scripts/seed_jira.py --dry-run
```

Output:
```
🔗 Testing JIRA connection to https://your-domain.atlassian.net...
✅ JIRA connection successful

📋 Loading template from assets/jira-seed-template.json...

============================================================
SEED RESULTS
============================================================
🔄 DRY-RUN   | SIGNUP-001   | Would create User Signup with Email and Password
🔄 DRY-RUN   | SIGNUP-002   | Would create User Signup with OAuth (Google/GitHub)

============================================================
Summary: 0 created, 0 errors, 0 skipped, 5 dry-run
============================================================

💡 This was a DRY RUN. Run without --dry-run to create issues.
```

### Create Issues
```bash
python scripts/seed_jira.py
```

Output:
```
============================================================
SEED RESULTS
============================================================
✅ SUCCESS   | SAAS-123     | Created SAAS-123
           → https://your-domain.atlassian.net/browse/SAAS-123
✅ SUCCESS   | SAAS-124     | Created SAAS-124
           → https://your-domain.atlassian.net/browse/SAAS-124

============================================================
Summary: 5 created, 0 errors, 0 skipped, 0 dry-run
============================================================
```

### Custom Template
```bash
python scripts/seed_jira.py --template assets/my-custom-template.json
```

## Filter Seeded Issues in JIRA

All seeded issues have the label `ai-seed`. In JIRA, use:

```
labels = ai-seed
```

Or delete them all (if needed):

```
labels = ai-seed AND type = Story
```

## GitHub Actions Setup (Optional)

Create `.github/workflows/seed-jira.yml`:

```yaml
name: Seed JIRA Issues

on:
  workflow_dispatch:  # Manual trigger
  schedule:
    - cron: '0 0 1 * *'  # Monthly, first day at 00:00 UTC

jobs:
  seed:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
      
      - name: Seed JIRA
        env:
          JIRA_BASE_URL: ${{ secrets.JIRA_BASE_URL }}
          JIRA_EMAIL: ${{ secrets.JIRA_EMAIL }}
          JIRA_API_TOKEN: ${{ secrets.JIRA_API_TOKEN }}
        run: python scripts/seed_jira.py
```

Then add secrets to your GitHub repo (Settings → Secrets and variables → Actions):
- `JIRA_BASE_URL`
- `JIRA_EMAIL`
- `JIRA_API_TOKEN`

Now you can trigger seeds manually from GitHub Actions or on schedule!

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `401 Unauthorized` | Check JIRA email and API token in `.env` |
| `404 Not Found` | Verify JIRA base URL (should be `https://domain.atlassian.net`) |
| `400 Bad Request` | Ensure project key exists and issue type is valid |
| `Missing project_key` | Add `"project_key"` to template JSON |

## Notes

- **No LLM overhead**: This script creates static issues from a template, saving API tokens
- **Dry-run first**: Always use `--dry-run` before creating issues
- **Bulk operations**: Perfect for seeding test data without manual JIRA UI clicks
- **Idempotent**: Can rerun multiple times (existing issues won't be duplicated unless you manually delete and reseed)

