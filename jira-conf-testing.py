import requests
from dotenv import load_dotenv
import os
import json

load_dotenv()

JIRA_API_URL = os.getenv("JIRA_API_URL")      # https://crisemy.atlassian.net
JIRA_EMAIL = os.getenv("JIRA_EMAIL")
JIRA_API_TOKEN = os.getenv("JIRA_API_TOKEN")

headers = {
    "Accept": "application/json",
    "Content-Type": "application/json"
}

# JQL enfocado en User Stories del proyecto KAN
jql = 'project = KAN AND issuetype = "User Story" ORDER BY created DESC'

url = f"{JIRA_API_URL}/rest/api/3/search/jql"

payload = {
    "jql": jql,
    "maxResults": 20,
    "fields": ["summary", "status", "description", "created", "updated", "assignee"]
}

response = requests.post(url, auth=(JIRA_EMAIL, JIRA_API_TOKEN), headers=headers, json=payload)

if response.status_code == 200:
    data = response.json()
    total = data.get('total', 0)
    print(f"✅ Conexión OK - Se encontraron {total} User Stories en el proyecto KAN\n")
    
    if total > 0:
        print("User Stories encontradas:")
        for issue in data.get('issues', []):
            key = issue['key']
            fields = issue['fields']
            summary = fields.get('summary', 'Sin título')
            status = fields.get('status', {}).get('name', 'Sin estado')
            print(f"   • {key} | {status} | {summary}")
    else:
        print("No hay User Stories todavía en KAN. Es normal si el proyecto está vacío.")
        print("Crea al menos una User Story desde la interfaz web para probar.")
        
else:
    print(f"❌ Error {response.status_code}")
    print(response.text)