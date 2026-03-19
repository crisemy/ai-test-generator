#!/usr/bin/env bash
# init-project.sh
# Simple project initializer — create new or clone existing repo
# Usage: ./init-project.sh

set -euo pipefail

cat << 'EOF'
┌──────────────────────────────────────────────────┐
│             Project Initializer                  │
└──────────────────────────────────────────────────┘

EOF

# ──────────────────────────────────────────────
#  Ask: clone or create new?
# ──────────────────────────────────────────────

read -p "Do you want to CLONE an existing repository? (y/n): " -n 1 -r clone_existing
echo ""

if [[ $clone_existing =~ ^[Yy]$ ]]; then
    echo ""
    read -p "Paste the repository URL (https or git@): " repo_url
    
    if [[ -z "$repo_url" ]]; then
        echo "✗ No URL provided. Aborting."
        exit 1
    fi

    # Guess folder name from repo
    folder_name=$(basename "$repo_url" .git)

    read -p "Folder name to clone into (Enter = $folder_name): " folder
    folder=${folder:-$folder_name}

    echo ""
    echo "Cloning $repo_url → ./$folder"
    git clone "$repo_url" "$folder"
    
    if [[ $? -eq 0 ]]; then
        cd "$folder" || exit 1
        echo ""
        echo "✓ Repository cloned successfully"
        echo "   → current directory: $(pwd)"
        echo ""
        exit 0
    else
        echo "✗ Cloning failed."
        exit 1
    fi
fi

# ──────────────────────────────────────────────
#  Create NEW project
# ──────────────────────────────────────────────

echo ""
read -p "Project name (used for folder & README): " project_name

# Basic sanitization
project_name=$(echo "$project_name" | sed 's/[^a-zA-Z0-9_-]/-/g' | sed 's/-+/-/g' | sed 's/^-*//;s/-*$//')

if [[ -z "$project_name" ]]; then
    echo "✗ You must provide a project name. Aborting."
    exit 1
fi

folder_name="$project_name"

if [[ -d "$folder_name" ]]; then
    echo "✗ Folder '$folder_name' already exists. Aborting to avoid overwrite."
    exit 1
fi

echo ""
echo "Creating new project: $project_name"
mkdir "$folder_name"
cd "$folder_name" || exit 1

# ──────────────────────────────────────────────
#  Create basic README
# ──────────────────────────────────────────────
cat > README.md << 'EOF'
# PROJECT_NAME

> Short description of the project

## Main technologies

- 
- 
- 

## Quick start

```bash

