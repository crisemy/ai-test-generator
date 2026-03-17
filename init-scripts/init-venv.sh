#!/usr/bin/env bash

# ──────────────────────────────────────────────
#  start-venv.sh
#  Creates .venv if missing → activates it
#  Usage:  source ./start-venv.sh OR . ./start-venv.sh 
#  Important: Do not use ./start-venv.sh or bash start-venv.sh because that will run in a subshell 
#  and won't activate the venv in your current shell.
# ──────────────────────────────────────────────

VENV_DIR=".venv"
PYTHON=${PYTHON:-python}   # you can override with: PYTHON=python start-venv.sh

echo ""
echo "Python virtual environment helper"
echo "─────────────────────────────────"

# 1. Create if missing
if [[ ! -d "$VENV_DIR" ]]; then
    echo "→ Creating virtual environment: $VENV_DIR"
    echo "   Command: $PYTHON -m venv $VENV_DIR"
    echo ""

    if ! $PYTHON -m venv "$VENV_DIR"; then
        echo "Error: Failed to create virtualenv."
        echo "Try setting PYTHON=python or PYTHON=python3.11 etc."
        exit 1
    fi

    echo "→ Virtual environment created."
else
    echo "→ Virtual environment already exists → $VENV_DIR"
fi

# 2. Activate — this only works when you SOURCE the script
echo ""
echo "Activating virtual environment..."

if [[ -f "$VENV_DIR/bin/activate" ]]; then
    # Linux / macOS / Git Bash (Unix-style)
    source "$VENV_DIR/bin/activate"
elif [[ -f "$VENV_DIR/Scripts/activate" ]]; then
    # Git Bash on Windows sometimes prefers this
    source "$VENV_DIR/Scripts/activate"
else
    echo "❌ Cannot find activation script."
    echo "   Looked for:"
    echo "     $VENV_DIR/bin/activate"
    echo "     $VENV_DIR/Scripts/activate"
    exit 1
fi

# 3. Quick success feedback
if [[ -n "${VIRTUAL_ENV:-}" ]]; then
    echo "→ Activated!  (VIRTUAL_ENV = $VIRTUAL_ENV)"
    python --version | head -n 1
    echo ""
    echo "You are now inside the virtual environment."
    echo "Install packages with:  pip install ..."
    echo "Deactivate later with:  deactivate"
else
    echo "Warning: Activation may have failed."
    echo "Make sure you ran:   source ./start-venv.sh"
    echo "(not ./start-venv.sh or bash start-venv.sh)"
fi

echo ""