#!/usr/bin/env bash
# Double-click on macOS (as start.command) or run `./start.sh` on Linux.
# Mirrors start.bat's first-run venv setup, then hands off to launch.py.
set -e
cd "$(dirname "$0")"

echo "ZivaWhite -- starting..."

PYCMD=""
for candidate in python3 python; do
    if command -v "$candidate" >/dev/null 2>&1; then
        PYCMD="$candidate"
        break
    fi
done

if [ -z "$PYCMD" ]; then
    echo
    echo "Python was not found. Install Python 3.10+ (python.org, or your"
    echo "system package manager), then run this script again."
    echo
    read -p "Press enter to close..."
    exit 1
fi

if [ ! -f ".venv/bin/python" ]; then
    echo "First run: creating a virtual environment and installing dependencies..."
    echo "This happens once and can take a few minutes."
    "$PYCMD" -m venv .venv
    ".venv/bin/python" -m pip install --upgrade pip --quiet
    ".venv/bin/python" -m pip install -r requirements.txt --quiet
    ".venv/bin/python" -m pip install -e . --quiet
fi

".venv/bin/python" launch.py

echo
echo "ZivaWhite stopped."
read -p "Press enter to close..."
