#!/usr/bin/env bash
# Set up a local virtual environment and install dependencies.
#
# Usage: ./setup.sh

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

REQUIRED_MAJOR=3
REQUIRED_MINOR=8
VENV_DIR="venv"

if ! command -v python3 >/dev/null 2>&1; then
    echo "Error: python3 not found on PATH." >&2
    exit 1
fi

PY_VERSION=$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')
PY_MAJOR=$(python3 -c 'import sys; print(sys.version_info[0])')
PY_MINOR=$(python3 -c 'import sys; print(sys.version_info[1])')
if [ "$PY_MAJOR" -lt "$REQUIRED_MAJOR" ] || { [ "$PY_MAJOR" -eq "$REQUIRED_MAJOR" ] && [ "$PY_MINOR" -lt "$REQUIRED_MINOR" ]; }; then
    echo "Error: Python ${REQUIRED_MAJOR}.${REQUIRED_MINOR}+ required, found ${PY_VERSION}." >&2
    exit 1
fi

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment in ./${VENV_DIR} ..."
    python3 -m venv "$VENV_DIR"
else
    echo "Virtual environment ./${VENV_DIR} already exists; reusing it."
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

pip install --upgrade pip
pip install -r requirements.txt

echo
echo "Setup complete. Activate the environment with:"
echo "    source ${VENV_DIR}/bin/activate"
echo
echo "Then run the pipeline, e.g.:"
echo "    python src/pipeline.py --validate"
