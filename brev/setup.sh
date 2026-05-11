#!/usr/bin/env bash
# Run once on the Brev instance after the repo is uploaded. Idempotent.
# Defaults to GDELT-only minimal install. Set FULL=1 to install the entire
# pipeline (torch, torch-geometric, hmmlearn, etc.).
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

pip install --upgrade pip

if [[ "${FULL:-0}" == "1" ]]; then
  pip install -r requirements.txt
else
  pip install requests pandas pyarrow
fi

echo
echo "Setup complete. Activate with: source .venv/bin/activate"
