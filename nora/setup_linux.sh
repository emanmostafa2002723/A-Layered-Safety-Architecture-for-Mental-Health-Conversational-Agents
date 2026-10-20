#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHON_BIN="${PYTHON_BIN:-python3}"
"$PYTHON_BIN" -c 'import sys; assert sys.version_info[:2] == (3,12), "Use Python 3.12"'
if [ -e .venv-api ]; then
  echo 'Existing .venv-api preserved. No environment was changed.'
  exit 2
fi
# Reuse compatible system-installed packages without copying CUDA wheels again.
# Do not upgrade this environment blindly: inspect before installing missing deps.
"$PYTHON_BIN" -m venv --system-site-packages .venv-api
echo 'Created project-local .venv-api using existing system packages.'
echo 'Next: .venv-api/bin/python scripts/check_dependencies.py'
echo 'No packages or model weights were downloaded.'
