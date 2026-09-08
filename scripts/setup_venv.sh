#!/usr/bin/env bash
# Create and populate the project virtualenv.
# Video DepthCrafter inference is CUDA-only (Colab T4 oriented).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-python3.10}"

if ! command -v "$PY" >/dev/null 2>&1; then
  echo "Python not found: $PY (set PYTHON=... or install python3.10)" >&2
  exit 1
fi

echo "Using $("$PY" --version) at $(command -v "$PY")"
"$PY" -m venv "$ROOT/.venv"
# shellcheck disable=SC1091
source "$ROOT/.venv/bin/activate"

pip install --upgrade pip setuptools wheel
pip install -r "$ROOT/requirements.txt"

echo
echo "Done. Activate with:"
echo "  source $ROOT/.venv/bin/activate"
echo
echo "Video depth uses DepthCrafter on CUDA — see README / Colab notebook."
echo
echo "If you hit OpenMP errors (libomp), run:"
echo "  export KMP_DUPLICATE_LIB_OK=TRUE"
