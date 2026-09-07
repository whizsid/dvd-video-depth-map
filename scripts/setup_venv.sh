#!/usr/bin/env bash
# Create and populate the project virtualenv (macOS / Apple Silicon).
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
pip install -e "$ROOT/vendor/DVD" --no-deps

# DA3: xformers is optional on Mac; install package without pulling it in.
pip install "git+https://github.com/ByteDance-Seed/Depth-Anything-3.git" --no-deps
pip install moviepy==1.0.3 e3nn omegaconf typer plyfile trimesh open3d evo pillow_heif pycolmap

echo
echo "Done. Activate with:"
echo "  source $ROOT/.venv/bin/activate"
echo
echo "If you hit OpenMP errors (libomp), run:"
echo "  export KMP_DUPLICATE_LIB_OK=TRUE"
