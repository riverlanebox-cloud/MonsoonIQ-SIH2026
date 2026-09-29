#!/usr/bin/env bash
# Vercel build: export the static API snapshot with the trained models, then build
# the console against it. vercel.json runs this; see docs/DEPLOY.md.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=python3
for cand in python3.13 python3.12 python3.11; do
  if command -v "$cand" >/dev/null 2>&1; then PY="$cand"; break; fi
done
echo "python: $($PY --version)"

# Install into a local directory so the build never depends on a writable system site.
DEPS="$PWD/.vercel-pydeps"
if command -v uv >/dev/null 2>&1; then
  uv pip install --python "$PY" --target "$DEPS" -r requirements-vercel.txt
else
  $PY -m pip --version >/dev/null 2>&1 || $PY -m ensurepip --user
  $PY -m pip install --disable-pip-version-check --target "$DEPS" -r requirements-vercel.txt
fi

export PYTHONPATH="$PWD:$DEPS"
export MONSOONIQ_GBM=sklearn   # the committed models use the scikit-learn backend
$PY scripts/export_static.py --out frontend/public/snapshot

cd frontend
VITE_STATIC=1 npm run build
