#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Task 1.1 — Step 01: Download the Arnaiz-Rodriguez dataset
# Owner: Eman (P1)
# Run from repo root: bash scripts/task1_1/step01_download_arnaiz.sh
# ─────────────────────────────────────────────────────────────────────────────

set -e   # stop on any error

DEST="data/raw/arnaiz-rodriguez"

if [ -d "$DEST" ]; then
  echo "[step01] Folder $DEST already exists. Skipping clone."
  echo "         Delete it and re-run if you want a fresh copy."
else
  echo "[step01] Cloning Arnaiz-Rodriguez repository..."
  git clone https://github.com/ellisalicante/LLMs-Mental-Health-Crisis "$DEST"
  echo "[step01] Done. Files downloaded to $DEST"
fi

echo ""
echo "[step01] Contents of $DEST:"
ls -lh "$DEST"

echo ""
echo "[step01] ✓ Next step: python scripts/task1_1/step02_inspect_arnaiz.py"
