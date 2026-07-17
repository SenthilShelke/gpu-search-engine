#!/bin/bash
# Assembles the native/ directory that gets published inside the npm
# package: the compiled Metal search binary + its metallib + the prebuilt
# corpus data. Run this on macOS (Apple Silicon) before `npm publish`.
#
# This does NOT rebuild the corpus or embeddings from Hugging Face -- run
# `python scripts/build_vector_db.py` first (from repo root, with the venv
# active and HF_TOKEN set in .env) if you want a fresh corpus. This script
# just packages whatever is currently in data/.

set -e

echo "[release] Checking arch..."
if [ "$(uname -m)" != "arm64" ]; then
  echo "This script must be run on Apple Silicon (arm64) to produce the arm64 binary this package ships." >&2
  exit 1
fi

if [ ! -f "data/vector_db.bin" ] || [ ! -f "data/corpus.json" ]; then
  echo "Missing data/vector_db.bin or data/corpus.json." >&2
  echo "Run: python scripts/build_vector_db.py (from repo root) first." >&2
  exit 1
fi

echo "[release] Building Metal library..."
xcrun -sdk macosx metal -c src/compute.metal -o compute.air
xcrun -sdk macosx metallib compute.air -o default.metallib

echo "[release] Building C++ engine..."
clang++ -std=c++17 -O2 -framework Metal -framework Foundation src/main.mm -o main

echo "[release] Assembling native/ package contents..."
rm -rf native
mkdir -p native/data
cp main native/main-darwin-arm64
chmod +x native/main-darwin-arm64
cp default.metallib native/default.metallib
cp data/vector_db.bin native/data/vector_db.bin
cp data/corpus.json native/data/corpus.json

echo "[release] Done. native/ is ready:"
ls -la native native/data

echo "[release] Next steps: bump version in package.json, then run 'npm publish'."
