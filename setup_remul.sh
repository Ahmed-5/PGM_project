#!/usr/bin/env bash
# Install dependencies and download REMUL paper datasets.
#
# Usage:
#   bash setup_remul.sh          # install + download all datasets
#   bash setup_remul.sh --skip-download   # install only
set -e

SKIP_DOWNLOAD=0
for arg in "$@"; do
  [ "$arg" = "--skip-download" ] && SKIP_DOWNLOAD=1
done

echo "=== REMUL setup (arXiv:2410.17878) ==="

# Create venv if missing
if [ ! -d .venv ]; then
  echo "[1/3] Creating virtual environment..."
  python3 -m venv .venv
else
  echo "[1/3] Virtual environment already exists."
fi
source .venv/bin/activate

echo "[2/3] Installing dependencies..."
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt
# torch_scatter often needs --no-build-isolation on some platforms
pip install --quiet --no-build-isolation torch_scatter 2>/dev/null || \
  pip install --quiet torch_scatter 2>/dev/null || \
  echo "Warning: torch_scatter install failed; some features may not work."

if [ "$SKIP_DOWNLOAD" = "0" ]; then
  echo "[3/3] Downloading datasets (MD17 × 8 + CMU MoCap 35 & 9)..."
  python -m remul.download 2>&1 | tee outputs/remul_download.log
else
  echo "[3/3] Skipping dataset download (--skip-download)."
fi

echo ""
echo "=== Setup complete ==="
echo "  Datasets:  data/remul/  (MD17 ~3.3 GB, MoCap ~18 MB; N-body is synthetic)"
echo "  Smoke test: SMOKE=1 bash remul/run_experiments.sh"
echo "  Full runs:  DEVICE=cuda bash remul/run_experiments.sh"
echo "  See remul/README.md for single-experiment examples."
