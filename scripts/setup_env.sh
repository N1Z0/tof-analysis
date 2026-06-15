#!/usr/bin/env bash
# One-time setup for TOF Analysis notebooks.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "→ Creating virtual environment…"
python3 -m venv .venv
source .venv/bin/activate

echo "→ Installing package…"
pip install --upgrade pip -q
pip install -e . -q

echo "→ Registering Jupyter kernel…"
python -m ipykernel install --user --name=tof-analysis --display-name="Python (TOF Analysis)"

echo ""
echo "✓ Ready. Activate and open a notebook:"
echo ""
echo "  source .venv/bin/activate"
echo "  jupyter notebook notebooks/calibration.ipynb            # fit & save"
echo "  jupyter notebook notebooks/analysis.ipynb               # plot spectra"
echo "  jupyter notebook notebooks/gas_45_dwell_comparison.ipynb  # dwell overlay"
echo ""
echo "  In Cursor: open notebook → kernel 'Python (TOF Analysis)'"
