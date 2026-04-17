#!/usr/bin/env bash
# setup_conda.sh — End-to-end conda environment setup for VHH Biosimilar Library Generator
set -euo pipefail

echo "==> Creating conda environment from environment.yml ..."
conda env create -f environment.yml

echo "==> Activating conda environment 'vhh_biosimilar' ..."
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vhh_biosimilar

echo "==> Installing package in editable mode ..."
pip install -e .

echo "==> Downloading AbNatiV model weights ..."
vhh-init

echo ""
echo "Setup complete!  Activate the environment with:"
echo "  conda activate vhh_biosimilar"
