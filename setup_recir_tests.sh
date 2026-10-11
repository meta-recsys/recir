#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# (c) Meta Platforms, Inc. and affiliates.
#
# ============================================================================
# Setup script for recir test dependencies.
#
# Installs recir (editable) + test extras, clones and installs
# facebookresearch/generative-recommenders, and validates that
#   python -m unittest recir.tests.stu_test -v
# can run.
#
# Usage:
#   ./setup_recir_tests.sh
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VENV_DIR:-$SCRIPT_DIR/.venv}"
GENERATIVE_RECOMMENDERS_DIR="${GENERATIVE_RECOMMENDERS_DIR:-$HOME/generative-recommenders}"

echo "===================================================="
echo "   Recir Test Environment Setup"
echo "===================================================="
echo "  Venv:   $VENV_DIR"
echo "  Repo:   $GENERATIVE_RECOMMENDERS_DIR"
echo "===================================================="

# --- 1. Create or reuse a virtual environment ---
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
pip install --upgrade pip --quiet

# --- 2. Fix missing __init__.py in recir/tests ---
if [ ! -f "$SCRIPT_DIR/recir/tests/__init__.py" ]; then
    echo ""
    echo "--- Creating missing recir/tests/__init__.py ---"
    touch "$SCRIPT_DIR/recir/tests/__init__.py"
fi

# --- 3. Install recir with test extras ---
echo ""
echo "--- Installing recir with test extras ---"
pip install -e "${SCRIPT_DIR}[test]" --quiet
echo "  recir installed."

# --- 4. Clone and install generative_recommenders ---
echo ""
echo "--- Installing generative_recommenders ---"
if [ -d "$GENERATIVE_RECOMMENDERS_DIR/.git" ]; then
    echo "  Using existing $GENERATIVE_RECOMMENDERS_DIR"
else
    echo "  Cloning https://github.com/facebookresearch/generative-recommenders.git ..."
    git clone https://github.com/facebookresearch/generative-recommenders.git "$GENERATIVE_RECOMMENDERS_DIR"
fi

# Install generative_recommenders in editable mode without strict deps
# (fbgemm_gpu and torchrec are already installed above).
pip install -e "$GENERATIVE_RECOMMENDERS_DIR" --no-deps --quiet

# Install missing runtime deps that generative_recommenders needs
pip install gin-config matplotlib pybind11 --quiet

echo "  generative_recommenders installed."

# --- 5. Validation ---
echo ""
echo "--- Validating environment ---"
python3 -c "
import torch
print(f'  PyTorch:      {torch.__version__}')

import recir
print(f'  recir:        OK')

import generative_recommenders
from generative_recommenders.common import HammerKernel, set_dev_mode
from generative_recommenders.modules.stu import STU, STULayer, STULayerConfig, STUStack
print(f'  generative_recommenders: OK')
"

echo ""
echo "===================================================="
echo "   SETUP COMPLETE"
echo "===================================================="
echo ""
echo "You can now run the tests with:"
echo "  cd $SCRIPT_DIR"
echo "  python -m unittest recir.tests.stu_test -v"
echo ""
