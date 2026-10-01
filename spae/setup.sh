#!/bin/bash
# One-command environment: a virtual environment with the pinned versions used for the paper (CUDA 13 build of PyTorch).
# Usage: ./setup.sh            (creates .venv next to this script; then: source .venv/bin/activate)
#        CUDA=cu126 ./setup.sh (another PyTorch build: cu126, cu128, cu130 or cpu)
#        PYTHON=python3.10 ./setup.sh
set -e
cd "$(dirname "$0")"
PY=${PYTHON:-$(command -v python3.10 || command -v python3)}
CUDA=${CUDA:-cu130}
"$PY" -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install "torch==2.11.0" --index-url "https://download.pytorch.org/whl/$CUDA"
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -c "import torch, transformers, pastalib; print('torch', torch.__version__, 'transformers', transformers.__version__, 'cuda', torch.cuda.is_available())"
echo "environment ready: source $(pwd)/.venv/bin/activate"
