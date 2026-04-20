#!/usr/bin/env bash
set -euo pipefail

cd /workspaces/condlanenet

# Change cu102 -> cu111 to support RTX 3060 (Ampere)
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html

# requirements.txt already pins torch/torchvision; skip those to avoid reinstall conflicts.
REQ_NO_TORCH_FILE="/workspaces/condlanenet/requirements.no_torch.txt"
awk '!/^(torch|torchvision)==/' requirements.txt > "${REQ_NO_TORCH_FILE}"
python -m pip install -r "${REQ_NO_TORCH_FILE}"
rm -f "${REQ_NO_TORCH_FILE}"

# Necessary for building the C++/CUDA extensions
python -m pip install --no-cache-dir \
  "pip==23.2.1" \
  "setuptools==65.7.0" \
  "packaging==23.2" \
  "wheel==0.41.2"

# This compiles the C++/CUDA extensions of your lane detection repo
python setup.py build develop
