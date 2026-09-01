#!/usr/bin/env bash
set -euo pipefail

# Optional Fedora host packages. Run only if the machine is missing them.
# This script is intentionally separate from the Python environment setup.

if ! command -v sudo >/dev/null 2>&1; then
  echo 'sudo not available; install the listed Fedora packages as an administrator.'
  exit 0
fi

sudo dnf install -y git git-lfs rsync wget curl unzip tar gcc gcc-c++ make cmake pkg-config

git lfs install || true

echo 'Fedora host prerequisites installed.'
echo 'NVIDIA driver/CUDA toolkit is not modified by this script.'
echo 'The PyTorch wheel supplies the user-space CUDA runtime.'
