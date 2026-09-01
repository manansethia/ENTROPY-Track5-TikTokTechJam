#!/usr/bin/env bash
set -euo pipefail
# Requires sudo. Run only if needed; NVIDIA driver is assumed already installed.
sudo dnf install -y git git-lfs gcc gcc-c++ make cmake ninja-build pkg-config python3-devel \
  ffmpeg libjpeg-turbo-devel libpng-devel openssl-devel zlib-devel \
  rsync wget curl jq unzip tmux htop nvtop

git lfs install || true
