#!/usr/bin/env bash
set -euo pipefail

# CUDA 12.1 environment matching the project design.
conda create -n aigc_forensics python=3.10 -y || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate aigc_forensics

python -m pip install --upgrade pip
python -m pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt

mkdir -p checkpoints logs data/train/real data/train/synthetic   data/val_demo/coco_val2017 data/val_demo/dalle_advanced

echo "Environment ready."
