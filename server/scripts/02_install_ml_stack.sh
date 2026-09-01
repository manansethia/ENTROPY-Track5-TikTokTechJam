#!/usr/bin/env bash
set -euo pipefail

ENV_DIR="${ENV_DIR:-$HOME/.venvs/aigc-detector}"
source "$ENV_DIR/bin/activate"

python -m pip install --upgrade pip

# PyTorch wheels ship their CUDA runtime; the host only needs a compatible NVIDIA driver.
DRIVER_MAJOR=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
if [[ "$DRIVER_MAJOR" =~ ^[0-9]+$ ]] && (( DRIVER_MAJOR >= 570 )); then
  TORCH_INDEX="https://download.pytorch.org/whl/cu128"
elif [[ "$DRIVER_MAJOR" =~ ^[0-9]+$ ]] && (( DRIVER_MAJOR >= 550 )); then
  TORCH_INDEX="https://download.pytorch.org/whl/cu126"
elif [[ "$DRIVER_MAJOR" =~ ^[0-9]+$ ]] && (( DRIVER_MAJOR >= 535 )); then
  TORCH_INDEX="https://download.pytorch.org/whl/cu124"
elif [[ "$DRIVER_MAJOR" =~ ^[0-9]+$ ]] && (( DRIVER_MAJOR >= 525 )); then
  TORCH_INDEX="https://download.pytorch.org/whl/cu121"
else
  echo "NVIDIA driver $DRIVER_MAJOR is too old for the supported CUDA wheels." >&2
  exit 1
fi

echo "Using PyTorch CUDA wheel index: $TORCH_INDEX"
pip install --upgrade torch torchvision torchaudio --index-url "$TORCH_INDEX"
pip install --upgrade \
  "transformers>=4.49" \
  "timm>=1.0" \
  "open_clip_torch>=2.30" \
  "albumentations>=2.0" \
  opencv-python-headless \
  scikit-learn \
  pandas \
  numpy \
  pillow \
  scipy \
  pywavelets \
  einops \
  accelerate \
  safetensors \
  bitsandbytes \
  matplotlib \
  seaborn \
  rich \
  psutil \
  datasets \
  huggingface_hub \
  modelscope

python - <<'PY'
import torch
print('torch:', torch.__version__)
print('cuda available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('gpu:', torch.cuda.get_device_name(0))
    print('capability:', torch.cuda.get_device_capability(0))
PY
