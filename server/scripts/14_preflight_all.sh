#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
AI_ROOT="${AI_ROOT:-/mnt/ai-storage/aigc_data}"
ENV_DIR="${ENV_DIR:-$HOME/.venvs/aigc-detector}"
source "$ENV_DIR/bin/activate"
python - <<'PY'
import importlib.util, sys, torch
required=['torch','torchvision','transformers','timm','albumentations','cv2','sklearn','PIL','scipy','pywt','einops','accelerate','datasets','huggingface_hub','safetensors']
missing=[x for x in required if importlib.util.find_spec(x) is None]
print('Python:',sys.version)
print('Torch:',torch.__version__)
print('CUDA available:',torch.cuda.is_available())
if torch.cuda.is_available():
 print('GPU:',torch.cuda.get_device_name(0),'VRAM GB:',round(torch.cuda.get_device_properties(0).total_memory/2**30,2))
if missing:
 print('MISSING:',missing); raise SystemExit(2)
print('Core dependencies: OK')
PY
bash "$ROOT/server/scripts/10_storage_guard.sh"
