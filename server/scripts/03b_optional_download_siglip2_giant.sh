#!/usr/bin/env bash
set -euo pipefail
AI_ROOT="${AI_ROOT:-/mnt/ai-storage/aigc_data}"
MODEL_ROOT="$AI_ROOT/models/siglip2_giant_384"
mkdir -p "$MODEL_ROOT"
source "${ENV_DIR:-$HOME/.venvs/aigc-detector}/bin/activate"
python - <<'PY'
from transformers import AutoConfig
from pathlib import Path
import json
model_id='google/siglip2-giant-opt-patch16-384'
cfg=AutoConfig.from_pretrained(model_id)
print('Loaded config:', cfg.__class__.__name__)
print('vision_config:', cfg.vision_config)
# Do not treat a published "1B" family label as proof of the full checkpoint count.
# The final submission must count the exact instantiated model parameters.
PY
hf download google/siglip2-giant-opt-patch16-384 --local-dir "$MODEL_ROOT" --max-workers 4
cat > "$MODEL_ROOT/COMPLIANCE_WARNING.txt" <<'TXT'
EXPERIMENTAL ONLY.
Before using this checkpoint in a final submission, instantiate the exact model and count
all parameters in the full checkpoint. The hackathon requires <2,000,000,000 parameters.
TXT
