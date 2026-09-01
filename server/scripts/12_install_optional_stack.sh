#!/usr/bin/env bash
set -euo pipefail
ENV_DIR="${ENV_DIR:-$HOME/.venvs/aigc-detector}"
source "$ENV_DIR/bin/activate"

# Install only after the core CUDA/PyTorch smoke test passes.
pip install --upgrade accelerate lightning
# DeepSpeed is optional on a single 6GB GPU; install only if a training experiment needs ZeRO/NVMe offload.
pip install --upgrade deepspeed
# OverflowML is an optional memory strategy/orchestration layer; keep it isolated from core model code.
pip install --upgrade overflowml
# Unsloth is primarily LLM/transformer fine-tuning oriented. Do not make it a dependency of the vision pipeline.
python - <<'PY'
mods=['accelerate','lightning','deepspeed','overflowml']
for m in mods:
    try:
        mod=__import__(m)
        print(m, getattr(mod,'__version__','installed'))
    except Exception as e:
        print(m, 'IMPORT FAILED:', e)
PY
