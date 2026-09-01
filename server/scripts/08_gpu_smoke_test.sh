#!/usr/bin/env bash
set -euo pipefail
ENV_DIR="${ENV_DIR:-$HOME/.venvs/aigc-detector}"
source "$ENV_DIR/bin/activate"
python - <<'PY'
import torch
assert torch.cuda.is_available(), 'CUDA is unavailable'
name=torch.cuda.get_device_name(0)
free,total=torch.cuda.mem_get_info()
print('GPU:', name)
print('CUDA:', torch.version.cuda)
print(f'VRAM free: {free/1024**3:.2f} GiB / {total/1024**3:.2f} GiB')
# Tiny matmul confirms actual CUDA execution.
a=torch.randn((2048,2048),device='cuda',dtype=torch.float16)
b=torch.randn((2048,2048),device='cuda',dtype=torch.float16)
c=a@b
torch.cuda.synchronize()
print('CUDA matmul OK:', tuple(c.shape))
PY
