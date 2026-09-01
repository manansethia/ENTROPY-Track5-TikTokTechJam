#!/usr/bin/env bash
set -euo pipefail

AI_ROOT="${AI_ROOT:-/mnt/ai-storage/aigc_data}"
ENV_DIR="${ENV_DIR:-$HOME/.venvs/aigc-detector}"
source "$ENV_DIR/bin/activate"

# This stage is intentionally a harness, not a fake claim of results.
# Implement adapters for each candidate and evaluate ONLY on allowed non-training
# probe sets first. The final challenge benchmark stays locked.

OUT="$AI_ROOT/benchmarks/model_pool"
mkdir -p "$OUT"

python - <<PY
from pathlib import Path
import json, os
root=Path('$AI_ROOT/models')
out=Path('$OUT/model_inventory.json')
rows=[]
for p in sorted(root.iterdir() if root.exists() else []):
    if p.is_dir():
        files=list(p.rglob('*'))
        size=sum(x.stat().st_size for x in files if x.is_file())
        rows.append({'name':p.name,'path':str(p),'bytes':size})
out.write_text(json.dumps(rows,indent=2))
print(out)
PY

echo
 echo 'Model inventory written.'
echo 'Next implementation target: adapters/benchmark_candidates.py'
echo 'Do NOT publish or treat these numbers as hackathon results until the locked benchmark is run exactly once after model selection.'
