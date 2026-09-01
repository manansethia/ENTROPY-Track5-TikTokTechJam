#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
AI_ROOT="${AI_ROOT:-/mnt/ai-storage/aigc_data}"
MODEL_ROOT="$AI_ROOT/models"
HF_HOME="$AI_ROOT/hf_cache"
export HF_HOME
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-300}"

# Load credentials from .env if available
if [[ -f "$PROJECT_ROOT/.env" ]]; then
  set -a
  source "$PROJECT_ROOT/.env"
  set +a
fi

if [[ -n "${HF_TOKEN:-}" ]]; then
  export HF_TOKEN
  echo "Hugging Face authentication token configured."
fi

mkdir -p "$MODEL_ROOT" "$HF_HOME" "$AI_ROOT/manifests"
source "${ENV_DIR:-$HOME/.venvs/aigc-detector}/bin/activate"

command -v hf >/dev/null || { echo "hf CLI missing from environment"; exit 1; }

# Core candidate models for benchmarking & teacher ensemble
declare -A MODELS=(
  [clip_vitl14]='openai/clip-vit-large-patch14'
  [siglip_base_224]='google/siglip-base-patch16-224'
  [siglip2_base_224]='google/siglip2-base-patch16-224'
  [siglip2_large_384]='google/siglip2-large-patch16-384'
  [dinov2_large]='facebook/dinov2-large'
  [convnext_tiny]='timm/convnext_tiny.fb_in1k'
  [aide_finetuned]='meet4150/AIDE_FINE_TUNED_98_acc'
  [aide_50epoch]='meet4150/50_epoch_aide'
)

echo "Starting download of model pool to $MODEL_ROOT..."

for NAME in "${!MODELS[@]}"; do
  ID="${MODELS[$NAME]}"
  DEST="$MODEL_ROOT/$NAME"
  echo "=================================================="
  echo "Downloading $NAME from $ID -> $DEST"
  echo "=================================================="
  case "$NAME" in
    aide_finetuned)
      hf download "$ID" --local-dir "$DEST" --include 'checkpoint42.pth' --include '*.py' --include '*.json' --include '*.txt' --include '*.md' --include 'LICENSE' --exclude 'data/*' --max-workers 4 || true
      ;;
    aide_50epoch)
      hf download "$ID" --local-dir "$DEST" --include 'model.safetensors' --include 'checkpoint-50.pth' --include '*.py' --include '*.json' --include '*.txt' --include '*.md' --include 'LICENSE' --exclude 'data/*' --max-workers 4 || true
      ;;
    *)
      hf download "$ID" --local-dir "$DEST" --max-workers 4 || true
      ;;
  esac
  printf '%s\t%s\t%s\n' "$NAME" "$ID" "$DEST" >> "$AI_ROOT/manifests/model_pool.tsv"
done

# DDA: fetch the official checkpoint only
DDA_DIR="$MODEL_ROOT/dda"
mkdir -p "$DDA_DIR"
echo "Downloading DDA checkpoint -> $DDA_DIR..."
hf download Junwei-Xi/Dual-Data-Alignment DDA_ckpt.pth --local-dir "$DDA_DIR" --max-workers 2 || true
printf '%s\t%s\t%s\n' 'dda' 'Junwei-Xi/Dual-Data-Alignment:DDA_ckpt.pth' "$DDA_DIR/DDA_ckpt.pth" >> "$AI_ROOT/manifests/model_pool.tsv"

echo "Model pool download finished. Generating parameter inventory..."

python - <<'PY'
import os
import json
from pathlib import Path

root = Path("/mnt/ai-storage/aigc_data/models")
out = Path("/mnt/ai-storage/aigc_data/manifests/model_inventory.json")
rows = []
if root.exists():
    for p in sorted(root.iterdir()):
        if p.is_dir():
            files = list(p.rglob("*"))
            size_bytes = sum(x.stat().st_size for x in files if x.is_file())
            rows.append({
                "name": p.name,
                "path": str(p),
                "size_mb": round(size_bytes / (1024 * 1024), 2),
                "file_count": len([x for x in files if x.is_file()])
            })

out.write_text(json.dumps(rows, indent=2))
print(f"Wrote model inventory ({len(rows)} models) to {out}")
for r in rows:
    print(f" - {r['name']:<20}: {r['size_mb']:>8.2f} MB ({r['file_count']} files)")
PY

echo "Model pool ready."
