#!/usr/bin/env bash
set -euo pipefail

AI_ROOT="${AI_ROOT:-/mnt/ai-storage/aigc_data}"
MODEL_ROOT="$AI_ROOT/models"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

mkdir -p "$MODEL_ROOT" "$AI_ROOT/manifests"

HF_TOKEN=""
if [[ -f "$PROJECT_ROOT/.env" ]]; then
  HF_TOKEN="$(grep HF_TOKEN "$PROJECT_ROOT/.env" | cut -d= -f2 | tr -d ' \r\n')"
fi

download_aria() {
  local DIR="$1"
  local URL="$2"
  local OUT="$3"

  mkdir -p "$DIR"
  if [[ -f "$DIR/$OUT" ]] && [[ ! -f "$DIR/$OUT.aria2" ]] && [[ -s "$DIR/$OUT" ]]; then
    echo "[EXISTS] $DIR/$OUT"
    return 0
  fi

  echo ">>> Downloading $OUT to $DIR..."
  if [[ -n "$HF_TOKEN" ]]; then
    aria2c -x 16 -s 16 -k 1M --allow-overwrite=true --auto-file-renaming=false \
      --header="Authorization: Bearer $HF_TOKEN" "$URL" -d "$DIR" -o "$OUT"
  else
    aria2c -x 16 -s 16 -k 1M --allow-overwrite=true --auto-file-renaming=false \
      "$URL" -d "$DIR" -o "$OUT"
  fi
  echo ">>> FINISHED $OUT"
}

echo "=================================================="
echo "STARTING HIGH-SPEED ARIA2 MODEL POOL DOWNLOAD"
echo "=================================================="

# 1. ConvNeXt Tiny
download_aria "$MODEL_ROOT/convnext_tiny" "https://huggingface.co/timm/convnext_tiny.fb_in1k/resolve/main/model.safetensors" "model.safetensors"
download_aria "$MODEL_ROOT/convnext_tiny" "https://huggingface.co/timm/convnext_tiny.fb_in1k/resolve/main/config.json" "config.json"

# 2. SigLIP Base 224
download_aria "$MODEL_ROOT/siglip_base_224" "https://huggingface.co/google/siglip-base-patch16-224/resolve/main/model.safetensors" "model.safetensors"
download_aria "$MODEL_ROOT/siglip_base_224" "https://huggingface.co/google/siglip-base-patch16-224/resolve/main/config.json" "config.json"
download_aria "$MODEL_ROOT/siglip_base_224" "https://huggingface.co/google/siglip-base-patch16-224/resolve/main/preprocessor_config.json" "preprocessor_config.json"

# 3. SigLIP2 Base 224
download_aria "$MODEL_ROOT/siglip2_base_224" "https://huggingface.co/google/siglip2-base-patch16-224/resolve/main/model.safetensors" "model.safetensors"
download_aria "$MODEL_ROOT/siglip2_base_224" "https://huggingface.co/google/siglip2-base-patch16-224/resolve/main/config.json" "config.json"
download_aria "$MODEL_ROOT/siglip2_base_224" "https://huggingface.co/google/siglip2-base-patch16-224/resolve/main/preprocessor_config.json" "preprocessor_config.json"

# 4. CLIP ViT-L/14
download_aria "$MODEL_ROOT/clip_vitl14" "https://huggingface.co/openai/clip-vit-large-patch14/resolve/main/model.safetensors" "model.safetensors"
download_aria "$MODEL_ROOT/clip_vitl14" "https://huggingface.co/openai/clip-vit-large-patch14/resolve/main/config.json" "config.json"
download_aria "$MODEL_ROOT/clip_vitl14" "https://huggingface.co/openai/clip-vit-large-patch14/resolve/main/preprocessor_config.json" "preprocessor_config.json"

# 5. DINOv2 Large
download_aria "$MODEL_ROOT/dinov2_large" "https://huggingface.co/facebook/dinov2-large/resolve/main/model.safetensors" "model.safetensors"
download_aria "$MODEL_ROOT/dinov2_large" "https://huggingface.co/facebook/dinov2-large/resolve/main/config.json" "config.json"

# 6. DDA Checkpoint
download_aria "$MODEL_ROOT/dda" "https://huggingface.co/Junwei-Xi/Dual-Data-Alignment/resolve/main/DDA_ckpt.pth" "DDA_ckpt.pth"

# 7. AIDE 50-Epoch
download_aria "$MODEL_ROOT/aide_50epoch" "https://huggingface.co/meet4150/50_epoch_aide/resolve/main/model.safetensors" "model.safetensors"
download_aria "$MODEL_ROOT/aide_50epoch" "https://huggingface.co/meet4150/50_epoch_aide/resolve/main/config.json" "config.json"
download_aria "$MODEL_ROOT/aide_50epoch/models" "https://huggingface.co/meet4150/50_epoch_aide/resolve/main/models/AIDE.py" "AIDE.py"
download_aria "$MODEL_ROOT/aide_50epoch/models" "https://huggingface.co/meet4150/50_epoch_aide/resolve/main/models/srm_filter_kernel.py" "srm_filter_kernel.py"

# 8. SigLIP2 Large 384
download_aria "$MODEL_ROOT/siglip2_large_384" "https://huggingface.co/google/siglip2-large-patch16-384/resolve/main/model.safetensors" "model.safetensors"
download_aria "$MODEL_ROOT/siglip2_large_384" "https://huggingface.co/google/siglip2-large-patch16-384/resolve/main/config.json" "config.json"
download_aria "$MODEL_ROOT/siglip2_large_384" "https://huggingface.co/google/siglip2-large-patch16-384/resolve/main/preprocessor_config.json" "preprocessor_config.json"

# 9. AIDE Fine-Tuned Checkpoint
download_aria "$MODEL_ROOT/aide_finetuned" "https://huggingface.co/meet4150/AIDE_FINE_TUNED_98_acc/resolve/main/checkpoint42.pth" "checkpoint42.pth"
download_aria "$MODEL_ROOT/aide_finetuned/models" "https://huggingface.co/meet4150/AIDE_FINE_TUNED_98_acc/resolve/main/models/AIDE.py" "AIDE.py"
download_aria "$MODEL_ROOT/aide_finetuned/models" "https://huggingface.co/meet4150/AIDE_FINE_TUNED_98_acc/resolve/main/models/srm_filter_kernel.py" "srm_filter_kernel.py"

echo "=================================================="
echo "ALL MODEL DOWNLOADS COMPLETE!"
echo "=================================================="

source "${ENV_DIR:-$HOME/.venvs/aigc-detector}/bin/activate"
python - <<'PY'
import json
from pathlib import Path

root = Path("/mnt/ai-storage/aigc_data/models")
out = Path("/mnt/ai-storage/aigc_data/manifests/model_inventory.json")
rows = []
if root.exists():
    for p in sorted(root.iterdir()):
        if p.is_dir():
            files = list(p.rglob("*"))
            size_mb = sum(x.stat().st_size for x in files if x.is_file() and not x.name.endswith('.aria2'))
            rows.append({
                "name": p.name,
                "path": str(p),
                "size_mb": round(size_mb / (1024 * 1024), 2),
                "file_count": len([x for x in files if x.is_file() and not x.name.endswith('.aria2')])
            })

out.write_text(json.dumps(rows, indent=2))
print(f"\nWrote model inventory ({len(rows)} models) to {out}:")
for r in rows:
    print(f" - {r['name']:<20}: {r['size_mb']:>8.2f} MB ({r['file_count']} files)")
PY
