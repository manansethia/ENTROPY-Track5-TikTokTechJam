#!/usr/bin/env python3
"""High-speed exact-file parallel model downloader for AIGC detector candidates.
Downloads candidate weights and configs directly to /mnt/ai-storage/aigc_data/models/.
"""

import concurrent.futures
import json
import os
import sys
import time
from pathlib import Path
from huggingface_hub import hf_hub_download

MODEL_ROOT = Path("/mnt/ai-storage/aigc_data/models")
MANIFEST_DIR = Path("/mnt/ai-storage/aigc_data/manifests")

MODEL_ROOT.mkdir(parents=True, exist_ok=True)
MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

# Load HF token
token = os.environ.get("HF_TOKEN")
if not token:
    env_file = Path(__file__).resolve().parents[2] / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.startswith("HF_TOKEN="):
                token = line.split("=", 1)[1].strip()
                break

TARGETS = [
    # 1. CLIP ViT-L/14
    ("clip_vitl14", "openai/clip-vit-large-patch14", "model.safetensors"),
    ("clip_vitl14", "openai/clip-vit-large-patch14", "config.json"),
    ("clip_vitl14", "openai/clip-vit-large-patch14", "preprocessor_config.json"),

    # 2. SigLIP Base 224
    ("siglip_base_224", "google/siglip-base-patch16-224", "model.safetensors"),
    ("siglip_base_224", "google/siglip-base-patch16-224", "config.json"),
    ("siglip_base_224", "google/siglip-base-patch16-224", "preprocessor_config.json"),

    # 3. SigLIP2 Base 224
    ("siglip2_base_224", "google/siglip2-base-patch16-224", "model.safetensors"),
    ("siglip2_base_224", "google/siglip2-base-patch16-224", "config.json"),
    ("siglip2_base_224", "google/siglip2-base-patch16-224", "preprocessor_config.json"),

    # 4. SigLIP2 Large 384
    ("siglip2_large_384", "google/siglip2-large-patch16-384", "model.safetensors"),
    ("siglip2_large_384", "google/siglip2-large-patch16-384", "config.json"),
    ("siglip2_large_384", "google/siglip2-large-patch16-384", "preprocessor_config.json"),

    # 5. DINOv2 Large
    ("dinov2_large", "facebook/dinov2-large", "model.safetensors"),
    ("dinov2_large", "facebook/dinov2-large", "config.json"),

    # 6. ConvNeXt Tiny
    ("convnext_tiny", "timm/convnext_tiny.fb_in1k", "model.safetensors"),
    ("convnext_tiny", "timm/convnext_tiny.fb_in1k", "config.json"),

    # 7. AIDE 50 epoch
    ("aide_50epoch", "meet4150/50_epoch_aide", "model.safetensors"),
    ("aide_50epoch", "meet4150/50_epoch_aide", "config.json"),
    ("aide_50epoch", "meet4150/50_epoch_aide", "models/AIDE.py"),
    ("aide_50epoch", "meet4150/50_epoch_aide", "models/srm_filter_kernel.py"),

    # 8. AIDE fine-tuned
    ("aide_finetuned", "meet4150/AIDE_FINE_TUNED_98_acc", "checkpoint42.pth"),
    ("aide_finetuned", "meet4150/AIDE_FINE_TUNED_98_acc", "models/AIDE.py"),
    ("aide_finetuned", "meet4150/AIDE_FINE_TUNED_98_acc", "models/srm_filter_kernel.py"),

    # 9. DDA
    ("dda", "Junwei-Xi/Dual-Data-Alignment", "DDA_ckpt.pth"),
]


def download_file(item):
    local_dir_name, repo_id, filename = item
    dest_dir = MODEL_ROOT / local_dir_name
    dest_dir.mkdir(parents=True, exist_ok=True)
    target_path = dest_dir / filename
    if target_path.exists() and target_path.stat().st_size > 0:
        print(f"[{local_dir_name}] Already exists: {filename} ({target_path.stat().st_size / 1024 / 1024:.1f} MB)")
        return local_dir_name, filename, "EXISTS"

    print(f"[{local_dir_name}] Downloading {filename} from {repo_id}...")
    t0 = time.time()
    try:
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(dest_dir),
            token=token,
        )
        elapsed = time.time() - t0
        sz_mb = target_path.stat().st_size / 1024 / 1024 if target_path.exists() else 0
        print(f"[{local_dir_name}] FINISHED {filename} in {elapsed:.1f}s ({sz_mb:.1f} MB)")
        return local_dir_name, filename, "OK"
    except Exception as exc:
        print(f"[{local_dir_name}] ERROR {filename}: {exc}")
        return local_dir_name, filename, f"ERROR: {exc}"


def main():
    print(f"Starting parallel download of {len(TARGETS)} target model files to {MODEL_ROOT}...")
    t_start = time.time()
    results = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        futures = {executor.submit(download_file, item): item for item in TARGETS}
        for fut in concurrent.futures.as_completed(futures):
            res = fut.result()
            results.append(res)

    print("\n" + "=" * 60)
    print(f"MODEL DOWNLOAD COMPLETE in {time.time() - t_start:.1f}s")
    print("=" * 60)

    inventory = []
    for p in sorted(MODEL_ROOT.iterdir()):
        if p.is_dir():
            files = list(p.rglob("*"))
            size_mb = sum(x.stat().st_size for x in files if x.is_file()) / (1024 * 1024)
            inventory.append({
                "name": p.name,
                "path": str(p),
                "size_mb": round(size_mb, 2),
                "file_count": len([x for x in files if x.is_file()])
            })
            print(f" - {p.name:<20}: {size_mb:>8.2f} MB ({len(files)} files)")

    out_file = MANIFEST_DIR / "model_inventory.json"
    out_file.write_text(json.dumps(inventory, indent=2))
    print(f"\nSaved inventory to {out_file}")


if __name__ == "__main__":
    main()
