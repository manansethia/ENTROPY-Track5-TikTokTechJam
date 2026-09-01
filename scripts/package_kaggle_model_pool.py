#!/usr/bin/env python3
"""
scripts/package_kaggle_model_pool.py
Packages all verified external model checkpoints into a versioned persistent
Kaggle dataset: doubleggunther/aigc-highres-model-pool-v1
"""

import os
import sys
import json
import shutil
import hashlib
import subprocess
from pathlib import Path

print("=== PACKAGING VERIFIED MODEL POOL FOR PERSISTENT KAGGLE DATASET ===")

UPLOAD_DIR = Path("/tmp/kaggle_model_pool_upload")
if UPLOAD_DIR.exists():
    shutil.rmtree(UPLOAD_DIR)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

models_base = Path("/mnt/ai-storage/aigc_data/models")
ckpt_dir = UPLOAD_DIR / "checkpoints"
ckpt_dir.mkdir(parents=True, exist_ok=True)

# 1. SPAI / TFG
spai_src = models_base / "spai_tfg/spai/weights/spai.pth"
if spai_src.exists():
    print("  [1/6] Copying SPAI / TFG checkpoint...")
    shutil.copy2(spai_src, ckpt_dir / "spai_tfg.pth")

# 2. CommunityForensics ViT-Small
cf_src = models_base / "community_forensics_vit_small/model.safetensors"
if cf_src.exists():
    print("  [2/6] Copying CommunityForensics ViT-Small checkpoint...")
    shutil.copy2(cf_src, ckpt_dir / "community_forensics_vit_small.safetensors")

# 3-6. divine2k Ensemble
d2k_base = models_base / "divine2k_ensemble"
for fname, target_name in [
    ("convNext_final.pth", "divine2k_convnext.pth"),
    ("convnext_tiny_final.pth", "divine2k_convnext_tiny.pth"),
    ("efficientNet_BO_Final.pth", "divine2k_efficientnet_b0.pth"),
    ("resnet50_ai_real_final.pth", "divine2k_resnet50.pth")
]:
    fpath = d2k_base / fname
    if fpath.exists():
        print(f"  Copying divine2k [{fname}] -> {target_name}...")
        shutil.copy2(fpath, ckpt_dir / target_name)

# 7. Metadata manifest & audit JSON
audit_json = {
    "version": "1.0.0",
    "dataset_slug": "doubleggunther/aigc-highres-model-pool-v1",
    "models": {
        "spai_tfg": {
            "file": "checkpoints/spai_tfg.pth",
            "size_mb": round((ckpt_dir / "spai_tfg.pth").stat().st_size / (1024**2), 2) if (ckpt_dir / "spai_tfg.pth").exists() else 0,
            "role": "FORENSIC SPECIALIST"
        },
        "community_forensics_vit_small": {
            "file": "checkpoints/community_forensics_vit_small.safetensors",
            "size_mb": round((ckpt_dir / "community_forensics_vit_small.safetensors").stat().st_size / (1024**2), 2) if (ckpt_dir / "community_forensics_vit_small.safetensors").exists() else 0,
            "role": "HIGH-RES SPECIALIST"
        },
        "divine2k_convnext": {
            "file": "checkpoints/divine2k_convnext.pth",
            "size_mb": round((ckpt_dir / "divine2k_convnext.pth").stat().st_size / (1024**2), 2) if (ckpt_dir / "divine2k_convnext.pth").exists() else 0,
            "role": "ROBUSTNESS SPECIALIST"
        },
        "divine2k_convnext_tiny": {
            "file": "checkpoints/divine2k_convnext_tiny.pth",
            "size_mb": round((ckpt_dir / "divine2k_convnext_tiny.pth").stat().st_size / (1024**2), 2) if (ckpt_dir / "divine2k_convnext_tiny.pth").exists() else 0,
            "role": "ROBUSTNESS SPECIALIST"
        },
        "divine2k_efficientnet_b0": {
            "file": "checkpoints/divine2k_efficientnet_b0.pth",
            "size_mb": round((ckpt_dir / "divine2k_efficientnet_b0.pth").stat().st_size / (1024**2), 2) if (ckpt_dir / "divine2k_efficientnet_b0.pth").exists() else 0,
            "role": "AUXILIARY EXPERT"
        },
        "divine2k_resnet50": {
            "file": "checkpoints/divine2k_resnet50.pth",
            "size_mb": round((ckpt_dir / "divine2k_resnet50.pth").stat().st_size / (1024**2), 2) if (ckpt_dir / "divine2k_resnet50.pth").exists() else 0,
            "role": "ROBUSTNESS SPECIALIST"
        }
    }
}

(UPLOAD_DIR / "model_metadata.json").write_text(json.dumps(audit_json, indent=2))
(UPLOAD_DIR / "README.md").write_text("""# AIGC High-Resolution Model Pool v1
Persistent verified external checkpoint bundle for multi-expert fine-tuning and fusion.
Contains: SPAI/TFG, CommunityForensics ViT-Small, and divine2k ensemble (ConvNeXt, ConvNeXt-Tiny, EfficientNet-B0, ResNet50).
""")

# Create dataset metadata
metadata = {
    "title": "AIGC High-Res Model Pool v1",
    "id": "doubleggunther/aigc-highres-model-pool-v1",
    "licenses": [{"name": "CC0-1.0"}]
}
(UPLOAD_DIR / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2))

print("\nModel pool package assembled. Total files:")
for f in ckpt_dir.glob("*"):
    print(f"  - {f.name} ({f.stat().st_size / (1024**2):.2f} MB)")

# Upload to Kaggle using kaggle datasets create
print("\nUploading persistent dataset to Kaggle...")
res = subprocess.run(["/home/manan/.venvs/aigc-detector/bin/kaggle", "datasets", "create", "-p", str(UPLOAD_DIR), "-r", "zip"], capture_output=True, text=True)
print("Kaggle Dataset Create STDOUT:", res.stdout)
if "already exists" in res.stderr or "already exists" in res.stdout:
    print("Dataset exists, creating new version...")
    res2 = subprocess.run(["/home/manan/.venvs/aigc-detector/bin/kaggle", "datasets", "version", "-p", str(UPLOAD_DIR), "-m", "Updated model pool with verified external checkpoints", "-r", "zip"], capture_output=True, text=True)
    print("Kaggle Dataset Version STDOUT:", res2.stdout)
    print("Kaggle Dataset Version STDERR:", res2.stderr)
else:
    print("Kaggle Dataset Create STDERR:", res.stderr)
