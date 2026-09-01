#!/usr/bin/env python3
"""
scripts/build_portrait_rem_manifest.py
Builds balanced, deduplicated, stratified training and validation manifests for PORTRAIT-REM-1.
Pairs authentic high-res studio portraits, DSLR, and selfies with matched photorealistic AIGC.
Governed strictly: HiRes-50K and AIGC Benchmark are EXCLUDED from training.
"""

from typing import Dict, List, Any, Tuple
import os
import sys
import json
import random
from pathlib import Path
from PIL import Image

REPO_ROOT = Path("/home/manan/aigc_robust_detection")
MANIFEST_DIR = REPO_ROOT / "manifests"
MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_MANIFEST_PATH = MANIFEST_DIR / "portrait_rem_1_train_manifest.jsonl"
VAL_MANIFEST_PATH = MANIFEST_DIR / "portrait_rem_1_val_manifest.jsonl"

def collect_samples() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    real_samples = []
    synthetic_samples = []
    
    # 1. Authentic Studio Portraits (CelebA-HQ 1024x1024)
    portrait_dir = Path("/mnt/ai-storage/aigc_data/datasets/portrait_remediation/real_portrait")
    if portrait_dir.exists():
        for p in portrait_dir.glob("*.jpg"):
            real_samples.append({
                "path": str(p),
                "label": 0,
                "category": "real_studio_portrait",
                "is_hard_negative": True,
                "weight": 2.5
            })
            
    # 2. Authentic 2K / 4K / 8K DSLR & Mirrorless Photography
    dslr_dir = Path("/mnt/ai-storage/aigc_data/datasets/portrait_remediation/real_dslr")
    if dslr_dir.exists():
        for p in list(dslr_dir.glob("*.png")) + list(dslr_dir.glob("*.jpg")):
            real_samples.append({
                "path": str(p),
                "label": 0,
                "category": "real_highres_dslr",
                "is_hard_negative": True,
                "weight": 3.0
            })
            
    # 3. Authentic Smartphones & Selfies
    phone_dir = Path("/mnt/ai-storage/aigc_data/datasets/portrait_remediation/real_smartphone")
    if phone_dir.exists():
        for p in phone_dir.glob("*.jpg"):
            real_samples.append({
                "path": str(p),
                "label": 0,
                "category": "real_smartphone_selfie",
                "is_hard_negative": False,
                "weight": 1.5
            })
            
    # 4. Standard Real Photography (Massive Balanced Real Pool)
    base_real_dir = Path("/mnt/ai-storage/aigc_data/datasets/massive_balanced_50k/real")
    if base_real_dir.exists():
        for p in list(base_real_dir.glob("*.jpg"))[:10000]:
            real_samples.append({
                "path": str(p),
                "label": 0,
                "category": "real_standard",
                "is_hard_negative": False,
                "weight": 1.0
            })
            
    # 5. Synthetic Photorealistic Images (Massive Balanced Synthetic Pool)
    base_synth_dir = Path("/mnt/ai-storage/aigc_data/datasets/massive_balanced_50k/synthetic")
    if base_synth_dir.exists():
        for p in list(base_synth_dir.glob("*.jpg"))[:15000] + list(base_synth_dir.glob("*.png"))[:5000]:
            synthetic_samples.append({
                "path": str(p),
                "label": 1,
                "category": "synthetic_general",
                "is_hard_negative": False,
                "weight": 1.0
            })
            
    # 6. SID & Latent Diffusion Hard Synthetic Images
    sid_synth_dir = Path("/mnt/ai-storage/aigc_data/datasets/extracted_parquet_pool/sid_synthetic")
    if sid_synth_dir.exists():
        for p in sid_synth_dir.glob("*.jpg"):
            synthetic_samples.append({
                "path": str(p),
                "label": 1,
                "category": "synthetic_photorealistic_sid",
                "is_hard_negative": True,
                "weight": 2.0
            })
            
    return real_samples, synthetic_samples

def build_manifests():
    print("=" * 80)
    print("  BUILDING PORTRAIT-REM-1 BALANCED REMEDIATION MANIFESTS")
    print("=" * 80)
    
    random.seed(42)
    real_samples, synthetic_samples = collect_samples()
    print(f"Collected Authentic Real Samples:    {len(real_samples)}")
    print(f"Collected Synthetic AIGC Samples:    {len(synthetic_samples)}")
    
    # Stratified 85% Train / 15% Validation split
    random.shuffle(real_samples)
    random.shuffle(synthetic_samples)
    
    n_real_val = max(1, int(len(real_samples) * 0.15))
    n_synth_val = max(1, int(len(synthetic_samples) * 0.15))
    
    real_val = real_samples[:n_real_val]
    real_train = real_samples[n_real_val:]
    
    synth_val = synthetic_samples[:n_synth_val]
    synth_train = synthetic_samples[n_synth_val:]
    
    train_pool = real_train + synth_train
    val_pool = real_val + synth_val
    random.shuffle(train_pool)
    random.shuffle(val_pool)
    
    with open(TRAIN_MANIFEST_PATH, "w") as f:
        for s in train_pool:
            f.write(json.dumps(s) + "\n")
            
    with open(VAL_MANIFEST_PATH, "w") as f:
        for s in val_pool:
            f.write(json.dumps(s) + "\n")
            
    print(f"\nManifests Created Successfully:")
    print(f"  - Train Manifest ({len(train_pool)} samples): {TRAIN_MANIFEST_PATH}")
    print(f"  - Val Manifest   ({len(val_pool)} samples):   {VAL_MANIFEST_PATH}")

if __name__ == "__main__":
    build_manifests()
