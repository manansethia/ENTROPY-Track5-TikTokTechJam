#!/usr/bin/env python3
"""
scripts/audit_geometry_resolution_shortcuts.py
Evaluates whether model predictions are shortcutting on image resolution, aspect ratio, or crop geometry.
Runs controlled counterfactual image manipulations on identical authentic high-res photos.
"""

import os
import sys
import json
import time
from pathlib import Path
from PIL import Image, ImageOps
import numpy as np
import torch
from torchvision import transforms

REPO_ROOT = Path("/home/manan/aigc_robust_detection")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deployment.portable_model import load_portable_champion_model

CHECKPOINT_PATH = REPO_ROOT / "checkpoints" / "production" / "final_champion_frozen_model.pt"
OUTPUT_REPORT_PATH = REPO_ROOT / "reports" / "geometry_shortcut_audit.json"

NORM_MEAN = [0.48145466, 0.4578275, 0.40821073]
NORM_STD = [0.26862954, 0.26130258, 0.27577711]

to_tensor_norm = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
])

def get_counterfactual_variants(img: Image.Image):
    """Generates controlled resolution and crop variants of a single authentic image."""
    w, h = img.size
    variants = {}
    
    # 1. Base Resize Variations (Full Field of View)
    variants["original_res"] = img.resize((224, 224), Image.BICUBIC)
    
    for target_dim in [1024, 512, 256]:
        if min(w, h) >= target_dim:
            scale = target_dim / min(w, h)
            new_w, new_h = int(w * scale), int(h * scale)
            rescaled = img.resize((new_w, new_h), Image.BICUBIC)
            variants[f"downscaled_{target_dim}px"] = rescaled.resize((224, 224), Image.BICUBIC)
            
    # 2. Crop Geometry Variations
    # Square Center Crop
    min_dim = min(w, h)
    left = (w - min_dim) // 2
    top = (h - min_dim) // 2
    square_crop = img.crop((left, top, left + min_dim, top + min_dim))
    variants["crop_square_1x1"] = square_crop.resize((224, 224), Image.BICUBIC)
    
    # Portrait Crop (4:5)
    target_ratio = 4.0 / 5.0
    if w / h > target_ratio:
        crop_w = int(h * target_ratio)
        crop_h = h
    else:
        crop_w = w
        crop_h = int(w / target_ratio)
    l = (w - crop_w) // 2
    t = (h - crop_h) // 2
    portrait_crop = img.crop((l, t, l + crop_w, t + crop_h))
    variants["crop_portrait_4x5"] = portrait_crop.resize((224, 224), Image.BICUBIC)
    
    # Landscape Crop (16:9)
    target_ratio_land = 16.0 / 9.0
    if w / h > target_ratio_land:
        crop_w_land = int(h * target_ratio_land)
        crop_h_land = h
    else:
        crop_w_land = w
        crop_h_land = int(w / target_ratio_land)
    l_l = (w - crop_w_land) // 2
    t_l = (h - crop_h_land) // 2
    landscape_crop = img.crop((l_l, t_l, l_l + crop_w_land, t_l + crop_h_land))
    variants["crop_landscape_16x9"] = landscape_crop.resize((224, 224), Image.BICUBIC)
    
    return variants

def run_geometry_audit():
    print("=" * 70)
    print("AUDIT: RESOLUTION & GEOMETRY SHORTCUT COUNTERFACTUAL ANALYSIS")
    print("=" * 70)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, meta = load_portable_champion_model(CHECKPOINT_PATH, device=device)
    T = meta.get("temperature", 1.5230212761606914)
    
    # Collect high-res authentic images
    candidate_paths = []
    sid_real_dir = Path("/mnt/ai-storage/aigc_data/datasets/extracted_parquet_pool/sid_real")
    if sid_real_dir.exists():
        candidate_paths.extend(list(sid_real_dir.glob("*.jpg"))[:100])
    coco_dir = Path("/mnt/ai-storage/aigc_data/datasets/massive_balanced_50k/real")
    if coco_dir.exists():
        candidate_paths.extend(list(coco_dir.glob("coco_*.jpg"))[:100])
    user_img = REPO_ROOT / "user_test_portrait.png"
    if user_img.exists():
        candidate_paths.append(user_img)
        
    print(f"Testing {len(candidate_paths)} authentic base images across 7 counterfactual variants...")
    
    variant_results = {}
    
    for idx, path in enumerate(candidate_paths):
        try:
            with Image.open(path) as raw_img:
                img = ImageOps.exif_transpose(raw_img).convert("RGB")
                variants = get_counterfactual_variants(img)
                
            for v_name, v_img in variants.items():
                tensor = to_tensor_norm(v_img).unsqueeze(0).to(device)
                with torch.inference_mode():
                    logits, _, srm_feats = model(tensor, return_evidence=True)
                    raw_logit = float(logits.cpu().item())
                    srm_energy = float(srm_feats.abs().mean().cpu().item())
                prob = float(torch.sigmoid(torch.tensor(raw_logit / T)).item())
                
                if v_name not in variant_results:
                    variant_results[v_name] = {"probs": [], "srm_energies": []}
                variant_results[v_name]["probs"].append(prob)
                variant_results[v_name]["srm_energies"].append(srm_energy)
        except Exception as e:
            print(f"Skipping {path}: {e}")
            
    summary = {}
    print("\n" + "=" * 75)
    print(f"{'Transformation Variant':30s} | {'Mean P(AIGC)':14s} | {'Median P':10s} | {'Mean SRM Energy':15s}")
    print("=" * 75)
    
    for v_name, data in variant_results.items():
        mean_p = float(np.mean(data["probs"]))
        med_p = float(np.median(data["probs"]))
        mean_srm = float(np.mean(data["srm_energies"]))
        summary[v_name] = {
            "mean_p_aigc": mean_p,
            "median_p_aigc": med_p,
            "mean_srm_energy": mean_srm,
            "sample_count": len(data["probs"])
        }
        print(f"{v_name:30s} | {mean_p:13.4f}% | {med_p:9.4f}% | {mean_srm:14.4f}")
        
    print("=" * 75)
    
    # Calculate Resolution Shortcut Flag
    res_delta = abs(summary.get("downscaled_256px", {}).get("mean_p_aigc", 0.0) - summary.get("original_res", {}).get("mean_p_aigc", 0.0))
    shortcut_flag = res_delta > 0.15
    print(f"\nRESOLUTION_GEOMETRY_SHORTCUT_OBSERVED: {shortcut_flag} (Delta={res_delta:.4f})")
    
    OUTPUT_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_REPORT_PATH, "w") as f:
        json.dump({
            "report_id": "GEOMETRY_RESOLUTION_SHORTCUT_AUDIT",
            "resolution_shortcut_flag": shortcut_flag,
            "delta_high_to_low_res": res_delta,
            "variants_summary": summary
        }, f, indent=2)
        
    print(f"Report saved to {OUTPUT_REPORT_PATH}")

if __name__ == "__main__":
    run_geometry_audit()
