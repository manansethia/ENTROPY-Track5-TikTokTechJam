#!/usr/bin/env python3
"""
scripts/audit_compression_shortcuts.py
Evaluates whether model predictions swing wildly under standard photographic JPEG compression.
Runs controlled compression sweeps on identical authentic images.
"""

import os
import sys
import io
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

from deployment.portable_model import load_portable_champion_model, portable_eval_transform

CHECKPOINT_PATH = REPO_ROOT / "checkpoints" / "production" / "final_champion_frozen_model.pt"
OUTPUT_REPORT_PATH = REPO_ROOT / "reports" / "compression_shortcut_audit.json"

def get_jpeg_compressed_image(img: Image.Image, quality: int) -> Image.Image:
    """Re-encodes an image at target JPEG quality level in memory."""
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")

def run_compression_audit():
    print("=" * 70)
    print("AUDIT: COMPRESSION SHORTCUT ANALYSIS (JPEG QUALITY SWEEP)")
    print("=" * 70)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, meta = load_portable_champion_model(CHECKPOINT_PATH, device=device)
    T = meta.get("temperature", 1.5230212761606914)
    
    # Collect representative authentic images
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
        
    qualities = [100, 95, 90, 80, 70, 50, 40]
    print(f"Testing {len(candidate_paths)} authentic base images across {len(qualities)} JPEG quality levels...")
    
    quality_results = {q: {"probs": [], "srm_energies": []} for q in qualities}
    
    for idx, path in enumerate(candidate_paths):
        try:
            with Image.open(path) as raw_img:
                img = ImageOps.exif_transpose(raw_img).convert("RGB")
                
            for q in qualities:
                compressed_img = get_jpeg_compressed_image(img, q)
                tensor = portable_eval_transform(compressed_img).unsqueeze(0).to(device)
                
                with torch.inference_mode():
                    logits, _, srm_feats = model(tensor, return_evidence=True)
                    raw_logit = float(logits.cpu().item())
                    srm_energy = float(srm_feats.abs().mean().cpu().item())
                    
                prob = float(torch.sigmoid(torch.tensor(raw_logit / T)).item())
                quality_results[q]["probs"].append(prob)
                quality_results[q]["srm_energies"].append(srm_energy)
        except Exception as e:
            print(f"Skipping {path}: {e}")
            
    summary = {}
    print("\n" + "=" * 70)
    print(f"{'JPEG Quality':15s} | {'Mean P(AIGC)':14s} | {'Median P':10s} | {'Mean SRM Energy':15s}")
    print("=" * 70)
    
    for q in qualities:
        probs = quality_results[q]["probs"]
        srms = quality_results[q]["srm_energies"]
        mean_p = float(np.mean(probs))
        med_p = float(np.median(probs))
        mean_srm = float(np.mean(srms))
        summary[f"JPEG_Q{q}"] = {
            "quality": q,
            "mean_p_aigc": mean_p,
            "median_p_aigc": med_p,
            "mean_srm_energy": mean_srm,
            "sample_count": len(probs)
        }
        print(f"JPEG Q={q:<8d} | {mean_p:13.4f}% | {med_p:9.4f}% | {mean_srm:14.4f}")
        
    print("=" * 70)
    
    q100_mean = summary["JPEG_Q100"]["mean_p_aigc"]
    q50_mean = summary["JPEG_Q50"]["mean_p_aigc"]
    compression_delta = abs(q100_mean - q50_mean)
    shortcut_flag = compression_delta > 0.15
    print(f"\nCOMPRESSION_SHORTCUT_OBSERVED: {shortcut_flag} (Delta Q100 vs Q50={compression_delta:.4f})")
    
    OUTPUT_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_REPORT_PATH, "w") as f:
        json.dump({
            "report_id": "COMPRESSION_SHORTCUT_AUDIT",
            "compression_shortcut_flag": shortcut_flag,
            "delta_q100_to_q50": compression_delta,
            "qualities_summary": summary
        }, f, indent=2)
        
    print(f"Report saved to {OUTPUT_REPORT_PATH}")

if __name__ == "__main__":
    run_compression_audit()
