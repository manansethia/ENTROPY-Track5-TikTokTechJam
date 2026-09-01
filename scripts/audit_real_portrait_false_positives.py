#!/usr/bin/env python3
"""
scripts/audit_real_portrait_false_positives.py
Quantifies Real High-Resolution and Portrait False Positives on Buildabot.
Evaluates the frozen production champion across stratified photographic subgroups.
"""

import os
import sys
import gc
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
OUTPUT_REPORT_PATH = REPO_ROOT / "reports" / "real_portrait_false_positive_audit.json"

def collect_diagnostic_real_images():
    """Collects diverse real images from existing datasets on Buildabot."""
    samples = []
    
    # 1. Defactify Real Images (Mixed Phone / Web / Social Media)
    defactify_real_dir = Path("/mnt/ai-storage/aigc_data/datasets/extracted_parquet_pool/defactify_real")
    if defactify_real_dir.exists():
        for p in list(defactify_real_dir.glob("*.jpg"))[:500] + list(defactify_real_dir.glob("*.png"))[:500]:
            samples.append({"path": str(p), "source": "Defactify_Real_Web", "type": "web_photography"})
            
    # 2. SID Real (DSLR / Clean Photography)
    sid_real_dir = Path("/mnt/ai-storage/aigc_data/datasets/extracted_parquet_pool/sid_real")
    if sid_real_dir.exists():
        for p in list(sid_real_dir.glob("*.jpg"))[:500] + list(sid_real_dir.glob("*.png"))[:500]:
            samples.append({"path": str(p), "source": "SID_Real_DSLR", "type": "dslr_clean"})
            
    # 3. ImageNet Authentic Photos (Diverse objects / scenes / lighting)
    imagenet_real_dir = Path("/mnt/ai-storage/aigc_data/datasets/remediation_expansion_pool/ImageNet_Authentic_Photo")
    if imagenet_real_dir.exists():
        for p in list(imagenet_real_dir.glob("*.jpg"))[:500]:
            samples.append({"path": str(p), "source": "ImageNet_Authentic", "type": "natural_objects"})
            
    # 4. COCO Authentic Photography from Massive 50k Pool
    coco_dir = Path("/mnt/ai-storage/aigc_data/datasets/massive_balanced_50k/real")
    if coco_dir.exists():
        for p in list(coco_dir.glob("coco_*.jpg"))[:500]:
            samples.append({"path": str(p), "source": "COCO_Authentic", "type": "complex_scene_human"})
            
    # 5. User Provided Diagnostic Image
    user_img = REPO_ROOT / "user_test_portrait.png"
    if user_img.exists():
        samples.append({"path": str(user_img), "source": "User_Formal_Portrait", "type": "studio_headshot"})
        
    return samples

def run_audit():
    print("=" * 70)
    print("AUDIT: REAL HIGH-RESOLUTION & PORTRAIT FALSE POSITIVE QUANTIFICATION")
    print("=" * 70)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Loading Frozen Champion Model on {device}...")
    model, meta = load_portable_champion_model(CHECKPOINT_PATH, device=device)
    
    samples = collect_diagnostic_real_images()
    print(f"Collected {len(samples)} authentic diagnostic evaluation samples.")
    
    T = meta.get("temperature", 1.5230212761606914)
    THRESH_ENTERPRISE = 0.984399  # FPR <= 0.10%
    THRESH_STANDARD = 0.500000    # FPR <= 1.00%
    
    results = []
    
    t0 = time.perf_counter()
    for idx, item in enumerate(samples):
        p = Path(item["path"])
        if not p.exists():
            continue
            
        try:
            with Image.open(p) as raw_img:
                w, h = raw_img.size
                aspect_ratio = round(w / h, 3)
                is_portrait = h > w
                is_high_res = (w * h) >= (1024 * 1024)
                
                img = ImageOps.exif_transpose(raw_img).convert("RGB")
                tensor = portable_eval_transform(img).unsqueeze(0).to(device)
                
            with torch.inference_mode():
                logits, ev_pred, srm_feats = model(tensor, return_evidence=True)
                raw_logit = float(logits.cpu().item())
                srm_energy = float(srm_feats.abs().mean().cpu().item())
                
            calibrated_prob = float(torch.sigmoid(torch.tensor(raw_logit / T)).item())
            
            results.append({
                "path": str(p),
                "source": item["source"],
                "type": item["type"],
                "width": w,
                "height": h,
                "aspect_ratio": aspect_ratio,
                "is_portrait": is_portrait,
                "is_high_res": is_high_res,
                "raw_logit": raw_logit,
                "calibrated_prob": calibrated_prob,
                "srm_energy": srm_energy,
                "is_fp_standard": calibrated_prob >= THRESH_STANDARD,
                "is_fp_enterprise": calibrated_prob >= THRESH_ENTERPRISE
            })
        except Exception as e:
            print(f"Skipping corrupt sample {p}: {e}")
            
        if (idx + 1) % 250 == 0:
            print(f"  Processed {idx + 1}/{len(samples)} samples ({time.perf_counter() - t0:.1f}s)...")
            
    print(f"\nAudit complete across {len(results)} valid authentic samples.")
    
    # Stratified Analysis
    categories = {
        "Overall_Authentic_Pool": lambda r: True,
        "High_Resolution (>1024x1024)": lambda r: r["is_high_res"],
        "Standard_Resolution (<=1024x1024)": lambda r: not r["is_high_res"],
        "Portrait_Orientation (H > W)": lambda r: r["is_portrait"],
        "Landscape_Orientation (W >= H)": lambda r: not r["is_portrait"],
        "Source_Defactify_Web": lambda r: r["source"] == "Defactify_Real_Web",
        "Source_SID_DSLR": lambda r: r["source"] == "SID_Real_DSLR",
        "Source_ImageNet_Objects": lambda r: r["source"] == "ImageNet_Authentic",
        "Source_COCO_Scenes": lambda r: r["source"] == "COCO_Authentic",
        "High_SRM_Energy (>4.0)": lambda r: r["srm_energy"] > 4.0,
        "Normal_SRM_Energy (<=4.0)": lambda r: r["srm_energy"] <= 4.0,
    }
    
    report_breakdown = {}
    print("\n" + "=" * 85)
    print(f"{'Category':35s} | {'Count':5s} | {'Mean P(AIGC)':12s} | {'Median P':9s} | {'P95 P':9s} | {'FPR @ 0.1%':10s}")
    print("=" * 85)
    
    for cat_name, cat_fn in categories.items():
        sub = [r for r in results if cat_fn(r)]
        if not sub:
            continue
        probs = [r["calibrated_prob"] for r in sub]
        fps_ent = sum(1 for r in sub if r["is_fp_enterprise"])
        fpr_ent = (fps_ent / len(sub)) * 100.0
        mean_p = float(np.mean(probs))
        median_p = float(np.median(probs))
        p95_p = float(np.percentile(probs, 95))
        
        report_breakdown[cat_name] = {
            "sample_count": len(sub),
            "false_positive_count_enterprise": fps_ent,
            "fpr_enterprise_pct": fpr_ent,
            "mean_p_aigc": mean_p,
            "median_p_aigc": median_p,
            "p95_p_aigc": p95_p,
            "mean_srm_energy": float(np.mean([r["srm_energy"] for r in sub]))
        }
        print(f"{cat_name:35s} | {len(sub):5d} | {mean_p:11.4f}% | {median_p:8.4f}% | {p95_p:8.4f}% | {fpr_ent:9.2f}%")
        
    print("=" * 85)
    
    OUTPUT_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_REPORT_PATH, "w") as f:
        json.dump({
            "report_id": "REAL_PORTRAIT_FALSE_POSITIVE_AUDIT",
            "evaluated_checkpoint": str(CHECKPOINT_PATH),
            "model_sha256": meta["file_sha256"],
            "parameter_hash": meta["parameter_hash"],
            "temperature": T,
            "enterprise_threshold": THRESH_ENTERPRISE,
            "total_samples_evaluated": len(results),
            "breakdown": report_breakdown
        }, f, indent=2)
        
    print(f"Report saved to {OUTPUT_REPORT_PATH}")

if __name__ == "__main__":
    run_audit()
