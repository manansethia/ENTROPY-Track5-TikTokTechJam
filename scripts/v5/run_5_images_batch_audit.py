#!/usr/bin/env python3
"""
run_5_images_batch_audit.py
---------------------------
Batch multi-specialist forensic evaluation for the 5 uploaded images:
  1. img1_crab_nebula.jpg
  2. img2_earth_globe.jpg
  3. img3_volcano_meteors.jpg
  4. img4_temple_reflection.jpg
  5. img5_scifi_globe_city.png
"""

import os
import sys
import json
import time
from pathlib import Path
from PIL import Image
import cv2
import numpy as np

sys.path.insert(0, "/home/manan/aigc_robust_detection")
from scripts.fused.master_fused_forensic_engine import ProductionForensicPipeline

def run_batch():
    batch_dir = "/home/manan/aigc_robust_detection/test_inputs/batch_eval"
    report_dir = "/home/manan/aigc_robust_detection/reports/batch_eval"
    os.makedirs(report_dir, exist_ok=True)
    
    images = [
        ("Image 1 (Crab Nebula - Hubble)", os.path.join(batch_dir, "img1_crab_nebula.jpg")),
        ("Image 2 (Earth from Space - Satellite)", os.path.join(batch_dir, "img2_earth_globe.jpg")),
        ("Image 3 (Volcanic Prehistoric Earth)", os.path.join(batch_dir, "img3_volcano_meteors.jpg")),
        ("Image 4 (Dakshineswar Temple Reflection)", os.path.join(batch_dir, "img4_temple_reflection.jpg")),
        ("Image 5 (Sci-Fi Floating Globe City)", os.path.join(batch_dir, "img5_scifi_globe_city.png"))
    ]
    
    pipeline = ProductionForensicPipeline()
    results = {}
    
    print("=" * 95)
    print("  BATCH MULTI-SPECIALIST FORENSIC AUDIT: 5 TEST IMAGES")
    print("=" * 95)
    
    for label, path in images:
        if not os.path.exists(path):
            print(f"  Missing: {path}")
            continue
            
        t0 = time.time()
        res = pipeline.analyze(path, save_heatmap=True)
        dur = time.time() - t0
        
        # Physical noise residual test
        img_np = np.array(Image.open(path).convert("RGB"))
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY).astype(np.float32)
        srm_kernel = np.array([
            [-1,  2, -2,  2, -1],
            [ 2, -6,  8, -6,  2],
            [-2,  8,-12,  8, -2],
            [ 2, -6,  8, -6,  2],
            [-1,  2, -2,  2, -1]
        ], dtype=np.float32) / 12.0
        srm_res = cv2.filter2D(gray, -1, srm_kernel)
        srm_var = float(np.var(srm_res))
        srm_kurt = float(np.mean((srm_res - np.mean(srm_res))**4) / (np.var(srm_res)**2 + 1e-6))
        
        res["physics_metrics"] = {
            "srm_variance": round(srm_var, 4),
            "srm_kurtosis": round(srm_kurt, 4)
        }
        res["label"] = label
        results[label] = res
        
        print(f"\n  👉 {label:42s}")
        print(f"     Verdict:                 {res['verdict']:12s} | Confidence: {res['confidence']:.4f}")
        print(f"     Class Probabilities:     Real={res['class_probabilities']['REAL']:.4f}, Partial={res['class_probabilities']['PARTIAL_AIGC']:.4f}, Full={res['class_probabilities']['FULL_AIGC']:.4f}")
        print(f"     Max Patch Anomaly:       {res['max_patch_anomaly']:.4f} | Affected Area: {res['affected_area_percentage']:.2f}%")
        print(f"     SRM Noise Kurtosis:      {srm_kurt:.4f} (Natural ~3.0-5.0, Synthetic >8.0)")

    out_file = os.path.join(report_dir, "batch_5_images_report.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
        
    print("\n" + "=" * 95)
    print(f"  Batch Forensic Audit Complete ✅ Saved to: {out_file}")
    print("=" * 95)

if __name__ == "__main__":
    run_batch()
