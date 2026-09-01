#!/usr/bin/env python3
"""
analyze_4women_quadrants.py
---------------------------
Analyzes the 4women.webp collage at:
  1. Full Collage Multi-Scale Spatial Tiling Level
  2. Individual Quadrant Sub-Image Level (Top-Left, Top-Right, Bottom-Left, Bottom-Right)
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

def analyze_collage(image_path: str):
    print("=" * 95)
    print("  EXECUTING MULTI-SCALE COLLAGE FORENSIC ANALYSIS: 4women.webp")
    print("=" * 95)
    
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    print(f"  Source Dimensions: {w} x {h} ({w*h/1e6:.2f} Megapixels)")
    
    pipeline = ProductionForensicPipeline()
    
    # 1. Full Collage Analysis
    print("\n  [1/2] Analyzing Full Collage with Multi-Scale Hierarchical Slicing...")
    full_res = pipeline.analyze(image_path, save_heatmap=True)
    print(f"    - Full Collage Verdict:       {full_res['verdict']}")
    print(f"    - Full Confidence:            {full_res['confidence']:.4f}")
    print(f"    - Tri-Class Distribution:     Real={full_res['class_probabilities']['REAL']:.4f}, Partial={full_res['class_probabilities']['PARTIAL_AIGC']:.4f}, Full={full_res['class_probabilities']['FULL_AIGC']:.4f}")
    print(f"    - Affected Area Estimated:    {full_res['affected_area_percentage']:.2f}%")
    print(f"    - Suspicious Patches Count:   {full_res['suspicious_regions_count']}")
    
    # 2. Quadrant-Level Sub-Image Breakdown
    print("\n  [2/2] Analyzing Individual Sub-Panel Quadrants...")
    quad_dir = "/home/manan/aigc_robust_detection/reports/quadrants_4women"
    os.makedirs(quad_dir, exist_ok=True)
    
    hw, hh = w // 2, h // 2
    quadrants = {
        "Top-Left (Bucket Hat Woman)": (0, 0, hw, hh),
        "Top-Right (Sunglasses Woman)": (hw, 0, w, hh),
        "Bottom-Left (Glasses Smile Woman)": (0, hh, hw, h),
        "Bottom-Right (Brooklyn Bridge Woman)": (hw, hh, w, h)
    }
    
    quadrant_results = {}
    for q_name, (qx1, qy1, qx2, qy2) in quadrants.items():
        q_crop = img.crop((qx1, qy1, qx2, qy2))
        q_path = os.path.join(quad_dir, f"{q_name.split()[0].lower()}.jpg")
        q_crop.save(q_path, quality=95)
        
        q_res = pipeline.analyze(q_path, save_heatmap=True)
        quadrant_results[q_name] = {
            "verdict": q_res["verdict"],
            "confidence": q_res["confidence"],
            "class_probabilities": q_res["class_probabilities"],
            "max_patch_anomaly": q_res["max_patch_anomaly"],
            "affected_area": q_res["affected_area_percentage"],
            "suspicious_count": q_res["suspicious_regions_count"],
            "heatmap_path": q_res["heatmap_path"]
        }
        print(f"    👉 {q_name:36s} -> Verdict: {q_res['verdict']:12s} | Confidence: {q_res['confidence']:.4f} | Max Patch Anomaly: {q_res['max_patch_anomaly']:.4f}")

    combined = {
        "source_image": os.path.basename(image_path),
        "dimensions": {"width": w, "height": h, "megapixels": round(w*h/1e6, 2)},
        "full_collage_analysis": full_res,
        "quadrant_breakdown": quadrant_results
    }
    
    out_json = "/home/manan/aigc_robust_detection/reports/4women_forensic_analysis.json"
    with open(out_json, "w") as f:
        json.dump(combined, f, indent=2)
        
    print(f"\n  Full Collage & Quadrant Report saved to: {out_json} ✅")
    print("=" * 95)

if __name__ == "__main__":
    analyze_collage("/home/manan/aigc_robust_detection/test_inputs/4women.webp")
