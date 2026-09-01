#!/usr/bin/env python3
"""
benchmark_v4_patch_ablations.py
-------------------------------
V4.3 Controlled Patch-Size Ablation and High-Resolution Benchmark Suite.
Compares:
  Config A: Global only
  Config B: Global + 1024px
  Config C: Global + 768px
  Config D: Global + 512px
  Config E: Global + 1024px + 768px
  Config F: Global + 1024px + 768px + 512px

Evaluates on ultra-high-resolution real DSLR images, ultra-high-res AIGC images,
and localized inpainting / test images using the frozen V3 model.
"""

import os
import sys
import json
import time
import glob
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score

REPO_ROOT = Path("/home/manan/aigc_robust_detection")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.v4.forensic_multiscale_engine import V4ForensicEngine, format_forensic_report_text, eval_transform

REPORT_OUTPUT_DIR = "/home/manan/aigc_robust_detection/reports/v4_forensics"
HEATMAP_OUTPUT_DIR = "/home/manan/aigc_robust_detection/reports/v4_heatmaps"
os.makedirs(REPORT_OUTPUT_DIR, exist_ok=True)
os.makedirs(HEATMAP_OUTPUT_DIR, exist_ok=True)

CONFIGS = {
    "Config_A_Global_Only": [],
    "Config_B_Global_1024": [1024],
    "Config_C_Global_768": [768],
    "Config_D_Global_512": [512],
    "Config_E_Global_1024_768": [1024, 768],
    "Config_F_Full_Hierarchical": [1024, 768, 512]
}

def gather_highres_evaluation_pool():
    """Gathers representative ultra-high-res real DSLR photos, AIGC samples, and user test images."""
    eval_pool = []
    
    # 1. Real Ultra High-Res DSLR Photography (3K - 14K px)
    real_dir = "/mnt/ai-storage/aigc_data/datasets/ultra_highres_gigapixel_pool/real_dslr_3k_10k"
    if os.path.exists(real_dir):
        files = sorted(glob.glob(f"{real_dir}/*.jpg"))[:15]
        for f in files:
            eval_pool.append({"path": f, "label": 0, "type": "real_dslr_gigapixel"})

    # 2. AIGC Ultra High-Res Synthetics (2K - 4K px)
    aigc_dir = "/mnt/ai-storage/aigc_data/datasets/ultra_highres_gigapixel_pool/aigc_counterpart_3k_10k"
    if os.path.exists(aigc_dir):
        files = sorted(glob.glob(f"{aigc_dir}/*.jpg"))[:15]
        for f in files:
            eval_pool.append({"path": f, "label": 1, "type": "aigc_highres_synthetic"})

    # 3. User Real-World Test Suite (Portraits, Lightroom hard negatives, web photos)
    test_dir = "/home/manan/aigc_robust_detection/test_inputs"
    if os.path.exists(test_dir):
        t_files = sorted(glob.glob(f"{test_dir}/*.*"))
        for f in t_files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.avif')):
                is_real = ("real" in f.lower() or "d3b177be" in f.lower() or "images.jpeg" in f.lower() or "photo-1472214103451" in f.lower())
                eval_pool.append({
                    "path": f,
                    "label": 0 if is_real else 1,
                    "type": "user_test_suite"
                })

    return eval_pool

def run_patch_ablation_benchmarks():
    print("=" * 95)
    print("  V4.3 CONTROLLED PATCH-SIZE ABLATION & HIGH-RESOLUTION FORENSIC BENCHMARK")
    print("=" * 95)

    engine = V4ForensicEngine()
    eval_pool = gather_highres_evaluation_pool()
    print(f"  Total High-Resolution Evaluation Samples: {len(eval_pool)} ({sum(1 for x in eval_pool if x['label']==0)} Real, {sum(1 for x in eval_pool if x['label']==1)} AIGC)")

    ablation_results = {}

    for cfg_name, scales in CONFIGS.items():
        print(f"\n" + "-" * 90)
        print(f"  >>> RUNNING {cfg_name} (Scales: {scales if scales else 'Global Resized Only'}) <<<")
        print("-" * 90)

        preds = []
        targets = []
        latencies = []
        patch_counts = []

        for sample in eval_pool:
            p = sample["path"]
            label = sample["label"]
            try:
                t_start = time.time()
                if not scales: # Global only
                    raw_img = Image.open(p).convert("RGB")
                    t_in = engine._eval_batch_logits(eval_transform(raw_img).unsqueeze(0).to(engine.device))
                    score = float(t_in[0][0].item())
                    elapsed = time.time() - t_start
                    p_count = 1
                else:
                    report = engine.analyze_image(p, scales=scales, overlap_ratio=0.20)
                    score = report["forensic_verdict"]["localized_ai_probability"]
                    elapsed = report["elapsed_seconds"]
                    p_count = report["patch_inference"]["total_patches_evaluated"]

                preds.append(score)
                targets.append(label)
                latencies.append(elapsed)
                patch_counts.append(p_count)

            except Exception as e:
                print(f"    Error processing {os.path.basename(p)}: {e}")

        y_true = np.array(targets)
        y_pred = np.array(preds)

        auc = float(roc_auc_score(y_true, y_pred))
        ap = float(average_precision_score(y_true, y_pred))
        acc = float(accuracy_score(y_true, (y_pred >= 0.50).astype(int))) * 100
        real_fpr = float(np.mean(y_pred[y_true == 0] >= 0.50)) * 100
        aigc_tpr = float(np.mean(y_pred[y_true == 1] >= 0.50)) * 100
        avg_time = float(np.mean(latencies))
        avg_patches = float(np.mean(patch_counts))

        ablation_results[cfg_name] = {
            "scales": scales,
            "roc_auc": round(auc, 4),
            "average_precision": round(ap, 4),
            "accuracy_50": round(acc, 2),
            "real_fpr_50": round(real_fpr, 2),
            "aigc_tpr_50": round(aigc_tpr, 2),
            "avg_latency_seconds": round(avg_time, 3),
            "avg_patches_per_image": round(avg_patches, 1)
        }

        print(f"  Result: AUC: {auc:.4f} | AP: {ap:.4f} | Acc: {acc:.2f}% | FPR: {real_fpr:.2f}% | TPR: {aigc_tpr:.2f}% | Avg Latency: {avg_time:.2f}s ({avg_patches:.0f} patches)")

    # -------------------------------------------------------------
    # Run Full Forensic Multi-Scale Heatmap on Key Representative Images
    # -------------------------------------------------------------
    print("\n" + "=" * 95)
    print("  GENERATING DETAILED V4.1 FORENSIC HEATMAPS & REPORTS FOR REPRESENTATIVE SAMPLES")
    print("=" * 95)

    key_samples = [
        "/home/manan/aigc_robust_detection/test_inputs/d3b177be-gp0su1gn2_medium-res-1200px-1024x683.jpg",
        "/home/manan/aigc_robust_detection/test_inputs/lake-landscape-photo-m.jpg",
        "/home/manan/aigc_robust_detection/test_inputs/whats-the-most-realistic-ai-photo-generator-online-v0-lav1uhmvubre1.webp",
        "/mnt/ai-storage/aigc_data/datasets/ultra_highres_gigapixel_pool/real_dslr_3k_10k/real_ultra_highres_0016_6016x4016.jpg",
        "/mnt/ai-storage/aigc_data/datasets/ultra_highres_gigapixel_pool/aigc_counterpart_3k_10k/aigc_ultra_highres_0001_3024x4032.jpg"
    ]

    detailed_reports = []
    for p in key_samples:
        if not os.path.exists(p): continue
        fname = os.path.basename(p)
        print(f"\n  Analyzing: {fname}...")
        report = engine.analyze_image(p, scales=[1024, 768, 512], overlap_ratio=0.20)
        
        # Save JSON & Heatmap Image
        json_path = os.path.join(REPORT_OUTPUT_DIR, f"{Path(fname).stem}_forensic_report.json")
        heatmap_path = os.path.join(HEATMAP_OUTPUT_DIR, f"{Path(fname).stem}_spatial_heatmap.jpg")
        
        json_clean = dict(report)
        json_clean.pop("continuous_heatmap", None)
        with open(json_path, "w") as f:
            json.dump(json_clean, f, indent=2)
            
        engine.render_and_save_visual_heatmap(report, heatmap_path)
        print(format_forensic_report_text(report))
        detailed_reports.append(json_clean)

    # Save summary report
    summary_path = os.path.join(REPORT_OUTPUT_DIR, "v4_patch_ablation_summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "ablation_comparison": ablation_results,
            "detailed_reports": detailed_reports
        }, f, indent=2)

    print("\n" + "=" * 95)
    print(f"  V4.1 / V4.3 EXPERIMENT COMPLETED ✅ All results saved to {summary_path}")
    print("=" * 95)

if __name__ == "__main__":
    run_patch_ablation_benchmarks()
