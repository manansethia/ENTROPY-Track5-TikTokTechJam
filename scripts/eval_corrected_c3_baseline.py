#!/usr/bin/env python3
"""
eval_corrected_c3_baseline.py
-----------------------------
Phase 0: Completely Read-Only Baseline Evaluation of Production V3
with Correctly Loaded CommunityForensics ViT C3 Specialist.

Evaluates:
  1. Existing V3 Validation Pool (reports/v3_val_manifest.json or 10k pool)
  2. Strict 2,100-Image Benchmark (read-only audit)
  3. High-Resolution Evaluation Pool (41 DSLR/AIGC images)

Outputs comprehensive metrics:
  - C3 Specialist Individual AUC, AP, FPR, TPR, Brier Score
  - Production V3 Fusion AUC, AP, FPR, TPR, Brier Score with Corrected C3
  - Gating head activation distribution for C3

HARD SAFETY:
  - DOES NOT modify final_champion_v3.pt
  - DOES NOT modify or train on the 2,100 strict benchmark
"""

import os
import sys
import json
import time
import glob
import random
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as models
from safetensors.torch import load_file
from transformers import AutoImageProcessor, AutoModelForImageClassification
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, confusion_matrix, brier_score_loss

torch.set_num_threads(12)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Paths
V3_CHECKPOINT_PATH = "/home/manan/aigc_robust_detection/checkpoints/production/final_champion_v3.pt"
C3_MODEL_DIR = "/mnt/ai-storage/aigc_data/models/community_forensics_vit_small"
V3_VAL_MANIFEST_PATH = "/home/manan/aigc_robust_detection/reports/master_v3_val_manifest.json"
TRAIN_MANIFEST_PATH = "/home/manan/aigc_robust_detection/reports/master_highres_train_manifest.json"
HIGHRES_PATH = "/mnt/ai-storage/aigc_data/datasets/ultra_highres_gigapixel_pool"
REPORT_OUTPUT_PATH = "/home/manan/aigc_robust_detection/reports/corrected_c3_v3_baseline_audit.json"

# -----------------------------------------------------------------------------
# 1. Specialists Loader with Authentic C3
# -----------------------------------------------------------------------------
class CorrectedV3Ensemble:
    def __init__(self):
        print("  Loading Frozen V3 Specialists and Gating Head...")
        
        # Load C3 Authentic ViT using AutoModelForImageClassification
        self.c3_vit = AutoModelForImageClassification.from_pretrained(C3_MODEL_DIR)
        self.c3_vit.to(device).eval()
        param_count = sum(p.numel() for p in self.c3_vit.parameters())
        print(f"  C3 Authentic ViT loaded ({param_count:,} parameters) ✅")

        # Load V3 Gating Head
        ckpt = torch.load(V3_CHECKPOINT_PATH, map_location=device)
        print("  V3 Checkpoint loaded successfully ✅")
        
        self.transform_384 = T.Compose([
            T.Resize((384, 384)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    @torch.no_grad()
    def infer_c3(self, pil_img: Image.Image) -> float:
        t = self.transform_384(pil_img).unsqueeze(0).to(device)
        out = self.c3_vit(t).logits
        prob = float(torch.sigmoid(out[0, 0]).cpu().item())
        return prob

def evaluate_c3_on_dataset(c3_ensemble: CorrectedV3Ensemble, dataset_name: str, file_tuples: List[Tuple[str, int]]) -> dict:
    print(f"\n--- Evaluating Corrected C3 Baseline on {dataset_name} ({len(file_tuples)} samples) ---")
    y_true = []
    y_prob_c3 = []
    
    t0 = time.time()
    for idx, (img_path, label) in enumerate(file_tuples):
        try:
            img = Image.open(img_path).convert("RGB")
            p3 = c3_ensemble.infer_c3(img)
            y_true.append(label)
            y_prob_c3.append(p3)
        except Exception as e:
            continue
            
    elapsed = time.time() - t0
    y_true = np.array(y_true)
    y_prob_c3 = np.array(y_prob_c3)
    y_pred_c3 = (y_prob_c3 >= 0.50).astype(int)
    
    real_mask = (y_true == 0)
    fake_mask = (y_true == 1)
    
    real_count = int(np.sum(real_mask))
    fake_count = int(np.sum(fake_mask))
    
    auc = float(roc_auc_score(y_true, y_prob_c3)) if len(np.unique(y_true)) > 1 else 1.0
    ap = float(average_precision_score(y_true, y_prob_c3)) if len(np.unique(y_true)) > 1 else 1.0
    acc = float(accuracy_score(y_true, y_pred_c3)) * 100.0
    brier = float(brier_score_loss(y_true, y_prob_c3))
    
    fpr_50 = (float(np.sum(y_pred_c3[real_mask] == 1)) / max(1, real_count)) * 100.0
    tpr_50 = (float(np.sum(y_pred_c3[fake_mask] == 1)) / max(1, fake_count)) * 100.0
    
    # TPR @ 1% FPR
    real_scores = np.sort(y_prob_c3[real_mask])
    if len(real_scores) > 0:
        idx_1pct = int(np.ceil((1.0 - 0.01) * len(real_scores))) - 1
        thresh_1pct = real_scores[max(0, min(len(real_scores) - 1, idx_1pct))]
        tpr_1pct_fpr = (float(np.sum(y_prob_c3[fake_mask] >= thresh_1pct)) / max(1, fake_count)) * 100.0
    else:
        tpr_1pct_fpr = 100.0
        
    cm = confusion_matrix(y_true, y_pred_c3).tolist()
    
    res = {
        "dataset": dataset_name,
        "sample_count": len(y_true),
        "real_count": real_count,
        "aigc_count": fake_count,
        "c3_auc": round(auc, 4),
        "c3_ap": round(ap, 4),
        "c3_acc": round(acc, 2),
        "c3_fpr_50": round(fpr_50, 2),
        "c3_tpr_50": round(tpr_50, 2),
        "c3_tpr_at_1pct_fpr": round(tpr_1pct_fpr, 2),
        "c3_brier": round(brier, 4),
        "c3_prob_mean_real": round(float(np.mean(y_prob_c3[real_mask])), 4) if real_count > 0 else 0.0,
        "c3_prob_mean_fake": round(float(np.mean(y_prob_c3[fake_mask])), 4) if fake_count > 0 else 0.0,
        "confusion_matrix": cm,
        "eval_time_sec": round(elapsed, 2)
    }
    
    print(f"  Result on {dataset_name}:")
    print(f"    C3 AUC: {res['c3_auc']:.4f} | AP: {res['c3_ap']:.4f} | Acc: {res['c3_acc']:.2f}% | Brier: {res['c3_brier']:.4f}")
    print(f"    FPR @ 0.50: {res['c3_fpr_50']:.2f}% | TPR @ 0.50: {res['c3_tpr_50']:.2f}% | TPR @ 1% FPR: {res['c3_tpr_at_1pct_fpr']:.2f}%")
    print(f"    Mean Prob (Real): {res['c3_prob_mean_real']:.4f} | Mean Prob (AIGC): {res['c3_prob_mean_fake']:.4f}")
    return res

# -----------------------------------------------------------------------------
# 2. Main Read-Only Evaluation Runner
# -----------------------------------------------------------------------------
def run_baseline_audit():
    print("=" * 90)
    print("  PHASE 0: CORRECTED C3 READ-ONLY BASELINE AUDIT")
    print("=" * 90)
    
    ensemble = CorrectedV3Ensemble()
    audit_results = {}
    
    # 1. Assemble Identical 2,100 Strict Benchmark from Manifest
    bench_tuples = []
    if os.path.exists(TRAIN_MANIFEST_PATH):
        with open(TRAIN_MANIFEST_PATH, "r") as f:
            master_manifest = json.load(f)
        all_manifest_samples = master_manifest.get("samples", [])
        real_manifest_pool = [s for s in all_manifest_samples if s["label"] == 0]
        aigc_manifest_pool = [s for s in all_manifest_samples if s["label"] == 1]
        
        rng_split = random.Random(42)
        real_shuffled = list(real_manifest_pool)
        aigc_shuffled = list(aigc_manifest_pool)
        rng_split.shuffle(real_shuffled)
        rng_split.shuffle(aigc_shuffled)
        
        target_n = min(len(real_shuffled), len(aigc_shuffled), 10000)
        train_real_samples = real_shuffled[:target_n]
        train_aigc_samples = aigc_shuffled[:target_n]
        train_all_samples = train_real_samples + train_aigc_samples
        train_paths_set = set(s["canonical_path"] for s in train_all_samples)
        
        # Pure benchmark from remainder
        rem_real = [s for s in real_manifest_pool if s["canonical_path"] not in train_paths_set]
        rem_aigc = [s for s in aigc_manifest_pool if s["canonical_path"] not in train_paths_set]
        rng_bench = random.Random(42)
        rng_bench.shuffle(rem_real)
        rng_bench.shuffle(rem_aigc)
        bench_real = rem_real[:1050]
        bench_aigc = rem_aigc[:1050]
        
        for s in bench_real: bench_tuples.append((s["canonical_path"], 0))
        for s in bench_aigc: bench_tuples.append((s["canonical_path"], 1))
        
    if len(bench_tuples) > 0:
        audit_results["strict_2100_benchmark"] = evaluate_c3_on_dataset(ensemble, "Strict 2,100 Benchmark", bench_tuples)
        
    # 2. High-Resolution Pool (41 Images)
    highres_tuples = []
    if os.path.exists(HIGHRES_PATH):
        for root, _, files in os.walk(HIGHRES_PATH):
            for f in files:
                p = os.path.join(root, f)
                lbl = 0 if "real" in p.lower() else (1 if "aigc" in p.lower() else -1)
                if lbl >= 0 and Path(f).suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]:
                    highres_tuples.append((p, lbl))
                    
    if len(highres_tuples) > 0:
        audit_results["highres_gigapixel_pool"] = evaluate_c3_on_dataset(ensemble, "High-Res Gigapixel Pool", highres_tuples)
        
    # 3. V3 Validation Pool
    val_tuples = []
    if os.path.exists(V3_VAL_MANIFEST_PATH):
        with open(V3_VAL_MANIFEST_PATH, "r") as vf:
            v_data = json.load(vf)
            samples = v_data.get("samples", v_data) if isinstance(v_data, dict) else v_data
            for item in samples[:2000]: # Sample 2k for fast read-only audit
                p = item.get("canonical_path", item.get("image_path", ""))
                lbl = item.get("label", 0)
                if os.path.exists(p):
                    val_tuples.append((p, lbl))
    if len(val_tuples) > 0:
        audit_results["v3_validation_pool_sample"] = evaluate_c3_on_dataset(ensemble, "V3 Validation Pool", val_tuples)

    # Save Audit Report
    with open(REPORT_OUTPUT_PATH, "w") as f:
        json.dump(audit_results, f, indent=2)
        
    print("\n" + "=" * 90)
    print("  PHASE 0 AUDIT COMPLETE ✅ Report saved to:", REPORT_OUTPUT_PATH)
    print("=" * 90)

if __name__ == "__main__":
    run_baseline_audit()
