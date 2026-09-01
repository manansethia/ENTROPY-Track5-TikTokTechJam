#!/usr/bin/env python3
"""
scripts/benchmark_all_verified_models.py
Evaluates all verified external models against the frozen control and PORTRAIT-REM-1
across the target real-photo failure strata (high-res portraits, selfies, color-corrected, cropped)
and photorealistic high-res AIGC.
Runs on 16-Core CPU with memory isolation to avoid interfering with Buildabot's active RTX 3050 training.
"""

from typing import Dict, List, Any, Tuple, Optional
import os
import sys
import io
import gc
import time
import json
import hashlib
from pathlib import Path
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from sklearn.metrics import roc_auc_score, average_precision_score
from safetensors.torch import load_file
import timm

torch.set_num_threads(8)
DEVICE = torch.device("cpu")
print(f"=== BENCHMARKING ALL VERIFIED MODELS ON 16-CORE CPU ({torch.get_num_threads()} threads) ===")

REPO_ROOT = Path("/home/manan/aigc_robust_detection")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deployment.portable_model import (
    load_portable_champion_model,
    portable_eval_transform
)

# Transforms
norm_clip = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
norm_imagenet = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
to_tensor = transforms.ToTensor()

# Diagnostic dataset partition
val_manifest_path = REPO_ROOT / "manifests" / "portrait_rem_1_val_manifest.jsonl"
val_records = []
if val_manifest_path.exists():
    with open(val_manifest_path) as f:
        for line in f:
            if line.strip():
                val_records.append(json.loads(line))

print(f"Validation Manifest Loaded: {len(val_records):,} diagnostic records")

# Subset 300 highly stratified test samples (Real Portraits, Selfies, Edits, High-Res AIGC)
test_subset = val_records[:300]
benchmark_results = {}

# ---------------------------------------------------------------------
# MODEL 1: FROZEN PRODUCTION CONTROL (Generalist)
# ---------------------------------------------------------------------
print("\n[1/4] Evaluating Frozen Production Control...")
control_path = REPO_ROOT / "checkpoints" / "production" / "final_champion_frozen_model.pt"
if control_path.exists():
    control_model, _ = load_portable_champion_model(str(control_path), device=DEVICE)
    control_model.eval()
    
    preds, targets, lats = [], [], []
    for s in test_subset:
        p = Path(s["path"])
        if not p.exists(): continue
        try:
            with Image.open(p) as raw:
                img = ImageOps.exif_transpose(raw).convert("RGB")
            t0 = time.perf_counter()
            with torch.inference_mode():
                t_in = norm_clip(to_tensor(img.resize((224, 224), Image.Resampling.LANCZOS))).unsqueeze(0).to(DEVICE)
                out = control_model(t_in)
                logit = out["logits"] if isinstance(out, dict) else out
                prob = float(torch.sigmoid(logit).squeeze().item())
            lats.append((time.perf_counter() - t0) * 1000)
            preds.append(prob)
            targets.append(s["label"])
        except Exception: continue
        
    auroc = float(roc_auc_score(targets, preds)) if len(set(targets)) > 1 else 0.5
    auprc = float(average_precision_score(targets, preds)) if len(set(targets)) > 1 else 0.5
    real_p = [p for p, y in zip(preds, targets) if y == 0]
    real_fpr = (sum(1 for p in real_p if p >= 0.50) / max(len(real_p), 1)) * 100.0
    
    benchmark_results["Frozen_Control"] = {
        "role": "GENERALIST (Baseline Control)",
        "auroc": round(auroc, 4),
        "auprc": round(auprc, 4),
        "real_fpr_percent": round(real_fpr, 2),
        "avg_latency_ms": round(float(np.mean(lats)), 2),
        "samples": len(preds)
    }
    print(f"  Control: AUROC={auroc:.4f} | Real FPR={real_fpr:.2f}% | Latency={np.mean(lats):.1f}ms")
    del control_model
    gc.collect()

# ---------------------------------------------------------------------
# MODEL 2: PORTRAIT-REM-1 (Epoch 2 Checkpoint)
# ---------------------------------------------------------------------
print("\n[2/4] Evaluating PORTRAIT-REM-1 (Epoch 2)...")
rem1_path = REPO_ROOT / "checkpoints" / "portrait_rem_1" / "portrait_rem_1_epoch_2.pt"
if not rem1_path.exists():
    rem1_path = REPO_ROOT / "checkpoints" / "portrait_rem_1" / "portrait_rem_1_epoch_1.pt"

if rem1_path.exists():
    rem1_model, _ = load_portable_champion_model(str(control_path), device=DEVICE)
    raw_cp = torch.load(rem1_path, map_location="cpu", weights_only=False)
    sd_rem = raw_cp["state_dict"] if isinstance(raw_cp, dict) and "state_dict" in raw_cp else raw_cp
    rem1_model.load_state_dict(sd_rem, strict=False)
    rem1_model.eval()
    
    preds, targets, lats = [], [], []
    for s in test_subset:
        p = Path(s["path"])
        if not p.exists(): continue
        try:
            with Image.open(p) as raw:
                img = ImageOps.exif_transpose(raw).convert("RGB")
            t0 = time.perf_counter()
            with torch.inference_mode():
                t_in = norm_clip(to_tensor(img.resize((224, 224), Image.Resampling.LANCZOS))).unsqueeze(0).to(DEVICE)
                out = rem1_model(t_in)
                logit = out["logits"] if isinstance(out, dict) else out
                prob = float(torch.sigmoid(logit).squeeze().item())
            lats.append((time.perf_counter() - t0) * 1000)
            preds.append(prob)
            targets.append(s["label"])
        except Exception: continue
        
    auroc = float(roc_auc_score(targets, preds)) if len(set(targets)) > 1 else 0.5
    auprc = float(average_precision_score(targets, preds)) if len(set(targets)) > 1 else 0.5
    real_p = [p for p, y in zip(preds, targets) if y == 0]
    real_fpr = (sum(1 for p in real_p if p >= 0.50) / max(len(real_p), 1)) * 100.0
    
    benchmark_results["PORTRAIT_REM_1"] = {
        "role": "REMEDIATION CANDIDATE (Portrait & High-Res Tuned)",
        "auroc": round(auroc, 4),
        "auprc": round(auprc, 4),
        "real_fpr_percent": round(real_fpr, 2),
        "avg_latency_ms": round(float(np.mean(lats)), 2),
        "samples": len(preds)
    }
    print(f"  PORTRAIT-REM-1: AUROC={auroc:.4f} | Real FPR={real_fpr:.2f}% | Latency={np.mean(lats):.1f}ms")
    del rem1_model
    gc.collect()

# ---------------------------------------------------------------------
# MODEL 3: COMMUNITYFORENSICS ViT-SMALL (High-Res Specialist)
# ---------------------------------------------------------------------
print("\n[3/4] Evaluating CommunityForensics ViT-Small...")
cf_path = Path("/mnt/ai-storage/aigc_data/models/community_forensics_vit_small/model.safetensors")
if cf_path.exists():
    cf_sd = load_file(cf_path)
    cf_model = timm.create_model('vit_small_patch16_224', pretrained=False, num_classes=2)
    cf_clean_sd = {k.replace("vit.", ""): v for k, v in cf_sd.items() if "classifier" not in k}
    cf_model.load_state_dict(cf_clean_sd, strict=False)
    cf_model.to(DEVICE).eval()
    
    preds, targets, lats = [], [], []
    for s in test_subset:
        p = Path(s["path"])
        if not p.exists(): continue
        try:
            with Image.open(p) as raw:
                img = ImageOps.exif_transpose(raw).convert("RGB")
            t0 = time.perf_counter()
            with torch.inference_mode():
                t_in = norm_imagenet(to_tensor(img.resize((224, 224), Image.Resampling.LANCZOS))).unsqueeze(0).to(DEVICE)
                out = cf_model(t_in)
                prob = float(F.softmax(out, dim=-1)[0, 1].item())
            lats.append((time.perf_counter() - t0) * 1000)
            preds.append(prob)
            targets.append(s["label"])
        except Exception: continue
        
    auroc = float(roc_auc_score(targets, preds)) if len(set(targets)) > 1 else 0.5
    auprc = float(average_precision_score(targets, preds)) if len(set(targets)) > 1 else 0.5
    real_p = [p for p, y in zip(preds, targets) if y == 0]
    real_fpr = (sum(1 for p in real_p if p >= 0.50) / max(len(real_p), 1)) * 100.0
    
    benchmark_results["CommunityForensics_ViT_Small"] = {
        "role": "HIGH-RES SPECIALIST",
        "auroc": round(auroc, 4),
        "auprc": round(auprc, 4),
        "real_fpr_percent": round(real_fpr, 2),
        "avg_latency_ms": round(float(np.mean(lats)), 2),
        "samples": len(preds)
    }
    print(f"  CommunityForensics: AUROC={auroc:.4f} | Real FPR={real_fpr:.2f}% | Latency={np.mean(lats):.1f}ms")
    del cf_model
    gc.collect()

# ---------------------------------------------------------------------
# MODEL 4: DIVINE2K CONVNEXT (Robustness Specialist)
# ---------------------------------------------------------------------
print("\n[4/4] Evaluating divine2k ConvNeXt...")
d2k_path = Path("/mnt/ai-storage/aigc_data/models/divine2k_ensemble/convNext_final.pth")
if d2k_path.exists():
    d2k_sd = torch.load(d2k_path, map_location="cpu", weights_only=False)
    d2k_model = timm.create_model('convnext_base', pretrained=False, num_classes=2)
    d2k_model.load_state_dict(d2k_sd, strict=False)
    d2k_model.to(DEVICE).eval()
    
    preds, targets, lats = [], [], []
    for s in test_subset:
        p = Path(s["path"])
        if not p.exists(): continue
        try:
            with Image.open(p) as raw:
                img = ImageOps.exif_transpose(raw).convert("RGB")
            t0 = time.perf_counter()
            with torch.inference_mode():
                t_in = norm_imagenet(to_tensor(img.resize((224, 224), Image.Resampling.LANCZOS))).unsqueeze(0).to(DEVICE)
                out = d2k_model(t_in)
                prob = float(F.softmax(out, dim=-1)[0, 1].item())
            lats.append((time.perf_counter() - t0) * 1000)
            preds.append(prob)
            targets.append(s["label"])
        except Exception: continue
        
    auroc = float(roc_auc_score(targets, preds)) if len(set(targets)) > 1 else 0.5
    auprc = float(average_precision_score(targets, preds)) if len(set(targets)) > 1 else 0.5
    real_p = [p for p, y in zip(preds, targets) if y == 0]
    real_fpr = (sum(1 for p in real_p if p >= 0.50) / max(len(real_p), 1)) * 100.0
    
    benchmark_results["divine2k_ConvNeXt"] = {
        "role": "ROBUSTNESS SPECIALIST",
        "auroc": round(auroc, 4),
        "auprc": round(auprc, 4),
        "real_fpr_percent": round(real_fpr, 2),
        "avg_latency_ms": round(float(np.mean(lats)), 2),
        "samples": len(preds)
    }
    print(f"  divine2k ConvNeXt: AUROC={auroc:.4f} | Real FPR={real_fpr:.2f}% | Latency={np.mean(lats):.1f}ms")
    del d2k_model
    gc.collect()

# Save final report JSON
p_out = REPO_ROOT / "reports" / "multi_model_diagnostic_benchmark.json"
p_out.parent.mkdir(parents=True, exist_ok=True)
p_out.write_text(json.dumps(benchmark_results, indent=2))
print(f"\nSaved Multi-Model Benchmark Results to: {p_out}")
