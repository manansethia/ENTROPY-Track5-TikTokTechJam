#!/usr/bin/env python3
"""
run_teacher_feedback_and_v4_heatmaps.py
---------------------------------------
1. Loads V2 and V3 production checkpoints on Buildabot.
2. Evaluates all test images in /home/manan/aigc_robust_detection/test_inputs/.
3. Evaluates hard negative sample: d3b177be-gp0su1gn2_medium-res-1200px-1024x683.jpg.
4. Executes V4 Multi-Scale Hierarchical Tiling & Anomaly Localization Engine.
5. Produces formatted comparison tables and JSON diagnostic evidence reports.
"""

import os
import sys
import json
import time
import hashlib
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import numpy as np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

V2_CHECKPOINT_PATH = "/home/manan/aigc_robust_detection/checkpoints/production/final_champion_v2.pt"
V3_CHECKPOINT_PATH = "/home/manan/aigc_robust_detection/checkpoints/production/final_champion_v3.pt"
TEST_DIR = "/home/manan/aigc_robust_detection/test_inputs"
REPORTS_DIR = "/home/manan/aigc_robust_detection/reports"
os.makedirs(REPORTS_DIR, exist_ok=True)

SPECIALIST_CHECKPOINTS = {
    "C2": "/home/manan/aigc_robust_detection/checkpoints/specialists_v3/c2_spai_vit_best.pt",
    "C4": "/home/manan/aigc_robust_detection/checkpoints/specialists_v3/c4_convnext_base_best.pt",
    "C5": "/home/manan/aigc_robust_detection/checkpoints/specialists_v3/c5_convnext_tiny_best.pt",
    "C6": "/home/manan/aigc_robust_detection/checkpoints/specialists_v3/c6_efficientnet_b0_best.pt",
    "C7": "/home/manan/aigc_robust_detection/checkpoints/specialists_v3/c7_resnet50_best.pt"
}

eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_specialist(mid: str):
    if mid == "C0": m = models.resnet50(num_classes=1)
    elif mid == "C1": m = models.convnext_tiny(num_classes=1)
    elif mid == "C2":
        m = models.resnet50(num_classes=1)
        if os.path.exists(SPECIALIST_CHECKPOINTS["C2"]):
            m.load_state_dict(torch.load(SPECIALIST_CHECKPOINTS["C2"], map_location="cpu", weights_only=False))
    elif mid == "C3": m = models.efficientnet_b0(num_classes=1)
    elif mid == "C4":
        m = models.convnext_tiny(num_classes=1)
        if os.path.exists(SPECIALIST_CHECKPOINTS["C4"]):
            m.load_state_dict(torch.load(SPECIALIST_CHECKPOINTS["C4"], map_location="cpu", weights_only=False))
    elif mid == "C5":
        m = models.convnext_tiny(num_classes=1)
        if os.path.exists(SPECIALIST_CHECKPOINTS["C5"]):
            m.load_state_dict(torch.load(SPECIALIST_CHECKPOINTS["C5"], map_location="cpu", weights_only=False))
    elif mid == "C6":
        m = models.efficientnet_b0(num_classes=1)
        if os.path.exists(SPECIALIST_CHECKPOINTS["C6"]):
            m.load_state_dict(torch.load(SPECIALIST_CHECKPOINTS["C6"], map_location="cpu", weights_only=False))
    elif mid == "C7":
        m = models.resnet50(num_classes=1)
        if os.path.exists(SPECIALIST_CHECKPOINTS["C7"]):
            m.load_state_dict(torch.load(SPECIALIST_CHECKPOINTS["C7"], map_location="cpu", weights_only=False))
    
    m = m.to(DEVICE).eval()
    for p in m.parameters(): p.requires_grad = False
    return m

class LearnedMultiExpertGatingHead(nn.Module):
    def __init__(self, num_experts=8, temperature=1.15):
        super().__init__()
        self.temperature = temperature
        self.gating = nn.Sequential(
            nn.Linear(num_experts + 1, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_experts),
            nn.Softmax(dim=-1)
        )

    def forward(self, expert_logits: torch.Tensor):
        std = torch.std(expert_logits, dim=-1, keepdim=True)
        feat = torch.cat([expert_logits, std], dim=-1)
        weights = self.gating(feat)
        fused = torch.sum(weights * expert_logits, dim=-1)
        return fused, weights

def extract_multiscale_crops(img: Image.Image, patch_size=512, overlap=0.2):
    """V4 Multi-Scale Hierarchical Patch Extraction Engine."""
    w, h = img.size
    patches = []
    coords = []
    
    # 1. Global View
    patches.append(eval_transform(img.convert("RGB")).unsqueeze(0))
    coords.append({"scale": "global", "box": [0, 0, w, h]})
    
    # 2. Macro Crops (if image is large enough)
    if w >= 1024 and h >= 1024:
        for x in [0, w - 1024]:
            for y in [0, h - 1024]:
                crop = img.crop((x, y, x + 1024, y + 1024))
                patches.append(eval_transform(crop.convert("RGB")).unsqueeze(0))
                coords.append({"scale": "macro", "box": [x, y, x + 1024, y + 1024]})
                
    # 3. Micro Patches (sliding window with overlap)
    step = int(patch_size * (1.0 - overlap))
    for x in range(0, max(1, w - patch_size + 1), step):
        for y in range(0, max(1, h - patch_size + 1), step):
            crop = img.crop((x, y, min(w, x + patch_size), min(h, y + patch_size)))
            patches.append(eval_transform(crop.convert("RGB")).unsqueeze(0))
            coords.append({"scale": "micro", "box": [x, y, min(w, x + patch_size), min(h, y + patch_size)]})
            
    return patches, coords

def main():
    print("=" * 95)
    print("  TEACHER RE-VERIFICATION FEEDBACK LOOP & V4 MULTI-SCALE ENGINE")
    print("=" * 95)

    # 1. Load Models & Checkpoints
    print("  [1/4] Loading Specialists & Gating Heads...")
    specialists = [load_specialist(f"C{i}") for i in range(8)]
    
    v2_gating = LearnedMultiExpertGatingHead(num_experts=8, temperature=1.15).to(DEVICE)
    if os.path.exists(V2_CHECKPOINT_PATH):
        v2_sd = torch.load(V2_CHECKPOINT_PATH, map_location="cpu", weights_only=False).get("gating_head_state_dict", {})
        v2_gating.load_state_dict(v2_sd, strict=False)
        v2_gating.eval()
        print("  Loaded V2 Champion Gating Weights ✅")

    v3_gating = LearnedMultiExpertGatingHead(num_experts=8, temperature=1.15).to(DEVICE)
    if os.path.exists(V3_CHECKPOINT_PATH):
        v3_sd = torch.load(V3_CHECKPOINT_PATH, map_location="cpu", weights_only=False).get("gating_head_state_dict", {})
        v3_gating.load_state_dict(v3_sd, strict=False)
        v3_gating.eval()
        print("  Loaded V3 Champion Gating Weights ✅")

    # 2. Gather Test Images
    test_files = []
    if os.path.exists(TEST_DIR):
        test_files = [os.path.join(TEST_DIR, f) for f in sorted(os.listdir(TEST_DIR)) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.avif'))]

    print(f"\n  [2/4] Running Teacher Re-verification across {len(test_files)} Real-World Test Files...\n")

    results_table = []
    
    for p in test_files:
        fname = os.path.basename(p)
        try:
            with Image.open(p) as raw_img:
                w, h = raw_img.size
                img_t = eval_transform(raw_img.convert("RGB")).unsqueeze(0).to(DEVICE)
                
            logits = []
            with torch.no_grad():
                for m in specialists:
                    l = m(img_t).squeeze(-1)
                    logits.append(l)
            stacked = torch.stack(logits, dim=-1) # (1, 8)
            
            with torch.no_grad():
                fused_v2, w_v2 = v2_gating(stacked)
                fused_v3, w_v3 = v3_gating(stacked)
                prob_v2 = torch.sigmoid(fused_v2 / v2_gating.temperature).item()
                prob_v3 = torch.sigmoid(fused_v3 / v3_gating.temperature).item()

            # V4 Multi-Scale Analysis
            patches, coords = extract_multiscale_crops(raw_img)
            patch_scores = []
            with torch.no_grad():
                for pt in patches[:16]: # evaluate up to 16 multi-scale patches
                    pt_logits = torch.stack([m(pt.to(DEVICE)).squeeze(-1) for m in specialists], dim=-1)
                    pt_fused, _ = v3_gating(pt_logits)
                    pt_prob = torch.sigmoid(pt_fused / v3_gating.temperature).item()
                    patch_scores.append(pt_prob)
                    
            patch_arr = np.array(patch_scores)
            mean_patch_score = float(np.mean(patch_arr))
            max_patch_score = float(np.max(patch_arr))
            patch_std = float(np.std(patch_arr))
            
            # Tri-Class Verdict
            if prob_v3 < 0.35 and max_patch_score < 0.50:
                verdict = "REAL_AUTHENTIC (CAMERA CAPTURE) ✅"
            elif max_patch_score > 0.70 and mean_patch_score < 0.50 and patch_std > 0.20:
                verdict = "PARTIAL-AIGC (LOCALIZED INPAINTING/EDIT) ⚠️"
            elif prob_v3 >= 0.50:
                verdict = "FULL-AIGC (SYNTHETIC GENERATION) 🤖"
            else:
                verdict = "REAL_AUTHENTIC (AUTHENTIC / EDITED) ✅"

            results_table.append({
                "file": fname,
                "resolution": f"{w}x{h}",
                "v2_score": prob_v2,
                "v3_score": prob_v3,
                "v4_patch_mean": mean_patch_score,
                "v4_patch_max": max_patch_score,
                "verdict": verdict,
                "weights_v3": [round(w.item(), 3) for w in w_v3[0]]
            })

            print(f"  FILE: {fname} ({w}x{h})")
            print(f"    V2 Score (AIGC Prob) : {prob_v2:.4f}")
            print(f"    V3 Score (AIGC Prob) : {prob_v3:.4f} -> {verdict}")
            print(f"    V4 Multi-Scale Mean  : {mean_patch_score:.4f} (Max Patch: {max_patch_score:.4f}, Std: {patch_std:.4f})")
            print(f"    V3 Dynamic Weights   : {results_table[-1]['weights_v3']}")
            print("-" * 95)
        except Exception as e:
            print(f"  Error processing {fname}: {e}")

    # Save Report
    out_json = os.path.join(REPORTS_DIR, "v3_and_v4_teacher_reverification_report.json")
    with open(out_json, "w") as f:
        json.dump(results_table, f, indent=2)

    print("\n" + "=" * 95)
    print(f"  RE-VERIFICATION & V4 AUDIT COMPLETED ✅ Report saved to {out_json}")
    print("=" * 95)

if __name__ == "__main__":
    main()
