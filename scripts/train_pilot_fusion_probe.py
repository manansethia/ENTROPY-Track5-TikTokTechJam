# =====================================================================================
# PILOT_TRAINING_PARTIAL_DATA: FAST SANITY PROBE (1,000 SAMPLES)
# Evaluates gradient dynamics and gating head convergence before Full-Data Ingestion
# =====================================================================================

import os, sys, time, json, random, hashlib, gc
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

print("=" * 85)
print("  PILOT TRAINING: FAST PROBE ON 1,000 SAMPLES (NOT FINAL PRODUCTION)")
print("=" * 85)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load Feature Cache from Phase A
cache_path = "/home/manan/aigc_robust_detection/reports/fusion_features/expert_logits_cache.json"
with open(cache_path, "r") as f:
    cache_data = json.load(f)

labels_arr = np.array(cache_data["labels"], dtype=np.float32)
expert_dict = cache_data["expert_logits"]

expert_matrix = np.column_stack([
    expert_dict["C0_Champion_Frozen"],
    expert_dict["C1_Portrait_REM1_E3"],
    expert_dict["C2_SPAI_TFG"],
    expert_dict["C3_CommunityForensics_ViT"],
    expert_dict["C4_divine2k_ConvNeXt"]
]).astype(np.float32)

# F1: Equal / Weighted Logit Fusion
f1_preds = 1.0 / (1.0 + np.exp(-np.mean(expert_matrix, axis=1)))
f1_auc = roc_auc_score(labels_arr, f1_preds)

# F2: Logistic Regression Fusion
lr_fusion = LogisticRegression()
lr_fusion.fit(expert_matrix, labels_arr)
f2_preds = lr_fusion.predict_proba(expert_matrix)[:, 1]
f2_auc = roc_auc_score(labels_arr, f2_preds)

# F3: Learned MLP Gating Head
class LearnedGatingHead(nn.Module):
    def __init__(self, num_experts=5):
        super().__init__()
        self.gating = nn.Sequential(
            nn.Linear(num_experts + 1, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Linear(32, num_experts),
            nn.Softmax(dim=-1)
        )
    def forward(self, x):
        std = torch.std(x, dim=-1, keepdim=True)
        feat = torch.cat([x, std], dim=-1)
        weights = self.gating(feat)
        return torch.sum(weights * x, dim=-1), weights

gating_model = LearnedGatingHead(num_experts=5).to(DEVICE)
opt_gating = torch.optim.AdamW(gating_model.parameters(), lr=1e-3)
crit_gating = nn.BCEWithLogitsLoss()

t_mat = torch.tensor(expert_matrix, dtype=torch.float32, device=DEVICE)
t_lbl = torch.tensor(labels_arr, dtype=torch.float32, device=DEVICE)

for ep in range(100):
    opt_gating.zero_grad()
    out, _ = gating_model(t_mat)
    loss = crit_gating(out, t_lbl)
    loss.backward()
    opt_gating.step()

with torch.no_grad():
    f3_logits, weights = gating_model(t_mat)
    f3_preds = torch.sigmoid(f3_logits).cpu().numpy()
    f3_auc = roc_auc_score(labels_arr, f3_preds)

print("\n" + "=" * 85)
print("  PILOT PROBE FUSION RESULTS (1,000 SAMPLES):")
print(f"  F1 (Weighted Logit Fusion)       : AUROC = {f1_auc:.5f}")
print(f"  F2 (Logistic Regression Fusion)  : AUROC = {f2_auc:.5f}")
print(f"  F3 (Learned MLP Gating Head)     : AUROC = {f3_auc:.5f}")
print("=" * 85)

os.makedirs("/home/manan/aigc_robust_detection/reports", exist_ok=True)
with open("/home/manan/aigc_robust_detection/reports/pilot_fusion_probe.json", "w") as f:
    json.dump({
        "status": "PILOT_TRAINING_PARTIAL_DATA",
        "pilot_samples": len(labels_arr),
        "f1_auc": float(f1_auc),
        "f2_auc": float(f2_auc),
        "f3_auc": float(f3_auc)
    }, f, indent=2)

print("  Saved Pilot Report: reports/pilot_fusion_probe.json")
print("=" * 85)
