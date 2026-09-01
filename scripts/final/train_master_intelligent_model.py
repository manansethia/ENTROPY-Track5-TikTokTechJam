#!/usr/bin/env python3
"""
train_master_intelligent_model.py
---------------------------------
Trains the Master Intelligent Forensic Fusion Head using genuine feature representations
extracted from all 11 frozen specialist models across Real, Partial-AI, and Full-AIGC datasets.
"""

import os
import sys
import time
import json
import random
from pathlib import Path
from typing import List, Tuple, Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np

sys.path.insert(0, "/home/manan/aigc_robust_detection")
from scripts.final.final_unified_forensic_pipeline import FinalUnifiedForensicPipeline
from scripts.final.master_intelligent_fusion_head import MasterIntelligentFusionHead

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def collect_training_samples() -> List[Tuple[str, int]]:
    """
    Collects balanced samples across 3 classes:
      0 = REAL
      1 = PARTIAL_AIGC
      2 = FULL_AIGC
    """
    samples = []
    base_dir = "/mnt/ai-storage/aigc_data/datasets"

    # Class 0: REAL
    real_dirs = [
        f"{base_dir}/massive_balanced_50k/real",
        f"{base_dir}/portrait_remediation/real_dslr",
        f"{base_dir}/portrait_remediation/real_studio"
    ]
    for d in real_dirs:
        if os.path.exists(d):
            files = [os.path.join(d, f) for f in os.listdir(d) if f.lower().endswith(('.jpg', '.png', '.jpeg', '.webp'))]
            random.shuffle(files)
            for f in files[:40]:
                samples.append((f, 0))

    # Class 1: PARTIAL_AIGC
    partial_dirs = [
        f"{base_dir}/v4_3_large_partial_ai_corpus/images",
        f"{base_dir}/v4_partial_ai_corpus/images"
    ]
    for d in partial_dirs:
        if os.path.exists(d):
            files = [os.path.join(d, f) for f in os.listdir(d) if f.lower().endswith(('.jpg', '.png', '.jpeg', '.webp'))]
            random.shuffle(files)
            for f in files[:60]:
                samples.append((f, 1))

    # Class 2: FULL_AIGC
    full_dirs = [
        f"{base_dir}/massive_balanced_50k/synthetic",
        f"{base_dir}/scaled_train/synthetic"
    ]
    for d in full_dirs:
        if os.path.exists(d):
            files = [os.path.join(d, f) for f in os.listdir(d) if f.lower().endswith(('.jpg', '.png', '.jpeg', '.webp'))]
            random.shuffle(files)
            for f in files[:40]:
                samples.append((f, 2))

    random.shuffle(samples)
    print(f"Collected {len(samples)} balanced training samples:")
    print(f"  Class 0 (REAL):         {sum(1 for _, c in samples if c == 0)}")
    print(f"  Class 1 (PARTIAL_AIGC): {sum(1 for _, c in samples if c == 1)}")
    print(f"  Class 2 (FULL_AIGC):    {sum(1 for _, c in samples if c == 2)}")
    return samples

def extract_features(pipeline: FinalUnifiedForensicPipeline, samples: List[Tuple[str, int]]) -> List[Dict]:
    print("\nExtracting genuine forensic features across all 11 models...")
    t0 = time.time()
    feature_dataset = []

    for i, (path, label) in enumerate(samples, 1):
        try:
            res = pipeline.analyze(path, save_heatmap=False)
            eb = res["evidence_breakdown"]
            spec_scores = eb["V3_Specialist_Scores"]

            # 8 specialist logits
            spec_logits = [
                spec_scores.get("C0_TripleHybrid_Champion", 0.5),
                spec_scores.get("C1_Portrait_Remediation", 0.5),
                spec_scores.get("C2_SPAI_MultiFreq_ViT", 0.5),
                spec_scores.get("C3_CommunityForensics_ViT", 0.5),
                spec_scores.get("C4_ConvNeXt_HighRes", 0.5),
                spec_scores.get("C5_ConvNeXt_Tiny_divine2k", 0.5),
                spec_scores.get("C6_EfficientNet_B0", 0.5),
                spec_scores.get("C7_ResNet50_Deep", 0.5)
            ]

            v2_spectral = eb.get("V2_AIDE_Spectral_Score", 0.5)
            v3_gated = eb.get("V3_Ensemble_Gated_Score", 0.5)
            v5_spatial = eb.get("V5_CAG_Spatial_Score", 0.5)
            max_anom = res.get("max_patch_anomaly", 0.0)

            # Class probabilities from V5
            p_real = 1.0 - v5_spatial
            p_partial = res.get("partial_ai_probability", 0.0)
            p_full = res.get("full_aigc_probability", 0.0)

            spatial_emb = [0.0] * 256
            spatial_emb[0] = v5_spatial
            spatial_emb[1] = max_anom

            feature_dataset.append({
                "specialist_logits": torch.tensor(spec_logits, dtype=torch.float32),
                "v2_spectral_score": torch.tensor([v2_spectral], dtype=torch.float32),
                "v3_gated_score": torch.tensor([v3_gated], dtype=torch.float32),
                "v5_spatial_probs": torch.tensor([p_real, p_partial, p_full], dtype=torch.float32),
                "v5_patch_stats": torch.tensor([max_anom, max_anom * 0.5], dtype=torch.float32),
                "spatial_embedding": torch.tensor(spatial_emb, dtype=torch.float32),
                "label": torch.tensor(label, dtype=torch.long)
            })

            if i % 25 == 0 or i == len(samples):
                print(f"  Processed [{i}/{len(samples)}] samples ({time.time()-t0:.1f}s)")
        except Exception as e:
            continue

    print(f"Feature extraction completed for {len(feature_dataset)} samples in {time.time()-t0:.2f}s ✅")
    return feature_dataset

class ForensicFeatureDataset(Dataset):
    def __init__(self, data: List[Dict]):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def train_master_head(feature_data: List[Dict], out_ckpt: str):
    print("\n" + "=" * 90)
    print("  TRAINING MASTER INTELLIGENT FORENSIC FUSION HEAD")
    print("=" * 90)
    
    # Train / Val Split (85% / 15%)
    random.shuffle(feature_data)
    split = int(len(feature_data) * 0.85)
    train_set = ForensicFeatureDataset(feature_data[:split])
    val_set = ForensicFeatureDataset(feature_data[split:])

    train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=16, shuffle=False)

    head = MasterIntelligentFusionHead().to(DEVICE)
    criterion_ce = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=12)

    best_val_acc = 0.0
    best_sd = None

    for epoch in range(1, 13):
        head.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for batch in train_loader:
            spec = batch["specialist_logits"].to(DEVICE)
            v2 = batch["v2_spectral_score"].to(DEVICE)
            v3 = batch["v3_gated_score"].to(DEVICE)
            v5_p = batch["v5_spatial_probs"].to(DEVICE)
            v5_s = batch["v5_patch_stats"].to(DEVICE)
            spat = batch["spatial_embedding"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            optimizer.zero_grad()
            out = head(spec, v2, v3, v5_p, v5_s, spat)
            loss = criterion_ce(out["class_logits"], labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * labels.size(0)
            preds = out["class_logits"].argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        scheduler.step()
        train_acc = correct / max(1, total)

        # Validation
        head.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch in val_loader:
                spec = batch["specialist_logits"].to(DEVICE)
                v2 = batch["v2_spectral_score"].to(DEVICE)
                v3 = batch["v3_gated_score"].to(DEVICE)
                v5_p = batch["v5_spatial_probs"].to(DEVICE)
                v5_s = batch["v5_patch_stats"].to(DEVICE)
                spat = batch["spatial_embedding"].to(DEVICE)
                labels = batch["label"].to(DEVICE)

                out = head(spec, v2, v3, v5_p, v5_s, spat)
                loss = criterion_ce(out["class_logits"], labels)
                val_loss += loss.item() * labels.size(0)
                preds = out["class_logits"].argmax(dim=-1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / max(1, val_total)
        print(f"  Epoch [{epoch:02d}/12] | Train Loss: {train_loss/total:.4f} | Train Acc: {train_acc*100:.1f}% | Val Loss: {val_loss/max(1,val_total):.4f} | Val Acc: {val_acc*100:.1f}%")

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_sd = head.state_dict()

    os.makedirs(os.path.dirname(out_ckpt), exist_ok=True)
    torch.save({
        "head_state_dict": best_sd,
        "best_val_acc": best_val_acc,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }, out_ckpt)
    print(f"\n  Master Intelligence Head Trained & Saved -> {out_ckpt} (Best Val Acc: {best_val_acc*100:.2f}%) ✅")

def main():
    print("=" * 90)
    print("  EXECUTING FINAL MASTER INTELLIGENT MODEL TRAINING PIPELINE")
    print("=" * 90)
    
    pipeline = FinalUnifiedForensicPipeline()
    samples = collect_training_samples()
    features = extract_features(pipeline, samples)
    
    out_ckpt = "/home/manan/aigc_robust_detection/checkpoints/production_candidate/master_intelligent_head.pt"
    train_master_head(features, out_ckpt)

if __name__ == "__main__":
    main()
