#!/usr/bin/env python3
"""
fast_train_master_intelligent_head.py
-------------------------------------
High-speed GPU trainer for Master Intelligent Forensic Fusion Head.
Extracts representations using the compiled FP16 master model on cuda:0
and trains the cross-attention fusion head in seconds.
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
import torchvision.transforms as T
import numpy as np

sys.path.insert(0, "/home/manan/aigc_robust_detection")
from scripts.final.compile_master_unified_model import MasterUnifiedForensicModel
from scripts.final.master_intelligent_fusion_head import MasterIntelligentFusionHead

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def collect_balanced_samples() -> List[Tuple[str, int]]:
    samples = []
    base_dir = "/mnt/ai-storage/aigc_data/datasets"

    # Class 0: REAL (20 samples)
    real_dirs = [
        f"{base_dir}/massive_balanced_50k/real",
        f"{base_dir}/portrait_remediation/real_dslr",
        f"{base_dir}/portrait_remediation/real_studio"
    ]
    for d in real_dirs:
        if os.path.exists(d):
            files = [os.path.join(d, f) for f in os.listdir(d) if f.lower().endswith(('.jpg', '.png', '.jpeg', '.webp'))]
            random.shuffle(files)
            for f in files[:10]:
                samples.append((f, 0))

    # Class 1: PARTIAL_AIGC (20 samples)
    partial_dirs = [
        f"{base_dir}/v4_3_large_partial_ai_corpus/images",
        f"{base_dir}/v4_partial_ai_corpus/images"
    ]
    for d in partial_dirs:
        if os.path.exists(d):
            files = [os.path.join(d, f) for f in os.listdir(d) if f.lower().endswith(('.jpg', '.png', '.jpeg', '.webp'))]
            random.shuffle(files)
            for f in files[:10]:
                samples.append((f, 1))

    # Class 2: FULL_AIGC (20 samples)
    full_dirs = [
        f"{base_dir}/massive_balanced_50k/synthetic",
        f"{base_dir}/scaled_train/synthetic"
    ]
    for d in full_dirs:
        if os.path.exists(d):
            files = [os.path.join(d, f) for f in os.listdir(d) if f.lower().endswith(('.jpg', '.png', '.jpeg', '.webp'))]
            random.shuffle(files)
            for f in files[:10]:
                samples.append((f, 2))

    random.shuffle(samples)
    print(f"Collected {len(samples)} balanced training samples:")
    print(f"  Class 0 (REAL):         {sum(1 for _, c in samples if c == 0)}")
    print(f"  Class 1 (PARTIAL_AIGC): {sum(1 for _, c in samples if c == 1)}")
    print(f"  Class 2 (FULL_AIGC):    {sum(1 for _, c in samples if c == 2)}")
    return samples

def extract_features_gpu(samples: List[Tuple[str, int]]) -> List[Dict]:
    print("\n[1/3] Loading Compiled Master FP16 Model on cuda:0...")
    t0 = time.time()
    ckpt_path = "/home/manan/aigc_robust_detection/checkpoints/compiled/master_unified_forensic_model_fp16.pt"
    
    model = MasterUnifiedForensicModel().half()
    sd = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(sd["model_state_dict"])
    model = model.to(DEVICE).eval()
    print(f"  Loaded Master Model in {time.time()-t0:.2f}s ✅")

    t_224 = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    t_256 = T.Compose([T.Resize((256, 256)), T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    t_384 = T.Compose([T.Resize((384, 384)), T.ToTensor(), T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

    print(f"\n[2/3] Extracting Genuine Multi-Modal Embeddings for {len(samples)} images on GPU...")
    t_f = time.time()
    feature_dataset = []

    for i, (path, label) in enumerate(samples, 1):
        try:
            img = Image.open(path).convert("RGB")
            img_224 = t_224(img).unsqueeze(0).to(DEVICE, dtype=torch.float16)
            img_256_5v = t_256(img).unsqueeze(0).unsqueeze(1).repeat(1, 5, 1, 1, 1).to(DEVICE, dtype=torch.float16)
            img_384 = t_384(img).unsqueeze(0).to(DEVICE, dtype=torch.float16)
            srm_feats = model.v3_c0_champion.srm_proj[0].weight.new_zeros((1, 36)).to(DEVICE, dtype=torch.float16)

            patch_tensors = [img_224.squeeze(0)]
            patch_coords = torch.tensor([[0.0, 0.0, 1.0, 1.0, 1.0]], dtype=torch.float16, device=DEVICE)
            patch_tensors_t = torch.stack(patch_tensors).to(DEVICE, dtype=torch.float16)

            with torch.no_grad():
                out = model(img_224, img_256_5v, img_384, srm_feats, patch_tensors_t, patch_coords)

                # Extract spatial feature
                g_feat = model.v5_pool(model.v5_backbone(img_224)).flatten(1)
                p_feats = model.v5_pool(model.v5_backbone(patch_tensors_t)).flatten(1)
                pos_emb = model.v5_cag_head.pos_mlp(patch_coords)
                combined = torch.cat([g_feat.expand(p_feats.shape[0], -1), p_feats, pos_emb], dim=-1)
                fused_spatial = model.v5_cag_head.fusion_mlp(combined)
                global_spatial = fused_spatial.mean(dim=0, keepdim=True).float().cpu() # (1, 256)

                spec_logits = torch.stack([v for v in out["specialist_logits"].values()], dim=-1).float().cpu().squeeze(0)
                v2_score = out["v2_spectral_score"].float().cpu().view(1)
                v3_score = out["v3_gated_score"].float().cpu().view(1)
                v5_probs = torch.tensor([out["real_probability"].item(), out["partial_ai_probability"].item(), out["full_aigc_probability"].item()], dtype=torch.float32)
                patch_stats = torch.tensor([out["patch_anomalies"].max().item(), out["patch_anomalies"].mean().item()], dtype=torch.float32)

            feature_dataset.append({
                "specialist_logits": spec_logits,
                "v2_spectral_score": v2_score,
                "v3_gated_score": v3_score,
                "v5_spatial_probs": v5_probs,
                "v5_patch_stats": patch_stats,
                "spatial_embedding": global_spatial.squeeze(0),
                "label": torch.tensor(label, dtype=torch.long)
            })

            if i % 10 == 0 or i == len(samples):
                print(f"  Extracted [{i}/{len(samples)}] samples ({time.time()-t_f:.1f}s)")
        except Exception as e:
            continue

    print(f"Feature extraction finished for {len(feature_dataset)} samples in {time.time()-t_f:.2f}s ✅")
    return feature_dataset

class ForensicFeatureDataset(Dataset):
    def __init__(self, data: List[Dict]):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def train_master_head(feature_data: List[Dict], out_ckpt: str):
    print("\n[3/3] Training Master Intelligent Cross-Attention Fusion Head...")
    random.shuffle(feature_data)
    split = int(len(feature_data) * 0.80)
    train_set = ForensicFeatureDataset(feature_data[:split])
    val_set = ForensicFeatureDataset(feature_data[split:])

    train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=8, shuffle=False)

    head = MasterIntelligentFusionHead().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(head.parameters(), lr=2e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)

    best_val_acc = 0.0
    best_sd = None

    for epoch in range(1, 16):
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
            loss = criterion(out["class_logits"], labels)
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
                loss = criterion(out["class_logits"], labels)
                val_loss += loss.item() * labels.size(0)
                preds = out["class_logits"].argmax(dim=-1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / max(1, val_total)
        print(f"  Epoch [{epoch:02d}/15] | Train Acc: {train_acc*100:.1f}% | Val Acc: {val_acc*100:.1f}% | Loss: {val_loss/max(1,val_total):.4f}")

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_sd = head.state_dict()

    os.makedirs(os.path.dirname(out_ckpt), exist_ok=True)
    torch.save({
        "head_state_dict": best_sd,
        "best_val_acc": best_val_acc,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }, out_ckpt)
    print(f"\n✅ Master Intelligence Head Trained Successfully -> {out_ckpt} (Best Val Acc: {best_val_acc*100:.2f}%)")

def main():
    print("=" * 95)
    print("  FAST GPU MASTER INTELLIGENT FORENSIC FUSION TRAINING PIPELINE")
    print("=" * 95)
    
    samples = collect_balanced_samples()
    features = extract_features_gpu(samples)
    
    out_ckpt = "/home/manan/aigc_robust_detection/checkpoints/production_candidate/master_intelligent_head.pt"
    train_master_head(features, out_ckpt)

if __name__ == "__main__":
    main()
