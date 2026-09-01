#!/usr/bin/env python3
"""Tri-Hybrid Dynamic Gating Robust Detector Training Engine.
Trains multi-stream fusion network combining SigLIP, CLIP, and ConvNeXt frequency branch.
"""

import argparse
import io
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from tqdm import tqdm

from scripts.augmentations import PERTURBATIONS, evaluation_transform


class CachedFeaturesDataset(Dataset):
    def __init__(self, h5_path):
        with h5py.File(h5_path, "r") as f:
            self.siglip_feats = torch.tensor(f["siglip_features"][:], dtype=torch.float32)
            self.clip_feats = torch.tensor(f["clip_features"][:], dtype=torch.float32)
            self.has_dinov2 = "dinov2_features" in f
            if self.has_dinov2:
                self.dinov2_feats = torch.tensor(f["dinov2_features"][:], dtype=torch.float32)
            else:
                self.dinov2_feats = None
            self.labels = torch.tensor(f["labels"][:], dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if self.has_dinov2:
            return self.siglip_feats[idx], self.clip_feats[idx], self.dinov2_feats[idx], self.labels[idx]
        return self.siglip_feats[idx], self.clip_feats[idx], self.labels[idx]


class DynamicGatingFusionHead(nn.Module):
    def __init__(self, in_dim_siglip=768, in_dim_clip=1024, in_dim_dinov2=1024, embed_dim=512, has_dinov2=False, num_classes=2):
        super().__init__()
        self.has_dinov2 = has_dinov2
        self.proj_siglip = nn.Sequential(
            nn.Linear(in_dim_siglip, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(0.2),
        )
        self.proj_clip = nn.Sequential(
            nn.Linear(in_dim_clip, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(0.2),
        )
        if self.has_dinov2:
            self.proj_dinov2 = nn.Sequential(
                nn.Linear(in_dim_dinov2, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.GELU(),
                nn.Dropout(0.2),
            )
            router_in = in_dim_siglip + in_dim_clip + in_dim_dinov2
            num_streams = 3
        else:
            router_in = in_dim_siglip + in_dim_clip
            num_streams = 2

        # Dynamic Gating Router
        self.gate_router = nn.Sequential(
            nn.Linear(router_in, 256),
            nn.GELU(),
            nn.Linear(256, num_streams),
        )
        # Final Robust Classifier
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, feat_siglip, feat_clip, feat_dinov2=None):
        p_siglip = self.proj_siglip(feat_siglip)
        p_clip = self.proj_clip(feat_clip)

        if self.has_dinov2 and feat_dinov2 is not None:
            p_dino = self.proj_dinov2(feat_dinov2)
            concat_feats = torch.cat([feat_siglip, feat_clip, feat_dinov2], dim=-1)
            gates = F.softmax(self.gate_router(concat_feats), dim=-1)  # [B, 3]
            fused = gates[:, 0:1] * p_siglip + gates[:, 1:2] * p_clip + gates[:, 2:3] * p_dino
        else:
            concat_feats = torch.cat([feat_siglip, feat_clip], dim=-1)
            gates = F.softmax(self.gate_router(concat_feats), dim=-1)  # [B, 2]
            fused = gates[:, 0:1] * p_siglip + gates[:, 1:2] * p_clip

        logits = self.classifier(fused)
        return logits, gates


def main():
    p = argparse.ArgumentParser(description="Train Tri-Hybrid Dynamic Gating Detector")
    p.add_argument("--cache_h5", default="/mnt/ai-storage/aigc_data/cache/features.h5")
    p.add_argument("--output_dir", default="checkpoints/tri_hybrid_v1")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print(f"Training on device: {device}")

    ds = CachedFeaturesDataset(args.cache_h5)
    train_size = int(0.85 * len(ds))
    val_size = len(ds) - train_size
    train_ds, val_ds = torch.utils.data.random_split(ds, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    print(f"Loaded dataset: {len(ds)} samples ({train_size} train, {val_size} val)")

    model = DynamicGatingFusionHead(has_dinov2=ds.has_dinov2).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Active Trainable Parameters: {total_params:,} (Strict limit: <2,000,000,000)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    scaler = GradScaler()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_val_auc = 0.0
    history = []

    print("\nStarting Dynamic Gating training loop...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{args.epochs:02d} [Train]")
        for batch in pbar:
            if ds.has_dinov2:
                f_siglip, f_clip, f_dino, labels = batch
                f_siglip, f_clip, f_dino, labels = f_siglip.to(device), f_clip.to(device), f_dino.to(device), labels.to(device)
            else:
                f_siglip, f_clip, labels = batch
                f_siglip, f_clip, labels = f_siglip.to(device), f_clip.to(device), labels.to(device)
                f_dino = None

            optimizer.zero_grad()

            with autocast():
                logits, gates = model(f_siglip, f_clip, f_dino)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{scheduler.get_last_lr()[0]:.6f}"})

        scheduler.step()
        train_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        y_true, y_prob, y_pred = [], [], []
        gate_weights = []

        with torch.no_grad():
            for batch in val_loader:
                if ds.has_dinov2:
                    f_siglip, f_clip, f_dino, labels = batch
                    f_siglip, f_clip, f_dino, labels = f_siglip.to(device), f_clip.to(device), f_dino.to(device), labels.to(device)
                else:
                    f_siglip, f_clip, labels = batch
                    f_siglip, f_clip, labels = f_siglip.to(device), f_clip.to(device), labels.to(device)
                    f_dino = None

                with autocast():
                    logits, gates = model(f_siglip, f_clip, f_dino)
                    loss = criterion(logits, labels)
                
                probs = F.softmax(logits, dim=-1)[:, 1]
                val_loss += loss.item()
                y_true.extend(labels.cpu().numpy())
                y_prob.extend(probs.cpu().numpy())
                y_pred.extend((probs >= 0.5).cpu().long().numpy())
                gate_weights.append(gates.cpu().numpy())

        val_loss /= len(val_loader)
        acc = accuracy_score(y_true, y_pred)
        b_acc = balanced_accuracy_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_prob)
        avg_gates = np.concatenate(gate_weights, axis=0).mean(axis=0)

        print(f"Epoch {epoch:02d} Summary: Train Loss={train_loss:.4f} | Val Loss={val_loss:.4f} | "
              f"Acc={acc:.4f} | B-Acc={b_acc:.4f} | AUROC={auc:.4f} | "
              f"Gates [SigLIP: {avg_gates[0]:.3f}, CLIP: {avg_gates[1]:.3f}]")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_acc": acc,
            "val_bacc": b_acc,
            "val_auroc": auc,
            "gate_siglip": avg_gates[0],
            "gate_clip": avg_gates[1],
        })

        if auc > best_val_auc:
            best_val_auc = auc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_auroc": auc,
                "val_acc": acc,
            }, out_dir / "best_model.pt")
            print(f"  --> Saved NEW BEST checkpoint (AUROC: {auc:.4f}) to {out_dir / 'best_model.pt'}!")

    # Save training log
    pd.DataFrame(history).to_csv(out_dir / "training_history.csv", index=False)
    print(f"\nTRAINING COMPLETE! Best Validation AUROC: {best_val_auc:.4f}")


if __name__ == "__main__":
    main()
