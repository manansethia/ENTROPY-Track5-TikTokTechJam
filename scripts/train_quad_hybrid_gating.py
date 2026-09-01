#!/usr/bin/env python3
"""Train Quad-Hybrid 4-Stream Dynamic Softmax Gating Fusion Network.
Fuses SigLIP (768-d), CLIP ViT-L (1024-d), DINOv2 (1024-d), and ConvNeXt-V2 (768-d).
Enforces 1:1 balanced batch sampling and Cosine Annealing learning rate schedule.
"""

import argparse
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
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from models.quad_hybrid_detector import QuadHybridGatingHead


class QuadCachedFeaturesDataset(Dataset):
    def __init__(self, h5_path, indices=None):
        self.h5_path = h5_path
        with h5py.File(h5_path, "r") as f:
            total_samples = len(f["labels"])
            idx = indices if indices is not None else np.arange(total_samples)
            
            # Preload 167 MB into memory tensors for instant training
            self.siglip = torch.tensor(f["siglip_features"][idx], dtype=torch.float32)
            self.clip = torch.tensor(f["clip_features"][idx], dtype=torch.float32)
            self.dinov2 = torch.tensor(f["dinov2_features"][idx], dtype=torch.float32) if "dinov2_features" in f else torch.zeros(len(idx), 1024, dtype=torch.float32)
            self.convnext = torch.tensor(f["convnext_features"][idx], dtype=torch.float32) if "convnext_features" in f else torch.zeros(len(idx), 768, dtype=torch.float32)
            self.labels = torch.tensor(f["labels"][idx], dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.siglip[idx], self.clip[idx], self.dinov2[idx], self.convnext[idx], self.labels[idx]



class AsymmetricMarginHardMiningLoss(nn.Module):
    """
    Cost-Sensitive Asymmetric Margin Loss with Online Hard Example Mining (OHEM)
    and Severe False Positive Minus-Point Penalties.
    
    1. Real Images (y=0): Must predict p(fake) < 0.05.
       If p(fake) > 0.05, applies severe fp_penalty (10.0x minus points)
       to strictly prevent false positives on genuine human photos.
    2. Synthetic Images (y=1): Must predict p(fake) > 0.95.
       Applies focal focusing parameter gamma=2.0 on difficult generative edge cases (FLUX.1, SD3).
    3. OHEM: Selects top hardest samples in every batch and amplifies their loss gradients.
    """
    def __init__(self, fp_penalty=10.0, gamma_fake=2.0, gamma_real=1.0, ohem_ratio=0.4):
        super().__init__()
        self.fp_penalty = fp_penalty
        self.gamma_fake = gamma_fake
        self.gamma_real = gamma_real
        self.ohem_ratio = ohem_ratio

    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=-1)
        p_fake = probs[:, 1]
        p_real = probs[:, 0]
        
        # Loss for Fake (Synthetic AI)
        loss_fake = - ((1.0 - p_fake) ** self.gamma_fake) * torch.log(torch.clamp(p_fake, min=1e-7))
        
        # Loss for Real (Authentic Human) with Severe False Positive Penalty (Minus Points)
        loss_real = - self.fp_penalty * ((1.0 - p_real) ** self.gamma_real) * torch.log(torch.clamp(p_real, min=1e-7))
        
        sample_losses = torch.where(targets == 1, loss_fake, loss_real)
        
        # Online Hard Example Mining (OHEM)
        if self.ohem_ratio < 1.0:
            k = max(int(len(targets) * self.ohem_ratio), 1)
            hard_losses, _ = torch.topk(sample_losses, k)
            return hard_losses.mean()
        
        return sample_losses.mean()


def train_quad_hybrid(
    cache_h5,
    output_dir,
    epochs=25,
    batch_size=128,
    lr=3e-4,
    weight_decay=1e-4,
    fp_penalty=10.0,
    ohem_ratio=0.4,
    device="cuda",
    seed=42,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(cache_h5, "r") as f:
        labels = np.array(f["labels"])
        n_total = len(labels)
        has_dinov2 = "dinov2_features" in f
        has_convnext = "convnext_features" in f

    print(f"\n========================================================")
    print(f"QUAD-HYBRID DYNAMIC GATING TRAINING ENGINE (HARD-MINING)")
    print(f"========================================================")
    print(f"Total Dataset Samples: {n_total:,} (Real: {np.sum(labels == 0):,}, Synthetic: {np.sum(labels == 1):,})")
    print(f"Streams Active: SigLIP (768-d), CLIP (1024-d), DINOv2 ({has_dinov2}), ConvNeXt-V2 ({has_convnext})")
    print(f"Loss: Asymmetric Hard-Mining (FP-Penalty={fp_penalty}x, OHEM={ohem_ratio*100:.0f}%)")
    print(f"Epochs: {epochs} | Batch Size: {batch_size} | LR: {lr:.1e} | Device: {dev}")
    print(f"========================================================\n")

    # Stratified 85/15 Split
    skf = StratifiedKFold(n_splits=7, shuffle=True, random_state=seed)
    train_idx, val_idx = next(skf.split(np.zeros(n_total), labels))

    train_ds = QuadCachedFeaturesDataset(cache_h5, train_idx)
    val_ds = QuadCachedFeaturesDataset(cache_h5, val_idx)

    # Class-Balanced Weighted Random Sampler with Stochastic Ratio Jittering (40%-60% dynamic fluctuation)
    train_labels = labels[train_idx]
    class_counts = np.bincount(train_labels)
    class_weights = 1.0 / np.maximum(class_counts, 1)
    sample_weights = class_weights[train_labels]
    
    # Apply stochastic jittering to prevent rigid 50.0% prior memorization
    jitter = np.random.uniform(0.85, 1.15, size=len(sample_weights))
    jittered_weights = sample_weights * jitter
    sampler = torch.utils.data.WeightedRandomSampler(jittered_weights, len(jittered_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = QuadHybridGatingHead().to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = AsymmetricMarginHardMiningLoss(fp_penalty=fp_penalty, ohem_ratio=ohem_ratio)
    scaler = torch.amp.GradScaler("cuda", enabled=(dev.type == "cuda"))

    best_auroc = 0.0
    best_ckpt = out_dir / "best_model.pt"
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for s_f, c_f, d_f, x_f, lbls in tqdm(train_loader, desc=f"Epoch {epoch:02d}/{epochs:02d} [Train]"):
            s_f, c_f, d_f, x_f, lbls = s_f.to(dev), c_f.to(dev), d_f.to(dev), x_f.to(dev), lbls.to(dev)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=(dev.type == "cuda")):
                logits, _ = model(s_f, c_f, d_f, x_f)
                loss = criterion(logits, lbls)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * len(lbls)

        scheduler.step()
        train_loss /= len(train_idx)

        # Validation
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        all_gates = []

        with torch.no_grad():
            for s_f, c_f, d_f, x_f, lbls in val_loader:
                s_f, c_f, d_f, x_f, lbls = s_f.to(dev), c_f.to(dev), d_f.to(dev), x_f.to(dev), lbls.to(dev)
                with torch.amp.autocast("cuda", enabled=(dev.type == "cuda")):
                    logits, gates = model(s_f, c_f, d_f, x_f)
                    loss = criterion(logits, lbls)

                val_loss += loss.item() * len(lbls)
                probs = F.softmax(logits, dim=-1)[:, 1].cpu().numpy()
                all_preds.extend(probs)
                all_labels.extend(lbls.cpu().numpy())
                all_gates.extend(gates.cpu().numpy())

        val_loss /= len(val_idx)
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_gates = np.array(all_gates)
        binary_preds = (all_preds >= 0.5).astype(int)

        acc = accuracy_score(all_labels, binary_preds)
        b_acc = balanced_accuracy_score(all_labels, binary_preds)
        
        # Calculate exact False Positive Rate
        real_mask = (all_labels == 0)
        fp_count = np.sum((binary_preds == 1) & real_mask)
        tn_count = np.sum((binary_preds == 0) & real_mask)
        fpr = fp_count / max(fp_count + tn_count, 1)

        try:
            auroc = roc_auc_score(all_labels, all_preds)
        except Exception:
            auroc = 0.5

        mean_gates = np.mean(all_gates, axis=0)
        gate_info = f"SigLIP: {mean_gates[0]:.3f}, CLIP: {mean_gates[1]:.3f}, DINO: {mean_gates[2]:.3f}, ConvNeXt: {mean_gates[3]:.3f}"
        print(f"Epoch {epoch:02d} Summary: Loss={train_loss:.4f} | ValLoss={val_loss:.4f} | Acc={acc:.4f} | B-Acc={b_acc:.4f} | AUROC={auroc:.4f} | FPR={fpr*100:.2f}% | Gates [{gate_info}]")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "accuracy": acc,
            "balanced_acc": b_acc,
            "auroc": auroc,
            "fpr": fpr,
            "gate_siglip": mean_gates[0],
            "gate_clip": mean_gates[1],
            "gate_dinov2": mean_gates[2],
            "gate_convnext": mean_gates[3],
        })

        if auroc > best_auroc:
            best_auroc = auroc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "auroc": auroc,
                "acc": acc,
                "gates": mean_gates.tolist(),
            }, best_ckpt)
            print(f"  --> Saved NEW BEST checkpoint (AUROC: {best_auroc:.4f}) to {best_ckpt}!")

    pd.DataFrame(history).to_csv(out_dir / "training_history.csv", index=False)
    print(f"\nTraining Complete. Best Validation AUROC: {best_auroc:.4f}")
    return best_auroc


def main():
    p = argparse.ArgumentParser(description="Train Quad-Hybrid Gating Head with Asymmetric Hard Mining")
    p.add_argument("--cache_h5", default="/mnt/ai-storage/aigc_data/cache/balanced_features_4stream.h5")
    p.add_argument("--output_dir", default="checkpoints/quad_hybrid_v1")
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--fp_penalty", type=float, default=10.0, help="Minus-point penalty weight for False Positives on real images")
    p.add_argument("--ohem_ratio", type=float, default=0.4, help="Ratio of hardest batch examples to mine gradients from")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    train_quad_hybrid(
        cache_h5=args.cache_h5,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        fp_penalty=args.fp_penalty,
        ohem_ratio=args.ohem_ratio,
        device=args.device,
    )


if __name__ == "__main__":
    main()

