"""
AetherForensics — Train Single-Student Forensic Model via Multi-Teacher Knowledge Distillation
Distills the 4-Stream Quad-Hybrid Ensemble (722M params) into a unified student backbone.
Uses in-memory HDF5 feature acceleration on CUDA.
"""

import os
import sys
import argparse
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.metrics import roc_auc_score, accuracy_score, balanced_accuracy_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.quad_hybrid_detector import QuadHybridGatingHead
from models.distilled_student import SingleStudentForensicDetector, DistillationLoss


def parse_args():
    parser = argparse.ArgumentParser(description="Train Single-Student Forensics Model")
    parser.add_argument("--h5_cache", type=str, default="/mnt/ai-storage/aigc_data/cache/balanced_features_4stream.h5")
    parser.add_argument("--teacher_checkpoint", type=str, default="checkpoints/quad_hybrid_v1/best_model.pt")
    parser.add_argument("--output_dir", type=str, default="checkpoints/distilled_student_v1")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("AETHERFORENSICS: SINGLE-STUDENT KNOWLEDGE DISTILLATION")
    print(f"Device: {device} | Temperature: {args.temperature} | Batch Size: {args.batch_size}")
    print("=" * 70)

    # 1. Load Pre-extracted 4-Teacher Feature Cache into RAM
    print(f"Loading cached multi-teacher features from {args.h5_cache}...")
    with h5py.File(args.h5_cache, "r") as f:
        siglip_feats = torch.tensor(f["siglip_features"][:], dtype=torch.float32)
        clip_feats = torch.tensor(f["clip_features"][:], dtype=torch.float32)
        dino_feats = torch.tensor(f["dinov2_features"][:], dtype=torch.float32)
        convnext_feats = torch.tensor(f["convnext_features"][:], dtype=torch.float32)
        labels = torch.tensor(f["labels"][:], dtype=torch.long)

    N = len(labels)
    n_real = (labels == 0).sum().item()
    n_fake = (labels == 1).sum().item()
    print(f"Loaded {N} samples in RAM: {n_real} Real, {n_fake} Synthetic.")

    # 2. Compute Teacher Ensemble Logits in Memory
    print(f"Generating teacher oracle soft-logits using {args.teacher_checkpoint}...")
    teacher = QuadHybridGatingHead().to(device)
    teacher_ckpt = torch.load(args.teacher_checkpoint, map_location=device, weights_only=True)
    teacher.load_state_dict(teacher_ckpt["model_state_dict"])
    teacher.eval()

    with torch.no_grad():
        # L2-Normalize teacher features
        s_norm = siglip_feats / siglip_feats.norm(dim=-1, keepdim=True)
        c_norm = clip_feats / clip_feats.norm(dim=-1, keepdim=True)
        d_norm = dino_feats / dino_feats.norm(dim=-1, keepdim=True)
        x_norm = convnext_feats / convnext_feats.norm(dim=-1, keepdim=True)

        teacher_logits_list = []
        batch_eval = 256
        for i in range(0, N, batch_eval):
            t_s = s_norm[i:i+batch_eval].to(device)
            t_c = c_norm[i:i+batch_eval].to(device)
            t_d = d_norm[i:i+batch_eval].to(device)
            t_x = x_norm[i:i+batch_eval].to(device)
            logits, _ = teacher(t_s, t_c, t_d, t_x)
            teacher_logits_list.append(logits.cpu())

        teacher_logits = torch.cat(teacher_logits_list, dim=0)

    # 3. Train / Validation Split (80 / 20 Stratified)
    indices = np.arange(N)
    np.random.seed(42)
    np.random.shuffle(indices)

    split = int(0.80 * N)
    train_idx = indices[:split]
    val_idx = indices[split:]

    # Student input representation: ConvNeXt-V2-Tiny (768-d)
    student_feats = convnext_feats

    train_dataset = TensorDataset(
        student_feats[train_idx],
        s_norm[train_idx],
        c_norm[train_idx],
        d_norm[train_idx],
        x_norm[train_idx],
        teacher_logits[train_idx],
        labels[train_idx]
    )

    val_dataset = TensorDataset(
        student_feats[val_idx],
        s_norm[val_idx],
        c_norm[val_idx],
        d_norm[val_idx],
        x_norm[val_idx],
        teacher_logits[val_idx],
        labels[val_idx]
    )

    # Balanced Class Sampler
    train_labels = labels[train_idx].numpy()
    class_counts = np.bincount(train_labels)
    class_weights = 1.0 / np.maximum(class_counts, 1)
    sample_weights = class_weights[train_labels]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(train_labels), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    print(f"Train samples: {len(train_idx)} | Validation samples: {len(val_idx)}")

    # 4. Instantiate Single Student & Distillation Loss
    student = SingleStudentForensicDetector(student_dim=768).to(device)
    distill_criterion = DistillationLoss(temperature=args.temperature, alpha_ce=0.4, alpha_kd=0.4, alpha_feat=0.2).to(device)
    optimizer = optim.AdamW(student.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_val_auroc = 0.0
    best_ckpt_path = os.path.join(args.output_dir, "best_student_model.pt")

    print("\nStarting Distillation Training across 25 Epochs...")
    for epoch in range(1, args.epochs + 1):
        student.train()
        train_loss = 0.0
        train_ce = 0.0
        train_kd = 0.0
        train_feat = 0.0

        for b_stu, b_s, b_c, b_d, b_x, b_t_logits, b_y in train_loader:
            b_stu = b_stu.to(device)
            b_t_feats = (b_s.to(device), b_c.to(device), b_d.to(device), b_x.to(device))
            b_t_logits = b_t_logits.to(device)
            b_y = b_y.to(device)

            optimizer.zero_grad()
            stu_logits, stu_projs = student(b_stu, return_projections=True)

            loss, loss_dict = distill_criterion(stu_logits, b_t_logits, stu_projs, b_t_feats, b_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(b_y)
            train_ce += loss_dict["loss_ce"] * len(b_y)
            train_kd += loss_dict["loss_kd"] * len(b_y)
            train_feat += loss_dict["loss_feat"] * len(b_y)

        scheduler.step()
        train_loss /= len(train_idx)
        train_ce /= len(train_idx)
        train_kd /= len(train_idx)
        train_feat /= len(train_idx)

        # 5. Validation Evaluation
        student.eval()
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for b_stu, _, _, _, _, _, b_y in val_loader:
                b_stu = b_stu.to(device)
                logits = student(b_stu, return_projections=False)
                probs = torch.softmax(logits, dim=-1)[:, 1]
                val_preds.extend(probs.cpu().numpy())
                val_targets.extend(b_y.numpy())

        val_preds = np.array(val_preds)
        val_targets = np.array(val_targets)
        val_bin = (val_preds >= 0.5).astype(int)

        val_auroc = roc_auc_score(val_targets, val_preds)
        val_acc = accuracy_score(val_targets, val_bin)
        val_bacc = balanced_accuracy_score(val_targets, val_bin)

        is_best = val_auroc > best_val_auroc
        if is_best:
            best_val_auroc = val_auroc
            torch.save({
                "epoch": epoch,
                "model_state_dict": student.state_dict(),
                "val_auroc": val_auroc,
                "val_acc": val_acc,
                "val_bacc": val_bacc
            }, best_ckpt_path)

        star = "★ NEW BEST" if is_best else ""
        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] Loss: {train_loss:.4f} (CE: {train_ce:.3f}, KD: {train_kd:.3f}, Feat: {train_feat:.3f}) | Val AUROC: {val_auroc:.4f} | Acc: {val_acc*100:.2f}% | BAcc: {val_bacc*100:.2f}% {star}")

    print("\n" + "=" * 70)
    print(f"DISTILLATION COMPLETE! Best Student Validation AUROC: {best_val_auroc:.4f}")
    print(f"Saved checkpoint to: {best_ckpt_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
