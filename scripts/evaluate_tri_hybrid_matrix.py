#!/usr/bin/env python3
"""Evaluation engine for Tri-Hybrid Dynamic Gating Detector across 15-condition matrix.
Compares robustness vs zero-shot baseline.
"""

import argparse
import io
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor, CLIPModel

from scripts.augmentations import PERTURBATIONS, evaluation_transform
from scripts.data import VALID_EXTS
from scripts.train_tri_hybrid_gating import DynamicGatingFusionHead


def find_image_files(root_dir):
    files = []
    for base, _, names in os.walk(root_dir):
        for n in sorted(names):
            if n.lower().endswith(VALID_EXTS):
                files.append(os.path.join(base, n))
    return sorted(files)


def main():
    p = argparse.ArgumentParser(description="Evaluate Tri-Hybrid Dynamic Gating Detector")
    p.add_argument("--checkpoint", default="checkpoints/tri_hybrid_v1/best_model.pt")
    p.add_argument("--coco_dir", default="/mnt/ai-storage/aigc_data/validation_LOCKED/val2017")
    p.add_argument("--fake_dir", default="/mnt/ai-storage/aigc_data/datasets/cf_slice/synthetic")
    p.add_argument("--siglip_dir", default="/mnt/ai-storage/aigc_data/models/siglip_base_224")
    p.add_argument("--clip_dir", default="/mnt/ai-storage/aigc_data/models/clip_vitl14")
    p.add_argument("--dinov2_dir", default="/mnt/ai-storage/aigc_data/models/dinov2_large")
    p.add_argument("--max_images", type=int, default=100)
    p.add_argument("--output_csv", default="reports/tri_hybrid_robustness_results.csv")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print(f"Running evaluation on device: {device}")

    # Load images
    real_paths = find_image_files(args.coco_dir)[:args.max_images]
    fake_paths = find_image_files(args.fake_dir)[:args.max_images]
    all_paths = real_paths + fake_paths
    labels = np.array([0] * len(real_paths) + [1] * len(fake_paths), dtype=np.int64)

    print(f"Loaded {len(real_paths)} authentic and {len(fake_paths)} synthetic benchmark samples.")

    # Load backbones
    print(f"Loading SigLIP from {args.siglip_dir}...")
    siglip_proc = AutoProcessor.from_pretrained(args.siglip_dir)
    siglip_model = AutoModel.from_pretrained(args.siglip_dir).to(device)
    siglip_model.eval()

    print(f"Loading CLIP from {args.clip_dir}...")
    clip_proc = AutoProcessor.from_pretrained(args.clip_dir)
    clip_model = CLIPModel.from_pretrained(args.clip_dir).to(device)
    clip_model.eval()

    # Load trained fusion head
    print(f"Loading trained fusion head from {args.checkpoint}...")
    ckpt = torch.load(args.checkpoint, map_location=device)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    has_dinov2 = "proj_dinov2.0.weight" in state_dict
    print(f"Model Architecture: {'3-Stream (SigLIP + CLIP + DINOv2)' if has_dinov2 else '2-Stream (SigLIP + CLIP)'}")

    dinov2_model = None
    dinov2_transform = None
    if has_dinov2:
        print(f"Loading DINOv2 from {args.dinov2_dir}...")
        from torchvision import transforms
        dinov2_transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        dinov2_model = AutoModel.from_pretrained(args.dinov2_dir).to(device)
        dinov2_model.eval()

    fusion_head = DynamicGatingFusionHead(has_dinov2=has_dinov2).to(device)
    fusion_head.load_state_dict(state_dict)
    fusion_head.eval()

    results = []
    print("\nStarting evaluation across 15 perturbation conditions...")

    for pert_name in PERTURBATIONS:
        pbar = tqdm(all_paths, desc=f"Evaluating {pert_name:<15}")
        y_probs, y_preds = [], []
        gate_weights = []

        for pth in pbar:
            img = np.array(Image.open(pth).convert("RGB"))
            # Apply perturbation
            pert_arr = evaluation_transform(pert_name, img)
            pert_img = Image.fromarray(pert_arr)

            # Extract features
            with torch.no_grad():
                s_inputs = siglip_proc(images=pert_img, return_tensors="pt").to(device)
                s_out = siglip_model.vision_model(**s_inputs)
                s_feat = s_out.pooler_output

                c_inputs = clip_proc(images=pert_img, return_tensors="pt").to(device)
                c_out = clip_model.vision_model(**c_inputs)
                c_feat = c_out.pooler_output
                c_feat = c_feat / c_feat.norm(dim=-1, keepdim=True)

                if has_dinov2 and dinov2_model is not None:
                    d_tensor = dinov2_transform(pert_img).unsqueeze(0).to(device)
                    d_out = dinov2_model(d_tensor)
                    d_feat = d_out.pooler_output if hasattr(d_out, "pooler_output") and d_out.pooler_output is not None else d_out.last_hidden_state[:, 0]
                    d_feat = d_feat / d_feat.norm(dim=-1, keepdim=True)
                else:
                    d_feat = None

                logits, gates = fusion_head(s_feat, c_feat, d_feat)
                prob = F.softmax(logits, dim=-1)[0, 1].item()

            y_probs.append(prob)
            y_preds.append(int(prob >= 0.5))
            gate_weights.append(gates[0].cpu().numpy())

        acc = accuracy_score(labels, y_preds)
        bacc = balanced_accuracy_score(labels, y_preds)
        f1 = f1_score(labels, y_preds, zero_division=0)
        try:
            auc = roc_auc_score(labels, y_probs)
        except Exception:
            auc = 0.5
        mean_gates = np.mean(gate_weights, axis=0)

        res_row = {
            "condition": pert_name,
            "auroc": auc,
            "accuracy": acc,
            "balanced_acc": bacc,
            "f1": f1,
            "gate_siglip": mean_gates[0],
            "gate_clip": mean_gates[1],
        }
        if has_dinov2:
            res_row["gate_dinov2"] = mean_gates[2]
        results.append(res_row)
        print(f"  [{pert_name:<15}] AUROC: {auc:.4f} | Acc: {acc:.4f} | B-Acc: {bacc:.4f} | F1: {f1:.4f} | " + 
              (f"Gates: [SigLIP: {mean_gates[0]:.3f}, CLIP: {mean_gates[1]:.3f}, DINOv2: {mean_gates[2]:.3f}]" if has_dinov2 else f"Gates: [SigLIP: {mean_gates[0]:.3f}, CLIP: {mean_gates[1]:.3f}]"))

    df = pd.DataFrame(results)
    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    clean_auc = df.loc[df["condition"] == "Clean", "auroc"].values[0]
    macro_auc = df["auroc"].mean()
    worst_row = df.loc[df["auroc"].idxmin()]

    print("\n" + "=" * 78)
    print("TRI-HYBRID DYNAMIC GATING ROBUSTNESS RESULTS")
    print("=" * 78)
    print(df.to_string(index=False, formatters={
        "Accuracy": "{:.4f}".format,
        "Balanced_Acc": "{:.4f}".format,
        "F1_Score": "{:.4f}".format,
        "AUROC": "{:.4f}".format,
        "Gate_SigLIP": "{:.3f}".format,
        "Gate_CLIP": "{:.3f}".format,
    }))
    print("-" * 78)
    print(f"Clean AUROC:             {clean_auc:.4f}")
    print(f"Macro-Robustness AUROC:  {macro_auc:.4f}")
    print(f"Worst-Case AUROC:        {worst_row['auroc']:.4f} ({worst_row['condition']}, Degradation: {clean_auc - worst_row['auroc']:.4f})")
    print("=" * 78)


if __name__ == "__main__":
    main()
