#!/usr/bin/env python3
"""Evaluation engine for Quad-Hybrid 4-Stream Dynamic Gating Detector across 15-condition matrix.
Fuses SigLIP (768-d), CLIP (1024-d), DINOv2 (1024-d), and ConvNeXt-V2 (768-d).
"""

import argparse
import io
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
from torchvision import transforms
from tqdm import tqdm
from transformers import AutoModel, CLIPModel, ConvNextV2Model

from models.quad_hybrid_detector import QuadHybridGatingHead
from scripts.augmentations import PERTURBATIONS, evaluation_transform
from scripts.data import VALID_EXTS


def find_image_files(root_dir):
    files = []
    for base, _, names in os.walk(root_dir):
        for n in sorted(names):
            if n.lower().endswith(VALID_EXTS):
                files.append(os.path.join(base, n))
    return sorted(files)


def main():
    p = argparse.ArgumentParser(description="Evaluate Quad-Hybrid 4-Stream Dynamic Gating Detector")
    p.add_argument("--checkpoint", default="checkpoints/quad_hybrid_v1/best_model.pt")
    p.add_argument("--coco_dir", default="/mnt/ai-storage/aigc_data/validation_LOCKED/val2017")
    p.add_argument("--fake_dir", default="/mnt/ai-storage/aigc_data/datasets/cf_slice/synthetic")
    p.add_argument("--siglip_dir", default="/mnt/ai-storage/aigc_data/models/siglip_base_224")
    p.add_argument("--clip_dir", default="/mnt/ai-storage/aigc_data/models/clip_vitl14")
    p.add_argument("--dinov2_dir", default="/mnt/ai-storage/aigc_data/models/dinov2_large")
    p.add_argument("--convnext_dir", default="/mnt/ai-storage/aigc_data/models/convnextv2_tiny")
    p.add_argument("--max_images", type=int, default=100)
    p.add_argument("--output_csv", default="reports/quad_hybrid_robustness_results.csv")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print(f"Running Quad-Hybrid evaluation on device: {device}")

    real_paths = find_image_files(args.coco_dir)[:args.max_images]
    fake_paths = find_image_files(args.fake_dir)[:args.max_images]
    all_paths = real_paths + fake_paths
    labels = np.array([0] * len(real_paths) + [1] * len(fake_paths), dtype=np.int64)

    print(f"Loaded {len(real_paths)} authentic and {len(fake_paths)} synthetic benchmark samples.")

    # Load All 4 Backbones in FP16 (Total VRAM: ~1.44 GB on RTX 3050)
    print("Loading 4 Orthogonal Vision Foundation Models in FP16 (VRAM: ~1.4 GB)...")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    siglip = AutoModel.from_pretrained(args.siglip_dir, torch_dtype=dtype).to(device).eval()
    clip_model = CLIPModel.from_pretrained(args.clip_dir, torch_dtype=dtype).to(device).eval()
    dinov2 = AutoModel.from_pretrained(args.dinov2_dir, torch_dtype=dtype).to(device).eval()
    convnext = ConvNextV2Model.from_pretrained(args.convnext_dir, torch_dtype=dtype).to(device).eval()

    # Load Quad-Hybrid Fusion Head
    fusion_head = QuadHybridGatingHead().to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    fusion_head.load_state_dict(ckpt["model_state_dict"])
    fusion_head.eval()

    siglip_norm = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    clip_norm = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
    dino_norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    results = []
    print(f"\nEvaluating across 15 standard perturbation conditions...")


    eval_preprocess = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
    ])

    for cond_name in PERTURBATIONS:
        preds = []
        gates_list = []

        with torch.no_grad():
            for p in tqdm(all_paths, desc=f"Evaluating {cond_name:18s}", leave=False):
                try:
                    img = Image.open(p).convert("RGB")
                    aug_np = evaluation_transform(cond_name, np.array(img))
                    aug_img = Image.fromarray(aug_np)

                    # Preprocessing
                    tensor_raw = eval_preprocess(aug_img)

                    # 1. SigLIP Preprocessing
                    t_siglip = siglip_norm(tensor_raw).unsqueeze(0).to(device)

                    # 2. CLIP Preprocessing
                    t_clip = clip_norm(tensor_raw).unsqueeze(0).to(device)

                    # 3. DINOv2 Preprocessing
                    t_dino = dino_norm(tensor_raw).unsqueeze(0).to(device)

                    # 4. ConvNeXt Preprocessing
                    t_convnext = dino_norm(tensor_raw).unsqueeze(0).to(device)


                    with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                        # Stream 1: SigLIP (768-d)
                        out_s = siglip(pixel_values=t_siglip) if not hasattr(siglip, "vision_model") else siglip.vision_model(pixel_values=t_siglip)
                        f_s = out_s.last_hidden_state[:, 0]
                        f_s = f_s / f_s.norm(dim=-1, keepdim=True)

                        # Stream 2: CLIP ViT-L/14 (1024-d)
                        out_c = clip_model.vision_model(pixel_values=t_clip)
                        f_c = out_c.last_hidden_state[:, 0]
                        f_c = f_c / f_c.norm(dim=-1, keepdim=True)

                        # Stream 3: DINOv2-Large (1024-d)
                        out_d = dinov2(pixel_values=t_dino)
                        f_d = out_d.last_hidden_state[:, 0]
                        f_d = f_d / f_d.norm(dim=-1, keepdim=True)

                        # Stream 4: ConvNeXt-V2-Tiny (768-d)
                        out_x = convnext(pixel_values=t_convnext)
                        f_x = out_x.last_hidden_state.mean(dim=[-2, -1])
                        f_x = f_x / f_x.norm(dim=-1, keepdim=True)


                        # Fusion Head
                        logits, gates = fusion_head(f_s.float(), f_c.float(), f_d.float(), f_x.float())
                        prob = F.softmax(logits, dim=-1)[0, 1].item()


                    preds.append(prob)
                    gates_list.append(gates.cpu().numpy()[0])
                except Exception as err:
                    print(f"\nInference error on {p}: {err}")
                    preds.append(0.5)
                    gates_list.append([0.25, 0.25, 0.25, 0.25])


        preds = np.array(preds)
        bin_preds = (preds >= 0.5).astype(int)
        acc = accuracy_score(labels, bin_preds)
        b_acc = balanced_accuracy_score(labels, bin_preds)
        f1 = f1_score(labels, bin_preds, zero_division=0)
        try:
            auroc = roc_auc_score(labels, preds)
        except Exception:
            auroc = 0.5

        mean_gates = np.mean(gates_list, axis=0)
        results.append({
            "condition": cond_name,
            "auroc": auroc,
            "accuracy": acc,
            "balanced_acc": b_acc,
            "f1": f1,
            "gate_siglip": mean_gates[0],
            "gate_clip": mean_gates[1],
            "gate_dinov2": mean_gates[2],
            "gate_convnext": mean_gates[3],
        })
        print(f"  --> {cond_name:18s} | AUROC: {auroc:.4f} | Acc: {acc*100:5.1f}% | Gates: [S: {mean_gates[0]:.2f}, C: {mean_gates[1]:.2f}, D: {mean_gates[2]:.2f}, X: {mean_gates[3]:.2f}]")

    df = pd.DataFrame(results)
    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"\nSaved Quad-Hybrid Robustness Results to {out_csv}!")
    print(f"Macro-Robustness AUROC (Mean over 15 conditions): {df['auroc'].mean():.4f}")


if __name__ == "__main__":
    main()
