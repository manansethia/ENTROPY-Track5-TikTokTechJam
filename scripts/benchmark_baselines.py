#!/usr/bin/env python3
"""Zero-shot and baseline benchmark runner across clean and 14 challenge degradation conditions.
Evaluates foundation vision backbones on the isolated benchmark (COCO val2017 authentic + AIGC).
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
from PIL import Image
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from tqdm import tqdm
import open_clip

from scripts.augmentations import PERTURBATIONS, evaluation_transform
from scripts.data import VALID_EXTS


def load_paths(root, label, max_count=0):
    out = []
    root_path = Path(root)
    if not root_path.exists():
        return out
    for base, _, names in os.walk(root):
        for n in sorted(names):
            if n.lower().endswith(VALID_EXTS):
                out.append((os.path.join(base, n), label))
                if max_count and len(out) >= max_count:
                    return out
    return out


class ZeroShotCLIPProbe:
    def __init__(self, model_name="ViT-L-14", pretrained="openai", device="cuda"):
        self.device = device
        print(f"Loading OpenCLIP {model_name} ({pretrained})...")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=device
        )
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)
        
        # Zero-shot forensic prompt embeddings
        prompts = ["a natural real authentic photograph", "a synthetic artificial AI generated image"]
        text = self.tokenizer(prompts).to(device)
        with torch.no_grad():
            self.text_features = self.model.encode_text(text)
            self.text_features /= self.text_features.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def predict_prob(self, img_pil):
        image = self.preprocess(img_pil).unsqueeze(0).to(self.device)
        image_features = self.model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        
        # Cosine similarity with temperature
        sim = (image_features @ self.text_features.T) * 100.0
        probs = torch.softmax(sim, dim=-1)
        return probs[0, 1].item()


def main():
    p = argparse.ArgumentParser(description="Zero-shot and baseline robustness evaluation")
    p.add_argument("--coco_dir", default="/mnt/ai-storage/aigc_data/validation_LOCKED/val2017")
    p.add_argument("--fake_dir", default="/mnt/ai-storage/aigc_data/datasets/cf_slice/synthetic")
    p.add_argument("--output_csv", default="reports/baseline_robustness_results.csv")
    p.add_argument("--max_images", type=int, default=200, help="Per-class evaluation sample count")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print(f"Running baseline benchmark on device: {device}")

    real_samples = load_paths(args.coco_dir, 0, max_count=args.max_images)
    fake_samples = load_paths(args.fake_dir, 1, max_count=args.max_images)
    samples = real_samples + fake_samples
    print(f"Loaded {len(real_samples)} authentic and {len(fake_samples)} synthetic benchmark samples.")

    if not samples:
        print("ERROR: No benchmark samples found. Check --coco_dir and --fake_dir.")
        sys.exit(1)

    probe = ZeroShotCLIPProbe(device=device)

    results = []
    print("\nStarting evaluation across 15 perturbation conditions...")

    for pert in PERTURBATIONS:
        y_true, y_pred, y_prob = [], [], []
        pbar = tqdm(samples, desc=f"Evaluating {pert:<16}")
        for path, label in pbar:
            try:
                img_raw = np.array(Image.open(path).convert("RGB"))
                img_aug = evaluation_transform(pert, img_raw)
                img_pil = Image.fromarray(img_aug)
                
                prob = probe.predict_prob(img_pil)
                y_true.append(label)
                y_prob.append(prob)
                y_pred.append(int(prob >= 0.5))
            except Exception as exc:
                print(f"[WARN] {pert}: {path}: {exc}")

        acc = accuracy_score(y_true, y_pred) if y_true else np.nan
        b_acc = balanced_accuracy_score(y_true, y_pred) if y_true else np.nan
        f1 = f1_score(y_true, y_pred, zero_division=0) if y_true else np.nan
        auroc = roc_auc_score(y_true, y_prob) if len(set(y_true)) == 2 else np.nan

        results.append({
            "Perturbation": pert,
            "N": len(y_true),
            "Accuracy": acc,
            "Balanced_Acc": b_acc,
            "F1_Score": f1,
            "AUROC": auroc,
        })
        pbar.set_postfix({"AUROC": f"{auroc:.4f}", "B-Acc": f"{b_acc:.4f}"})

    df = pd.DataFrame(results)
    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    
    print("\n" + "=" * 70)
    print("BASELINE ROBUSTNESS BENCHMARK RESULTS")
    print("=" * 70)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    
    # Summary metrics
    clean_auroc = df[df["Perturbation"] == "Clean"]["AUROC"].values[0]
    mean_auroc = df["AUROC"].mean()
    worst_auroc = df["AUROC"].min()
    print("\n" + "-" * 70)
    print(f"Clean AUROC:             {clean_auroc:.4f}")
    print(f"Macro-Robustness AUROC:  {mean_auroc:.4f}")
    print(f"Worst-Case AUROC:        {worst_auroc:.4f} (Degradation: {clean_auroc - worst_auroc:.4f})")
    print("=" * 70)


if __name__ == "__main__":
    main()
