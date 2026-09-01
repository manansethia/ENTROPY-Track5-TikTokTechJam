import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import os

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from tqdm import tqdm

from models.tri_hybrid_detector import MasterEnsembleDetector
from scripts.augmentations import PERTURBATIONS, evaluation_transform
from scripts.data import VALID_EXTS
from scripts.transforms import prepare_inputs


def load_paths(root, label):
    out = []
    for base, _, names in os.walk(root):
        for n in names:
            if n.lower().endswith(VALID_EXTS):
                out.append((os.path.join(base, n), label))
    return out


def main():
    p = argparse.ArgumentParser(description="Robustness evaluation on isolated demo benchmark")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--coco_dir", required=True)
    p.add_argument("--dalle_dir", required=True)
    p.add_argument("--config", default="configs/train_config.yaml")
    p.add_argument("--output_csv", default="robustness_results.csv")
    p.add_argument("--device", default="cuda")
    p.add_argument("--max_images", type=int, default=0, help="0 = all images")
    args = p.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    device = torch.device(
        args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    )
    model = MasterEnsembleDetector(**cfg["models"])
    state = torch.load(args.checkpoint, map_location=device)
    if "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.to(device).eval()

    samples = load_paths(args.coco_dir, 0) + load_paths(args.dalle_dir, 1)
    samples.sort(key=lambda x: x[0])
    if args.max_images:
        samples = samples[: args.max_images]

    rows = []
    for name in PERTURBATIONS:
        y_true, y_pred, y_prob = [], [], []
        for path, label in tqdm(samples, desc=name):
            try:
                img = np.array(Image.open(path).convert("RGB"))
                img = evaluation_transform(name, img)
                raw = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0) / 255.0
                raw = raw.to(device)
                clip, siglip, raw_224 = prepare_inputs(raw)
                with torch.inference_mode():
                    prob = torch.sigmoid(model(clip, siglip, raw_224)).item()
                y_true.append(label)
                y_prob.append(prob)
                y_pred.append(int(prob >= 0.5))
            except Exception as exc:
                print(f"[WARN] {name}: {path}: {exc}")

        row = {
            "transform": name,
            "n": len(y_true),
            "accuracy": accuracy_score(y_true, y_pred) if y_true else np.nan,
            "balanced_accuracy": balanced_accuracy_score(y_true, y_pred) if y_true else np.nan,
            "f1": f1_score(y_true, y_pred, zero_division=0) if y_true else np.nan,
            "auroc": roc_auc_score(y_true, y_prob) if len(set(y_true)) == 2 else np.nan,
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(args.output_csv, index=False)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"Wrote {args.output_csv}")


if __name__ == "__main__":
    main()
