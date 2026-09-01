import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import os

import numpy as np
import torch
import yaml
from PIL import Image
from tqdm import tqdm

from models.tri_hybrid_detector import MasterEnsembleDetector
from scripts.data import VALID_EXTS
from scripts.transforms import prepare_inputs


def load_model(checkpoint, config_path, device):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    model = MasterEnsembleDetector(**cfg["models"])
    state = torch.load(checkpoint, map_location=device)
    if "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.to(device).eval()
    return model


def main():
    p = argparse.ArgumentParser(description="Batch AIGC image inference")
    p.add_argument("--image_dir", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", default="results.json")
    p.add_argument("--config", default="configs/train_config.yaml")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device = torch.device(
        args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    )
    model = load_model(args.checkpoint, args.config, device)

    paths = []
    for root, _, names in os.walk(args.image_dir):
        for name in names:
            if name.lower().endswith(VALID_EXTS):
                paths.append(os.path.join(root, name))
    paths.sort()

    results = []
    with torch.inference_mode():
        for path in tqdm(paths, desc="Inference"):
            try:
                img = np.array(Image.open(path).convert("RGB"))
                raw = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0) / 255.0
                raw = raw.to(device)
                clip, siglip, raw_224 = prepare_inputs(raw)
                prob = torch.sigmoid(model(clip, siglip, raw_224)).item()
                results.append(
                    {
                        "image_path": os.path.relpath(path, args.image_dir),
                        "pred": round(float(prob), 4),
                    }
                )
            except Exception as exc:
                print(f"[WARN] Failed: {path}: {exc}")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {len(results)} predictions to {args.output}")


if __name__ == "__main__":
    main()
